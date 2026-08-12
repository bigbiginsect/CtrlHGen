"""Run the one repaired Phase D frozen test after terminal selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from akgr.abduction_model.experiment_runner import (
    _file_sha256,
    _graph_samplers,
    _require_cuda,
    _require_phase_d_parent_checkpoint,
)
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.phase_d_pilot import (
    _compare_validation,
    _evaluate_model,
    _write_json,
)
from akgr.utils.load_util import load_reproduction_checkpoint


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _validate_terminal(checkpoint: Path, expected_sha256: str) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name != "checkpoint-13000":
        raise ValueError(f"Frozen test requires checkpoint-13000, got {checkpoint}")
    model_path = checkpoint / "model.safetensors"
    state_path = checkpoint / "trainer_state.json"
    required = (model_path, state_path, checkpoint / "tokenizer.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Terminal checkpoint is incomplete: {missing}")
    actual_sha256 = _file_sha256(model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Terminal model SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != 13000:
        raise ValueError(f"Terminal trainer state is not step 13000: {state.get('global_step')}")
    return {
        "checkpoint": str(checkpoint),
        "global_step": 13000,
        "model_sha256": actual_sha256,
        "trainer_state_sha256": _file_sha256(state_path),
    }


def _frozen_decision(comparison: dict[str, Any]) -> dict[str, Any]:
    health = all(
        float(comparison[decode]["candidate"]["parse_ok"]) >= 0.98
        and float(comparison[decode]["candidate"]["eos_rate"]) >= 0.98
        and float(comparison[decode]["candidate"]["condition_accuracy"]) >= 0.90
        for decode in ("greedy", "sampled")
    )
    semantic = all(
        float(comparison[decode]["delta"]["semantic_average"]) > 0.0
        for decode in ("greedy", "sampled")
    )
    return {
        "classification": "pass" if health and semantic else "fail",
        "health_pass": health,
        "semantic_both_decodes_positive": semantic,
        "rule": (
            "pass iff repaired terminal has positive Jaccard/Dice/Overlap average delta "
            "for greedy and sampled, with parse/EOS >= 0.98 and condition >= 0.90"
        ),
        "post_test_training_or_checkpoint_selection_allowed": False,
    }


def run_frozen_test(args: argparse.Namespace) -> Path:
    config = load_experiment_config(args.experiment_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    terminal = _validate_terminal(
        Path(args.terminal_checkpoint), args.expected_model_sha256
    )
    parent = _require_phase_d_parent_checkpoint(args.parent_checkpoint)
    parent_loaded = load_reproduction_checkpoint(
        parent,
        mode="parent",
        expected_stage="conditional",
        expected_condition=config.condition,
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    preflight = {
        "schema_version": 1,
        "kind": "phase_d_repaired_frozen_test_preflight",
        "created_at": _utc_now(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_semantic_hash": config.semantic_hash,
        "data_manifest_sha256": _file_sha256(config.sampling_manifest_path),
        "parent_checkpoint": str(parent),
        "parent_model_config_hash": parent_loaded.metadata["model_config_hash"],
        "terminal": terminal,
        "split": "test",
        "decoding": ["greedy", "sampled"],
        "selection_frozen_before_test": True,
        "no_post_test_training_or_checkpoint_selection": True,
        "decision_rule": _frozen_decision({
            decode: {
                "candidate": {"parse_ok": 1, "eos_rate": 1, "condition_accuracy": 1},
                "delta": {"semantic_average": 1},
            }
            for decode in ("greedy", "sampled")
        })["rule"],
    }
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(status_path, {
        "schema_version": 1, "status": "running", "started_at": _utc_now()
    })

    try:
        graph_samplers = _graph_samplers(config)
        device = _require_cuda("repaired Phase D frozen test")
        parent_model = parent_loaded.model.to(device)
        baseline = _evaluate_model(
            config=config,
            model=parent_model,
            tokenizer=parent_loaded.tokenizer,
            graph_samplers=graph_samplers,
            device=device,
            output_dir=output_dir,
            label="parent",
            split="test",
            seed_namespace="phase_d_repaired_frozen_test",
        )
        del parent_model
        torch.cuda.empty_cache()

        terminal_path = Path(args.terminal_checkpoint).expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(terminal_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            terminal_path, local_files_only=True
        ).to(device)
        candidate = _evaluate_model(
            config=config,
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            device=device,
            output_dir=output_dir,
            label="repaired-terminal",
            split="test",
            seed_namespace="phase_d_repaired_frozen_test",
        )
        comparison = _compare_validation(baseline, candidate, seed=config.seed)
        comparison["decision"] = _frozen_decision(comparison)
        _write_json(output_dir / "comparison.json", comparison)
        del model
        torch.cuda.empty_cache()

        artifacts = {
            path.name: _artifact(path, output_dir)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name not in {"status.json", "result.json"}
        }
        result = {
            "schema_version": 1,
            "kind": "phase_d_repaired_frozen_test",
            "status": "completed",
            "finished_at": _utc_now(),
            "preflight": preflight,
            "comparison": comparison,
            "artifacts": artifacts,
        }
        result_path = output_dir / "result.json"
        _write_json(result_path, result)
        _write_json(status_path, {
            "schema_version": 1,
            "status": "completed",
            "finished_at": result["finished_at"],
            "decision": comparison["decision"]["classification"],
            "result": result_path.name,
        })
        return result_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": 1,
            "status": "failed",
            "finished_at": _utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--terminal-checkpoint", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    print(run_frozen_test(parser.parse_args()))


if __name__ == "__main__":
    main()

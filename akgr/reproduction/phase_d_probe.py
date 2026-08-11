"""Evaluate one complete GRPO Trainer checkpoint on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from akgr.abduction_model.experiment_runner import _graph_samplers, _require_cuda
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.phase_d_pilot import (
    _compare_validation,
    _evaluate_model,
    _write_json,
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_parent_validation(validation_dir: Path) -> dict[str, Any]:
    result = {}
    for decode in ("greedy", "sampled"):
        records_path = validation_dir / f"parent-valid-{decode}.jsonl"
        metrics_path = validation_dir / f"parent-valid-{decode}.metrics.json"
        if not records_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(
                f"Parent validation artifacts are incomplete in {validation_dir}"
            )
        result[decode] = {
            "records": _load_records(records_path),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        }
    return result


def _checkpoint_step(checkpoint: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(f"Expected checkpoint-<step>, got {checkpoint}")
    required = ("model.safetensors", "trainer_state.json", "tokenizer.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}; missing {missing}")
    return int(match.group(1))


def _recommendation(comparison: dict[str, Any], step: int) -> dict[str, Any]:
    greedy = float(comparison["greedy"]["delta"]["semantic_average"])
    sampled = float(comparison["sampled"]["delta"]["semantic_average"])
    health = bool(comparison["decision"]["health_pass"])
    if not health:
        action = "stop_health_regression"
    elif step >= 4000 and greedy <= 0.002 and sampled <= 0.002:
        action = "stop_negligible_gain_after_4000"
    elif step >= 8000 and greedy < 0.005 and sampled < 0.005:
        action = "stop_below_pilot_scale_after_8000"
    else:
        action = "continue"
    return {
        "action": action,
        "rule": {
            "health_regression": "stop immediately",
            "after_step_4000": "stop if greedy and sampled semantic deltas are both <= 0.002",
            "after_step_8000": "stop if greedy and sampled semantic deltas are both < 0.005",
        },
    }


def run_probe(args: argparse.Namespace) -> Path:
    config = load_experiment_config(args.experiment_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    step = _checkpoint_step(checkpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline = _load_parent_validation(
        Path(args.parent_validation_dir).expanduser().resolve()
    )
    graph_samplers = _graph_samplers(config)
    device = _require_cuda("Phase D validation probe")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device)
    candidate = _evaluate_model(
        config=config,
        model=model,
        tokenizer=tokenizer,
        graph_samplers=graph_samplers,
        device=device,
        output_dir=output_dir,
        label=f"step-{step}",
    )
    comparison = _compare_validation(baseline, candidate, seed=config.seed)
    result = {
        "schema_version": 1,
        "kind": "phase_d_validation_probe",
        "checkpoint": str(checkpoint),
        "step": step,
        "validation": comparison,
        "recommendation": _recommendation(comparison, step),
    }
    result_path = output_dir / "probe-result.json"
    _write_json(result_path, result)
    del model
    torch.cuda.empty_cache()
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run_probe(args))


if __name__ == "__main__":
    main()

"""Run the paired, single-seed SC-IDC Experiment 2 GRPO branches.

This entry point is intentionally separate from the generic reproduction GRPO
runner.  It consumes the frozen fresh-query RL manifest produced by the SC-IDC
preflight and never opens the sealed final-evaluation manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

# PyTorch requires this to make CUDA GEMM deterministic.  Set it before any
# project import can initialize a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd

from akgr.abduction_model.experiment_runner import _graph_samplers
from akgr.abduction_model.reproduction import build_grpo_config, train_grpo
from akgr.dataloader import pre_pre_processing
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.sc_idc import (
    action_string,
    parse_action,
    parse_raw_query,
    value_occurrences,
)
from akgr.reproduction.sc_idc_phase2_data import (
    checkpoint_tree_sha256,
    load_frozen_condition_manifest,
)
from akgr.reproduction.sc_idc_phase2_reward import (
    CachedQueryExecutor,
    StaticRelationMatcher,
    audit_relation_value,
    set_semantic_scores,
)
from akgr.reproduction.seed import seed_everything
from akgr.tokenizer import build_generation_prompt
from akgr.utils.load_util import (
    load_reproduction_checkpoint,
    save_reproduction_checkpoint,
)
from akgr.utils.parsing_util import ans_unshift_indices


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
BRANCHES = ("baseline", "sc_idc")
EXPECTED_ALPHA = 0.05
REWARD_SEED = 42
CACHE_ENTRIES = 50_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _git_state() -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, encoding="utf-8"
    ).strip())
    return {"sha": sha, "dirty": dirty}


def _query_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({
        "query": record["query"], "pattern_str": record["pattern_str"]
    })


def _supervision_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({
        "answers": record["answers"],
        "query": record["query"],
        "pattern_str": record["pattern_str"],
    })


def _target_action(record: Mapping[str, Any]) -> str:
    return action_string(parse_raw_query(record["query"]))


def _require_parent(parent: Path, *, config, expected_tree_sha256: str):
    pointer_path = parent.parent / "conditional-best.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    selected = (parent.parent / pointer["checkpoint"]).resolve()
    if selected != parent.resolve():
        raise ValueError(f"Paired GRPO requires selected conditional parent {selected}")
    if pointer.get("validation", {}).get("health_pass") is not True:
        raise ValueError("Selected conditional parent did not pass health gates")
    actual_tree = checkpoint_tree_sha256(parent)
    if actual_tree != expected_tree_sha256:
        raise ValueError("Conditional parent checkpoint tree hash mismatch")
    loaded = load_reproduction_checkpoint(
        parent,
        mode="parent",
        expected_stage="conditional",
        expected_condition="specific_relation",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    return loaded, actual_tree


def load_grpo_contract(
    *,
    config,
    branch: str,
    preflight_path: Path,
    alpha_freeze_path: Path,
    parent: Path,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """Verify the frozen Experiment 2 inputs without opening final evaluation."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    alpha_freeze = json.loads(alpha_freeze_path.read_text(encoding="utf-8"))
    if preflight.get("kind") != "sc_idc_specific_relation_sft_preflight":
        raise ValueError("Not an SC-IDC specific-relation preflight")
    if preflight.get("status") != "ready":
        raise ValueError("SC-IDC preflight is not ready")
    if preflight.get("isolation", {}).get("final_evaluation_sealed") is not True:
        raise ValueError("Final evaluation is not sealed")
    for key, expected in {
        "config_semantic_hash": config.semantic_hash,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "condition": "specific_relation",
    }.items():
        if preflight.get("target", {}).get(key) != expected:
            raise ValueError(f"Phase 2 target {key} mismatch")
    if alpha_freeze.get("kind") != "sc_idc_alpha_freeze":
        raise ValueError("Not an SC-IDC alpha freeze")
    if alpha_freeze.get("status") != "frozen":
        raise ValueError("SC-IDC alpha is not frozen")
    if alpha_freeze.get("decision") != "go_to_grpo_preparation":
        raise ValueError("Experiment 1 did not authorize GRPO preparation")
    if alpha_freeze.get("final_evaluation_manifest_loaded") is not False:
        raise ValueError("Experiment 1 did not preserve sealed evaluation isolation")
    if float(alpha_freeze.get("selected_alpha_idc")) != EXPECTED_ALPHA:
        raise ValueError("Frozen alpha_IDC is not 0.05")

    rl_ref = preflight["conditions"]["rl_train"]
    rl_hash = str(rl_ref["manifest_sha256"])
    if alpha_freeze.get("rl_train_manifest_sha256") != rl_hash:
        raise ValueError("Experiment 1 and preflight RL manifest hashes differ")
    manifest_path = (preflight_path.parent / rl_ref["manifest_path"]).resolve()
    conditions, rl_manifest = load_frozen_condition_manifest(
        manifest_path,
        target_config=config,
        purpose="rl_train",
        consumer="baseline_grpo" if branch == "baseline" else "sc_idc_grpo",
        expected_manifest_sha256=rl_hash,
    )
    source = rl_manifest["source"]
    if source.get("kind") != "condition_agnostic_fresh_query_rebind":
        raise ValueError("RL train is not backed by the frozen fresh-query source")
    fresh_manifest_path = Path(source["manifest_path"]).expanduser().resolve()
    if _file_sha256(fresh_manifest_path) != source["manifest_sha256"]:
        raise ValueError("Fresh source manifest hash mismatch")
    fresh_manifest = json.loads(fresh_manifest_path.read_text(encoding="utf-8"))
    if fresh_manifest.get("exclusion", {}).get("test_artifact_opened") is not False:
        raise ValueError("Fresh source did not preserve sealed test isolation")
    source_path = Path(source["artifact_path"]).expanduser().resolve()
    if _file_sha256(source_path) != source["artifact_sha256"]:
        raise ValueError("Fresh source artifact hash mismatch")
    all_rows = _read_jsonl(source_path)
    if len(all_rows) != int(source["artifact_count"]):
        raise ValueError("Fresh source artifact count mismatch")
    by_id = {str(row["record_id"]): row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Fresh source contains duplicate record IDs")

    condition_path = manifest_path.parent / rl_manifest["artifact"]["path"]
    condition_rows = {
        str(row["record_id"]): row for row in _read_jsonl(condition_path)
    }
    if set(conditions) != set(condition_rows):
        raise ValueError("RL condition mapping and contract rows differ")
    missing = sorted(set(conditions) - set(by_id))
    if missing:
        raise ValueError(f"Fresh source is missing RL record IDs: {missing[:3]}")
    selected = []
    for record_id in sorted(conditions):
        record = by_id[record_id]
        row = condition_rows[record_id]
        query_hash = hashlib.sha256(_query_signature(record).encode()).hexdigest()
        supervision_hash = hashlib.sha256(
            _supervision_signature(record).encode()
        ).hexdigest()
        target_hash = hashlib.sha256(_target_action(record).encode()).hexdigest()
        if query_hash != row["query_sha256"]:
            raise ValueError(f"RL query identity mismatch for {record_id}")
        if supervision_hash != row["supervision_sha256"]:
            raise ValueError(f"RL supervision identity mismatch for {record_id}")
        if target_hash != row["target_sha256"]:
            raise ValueError(f"RL target identity mismatch for {record_id}")
        selected.append({
            **record,
            "condition_value": str(conditions[record_id]),
            "topology": str(row["topology"]),
        })
    if len(selected) != int(rl_ref["count"]):
        raise ValueError("Frozen RL train count mismatch")

    loaded, parent_tree = _require_parent(
        parent,
        config=config,
        expected_tree_sha256=str(alpha_freeze["parent_checkpoint_tree_sha256"]),
    )
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_experiment_2_grpo_lineage",
        "branch": branch,
        "preflight_path": str(preflight_path),
        "preflight_sha256": _file_sha256(preflight_path),
        "alpha_freeze_path": str(alpha_freeze_path),
        "alpha_freeze_sha256": _file_sha256(alpha_freeze_path),
        "selected_alpha_idc": EXPECTED_ALPHA,
        "rl_train_manifest_path": str(manifest_path),
        "rl_train_manifest_sha256": rl_hash,
        "rl_train_condition_artifact_sha256": rl_manifest["artifact"]["sha256"],
        "fresh_manifest_path": str(fresh_manifest_path),
        "fresh_manifest_sha256": source["manifest_sha256"],
        "fresh_artifact_path": str(source_path),
        "fresh_artifact_sha256": source["artifact_sha256"],
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_tree_sha256": parent_tree,
        "seed": int(config.seed),
        "reward_seed": REWARD_SEED,
        "final_evaluation_manifest_loaded": False,
    }
    return selected, loaded, lineage


def build_grpo_dataset(records: Sequence[Mapping[str, Any]], *, config, tokenizer):
    from datasets import Dataset

    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")
    pattern_map = dict(zip(patterns["pattern_str"], patterns.index))
    frame = pre_pre_processing(list(records), pattern_map, is_act=True)
    values = {str(row["record_id"]): str(row["condition_value"]) for row in records}
    topology = {str(row["record_id"]): str(row["topology"]) for row in records}
    dataset = Dataset.from_pandas(frame, split="train", preserve_index=False)

    def add_prompt(example):
        record_id = str(example["record_id"])
        condition = values[record_id]
        return {
            "prompt": build_generation_prompt(
                example["source"],
                ConditionSpec("specific_relation", condition),
                tokenizer,
                condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
            ),
            "condition": condition,
            "topology": topology[record_id],
        }

    return dataset.map(add_prompt)


class PairedGRPORewardAdapter:
    """One-pass base reward plus the optional value-level SC-IDC component."""

    def __init__(
        self,
        graph_sampler,
        *,
        branch: str,
        progress_path: Path,
        alpha_idc: float = EXPECTED_ALPHA,
        reward_seed: int = REWARD_SEED,
        cache_entries: int = CACHE_ENTRIES,
    ) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}")
        self.branch = branch
        self.alpha_idc = float(alpha_idc)
        self.reward_seed = int(reward_seed)
        self.progress_path = progress_path
        self.cache = CachedQueryExecutor(
            graph_sampler, max_entries=int(cache_entries)
        )
        self.matcher = StaticRelationMatcher(graph_sampler)
        self.started = time.monotonic()
        self.calls = 0
        self.completions = 0
        self.parse_ok = 0
        self.nominal = 0
        self.raw_bss_sum = 0.0
        self.classifications: Counter[str] = Counter()
        self._pending_key = None
        self._pending_rows: list[dict[str, Any]] | None = None

    @staticmethod
    def _signature(
        completions: Sequence[str],
        source: Sequence[str],
        condition: Sequence[str],
        record_id: Sequence[str],
    ) -> str:
        payload = [list(completions), list(source), list(condition), list(record_id)]
        return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()

    @staticmethod
    def _observation(source: str) -> frozenset[int]:
        shifted = [int(token) for token in str(source).split() if token]
        return frozenset(ans_unshift_indices(shifted))

    def base_reward(
        self,
        prompts,
        completions,
        source,
        target,
        condition,
        record_id,
        **kwargs,
    ):
        del prompts, target, kwargs
        signature = self._signature(completions, source, condition, record_id)
        rows = []
        rewards = []
        for completion, answer_text, value, row_id in zip(
            completions, source, condition, record_id
        ):
            observation = self._observation(answer_text)
            query = None
            denotation = None
            nominal = False
            scores = {"jaccard": 0.0, "dice": 0.0, "overlap": 0.0}
            try:
                query = parse_action(str(completion))
                denotation, _ = self.cache.execute(query)
                scores = set_semantic_scores(denotation, observation)
                nominal = bool(value_occurrences(
                    query, "specific_relation", int(value)
                ))
                self.parse_ok += 1
                self.nominal += int(nominal)
            except Exception:
                query = None
                denotation = None
                nominal = False
            reward = (
                scores["jaccard"]
                + 0.5 * scores["dice"]
                + 0.5 * scores["overlap"]
                + float(nominal)
            )
            rewards.append(float(reward))
            rows.append({
                "query": query,
                "denotation": denotation,
                "observation": observation,
                "condition": int(value),
                "record_id": str(row_id),
                "nominal": nominal,
            })
        self.calls += 1
        self.completions += len(rewards)
        self._pending_key = signature
        self._pending_rows = rows
        if self.branch == "baseline":
            self._pending_key = None
            self._pending_rows = None
        self._persist_progress()
        LOGGER.info(
            "GRPO base reward branch=%s mean=%.6f parse_rate=%.6f nominal_rate=%.6f",
            self.branch,
            sum(rewards) / len(rewards) if rewards else 0.0,
            self.parse_ok / self.completions if self.completions else 0.0,
            self.nominal / self.completions if self.completions else 0.0,
        )
        return rewards

    def sc_idc_reward(
        self,
        prompts,
        completions,
        source,
        target,
        condition,
        record_id,
        **kwargs,
    ):
        del prompts, target, kwargs
        signature = self._signature(completions, source, condition, record_id)
        if signature != self._pending_key or self._pending_rows is None:
            raise RuntimeError("SC-IDC reward was not paired with its base-reward batch")
        values = []
        for row in self._pending_rows:
            raw_bss = 0.0
            if row["query"] is not None and row["nominal"]:
                audit = audit_relation_value(
                    query=row["query"],
                    observation=row["observation"],
                    condition_value=row["condition"],
                    matcher=self.matcher,
                    cache=self.cache,
                    record_id=row["record_id"],
                    reward_seed=self.reward_seed,
                    base_denotation=row["denotation"],
                )
                raw_bss = float(audit["raw_bss"])
                self.classifications[audit["classification"]] += 1
            values.append(raw_bss)
            self.raw_bss_sum += raw_bss
        self._pending_key = None
        self._pending_rows = None
        self._persist_progress()
        LOGGER.info(
            "GRPO SC-IDC raw reward mean=%.6f alpha_idc=%.6f graph_executions=%d",
            sum(values) / len(values) if values else 0.0,
            self.alpha_idc,
            self.cache.executions,
        )
        return values

    def reward_functions(self):
        if self.branch == "baseline":
            return [self.base_reward], [1.0]
        return [self.base_reward, self.sc_idc_reward], [1.0, self.alpha_idc]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_reward_progress",
            "updated_at": _utc_now(),
            "branch": self.branch,
            "alpha_idc": 0.0 if self.branch == "baseline" else self.alpha_idc,
            "reward_seed": self.reward_seed,
            "calls": self.calls,
            "completions": self.completions,
            "parse_ok_rate": self.parse_ok / self.completions if self.completions else None,
            "nominal_adherence_rate": (
                self.nominal / self.completions if self.completions else None
            ),
            "mean_raw_bss": (
                self.raw_bss_sum / self.completions if self.completions else None
            ),
            "classification_counts": dict(sorted(self.classifications.items())),
            "graph": {
                "requests": self.cache.requests,
                "executions": self.cache.executions,
                "cache_hits": self.cache.requests - self.cache.executions,
                "cache_entries": len(self.cache.cache),
                "cache_limit": self.cache.max_entries,
                "cache_evictions": self.cache.evictions,
            },
            "elapsed_seconds": time.monotonic() - self.started,
        }

    def _persist_progress(self) -> None:
        _write_json(self.progress_path, self.snapshot())


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def run_branch(args: argparse.Namespace) -> Path:
    if args.branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    code = _git_state()
    if code["dirty"]:
        raise ValueError("Experiment 2 requires a clean Git worktree")
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"Checkpoint directory is not empty: {checkpoint_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "training.log")
    status_path = run_dir / "status.json"
    _write_json(status_path, {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_experiment_2_branch_status",
        "branch": args.branch,
        "status": "preflight",
        "started_at": _utc_now(),
        "code_sha": code["sha"],
        "pid": os.getpid(),
    })
    try:
        config = load_experiment_config(args.experiment_config)
        records, loaded, lineage = load_grpo_contract(
            config=config,
            branch=args.branch,
            preflight_path=Path(args.phase2_preflight).expanduser().resolve(),
            alpha_freeze_path=Path(args.alpha_freeze).expanduser().resolve(),
            parent=Path(args.parent_checkpoint).expanduser().resolve(),
        )
        seed_everything(config.seed)
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        dataset = build_grpo_dataset(records, config=config, tokenizer=loaded.tokenizer)
        graph_samplers = _graph_samplers(config)
        adapter = PairedGRPORewardAdapter(
            graph_samplers["train"],
            branch=args.branch,
            progress_path=run_dir / "reward-progress.json",
        )
        reward_functions, reward_weights = adapter.reward_functions()
        from trl import GRPOTrainer
        from trl.trainer.utils import disable_dropout_in_model

        loaded.tokenizer.padding_side = "left"
        disable_dropout_in_model(loaded.model)
        training_args = build_grpo_config(
            config,
            checkpoint_dir,
            max_steps=int(args.max_steps),
            save_steps=int(args.save_steps) if args.save_steps else None,
        )
        training_args.reward_weights = reward_weights
        trainer = GRPOTrainer(
            model=loaded.model,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=dataset,
            processing_class=loaded.tokenizer,
        )
        contract = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_branch_manifest",
            "status": "running",
            "created_at": _utc_now(),
            "code": code,
            "branch": args.branch,
            "method_name": (
                "uniform-value control + original-reward baseline"
                if args.branch == "baseline"
                else "SC-IDC GRPO"
            ),
            "lineage": lineage,
            "dataset_count": len(dataset),
            "optimizer_updates": {
                "max_steps": int(args.max_steps),
                "epochs": float(config.raw["grpo"]["epochs"]),
            },
            "reward": {
                "base": "jaccard + 0.5*dice + 0.5*overlap + nominal",
                "alpha_idc": 0.0 if args.branch == "baseline" else EXPECTED_ALPHA,
                "tau_marg": 0.1,
                "tau_match": 0.1,
                "epsilon": 0.0,
                "matched_replacements": 3,
            },
            "generation_and_optimizer_config": config.raw["grpo"],
            "final_evaluation_manifest_loaded": False,
            "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
        }
        _write_json(run_dir / "manifest.json", contract)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_branch_status",
            "branch": args.branch,
            "status": "running",
            "started_at": _utc_now(),
            "code_sha": code["sha"],
            "pid": os.getpid(),
            "dataset_count": len(dataset),
            "checkpoint_dir": str(checkpoint_dir),
        })
        train_grpo(trainer, resume_checkpoint=None)
        loaded.tokenizer.padding_side = "right"
        loaded.tokenizer.backend_tokenizer.no_padding()
        evaluation_checkpoint = checkpoint_dir / f"evaluation-step-{int(trainer.state.global_step)}"
        save_reproduction_checkpoint(
            evaluation_checkpoint,
            model=trainer.model,
            tokenizer=loaded.tokenizer,
            stage="grpo",
            stage_epoch=int(config.raw["grpo"]["epochs"]),
            global_step=int(trainer.state.global_step),
            condition=config.condition,
            experiment_config=config.raw,
            experiment_config_hash=config.semantic_hash,
            seed=config.seed,
            parent_checkpoint=args.parent_checkpoint,
            data_manifest_hash=_file_sha256(config.sampling_manifest_path),
            condition_lineage=lineage,
        )
        reward_summary = adapter.snapshot()
        _write_json(run_dir / "reward-summary.json", reward_summary)
        contract["status"] = "completed"
        contract["completed_at"] = _utc_now()
        contract["global_step"] = int(trainer.state.global_step)
        contract["evaluation_checkpoint"] = str(evaluation_checkpoint)
        contract["reward_summary_sha256"] = _file_sha256(
            run_dir / "reward-summary.json"
        )
        _write_json(run_dir / "manifest.json", contract)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_branch_status",
            "branch": args.branch,
            "status": "completed",
            "completed_at": _utc_now(),
            "code_sha": code["sha"],
            "pid": os.getpid(),
            "global_step": int(trainer.state.global_step),
            "evaluation_checkpoint": str(evaluation_checkpoint),
        })
        return evaluation_checkpoint
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_branch_status",
            "branch": args.branch,
            "status": "failed",
            "failed_at": _utc_now(),
            "code_sha": code["sha"],
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--phase2-preflight", required=True)
    parser.add_argument("--alpha-freeze", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int)
    return parser


def main() -> None:
    result = run_branch(build_parser().parse_args())
    print(result)


if __name__ == "__main__":
    main()

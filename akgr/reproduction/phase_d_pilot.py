"""Run an isolated, validation-only pilot for a repaired Phase D contract.

The pilot changes exactly two inputs relative to the archived Phase D run:

* raw GRPO prompts end with the causal target-boundary ``SEP`` token;
* policy optimization uses freshly sampled train-graph queries that do not
  occur in the SFT train data or the validation set.

It never loads the test artifact.  A rollout signal gate runs before any
optimizer step, and the parent and terminal model are compared only on the
fixed validation split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from akgr.abduction_model.experiment_runner import (
    _aggregate_evaluation_metrics,
    _datasets,
    _evaluation_records,
    _file_sha256,
    _graph_samplers,
    _loader,
    _require_cuda,
    _require_phase_d_parent_checkpoint,
)
from akgr.abduction_model.reproduction import (
    configure_reproduction_logging,
    create_grpo_trainer,
)
from akgr.dataloader import _load_manifest_artifact, pre_pre_processing
from akgr.evaluation import parse_action_status, scoring_input_act_batch_condition, write_evaluation_jsonl
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.sampling.sample_parallel import _patterns, run_tasks
from akgr.tokenizer import build_generation_prompt, condition_value_from_target
from akgr.utils.load_util import load_reproduction_checkpoint


PILOT_SCHEMA_VERSION = 1
SEMANTIC_METRICS = ("jaccard", "dice", "overlap")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _query_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({"query": record["query"], "pattern_str": record["pattern_str"]})


def _supervision_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({
        "answers": record["answers"],
        "query": record["query"],
        "pattern_str": record["pattern_str"],
    })


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _load_forbidden_queries(config) -> tuple[set[str], dict[str, Any]]:
    """Load SFT train and validation identities without opening test records."""
    manifest_path = config.sampling_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    train = _load_manifest_artifact(manifest_path, artifacts["merged"]["train"])
    valid = _load_manifest_artifact(manifest_path, artifacts["base"]["valid"])
    forbidden = {_query_signature(record) for record in train}
    forbidden.update(_query_signature(record) for record in valid)
    audit = {
        "policy": "query+pattern disjoint from merged SFT train and base validation",
        "test_artifact_opened": False,
        "source_counts": {"merged_train": len(train), "base_valid": len(valid)},
        "source_artifacts": {
            "merged_train": dict(artifacts["merged"]["train"]),
            "base_valid": dict(artifacts["base"]["valid"]),
        },
        "forbidden_query_count": len(forbidden),
        "forbidden_query_set_sha256": _sha256_bytes(
            "\n".join(sorted(forbidden)).encode("utf-8")
        ),
    }
    return forbidden, audit


def sample_fresh_rl_records(
    *,
    config,
    graph_samplers,
    count_per_pattern: int,
    forbidden_queries: set[str],
    workers: int,
    sampling_namespace: str = "phase_d_repaired_pilot",
    max_rounds: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample an exact balanced RL-only set and reject all identity overlap."""
    patterns = _patterns("akgr/metadata/pattern_table.csv")
    accepted: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern, _ in patterns}
    accepted_queries: set[str] = set()
    accepted_supervision: set[str] = set()
    rejected_forbidden = 0
    rejected_duplicate_query = 0
    rejected_duplicate_supervision = 0
    next_ordinal = defaultdict(int)
    rounds = 0

    while any(len(accepted[pattern]) < int(count_per_pattern) for pattern, _ in patterns):
        rounds += 1
        if rounds > int(max_rounds):
            missing = {
                abbreviation: int(count_per_pattern) - len(accepted[pattern])
                for pattern, abbreviation in patterns
                if len(accepted[pattern]) < int(count_per_pattern)
            }
            raise RuntimeError(
                f"Unable to fill fresh RL dataset after {max_rounds} rounds: {missing}"
            )
        tasks = []
        for pattern, abbreviation in patterns:
            missing = int(count_per_pattern) - len(accepted[pattern])
            for _ in range(missing):
                ordinal = next_ordinal[abbreviation]
                next_ordinal[abbreviation] += 1
                tasks.append({
                    "task_index": len(tasks),
                    "dataset": config.dataset,
                    "mode": "train",
                    "pattern_str": pattern,
                    "pattern_abbr": abbreviation,
                    "ordinal": ordinal,
                    "task_seed": derive_seed(
                        config.seed, sampling_namespace, abbreviation, ordinal
                    ),
                    "max_answers": int(config.raw["data"]["max_answers"]),
                    "max_attempts": int(config.raw["data"]["max_attempts_per_record"]),
                })
        sampled, failures = run_tasks(tasks, graph_samplers, int(workers))
        if failures:
            raise RuntimeError(f"Fresh RL sampling failed for {len(failures)} tasks")
        for record in sampled:
            pattern = str(record["pattern_str"])
            if len(accepted[pattern]) >= int(count_per_pattern):
                continue
            query_signature = _query_signature(record)
            supervision_signature = _supervision_signature(record)
            if query_signature in forbidden_queries:
                rejected_forbidden += 1
                continue
            if query_signature in accepted_queries:
                rejected_duplicate_query += 1
                continue
            if supervision_signature in accepted_supervision:
                rejected_duplicate_supervision += 1
                continue
            accepted[pattern].append(record)
            accepted_queries.add(query_signature)
            accepted_supervision.add(supervision_signature)

    records = [record for pattern, _ in patterns for record in accepted[pattern]]
    counts = {abbreviation: len(accepted[pattern]) for pattern, abbreviation in patterns}
    if set(counts.values()) != {int(count_per_pattern)}:
        raise AssertionError(f"Fresh RL pattern imbalance: {counts}")
    if any(_query_signature(record) in forbidden_queries for record in records):
        raise AssertionError("Fresh RL data overlaps a forbidden query")
    audit = {
        "seed_namespace": sampling_namespace,
        "max_rounds": int(max_rounds),
        "rounds": rounds,
        "count": len(records),
        "count_per_pattern": int(count_per_pattern),
        "counts_by_pattern_abbreviation": counts,
        "unique_query_count": len(accepted_queries),
        "unique_supervision_count": len(accepted_supervision),
        "overlap_with_sft_or_validation": 0,
        "internal_query_duplicates": 0,
        "internal_supervision_duplicates": 0,
        "rejected_forbidden": rejected_forbidden,
        "rejected_duplicate_query": rejected_duplicate_query,
        "rejected_duplicate_supervision": rejected_duplicate_supervision,
    }
    return records, audit


def _pilot_dataset(records, config, tokenizer) -> Dataset:
    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")
    pattern_to_id = dict(zip(patterns["pattern_str"], patterns.index))
    frame = pre_pre_processing(records, pattern_to_id, is_act=True)
    dataset = Dataset.from_pandas(frame, split="train", preserve_index=False)

    def add_prompt(example):
        value = condition_value_from_target(config.condition, example["target"])
        return {
            "prompt": build_generation_prompt(
                example["source"],
                ConditionSpec(config.condition, value),
                tokenizer,
                condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
            ),
            "condition": value,
        }

    return dataset.map(add_prompt)


def _rollout_signal_audit(
    *, trainer, tokenizer, graph_samplers, prompt_groups: int, output_path: Path
) -> dict[str, Any]:
    group_size = int(trainer.num_generations)
    if int(prompt_groups) <= 0:
        raise ValueError("signal prompt count must be positive")
    completions_all: list[str] = []
    rows_all: list[dict[str, Any]] = []
    group_unique_counts: list[int] = []
    group_zero_variance: list[bool] = []
    eos_flags: list[bool] = []
    sample_rows: list[dict[str, Any]] = []
    weights = {"jaccard": 1.0, "dice": 0.5, "overlap": 0.5, "validity": 1.0}

    for inputs in trainer.get_train_dataloader():
        prepared = trainer._generate_and_score_completions(inputs)
        decoded = tokenizer.batch_decode(prepared["completion_ids"], skip_special_tokens=True)
        if len(decoded) % group_size:
            raise AssertionError("GRPO rollout batch split a generation group")
        remaining_groups = int(prompt_groups) - len(group_unique_counts)
        keep = min(len(decoded) // group_size, remaining_groups) * group_size
        decoded = decoded[:keep]
        kept_inputs = inputs[:keep]
        completion_ids = prepared["completion_ids"][:keep]
        scores, _ = scoring_input_act_batch_condition(
            pred_word_batch=decoded,
            label_word_batch=[row["target"] for row in kept_inputs],
            ans_word_batch=[row["source"] for row in kept_inputs],
            condition_batch=[row["condition"] for row in kept_inputs],
            scoring_method=["jaccard", "dice", "overlap", "validity"],
            graph_samplers=graph_samplers,
            searching_split="train",
            return_failures=True,
        )
        rewards = [
            sum(weights[name] * float(score[name]) for name in weights)
            for score in scores
        ]
        completions_all.extend(decoded)
        rows_all.extend(scores)
        eos_flags.extend(
            bool((row == tokenizer.eos_token_id).any().item()) for row in completion_ids
        )
        for start in range(0, keep, group_size):
            group_completions = decoded[start:start + group_size]
            group_rewards = rewards[start:start + group_size]
            group_unique_counts.append(len(set(group_completions)))
            group_zero_variance.append(max(group_rewards) - min(group_rewards) <= 1e-8)
            if len(sample_rows) < 128:
                sample_rows.append({
                    "prompt": kept_inputs[start]["prompt"],
                    "source": kept_inputs[start]["source"],
                    "target": kept_inputs[start]["target"],
                    "condition": kept_inputs[start]["condition"],
                    "completions": group_completions,
                    "combined_rewards": group_rewards,
                })
        if len(group_unique_counts) >= int(prompt_groups):
            break

    if len(group_unique_counts) != int(prompt_groups):
        raise ValueError(
            f"Requested {prompt_groups} rollout groups, observed {len(group_unique_counts)}"
        )
    parse_rate = sum(parse_action_status(value)[0] for value in completions_all) / len(completions_all)
    combined_rewards = [
        sum(weights[name] * float(score[name]) for name in weights)
        for score in rows_all
    ]
    audit = {
        "prompt_group_count": len(group_unique_counts),
        "completion_count": len(completions_all),
        "num_generations": group_size,
        "zero_reward_variance_group_rate": sum(group_zero_variance) / len(group_zero_variance),
        "all_strings_identical_group_rate": sum(value == 1 for value in group_unique_counts) / len(group_unique_counts),
        "mean_unique_strings_per_group": sum(group_unique_counts) / len(group_unique_counts),
        "mean_combined_reward": sum(combined_rewards) / len(combined_rewards),
        "mean_jaccard": sum(float(row["jaccard"]) for row in rows_all) / len(rows_all),
        "mean_dice": sum(float(row["dice"]) for row in rows_all) / len(rows_all),
        "mean_overlap": sum(float(row["overlap"]) for row in rows_all) / len(rows_all),
        "condition_accuracy": sum(float(row["validity"]) for row in rows_all) / len(rows_all),
        "parse_rate": parse_rate,
        "eos_rate": sum(eos_flags) / len(eos_flags),
    }
    audit["gates"] = {
        "zero_reward_variance_group_rate_lte_0.70": audit["zero_reward_variance_group_rate"] <= 0.70,
        "mean_unique_strings_per_group_gte_1.50": audit["mean_unique_strings_per_group"] >= 1.50,
        "parse_rate_gte_0.95": audit["parse_rate"] >= 0.95,
        "eos_rate_gte_0.98": audit["eos_rate"] >= 0.98,
    }
    audit["gate_pass"] = all(audit["gates"].values())
    _write_jsonl(output_path, sample_rows)
    return audit


def _evaluate_model(
    *, config, model, tokenizer, graph_samplers, device, output_dir: Path,
    label: str, split: str = "valid", seed_namespace: str = "phase_d_repaired_pilot",
):
    if split not in {"valid", "test"}:
        raise ValueError("Evaluation split must be valid or test")
    dataset_dict, _, _ = _datasets(config, [split], train_variant="base")
    loader = _loader(
        dataset_dict[split], config.raw["training"]["validation"]["batch_size"],
        config.seed, False,
    )
    output: dict[str, Any] = {}
    for decode, do_sample in (("greedy", False), ("sampled", True)):
        seed_everything(derive_seed(config.seed, seed_namespace, split, decode))
        records = _evaluation_records(
            config=config,
            dataloader=loader,
            model=model,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            device=device,
            split=split,
            condition_kind=config.condition,
            do_sample=do_sample,
        )
        write_evaluation_jsonl(output_dir / f"{label}-{split}-{decode}.jsonl", records)
        metrics = _aggregate_evaluation_metrics(records)
        _write_json(output_dir / f"{label}-{split}-{decode}.metrics.json", metrics)
        output[decode] = {"metrics": metrics, "records": records}
    return output


def _paired_bootstrap(values: list[float], *, seed: int, samples: int = 10_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = []
    for start in range(0, int(samples), 500):
        size = min(500, int(samples) - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means.append(array[indices].mean(axis=1))
    distribution = np.concatenate(means)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_samples": int(samples),
    }


def _compare_validation(baseline, candidate, *, seed: int) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for decode in ("greedy", "sampled"):
        base_metrics = baseline[decode]["metrics"]
        candidate_metrics = candidate[decode]["metrics"]
        deltas = {
            key: float(candidate_metrics[key]) - float(base_metrics[key])
            for key in (*SEMANTIC_METRICS, "condition_accuracy", "smatch", "parse_ok", "eos_rate")
        }
        deltas["semantic_average"] = sum(deltas[name] for name in SEMANTIC_METRICS) / 3.0
        paired = []
        for base_record, candidate_record in zip(
            baseline[decode]["records"], candidate[decode]["records"]
        ):
            if base_record["record_id"] != candidate_record["record_id"]:
                raise AssertionError("Validation records are not paired")
            base_average = sum(float(base_record[name]) for name in SEMANTIC_METRICS) / 3.0
            candidate_average = sum(float(candidate_record[name]) for name in SEMANTIC_METRICS) / 3.0
            paired.append(candidate_average - base_average)
        comparison[decode] = {
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "delta": deltas,
            "paired_semantic_average": _paired_bootstrap(
                paired, seed=derive_seed(seed, "bootstrap", decode)
            ),
        }

    health_pass = all(
        float(comparison[decode]["candidate"]["parse_ok"]) >= 0.98
        and float(comparison[decode]["candidate"]["eos_rate"]) >= 0.98
        and float(comparison[decode]["delta"]["condition_accuracy"]) >= -0.01
        for decode in ("greedy", "sampled")
    )
    greedy_gain = float(comparison["greedy"]["delta"]["semantic_average"])
    sampled_gain = float(comparison["sampled"]["delta"]["semantic_average"])
    if not health_pass:
        decision = "negative_health_regression"
    elif greedy_gain >= 0.005 and sampled_gain > 0.0:
        decision = "promising_repaired_phase_d"
    elif greedy_gain > 0.0 or sampled_gain > 0.0:
        decision = "inconclusive_small_positive"
    else:
        decision = "negative_no_validation_gain"
    comparison["decision"] = {
        "classification": decision,
        "health_pass": health_pass,
        "promising_rule": "greedy semantic-average delta >= 0.005 and sampled delta > 0",
    }
    return comparison


def _relative_parameter_delta(initial_state: Mapping[str, torch.Tensor], model) -> dict[str, float]:
    delta_squared = 0.0
    initial_squared = 0.0
    final_squared = 0.0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            initial = initial_state[name].to(dtype=torch.float64)
            final = parameter.detach().cpu().to(dtype=torch.float64)
            delta_squared += float(torch.sum((final - initial) ** 2))
            initial_squared += float(torch.sum(initial ** 2))
            final_squared += float(torch.sum(final ** 2))
    return {
        "relative_l2_delta": math.sqrt(delta_squared) / math.sqrt(initial_squared),
        "initial_l2": math.sqrt(initial_squared),
        "final_l2": math.sqrt(final_squared),
    }


def run_pilot(args: argparse.Namespace) -> Path:
    config = load_experiment_config(args.experiment_config)
    parent = _require_phase_d_parent_checkpoint(args.parent_checkpoint)
    control_dir = Path(args.control_dir).expanduser().resolve()
    control_dir.mkdir(parents=True, exist_ok=False)
    status_path = control_dir / "status.json"
    status: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "pid": os.getpid(),
        "command": [sys.executable, "-m", "akgr.reproduction.phase_d_pilot", *sys.argv[1:]],
    }
    _write_json(status_path, status)

    try:
        seed_everything(config.seed)
        configure_reproduction_logging(control_dir / "reproduction.log")
        graph_samplers = _graph_samplers(config)
        forbidden, exclusion_audit = _load_forbidden_queries(config)
        extra_forbidden_counts = {}
        for raw_path in args.extra_forbidden_jsonl:
            path = Path(raw_path).expanduser().resolve()
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            forbidden.update(_query_signature(record) for record in rows)
            extra_forbidden_counts[str(path)] = len(rows)
        if extra_forbidden_counts:
            exclusion_audit["extra_forbidden_jsonl"] = extra_forbidden_counts
            exclusion_audit["forbidden_query_count"] = len(forbidden)
            exclusion_audit["forbidden_query_set_sha256"] = _sha256_bytes(
                "\n".join(sorted(forbidden)).encode("utf-8")
            )
        records, sampling_audit = sample_fresh_rl_records(
            config=config,
            graph_samplers=graph_samplers,
            count_per_pattern=args.fresh_per_pattern,
            forbidden_queries=forbidden,
            workers=args.workers,
            sampling_namespace=args.sampling_namespace,
            max_rounds=args.max_sampling_rounds,
        )
        data_path = control_dir / "fresh-rl-train.jsonl"
        _write_jsonl(data_path, records)
        fresh_manifest = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "kind": "phase_d_repaired_pilot_rl_only",
            "created_at": _utc_now(),
            "config_semantic_hash": config.semantic_hash,
            "original_sampling_manifest_sha256": _file_sha256(config.sampling_manifest_path),
            "exclusion": exclusion_audit,
            "sampling": sampling_audit,
            "artifact": {
                "path": data_path.name,
                "count": len(records),
                "sha256": _file_sha256(data_path),
            },
        }
        _write_json(control_dir / "fresh-rl-manifest.json", fresh_manifest)

        loaded = load_reproduction_checkpoint(
            parent,
            mode="parent",
            expected_stage="conditional",
            expected_condition=config.condition,
            expected_config_hash=config.semantic_hash,
            expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
        )
        model, tokenizer = loaded.model, loaded.tokenizer
        initial_state = {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }
        dataset = _pilot_dataset(records, config, tokenizer)
        if any(not str(prompt).split()[-1] == tokenizer.sep_token for prompt in dataset["prompt"]):
            raise AssertionError("A repaired GRPO prompt does not end with SEP")

        device = _require_cuda("Phase D repaired pilot")
        model.to(device)
        baseline = _evaluate_model(
            config=config, model=model, tokenizer=tokenizer, graph_samplers=graph_samplers,
            device=device, output_dir=control_dir / "validation", label="parent",
        )

        audit_trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            output_dir=control_dir / "rollout-gate-tmp",
            graph_samplers=graph_samplers,
            max_steps=args.max_steps,
        )
        rollout_audit = _rollout_signal_audit(
            trainer=audit_trainer,
            tokenizer=tokenizer,
            graph_samplers=graph_samplers,
            prompt_groups=args.signal_prompts,
            output_path=control_dir / "rollout-samples.jsonl",
        )
        _write_json(control_dir / "rollout-signal-audit.json", rollout_audit)
        del audit_trainer
        torch.cuda.empty_cache()
        if not rollout_audit["gate_pass"]:
            raise RuntimeError(f"Rollout signal gate failed: {rollout_audit['gates']}")

        seed_everything(config.seed)
        trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            output_dir=control_dir / "grpo",
            graph_samplers=graph_samplers,
            max_steps=args.max_steps,
        )
        trainer.train()
        tokenizer.padding_side = "right"
        tokenizer.backend_tokenizer.no_padding()
        final_model_dir = control_dir / "grpo" / "final-model"
        trainer.save_model(str(final_model_dir))
        trainer.save_state()
        if int(trainer.state.global_step) != int(args.max_steps):
            raise AssertionError(
                f"Pilot stopped at step {trainer.state.global_step}, expected {args.max_steps}"
            )

        candidate = _evaluate_model(
            config=config, model=trainer.model, tokenizer=tokenizer,
            graph_samplers=graph_samplers, device=device,
            output_dir=control_dir / "validation", label="pilot",
        )
        comparison = _compare_validation(baseline, candidate, seed=config.seed)
        parameter_delta = _relative_parameter_delta(initial_state, trainer.model)
        _write_json(control_dir / "validation-comparison.json", comparison)

        result = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "kind": "phase_d_repaired_pilot",
            "status": "completed",
            "finished_at": _utc_now(),
            "code_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config": {
                "path": str(config.source_path),
                "semantic_hash": config.semantic_hash,
                "data_hash": config.data_hash,
                "kg_hash": config.kg_hash,
            },
            "parent_checkpoint": str(parent),
            "parent_metadata": loaded.metadata,
            "interventions": {
                "explicit_target_boundary_sep": True,
                "fresh_rl_only_data": True,
                "test_evaluation_or_artifact_access": False,
            },
            "training": {
                "max_steps": int(args.max_steps),
                "fresh_per_pattern": int(args.fresh_per_pattern),
                "fresh_record_count": len(records),
                "learning_rate": float(config.raw["grpo"]["learning_rate"]),
                "lr_scheduler": "linear",
                "beta": float(config.raw["grpo"]["beta"]),
                "num_generations": 4,
                "per_device_train_batch_size": int(
                    config.raw["grpo"]["per_device_train_batch_size"]
                ),
                "global_step": int(trainer.state.global_step),
            },
            "fresh_data_manifest": fresh_manifest,
            "rollout_signal": rollout_audit,
            "parameter_delta": parameter_delta,
            "validation": comparison,
            "final_model": str(final_model_dir),
        }
        result_path = control_dir / "pilot-result.json"
        _write_json(result_path, result)
        status.update({
            "status": "completed",
            "finished_at": result["finished_at"],
            "result": result_path.name,
            "decision": comparison["decision"]["classification"],
        })
        _write_json(status_path, status)
        return result_path
    except BaseException as exc:
        status.update({
            "status": "failed",
            "finished_at": _utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        _write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--fresh-per-pattern", type=int, default=1024)
    parser.add_argument("--signal-prompts", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1664)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sampling-namespace", default="phase_d_repaired_pilot")
    parser.add_argument("--max-sampling-rounds", type=int, default=100)
    parser.add_argument("--extra-forbidden-jsonl", action="append", default=[])
    args = parser.parse_args()
    if (
        args.fresh_per_pattern <= 0
        or args.signal_prompts <= 0
        or args.max_steps <= 0
        or args.max_sampling_rounds <= 0
    ):
        parser.error(
            "fresh-per-pattern, signal-prompts, max-steps, and max-sampling-rounds "
            "must be positive"
        )
    result = run_pilot(args)
    print(result)


if __name__ == "__main__":
    main()

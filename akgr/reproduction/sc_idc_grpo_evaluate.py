"""Validation-only paired evaluation for SC-IDC Experiment 2.

This command evaluates two completed GRPO branches on the frozen validation
conditions.  It deliberately has no option for the sealed final-evaluation
manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from akgr.abduction_model.experiment_runner import (
    _aggregate_evaluation_metrics,
    _datasets,
    _evaluation_records,
    _file_sha256,
    _graph_samplers,
    _loader,
)
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc import parse_action, value_occurrences
from akgr.reproduction.sc_idc_grpo import EXPECTED_ALPHA, REWARD_SEED
from akgr.reproduction.sc_idc_phase2_data import (
    checkpoint_tree_sha256,
    load_frozen_condition_manifest,
    verify_frozen_validation_dataset,
)
from akgr.reproduction.sc_idc_phase2_reward import (
    CachedQueryExecutor,
    StaticRelationMatcher,
    audit_relation_value,
)
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.utils.load_util import load_reproduction_checkpoint
from akgr.utils.parsing_util import ans_unshift_indices


SCHEMA_VERSION = 1
SEMANTIC_METRICS = ("jaccard", "dice", "overlap")
TOPOLOGIES = ("first_only", "non_first_only", "repeated")
DECODE_MODES = (("greedy", False), ("sampled", True))
BOOTSTRAP_SAMPLES = 10_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        count = 0
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": _file_sha256(path)}


def _git_state() -> dict[str, Any]:
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()),
    }


def _read_condition_rows(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact_path = (manifest_path.parent / manifest["artifact"]["path"]).resolve()
    rows: dict[str, dict[str, Any]] = {}
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in rows:
                raise ValueError(f"Duplicate validation record ID at line {line_number}")
            rows[record_id] = row
    if len(rows) != int(manifest["artifact"]["count"]):
        raise ValueError("Frozen validation condition row count mismatch")
    return rows


def _load_validation_contract(config, preflight_path: Path):
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
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
    reference = preflight["conditions"]["validation"]
    if reference.get("sealed") is not False:
        raise ValueError("Validation conditions unexpectedly marked sealed")
    manifest_path = (preflight_path.parent / reference["manifest_path"]).resolve()
    conditions, manifest = load_frozen_condition_manifest(
        manifest_path,
        target_config=config,
        purpose="validation",
        consumer="reporting",
        expected_manifest_sha256=str(reference["manifest_sha256"]),
    )
    rows = _read_condition_rows(manifest_path, manifest)
    datasets, _, _ = _datasets(config, ["valid"])
    dataset = datasets["valid"]
    verify_frozen_validation_dataset(
        dataset, condition_values=conditions, record_contracts=rows
    )
    if len(dataset) != int(reference["count"]):
        raise ValueError("Frozen validation dataset count mismatch")
    return dataset, conditions, rows, {
        "preflight_path": str(preflight_path),
        "preflight_sha256": _file_sha256(preflight_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(reference["manifest_sha256"]),
        "artifact_sha256": str(reference["artifact_sha256"]),
        "count": len(dataset),
        "topology_counts": manifest["topology_counts"],
        "final_evaluation_manifest_loaded": False,
    }


def _load_branch_checkpoint(
    checkpoint: Path, *, branch: str, config, pair_run_dir: Path, expected_rl_hash: str
):
    run_dir = pair_run_dir / branch
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError(f"{branch} training is not completed")
    if Path(manifest["evaluation_checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError(f"{branch} checkpoint does not match its manifest")
    if manifest.get("final_evaluation_manifest_loaded") is not False:
        raise ValueError(f"{branch} did not preserve sealed evaluation isolation")
    lineage = manifest["lineage"]
    if lineage.get("rl_train_manifest_sha256") != expected_rl_hash:
        raise ValueError(f"{branch} RL manifest hash mismatch")
    if int(lineage.get("seed")) != int(config.seed):
        raise ValueError(f"{branch} seed mismatch")
    loaded = load_reproduction_checkpoint(
        checkpoint,
        mode="test",
        expected_stage="grpo",
        expected_condition="specific_relation",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    metadata = loaded.metadata
    if metadata.get("condition_lineage", {}).get("rl_train_manifest_sha256") != expected_rl_hash:
        raise ValueError(f"{branch} checkpoint RL lineage mismatch")
    if int(metadata["global_step"]) != int(manifest["global_step"]):
        raise ValueError(f"{branch} checkpoint global step mismatch")
    return loaded, manifest, {
        "path": str(checkpoint),
        "tree_sha256": checkpoint_tree_sha256(checkpoint),
        "global_step": int(metadata["global_step"]),
        "parent_checkpoint": str(metadata["parent_checkpoint"]),
        "seed": int(metadata["seed"]),
        "reload_verified": True,
        "manifest_sha256": _file_sha256(run_dir / "manifest.json"),
        "reward_summary_sha256": _file_sha256(run_dir / "reward-summary.json"),
    }


def _observation(text: str) -> frozenset[int]:
    shifted = [int(token) for token in str(text).split() if token]
    return frozenset(ans_unshift_indices(shifted))


def _audit_records(records: Sequence[Mapping[str, Any]], graph_sampler) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matcher = StaticRelationMatcher(graph_sampler)
    cache = CachedQueryExecutor(graph_sampler, max_entries=50_000)
    audited = []
    for record in records:
        prediction = str(record["prediction"])
        condition = int(record["condition"]["value"])
        parse_ok = False
        nominal = False
        classification = None
        branch_delta = None
        matched_delta = None
        raw_bss = 0.0
        try:
            query = parse_action(prediction)
            parse_ok = True
            nominal = bool(value_occurrences(query, "specific_relation", condition))
            if nominal:
                audit = audit_relation_value(
                    query=query,
                    observation=_observation(str(record["observation"])),
                    condition_value=condition,
                    matcher=matcher,
                    cache=cache,
                    record_id=str(record["record_id"]),
                    reward_seed=REWARD_SEED,
                )
                classification = str(audit["classification"])
                branch_delta = float(audit["branch_marginal_delta"])
                matched_delta = (
                    None if audit["matched_delta"] is None
                    else float(audit["matched_delta"])
                )
                raw_bss = float(audit["raw_bss"])
        except Exception:
            parse_ok = False
            nominal = False
        audited.append({
            **record,
            "audit_parse_ok": parse_ok,
            "nominal_adherence": nominal,
            "sc_idc_classification": classification,
            "branch_marginal_delta": branch_delta,
            "matched_delta": matched_delta,
            "raw_bss": raw_bss,
        })
    return audited, {
        "requests": cache.requests,
        "executions": cache.executions,
        "cache_hits": cache.requests - cache.executions,
        "cache_evictions": cache.evictions,
    }


def _audit_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    classifications = Counter(
        record["sc_idc_classification"] for record in records
        if record["sc_idc_classification"] is not None
    )
    nominal = sum(bool(record["nominal_adherence"]) for record in records)
    supported = (
        classifications["branch_supported_nonselective"]
        + classifications["branch_supported_selective"]
    )
    selective = classifications["branch_supported_selective"]
    scorable = sum(classifications.values())
    matched_positive = sum(
        record["matched_delta"] is not None and float(record["matched_delta"]) > 0.0
        for record in records
    )
    return {
        "count": count,
        "nominal_adherence": nominal / count if count else None,
        "scorable_rate": scorable / count if count else None,
        "branch_supported_rate": supported / count if count else None,
        "matched_selectivity_rate": matched_positive / count if count else None,
        "branch_supported_selectivity_rate": selective / count if count else None,
        "branch_nonmarginal_rate": classifications["branch_nonmarginal"] / count if count else None,
        "mean_raw_bss": (
            math.fsum(float(record["raw_bss"]) for record in records) / count
            if count else None
        ),
        "classification_counts": dict(sorted(classifications.items())),
        "conditional_on_nominal": {
            "branch_supported_rate": supported / nominal if nominal else None,
            "matched_selectivity_rate": matched_positive / nominal if nominal else None,
            "branch_supported_selectivity_rate": selective / nominal if nominal else None,
            "branch_nonmarginal_rate": (
                classifications["branch_nonmarginal"] / nominal if nominal else None
            ),
        },
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "standard": _aggregate_evaluation_metrics(records),
        "sc_idc": _audit_metrics(records),
    }


def _paired_bootstrap(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    chunks = []
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        size = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        chunks.append(array[indices].mean(axis=1))
    distribution = np.concatenate(chunks)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def compare_paired_records(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("Paired validation record counts differ")
    semantic_deltas = []
    nominal_deltas = []
    for left, right in zip(baseline, candidate):
        if left["record_id"] != right["record_id"]:
            raise ValueError("Paired validation record IDs differ")
        semantic_deltas.append(
            math.fsum(float(right[key]) - float(left[key]) for key in SEMANTIC_METRICS) / 3.0
        )
        nominal_deltas.append(
            float(right["nominal_adherence"]) - float(left["nominal_adherence"])
        )
    base_summary = _summarize(baseline)
    candidate_summary = _summarize(candidate)
    standard_delta = {
        key: float(candidate_summary["standard"][key]) - float(base_summary["standard"][key])
        for key in (*SEMANTIC_METRICS, "condition_accuracy", "smatch", "parse_ok", "eos_rate", "mean_generated_tokens")
    }
    standard_delta["semantic_average"] = math.fsum(
        standard_delta[key] for key in SEMANTIC_METRICS
    ) / 3.0
    sc_delta = {
        key: float(candidate_summary["sc_idc"][key]) - float(base_summary["sc_idc"][key])
        for key in (
            "nominal_adherence", "scorable_rate", "branch_supported_rate",
            "matched_selectivity_rate", "branch_supported_selectivity_rate",
            "branch_nonmarginal_rate", "mean_raw_bss",
        )
    }
    return {
        "baseline": base_summary,
        "sc_idc": candidate_summary,
        "delta": {"standard": standard_delta, "sc_idc": sc_delta},
        "paired_semantic_average": _paired_bootstrap(semantic_deltas, seed=seed),
        "paired_nominal_adherence": _paired_bootstrap(
            nominal_deltas, seed=derive_seed(seed, "nominal")
        ),
    }


def _evaluate_one(
    *, config, dataset, conditions, condition_rows, loaded, graph_samplers,
    device, branch: str, output_dir: Path
) -> dict[str, Any]:
    loaded.model.to(device)
    loaded.model.eval()
    loader = _loader(
        dataset, config.raw["training"]["validation"]["batch_size"],
        derive_seed(config.seed, "sc-idc-experiment-2-validation-loader"), False,
    )
    result = {}
    for decode, do_sample in DECODE_MODES:
        seed_everything(derive_seed(config.seed, "sc-idc-experiment-2-validation", decode))
        records = _evaluation_records(
            config=config, dataloader=loader, model=loaded.model,
            tokenizer=loaded.tokenizer, graph_samplers=graph_samplers,
            device=device, split="valid", condition_kind="specific_relation",
            do_sample=do_sample, condition_value_by_record_id=conditions,
        )
        for record in records:
            record["topology"] = str(condition_rows[str(record["record_id"])]["topology"])
        records, accounting = _audit_records(records, graph_samplers["train"])
        artifact = _write_jsonl(output_dir / f"{branch}-{decode}.jsonl", records)
        by_topology = {
            topology: _summarize([row for row in records if row["topology"] == topology])
            for topology in TOPOLOGIES
        }
        result[decode] = {
            "summary": _summarize(records),
            "topology": by_topology,
            "graph_accounting": accounting,
            "artifact": artifact,
            "records": records,
        }
    loaded.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> Path:
    code = _git_state()
    if code["dirty"]:
        raise ValueError("Formal validation evaluation requires a clean Git worktree")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _write_json(status_path, {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_experiment_2_validation_status",
        "status": "running", "started_at": _utc_now(), "code_sha": code["sha"],
    })
    try:
        config = load_experiment_config(args.experiment_config)
        preflight_path = Path(args.phase2_preflight).expanduser().resolve()
        dataset, conditions, condition_rows, validation_contract = _load_validation_contract(
            config, preflight_path
        )
        rl_hash = str(json.loads(preflight_path.read_text(encoding="utf-8"))[
            "conditions"
        ]["rl_train"]["manifest_sha256"])
        pair_run_dir = Path(args.pair_run_dir).expanduser().resolve()
        loaded = {}
        branch_manifests = {}
        checkpoint_contracts = {}
        for branch, value in (
            ("baseline", args.baseline_checkpoint), ("sc_idc", args.sc_idc_checkpoint)
        ):
            loaded[branch], branch_manifests[branch], checkpoint_contracts[branch] = (
                _load_branch_checkpoint(
                    Path(value).expanduser().resolve(), branch=branch, config=config,
                    pair_run_dir=pair_run_dir, expected_rl_hash=rl_hash,
                )
            )
        baseline_contract = checkpoint_contracts["baseline"]
        candidate_contract = checkpoint_contracts["sc_idc"]
        for key in ("global_step", "parent_checkpoint", "seed"):
            if baseline_contract[key] != candidate_contract[key]:
                raise ValueError(f"Paired checkpoint {key} mismatch")
        if int(branch_manifests["baseline"]["dataset_count"]) != int(
            branch_manifests["sc_idc"]["dataset_count"]
        ):
            raise ValueError("Paired branch dataset counts differ")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        graph_samplers = _graph_samplers(config)
        evaluations = {}
        for branch in ("baseline", "sc_idc"):
            evaluations[branch] = _evaluate_one(
                config=config, dataset=dataset, conditions=conditions,
                condition_rows=condition_rows, loaded=loaded[branch],
                graph_samplers=graph_samplers, device=device, branch=branch,
                output_dir=output_dir,
            )
        comparisons = {}
        for decode, _ in DECODE_MODES:
            comparisons[decode] = compare_paired_records(
                evaluations["baseline"][decode]["records"],
                evaluations["sc_idc"][decode]["records"],
                seed=derive_seed(config.seed, "sc-idc-experiment-2-bootstrap", decode),
            )
            comparisons[decode]["topology"] = {
                topology: compare_paired_records(
                    [row for row in evaluations["baseline"][decode]["records"] if row["topology"] == topology],
                    [row for row in evaluations["sc_idc"][decode]["records"] if row["topology"] == topology],
                    seed=derive_seed(config.seed, "sc-idc-experiment-2-bootstrap", decode, topology),
                )
                for topology in TOPOLOGIES
            }
        artifacts = {
            branch: {
                decode: evaluations[branch][decode]["artifact"] for decode, _ in DECODE_MODES
            } for branch in ("baseline", "sc_idc")
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_validation_report",
            "status": "completed", "completed_at": _utc_now(), "code": code,
            "method_names": {
                "baseline": "uniform-value control + original-reward baseline",
                "sc_idc": "SC-IDC GRPO",
            },
            "alpha_idc": EXPECTED_ALPHA,
            "seed": int(config.seed),
            "validation": validation_contract,
            "rl_train_manifest_sha256": rl_hash,
            "checkpoints": checkpoint_contracts,
            "training_dataset_count": int(branch_manifests["baseline"]["dataset_count"]),
            "evaluations": {
                branch: {
                    decode: {
                        key: value for key, value in evaluations[branch][decode].items()
                        if key != "records"
                    } for decode, _ in DECODE_MODES
                } for branch in ("baseline", "sc_idc")
            },
            "comparison": comparisons,
            "artifacts": artifacts,
            "final_evaluation_manifest_loaded": False,
            "limitations": [
                "single seed pilot; no stable multi-seed gain claim",
                "natural branch nonmarginality is not labeled laundering",
                "validation-only result; sealed final evaluation remains unopened",
            ],
            "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
        }
        report_path = output_dir / "validation-report.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_validation_status",
            "status": "completed", "completed_at": _utc_now(),
            "code_sha": code["sha"], "report_sha256": _file_sha256(report_path),
            "final_evaluation_manifest_loaded": False,
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_experiment_2_validation_status",
            "status": "failed", "failed_at": _utc_now(), "code_sha": code["sha"],
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "final_evaluation_manifest_loaded": False,
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--phase2-preflight", required=True)
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--sc-idc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(result)


if __name__ == "__main__":
    main()

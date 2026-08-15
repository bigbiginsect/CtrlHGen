"""Run the first-stage, reference-only SC-IDC audit on an observable KG.

The command consumes a repaired Phase-D fresh-data manifest, verifies every
available identity/hash contract, stratifies records by the thirteen patterns,
and audits every eligible entity and relation slot.  It never loads a model,
checkpoint, rollout, validation record, or test artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from akgr.kgdata import GraphSampler, load_kg
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc import (
    MatchedReplacementIndex,
    QueryExecutor,
    audit_slot,
    extract_semantic_slots,
    legacy_denotation,
    parse_action,
    parse_raw_query,
    pattern_signature,
    slot_counts,
)


SCHEMA_VERSION = 1
CONDITION_KINDS = ("specific_entity", "specific_relation")
THRESHOLDS = (0.0, 0.01, 0.05)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_verified_fresh_records(
    manifest_path: Path,
    *,
    expected_config_hash: str,
    expected_sampling_manifest_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Load a fresh-RL artifact only after its provenance and bytes verify."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Fresh-RL manifest schema_version must be 1")
    if manifest.get("kind") != "phase_d_repaired_pilot_rl_only":
        raise ValueError("Input is not a repaired Phase-D fresh-RL manifest")
    if manifest.get("config_semantic_hash") != expected_config_hash:
        raise ValueError("Fresh-RL manifest experiment config hash mismatch")
    if manifest.get("original_sampling_manifest_sha256") != expected_sampling_manifest_hash:
        raise ValueError("Fresh-RL manifest base sampling-manifest hash mismatch")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Fresh-RL manifest is missing its artifact contract")
    data_path = (manifest_path.parent / str(artifact.get("path", ""))).resolve()
    if data_path.parent != manifest_path.parent:
        raise ValueError("Fresh-RL artifact must be colocated with its manifest")
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    actual_hash = _sha256(data_path)
    if actual_hash != artifact.get("sha256"):
        raise ValueError(
            f"Fresh-RL artifact SHA256 mismatch: expected {artifact.get('sha256')}, "
            f"found {actual_hash}"
        )
    rows = []
    with data_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {data_path}:{line_number}") from exc
            required = {"answers", "query", "pattern_str", "record_id"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(
                    f"Fresh-RL record {line_number} is missing {sorted(required)}"
                )
            if (
                not isinstance(row["answers"], list)
                or not all(isinstance(value, int) for value in row["answers"])
                or not isinstance(row["query"], list)
                or not isinstance(row["pattern_str"], str)
                or not isinstance(row["record_id"], str)
                or not row["record_id"]
            ):
                raise ValueError(f"Fresh-RL record {line_number} has invalid field types")
            rows.append(row)
    if len(rows) != int(artifact.get("count", -1)):
        raise ValueError(
            f"Fresh-RL record count mismatch: expected {artifact.get('count')}, "
            f"found {len(rows)}"
        )
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Fresh-RL artifact contains duplicate record_id values")
    return rows, manifest, data_path


def _pattern_abbreviations() -> dict[str, str]:
    pattern_table = REPOSITORY_ROOT / "akgr" / "metadata" / "pattern_table.csv"
    with pattern_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        str(row["original"]): str(row["pattern_abbr"])
        for row in rows
        if int(row["original_depth"]) <= 2
    }
    if len(mapping) != 13:
        raise ValueError(f"Expected 13 audit patterns, found {len(mapping)}")
    return mapping


def _registered_pattern_signature(pattern: str) -> str:
    operators = {"e", "p", "i", "u", "n"}
    return " ".join(character for character in pattern if character in operators)


def stratified_records(
    records: Sequence[dict[str, Any]], *, per_pattern: int, seed: int
) -> list[dict[str, Any]]:
    if int(per_pattern) <= 0:
        raise ValueError("per_pattern must be positive")
    abbreviations = _pattern_abbreviations()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pattern = str(row["pattern_str"])
        if pattern not in abbreviations:
            raise ValueError(f"Fresh-RL record uses an unregistered pattern: {pattern}")
        grouped[pattern].append(row)
    if set(grouped) != set(abbreviations):
        missing = sorted(set(abbreviations) - set(grouped))
        raise ValueError(f"Fresh-RL artifact does not cover all patterns: {missing}")

    selected = []
    for pattern, abbreviation in sorted(abbreviations.items(), key=lambda item: item[1]):
        candidates = sorted(grouped[pattern], key=lambda row: str(row["record_id"]))
        if len(candidates) < int(per_pattern):
            raise ValueError(
                f"Pattern {abbreviation} has {len(candidates)} records, "
                f"fewer than requested {per_pattern}"
            )
        rng = random.Random(_stable_seed(seed, "sc_idc_stratified", abbreviation))
        indices = sorted(rng.sample(range(len(candidates)), int(per_pattern)))
        for index in indices:
            row = dict(candidates[index])
            row["pattern_abbreviation"] = abbreviation
            selected.append(row)
    selected.sort(key=lambda row: (str(row["pattern_abbreviation"]), str(row["record_id"])))
    return selected


def _weighted_sum(rows: Sequence[Mapping[str, Any]], predicate) -> float:
    return sum(float(row["slot_weight"]) for row in rows if predicate(row))


def _weighted_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    eligible = [row for row in rows if row.get(field) is not None]
    denominator = sum(float(row["slot_weight"]) for row in eligible)
    if denominator == 0.0:
        return None
    return sum(float(row["slot_weight"]) * float(row[field]) for row in eligible) / denominator


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_weight = sum(float(row["slot_weight"]) for row in rows)
    if not rows or total_weight <= 0.0:
        raise ValueError("Cannot summarize an empty or zero-weight audit group")
    replacement_weight = sum(
        float(row["slot_weight"]) * len(row["replacements"])
        for row in rows
    )
    thresholds = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:g}"
        thresholds[key] = {
            "marginal_effective_rate": _weighted_sum(
                rows, lambda row: float(row["branch_marginal_delta"]) > threshold
            ) / total_weight,
            "matched_selectivity_rate": _weighted_sum(
                rows,
                lambda row: row.get("matched_delta") is not None
                and float(row["matched_delta"]) > threshold,
            ) / total_weight,
            "strict_effective_rate": _weighted_sum(
                rows,
                lambda row: float(row["branch_marginal_delta"]) > threshold
                and row.get("matched_delta") is not None
                and float(row["matched_delta"]) > threshold,
            ) / total_weight,
            "control_laundering_rate": _weighted_sum(
                rows, lambda row: float(row["branch_marginal_delta"]) <= threshold
            ) / total_weight,
        }
    return {
        "row_count": len(rows),
        "execution_count": sum(int(row["execution_count"]) for row in rows),
        "mean_execution_count": _weighted_mean(rows, "execution_count"),
        "total_slot_weight": total_weight,
        "unscorable_rate": _weighted_sum(
            rows, lambda row: row["classification"] == "unscorable"
        ) / total_weight,
        "reference_exact_rate": _weighted_sum(
            rows, lambda row: bool(row["reference_exact"])
        ) / total_weight,
        "reference_jaccard_one_rate": _weighted_sum(
            rows, lambda row: float(row["base_semantic_score"]) == 1.0
        ) / total_weight,
        "legacy_executor_parity_rate": _weighted_sum(
            rows, lambda row: bool(row["legacy_executor_parity"])
        ) / total_weight,
        "structure_preservation_rate": (
            1.0 - sum(
                float(row["slot_weight"])
                for row in rows
                for candidate in row["replacements"]
                if not candidate["structure_preserved"]
            )
            / replacement_weight
            if replacement_weight > 0.0
            else 0.0
        ),
        "mean_branch_marginal_delta": _weighted_mean(rows, "branch_marginal_delta"),
        "mean_matched_delta": _weighted_mean(rows, "matched_delta"),
        "mean_base_semantic_score": _weighted_mean(rows, "base_semantic_score"),
        "mean_diagnostic_score": _weighted_mean(rows, "diagnostic_score"),
        "thresholds": thresholds,
    }


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    selected_record_count: int,
    shared_reference_execution_count: int = 0,
    legacy_executor_execution_count: int = 0,
) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    dimensions = {
        "condition_kind": lambda row: str(row["condition_kind"]),
        "pattern": lambda row: str(row["pattern_abbreviation"]),
        "branch_operator": lambda row: str(
            row["neutralization"]["branch_operator"] or "root"
        ),
        "slot_position": lambda row: str(row["slot_position"]),
    }
    for dimension, key_fn in dimensions.items():
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[key_fn(row)].append(row)
        by[dimension] = {
            key: _group_summary(group_rows)
            for key, group_rows in sorted(grouped.items())
        }

    weights: defaultdict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        weights[(str(row["record_id"]), str(row["condition_kind"]))] += float(
            row["slot_weight"]
        )
    weight_errors = [abs(value - 1.0) for value in weights.values()]
    fallback_counts = Counter(
        candidate["fallback"]
        for row in rows
        for candidate in row["replacements"]
    )
    overall = _group_summary(rows)
    integrity_gates = {
        "thirteen_patterns": len(by["pattern"]) == 13,
        "reference_exact_100_percent": overall["reference_exact_rate"] == 1.0,
        "reference_jaccard_100_percent_one": (
            overall["reference_jaccard_one_rate"] == 1.0
        ),
        "legacy_executor_parity_100_percent": (
            overall["legacy_executor_parity_rate"] == 1.0
        ),
        "structure_preservation_100_percent": (
            overall["structure_preservation_rate"] == 1.0
        ),
        "no_unscorable_slots": overall["unscorable_rate"] == 0.0,
        "slot_weight_contract": all(error <= 1e-12 for error in weight_errors),
    }
    slot_execution_count = sum(int(row["execution_count"]) for row in rows)
    total_execution_count = (
        slot_execution_count
        + int(shared_reference_execution_count)
        + int(legacy_executor_execution_count)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_reference_audit_summary",
        "evidence_boundary": {
            "reference_queries": True,
            "nominal_adherence_is_trivial": True,
            "model_control_claim_supported": False,
            "test_artifact_accessed": False,
        },
        "selected_record_count": int(selected_record_count),
        "slot_row_count": len(rows),
        "execution_count": total_execution_count,
        "execution_cost": {
            "shared_reference_query_executions": int(shared_reference_execution_count),
            "slot_counterfactual_executions": slot_execution_count,
            "legacy_parity_executions": int(legacy_executor_execution_count),
            "total_executions": total_execution_count,
        },
        "slot_weight_contract": {
            "record_kind_pair_count": len(weights),
            "max_absolute_error": max(weight_errors, default=0.0),
            "pass": all(error <= 1e-12 for error in weight_errors),
        },
        "integrity_gates": {
            **integrity_gates,
            "all_single_run_gates_pass": all(integrity_gates.values()),
            "repeat_run_byte_identity_requires_second_output_directory": True,
        },
        "replacement_fallback_counts": dict(sorted(fallback_counts.items())),
        "overall": overall,
        "by": by,
    }


def _graph_samplers(config):
    data = config.raw["data"]
    return load_kg(
        config.dataset,
        data_root=config.runtime_paths["data_root"],
        seed=config.seed,
        split_ratios=data["split_ratios"],
        reverse_edges_flag=data["reverse_edges"],
        semantic_hash=config.kg_hash,
        offline=True,
    ).graph_samplers


def run_audit(args: argparse.Namespace) -> Path:
    if args.split != "train":
        raise ValueError("First-stage SC-IDC audit is restricted to the train graph")
    if args.condition_kind not in {"both", *CONDITION_KINDS}:
        raise ValueError(f"Unsupported condition kind: {args.condition_kind}")
    if int(args.per_pattern) <= 0 or not 1 <= int(args.replacements) <= 8:
        raise ValueError("per_pattern must be positive and replacements between 1 and 8")
    if int(args.seed) < 0:
        raise ValueError("seed must be non-negative")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started_at = _utc_now()
    started = time.monotonic()

    config = load_experiment_config(args.experiment_config)
    if config.dataset != "WN18RR" or config.condition != "pattern":
        raise ValueError(
            "First-stage SC-IDC audit requires the WN18RR pattern reproduction config"
        )
    sampling_manifest_hash = _sha256(config.sampling_manifest_path)
    input_manifest_path = Path(args.fresh_manifest).expanduser().resolve()
    records, input_manifest, input_path = load_verified_fresh_records(
        input_manifest_path,
        expected_config_hash=config.semantic_hash,
        expected_sampling_manifest_hash=sampling_manifest_hash,
    )
    selected = stratified_records(records, per_pattern=args.per_pattern, seed=args.seed)
    graph_sampler = _graph_samplers(config)["train"]
    matcher = MatchedReplacementIndex(graph_sampler)
    executor = QueryExecutor(graph_sampler)
    condition_kinds = CONDITION_KINDS if args.condition_kind == "both" else (args.condition_kind,)

    audit_rows: list[dict[str, Any]] = []
    for record in selected:
        query = parse_raw_query(record["query"])
        expected_signature = _registered_pattern_signature(str(record["pattern_str"]))
        if pattern_signature(query) != expected_signature:
            raise ValueError(
                f"Record {record['record_id']} query does not match its pattern_str"
            )
        observation = frozenset(int(value) for value in record["answers"])
        base = executor.execute(query)
        legacy = legacy_denotation(graph_sampler, query)
        for condition_kind in condition_kinds:
            slots = extract_semantic_slots(query, condition_kind)
            if not slots:
                raise ValueError(
                    f"Record {record['record_id']} has no {condition_kind} slots"
                )
            slot_weight = 1.0 / len(slots)
            for slot in slots:
                row = audit_slot(
                    query=query,
                    observation=observation,
                    slot=slot,
                    matcher=matcher,
                    record_id=str(record["record_id"]),
                    seed=args.seed,
                    replacements=args.replacements,
                    epsilon=0.0,
                    tau=0.1,
                    base_denotation=base,
                )
                row.update({
                    "schema_version": SCHEMA_VERSION,
                    "pattern_abbreviation": record["pattern_abbreviation"],
                    "pattern_str": record["pattern_str"],
                    "slot_weight": slot_weight,
                    "legacy_executor_parity": base == legacy,
                })
                audit_rows.append(row)

    audit_path = output_dir / "slot-audit.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(audit_path, audit_rows)
    summary = summarize(
        audit_rows,
        selected_record_count=len(selected),
        shared_reference_execution_count=len(selected),
        legacy_executor_execution_count=len(selected),
    )
    _write_json(summary_path, summary)

    code_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, cwd=REPOSITORY_ROOT
    ).strip())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_reference_audit_manifest",
        "status": "completed",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        "code_sha": code_sha,
        "git_dirty": git_dirty,
        "command": [sys.executable, "-m", "akgr.reproduction.sc_idc_audit", *sys.argv[1:]],
        "config": {
            "path": str(config.source_path),
            "sha256": _sha256(config.source_path),
            "semantic_hash": config.semantic_hash,
            "data_hash": config.data_hash,
            "kg_hash": config.kg_hash,
            "sampling_manifest_sha256": sampling_manifest_hash,
        },
        "input": {
            "fresh_manifest": str(input_manifest_path),
            "fresh_manifest_sha256": _sha256(input_manifest_path),
            "fresh_artifact": str(input_path),
            "fresh_artifact_sha256": _sha256(input_path),
            "fresh_record_count": len(records),
            "manifest_kind": input_manifest.get("kind"),
        },
        "audit": {
            "split": "train",
            "condition_kind": args.condition_kind,
            "seed": int(args.seed),
            "per_pattern": int(args.per_pattern),
            "selected_record_count": len(selected),
            "replacements": int(args.replacements),
            "thresholds": list(THRESHOLDS),
            "tau": 0.1,
            "test_artifact_accessed": False,
        },
        "artifacts": {
            audit_path.name: {"sha256": _sha256(audit_path), "rows": len(audit_rows)},
            summary_path.name: {"sha256": _sha256(summary_path)},
        },
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def run_self_check() -> dict[str, Any]:
    """Exercise the key falsification case without PyTorch/model dependencies."""
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(6))
    graph.add_edge(0, 2, key=0)  # controlled branch C -> observed answer
    graph.add_edge(1, 2, key=1)  # dominant branch H0 -> observed answer
    graph.add_edge(0, 3, key=2)  # matched C' -> false positive
    graph.add_edge(4, 2, key=3)
    graph.add_edge(4, 3, key=3)
    sampler = GraphSampler(
        graph,
        {0: "+controlled", 1: "+dominant", 2: "+replacement", 3: "+broad"},
    )
    matcher = MatchedReplacementIndex(sampler)
    query = parse_action("u -2 2 -1 1")
    relation_slot = extract_semantic_slots(query, "specific_relation")[1]
    row = audit_slot(
        query=query,
        observation={2},
        slot=relation_slot,
        matcher=matcher,
        record_id="or-laundering",
        seed=42,
        replacements=3,
    )
    checks = {
        "legacy_executor_parity": QueryExecutor(sampler).execute(query)
        == legacy_denotation(sampler, query),
        "matched_delta_positive": row["matched_delta"] is not None
        and float(row["matched_delta"]) > 0.0,
        "branch_marginal_zero": abs(float(row["branch_marginal_delta"])) <= 1e-12,
        "classified_laundered": row["classification"] == "laundered",
        "structure_preserved": all(
            candidate["structure_preserved"] for candidate in row["replacements"]
        ),
        "slot_counts": slot_counts(query) == {
            "specific_entity": 2,
            "specific_relation": 2,
        },
    }
    if not all(checks.values()):
        raise AssertionError(f"SC-IDC self-check failed: {checks}")
    return {"status": "pass", "checks": checks, "audit": row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--experiment-config")
    parser.add_argument("--fresh-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument(
        "--condition-kind",
        default="both",
        choices=["both", *CONDITION_KINDS],
    )
    parser.add_argument("--per-pattern", type=int, default=16)
    parser.add_argument("--replacements", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(run_self_check(), indent=2, sort_keys=True, ensure_ascii=False))
        return
    missing = [
        name
        for name in ("experiment_config", "fresh_manifest", "output_dir")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(f"audit mode requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    if args.per_pattern <= 0 or not 1 <= args.replacements <= 8 or args.seed < 0:
        parser.error(
            "per-pattern must be positive, replacements in [1, 8], and seed non-negative"
        )
    print(run_audit(args))


if __name__ == "__main__":
    main()

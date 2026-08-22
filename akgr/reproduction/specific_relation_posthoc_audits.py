"""Post-hoc audits for the specific-relation verifier experiment.

The selector audit is strictly offline: it reuses the four candidates already
stored by P2.  The graph-view audit generates a fresh validation-only sample on
the train graph, freezes all selections there, and evaluates the same selected
hypotheses on cumulative-valid and valid-exclusive graph views.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import networkx as nx
import numpy as np
import torch

from akgr.abduction_model.experiment_runner import _datasets, _graph_samplers, _require_cuda
from akgr.kgdata.kgclass import GraphSampler
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc import (
    QueryExecutor,
    denotation_sha256,
    parse_action,
    value_occurrences,
)
from akgr.reproduction.ss_csc_common import mean, sha256_file
from akgr.reproduction.verifier_best_of_n import (
    _audit_candidate,
    _evaluate_examples,
    _load_frozen_split,
    _observation,
    _verify_checkpoint,
)
from akgr.utils.parsing_util import ans_shift_indices


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "specific-relation-posthoc-audits-v1"
EXPECTED_P2_SUMMARY_SHA256 = "21ec03b56840a447cd93ae9bb027f0c2e10cdc5d41184d2427da144aeed1001e"
EXPECTED_P2_RECORDS_SHA256 = "6d6a159b1d4f5d0d4f76aaa7ef649c5bacd0fa32f00d4472b48d28054d81568d"
EXPECTED_P2_RECORD_COUNT = 1664
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 10_000
GRAPH_AUDIT_SEED = 161803
K = 4
SELECTORS = (
    "first_sample",
    "likelihood_only",
    "semantic_only",
    "exact_semantic",
    "exact_branch",
    "nominal_aware",
    "branch_aware",
)
METRICS = (
    "jaccard",
    "dice",
    "overlap",
    "exact",
    "nominal",
    "branch_supported",
    "nonroot_branch_supported",
    "parse_ok",
    "eos_emitted",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict[str, Any]:
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    temporary = path.with_name("." + path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prepare_output(output_dir: Path, kind: str, code: Mapping[str, Any]) -> Path:
    if code["dirty"]:
        raise ValueError("Formal post-hoc audits require a clean Git worktree")
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _write_json(status_path, {
        "schema_version": SCHEMA_VERSION,
        "kind": kind + "_status",
        "status": "running",
        "started_at": _utc_now(),
        "code": dict(code),
    })
    return status_path


def _verify_p2(path: Path) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    actual_summary = sha256_file(path)
    if actual_summary != EXPECTED_P2_SUMMARY_SHA256:
        raise ValueError("P2 summary SHA256 differs from the preregistered artifact")
    summary = _read_json(path)
    if summary.get("kind") != "verifier_best_of_n_p2" or summary.get("status") != "completed":
        raise ValueError("Not a completed verifier P2 summary")
    artifact = summary.get("artifact", {})
    records_path = (path.parent / str(artifact["path"])).resolve()
    if (
        sha256_file(records_path) != EXPECTED_P2_RECORDS_SHA256
        or str(artifact.get("sha256")) != EXPECTED_P2_RECORDS_SHA256
    ):
        raise ValueError("P2 record SHA256 differs from the preregistered artifact")
    rows = _read_jsonl(records_path)
    if len(rows) != EXPECTED_P2_RECORD_COUNT or int(artifact.get("count", -1)) != len(rows):
        raise ValueError("P2 record count mismatch")
    for row in rows:
        if len(row.get("sample_candidates", [])) != K:
            raise ValueError("P2 row does not contain exactly four candidates")
    return summary, records_path, rows


def _tie_key(candidate: Mapping[str, Any]) -> tuple[float, str]:
    return (
        -float(candidate["model_mean_log_probability"]),
        str(candidate["canonical_hypothesis_sha256"]),
    )


def select_with(selector: str, candidates: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    """Select from a frozen K-candidate set using one ablation rule."""
    if selector not in SELECTORS or not candidates:
        raise ValueError("Unknown selector or empty candidates: " + selector)
    if selector == "first_sample":
        return 0, "first_sample"

    indices = list(range(len(candidates)))
    if selector == "likelihood_only":
        return min(indices, key=lambda index: _tie_key(candidates[index])), "likelihood_only"

    def semantic_key(index: int) -> tuple[float, float, str]:
        candidate = candidates[index]
        return (
            -float(candidate["semantic_average"]),
            -float(candidate["model_mean_log_probability"]),
            str(candidate["canonical_hypothesis_sha256"]),
        )

    if selector == "semantic_only":
        return min(indices, key=semantic_key), "semantic_only"

    if selector == "exact_semantic":
        best = max(int(bool(candidate["exact"])) for candidate in candidates)
        eligible = [index for index in indices if int(bool(candidates[index]["exact"])) == best]
        if best:
            return min(eligible, key=lambda index: _tie_key(candidates[index])), "exact"
        return min(eligible, key=semantic_key), "highest_semantic_average"

    if selector == "exact_branch":
        tiers = [
            2 if candidate["exact"] and candidate["branch_supported"]
            else 1 if candidate["exact"] else 0
            for candidate in candidates
        ]
        reasons = {
            2: "exact+branch_supported",
            1: "exact",
            0: "highest_semantic_average",
        }
    elif selector == "nominal_aware":
        tiers = [
            2 if candidate["exact"] and candidate["nominal"]
            else 1 if candidate["exact"] else 0
            for candidate in candidates
        ]
        reasons = {2: "exact+nominal", 1: "exact", 0: "highest_semantic_average"}
    else:
        tiers = [
            3 if candidate["exact"] and candidate["branch_supported"]
            else 2 if candidate["exact"] and candidate["nominal"]
            else 1 if candidate["exact"] else 0
            for candidate in candidates
        ]
        reasons = {
            3: "exact+branch_supported",
            2: "exact+nominal",
            1: "exact",
            0: "highest_semantic_average",
        }
    best = max(tiers)
    eligible = [index for index, tier in enumerate(tiers) if tier == best]
    if best == 0:
        chosen = min(eligible, key=semantic_key)
    else:
        chosen = min(eligible, key=lambda index: _tie_key(candidates[index]))
    return chosen, reasons[best]


def _attach_selections(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        row["posthoc_selections"] = {}
        for selector in SELECTORS:
            index, reason = select_with(selector, row["sample_candidates"])
            row["posthoc_selections"][selector] = {
                "selected_index": index,
                "selection_reason": reason,
            }


def _selected(row: Mapping[str, Any], selector: str) -> Mapping[str, Any]:
    index = int(row["posthoc_selections"][selector]["selected_index"])
    return row["sample_candidates"][index]


def _metric_value(candidate: Mapping[str, Any], metric: str) -> float:
    return float(candidate[metric])


def _summarize(rows: Sequence[Mapping[str, Any]], selector: str, *, view: str | None = None):
    selected = [_selected(row, selector) for row in rows]
    if view is not None:
        selected = [candidate["graph_view_audits"][view] for candidate in selected]
    result = {"count": len(selected)}
    for metric in METRICS:
        key = metric if metric in {"jaccard", "dice", "overlap"} else metric + "_rate"
        result[key] = mean([_metric_value(candidate, metric) for candidate in selected])
    return result


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    metric: str,
    *,
    view: str | None = None,
) -> dict[str, Any]:
    deltas = []
    for row in rows:
        left_candidate, right_candidate = _selected(row, left), _selected(row, right)
        if view is not None:
            left_candidate = left_candidate["graph_view_audits"][view]
            right_candidate = right_candidate["graph_view_audits"][view]
        deltas.append(_metric_value(left_candidate, metric) - _metric_value(right_candidate, metric))
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        stop = min(start + 250, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "count": len(values),
        "mean_delta": float(values.mean()),
        "ci95_percentile": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "win_rate": float(np.mean(values > 0)),
        "tie_rate": float(np.mean(values == 0)),
        "loss_rate": float(np.mean(values < 0)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _contrasts(rows, pairs, *, view: str | None = None):
    return {
        left + "_minus_" + right: {
            metric: _paired_bootstrap(rows, left, right, metric, view=view)
            for metric in METRICS
        }
        for left, right in pairs
    }


def run_selector(args: argparse.Namespace) -> Path:
    code = _git_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_path = _prepare_output(output_dir, "specific_relation_selector_ablation", code)
    try:
        started = time.perf_counter()
        p2_path = Path(args.p2_summary).expanduser().resolve()
        _, records_path, rows = _verify_p2(p2_path)
        _attach_selections(rows)
        mismatches = []
        for row in rows:
            expected = int(row["selections"]["sample_k4"]["selected_index"])
            actual = int(row["posthoc_selections"]["branch_aware"]["selected_index"])
            if expected != actual:
                mismatches.append(str(row["record_id"]))
        if mismatches:
            raise ValueError("branch-aware replay mismatch: " + ",".join(mismatches[:10]))
        artifact = _write_jsonl(output_dir / "selector-ablation-records.jsonl", rows)
        report = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "kind": "specific_relation_selector_ablation",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "command": [sys.executable, "-m", __name__, "selector", *sys.argv[2:]],
            "method": {
                "offline_only": True,
                "generation_performed": False,
                "selectors": list(SELECTORS),
                "metrics": list(METRICS),
                "k": K,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "primary_contrasts": [
                    "branch_aware_minus_exact_semantic",
                    "branch_aware_minus_nominal_aware",
                ],
            },
            "inputs": {
                "p2_summary": str(p2_path),
                "p2_summary_sha256": sha256_file(p2_path),
                "p2_records": str(records_path),
                "p2_records_sha256": sha256_file(records_path),
            },
            "replay": {
                "branch_aware_matches_historical_sample_k4": True,
                "mismatch_count": 0,
            },
            "summaries": {selector: _summarize(rows, selector) for selector in SELECTORS},
            "contrasts": _contrasts(rows, (
                ("branch_aware", "exact_semantic"),
                ("branch_aware", "nominal_aware"),
                ("branch_aware", "semantic_only"),
                ("branch_aware", "likelihood_only"),
                ("branch_aware", "first_sample"),
            )),
            "artifact": artifact,
            "total_wall_seconds": time.perf_counter() - started,
            "post_evaluation_tuning_permitted": False,
        }
        report_path = output_dir / "summary.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_selector_ablation_status",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "summary_sha256": sha256_file(report_path),
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_selector_ablation_status",
            "status": "failed",
            "failed_at": _utc_now(),
            "code": code,
            "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


AVAILABILITY_QUALITIES = (
    "parse_ok",
    "exact",
    "nominal",
    "branch_supported",
    "nonroot_branch_supported",
    "exact_nominal",
    "exact_branch_supported",
    "exact_nonroot_branch_supported",
)


def _qualifies(candidate: Mapping[str, Any], quality: str) -> bool:
    if quality in {
        "parse_ok",
        "exact",
        "nominal",
        "branch_supported",
        "nonroot_branch_supported",
    }:
        return bool(candidate[quality])
    if quality == "exact_nominal":
        return bool(candidate["exact"] and candidate["nominal"])
    if quality == "exact_branch_supported":
        return bool(candidate["exact"] and candidate["branch_supported"])
    if quality == "exact_nonroot_branch_supported":
        return bool(candidate["exact"] and candidate["nonroot_branch_supported"])
    raise ValueError("Unknown availability quality: " + quality)


def _prefix_selected(row: Mapping[str, Any], selector: str, k: int):
    candidates = row["sample_candidates"][: int(k)]
    index, reason = select_with(selector, candidates)
    return index, reason, candidates[index]


def _bootstrap_binary_delta(left: Sequence[bool], right: Sequence[bool]):
    values = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        stop = min(start + 250, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci95_percentile": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "gain_rate": float(np.mean(values > 0)),
        "unchanged_rate": float(np.mean(values == 0)),
        "loss_rate": float(np.mean(values < 0)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _availability_dataset(rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]):
    result: dict[str, Any] = {
        "count": len(rows),
        "k_values": list(k_values),
        "availability": {},
        "selectors": {},
        "full_verifier_rescue_harm": {},
        "selector_disagreement": {},
    }
    availability_vectors: dict[int, dict[str, list[bool]]] = {}
    for k in k_values:
        availability_vectors[k] = {
            quality: [
                any(_qualifies(candidate, quality) for candidate in row["sample_candidates"][:k])
                for row in rows
            ]
            for quality in AVAILABILITY_QUALITIES
        }
        result["availability"][f"k{k}"] = {}
        for quality in AVAILABILITY_QUALITIES:
            available = availability_vectors[k][quality]
            counts = [
                sum(
                    _qualifies(candidate, quality)
                    for candidate in row["sample_candidates"][:k]
                )
                for row in rows
            ]
            result["availability"][f"k{k}"][quality] = {
                "available_count": int(sum(available)),
                "availability_rate": mean([float(value) for value in available]),
                "proposal_failure_count": int(len(rows) - sum(available)),
                "proposal_failure_rate": mean([float(not value) for value in available]),
                "mean_qualifying_candidate_count": mean([float(value) for value in counts]),
            }
    result["availability_adjacent_k_deltas"] = {}
    for left_k, right_k in zip(k_values, k_values[1:]):
        result["availability_adjacent_k_deltas"][f"k{right_k}_minus_k{left_k}"] = {
            quality: _bootstrap_binary_delta(
                availability_vectors[left_k][quality],
                availability_vectors[right_k][quality],
            )
            for quality in AVAILABILITY_QUALITIES
        }

    selected_cache: dict[tuple[int, str], list[tuple[int, Mapping[str, Any]]]] = {}
    for k in k_values:
        result["selectors"][f"k{k}"] = {}
        for selector in SELECTORS:
            selected = []
            reasons: dict[str, int] = {}
            for row in rows:
                index, reason, candidate = _prefix_selected(row, selector, k)
                selected.append((index, candidate))
                reasons[reason] = reasons.get(reason, 0) + 1
            selected_cache[(k, selector)] = selected
            metric_summary = {}
            for metric in METRICS:
                key = metric if metric in {"jaccard", "dice", "overlap"} else metric + "_rate"
                metric_summary[key] = mean([
                    float(candidate[metric]) for _, candidate in selected
                ])
            captures = {}
            for quality in AVAILABILITY_QUALITIES:
                available = availability_vectors[k][quality]
                selected_qualifying = [
                    _qualifies(candidate, quality) for _, candidate in selected
                ]
                available_count = int(sum(available))
                captured_count = sum(
                    bool(is_available and is_selected)
                    for is_available, is_selected in zip(available, selected_qualifying)
                )
                captures[quality] = {
                    "selected_qualifying_count": int(sum(selected_qualifying)),
                    "selected_qualifying_rate": mean([
                        float(value) for value in selected_qualifying
                    ]),
                    "captured_available_count": int(captured_count),
                    "conditional_capture_rate": (
                        float(captured_count) / available_count
                        if available_count else None
                    ),
                    "missed_available_count": int(available_count - captured_count),
                }
            result["selectors"][f"k{k}"][selector] = {
                "metrics": metric_summary,
                "captures": captures,
                "selection_reasons": reasons,
            }

        full = selected_cache[(k, "branch_aware")]
        for baseline in ("likelihood_only", "exact_semantic"):
            baseline_selected = selected_cache[(k, baseline)]
            comparison = {}
            for quality in AVAILABILITY_QUALITIES:
                available = availability_vectors[k][quality]
                full_values = [
                    _qualifies(candidate, quality) for _, candidate in full
                ]
                baseline_values = [
                    _qualifies(candidate, quality) for _, candidate in baseline_selected
                ]
                rescue = [
                    a and f and not b
                    for a, f, b in zip(available, full_values, baseline_values)
                ]
                harm = [
                    a and b and not f
                    for a, f, b in zip(available, full_values, baseline_values)
                ]
                available_count = int(sum(available))
                comparison[quality] = {
                    "rescue_count": int(sum(rescue)),
                    "rescue_rate": mean([float(value) for value in rescue]),
                    "rescue_rate_given_available": (
                        float(sum(rescue)) / available_count if available_count else None
                    ),
                    "harm_count": int(sum(harm)),
                    "harm_rate": mean([float(value) for value in harm]),
                    "harm_rate_given_available": (
                        float(sum(harm)) / available_count if available_count else None
                    ),
                }
            result["full_verifier_rescue_harm"].setdefault(f"k{k}", {})[
                "branch_aware_vs_" + baseline
            ] = comparison

        for left, right in (
            ("branch_aware", "exact_branch"),
            ("branch_aware", "nominal_aware"),
            ("branch_aware", "exact_semantic"),
            ("branch_aware", "likelihood_only"),
        ):
            left_indices = [index for index, _ in selected_cache[(k, left)]]
            right_indices = [index for index, _ in selected_cache[(k, right)]]
            disagreements = [
                left_index != right_index
                for left_index, right_index in zip(left_indices, right_indices)
            ]
            result["selector_disagreement"].setdefault(f"k{k}", {})[
                left + "_vs_" + right
            ] = {
                "count": int(sum(disagreements)),
                "rate": mean([float(value) for value in disagreements]),
            }
    return result


def _compact_availability_records(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
):
    for row in rows:
        prefixes = {}
        for k in k_values:
            prefixes[f"k{k}"] = {
                "availability": {
                    quality: any(
                        _qualifies(candidate, quality)
                        for candidate in row["sample_candidates"][:k]
                    )
                    for quality in AVAILABILITY_QUALITIES
                },
                "qualifying_candidate_counts": {
                    quality: sum(
                        _qualifies(candidate, quality)
                        for candidate in row["sample_candidates"][:k]
                    )
                    for quality in AVAILABILITY_QUALITIES
                },
                "selections": {
                    selector: {
                        "selected_index": _prefix_selected(row, selector, k)[0],
                        "selection_reason": _prefix_selected(row, selector, k)[1],
                    }
                    for selector in SELECTORS
                },
            }
        yield {
            "dataset": dataset,
            "record_id": str(row["record_id"]),
            "prefixes": prefixes,
        }


def _p1_original_from_p2(p2: Mapping[str, Any]):
    p1_path = Path(str(p2["p1"]["path"])).expanduser().resolve()
    if sha256_file(p1_path) != str(p2["p1"]["sha256"]):
        raise ValueError("P1 summary hash differs from P2 contract")
    p1 = _read_json(p1_path)
    if p1.get("kind") != "verifier_best_of_n_p1" or p1.get("status") != "passed":
        raise ValueError("P2 does not reference a passed P1")
    artifact = p1["artifacts"]["original_validation_records"]
    records_path = (p1_path.parent / str(artifact["path"])).resolve()
    if sha256_file(records_path) != str(artifact["sha256"]):
        raise ValueError("P1 original-validation record hash mismatch")
    rows = _read_jsonl(records_path)
    if len(rows) != int(artifact["count"]):
        raise ValueError("P1 original-validation record count mismatch")
    if any(len(row.get("sample_candidates", [])) != 8 for row in rows):
        raise ValueError("P1 original-validation rows must contain K=8 candidates")
    return p1_path, records_path, rows


def run_availability(args: argparse.Namespace) -> Path:
    code = _git_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_path = _prepare_output(output_dir, "specific_relation_candidate_availability", code)
    try:
        started = time.perf_counter()
        p2_path = Path(args.p2_summary).expanduser().resolve()
        p2, p2_records_path, p2_rows = _verify_p2(p2_path)
        p1_path, p1_records_path, p1_rows = _p1_original_from_p2(p2)
        p1_k, p2_k = (1, 2, 4, 8), (1, 2, 4)
        compact_rows = [
            *_compact_availability_records("p1_original_validation", p1_rows, p1_k),
            *_compact_availability_records("p2_final_posthoc", p2_rows, p2_k),
        ]
        artifact = _write_jsonl(
            output_dir / "candidate-availability-records.jsonl",
            compact_rows,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "kind": "specific_relation_candidate_availability",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "command": [sys.executable, "-m", __name__, "availability", *sys.argv[2:]],
            "method": {
                "offline_only": True,
                "generation_performed": False,
                "post_hoc_diagnostic_only": True,
                "p1_is_primary_k_curve": True,
                "p2_is_post_hoc_confirmation": True,
                "k_selection_or_tuning_permitted": False,
                "qualities": list(AVAILABILITY_QUALITIES),
                "selectors": list(SELECTORS),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
            "inputs": {
                "p1_summary": str(p1_path),
                "p1_summary_sha256": sha256_file(p1_path),
                "p1_records": str(p1_records_path),
                "p1_records_sha256": sha256_file(p1_records_path),
                "p2_summary": str(p2_path),
                "p2_summary_sha256": sha256_file(p2_path),
                "p2_records": str(p2_records_path),
                "p2_records_sha256": sha256_file(p2_records_path),
            },
            "datasets": {
                "p1_original_validation": _availability_dataset(p1_rows, p1_k),
                "p2_final_posthoc": _availability_dataset(p2_rows, p2_k),
            },
            "artifact": artifact,
            "total_wall_seconds": time.perf_counter() - started,
            "post_evaluation_tuning_permitted": False,
        }
        report_path = output_dir / "summary.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_candidate_availability_status",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "summary_sha256": sha256_file(report_path),
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_candidate_availability_status",
            "status": "failed",
            "failed_at": _utc_now(),
            "code": code,
            "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def _edges(graph_sampler) -> set[tuple[int, int, int]]:
    return {
        (int(source), int(target), int(relation))
        for source, target, relation in graph_sampler.graph.edges(keys=True)
    }


def _edge_sha256(edges: Iterable[tuple[int, int, int]]) -> str:
    payload = json.dumps(sorted(edges), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_valid_exclusive_sampler(train_sampler, valid_sampler) -> tuple[GraphSampler, dict[str, Any]]:
    """Construct valid-exclusive edges while preserving the valid entity universe."""
    train_edges, valid_edges = _edges(train_sampler), _edges(valid_sampler)
    if not train_edges < valid_edges:
        raise ValueError("Expected train graph to be a strict subset of cumulative valid")
    exclusive_edges = valid_edges - train_edges
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(valid_sampler.graph.nodes(data=True))
    for source, target, relation in sorted(exclusive_edges):
        graph.add_edge(source, target, key=relation)
    sampler = GraphSampler(graph, valid_sampler.id2rel)
    reconstructed = train_edges | _edges(sampler)
    if train_edges & _edges(sampler):
        raise ValueError("Train and valid-exclusive edges overlap")
    if reconstructed != valid_edges:
        raise ValueError("Train plus valid-exclusive does not reconstruct cumulative valid")
    return sampler, {
        "train": {
            "edge_count": len(train_edges),
            "edge_sha256": _edge_sha256(train_edges),
            "node_count": train_sampler.graph.number_of_nodes(),
        },
        "valid_cumulative": {
            "edge_count": len(valid_edges),
            "edge_sha256": _edge_sha256(valid_edges),
            "node_count": valid_sampler.graph.number_of_nodes(),
        },
        "valid_exclusive": {
            "edge_count": len(exclusive_edges),
            "edge_sha256": _edge_sha256(exclusive_edges),
            "node_count": sampler.graph.number_of_nodes(),
        },
        "checks": {
            "train_exclusive_edge_disjoint": True,
            "train_union_exclusive_equals_valid": True,
            "valid_node_universe_preserved": (
                set(sampler.graph.nodes) == set(valid_sampler.graph.nodes)
            ),
        },
    }


def _source_from_observation(observation: Iterable[int]) -> str:
    return " ".join(str(value) for value in ans_shift_indices(sorted(observation)))


def _reference_examples(config, preflight_path: Path, samplers):
    frozen, contract = _load_frozen_split(
        config=config,
        preflight_path=preflight_path,
        purpose="validation",
        split="valid",
    )
    datasets, _, _ = _datasets(config, ["valid"], train_variant="base")
    dataset_rows = {str(item["record_id"]): item for item in datasets["valid"]}
    train_executor = QueryExecutor(samplers["train"])
    valid_executor = QueryExecutor(samplers["valid"])
    exclusive_executor = QueryExecutor(samplers["valid_exclusive"])
    max_answers = int(config.raw["data"]["max_answers"])
    examples, audit = [], {
        "frozen_count": len(frozen),
        "eligible_count": 0,
        "filtered_train_empty": 0,
        "filtered_train_too_large": 0,
        "valid_exclusive_reference_nonempty_count": 0,
    }
    for example in frozen:
        record_id = str(example["record_id"])
        item = dataset_rows[record_id]
        reference = str(item["target"])
        query = parse_action(reference)
        condition = str(example["condition"])
        if not value_occurrences(query, "specific_relation", int(condition)):
            raise ValueError("Frozen reference omits its condition at " + record_id)
        train_observation = frozenset(train_executor.execute(query))
        valid_observation = frozenset(valid_executor.execute(query))
        exclusive_observation = frozenset(exclusive_executor.execute(query))
        if valid_observation != _observation(str(item["source"])):
            raise ValueError("Frozen validation source does not match cumulative-valid execution")
        if not train_observation:
            audit["filtered_train_empty"] += 1
            continue
        if len(train_observation) > max_answers:
            audit["filtered_train_too_large"] += 1
            continue
        if exclusive_observation:
            audit["valid_exclusive_reference_nonempty_count"] += 1
        examples.append({
            "record_id": record_id,
            "source": _source_from_observation(train_observation),
            "condition": condition,
            "group_id": record_id,
            "side": None,
            "alternative_condition": None,
            "reference_hypothesis": reference,
            "reference_query_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
            "reference_observations": {
                "train": sorted(train_observation),
                "valid_cumulative": sorted(valid_observation),
                "valid_exclusive": sorted(exclusive_observation),
            },
            "reference_denotation_sha256": {
                "train": denotation_sha256(train_observation),
                "valid_cumulative": denotation_sha256(valid_observation),
                "valid_exclusive": denotation_sha256(exclusive_observation),
            },
        })
    audit["eligible_count"] = len(examples)
    audit["eligible_rate"] = len(examples) / len(frozen) if frozen else 0.0
    return examples, contract, audit


def _attach_graph_view_audits(rows, samplers) -> None:
    executors = {
        "valid_cumulative": QueryExecutor(samplers["valid"]),
        "valid_exclusive": QueryExecutor(samplers["valid_exclusive"]),
    }
    for row in rows:
        reference = row["reference_observations"]
        candidates = [row["greedy"], *row["sample_candidates"]]
        for candidate in candidates:
            candidate["graph_view_audits"] = {}
            for view, executor in executors.items():
                audit = _audit_candidate(
                    str(candidate["prediction"]),
                    condition=str(row["condition"]),
                    observation=frozenset(int(value) for value in reference[view]),
                    executor=executor,
                )
                audit["eos_emitted"] = bool(candidate["eos_emitted"])
                candidate["graph_view_audits"][view] = audit


def _view_report(rows, view: str):
    pairs = (
        ("branch_aware", "exact_semantic"),
        ("branch_aware", "nominal_aware"),
        ("branch_aware", "first_sample"),
    )
    return {
        "summaries": {selector: _summarize(rows, selector, view=view) for selector in SELECTORS},
        "contrasts": _contrasts(rows, pairs, view=view),
    }


def run_graph_audit(args: argparse.Namespace) -> Path:
    code = _git_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_path = _prepare_output(output_dir, "specific_relation_graph_view_audit", code)
    try:
        started = time.perf_counter()
        p2_path = Path(args.p2_summary).expanduser().resolve()
        p2, _, _ = _verify_p2(p2_path)
        p1_path = Path(str(p2["p1"]["path"])).expanduser().resolve()
        if sha256_file(p1_path) != str(p2["p1"]["sha256"]):
            raise ValueError("P1 summary hash differs from the P2 contract")
        p1 = _read_json(p1_path)
        config_path = Path(str(p1["inputs"]["experiment_config"])).expanduser().resolve()
        if sha256_file(config_path) != str(p1["inputs"]["experiment_config_sha256"]):
            raise ValueError("Experiment config hash differs from P1")
        config = load_experiment_config(config_path)
        preflight_path = Path(
            str(p1["inputs"]["original_validation"]["preflight_path"])
        ).expanduser().resolve()
        if sha256_file(preflight_path) != str(
            p1["inputs"]["original_validation"]["preflight_sha256"]
        ):
            raise ValueError("Preflight hash differs from P1")

        samplers = _graph_samplers(config)
        exclusive, graph_contract = build_valid_exclusive_sampler(
            samplers["train"], samplers["valid"]
        )
        samplers["valid_exclusive"] = exclusive
        examples, validation_contract, eligibility = _reference_examples(
            config, preflight_path, samplers
        )
        if args.limit is not None:
            examples = examples[: int(args.limit)]
        if not examples:
            raise ValueError("No eligible validation examples")

        checkpoint = p1["inputs"]["checkpoint"]
        loaded, verified_checkpoint = _verify_checkpoint(
            checkpoint=Path(str(checkpoint["path"])).expanduser().resolve(),
            selection_path=Path(str(checkpoint["selection_path"])).expanduser().resolve(),
            expected_tree=str(checkpoint["tree_sha256"]),
            expected_pair_summary_sha256=str(
                p1["inputs"]["pair_validation"]["pair_summary_sha256"]
            ),
            config=config,
        )
        device = _require_cuda("specific-relation graph-view audit")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        rows, costs = _evaluate_examples(
            name="specific_relation_graph_view_audit",
            examples=examples,
            model=loaded.model.to(device),
            tokenizer=loaded.tokenizer,
            device=device,
            graph_sampler=samplers["train"],
            seed=GRAPH_AUDIT_SEED,
            batch_size=int(args.batch_size),
            max_new_tokens=int(p1["method"]["max_new_tokens"]),
            max_k=K,
            include_greedy=True,
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]),
        )
        _attach_selections(rows)
        _attach_graph_view_audits(rows, samplers)
        exclusive_nonempty = [
            row for row in rows if bool(row["reference_observations"]["valid_exclusive"])
        ]
        artifact = _write_jsonl(output_dir / "graph-view-audit-records.jsonl", rows)
        report = {
            "schema_version": SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "kind": "specific_relation_graph_view_audit",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "command": [sys.executable, "-m", __name__, "graph-audit", *sys.argv[2:]],
            "method": {
                "post_hoc_diagnostic_only": True,
                "historical_test_reopened": False,
                "selection_view": "train",
                "selection_frozen_before_audit": True,
                "audit_views": ["valid_cumulative", "valid_exclusive"],
                "graph_audit_seed": GRAPH_AUDIT_SEED,
                "k": K,
                "selectors": list(SELECTORS),
                "metrics": list(METRICS),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "limit": args.limit,
            },
            "inputs": {
                "p2_summary": str(p2_path),
                "p2_summary_sha256": sha256_file(p2_path),
                "p1_summary": str(p1_path),
                "p1_summary_sha256": sha256_file(p1_path),
                "checkpoint": verified_checkpoint,
                "validation": validation_contract,
            },
            "graph_contract": graph_contract,
            "eligibility": {
                **eligibility,
                "executed_count": len(rows),
                "executed_exclusive_reference_nonempty_count": len(exclusive_nonempty),
            },
            "selection_train": {
                "summaries": {
                    selector: _summarize(rows, selector) for selector in SELECTORS
                },
                "contrasts": _contrasts(rows, (
                    ("branch_aware", "exact_semantic"),
                    ("branch_aware", "nominal_aware"),
                    ("branch_aware", "first_sample"),
                )),
            },
            "audit_valid_cumulative": _view_report(rows, "valid_cumulative"),
            "audit_valid_exclusive_all": _view_report(rows, "valid_exclusive"),
            "audit_valid_exclusive_reference_nonempty": (
                _view_report(exclusive_nonempty, "valid_exclusive")
                if exclusive_nonempty else None
            ),
            "generation_cost": costs,
            "artifact": artifact,
            "total_wall_seconds": time.perf_counter() - started,
            "post_evaluation_tuning_permitted": False,
            "limitations": [
                "Validation-only post-hoc audit; it does not revise the historical P2 estimate.",
                "The valid-exclusive view is sparse and is reported both overall and with nonempty references.",
                "Graph-view stability is a diagnostic, not proof of logical equivalence.",
            ],
        }
        report_path = output_dir / "summary.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_graph_view_audit_status",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "summary_sha256": sha256_file(report_path),
            "historical_test_reopened": False,
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "specific_relation_graph_view_audit_status",
            "status": "failed",
            "failed_at": _utc_now(),
            "code": code,
            "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(),
            "historical_test_reopened": False,
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    selector = subparsers.add_parser("selector")
    selector.add_argument("--p2-summary", required=True)
    selector.add_argument("--output-dir", required=True)
    availability = subparsers.add_parser("availability")
    availability.add_argument("--p2-summary", required=True)
    availability.add_argument("--output-dir", required=True)
    graph = subparsers.add_parser("graph-audit")
    graph.add_argument("--p2-summary", required=True)
    graph.add_argument("--output-dir", required=True)
    graph.add_argument("--batch-size", type=int, default=32)
    graph.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("batch size must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.stage == "selector":
        result = run_selector(args)
    elif args.stage == "availability":
        result = run_availability(args)
    else:
        result = run_graph_audit(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc diagnostics for an SC-IDC Experiment 1 completion audit.

The command is deliberately read-only: it derives group-level optimization
statistics and reward-construction diagnostics from the frozen JSONL artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ALPHA_IDC = 0.05
TAU_MARG = 0.1
TAU_MATCH = 0.1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _rate(values: Iterable[bool]) -> float | None:
    materialized = [bool(value) for value in values]
    return _mean([float(value) for value in materialized])


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return math.fsum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _normalized_advantages(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    deviation = statistics.stdev(values)
    if deviation == 0.0:
        return [0.0 for _ in values]
    center = statistics.mean(values)
    return [(value - center) / (deviation + 1e-4) for value in values]


def _ordering(values: Sequence[float]) -> tuple[int, ...]:
    result = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            delta = values[left] - values[right]
            result.append(0 if delta == 0.0 else (1 if delta > 0.0 else -1))
    return tuple(result)


def _gate(row: Mapping[str, Any]) -> float:
    audit = row.get("sc_idc")
    if not isinstance(audit, dict):
        return 0.0
    branch = max(0.0, min(1.0, float(audit["branch_marginal_delta"]) / TAU_MARG))
    matched_value = audit.get("matched_delta")
    matched = (
        0.0
        if matched_value is None
        else max(0.0, min(1.0, float(matched_value) / TAU_MATCH))
    )
    return branch * matched


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Completion audit is empty")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_index"])].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row["generation_index"]))

    nominal_rows = [row for row in rows if isinstance(row.get("sc_idc"), dict)]
    positive_rows = [row for row in nominal_rows if float(row["raw_bss"]) > 0.0]
    base = [float(row["base_reward"]) for row in nominal_rows]
    raw = [float(row["raw_bss"]) for row in nominal_rows]
    gates = [_gate(row) for row in nominal_rows]

    advantage_changed = []
    ordering_changed = []
    advantage_l1 = []
    gate_advantage_changed = []
    candidate_set_variation = []
    for group_rows in groups.values():
        base_values = [float(row["base_reward"]) for row in group_rows]
        raw_values = [float(row["raw_bss"]) for row in group_rows]
        gate_values = [_gate(row) for row in group_rows]
        augmented = [a + ALPHA_IDC * b for a, b in zip(base_values, raw_values)]
        gate_augmented = [a + ALPHA_IDC * b for a, b in zip(base_values, gate_values)]
        base_adv = _normalized_advantages(base_values)
        augmented_adv = _normalized_advantages(augmented)
        gate_adv = _normalized_advantages(gate_augmented)
        deltas = [abs(a - b) for a, b in zip(base_adv, augmented_adv)]
        advantage_changed.append(max(deltas, default=0.0) > 1e-8)
        advantage_l1.append(_mean(deltas) or 0.0)
        ordering_changed.append(_ordering(base_values) != _ordering(augmented))
        gate_advantage_changed.append(
            max((abs(a - b) for a, b in zip(base_adv, gate_adv)), default=0.0) > 1e-8
        )
        candidate_sets = {
            tuple(sorted(int(item["token"]) for item in row["sc_idc"]["replacements"]))
            for row in group_rows
            if isinstance(row.get("sc_idc"), dict)
        }
        if len(candidate_sets) >= 1:
            candidate_set_variation.append(len(candidate_sets) > 1)

    replacements = [
        replacement
        for row in nominal_rows
        for replacement in row["sc_idc"].get("replacements", [])
    ]
    root_rows = [
        row for row in nominal_rows
        if any(
            str(branch.get("branch_path")) == "root"
            for branch in row["sc_idc"]["neutralization"].get("branches", [])
        )
    ]
    all_empty_rows = [
        row for row in nominal_rows
        if row["sc_idc"].get("replacements")
        and all(
            int(replacement["denotation_cardinality"]) == 0
            for replacement in row["sc_idc"]["replacements"]
        )
    ]
    occurrence_buckets: dict[str, list[float]] = defaultdict(list)
    for row in nominal_rows:
        occurrence_buckets[str(int(row["sc_idc"]["occurrence_count"]))].append(
            float(row["raw_bss"])
        )

    return {
        "counts": {
            "groups": len(groups),
            "completions": len(rows),
            "nominal_scorable": len(nominal_rows),
            "positive_bss": len(positive_rows),
        },
        "optimization_signal": {
            "alpha_idc": ALPHA_IDC,
            "groups_with_pairwise_ordering_change_rate": _rate(ordering_changed),
            "groups_with_normalized_advantage_change_rate": _rate(advantage_changed),
            "mean_group_normalized_advantage_l1_change": _mean(advantage_l1),
            "groups_with_advantage_change_without_base_jaccard_multiplier_rate": _rate(
                gate_advantage_changed
            ),
            "base_reward_raw_bss_pearson_on_nominal": _pearson(base, raw),
            "base_reward_gate_pearson_on_nominal": _pearson(base, gates),
        },
        "saturation": {
            "branch_clip_saturated_rate_on_nominal": _rate(
                float(row["sc_idc"]["branch_marginal_delta"]) >= TAU_MARG
                for row in nominal_rows
            ),
            "matched_clip_saturated_rate_on_nominal": _rate(
                row["sc_idc"].get("matched_delta") is not None
                and float(row["sc_idc"]["matched_delta"]) >= TAU_MATCH
                for row in nominal_rows
            ),
            "both_clips_saturated_rate_on_nominal": _rate(
                float(row["sc_idc"]["branch_marginal_delta"]) >= TAU_MARG
                and row["sc_idc"].get("matched_delta") is not None
                and float(row["sc_idc"]["matched_delta"]) >= TAU_MATCH
                for row in nominal_rows
            ),
            "mean_raw_bss_when_positive": _mean(
                [float(row["raw_bss"]) for row in positive_rows]
            ),
        },
        "intervention_bias": {
            "root_neutralization_rate_on_nominal": len(root_rows) / len(nominal_rows),
            "replacement_empty_denotation_rate": _rate(
                int(replacement["denotation_cardinality"]) == 0
                for replacement in replacements
            ),
            "all_replacements_empty_rate_on_nominal": len(all_empty_rows) / len(nominal_rows),
            "repeated_condition_value_rate_on_nominal": _rate(
                int(row["sc_idc"]["occurrence_count"]) > 1 for row in nominal_rows
            ),
            "groups_with_multiple_candidate_sets_rate": _rate(candidate_set_variation),
            "raw_bss_by_occurrence_count": {
                count: {"count": len(values), "mean": _mean(values)}
                for count, values in sorted(occurrence_buckets.items(), key=lambda item: int(item[0]))
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.completion_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = _read_jsonl(source)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_offline_diagnostics",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": _sha256(source), "count": len(rows)},
        "summary": summarize(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

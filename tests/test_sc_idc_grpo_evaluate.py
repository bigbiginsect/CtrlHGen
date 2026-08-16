from __future__ import annotations

import pytest

pytest.importorskip("torch")

from akgr.reproduction.sc_idc_grpo_evaluate import (
    _audit_metrics,
    compare_paired_records,
)


def _record(record_id, semantic, nominal, classification, *, length=3):
    return {
        "record_id": record_id,
        "jaccard": semantic,
        "dice": semantic,
        "overlap": semantic,
        "condition_accuracy": float(nominal),
        "smatch": semantic,
        "parse_ok": True,
        "eos_emitted": True,
        "hit_max_new_tokens": False,
        "generated_token_count": length,
        "nominal_adherence": nominal,
        "sc_idc_classification": classification,
        "matched_delta": 0.2 if classification == "branch_supported_selective" else -0.1,
        "raw_bss": 0.4 if classification == "branch_supported_selective" else 0.0,
    }


def test_audit_metrics_keep_overall_and_nominal_denominators_explicit():
    records = [
        _record("a", 0.1, True, "branch_supported_selective"),
        _record("b", 0.1, True, "branch_nonmarginal"),
        _record("c", 0.1, False, None),
    ]
    metrics = _audit_metrics(records)
    assert metrics["nominal_adherence"] == pytest.approx(2 / 3)
    assert metrics["branch_supported_selectivity_rate"] == pytest.approx(1 / 3)
    assert metrics["conditional_on_nominal"]["branch_supported_selectivity_rate"] == 0.5


def test_paired_comparison_rejects_record_identity_drift():
    with pytest.raises(ValueError, match="record IDs"):
        compare_paired_records(
            [_record("a", 0.1, True, "branch_supported_selective")],
            [_record("b", 0.2, True, "branch_supported_selective")],
            seed=42,
        )


def test_paired_comparison_reports_semantic_and_control_deltas():
    baseline = [_record("a", 0.1, True, "branch_nonmarginal")]
    candidate = [_record("a", 0.3, True, "branch_supported_selective", length=4)]
    result = compare_paired_records(baseline, candidate, seed=42)
    assert result["delta"]["standard"]["semantic_average"] == pytest.approx(0.2)
    assert result["delta"]["standard"]["mean_generated_tokens"] == 1.0
    assert result["delta"]["sc_idc"]["branch_supported_selectivity_rate"] == 1.0

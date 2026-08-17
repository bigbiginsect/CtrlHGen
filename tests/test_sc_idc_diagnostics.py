from types import SimpleNamespace

from akgr.reproduction.sc_idc_condition_causality import _counterfactual_condition
from akgr.reproduction.sc_idc_offline_diagnostics import summarize


def _audit(raw_bss, *, replacement_token, replacement_empty=False):
    return {
        "base_reward": 1.0,
        "raw_bss": raw_bss,
        "sc_idc": {
            "branch_marginal_delta": raw_bss,
            "matched_delta": raw_bss,
            "occurrence_count": 1,
            "neutralization": {"branches": [{"branch_path": "0"}]},
            "replacements": [{
                "token": replacement_token,
                "denotation_cardinality": 0 if replacement_empty else 1,
            }],
        },
    }


def test_offline_summary_exposes_new_group_signal_and_candidate_noise():
    rows = []
    for generation_index, raw_bss in enumerate((0.0, 1.0, 0.0, 1.0)):
        rows.append({
            **_audit(
                raw_bss,
                replacement_token=-(generation_index + 1),
                replacement_empty=generation_index == 0,
            ),
            "group_index": 0,
            "generation_index": generation_index,
        })
    result = summarize(rows)
    signal = result["optimization_signal"]
    assert signal["base_zero_variance_group_rate"] == 1.0
    assert signal["augmented_zero_variance_group_rate"] == 0.0
    assert signal["groups_with_pairwise_ordering_change_rate"] == 1.0
    assert result["intervention_bias"]["groups_with_multiple_candidate_sets_rate"] == 1.0


def test_counterfactual_condition_skips_relations_in_reference():
    matcher = SimpleNamespace(ranked=lambda value: [
        {"token": -2, "rank": 0},
        {"token": -3, "rank": 1},
    ])
    value, metadata = _counterfactual_condition(
        target="i -1 10 -2 11", assigned="-1", matcher=matcher
    )
    assert value == "-3"
    assert metadata["rank"] == 1

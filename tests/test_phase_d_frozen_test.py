from akgr.reproduction.phase_d_frozen_test import _frozen_decision


def _comparison(greedy=0.01, sampled=0.01, parse=1.0, eos=1.0, condition=0.95):
    return {
        decode: {
            "candidate": {
                "parse_ok": parse,
                "eos_rate": eos,
                "condition_accuracy": condition,
            },
            "delta": {"semantic_average": delta},
        }
        for decode, delta in (("greedy", greedy), ("sampled", sampled))
    }


def test_frozen_decision_passes_both_positive_healthy_decodes():
    result = _frozen_decision(_comparison())
    assert result["classification"] == "pass"
    assert result["post_test_training_or_checkpoint_selection_allowed"] is False


def test_frozen_decision_fails_nonpositive_decode():
    assert _frozen_decision(_comparison(sampled=0.0))["classification"] == "fail"


def test_frozen_decision_fails_health_gate():
    assert _frozen_decision(_comparison(parse=0.97))["classification"] == "fail"

from akgr.reproduction.phase_d_probe import _recommendation


def _comparison(greedy, sampled, health=True):
    return {
        "greedy": {"delta": {"semantic_average": greedy}},
        "sampled": {"delta": {"semantic_average": sampled}},
        "decision": {"health_pass": health},
    }


def test_probe_stops_on_health_regression():
    result = _recommendation(_comparison(0.1, 0.1, health=False), 1000)
    assert result["action"] == "stop_health_regression"


def test_probe_waits_until_minimum_step():
    result = _recommendation(_comparison(0.0, 0.0), 3999)
    assert result["action"] == "continue"


def test_probe_stops_negligible_gain_after_4000():
    result = _recommendation(_comparison(0.002, 0.001), 4000)
    assert result["action"] == "stop_negligible_gain_after_4000"


def test_probe_stops_below_pilot_scale_after_8000():
    result = _recommendation(_comparison(0.004, 0.0049), 8000)
    assert result["action"] == "stop_below_pilot_scale_after_8000"


def test_probe_continues_one_strong_decode():
    result = _recommendation(_comparison(0.001, 0.01), 9000)
    assert result["action"] == "continue"

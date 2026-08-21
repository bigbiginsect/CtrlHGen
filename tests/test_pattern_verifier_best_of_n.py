import pytest

from akgr.evaluation import condition_accuracy
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.pattern_verifier_best_of_n import (
    audit_candidate,
    canonical_pattern,
    p1_gate,
    select_candidate,
)
from akgr.reproduction.sc_idc import parse_action


def candidate(*, exact=False, pattern=False, semantic=0.0, logp=-1.0, sha="a"):
    return {"exact": exact, "pattern_match": pattern, "semantic_average": semantic,
            "model_mean_log_probability": logp, "canonical_hypothesis_sha256": sha}


def test_canonical_pattern_matches_existing_condition_accuracy():
    prediction = "i -1 2 -3 4"
    condition = canonical_pattern(parse_action(prediction))
    assert condition_accuracy(prediction, ConditionSpec("pattern", condition)) == 1.0


def test_malformed_surface_pattern_cannot_match():
    class NeverExecutor:
        def execute(self, query):
            raise AssertionError("malformed input must not execute")
    row = audit_candidate("i -1 2", condition="i p e p e", observation=frozenset(), executor=NeverExecutor())
    assert not row["parse_ok"] and not row["pattern_match"]


def test_exact_pattern_precedes_exact_only():
    rows = [candidate(exact=True, semantic=1.0, logp=-0.1),
            candidate(exact=True, pattern=True, semantic=1.0, logp=-9.0, sha="b")]
    assert select_candidate(rows) == (1, "exact+pattern_match")


def test_exact_only_precedes_nonexact_pattern():
    rows = [candidate(exact=True, semantic=0.9, logp=-9.0),
            candidate(pattern=True, semantic=0.999, logp=-0.1, sha="b")]
    assert select_candidate(rows)[0] == 0


def test_semantic_precedes_pattern_without_exact():
    rows = [candidate(pattern=True, semantic=0.5, logp=-0.1), candidate(semantic=0.8, logp=-9.0, sha="b")]
    assert select_candidate(rows)[0] == 1


def test_pattern_only_breaks_equal_semantic_tie():
    rows = [candidate(semantic=0.8, logp=-0.1), candidate(pattern=True, semantic=0.8, logp=-9.0, sha="b")]
    assert select_candidate(rows)[0] == 1


def test_likelihood_then_hash_are_deterministic():
    rows = [candidate(semantic=0.8, logp=-1.0, sha="z"), candidate(semantic=0.8, logp=-1.0, sha="b")]
    assert select_candidate(rows, "semantic_only")[0] == 1


def test_candidate_prefixes_are_nested_by_construction():
    stream = [candidate(sha=str(index)) for index in range(8)]
    assert stream[:1] == stream[:2][:1] == stream[:4][:1] == stream[:8][:1]
    assert stream[:2] == stream[:4][:2] == stream[:8][:2]
    assert stream[:4] == stream[:8][:4]


def test_reference_change_cannot_affect_selector():
    rows = [candidate(semantic=0.7), candidate(pattern=True, semantic=0.8, sha="b")]
    before = select_candidate(rows)
    references = ["-1 2", "u -2 3 -4 5"]
    assert all(select_candidate(rows) == before for _ in references)


def test_gate_contract():
    base = {"semantic_average": 0.7, "exact_rate": 0.2, "pattern_accuracy": 0.95,
            "parse_rate": 1.0, "eos_rate": 1.0, "cost": {"wall_seconds": 1.0,
            "generated_sequences": 4, "generated_tokens": 20, "graph_executions": 4}}
    summaries = {"greedy": dict(base), "pattern_aware": {**base, "semantic_average": 0.8, "exact_rate": 0.3},
                 "exact_semantic": {**base, "semantic_average": 0.801, "pattern_accuracy": 0.94}}
    assert p1_gate(summaries)["passed"]
    summaries["pattern_aware"]["pattern_accuracy"] = 0.93
    assert not p1_gate(summaries)["passed"]

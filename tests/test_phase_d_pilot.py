from types import SimpleNamespace

from akgr.reproduction.phase_d_pilot import (
    _query_signature,
    sample_fresh_rl_records,
)


def _config():
    return SimpleNamespace(
        dataset="WN18RR",
        seed=42,
        raw={
            "data": {"max_answers": 32, "max_attempts_per_record": 10},
        },
    )


def test_query_signature_ignores_record_id_and_answers():
    first = {"record_id": "a", "answers": [1], "query": ["p", 1], "pattern_str": "p(e)"}
    second = {"record_id": "b", "answers": [2], "query": ["p", 1], "pattern_str": "p(e)"}
    assert _query_signature(first) == _query_signature(second)


def test_fresh_sampler_rejects_forbidden_and_internal_query_duplicates(monkeypatch):
    patterns = [("p(e)", "1p"), ("i(p(e),p(e))", "2i")]
    monkeypatch.setattr("akgr.reproduction.phase_d_pilot._patterns", lambda _: patterns)
    calls = 0

    def fake_run_tasks(tasks, graph_samplers, workers):
        nonlocal calls
        calls += 1
        records = []
        for task in tasks:
            # The first round deliberately emits one forbidden query for 1p
            # and one duplicated query for 2i.  Later rounds fill the holes.
            if calls == 1 and task["pattern_abbr"] == "1p" and task["ordinal"] == 0:
                query = ["forbidden"]
            elif calls == 1 and task["pattern_abbr"] == "2i":
                query = ["duplicate"]
            else:
                query = [task["pattern_abbr"], task["ordinal"], calls]
            records.append({
                "record_id": f"r-{task['pattern_abbr']}-{task['ordinal']}-{calls}",
                "answers": [task["ordinal"]],
                "query": query,
                "pattern_str": task["pattern_str"],
            })
        return records, []

    monkeypatch.setattr("akgr.reproduction.phase_d_pilot.run_tasks", fake_run_tasks)
    forbidden_record = {
        "answers": [99], "query": ["forbidden"], "pattern_str": "p(e)"
    }
    records, audit = sample_fresh_rl_records(
        config=_config(),
        graph_samplers={},
        count_per_pattern=2,
        forbidden_queries={_query_signature(forbidden_record)},
        workers=1,
    )
    assert len(records) == 4
    assert audit["counts_by_pattern_abbreviation"] == {"1p": 2, "2i": 2}
    assert audit["overlap_with_sft_or_validation"] == 0
    assert audit["internal_query_duplicates"] == 0
    assert audit["rejected_forbidden"] == 1
    assert audit["seed_namespace"] == "phase_d_repaired_pilot"


def test_fresh_sampler_uses_explicit_namespace(monkeypatch):
    monkeypatch.setattr(
        "akgr.reproduction.phase_d_pilot._patterns", lambda _: [("p(e)", "1p")]
    )
    observed_seeds = []

    def fake_run_tasks(tasks, graph_samplers, workers):
        observed_seeds.extend(task["task_seed"] for task in tasks)
        return ([{
            "record_id": "r",
            "answers": [1],
            "query": ["fresh"],
            "pattern_str": "p(e)",
        }], [])

    monkeypatch.setattr("akgr.reproduction.phase_d_pilot.run_tasks", fake_run_tasks)
    _, audit = sample_fresh_rl_records(
        config=_config(), graph_samplers={}, count_per_pattern=1,
        forbidden_queries=set(), workers=1, sampling_namespace="full-v1",
    )
    assert audit["seed_namespace"] == "full-v1"
    assert len(observed_seeds) == 1

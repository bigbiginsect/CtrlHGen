import hashlib
import json

import networkx as nx

from akgr.kgdata.kgclass import GraphSampler
from akgr.sampling import sample_parallel


class NeverSampler:
    def try_sample_query_given_pattern(self, pattern, rng=None):
        return None


def _graph_samplers():
    graph = nx.MultiDiGraph()
    for source in range(10):
        graph.add_edge(source, source + 1, key=0)
    sampler = GraphSampler(graph, {0: "+r"})
    return {"train": sampler, "valid": sampler, "test": sampler}


def _digest(records):
    payload = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_task_count_is_exact_per_pattern():
    patterns = [("(p,(e))", "1p"), ("(p,(p,(e)))", "2p")]
    tasks = sample_parallel.build_tasks(
        dataset="WN18RR",
        mode="train",
        patterns=patterns,
        count_per_pattern=8,
        seed=42,
        max_answers=32,
        max_attempts=100,
    )
    assert len(tasks) == 16
    assert len({task["task_seed"] for task in tasks}) == 16


def test_worker_count_does_not_change_records():
    tasks = sample_parallel.build_tasks(
        dataset="WN18RR",
        mode="train",
        patterns=[("(p,(e))", "1p")],
        count_per_pattern=12,
        seed=42,
        max_answers=32,
        max_attempts=100,
    )
    serial, serial_failures = sample_parallel.run_tasks(tasks, _graph_samplers(), workers=1)
    parallel, parallel_failures = sample_parallel.run_tasks(tasks, _graph_samplers(), workers=2)
    assert not serial_failures
    assert not parallel_failures
    assert _digest(serial) == _digest(parallel)


def test_bounded_failure_reports_exact_attempt_budget():
    sample_parallel.init_workers({"train": NeverSampler()})
    task = sample_parallel.build_tasks(
        dataset="WN18RR",
        mode="train",
        patterns=[("(p,(e))", "1p")],
        count_per_pattern=1,
        seed=42,
        max_answers=32,
        max_attempts=7,
    )[0]
    result = sample_parallel.sample_task(task)
    assert result["ok"] is False
    assert result["attempts"] == 7


def test_cross_split_dedup_freezes_test_and_resamples_train(monkeypatch):
    duplicate = {
        "answers": [1],
        "query": ["(", "e", "(", 1, ")", ")"],
        "pattern_str": "(e)",
        "record_id": "duplicate",
    }
    valid = {**duplicate, "answers": [2], "record_id": "valid"}
    sampled = {
        "train": [dict(duplicate)],
        "valid": [valid],
        "test": [dict(duplicate)],
    }
    task = {
        "task_index": 0,
        "dataset": "WN18RR",
        "mode": "train",
        "pattern_str": "(e)",
        "pattern_abbr": "1e",
        "ordinal": 0,
        "task_seed": 42,
        "max_answers": 32,
        "max_attempts": 10,
    }
    tasks = {
        split: [{**task, "mode": split}] for split in ("train", "valid", "test")
    }

    def replacement(replacement_task):
        return {
            "ok": True,
            "record": {
                **duplicate,
                "answers": [3],
                "query": ["(", "e", "(", 3, ")", ")"],
                "record_id": "replacement",
            },
        }

    monkeypatch.setattr(sample_parallel, "sample_task", replacement)
    report = sample_parallel.deduplicate_full_supervision_across_splits(
        sampled, tasks, graph_samplers={}
    )

    assert sampled["test"] == [duplicate]
    assert sampled["valid"] == [valid]
    assert sampled["train"][0]["record_id"] == "replacement"
    assert report["replacement_count"] == 1
    assert not any(report["post_dedup_overlap"].values())

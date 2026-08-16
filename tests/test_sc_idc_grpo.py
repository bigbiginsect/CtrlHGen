from __future__ import annotations

import json

import networkx as nx
import pytest

pytest.importorskip("torch")

from akgr.kgdata import GraphSampler
from akgr.reproduction.sc_idc import parse_action
from akgr.reproduction.sc_idc_grpo import PairedGRPORewardAdapter
from akgr.reproduction.sc_idc_phase2_reward import CachedQueryExecutor


def _sampler():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(8))
    graph.add_edge(0, 2, key=0)
    graph.add_edge(1, 3, key=1)
    graph.add_edge(4, 5, key=2)
    graph.add_edge(6, 7, key=3)
    return GraphSampler(
        graph,
        {0: "+controlled", 1: "+other-a", 2: "+other-b", 3: "+other-c"},
    )


def _batch(adapter):
    kwargs = {
        "prompts": ["prompt"],
        "completions": ["-1 1"],
        "source": ["3"],
        "target": ["-1 1"],
        "condition": ["-1"],
        "record_id": ["record-a"],
    }
    base = adapter.base_reward(**kwargs)
    return kwargs, base


def test_baseline_adapter_returns_the_original_combined_reward(tmp_path):
    adapter = PairedGRPORewardAdapter(
        _sampler(), branch="baseline", progress_path=tmp_path / "progress.json"
    )
    _, base = _batch(adapter)
    assert base == pytest.approx([1.0 + 0.5 + 0.5 / 1.00001 + 1.0])
    funcs, weights = adapter.reward_functions()
    assert [func.__name__ for func in funcs] == ["base_reward"]
    assert weights == [1.0]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["parse_ok_rate"] == 1.0
    assert progress["nominal_adherence_rate"] == 1.0
    assert progress["graph"]["executions"] == 1


def test_sc_idc_adapter_adds_only_alpha_weighted_raw_bss(tmp_path):
    adapter = PairedGRPORewardAdapter(
        _sampler(), branch="sc_idc", progress_path=tmp_path / "progress.json"
    )
    kwargs, base = _batch(adapter)
    raw_bss = adapter.sc_idc_reward(**kwargs)
    funcs, weights = adapter.reward_functions()
    assert [func.__name__ for func in funcs] == ["base_reward", "sc_idc_reward"]
    assert weights == [1.0, 0.05]
    assert raw_bss == pytest.approx([1.0])
    assert base[0] + weights[1] * raw_bss[0] == pytest.approx(base[0] + 0.05)
    snapshot = adapter.snapshot()
    assert snapshot["classification_counts"] == {
        "branch_supported_selective": 1
    }
    assert snapshot["graph"]["executions"] <= 5


def test_sc_idc_reward_rejects_an_unpaired_batch(tmp_path):
    adapter = PairedGRPORewardAdapter(
        _sampler(), branch="sc_idc", progress_path=tmp_path / "progress.json"
    )
    with pytest.raises(RuntimeError, match="not paired"):
        adapter.sc_idc_reward(
            prompts=["prompt"], completions=["-1 1"], source=["3"],
            target=["-1 1"], condition=["-1"], record_id=["record-a"],
        )


def test_bounded_query_cache_evicts_lru_entries():
    cache = CachedQueryExecutor(_sampler(), max_entries=1)
    first = parse_action("-1 1")
    second = parse_action("-2 2")
    cache.execute(first)
    cache.execute(second)
    assert cache.evictions == 1
    assert len(cache.cache) == 1
    cache.execute(first)
    assert cache.executions == 3
    with pytest.raises(ValueError, match="positive"):
        CachedQueryExecutor(_sampler(), max_entries=0)

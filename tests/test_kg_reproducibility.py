import json
from types import SimpleNamespace

import pandas as pd
import pytest

from akgr.kgdata import load_kg_util


def _triples(count=20):
    return pd.DataFrame(
        [(index % 7, (index * 3) % 7, index % 3) for index in range(count)],
        columns=load_kg_util.TRIPLE_COLUMNS,
    )


def test_module_distribution_version_supports_pykeen_without_dunder_version(monkeypatch):
    module = SimpleNamespace(__name__="pykeen")
    monkeypatch.setattr(load_kg_util.importlib_metadata, "version", lambda name: "1.11.0")
    assert load_kg_util._module_distribution_version(module) == "1.11.0"

    module.__version__ = "test"
    assert load_kg_util._module_distribution_version(module) == "test"


def test_canonical_split_is_input_order_independent():
    triples = _triples()
    shuffled = triples.sample(frac=1, random_state=999)
    first = load_kg_util.split_triples(triples, seed=42)
    second = load_kg_util.split_triples(shuffled, seed=42)

    assert {key: load_kg_util.triples_hash(value) for key, value in first.items()} == {
        key: load_kg_util.triples_hash(value) for key, value in second.items()
    }
    assert {key: len(value) for key, value in first.items()} == {"train": 16, "valid": 2, "test": 2}
    row_sets = {
        split: set(map(tuple, frame[load_kg_util.CANONICAL_COLUMNS].itertuples(index=False, name=None)))
        for split, frame in first.items()
    }
    assert row_sets["train"].isdisjoint(row_sets["valid"])
    assert row_sets["train"].isdisjoint(row_sets["test"])
    assert row_sets["valid"].isdisjoint(row_sets["test"])
    assert set().union(*row_sets.values()) == set(
        map(
            tuple,
            load_kg_util.canonicalize_triples(triples)[load_kg_util.CANONICAL_COLUMNS]
            .itertuples(index=False, name=None),
        )
    )


def test_builder_records_exclusive_and_cumulative_hashes():
    kg, metadata = load_kg_util.build_kg_from_triples(
        _triples(), num_ent=7, rel_id2name={0: "a", 1: "b", 2: "c"}, seed=42
    )
    assert kg.num_rel == 6
    assert [metadata["exclusive_splits"][split]["count"] for split in ("train", "valid", "test")] == [16, 2, 2]
    assert [metadata["cumulative_graphs"][split]["count"] for split in ("train", "valid", "test")] == [32, 36, 40]


def test_loader_hits_exact_cache_before_raw_dataset(monkeypatch, tmp_path):
    calls = []

    def fake_raw(_dataname):
        calls.append("raw")
        return (
            SimpleNamespace(__version__="test"),
            "SyntheticDataset",
            _triples(),
            {index: str(index) for index in range(7)},
            {0: "a", 1: "b", 2: "c"},
        )

    monkeypatch.setattr(load_kg_util, "_load_raw_dataset", fake_raw)
    first = load_kg_util.load_kg(
        "WN18RR", data_root=tmp_path, seed=42, semantic_hash="config-a"
    )
    assert calls == ["raw"]

    monkeypatch.setattr(
        load_kg_util,
        "_load_raw_dataset",
        lambda _: (_ for _ in ()).throw(AssertionError("offline cache attempted a raw load")),
    )
    second = load_kg_util.load_kg(
        "WN18RR", data_root=tmp_path, seed=42, semantic_hash="config-a", offline=True
    )
    assert second.num_ent == first.num_ent

    with pytest.raises(FileNotFoundError):
        load_kg_util.load_kg(
            "WN18RR", data_root=tmp_path, seed=43, semantic_hash="config-a", offline=True
        )
    with pytest.raises(FileNotFoundError):
        load_kg_util.load_kg(
            "WN18RR", data_root=tmp_path, seed=42, semantic_hash="config-a",
            reverse_edges_flag=False, offline=True,
        )

    manifests = list((tmp_path / "WN18RR").glob("*.manifest.json"))
    manifest = json.loads(manifests[0].read_text())
    assert manifest["seed"] == 42
    assert manifest["source"] == {
        "provider": "pykeen",
        "dataset": "WN18RR",
        "dataset_class": "SyntheticDataset",
        "pykeen_version": "test",
    }
    assert manifest["raw"]["sha256"]
    assert manifest["exclusive_splits"]["train"]["count"] == 16
    assert manifest["mappings"]["entity"]["count"] == 7
    assert manifest["inverse_edges"] == {
        "enabled": True,
        "applied_after_exclusive_split": True,
    }
    assert manifest["split"]["ratios"] == [0.8, 0.1, 0.1]


def test_loader_rejects_tampered_manifest_instead_of_rebuilding(monkeypatch, tmp_path):
    calls = []

    def fake_raw(_dataname):
        calls.append("raw")
        return (
            SimpleNamespace(__version__="test"),
            "SyntheticDataset",
            _triples(),
            {index: str(index) for index in range(7)},
            {0: "a", 1: "b", 2: "c"},
        )

    monkeypatch.setattr(load_kg_util, "_load_raw_dataset", fake_raw)
    load_kg_util.load_kg(
        "WN18RR", data_root=tmp_path, seed=42, semantic_hash="config-a"
    )
    manifest_path = next((tmp_path / "WN18RR").glob("*.manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_kg_util.load_kg(
            "WN18RR", data_root=tmp_path, seed=42, semantic_hash="config-a"
        )
    assert calls == ["raw"]

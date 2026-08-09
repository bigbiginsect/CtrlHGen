import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from akgr.dataloader import create_reproduction_dataset
from akgr.reproduction.config import load_experiment_config


def _write_jsonl(path: Path, records):
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(payload)
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "count": len(records)}


def test_manifest_loader_selects_merged_train_and_base_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CTRLHGEN_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CTRLHGEN_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("CTRLHGEN_RUN_ROOT", str(tmp_path / "runs"))
    config = load_experiment_config("akgr/configs/reproduce/wn-pattern-tiny.yml")
    config.artifact_dir.mkdir(parents=True)

    base = {
        "answers": [0],
        "query": ["(", "p", "(", -1, ")", "(", "e", "(", 0, ")", ")", ")"],
        "pattern_str": "(p,(e))",
        "record_id": "base-0",
    }
    augmented = dict(base, record_id="augmented-0", parent_record_id="base-0")
    train_base = _write_jsonl(config.artifact_dir / "train-base.jsonl", [base])
    valid_base = _write_jsonl(config.artifact_dir / "valid-base.jsonl", [base])
    test_base = _write_jsonl(config.artifact_dir / "test-base.jsonl", [base])
    train_augmented = _write_jsonl(config.artifact_dir / "train-augmented.jsonl", [augmented])
    train_merged = _write_jsonl(config.artifact_dir / "train-merged.jsonl", [base, augmented])
    manifest = {
        "schema_version": 2,
        "dataset": config.dataset,
        "profile": config.experiment["profile"],
        "seed": config.seed,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "stats": {"nentity": 10, "nrelation": 4},
        "artifacts": {
            "base": {"train": train_base, "valid": valid_base, "test": test_base},
            "augmented_only": {"train": train_augmented},
            "merged": {"train": train_merged},
        },
    }
    config.sampling_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")
    datasets, nentity, nrelation = create_reproduction_dataset(
        config, patterns, splits=["train", "valid"], is_act=False,
        train_variant="merged",
    )
    assert len(datasets["train"]) == 2
    assert len(datasets["valid"]) == 1
    assert datasets["train"][0]["record_id"] == "base-0"
    assert datasets["train"][1]["record_id"] == "augmented-0"
    assert (nentity, nrelation) == (10, 4)

    base_datasets, _, _ = create_reproduction_dataset(
        config, patterns, splits=["train"], is_act=False, train_variant="base"
    )
    assert len(base_datasets["train"]) == 1
    assert base_datasets["train"][0]["record_id"] == "base-0"


def test_manifest_loader_rejects_artifact_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("CTRLHGEN_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CTRLHGEN_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("CTRLHGEN_RUN_ROOT", str(tmp_path / "runs"))
    config = load_experiment_config("akgr/configs/reproduce/wn-pattern-tiny.yml")
    config.artifact_dir.mkdir(parents=True)
    artifact_path = config.artifact_dir / "train-merged.jsonl"
    artifact_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "dataset": config.dataset,
        "profile": config.experiment["profile"],
        "seed": config.seed,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "stats": {"nentity": 1, "nrelation": 1},
        "artifacts": {
            "base": {},
            "augmented_only": {},
            "merged": {"train": {"path": artifact_path.name, "sha256": "bad", "count": 1}},
        },
    }
    config.sampling_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")

    try:
        create_reproduction_dataset(
            config, patterns, splits=["train"], is_act=False, train_variant="merged"
        )
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch should fail before dataset creation")

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.phase_c_iv_audit import (
    _paper_average,
    _pareto_frontier,
    _target_tokens,
    audit_data,
)


def _artifact(path: Path, records: list[dict]) -> dict:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "count": len(records),
    }


def test_target_coverage_token_extraction_distinguishes_entities_and_relations() -> None:
    query = ["(", "i", "(", "p", "(", -3, ")", "(", "e", "(", 7, ")", ")", ")", ")"]
    assert _target_tokens(query) == ([7], [3])


def test_paper_average_and_pareto_use_all_five_registered_metrics() -> None:
    first = {
        "stage_epoch": 2, "jaccard": 0.2, "dice": 0.3, "overlap": 0.4,
        "condition_accuracy": 0.5, "smatch": 0.6,
    }
    second = {**first, "stage_epoch": 4, "jaccard": 0.3}
    tradeoff = {**first, "stage_epoch": 6, "condition_accuracy": 0.7, "smatch": 0.5}
    assert _paper_average(first) == pytest.approx(0.4)
    assert _pareto_frontier([first, second, tradeoff]) == [4, 6]


def test_data_audit_verifies_hash_counts_split_dedup_and_coverage(tmp_path: Path, monkeypatch) -> None:
    env = {
        "CTRLHGEN_DATA_ROOT": str(tmp_path / "data"),
        "CTRLHGEN_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
        "CTRLHGEN_RUN_ROOT": str(tmp_path / "runs"),
    }
    config = load_experiment_config("akgr/configs/reproduce/wn-pattern-tiny.yml", env=env)
    config.artifact_dir.mkdir(parents=True)
    patterns = {f"pattern-{index}": f"p{index}" for index in range(13)}
    monkeypatch.setattr(
        "akgr.reproduction.phase_c_iv_audit._pattern_names", lambda: patterns
    )

    def records(split: str, per_pattern: int, answer_offset: int) -> list[dict]:
        result = []
        for pattern_index, pattern in enumerate(patterns):
            for ordinal in range(per_pattern):
                entity = pattern_index * 100 + ordinal
                result.append({
                    "answers": [answer_offset + entity],
                    "query": ["(", "p", "(", -(pattern_index % 4), ")", "(", "e", "(", entity, ")", ")", ")"],
                    "pattern_str": pattern,
                    "record_id": f"{split}-{pattern_index}-{ordinal}",
                })
        return result

    base = {
        "train": records("train", 8, 0),
        "valid": records("valid", 2, 10_000),
        "test": records("test", 2, 20_000),
    }
    artifacts = {
        "base": {
            split: _artifact(config.artifact_dir / f"{split}.jsonl", split_records)
            for split, split_records in base.items()
        },
        "augmented_only": {
            "train": _artifact(config.artifact_dir / "augmented.jsonl", [])
        },
        "merged": {
            "train": _artifact(config.artifact_dir / "merged.jsonl", base["train"])
        },
    }
    manifest = {
        "schema_version": 2,
        "dataset": config.dataset,
        "profile": config.experiment["profile"],
        "seed": config.seed,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "stats": {"nentity": 2000, "nrelation": 4},
        "artifacts": artifacts,
        "kg": {},
        "pattern_table": {},
        "augmentation": {},
    }
    config.sampling_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_data(config)

    assert report["status"] == "pass"
    assert report["counts"]["base"] == {"train": 104, "valid": 26, "test": 26}
    assert not any(report["split_dedup"]["record_id_overlap"].values())
    assert not any(report["split_dedup"]["full_supervision_overlap"].values())
    assert report["target_token_coverage"]["relation"]["train_coverage_rate"] == 1.0
    assert report["target_token_coverage"]["entity"]["valid"]["unseen_unique_rate"] == 0.0

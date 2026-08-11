from __future__ import annotations

import hashlib
import json

import pytest

from akgr.reproduction.phase_d_parent import publish_phase_d_parent


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_phase_d_parent_preserves_original_best_and_verifies_bakeoff(tmp_path):
    checkpoint = tmp_path / "conditional-epoch-45"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({
        "stage": "conditional",
        "stage_epoch": 45,
        "experiment_config_hash": "config-hash",
        "data_manifest_hash": "manifest-hash",
    }))
    original = tmp_path / "conditional-epoch-50"
    original.mkdir()
    (tmp_path / "conditional-best.json").write_text(json.dumps({
        "stage": "conditional",
        "stage_epoch": 50,
        "checkpoint": original.name,
        "validation": {"health_pass": True},
    }))
    history = tmp_path / "conditional-history.jsonl"
    history.write_text(json.dumps({
        "stage_epoch": 45,
        "validation": {"health_pass": True, "jaccard": 0.6, "condition_accuracy": 0.94},
    }) + "\n")
    artifact = tmp_path / "test-greedy.jsonl"
    artifact.write_text("{}\n")
    candidate_metrics = {
        "jaccard": 0.60,
        "dice": 0.65,
        "overlap": 0.72,
        "condition_accuracy": 0.94,
        "smatch": 0.82,
        "paper_average": 0.746,
        "parse_ok": 0.99,
        "eos_rate": 1.0,
    }
    deltas = {
        "jaccard": 0.006,
        "dice": 0.007,
        "overlap": 0.006,
        "condition_accuracy": -0.001,
        "smatch": -0.003,
        "paper_average": 0.003,
    }
    summary = tmp_path / "bakeoff.json"
    summary.write_text(json.dumps({
        "status": "pass",
        "candidate_checkpoint": str(checkpoint),
        "baseline_checkpoint": str(original),
        "config_semantic_hash": "config-hash",
        "data_manifest_sha256": "manifest-hash",
        "decoding": {
            "greedy": {"epoch45": candidate_metrics, "epoch45_minus_epoch50": deltas},
            "sampled": {"epoch45": candidate_metrics, "epoch45_minus_epoch50": deltas},
        },
        "artifacts": {
            str(artifact): {"sha256": _sha256(artifact), "bytes": artifact.stat().st_size},
        },
    }))

    pointer = publish_phase_d_parent(
        checkpoint=checkpoint,
        history_path=history,
        bakeoff_summary_path=summary,
        expected_config_hash="config-hash",
        expected_data_manifest_hash="manifest-hash",
    )
    payload = json.loads(pointer.read_text())
    assert payload["checkpoint"] == "conditional-epoch-45"
    assert payload["original_training_selection"] == "conditional-epoch-50"
    assert payload["selection"]["policy"] == "post_phase_c_reproduction_oriented_bakeoff"
    assert (tmp_path / "phase-d-parent").resolve() == checkpoint.resolve()
    assert json.loads((tmp_path / "conditional-best.json").read_text())["checkpoint"] == original.name

    assert publish_phase_d_parent(
        checkpoint=checkpoint,
        history_path=history,
        bakeoff_summary_path=summary,
        expected_config_hash="config-hash",
        expected_data_manifest_hash="manifest-hash",
    ) == pointer


def test_publish_phase_d_parent_rejects_non_improving_semantic_bakeoff(tmp_path):
    checkpoint = tmp_path / "conditional-epoch-45"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({
        "stage": "conditional", "stage_epoch": 45,
        "experiment_config_hash": "config-hash", "data_manifest_hash": "manifest-hash",
    }))
    original = tmp_path / "conditional-epoch-50"
    original.mkdir()
    (tmp_path / "conditional-best.json").write_text(json.dumps({
        "stage": "conditional", "stage_epoch": 50,
        "checkpoint": original.name, "validation": {"health_pass": True},
    }))
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps({
        "stage_epoch": 45, "validation": {"health_pass": True},
    }) + "\n")
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("{}\n")
    summary = tmp_path / "bakeoff.json"
    report = {
        "epoch45": {"parse_ok": 0.99, "eos_rate": 1.0, "condition_accuracy": 0.94},
        "epoch45_minus_epoch50": {
            "jaccard": -0.001, "dice": 0.001, "overlap": 0.001, "paper_average": 0.001,
        },
    }
    summary.write_text(json.dumps({
        "status": "pass", "candidate_checkpoint": str(checkpoint),
        "baseline_checkpoint": str(original),
        "config_semantic_hash": "config-hash", "data_manifest_sha256": "manifest-hash",
        "decoding": {"greedy": report, "sampled": report},
        "artifacts": {str(artifact): {"sha256": _sha256(artifact), "bytes": artifact.stat().st_size}},
    }))

    with pytest.raises(ValueError, match="improve all semantic metrics"):
        publish_phase_d_parent(
            checkpoint=checkpoint,
            history_path=history,
            bakeoff_summary_path=summary,
            expected_config_hash="config-hash",
            expected_data_manifest_hash="manifest-hash",
        )

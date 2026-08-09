import json
import hashlib
from types import SimpleNamespace

from akgr.sampling import sample_parallel


def test_sample_manifest_schema_and_profile_counts(monkeypatch, tmp_path):
    cache_path = tmp_path / "synthetic-kg.pkl"
    cache_path.write_bytes(b"synthetic-cache")
    cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    kg_metadata = {
        "schema_version": 1,
        "dataset": "WN18RR",
        "source": {"provider": "pykeen", "dataset": "WN18RR", "dataset_class": "Synthetic", "pykeen_version": "test"},
        "raw": {"count": 10, "sha256": "raw-hash"},
        "mappings": {
            "entity": {"count": 10, "sha256": "entity-hash"},
            "relation": {"count": 4, "sha256": "relation-hash"},
        },
        "exclusive_splits": {split: {"count": count, "sha256": f"{split}-hash"} for split, count in {"train": 8, "valid": 1, "test": 1}.items()},
        "cumulative_graphs": {split: {"count": count, "sha256": f"graph-{split}-hash"} for split, count in {"train": 16, "valid": 18, "test": 20}.items()},
        "inverse_edges": {"enabled": True, "applied_after_exclusive_split": True},
        "split": {"algorithm": "synthetic", "seed": 42, "effective_seed": 42, "ratios": [0.8, 0.1, 0.1]},
        "cache": {"path": cache_path.name, "sha256": cache_sha},
    }
    kg_manifest_path = tmp_path / "synthetic-kg.manifest.json"
    kg_manifest_path.write_text(json.dumps(kg_metadata), encoding="utf-8")
    fake_kg = SimpleNamespace(
        num_ent=10,
        num_rel=4,
        graph_samplers={"train": object(), "valid": object(), "test": object()},
        cache_path=cache_path,
        cache_manifest_path=kg_manifest_path,
        cache_manifest=kg_metadata,
    )
    patterns = [(f"pattern-{index}", f"p{index}") for index in range(13)]

    def fake_run(tasks, graph_samplers, workers):
        return (
            [
                {
                    "answers": [1],
                    "query": ["(", "e", "(", task["ordinal"], ")", ")"],
                    "pattern_str": task["pattern_str"],
                    "record_id": f"record-{task['mode']}-{task['task_index']}",
                }
                for task in tasks
            ],
            [],
        )

    monkeypatch.setattr(sample_parallel, "load_kg", lambda *args, **kwargs: fake_kg)
    monkeypatch.setattr(sample_parallel, "_patterns", lambda _: patterns)
    monkeypatch.setattr(sample_parallel, "run_tasks", fake_run)

    manifest_path = sample_parallel.sample_dataset(
        dataset="WN18RR",
        profile="tiny",
        seed=42,
        data_hash="data-hash",
        kg_hash="kg-hash",
        data_root=tmp_path,
        counts={"train": 8, "valid": 2, "test": 2},
        max_answers=32,
        max_attempts=10,
        workers=4,
        reverse_edges=True,
        split_ratios=(0.8, 0.1, 0.1),
        pattern_table_file="akgr/metadata/pattern_table.csv",
        augmentation_enabled=False,
        augmentation_patterns=(),
    )
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {
        "schema_version", "dataset", "profile", "seed", "data_hash", "kg_hash", "stats", "artifacts",
        "kg", "pattern_table", "augmentation",
    }
    assert manifest["schema_version"] == 2
    assert manifest["data_hash"] == "data-hash"
    assert manifest["kg_hash"] == "kg-hash"
    assert manifest["artifacts"]["base"]["train"]["count"] == 13 * 8
    assert manifest["artifacts"]["base"]["valid"]["count"] == 13 * 2
    assert manifest["artifacts"]["base"]["test"]["count"] == 13 * 2
    assert manifest["artifacts"]["augmented_only"]["train"]["count"] == 0
    assert manifest["artifacts"]["merged"]["train"]["count"] == 13 * 8
    for group in manifest["artifacts"].values():
        for artifact in group.values():
            assert not artifact["path"].startswith("/")
            assert len(artifact["sha256"]) == 64
    assert manifest["kg"]["manifest"]["sha256"] == hashlib.sha256(kg_manifest_path.read_bytes()).hexdigest()
    assert manifest["kg"]["cache"]["sha256"] == cache_sha
    assert manifest["kg"]["source"]["provider"] == "pykeen"
    assert manifest["kg"]["raw"]["sha256"] == "raw-hash"
    assert manifest["kg"]["mappings"]["entity"]["sha256"] == "entity-hash"
    assert manifest["kg"]["split"]["ratios"] == [0.8, 0.1, 0.1]
    assert len(manifest["pattern_table"]["sha256"]) == 64
    assert len(manifest["augmentation"]["report"]["sha256"]) == 64
    assert manifest["augmentation"]["counts"]["base_train"]["total"] == 104
    assert manifest["augmentation"]["counts"]["merged_train"]["total"] == 104


def _runtime_env(tmp_path):
    return {
        "CTRLHGEN_DATA_ROOT": str(tmp_path / "data"),
        "CTRLHGEN_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
        "CTRLHGEN_RUN_ROOT": str(tmp_path / "runs"),
    }


def test_strict_sampling_run_writes_snapshots_and_completed_status(monkeypatch, tmp_path):
    for name, value in _runtime_env(tmp_path).items():
        monkeypatch.setenv(name, value)

    def fake_sample_dataset(**kwargs):
        path = kwargs["artifact_dir"] / "sampling-manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema_version":1}\n', encoding="utf-8")
        return path

    monkeypatch.setattr(sample_parallel, "sample_dataset", fake_sample_dataset)
    sample_parallel._strict_main("akgr/configs/reproduce/wn-pattern-tiny.yml")
    run_dirs = list((tmp_path / "runs" / "sampling").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "experiment.raw.yml").is_file()
    assert (run_dir / "experiment.resolved.yml").is_file()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["sampling_manifest"]["sha256"] == hashlib.sha256(
        (tmp_path / "data" / "WN18RR" / "tiny" / "seed-42" / status["data_hash"][:12] / "sampling-manifest.json").read_bytes()
    ).hexdigest()


def test_strict_sampling_run_writes_failed_status(monkeypatch, tmp_path):
    for name, value in _runtime_env(tmp_path).items():
        monkeypatch.setenv(name, value)

    def fail_sampling(**kwargs):
        raise RuntimeError("synthetic sampling failure")

    monkeypatch.setattr(sample_parallel, "sample_dataset", fail_sampling)
    try:
        sample_parallel._strict_main("akgr/configs/reproduce/wn-pattern-tiny.yml")
    except RuntimeError as exc:
        assert "synthetic sampling failure" in str(exc)
    else:
        raise AssertionError("strict sampling failure should be re-raised")
    run_dir = next((tmp_path / "runs" / "sampling").iterdir())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == {"type": "RuntimeError", "message": "synthetic sampling failure"}
    assert (run_dir / "experiment.raw.yml").is_file()
    assert (run_dir / "experiment.resolved.yml").is_file()

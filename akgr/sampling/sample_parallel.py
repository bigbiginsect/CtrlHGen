"""Deterministic bounded query sampling for CtrlHGen.

Use ``--experiment-config`` for the strict WN18RR reproduction path. Historical
``--config-sampling/--scale`` arguments remain available as a legacy interface.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from akgr.kgdata import load_kg
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.seed import derive_seed
from akgr.sampling.sample_add import augment_records, ensure_record_ids


_GRAPH_SAMPLERS = None


def init_workers(graph_samplers) -> None:
    global _GRAPH_SAMPLERS
    _GRAPH_SAMPLERS = graph_samplers


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def judge(answers_from: Mapping[str, list[int]], mode: str) -> bool:
    if mode == "train":
        return bool(answers_from["train"])
    if mode == "valid":
        return (
            bool(answers_from["train"])
            and bool(answers_from["valid"])
            and len(answers_from["train"]) != len(answers_from["valid"])
        )
    if mode == "test":
        return (
            bool(answers_from["train"])
            and bool(answers_from["valid"])
            and bool(answers_from["test"])
            and len(answers_from["test"]) != len(answers_from["valid"])
        )
    raise ValueError(f"Unknown split: {mode}")


def _record_id(task: Mapping[str, Any], query: list[Any]) -> str:
    payload = [task["dataset"], task["mode"], task["pattern_abbr"], task["ordinal"], query]
    return f"rec-{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()[:20]}"


def sample_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Run one independently seeded, bounded sampling task."""
    if _GRAPH_SAMPLERS is None:
        raise RuntimeError("Sampling workers have not been initialized")
    rng = random.Random(int(task["task_seed"]))
    last_reason = "query construction failed"
    mode = task["mode"]
    required_splits = {"train": ("train",), "valid": ("train", "valid"), "test": ("train", "valid", "test")}[mode]
    for attempt in range(1, int(task["max_attempts"]) + 1):
        query = _GRAPH_SAMPLERS[mode].try_sample_query_given_pattern(task["pattern_str"], rng=rng)
        if query is None:
            continue
        answers_from = {
            split: sorted(int(value) for value in _GRAPH_SAMPLERS[split].search_answers_to_query(query))
            for split in required_splits
        }
        if len(answers_from[mode]) > int(task["max_answers"]):
            last_reason = f"answer count {len(answers_from[mode])} exceeds max_answers"
            continue
        if not judge(answers_from, mode):
            last_reason = "query does not satisfy cross-split novelty constraints"
            continue
        normalized_query = _normalize(query)
        return {
            "ok": True,
            "task_index": task["task_index"],
            "attempts": attempt,
            "record": {
                "answers": answers_from[mode],
                "query": normalized_query,
                "pattern_str": task["pattern_str"],
                "record_id": _record_id(task, normalized_query),
            },
        }
    return {
        "ok": False,
        "task_index": task["task_index"],
        "dataset": task["dataset"],
        "split": mode,
        "pattern": task["pattern_str"],
        "pattern_abbr": task["pattern_abbr"],
        "ordinal": task["ordinal"],
        "task_seed": task["task_seed"],
        "attempts": task["max_attempts"],
        "reason": last_reason,
    }


def build_tasks(
    *,
    dataset: str,
    mode: str,
    patterns: Iterable[tuple[str, str]],
    count_per_pattern: int,
    seed: int,
    max_answers: int,
    max_attempts: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for pattern_str, pattern_abbr in patterns:
        for ordinal in range(int(count_per_pattern)):
            task_index = len(tasks)
            tasks.append(
                {
                    "task_index": task_index,
                    "dataset": dataset,
                    "mode": mode,
                    "pattern_str": pattern_str,
                    "pattern_abbr": pattern_abbr,
                    "ordinal": ordinal,
                    "task_seed": derive_seed(seed, dataset, mode, pattern_abbr, ordinal),
                    "max_answers": int(max_answers),
                    "max_attempts": int(max_attempts),
                }
            )
    return tasks


def run_tasks(tasks: list[dict[str, Any]], graph_samplers, workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        init_workers(graph_samplers)
        results = [sample_task(task) for task in tasks]
    else:
        with Pool(processes=workers, initializer=init_workers, initargs=(graph_samplers,)) as pool:
            results = list(pool.imap(sample_task, tasks, chunksize=1))
    results.sort(key=lambda result: result["task_index"])
    records = [result["record"] for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    return records, failures


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def _artifact(path: Path, manifest_dir: Path, count: int) -> dict[str, Any]:
    return {"path": str(path.relative_to(manifest_dir)), "sha256": _file_sha256(path), "count": int(count)}


def _file_reference(path: Path, manifest_dir: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": os.path.relpath(resolved, manifest_dir.resolve()),
        "resolved_path": str(resolved),
        "sha256": _file_sha256(resolved),
    }


def _pattern_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        pattern = str(record["pattern_str"])
        counts[pattern] = counts.get(pattern, 0) + 1
    return dict(sorted(counts.items()))


def _kg_audit(kg, manifest_dir: Path) -> dict[str, Any]:
    manifest_path = getattr(kg, "cache_manifest_path", None)
    cache_path = getattr(kg, "cache_path", None)
    metadata = getattr(kg, "cache_manifest", None)
    if manifest_path is None or cache_path is None or not isinstance(metadata, dict):
        raise RuntimeError("KG loader did not expose cache manifest provenance")
    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path)
    on_disk_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if on_disk_metadata != metadata:
        raise ValueError("KG manifest changed after the cache was loaded")
    manifest_reference = _file_reference(manifest_path, manifest_dir)
    cache_reference = _file_reference(cache_path, manifest_dir)
    expected_cache_hash = metadata.get("cache", {}).get("sha256")
    if cache_reference["sha256"] != expected_cache_hash:
        raise ValueError(
            f"KG cache checksum changed after load: expected {expected_cache_hash}, "
            f"found {cache_reference['sha256']}"
        )
    source = metadata.get("source")
    if source is None:
        source = {
            "provider": "pykeen",
            "dataset": metadata.get("dataset"),
            "dataset_class": metadata.get("dataset_class"),
            "pykeen_version": metadata.get("pykeen_version"),
        }
    mappings = metadata.get("mappings")
    if mappings is None:
        mappings = {
            "entity": {"sha256": metadata["entity_mapping_sha256"]},
            "relation": {"sha256": metadata["relation_mapping_sha256"]},
        }
    inverse_edges = metadata.get("inverse_edges")
    if inverse_edges is None:
        inverse_edges = {
            "enabled": metadata["reverse_edges"],
            "applied_after_exclusive_split": True,
        }
    split = metadata.get("split")
    if split is None:
        split = {
            "algorithm": "canonical-sort+numpy-default-rng-permutation-v1",
            "seed": metadata.get("seed"),
            "effective_seed": metadata.get("effective_seed"),
            "ratios": metadata["split_ratios"],
        }
    return {
        "manifest": manifest_reference,
        "cache": cache_reference,
        "source": source,
        "raw": metadata["raw"],
        "mappings": mappings,
        "exclusive_splits": metadata["exclusive_splits"],
        "cumulative_graphs": metadata["cumulative_graphs"],
        "inverse_edges": inverse_edges,
        "split": split,
    }


def _patterns(pattern_table_file: str) -> list[tuple[str, str]]:
    table = pd.read_csv(pattern_table_file)
    filtered = table.loc[table["original_depth"] <= 2, ["original", "pattern_abbr"]]
    result = [(str(row.original), str(row.pattern_abbr)) for row in filtered.itertuples()]
    if len(result) != 13:
        raise ValueError(f"Expected 13 reproduction patterns, found {len(result)}")
    return result


def sample_dataset(
    *,
    dataset: str,
    profile: str,
    seed: int,
    data_hash: str | None,
    kg_hash: str | None,
    data_root: Path,
    artifact_dir: Path | None = None,
    manifest_filename: str = "sampling-manifest.json",
    counts: Mapping[str, int],
    max_answers: int,
    max_attempts: int,
    workers: int,
    reverse_edges: bool,
    split_ratios: Iterable[float],
    pattern_table_file: str,
    augmentation_enabled: bool,
    augmentation_patterns: Iterable[str],
) -> Path:
    augmentation_patterns = tuple(augmentation_patterns)
    kg = load_kg(
        dataset,
        reverse_edges_flag=reverse_edges,
        data_root=data_root,
        seed=seed,
        split_ratios=tuple(split_ratios),
        semantic_hash=kg_hash,
    )
    pattern_table_path = Path(pattern_table_file).expanduser().resolve()
    patterns = _patterns(str(pattern_table_path))
    sampled: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        tasks = build_tasks(
            dataset=dataset,
            mode=split,
            patterns=patterns,
            count_per_pattern=int(counts[split]),
            seed=seed,
            max_answers=max_answers,
            max_attempts=max_attempts,
        )
        sampled[split], split_failures = run_tasks(tasks, kg.graph_samplers, workers)
        failures.extend(split_failures)

    output_dir = artifact_dir if artifact_dir is not None else data_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / f"{dataset}-{profile}-seed{seed}-sampling-failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Sampling failed for {len(failures)} tasks; no sample manifest was published. See {failure_path}"
        )

    expected_counts = {split: len(patterns) * int(counts[split]) for split in counts}
    actual_counts = {split: len(sampled[split]) for split in sampled}
    if actual_counts != expected_counts:
        raise AssertionError(f"Profile count mismatch: expected {expected_counts}, got {actual_counts}")

    (output_dir / "stats.txt").write_text(
        f"nentity\t{kg.num_ent}\nnrelation\t{kg.num_rel}\n", encoding="utf-8"
    )
    base_paths: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        path = output_dir / f"{dataset}-{profile}-{max_answers}-{split}-a2q.jsonl"
        _write_jsonl(path, sampled[split])
        base_paths[split] = path

    augmented: list[dict[str, Any]] = []
    augmentation_report: dict[str, Any] = {"generated_count": 0, "skipped_count": 0}
    if augmentation_enabled:
        augmented, augmentation_report = augment_records(
            sampled["train"],
            kg.graph_samplers["train"],
            max_answers=max_answers,
            source_patterns=augmentation_patterns,
        )
        missing_augmentations = [
            pattern
            for pattern in augmentation_patterns
            if not augmentation_report["generated_source_counts"].get(pattern)
        ]
        if missing_augmentations:
            raise RuntimeError(
                "Augmentation produced no legal records for source patterns: "
                + ", ".join(missing_augmentations)
            )
    augmented_path = output_dir / f"{dataset}-{profile}-{max_answers}-train-augmented-only-a2q.jsonl"
    _write_jsonl(augmented_path, augmented)
    merged = ensure_record_ids(sampled["train"], "train") + augmented
    merged_path = output_dir / f"{dataset}-{profile}-{max_answers}-train-merged-a2q.jsonl"
    _write_jsonl(merged_path, merged)
    report_path = output_dir / f"{dataset}-{profile}-{max_answers}-augmentation-report.json"
    report_path.write_text(json.dumps(augmentation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = output_dir / manifest_filename
    artifacts = {
        "base": {
            split: _artifact(path, output_dir, len(sampled[split]))
            for split, path in base_paths.items()
        },
        "augmented_only": {"train": _artifact(augmented_path, output_dir, len(augmented))},
        "merged": {"train": _artifact(merged_path, output_dir, len(merged))},
    }
    manifest = {
        "schema_version": 2,
        "dataset": dataset,
        "profile": profile,
        "seed": int(seed),
        "data_hash": data_hash,
        "kg_hash": kg_hash,
        "stats": {"nentity": kg.num_ent, "nrelation": kg.num_rel},
        "artifacts": artifacts,
        "kg": _kg_audit(kg, output_dir),
        "pattern_table": {
            **_file_reference(pattern_table_path, output_dir),
            "pattern_count": len(patterns),
        },
        "augmentation": {
            "enabled": bool(augmentation_enabled),
            "source_patterns": list(augmentation_patterns),
            "report": _file_reference(report_path, output_dir),
            "counts": {
                "base_train": {
                    "total": len(sampled["train"]),
                    "by_pattern": _pattern_counts(sampled["train"]),
                },
                "augmented_only_train": {
                    "total": len(augmented),
                    "by_pattern": _pattern_counts(augmented),
                },
                "merged_train": {
                    "total": len(merged),
                    "by_pattern": _pattern_counts(merged),
                },
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _strict_main(experiment_config: str) -> None:
    config = load_experiment_config(experiment_config)
    raw = config.raw
    timestamp = datetime.now(timezone.utc)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw["experiment"]["name"])).strip("-")
    run_id = f"{safe_name}-seed{config.seed}-{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-pid{os.getpid()}"
    run_dir = config.runtime_paths["run_root"] / "sampling" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    status = {
        "schema_version": 1,
        "stage": "sampling",
        "status": "running",
        "run_id": run_id,
        "started_at": timestamp.isoformat(),
        "pid": os.getpid(),
        "command": [sys.executable, "-m", "akgr.sampling.sample_parallel", "--experiment-config", str(config.source_path)],
        "semantic_hash": config.semantic_hash,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "artifact_dir": str(config.artifact_dir),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        config.write_snapshots(run_dir)
        manifest = sample_dataset(
            dataset=config.dataset,
            profile=raw["experiment"]["profile"],
            seed=config.seed,
            data_hash=config.data_hash,
            kg_hash=config.kg_hash,
            data_root=config.runtime_paths["data_root"],
            artifact_dir=config.artifact_dir,
            counts={
                "train": raw["sampling"]["train_per_pattern"],
                "valid": raw["sampling"]["valid_per_pattern"],
                "test": raw["sampling"]["test_per_pattern"],
            },
            max_answers=raw["data"]["max_answers"],
            max_attempts=raw["data"]["max_attempts_per_record"],
            workers=raw["data"]["workers"],
            reverse_edges=raw["data"]["reverse_edges"],
            split_ratios=raw["data"]["split_ratios"],
            pattern_table_file="akgr/metadata/pattern_table.csv",
            augmentation_enabled=raw["augmentation"]["enabled"],
            augmentation_patterns=raw["augmentation"]["source_patterns"],
        )
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    status.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "sampling_manifest": _file_reference(manifest, run_dir),
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"# Sampling run: {run_dir}")
    print(f"# Sample manifest: {manifest}")


def _legacy_main(args: argparse.Namespace) -> None:
    with Path(args.config_sampling).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.scale not in config:
        raise ValueError(f"Unknown legacy sampling scale: {args.scale!r}")
    profile_config = config[args.scale]
    datasets = profile_config.get("datasets") or [
        key for key, value in profile_config.items() if isinstance(value, dict)
    ]
    if not datasets:
        raise ValueError(f"Legacy profile {args.scale!r} does not define datasets")
    for dataset in datasets:
        explicit = profile_config.get(dataset)
        if explicit is None:
            scaling_factor = profile_config.get("scale")
            if not scaling_factor:
                raise ValueError(
                    f"Legacy profile {args.scale!r} has neither explicit counts nor an edge scale for {dataset}"
                )
            legacy_kg = load_kg(
                dataset,
                data_root=Path(args.data_root).expanduser().resolve(),
                seed=args.seed,
            )
            train_count = legacy_kg.num_train_edges // int(scaling_factor)
            explicit = {"train": train_count, "valid": train_count // 8, "test": train_count // 8}
        manifest = sample_dataset(
            dataset=dataset,
            profile=args.scale,
            seed=args.seed,
            data_hash=None,
            kg_hash=None,
            data_root=Path(args.data_root).expanduser().resolve(),
            manifest_filename=f"{dataset}-{args.scale}-seed{args.seed}-sample-manifest.json",
            counts=explicit,
            max_answers=args.max_answer_size,
            max_attempts=args.max_attempts,
            workers=args.nproc,
            reverse_edges=True,
            split_ratios=(0.8, 0.1, 0.1),
            pattern_table_file=config["pattern_table_file"],
            augmentation_enabled=False,
            augmentation_patterns=(),
        )
        print(f"# Sample manifest: {manifest}")


def my_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config")
    parser.add_argument("-c", "--config-sampling")
    parser.add_argument("-s", "--scale")
    parser.add_argument("-a", "--max-answer-size", type=int)
    parser.add_argument("-p", "--nproc", type=int)
    parser.add_argument("--data_root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    legacy_values = [
        args.config_sampling,
        args.scale,
        args.max_answer_size,
        args.nproc,
        args.data_root,
        args.seed,
        args.max_attempts,
    ]
    if args.experiment_config and any(value is not None for value in legacy_values):
        parser.error("--experiment-config cannot be combined with legacy sampling arguments")
    if not args.experiment_config:
        args.config_sampling = args.config_sampling or "akgr/configs/config-sampling.yml"
        args.scale = args.scale or "debug"
        args.max_answer_size = args.max_answer_size or 32
        args.nproc = args.nproc or 1
        args.data_root = args.data_root or "./sampled_data"
        args.seed = 42 if args.seed is None else args.seed
        args.max_attempts = args.max_attempts or 10000
    return args


def main() -> None:
    args = my_parse_args()
    if args.experiment_config:
        _strict_main(args.experiment_config)
    else:
        _legacy_main(args)


if __name__ == "__main__":
    main()

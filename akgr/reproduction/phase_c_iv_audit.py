"""Recompute the hard data, SFT, and evaluation gates for Phase C-IV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from akgr.reproduction.config import ExperimentConfig, load_experiment_config


PAPER_METRICS = ("jaccard", "dice", "overlap", "condition_accuracy", "smatch")
MANIFEST_KEYS = {
    "schema_version", "dataset", "profile", "seed", "data_hash", "kg_hash",
    "stats", "artifacts", "kg", "pattern_table", "augmentation",
}
V2_GREEDY = {
    "jaccard": 0.3299, "dice": 0.3656, "overlap": 0.4244,
    "condition_accuracy": 0.3185, "smatch": 0.6028,
}
V3_GREEDY = {
    "jaccard": 0.2690, "dice": 0.2993, "overlap": 0.3493,
    "condition_accuracy": 0.5883, "smatch": 0.6613,
}
V2_SAMPLED = {
    "jaccard": 0.2419, "dice": 0.2622, "overlap": 0.2930,
    "condition_accuracy": 0.3191, "smatch": 0.5987,
}
V3_SAMPLED = {
    "jaccard": 0.2071, "dice": 0.2263, "overlap": 0.2580,
    "condition_accuracy": 0.5787, "smatch": 0.6478,
}
PAPER_WITHOUT_RL = {
    "jaccard": 0.715, "dice": 0.758, "overlap": 0.837,
    "condition_accuracy": 0.815, "smatch": 0.790,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_artifact(manifest_path: Path, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    path = manifest_path.parent / artifact["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(path)
    if actual_hash != artifact["sha256"]:
        raise ValueError(f"Artifact hash mismatch for {path}: {actual_hash}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != int(artifact["count"]):
        raise ValueError(f"Artifact count mismatch for {path}: {len(records)}")
    return records


def _target_tokens(query: Iterable[Any]) -> tuple[list[int], list[int]]:
    entities: list[int] = []
    relations: list[int] = []
    operator = None
    for token in query:
        if token in {"p", "e"}:
            operator = token
        elif isinstance(token, int):
            (relations if operator == "p" else entities).append(abs(int(token)))
    return entities, relations


def _overlaps(values: dict[str, set[str]]) -> dict[str, int]:
    return {
        f"{left}-{right}": len(values[left] & values[right])
        for left, right in combinations(("train", "valid", "test"), 2)
    }


def _pattern_names() -> dict[str, str]:
    with Path("akgr/metadata/pattern_table.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["original"]: row["pattern_abbr"]
        for row in rows if int(row["original_depth"]) <= 2
    }


def audit_data(config: ExperimentConfig, reference_manifest: Path | None = None) -> dict[str, Any]:
    manifest_path = config.sampling_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    deduplication = config.raw["sampling"].get("deduplication")
    manifest_keys = MANIFEST_KEYS | ({"deduplication"} if deduplication is not None else set())
    if set(manifest) != manifest_keys:
        raise ValueError(f"Manifest schema keys differ: {sorted(set(manifest) ^ manifest_keys)}")
    expected_identity = {
        "schema_version": 3 if deduplication is not None else 2,
        "dataset": config.dataset,
        "profile": config.experiment["profile"],
        "seed": config.seed,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}")
    if deduplication is not None:
        report = manifest["deduplication"]
        if report.get("policy") != deduplication or any(report.get("post_dedup_overlap", {}).values()):
            raise ValueError("Manifest deduplication policy or post-dedup overlap is invalid")
    artifacts = manifest["artifacts"]
    if set(artifacts) != {"base", "augmented_only", "merged"}:
        raise ValueError("Manifest artifact groups differ from the strict schema")
    if set(artifacts["base"]) != {"train", "valid", "test"}:
        raise ValueError("Base artifacts must contain exactly train/valid/test")
    if set(artifacts["augmented_only"]) != {"train"} or set(artifacts["merged"]) != {"train"}:
        raise ValueError("Augmentation and merged artifacts must be train-only")

    base = {split: _load_artifact(manifest_path, artifacts["base"][split]) for split in ("train", "valid", "test")}
    augmented = _load_artifact(manifest_path, artifacts["augmented_only"]["train"])
    merged = _load_artifact(manifest_path, artifacts["merged"]["train"])
    counts_per_pattern = {
        "train": int(config.raw["sampling"]["train_per_pattern"]),
        "valid": int(config.raw["sampling"]["valid_per_pattern"]),
        "test": int(config.raw["sampling"]["test_per_pattern"]),
    }
    names = _pattern_names()
    pattern_counts = {}
    for split, records in base.items():
        counts = Counter(names[record["pattern_str"]] for record in records)
        expected = counts_per_pattern[split]
        if set(counts) != set(names.values()) or any(count != expected for count in counts.values()):
            raise ValueError(f"{split} pattern counts are not exactly {expected}: {dict(counts)}")
        pattern_counts[split] = dict(sorted(counts.items()))

    ids = {split: {str(record["record_id"]) for record in records} for split, records in base.items()}
    if any(len(ids[split]) != len(base[split]) for split in base):
        raise ValueError("Duplicate record_id within a base split")
    supervision = {
        split: {
            _canonical({key: record[key] for key in ("answers", "query", "pattern_str")})
            for record in records
        }
        for split, records in base.items()
    }
    id_overlap = _overlaps(ids)
    supervision_overlap = _overlaps(supervision)
    if any(id_overlap.values()) or any(supervision_overlap.values()):
        raise ValueError(
            f"Cross-split leakage: record_id={id_overlap}, supervision={supervision_overlap}"
        )
    query_only = {
        split: {_canonical(record["query"]) for record in records}
        for split, records in base.items()
    }
    augmented_ids = [str(record["record_id"]) for record in augmented]
    if len(set(augmented_ids)) != len(augmented_ids):
        raise ValueError("Duplicate record_id in augmented-only train")
    if any(record.get("parent_record_id") not in ids["train"] for record in augmented):
        raise ValueError("Augmented record has a parent outside base train")
    merged_ids = [str(record["record_id"]) for record in merged]
    expected_merged_ids = [str(record["record_id"]) for record in base["train"]] + augmented_ids
    if merged_ids != expected_merged_ids or len(set(merged_ids)) != len(merged_ids):
        raise ValueError("Merged train is not the exact unique base+augmentation concatenation")

    token_sets: dict[str, dict[str, set[int]]] = {}
    token_occurrences: dict[str, dict[str, list[int]]] = {}
    for split, records in base.items():
        entity_tokens: list[int] = []
        relation_tokens: list[int] = []
        for record in records:
            entities, relations = _target_tokens(record["query"])
            entity_tokens.extend(entities)
            relation_tokens.extend(relations)
        token_occurrences[split] = {"entity": entity_tokens, "relation": relation_tokens}
        token_sets[split] = {"entity": set(entity_tokens), "relation": set(relation_tokens)}
    coverage = {}
    for kind, denominator_key in (("entity", "nentity"), ("relation", "nrelation")):
        train_tokens = token_sets["train"][kind]
        denominator = int(manifest["stats"][denominator_key])
        kind_report: dict[str, Any] = {
            "train_unique": len(train_tokens),
            "vocabulary_size": denominator,
            "train_coverage_rate": len(train_tokens) / denominator,
        }
        for split in ("valid", "test"):
            split_unique = token_sets[split][kind]
            occurrences = token_occurrences[split][kind]
            unseen_unique = split_unique - train_tokens
            unseen_occurrences = sum(token not in train_tokens for token in occurrences)
            kind_report[split] = {
                "unique": len(split_unique),
                "unseen_unique": len(unseen_unique),
                "unseen_unique_rate": len(unseen_unique) / len(split_unique) if split_unique else 0.0,
                "occurrences": len(occurrences),
                "unseen_occurrences": unseen_occurrences,
                "unseen_occurrence_rate": unseen_occurrences / len(occurrences) if occurrences else 0.0,
            }
        coverage[kind] = kind_report

    reference = None
    if reference_manifest is not None:
        reference_payload = json.loads(reference_manifest.read_text(encoding="utf-8"))
        reference = {}
        for split in ("valid", "test"):
            current = artifacts["base"][split]
            prior = reference_payload["artifacts"]["base"][split]
            reference[split] = {
                "current_sha256": current["sha256"],
                "reference_sha256": prior["sha256"],
                "byte_identical": current["sha256"] == prior["sha256"],
                "count_identical": int(current["count"]) == int(prior["count"]),
            }
            if not reference[split]["byte_identical"] or not reference[split]["count_identical"]:
                raise ValueError(f"{split} differs from the frozen reference manifest")

    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "identity": expected_identity,
        "counts": {
            "base": {split: len(records) for split, records in base.items()},
            "augmented_only_train": len(augmented),
            "merged_train": len(merged),
            "by_pattern": pattern_counts,
        },
        "split_dedup": {
            "record_id_overlap": id_overlap,
            "full_supervision_overlap": supervision_overlap,
            "query_only_overlap_reported_not_gated": _overlaps(query_only),
            "within_split_duplicate_supervision_reported_not_gated": {
                split: len(base[split]) - len(supervision[split]) for split in base
            },
        },
        "target_token_coverage": coverage,
        "frozen_reference": reference,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _paper_average(record: dict[str, Any]) -> float:
    return sum(float(record[name]) for name in PAPER_METRICS) / len(PAPER_METRICS)


def _pareto_frontier(points: list[dict[str, Any]]) -> list[int]:
    frontier = []
    for candidate in points:
        dominated = any(
            all(float(other[name]) >= float(candidate[name]) for name in PAPER_METRICS)
            and any(float(other[name]) > float(candidate[name]) for name in PAPER_METRICS)
            for other in points if other is not candidate
        )
        if not dominated:
            frontier.append(int(candidate["stage_epoch"]))
    return frontier


def audit_training(config: ExperimentConfig, *, pilot: bool) -> dict[str, Any]:
    run_root = config.runtime_paths["run_root"] / config.experiment["name"]
    checkpoint_root = config.runtime_paths["checkpoint_root"] / config.experiment["name"]
    manifest = json.loads(config.sampling_manifest_path.read_text(encoding="utf-8"))
    merged_count = int(manifest["artifacts"]["merged"]["train"]["count"])
    expected_steps = {
        "unconditional": math.ceil(merged_count / int(config.raw["training"]["unconditional"]["micro_batch_size"])),
        "conditional": math.ceil(
            int(manifest["artifacts"]["base"]["train"]["count"])
            / int(config.raw["training"]["conditional"]["micro_batch_size"])
        ),
    }
    if expected_steps["conditional"] != 650:
        raise ValueError(f"Conditional steps must be 650, got {expected_steps['conditional']}")
    stages = {}
    conditional_validations: list[dict[str, Any]] = []
    for stage in ("unconditional", "conditional"):
        history = _read_jsonl(run_root / f"{stage}-history.jsonl")
        expected_epochs = int(config.raw["training"][stage]["epochs"])
        if [row["stage_epoch"] for row in history] != list(range(1, expected_epochs + 1)):
            raise ValueError(f"{stage} history does not contain exact epochs 1..{expected_epochs}")
        if any(not math.isfinite(float(row["train_loss"])) for row in history):
            raise ValueError(f"{stage} contains non-finite train loss")
        if any(int(row["optimizer_steps"]) != expected_steps[stage] for row in history):
            raise ValueError(f"{stage} optimizer-step drift")
        stage_config = config.raw["training"][stage]
        learning_rate = float(stage_config["learning_rate"])
        first_lr = learning_rate * float(stage_config["scheduler"]["start_factor"])
        for index, row in enumerate(history):
            expected_start = first_lr if index == 0 else learning_rate
            if not math.isclose(float(row["learning_rate_start"]), expected_start, abs_tol=1e-12):
                raise ValueError(f"{stage} epoch {index + 1} LR start drift")
            if not math.isclose(float(row["learning_rate_end"]), learning_rate, abs_tol=1e-12):
                raise ValueError(f"{stage} epoch {index + 1} LR end drift")
            schedule = row["optimizer_schedule"]
            if (
                schedule["optimizer"] != "adam"
                or schedule["scheduler"] != "linear_warmup_constant"
                or int(schedule["warmup_steps"]) != 5
                or int(schedule["optimizer_steps_per_epoch"]) != expected_steps[stage]
            ):
                raise ValueError(f"{stage} optimizer schedule drift")
        loss_decrease = 1.0 - float(history[-1]["train_loss"]) / float(history[0]["train_loss"])
        if loss_decrease < 0.25:
            raise ValueError(f"{stage} loss decrease {loss_decrease:.6f} is below 25%")
        pointer_path = checkpoint_root / f"{stage}-best.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        selected = (checkpoint_root / pointer["checkpoint"]).resolve()
        link = (checkpoint_root / f"{stage}-best").resolve()
        if selected != link or not selected.is_dir():
            raise ValueError(f"{stage} best checkpoint pointer/link contract failed")
        metadata = json.loads((selected / "metadata.json").read_text(encoding="utf-8"))
        if metadata["experiment_config_hash"] != config.semantic_hash:
            raise ValueError(f"{stage} checkpoint config hash mismatch")
        if metadata["data_manifest_hash"] != _sha256(config.sampling_manifest_path):
            raise ValueError(f"{stage} checkpoint data hash mismatch")
        if stage == "unconditional" and metadata.get("parent_checkpoint") is not None:
            raise ValueError("Unconditional formal/pilot run was not freshly initialized")
        if stage == "conditional" and Path(metadata["parent_checkpoint"]).resolve() != (checkpoint_root / "unconditional-best").resolve():
            raise ValueError("Conditional parent is not the run's selected unconditional checkpoint")
        validation = pointer["validation"]
        if validation.get("health_pass") is not True:
            raise ValueError(f"{stage} selected checkpoint is unhealthy")
        validations = [row["validation"] for row in history if row.get("validation") is not None]
        if stage == "conditional":
            for record in validations:
                record["paper_average"] = _paper_average(record)
            conditional_validations = validations
        selected_validation = dict(validation)
        if stage == "conditional":
            selected_validation["paper_average"] = _paper_average(validation)
        stages[stage] = {
            "epochs": expected_epochs,
            "optimizer_steps_per_epoch": expected_steps[stage],
            "loss_epoch_1": float(history[0]["train_loss"]),
            "loss_final_epoch": float(history[-1]["train_loss"]),
            "loss_decrease_rate": loss_decrease,
            "selected_checkpoint": str(selected),
            "selected_validation": selected_validation,
            "validation_trajectory": validations,
            "pareto_epochs": _pareto_frontier(validations) if stage == "conditional" else [],
        }
    selected = stages["conditional"]["selected_validation"]
    pilot_decision = None
    if pilot:
        if float(selected["condition_accuracy"]) < 0.45 or float(selected["jaccard"]) < 0.24:
            raise ValueError("Pilot conditional control/Jaccard gate failed")
        averages = [float(record["paper_average"]) for record in conditional_validations]
        last_three_rising = len(averages) >= 3 and averages[-3] < averages[-2] < averages[-1]
        pilot_decision = {
            "baseline_average": 0.4516,
            "selected_meets_average": float(selected["paper_average"]) >= 0.4516,
            "last_three_validation_averages": averages[-3:],
            "last_three_strictly_rising": last_three_rising,
        }
        if not pilot_decision["selected_meets_average"] and not last_three_rising:
            raise ValueError("Pilot paper-average missed v3 without a rising last-three trajectory")
        pilot_decision["status"] = (
            "pass" if pilot_decision["selected_meets_average"] else "requires_pre_test_judgment"
        )
    return {
        "status": "pass" if not pilot_decision or pilot_decision["status"] == "pass" else pilot_decision["status"],
        "experiment": config.experiment["name"],
        "config_semantic_hash": config.semantic_hash,
        "manifest_sha256": _sha256(config.sampling_manifest_path),
        "stages": stages,
        "pilot_decision": pilot_decision,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    result = {
        name: sum(float(record[name]) for record in records) / len(records)
        for name in (*PAPER_METRICS, "parse_ok")
    }
    result["eos_rate"] = sum(float(record["eos_emitted"]) for record in records) / len(records)
    result["max_length_rate"] = sum(float(record["hit_max_new_tokens"]) for record in records) / len(records)
    result["mean_generated_tokens"] = sum(float(record["generated_token_count"]) for record in records) / len(records)
    result["paper_average"] = sum(result[name] for name in PAPER_METRICS) / len(PAPER_METRICS)
    return result


def _csv_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return {row[0]: float(row[1]) for row in rows[1:]}


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"count": len(records), **_aggregate(records)} if records else {"count": 0}


def audit_evaluation(config: ExperimentConfig) -> dict[str, Any]:
    run_root = config.runtime_paths["run_root"] / config.experiment["name"]
    manifest = json.loads(config.sampling_manifest_path.read_text(encoding="utf-8"))
    raw_test = _load_artifact(
        config.sampling_manifest_path, manifest["artifacts"]["base"]["test"]
    )
    pattern_names = _pattern_names()
    reports = {}
    for decode in ("greedy", "sampled"):
        jsonl_path = run_root / f"test-{decode}.jsonl"
        csv_path = run_root / f"test-{decode}.csv"
        records = _read_jsonl(jsonl_path)
        if len(records) != 1664 or len({record["record_id"] for record in records}) != 1664:
            raise ValueError(f"{decode} evaluation must have exactly 1,664 unique record IDs")
        aggregate = _aggregate(records)
        published = _csv_metrics(csv_path)
        for name, value in aggregate.items():
            if name == "paper_average":
                continue
            if not math.isclose(value, published[name], abs_tol=1e-12):
                raise ValueError(f"{decode} CSV mismatch for {name}: {value} != {published[name]}")
        by_pattern: dict[str, list[dict[str, Any]]] = {}
        for index, record in enumerate(records):
            pattern = pattern_names[raw_test[index]["pattern_str"]]
            by_pattern.setdefault(pattern, []).append(record)
        reference_lengths = sorted(len(record["reference"].split()) for record in records)
        long_threshold = reference_lengths[math.ceil(0.75 * len(reference_lengths)) - 1]
        groups = {
            "union": [
                record for index, record in enumerate(records)
                if "(u," in raw_test[index]["pattern_str"]
            ],
            "negation": [
                record for index, record in enumerate(records)
                if "(n," in raw_test[index]["pattern_str"]
            ],
            "long_structure": [record for record in records if len(record["reference"].split()) >= long_threshold],
        }
        reports[decode] = {
            "record_count": len(records),
            "jsonl_sha256": _sha256(jsonl_path),
            "csv_sha256": _sha256(csv_path),
            "aggregate": aggregate,
            "csv_recomputed_max_abs_error": max(
                abs(aggregate[name] - published[name])
                for name in published
            ),
            "deltas": {
                baseline: {name: aggregate[name] - values[name] for name in PAPER_METRICS}
                for baseline, values in (
                    (("v2_greedy", V2_GREEDY) if decode == "greedy" else ("v2_sampled", V2_SAMPLED)),
                    (("v3_greedy", V3_GREEDY) if decode == "greedy" else ("v3_sampled", V3_SAMPLED)),
                    ("paper_without_rl", PAPER_WITHOUT_RL),
                )
            },
            "by_pattern": {
                pattern: _summarize_group(group) for pattern, group in sorted(by_pattern.items())
            },
            "focus_groups": {
                name: _summarize_group(group) for name, group in groups.items()
            },
            "long_structure_reference_token_threshold": long_threshold,
        }
    return {"status": "pass", "experiment": config.experiment["name"], "decoding": reports}


def _write_result(result: dict[str, Any], output: str | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "training", "evaluation"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--experiment-config", required=True)
        subparser.add_argument("--output")
        if name == "data":
            subparser.add_argument("--reference-manifest")
        elif name == "training":
            subparser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.experiment_config)
    if args.command == "data":
        reference = Path(args.reference_manifest).expanduser().resolve() if args.reference_manifest else None
        result = audit_data(config, reference)
    elif args.command == "training":
        result = audit_training(config, pilot=args.pilot)
    else:
        result = audit_evaluation(config)
    _write_result(result, args.output)


if __name__ == "__main__":
    main()

"""Fail-closed Phase 2 data and parent-import contracts for SC-IDC.

This module prepares the artifacts required immediately before the formal
specific-relation conditional SFT.  It deliberately does not start training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

from akgr.dataloader import _load_manifest_artifact
from akgr.reproduction.config import ExperimentConfig, load_experiment_config
from akgr.reproduction.seed import derive_seed
from akgr.tokenizer import (
    create_reproduction_tokenizer,
    unique_condition_values_from_target,
)
from akgr.utils.load_util import (
    _json_sha256,
    _tokenizer_contract,
    load_reproduction_checkpoint,
)
from akgr.utils.parsing_util import qry_shift_indices, qry_str_2_actionstr, list_to_str


SCHEMA_VERSION = 1
CONDITION_KIND = "specific_relation"
PURPOSES = ("validation", "signal", "rl_train", "final_evaluation")
SEALED_PURPOSES = {"final_evaluation"}
CONSUMERS = {
    "validation": {"conditional_sft", "checkpoint_selection", "reporting"},
    "signal": {"signal_audit", "reporting"},
    "rl_train": {"baseline_grpo", "sc_idc_grpo", "reporting"},
    "final_evaluation": {"final_evaluation", "reporting"},
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tree_sha256(path: os.PathLike[str] | str) -> str:
    """Hash names, sizes, and bytes of every regular checkpoint file."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Checkpoint has no files: {root}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        count = 0
        for record in records:
            handle.write(_canonical_json(record) + "\n")
            count += 1
    return {"path": path.name, "count": count, "sha256": _file_sha256(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def _query_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({"query": record["query"], "pattern_str": record["pattern_str"]})


def _supervision_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({
        "answers": record["answers"],
        "query": record["query"],
        "pattern_str": record["pattern_str"],
    })


def _target_from_record(record: Mapping[str, Any]) -> str:
    shifted = list_to_str(qry_shift_indices(list(record["query"])))
    return qry_str_2_actionstr(shifted)


def _topology(value: str, target: str) -> str:
    relations = [token for token in target.split() if token.startswith("-") and token[1:].isdigit()]
    positions = [index for index, token in enumerate(relations) if token == value]
    if not positions:
        raise ValueError(f"Condition value {value!r} is absent from target {target!r}")
    if len(positions) > 1:
        return "repeated"
    return "first_only" if positions[0] == 0 else "non_first_only"


def _select_condition_value(
    record: Mapping[str, Any], *, seed: int, purpose: str
) -> tuple[str, str, int, str]:
    target = _target_from_record(record)
    values = unique_condition_values_from_target(CONDITION_KIND, target)
    selection_seed = derive_seed(
        int(seed), "frozen-condition", purpose, str(record["record_id"]), CONDITION_KIND
    )
    value = values[random.Random(selection_seed).randrange(len(values))]
    return value, target, selection_seed, _topology(value, target)


def build_condition_rows(
    records: Iterable[Mapping[str, Any]], *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown condition-manifest purpose: {purpose}")
    rows = []
    record_ids = set()
    for record in records:
        record_id = str(record["record_id"])
        if record_id in record_ids:
            raise ValueError(f"Duplicate record_id in {purpose}: {record_id}")
        record_ids.add(record_id)
        value, target, selection_seed, topology = _select_condition_value(
            record, seed=seed, purpose=purpose
        )
        rows.append({
            "record_id": record_id,
            "condition_kind": CONDITION_KIND,
            "condition_value": value,
            "selection_seed": selection_seed,
            "topology": topology,
            "pattern_str": str(record["pattern_str"]),
            "query_sha256": _sha256_bytes(_query_signature(record).encode("utf-8")),
            "supervision_sha256": _sha256_bytes(
                _supervision_signature(record).encode("utf-8")
            ),
            "target_sha256": _sha256_bytes(target.encode("utf-8")),
        })
    return sorted(rows, key=lambda row: row["record_id"])


def _require_selected_unconditional_parent(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    pointer_path = checkpoint.parent / "unconditional-best.json"
    if not pointer_path.is_file():
        raise ValueError(f"Missing selected unconditional parent pointer: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    selected = (checkpoint.parent / pointer["checkpoint"]).resolve()
    if selected != checkpoint:
        raise ValueError(
            f"Parent import requires selected unconditional checkpoint {selected}, got {checkpoint}"
        )
    if pointer.get("stage") not in {None, "unconditional"}:
        raise ValueError("Unconditional parent pointer has the wrong stage")
    if pointer.get("validation", {}).get("health_pass") is not True:
        raise ValueError("Selected unconditional parent did not pass health gates")
    return {
        "path": str(pointer_path),
        "sha256": _file_sha256(pointer_path),
        "selection": pointer.get("selection"),
        "validation": pointer.get("validation"),
    }


def create_parent_import_contract(
    *,
    source_config: ExperimentConfig,
    target_config: ExperimentConfig,
    checkpoint: os.PathLike[str] | str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Validate an unconditional parent across condition configs, fail closed."""
    if target_config.condition != CONDITION_KIND:
        raise ValueError("SC-IDC Phase 2 target config must use specific_relation")
    if source_config.condition == target_config.condition:
        raise ValueError("Parent import must cross from a distinct source condition config")
    if source_config.data_hash != target_config.data_hash:
        raise ValueError("Source and target data hashes differ")
    if source_config.kg_hash != target_config.kg_hash:
        raise ValueError("Source and target KG hashes differ")
    if source_config.raw["model"] != target_config.raw["model"]:
        raise ValueError("Source and target model architecture configs differ")
    if source_config.raw["tokenizer"] != target_config.raw["tokenizer"]:
        raise ValueError("Source and target tokenizer configs differ")
    source_sampling = source_config.sampling_manifest_path
    target_sampling = target_config.sampling_manifest_path
    if source_sampling.resolve() != target_sampling.resolve():
        raise ValueError("Source and target configs do not resolve to the same sampled data")
    sampling_sha = _file_sha256(source_sampling)
    pointer = _require_selected_unconditional_parent(Path(checkpoint))
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    loaded = load_reproduction_checkpoint(
        checkpoint_path,
        mode="parent",
        expected_stage="unconditional",
        expected_condition="unconditional",
        expected_config_hash=source_config.semantic_hash,
        expected_data_manifest_hash=sampling_sha,
    )
    metadata = loaded.metadata
    if metadata.get("experiment_config") != source_config.raw:
        raise ValueError("Checkpoint embedded source config does not match source config")
    sampling_manifest = json.loads(source_sampling.read_text(encoding="utf-8"))
    stats = sampling_manifest.get("stats", {})
    expected_tokenizer = create_reproduction_tokenizer(
        int(stats["nentity"]), int(stats["nrelation"])
    )
    expected_vocab_hash = _json_sha256(expected_tokenizer.get_vocab())
    if metadata.get("tokenizer_vocab_hash") != expected_vocab_hash:
        raise ValueError("Checkpoint tokenizer vocabulary is incompatible with target data")
    if metadata.get("tokenizer_contract") != _tokenizer_contract(expected_tokenizer):
        raise ValueError("Checkpoint tokenizer contract is incompatible with target config")
    expected_model_fields = {
        "vocab_size": len(expected_tokenizer),
        "n_layer": int(target_config.raw["model"]["n_layer"]),
        "n_embd": int(target_config.raw["model"]["n_embd"]),
        "n_head": int(target_config.raw["model"]["n_head"]),
        "n_positions": int(target_config.raw["model"]["n_positions"]),
        "n_ctx": int(target_config.raw["model"]["n_ctx"]),
        "tie_word_embeddings": bool(target_config.raw["model"]["tie_word_embeddings"]),
        "pad_token_id": expected_tokenizer.pad_token_id,
        "bos_token_id": expected_tokenizer.bos_token_id,
        "eos_token_id": expected_tokenizer.eos_token_id,
    }
    for key, value in expected_model_fields.items():
        if getattr(loaded.model.config, key) != value:
            raise ValueError(f"Checkpoint model field {key} is incompatible with target config")
    target_model_hash = _json_sha256(target_config.raw["model"])
    source_model_hash = _json_sha256(source_config.raw["model"])
    if source_model_hash != target_model_hash:
        raise ValueError("Source and target model identity hashes differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_import_unconditional_parent",
        "source": {
            "config_path": str(source_config.source_path),
            "config_file_sha256": _file_sha256(source_config.source_path),
            "config_semantic_hash": source_config.semantic_hash,
            "data_hash": source_config.data_hash,
            "kg_hash": source_config.kg_hash,
            "stage": metadata["stage"],
            "condition": metadata["condition"],
            "sampling_manifest_path": str(source_sampling),
            "sampling_manifest_sha256": sampling_sha,
            "selected_checkpoint_pointer": pointer,
        },
        "target": {
            "config_path": str(target_config.source_path),
            "config_file_sha256": _file_sha256(target_config.source_path),
            "config_semantic_hash": target_config.semantic_hash,
            "data_hash": target_config.data_hash,
            "kg_hash": target_config.kg_hash,
            "condition": target_config.condition,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "tree_sha256": checkpoint_tree_sha256(checkpoint_path),
            "metadata_sha256": _file_sha256(checkpoint_path / "metadata.json"),
            "model_config_hash": metadata["model_config_hash"],
            "tokenizer_vocab_hash": metadata["tokenizer_vocab_hash"],
            "tokenizer_json_hash": metadata["tokenizer_json_hash"],
        },
        "identity": {
            "source_model_config_sha256": source_model_hash,
            "target_model_config_sha256": target_model_hash,
            "target_tokenizer_vocab_hash": expected_vocab_hash,
            "target_tokenizer_contract": _tokenizer_contract(expected_tokenizer),
            "target_model_fields": expected_model_fields,
        },
        "transition": {
            "weights_only": True,
            "optimizer_reset": True,
            "scheduler_reset": True,
            "rng_not_restored": True,
        },
        "command": list(command or []),
    }


def verify_parent_import_contract(
    payload: Mapping[str, Any], *, target_config: ExperimentConfig, checkpoint: Path
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported parent-import schema")
    if payload.get("kind") != "sc_idc_import_unconditional_parent":
        raise ValueError("Not an SC-IDC parent-import contract")
    target = payload["target"]
    expected = {
        "config_semantic_hash": target_config.semantic_hash,
        "data_hash": target_config.data_hash,
        "kg_hash": target_config.kg_hash,
        "condition": target_config.condition,
    }
    for key, value in expected.items():
        if target.get(key) != value:
            raise ValueError(f"Parent-import target {key} mismatch")
    checkpoint = checkpoint.expanduser().resolve()
    if Path(payload["checkpoint"]["path"]).expanduser().resolve() != checkpoint:
        raise ValueError("Parent-import checkpoint path mismatch")
    if checkpoint_tree_sha256(checkpoint) != payload["checkpoint"]["tree_sha256"]:
        raise ValueError("Parent-import checkpoint tree hash mismatch")
    if _file_sha256(checkpoint / "metadata.json") != payload["checkpoint"]["metadata_sha256"]:
        raise ValueError("Parent-import checkpoint metadata hash mismatch")
    if payload.get("transition") != {
        "weights_only": True,
        "optimizer_reset": True,
        "scheduler_reset": True,
        "rng_not_restored": True,
    }:
        raise ValueError("Parent-import optimizer/scheduler reset contract is invalid")


def _fresh_records(
    fresh_manifest_path: Path, *, source_config: ExperimentConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fresh_manifest = json.loads(fresh_manifest_path.read_text(encoding="utf-8"))
    if fresh_manifest.get("config_semantic_hash") != source_config.semantic_hash:
        raise ValueError("Fresh-query source config semantic hash mismatch")
    sampling_sha = _file_sha256(source_config.sampling_manifest_path)
    if fresh_manifest.get("original_sampling_manifest_sha256") != sampling_sha:
        raise ValueError("Fresh-query source sampling manifest hash mismatch")
    artifact = fresh_manifest.get("artifact", {})
    artifact_path = (fresh_manifest_path.parent / artifact.get("path", "")).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if _file_sha256(artifact_path) != artifact.get("sha256"):
        raise ValueError("Fresh-query artifact hash mismatch")
    records = _read_jsonl(artifact_path)
    if len(records) != artifact.get("count"):
        raise ValueError("Fresh-query artifact count mismatch")
    if fresh_manifest.get("exclusion", {}).get("test_artifact_opened") is not False:
        raise ValueError("Fresh-query source does not prove test isolation")
    return records, {
        "manifest_path": str(fresh_manifest_path),
        "manifest_sha256": _file_sha256(fresh_manifest_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact["sha256"],
        "artifact_count": artifact["count"],
        "source_sampling_manifest_sha256": sampling_sha,
    }


def _load_sampling_inputs(config: ExperimentConfig) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    path = config.sampling_manifest_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": config.dataset,
        "profile": config.experiment["profile"],
        "seed": config.seed,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Sampling manifest {key} mismatch")
    artifacts = manifest["artifacts"]
    conditional_variant = config.raw["training"]["conditional"]["data_variant"]
    records = {
        "sft_train": _load_manifest_artifact(
            path, artifacts[conditional_variant]["train"]
        ),
        "merged_train": _load_manifest_artifact(path, artifacts["merged"]["train"]),
        "validation": _load_manifest_artifact(path, artifacts["base"]["valid"]),
        "final_evaluation": _load_manifest_artifact(path, artifacts["base"]["test"]),
    }
    return manifest, records


def _require_unique(
    records: Iterable[Mapping[str, Any]],
    *,
    label: str,
    require_query_and_supervision_unique: bool = False,
) -> dict[str, Any]:
    record_ids = [str(record["record_id"]) for record in records]
    queries = [_query_signature(record) for record in records]
    supervision = [_supervision_signature(record) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{label} has duplicate record IDs")
    query_duplicates = len(queries) - len(set(queries))
    supervision_duplicates = len(supervision) - len(set(supervision))
    if require_query_and_supervision_unique and query_duplicates:
        raise ValueError(f"{label} has duplicate queries")
    if require_query_and_supervision_unique and supervision_duplicates:
        raise ValueError(f"{label} has duplicate supervision")
    return {
        "record_id": set(record_ids),
        "query": set(queries),
        "supervision": set(supervision),
        "audit": {
            "count": len(record_ids),
            "record_id_duplicates": 0,
            "query_duplicates": query_duplicates,
            "supervision_duplicates": supervision_duplicates,
        },
    }


def _partition_fresh(
    records: list[dict[str, Any]], *, seed: int, signal_per_pattern: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pattern_str"])].append(record)
    signal = []
    rl_train = []
    counts = {}
    for pattern in sorted(grouped):
        ranked = sorted(
            grouped[pattern],
            key=lambda record: (
                derive_seed(seed, "fresh-partition", pattern, str(record["record_id"])),
                str(record["record_id"]),
            ),
        )
        if len(ranked) <= int(signal_per_pattern):
            raise ValueError(
                f"Pattern {pattern} has {len(ranked)} fresh records; need more than "
                f"signal_per_pattern={signal_per_pattern}"
            )
        signal.extend(ranked[: int(signal_per_pattern)])
        rl_train.extend(ranked[int(signal_per_pattern) :])
        counts[pattern] = {
            "source": len(ranked),
            "signal": int(signal_per_pattern),
            "rl_train": len(ranked) - int(signal_per_pattern),
        }
    return signal, rl_train, counts


def _condition_manifest(
    *,
    output_dir: Path,
    purpose: str,
    rows: list[dict[str, Any]],
    target_config: ExperimentConfig,
    source_reference: Mapping[str, Any],
) -> dict[str, Any]:
    rows_path = output_dir / f"{purpose}.conditions.jsonl"
    artifact = _write_jsonl(rows_path, rows)
    topology = Counter(row["topology"] for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_frozen_condition_manifest",
        "purpose": purpose,
        "condition_kind": CONDITION_KIND,
        "sealed": purpose in SEALED_PURPOSES,
        "allowed_consumers": sorted(CONSUMERS[purpose]),
        "target": {
            "config_semantic_hash": target_config.semantic_hash,
            "data_hash": target_config.data_hash,
            "kg_hash": target_config.kg_hash,
        },
        "selection": {
            "unit": "unique_relation_value",
            "distribution": "uniform",
            "seed": target_config.seed,
            "seed_components": [
                "seed", "frozen-condition", "purpose", "record_id", "condition_kind"
            ],
        },
        "source": dict(source_reference),
        "artifact": artifact,
        "topology_counts": dict(sorted(topology.items())),
    }
    manifest_path = output_dir / f"{purpose}.manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path.name,
        "manifest_sha256": _file_sha256(manifest_path),
        "artifact_path": artifact["path"],
        "artifact_sha256": artifact["sha256"],
        "count": artifact["count"],
        "sealed": manifest["sealed"],
    }


def load_frozen_condition_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    target_config: ExperimentConfig,
    purpose: str,
    consumer: str,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    if expected_manifest_sha256 and _file_sha256(path) != expected_manifest_sha256:
        raise ValueError(f"Frozen {purpose} condition manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported condition-manifest schema")
    if manifest.get("kind") != "sc_idc_frozen_condition_manifest":
        raise ValueError("Not an SC-IDC condition manifest")
    if manifest.get("purpose") != purpose:
        raise ValueError(f"Condition manifest purpose is not {purpose}")
    if consumer not in set(manifest.get("allowed_consumers", [])):
        raise PermissionError(
            f"Consumer {consumer!r} cannot load {purpose} conditions"
        )
    target = manifest["target"]
    for key, value in {
        "config_semantic_hash": target_config.semantic_hash,
        "data_hash": target_config.data_hash,
        "kg_hash": target_config.kg_hash,
    }.items():
        if target.get(key) != value:
            raise ValueError(f"Condition manifest target {key} mismatch")
    artifact = manifest["artifact"]
    artifact_path = (path.parent / artifact["path"]).resolve()
    if _file_sha256(artifact_path) != artifact["sha256"]:
        raise ValueError(f"Frozen {purpose} condition artifact hash mismatch")
    rows = _read_jsonl(artifact_path)
    if len(rows) != artifact["count"]:
        raise ValueError(f"Frozen {purpose} condition artifact count mismatch")
    mapping = {}
    for row in rows:
        if row.get("condition_kind") != CONDITION_KIND:
            raise ValueError("Frozen condition row has the wrong kind")
        record_id = str(row["record_id"])
        if record_id in mapping:
            raise ValueError(f"Duplicate frozen condition record ID: {record_id}")
        mapping[record_id] = str(row["condition_value"])
    return mapping, manifest


def require_shared_rl_manifest_hash(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    first_hash = first["conditions"]["rl_train"]["manifest_sha256"]
    second_hash = second["conditions"]["rl_train"]["manifest_sha256"]
    if first_hash != second_hash:
        raise ValueError("Paired GRPO branches must share the same RL condition manifest hash")
    return str(first_hash)


def verify_frozen_validation_dataset(
    dataset,
    *,
    condition_values: Mapping[str, str],
    record_contracts: Mapping[str, Mapping[str, str]],
) -> None:
    dataset_ids = {str(value) for value in dataset["record_id"]}
    manifest_ids = set(condition_values)
    if dataset_ids != manifest_ids or dataset_ids != set(record_contracts):
        raise ValueError("Frozen validation condition record IDs do not match validation data")
    for example in dataset:
        record_id = str(example["record_id"])
        contract = record_contracts[record_id]
        target_hash = hashlib.sha256(example["target"].encode("utf-8")).hexdigest()
        if target_hash != contract["target_sha256"]:
            raise ValueError(f"Frozen validation target hash mismatch for {record_id}")
        if str(condition_values[record_id]) != str(contract["condition_value"]):
            raise ValueError(f"Frozen validation condition mismatch for {record_id}")


def prepare_phase2_data(args: argparse.Namespace) -> Path:
    source_config = load_experiment_config(args.source_config)
    target_config = load_experiment_config(args.target_config)
    if source_config.data_hash != target_config.data_hash or source_config.kg_hash != target_config.kg_hash:
        raise ValueError("Source and target configs must share data/KG identity")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        parent_contract = create_parent_import_contract(
            source_config=source_config,
            target_config=target_config,
            checkpoint=args.parent_checkpoint,
            command=[sys.executable, "-m", __name__, *sys.argv[1:]],
        )
        parent_path = output_dir / "parent-import.json"
        _write_json(parent_path, parent_contract)

        sampling_manifest, sampled = _load_sampling_inputs(target_config)
        fresh, fresh_reference = _fresh_records(
            Path(args.fresh_manifest).expanduser().resolve(), source_config=source_config
        )
        final_queries = {_query_signature(record) for record in sampled["final_evaluation"]}
        excluded_final_query_overlap = [
            record for record in fresh if _query_signature(record) in final_queries
        ]
        fresh = [record for record in fresh if _query_signature(record) not in final_queries]
        identities = {}
        for label, records in {
            "conditional_sft_train": sampled["sft_train"],
            "validation": sampled["validation"],
            "final_evaluation": sampled["final_evaluation"],
            "fresh": fresh,
        }.items():
            identities[label] = _require_unique(
                records,
                label=label,
                require_query_and_supervision_unique=(label == "fresh"),
            )
        for left, right in (
            ("conditional_sft_train", "validation"),
            ("conditional_sft_train", "final_evaluation"),
            ("validation", "final_evaluation"),
            ("fresh", "conditional_sft_train"),
            ("fresh", "validation"),
            ("fresh", "final_evaluation"),
        ):
            record_overlap = identities[left]["record_id"] & identities[right]["record_id"]
            if record_overlap:
                raise ValueError(f"Record ID overlap between {left} and {right}")
            supervision_overlap = (
                identities[left]["supervision"] & identities[right]["supervision"]
            )
            if supervision_overlap:
                raise ValueError(f"Supervision identity overlap between {left} and {right}")
            if "fresh" in {left, right}:
                query_overlap = identities[left]["query"] & identities[right]["query"]
                if query_overlap:
                    raise ValueError(f"Query identity overlap between {left} and {right}")

        signal, rl_train, partition_counts = _partition_fresh(
            fresh, seed=target_config.seed, signal_per_pattern=args.signal_per_pattern
        )
        condition_sources = {
            "validation": {
                "kind": "sampling_manifest_artifact",
                "sampling_manifest_sha256": _file_sha256(target_config.sampling_manifest_path),
                "artifact": dict(sampling_manifest["artifacts"]["base"]["valid"]),
                "split": "valid",
            },
            "signal": {
                "kind": "condition_agnostic_fresh_query_rebind",
                **fresh_reference,
                "excluded_final_evaluation_query_overlap_count": len(
                    excluded_final_query_overlap
                ),
                "subset": "signal",
            },
            "rl_train": {
                "kind": "condition_agnostic_fresh_query_rebind",
                **fresh_reference,
                "excluded_final_evaluation_query_overlap_count": len(
                    excluded_final_query_overlap
                ),
                "subset": "rl_train",
            },
            "final_evaluation": {
                "kind": "sealed_sampling_manifest_artifact",
                "sampling_manifest_sha256": _file_sha256(target_config.sampling_manifest_path),
                "artifact": dict(sampling_manifest["artifacts"]["base"]["test"]),
                "split": "test",
            },
        }
        condition_records = {
            "validation": sampled["validation"],
            "signal": signal,
            "rl_train": rl_train,
            "final_evaluation": sampled["final_evaluation"],
        }
        conditions = {}
        for purpose in PURPOSES:
            rows = build_condition_rows(
                condition_records[purpose], seed=target_config.seed, purpose=purpose
            )
            conditions[purpose] = _condition_manifest(
                output_dir=output_dir,
                purpose=purpose,
                rows=rows,
                target_config=target_config,
                source_reference=condition_sources[purpose],
            )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sc_idc_specific_relation_sft_preflight",
            "status": "ready",
            "target": {
                "config_path": str(target_config.source_path),
                "config_file_sha256": _file_sha256(target_config.source_path),
                "config_semantic_hash": target_config.semantic_hash,
                "data_hash": target_config.data_hash,
                "kg_hash": target_config.kg_hash,
                "condition": target_config.condition,
            },
            "source": {
                "config_path": str(source_config.source_path),
                "config_file_sha256": _file_sha256(source_config.source_path),
                "config_semantic_hash": source_config.semantic_hash,
                "sampling_manifest_sha256": _file_sha256(source_config.sampling_manifest_path),
            },
            "parent_import": {
                "path": parent_path.name,
                "sha256": _file_sha256(parent_path),
                "checkpoint_tree_sha256": parent_contract["checkpoint"]["tree_sha256"],
            },
            "train_sampler": {
                "kind": "dynamic_uniform_unique_relation_value",
                "condition_kind": CONDITION_KIND,
                "seed": target_config.seed,
                "seed_components": ["seed", "record_id", "epoch", "condition_kind"],
                "data_variant": target_config.raw["training"]["conditional"]["data_variant"],
                "occurrence_weighting": False,
            },
            "conditions": conditions,
            "isolation": {
                "query_overlap_all_required_pairs": 0,
                "supervision_overlap_all_required_pairs": 0,
                "record_id_overlap_all_required_pairs": 0,
                "fresh_internal_query_duplicates": 0,
                "fresh_internal_supervision_duplicates": 0,
                "fresh_excluded_for_final_evaluation_query_overlap": len(
                    excluded_final_query_overlap
                ),
                "source_internal_identity_audit": {
                    label: inventory["audit"]
                    for label, inventory in sorted(identities.items())
                },
                "signal_rl_query_overlap": 0,
                "signal_per_pattern": int(args.signal_per_pattern),
                "fresh_partition_counts": partition_counts,
                "final_evaluation_sealed": True,
                "conditional_sft_loads_final_evaluation_manifest": False,
            },
            "checks": {
                "parent_source_target_identity": True,
                "checkpoint_tree_hash_frozen": True,
                "optimizer_scheduler_reset": True,
                "fresh_query_rebind_verified": True,
                "unique_value_conditions": True,
                "fixed_seed_byte_stable_format": True,
                "paired_grpo_rl_manifest_single_hash": True,
                "split_isolation": True,
            },
        }
        summary_path = output_dir / "sft-preflight.json"
        _write_json(summary_path, summary)
        verify_sft_preflight(
            summary_path,
            target_config=target_config,
            checkpoint=Path(args.parent_checkpoint),
        )
        return summary_path
    except BaseException:
        # Preserve a small failure marker without pretending a partial directory is ready.
        failure_path = output_dir / "FAILED"
        failure_path.write_text("SC-IDC Phase 2 data preflight did not complete.\n", encoding="utf-8")
        raise


def verify_sft_preflight(
    path: os.PathLike[str] | str,
    *,
    target_config: ExperimentConfig,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, dict[str, str]]]:
    """Verify only SFT-authorized inputs; never open the sealed final manifest."""
    summary_path = Path(path).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported SFT preflight schema")
    if summary.get("kind") != "sc_idc_specific_relation_sft_preflight":
        raise ValueError("Not an SC-IDC specific-relation SFT preflight")
    if summary.get("status") != "ready":
        raise ValueError("SC-IDC specific-relation SFT preflight is not ready")
    for key, value in {
        "config_semantic_hash": target_config.semantic_hash,
        "data_hash": target_config.data_hash,
        "kg_hash": target_config.kg_hash,
        "condition": target_config.condition,
    }.items():
        if summary["target"].get(key) != value:
            raise ValueError(f"SFT preflight target {key} mismatch")
    parent_ref = summary["parent_import"]
    parent_path = (summary_path.parent / parent_ref["path"]).resolve()
    if _file_sha256(parent_path) != parent_ref["sha256"]:
        raise ValueError("Parent-import artifact hash mismatch")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    verify_parent_import_contract(
        parent_payload, target_config=target_config, checkpoint=checkpoint
    )
    validation_ref = summary["conditions"]["validation"]
    validation_path = (summary_path.parent / validation_ref["manifest_path"]).resolve()
    validation_map, validation_manifest = load_frozen_condition_manifest(
        validation_path,
        target_config=target_config,
        purpose="validation",
        consumer="conditional_sft",
        expected_manifest_sha256=validation_ref["manifest_sha256"],
    )
    if len(validation_map) != validation_ref["count"]:
        raise ValueError("SFT validation condition count mismatch")
    # Intentionally do not open conditions.final_evaluation here.
    validation_artifact = validation_manifest["artifact"]
    validation_rows = _read_jsonl(
        (validation_path.parent / validation_artifact["path"]).resolve()
    )
    validation_contracts = {
        str(row["record_id"]): {
            "condition_value": str(row["condition_value"]),
            "target_sha256": str(row["target_sha256"]),
        }
        for row in validation_rows
    }
    return summary, validation_map, {
        "preflight_path": str(summary_path),
        "preflight_sha256": _file_sha256(summary_path),
        "parent_import_path": str(parent_path),
        "parent_import_sha256": parent_ref["sha256"],
        "parent_checkpoint_tree_sha256": parent_ref["checkpoint_tree_sha256"],
        "validation_manifest_path": str(validation_path),
        "validation_manifest_sha256": validation_ref["manifest_sha256"],
        "validation_conditions_sha256": validation_manifest["artifact"]["sha256"],
        "train_sampler": summary["train_sampler"],
        "final_evaluation_manifest_loaded": False,
    }, validation_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--fresh-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signal-per-pattern", type=int, default=16)
    args = parser.parse_args()
    if args.signal_per_pattern <= 0:
        parser.error("--signal-per-pattern must be positive")
    output = prepare_phase2_data(args)
    print(output)


if __name__ == "__main__":
    main()

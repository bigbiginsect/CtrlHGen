"""Run Experiment 1: train-only SC-IDC rollout and offline reward audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.sc_idc import (
    action_string,
    canonical_query_sha256,
    parse_action,
    parse_raw_query,
    pattern_signature,
    slot_counts,
    value_occurrences,
)
from akgr.reproduction.sc_idc_phase2_data import (
    checkpoint_tree_sha256,
    load_frozen_condition_manifest,
)
from akgr.reproduction.sc_idc_phase2_reward import (
    CachedQueryExecutor,
    StaticRelationMatcher,
    audit_relation_value,
    set_semantic_scores,
)
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.tokenizer import build_generation_prompt
from akgr.utils.load_util import load_reproduction_checkpoint
from akgr.utils.parsing_util import ans_shift_indices, list_to_str


SCHEMA_VERSION = 1
ALPHA_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
BOOTSTRAP_SAMPLES = 10_000
RESOURCE_BUDGET = {
    "accelerator": "one NVIDIA L20",
    "maximum_wall_time_seconds": 3600,
    "prompt_groups": 208,
    "generations_per_group": 4,
    "matched_replacements_per_scorable_completion": 3,
    "maximum_graph_executions_per_completion": 5,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(_canonical_json(value) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _query_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({"query": record["query"], "pattern_str": record["pattern_str"]})


def _supervision_signature(record: Mapping[str, Any]) -> str:
    return _canonical_json({
        "answers": record["answers"], "query": record["query"],
        "pattern_str": record["pattern_str"],
    })


def _target_action(record: Mapping[str, Any]) -> str:
    return action_string(parse_raw_query(record["query"]))


def _load_signal_contract(
    *, config, preflight_path: Path, expected_signal_manifest_sha256: str | None
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("kind") != "sc_idc_specific_relation_sft_preflight":
        raise ValueError("Not an SC-IDC Phase 2 preflight")
    target = preflight["target"]
    for key, expected in {
        "config_semantic_hash": config.semantic_hash,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "condition": "specific_relation",
    }.items():
        if target.get(key) != expected:
            raise ValueError(f"Phase 2 preflight target {key} mismatch")
    isolation = preflight.get("isolation", {})
    required_zeroes = (
        "query_overlap_all_required_pairs", "supervision_overlap_all_required_pairs",
        "record_id_overlap_all_required_pairs", "signal_rl_query_overlap",
        "fresh_internal_query_duplicates", "fresh_internal_supervision_duplicates",
    )
    if any(int(isolation.get(key, -1)) != 0 for key in required_zeroes):
        raise ValueError("Phase 2 preflight does not prove train-only signal isolation")
    if isolation.get("final_evaluation_sealed") is not True:
        raise ValueError("Final evaluation is not sealed")
    signal_ref = preflight["conditions"]["signal"]
    expected_hash = expected_signal_manifest_sha256 or signal_ref["manifest_sha256"]
    if expected_hash != signal_ref["manifest_sha256"]:
        raise ValueError("Requested signal manifest differs from the frozen preflight")
    manifest_path = (preflight_path.parent / signal_ref["manifest_path"]).resolve()
    conditions, signal_manifest = load_frozen_condition_manifest(
        manifest_path, target_config=config, purpose="signal", consumer="signal_audit",
        expected_manifest_sha256=expected_hash,
    )
    source = signal_manifest["source"]
    if source.get("kind") != "condition_agnostic_fresh_query_rebind":
        raise ValueError("Signal conditions are not derived from fresh train-only queries")
    fresh_manifest_path = Path(source["manifest_path"]).expanduser().resolve()
    if _file_sha256(fresh_manifest_path) != source["manifest_sha256"]:
        raise ValueError("Fresh source manifest hash mismatch")
    fresh_manifest = json.loads(fresh_manifest_path.read_text(encoding="utf-8"))
    if fresh_manifest.get("exclusion", {}).get("test_artifact_opened") is not False:
        raise ValueError("Fresh source does not prove sealed-test isolation")
    data_path = Path(source["artifact_path"]).expanduser().resolve()
    if _file_sha256(data_path) != source["artifact_sha256"]:
        raise ValueError("Fresh source artifact hash mismatch")
    all_rows = _read_jsonl(data_path)
    if len(all_rows) != int(source["artifact_count"]):
        raise ValueError("Fresh source artifact count mismatch")
    by_id = {str(row["record_id"]): row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("Fresh source contains duplicate record IDs")
    missing = sorted(set(conditions) - set(by_id))
    if missing:
        raise ValueError(f"Signal source is missing record IDs: {missing[:3]}")
    condition_rows_path = manifest_path.parent / signal_manifest["artifact"]["path"]
    condition_rows = {str(row["record_id"]): row for row in _read_jsonl(condition_rows_path)}
    selected = []
    for record_id in sorted(conditions):
        record = by_id[record_id]
        contract = condition_rows[record_id]
        if _sha256_bytes(_query_signature(record).encode("utf-8")) != contract["query_sha256"]:
            raise ValueError(f"Signal query identity mismatch for {record_id}")
        if _sha256_bytes(_supervision_signature(record).encode("utf-8")) != contract["supervision_sha256"]:
            raise ValueError(f"Signal supervision identity mismatch for {record_id}")
        target_hash = _sha256_bytes(_target_action(record).encode("utf-8"))
        if target_hash != contract["target_sha256"]:
            raise ValueError(f"Signal target identity mismatch for {record_id}")
        selected.append({**record, "condition_value": conditions[record_id], "topology": contract["topology"]})
    pattern_counts = Counter(str(row["pattern_str"]) for row in selected)
    if len(selected) != RESOURCE_BUDGET["prompt_groups"] or set(pattern_counts.values()) != {16}:
        raise ValueError(f"Signal set is not the frozen 13 x 16 pilot: {pattern_counts}")
    lineage = {
        "preflight_path": str(preflight_path),
        "preflight_sha256": _file_sha256(preflight_path),
        "signal_manifest_path": str(manifest_path),
        "signal_manifest_sha256": expected_hash,
        "signal_conditions_sha256": signal_manifest["artifact"]["sha256"],
        "fresh_manifest_path": str(fresh_manifest_path),
        "fresh_manifest_sha256": source["manifest_sha256"],
        "fresh_artifact_path": str(data_path),
        "fresh_artifact_sha256": source["artifact_sha256"],
        "rl_train_manifest_sha256": preflight["conditions"]["rl_train"]["manifest_sha256"],
        "final_evaluation_manifest_loaded": False,
    }
    return selected, conditions, lineage


def _graph_sampler(config):
    from akgr.kgdata import load_kg
    data = config.raw["data"]
    return load_kg(
        config.dataset, data_root=config.runtime_paths["data_root"], seed=config.seed,
        split_ratios=data["split_ratios"], reverse_edges_flag=data["reverse_edges"],
        semantic_hash=config.kg_hash, offline=True,
    ).graph_samplers["train"]


def _reference_preflight(
    records: Sequence[Mapping[str, Any]], graph_sampler, *, reward_seed: int
) -> dict[str, Any]:
    matcher = StaticRelationMatcher(graph_sampler)
    cache = CachedQueryExecutor(graph_sampler)
    rows = []
    for record in records:
        query = parse_raw_query(record["query"])
        condition = int(record["condition_value"])
        if not value_occurrences(query, "specific_relation", condition):
            raise AssertionError("Frozen signal condition is absent from its reference")
        base, _ = cache.execute(query)
        row = audit_relation_value(
            query=query, observation=record["answers"], condition_value=condition,
            matcher=matcher, cache=cache, record_id=str(record["record_id"]),
            reward_seed=reward_seed, base_denotation=base,
        )
        row["reference_exact"] = base == frozenset(int(value) for value in record["answers"])
        rows.append(row)
    gates = {
        "all_reference_exact": all(row["reference_exact"] for row in rows),
        "all_replacements_structure_preserved": all(
            candidate["structure_preserved"] for row in rows for candidate in row["replacements"]
        ),
        "all_scorable": all(row["classification"] != "unscorable" for row in rows),
        "all_joint_occurrences_accounted": all(row["occurrence_count"] >= 1 for row in rows),
        "no_predicate_necessity_claim": all(
            row["terminology"]["predicate_necessity_claim"] is False for row in rows
        ),
    }
    return {
        "row_count": len(rows), "gates": gates, "pass": all(gates.values()),
        "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
        "graph_executions": cache.executions,
    }


def _load_parent(config, checkpoint: Path):
    pointer = checkpoint.parent / "conditional-best.json"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    selection = json.loads(pointer.read_text(encoding="utf-8"))
    selected = (checkpoint.parent / selection["checkpoint"]).resolve()
    if selected != checkpoint.resolve():
        raise ValueError(f"Experiment 1 requires selected conditional parent {selected}")
    if selection.get("validation", {}).get("health_pass") is not True:
        raise ValueError("Selected conditional parent did not pass health gates")
    loaded = load_reproduction_checkpoint(
        checkpoint, mode="parent", expected_stage="conditional",
        expected_condition="specific_relation", expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    return loaded, selection


def _rollout(
    *, records: Sequence[Mapping[str, Any]], config, model, tokenizer, device
) -> list[dict[str, Any]]:
    import torch
    group_size = RESOURCE_BUDGET["generations_per_group"]
    prompts = []
    for record in records:
        source = list_to_str(ans_shift_indices(record["answers"]))
        prompt = build_generation_prompt(
            source, ConditionSpec("specific_relation", str(record["condition_value"])), tokenizer,
            condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
        )
        prompts.append(prompt)
    batch_size = 32
    generation = config.raw["generation"]
    rows = []
    model.eval()
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        seed_everything(derive_seed(config.seed, "sc-idc-experiment-1-rollout"))
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                prompt_batch = prompts[start:start + batch_size]
                encoded = tokenizer(
                    prompt_batch, padding="longest", add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                    do_sample=True, top_k=int(generation["top_k"]),
                    top_p=float(generation["top_p"]), temperature=float(generation["temperature"]),
                    max_new_tokens=int(generation["max_new_tokens"]),
                    num_return_sequences=group_size, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion_ids = generated[:, encoded.input_ids.shape[1]:]
                decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                for offset, record in enumerate(records[start:start + batch_size]):
                    for generation_index in range(group_size):
                        flat = offset * group_size + generation_index
                        token_ids = completion_ids[flat].tolist()
                        eos = tokenizer.eos_token_id in token_ids
                        rows.append({
                            "group_index": start + offset,
                            "generation_index": generation_index,
                            "record_id": str(record["record_id"]),
                            "pattern_str": str(record["pattern_str"]),
                            "topology": str(record["topology"]),
                            "condition_value": str(record["condition_value"]),
                            "observation": [int(value) for value in record["answers"]],
                            "reference": _target_action(record),
                            "prompt": prompt_batch[offset],
                            "completion": decoded[flat],
                            "eos_emitted": eos,
                            "generated_token_count": (
                                token_ids.index(tokenizer.eos_token_id) + 1 if eos else len(token_ids)
                            ),
                        })
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    if len(rows) != len(records) * group_size:
        raise AssertionError("Rollout count mismatch")
    return rows


def _deterministic_view(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("reward_wall_time_seconds", None)
    if isinstance(result.get("sc_idc"), dict):
        sc_idc = dict(result["sc_idc"])
        sc_idc.pop("wall_time_seconds", None)
        result["sc_idc"] = sc_idc
    return result


def score_rollouts(
    rollout_rows: Sequence[Mapping[str, Any]], graph_sampler, *, reward_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matcher = StaticRelationMatcher(graph_sampler)
    cache = CachedQueryExecutor(graph_sampler)
    rows = []
    for rollout in rollout_rows:
        started = time.perf_counter()
        before = cache.executions
        completion = str(rollout["completion"])
        condition = int(rollout["condition_value"])
        base_scores = {"jaccard": 0.0, "dice": 0.0, "overlap": 0.0}
        nominal = False
        parse_ok = False
        parse_error = None
        query = None
        denotation = None
        sc_idc = None
        try:
            query = parse_action(completion)
            parse_ok = True
            occurrences = value_occurrences(query, "specific_relation", condition)
            nominal = bool(occurrences)
            denotation, _ = cache.execute(query)
            base_scores = set_semantic_scores(denotation, rollout["observation"])
            if nominal:
                sc_idc = audit_relation_value(
                    query=query, observation=rollout["observation"], condition_value=condition,
                    matcher=matcher, cache=cache, record_id=str(rollout["record_id"]),
                    reward_seed=reward_seed, base_denotation=denotation,
                )
        except Exception as exc:
            parse_ok = False
            parse_error = f"{type(exc).__name__}: {exc}"
            base_scores = {"jaccard": 0.0, "dice": 0.0, "overlap": 0.0}
            nominal = False
            sc_idc = None
        base_reward = (
            base_scores["jaccard"] + 0.5 * base_scores["dice"]
            + 0.5 * base_scores["overlap"] + float(nominal)
        )
        raw_bss = 0.0 if sc_idc is None else float(sc_idc["raw_bss"])
        augmented = {f"{alpha:g}": base_reward + alpha * raw_bss for alpha in ALPHA_GRID}
        rows.append({
            **rollout,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "nominal_adherence": nominal,
            "canonical_query_sha256": None if query is None else canonical_query_sha256(query),
            "executable_ast": None if query is None else action_string(query),
            "denotation_sha256": None if denotation is None else _sha256_bytes(
                _canonical_json(sorted(denotation)).encode("utf-8")
            ),
            "semantic_scores": base_scores,
            "base_reward": base_reward,
            "raw_bss": raw_bss,
            "augmented_rewards": augmented,
            "sc_idc": sc_idc,
            "graph_executions": cache.executions - before,
            "reward_wall_time_seconds": time.perf_counter() - started,
        })
    accounting = {
        "requests": cache.requests, "executions": cache.executions,
        "cache_hits": cache.requests - cache.executions,
        "cache_hit_rate": (
            (cache.requests - cache.executions) / cache.requests if cache.requests else 0.0
        ),
    }
    return rows, accounting


def _groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["group_index"])].append(row)
    result = [sorted(grouped[index], key=lambda row: int(row["generation_index"])) for index in sorted(grouped)]
    if any(len(group) != RESOURCE_BUDGET["generations_per_group"] for group in result):
        raise ValueError("Incomplete rollout group")
    return result


def _pair_counts(group: Sequence[Mapping[str, Any]], alpha: float) -> dict[str, int]:
    counts = Counter()
    for left_index in range(len(group)):
        for right_index in range(left_index + 1, len(group)):
            left, right = group[left_index], group[right_index]
            if left["completion"] == right["completion"]:
                continue
            counts["nonidentical"] += 1
            base_delta = float(left["base_reward"]) - float(right["base_reward"])
            aug_delta = float(left["augmented_rewards"][f"{alpha:g}"]) - float(right["augmented_rewards"][f"{alpha:g}"])
            if abs(base_delta) <= 1e-12:
                counts["base_tie"] += 1
                if abs(aug_delta) > 1e-12:
                    counts["base_tie_resolved"] += 1
            if base_delta * aug_delta < -1e-12:
                counts["base_order_reversal"] += 1
            # An IDC-caused Pareto error is algebraically impossible for a correct
            # positive scalar adapter; retain this explicit implementation gate.
            if base_delta <= 1e-12 and aug_delta > 1e-12:
                left_sem = float(left["semantic_scores"]["jaccard"])
                right_sem = float(right["semantic_scores"]["jaccard"])
                if left_sem < right_sem - 1e-12 and float(left["raw_bss"]) <= float(right["raw_bss"]) + 1e-12:
                    counts["pareto_harmful_inversion"] += 1
            if base_delta >= -1e-12 and aug_delta < -1e-12:
                left_sem = float(left["semantic_scores"]["jaccard"])
                right_sem = float(right["semantic_scores"]["jaccard"])
                if right_sem < left_sem - 1e-12 and float(right["raw_bss"]) <= float(left["raw_bss"]) + 1e-12:
                    counts["pareto_harmful_inversion"] += 1
    return dict(counts)


def _bootstrap(values: Sequence[float], *, seed: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "samples": BOOTSTRAP_SAMPLES}
    rng = np.random.default_rng(int(seed))
    chunks = []
    for start in range(0, BOOTSTRAP_SAMPLES, 500):
        count = min(500, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        chunks.append(array[indices].mean(axis=1))
    distribution = np.concatenate(chunks)
    return {
        "mean": float(array.mean()), "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)), "samples": BOOTSTRAP_SAMPLES,
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, elapsed_seconds: float, accounting: Mapping[str, Any], seed: int) -> dict[str, Any]:
    groups = _groups(rows)
    informative = []
    for group in groups:
        scorable = {
            row["executable_ast"] for row in group
            if row["sc_idc"] is not None and row["sc_idc"]["classification"] != "unscorable"
        }
        if len(scorable) >= 2:
            informative.append(group)
    alpha_table = []
    for alpha in ALPHA_GRID:
        pair_totals = Counter()
        zero_variance = []
        ordering_difference_groups = 0
        for group in groups:
            pair_totals.update(_pair_counts(group, alpha))
            rewards = [float(row["augmented_rewards"][f"{alpha:g}"]) for row in group]
            zero_variance.append(float(max(rewards) - min(rewards) <= 1e-12))
            if alpha > 0:
                pairs = _pair_counts(group, alpha)
                if pairs.get("base_tie_resolved", 0) or pairs.get("base_order_reversal", 0):
                    ordering_difference_groups += 1
        alpha_table.append({
            "alpha": alpha,
            "zero_reward_variance_group_rate": sum(zero_variance) / len(zero_variance),
            "nonidentical_pair_count": pair_totals["nonidentical"],
            "base_tie_pair_count": pair_totals["base_tie"],
            "nonidentical_completion_reward_tie_rate": (
                pair_totals["base_tie"] / pair_totals["nonidentical"]
                if pair_totals["nonidentical"] else 0.0
            ) if alpha == 0.0 else (
                (pair_totals["base_tie"] - pair_totals["base_tie_resolved"])
                / pair_totals["nonidentical"] if pair_totals["nonidentical"] else 0.0
            ),
            "base_tie_resolved_count": pair_totals["base_tie_resolved"],
            "base_tie_resolution_rate": (
                pair_totals["base_tie_resolved"] / pair_totals["base_tie"]
                if pair_totals["base_tie"] else None
            ),
            "base_order_reversal_count": pair_totals["base_order_reversal"],
            "pareto_harmful_inversion_count": pair_totals["pareto_harmful_inversion"],
            "ordering_difference_group_count": ordering_difference_groups,
        })
    eligible = [
        row for row in alpha_table if row["alpha"] > 0
        and row["pareto_harmful_inversion_count"] == 0
    ]
    if not eligible:
        selected_alpha = None
    else:
        # Frozen conservative lexicographic rule: maximize resolved base ties,
        # then minimize strict base-order reversals, then choose the smallest alpha.
        selected_alpha = min(
            eligible,
            key=lambda row: (
                -row["base_tie_resolved_count"], row["base_order_reversal_count"], row["alpha"]
            ),
        )["alpha"]
    selected = next((row for row in alpha_table if row["alpha"] == selected_alpha), None)
    base_zero = [float(max(row["base_reward"] for row in group) - min(row["base_reward"] for row in group) <= 1e-12) for group in groups]
    selected_zero = [
        float(max(row["augmented_rewards"][f"{selected_alpha:g}"] for row in group) - min(row["augmented_rewards"][f"{selected_alpha:g}"] for row in group) <= 1e-12)
        for group in groups
    ] if selected_alpha is not None else base_zero
    tie_reduction = [base - augmented for base, augmented in zip(base_zero, selected_zero)]
    wall_times = [float(row["reward_wall_time_seconds"]) for row in rows]
    raw_varying_groups = 0
    raw_ordering_signal_groups = 0
    for group in informative:
        if max(float(row["raw_bss"]) for row in group) - min(float(row["raw_bss"]) for row in group) > 1e-12:
            raw_varying_groups += 1
        differs = False
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left, right = group[left_index], group[right_index]
                if left["completion"] == right["completion"]:
                    continue
                base_delta = float(left["base_reward"]) - float(right["base_reward"])
                bss_delta = float(left["raw_bss"]) - float(right["raw_bss"])
                if abs(bss_delta) > 1e-12 and (
                    abs(base_delta) <= 1e-12 or base_delta * bss_delta < 0.0
                ):
                    differs = True
        raw_ordering_signal_groups += int(differs)
    classification = Counter(
        row["sc_idc"]["classification"] for row in rows if row["sc_idc"] is not None
    )
    topology = {}
    for name in sorted({str(row["topology"]) for row in rows}):
        subset = [row for row in rows if row["topology"] == name]
        topology[name] = {
            "completion_count": len(subset), "prompt_group_count": len({row["group_index"] for row in subset}),
            "nominal_rate": sum(bool(row["nominal_adherence"]) for row in subset) / len(subset),
            "scorable_rate": sum(row["sc_idc"] is not None and row["sc_idc"]["classification"] != "unscorable" for row in subset) / len(subset),
            "mean_raw_bss": statistics.fmean(float(row["raw_bss"]) for row in subset),
        }
    nonidentical_ties = sum(_pair_counts(group, 0.0).get("base_tie", 0) for group in groups)
    go_no_go = {
        "deterministic_reward_rows": True,
        "train_only_no_split_leakage": True,
        "informative_group_count_gte_30": len(informative) >= 30,
        "raw_sc_idc_changes_action_ordering": raw_ordering_signal_groups > 0,
        "raw_sc_idc_breaks_some_existing_base_ties": (
            True if nonidentical_ties == 0 else bool(selected and selected["base_tie_resolved_count"] > 0)
        ),
        "no_pareto_harmful_inversion": bool(selected and selected["pareto_harmful_inversion_count"] == 0),
        "within_resource_budget": (
            elapsed_seconds <= RESOURCE_BUDGET["maximum_wall_time_seconds"]
            and int(accounting["executions"]) <= len(rows) * RESOURCE_BUDGET["maximum_graph_executions_per_completion"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_experiment_1_summary",
        "prompt_group_count": len(groups), "completion_count": len(rows),
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / len(rows),
        "eos_rate": sum(bool(row["eos_emitted"]) for row in rows) / len(rows),
        "nominal_rate": sum(bool(row["nominal_adherence"]) for row in rows) / len(rows),
        "scorable_rate": sum(row["sc_idc"] is not None and row["sc_idc"]["classification"] != "unscorable" for row in rows) / len(rows),
        "informative_group_count": len(informative),
        "unique_completion_strings": len({row["completion"] for row in rows}),
        "unique_executable_asts": len({row["executable_ast"] for row in rows if row["executable_ast"] is not None}),
        "unique_denotations": len({row["denotation_sha256"] for row in rows if row["denotation_sha256"] is not None}),
        "classification_counts": dict(sorted(classification.items())),
        "informative_groups_with_raw_bss_variation": raw_varying_groups,
        "informative_groups_with_raw_ordering_distinct_from_base": raw_ordering_signal_groups,
        "raw_bss_distribution": {
            "mean": statistics.fmean(float(row["raw_bss"]) for row in rows),
            "p50": float(np.quantile([float(row["raw_bss"]) for row in rows], 0.5)),
            "p95": float(np.quantile([float(row["raw_bss"]) for row in rows], 0.95)),
            "positive_rate": sum(float(row["raw_bss"]) > 0.0 for row in rows) / len(rows),
            "mean_jaccard_when_positive": (
                statistics.fmean(
                    float(row["semantic_scores"]["jaccard"])
                    for row in rows if float(row["raw_bss"]) > 0.0
                ) if any(float(row["raw_bss"]) > 0.0 for row in rows) else None
            ),
        },
        "base_zero_reward_variance_group_rate": sum(base_zero) / len(base_zero),
        "selected_zero_reward_variance_group_rate": sum(selected_zero) / len(selected_zero),
        "zero_variance_rate_reduction_bootstrap": _bootstrap(
            tie_reduction, seed=derive_seed(seed, "experiment-1-bootstrap", "zero-variance")
        ),
        "alpha_selection": {
            "grid": list(ALPHA_GRID), "table": alpha_table, "selected_alpha": selected_alpha,
            "rule": "among positive alphas with zero Pareto-harmful inversions, maximize resolved base ties, minimize strict base-order reversals, then choose the smallest alpha",
            "train_only": True,
        },
        "reward_cost": {
            **accounting, "elapsed_seconds": elapsed_seconds,
            "per_completion_wall_time_p50": float(np.quantile(wall_times, 0.5)),
            "per_completion_wall_time_p95": float(np.quantile(wall_times, 0.95)),
            "mean_wall_time_bootstrap": _bootstrap(
                wall_times, seed=derive_seed(seed, "experiment-1-bootstrap", "wall-time")
            ),
        },
        "topology": topology,
        "resource_budget": RESOURCE_BUDGET,
        "go_no_go_gates": go_no_go,
        "decision": "go_to_grpo_preparation" if all(go_no_go.values()) else "no_go_fix_before_grpo",
        "evidence_boundary": {
            "train_graph_only": True, "final_evaluation_manifest_loaded": False,
            "predicate_necessity_claim": False, "natural_nonmarginal_called_laundering": False,
        },
    }


def run_experiment(args: argparse.Namespace) -> Path:
    started_at = _utc_now()
    started = time.monotonic()
    config = load_experiment_config(args.experiment_config)
    if config.dataset != "WN18RR" or config.condition != "specific_relation":
        raise ValueError("Experiment 1 requires the WN18RR specific_relation config")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPOSITORY_ROOT).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=REPOSITORY_ROOT).strip())
    if git_dirty:
        raise ValueError("Formal Experiment 1 requires a clean Git worktree")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _write_json(status_path, {"schema_version": 1, "status": "running", "started_at": started_at})
    try:
        preflight_path = Path(args.phase2_data_manifest).expanduser().resolve()
        records, _, lineage = _load_signal_contract(
            config=config, preflight_path=preflight_path,
            expected_signal_manifest_sha256=args.expected_signal_manifest_sha256,
        )
        graph_sampler = _graph_sampler(config)
        reference = _reference_preflight(records, graph_sampler, reward_seed=args.reward_seed)
        if not reference["pass"]:
            raise RuntimeError(f"Reward-3 reference preflight failed: {reference['gates']}")
        checkpoint = Path(args.parent_checkpoint).expanduser().resolve()
        loaded, selection = _load_parent(config, checkpoint)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Experiment 1 rollout requires CUDA")
        device = torch.device("cuda:0")
        model = loaded.model.to(device)
        rollout_started = time.monotonic()
        rollout_rows = _rollout(
            records=records, config=config, model=model, tokenizer=loaded.tokenizer, device=device
        )
        rollout_elapsed = time.monotonic() - rollout_started
        del model
        torch.cuda.empty_cache()
        rollout_path = output_dir / "rollouts.jsonl"
        _write_jsonl(rollout_path, rollout_rows)
        reward_started = time.monotonic()
        scored, accounting = score_rollouts(
            rollout_rows, graph_sampler, reward_seed=args.reward_seed
        )
        reward_elapsed = time.monotonic() - reward_started
        # Independent in-process rescore proves deterministic data/reward rows while
        # excluding intentionally observational wall-time fields.
        rescored, second_accounting = score_rollouts(
            rollout_rows, graph_sampler, reward_seed=args.reward_seed
        )
        first_bytes = _canonical_json([_deterministic_view(row) for row in scored]).encode("utf-8")
        second_bytes = _canonical_json([_deterministic_view(row) for row in rescored]).encode("utf-8")
        if first_bytes != second_bytes or accounting != second_accounting:
            raise AssertionError("Independent reward rescore is not byte deterministic")
        audit_path = output_dir / "completion-audit.jsonl"
        _write_jsonl(audit_path, scored)
        elapsed = time.monotonic() - started
        summary = summarize(
            scored, elapsed_seconds=elapsed, accounting=accounting, seed=config.seed
        )
        summary["reward_preflight"] = reference
        summary["deterministic_rescore"] = {
            "pass": True, "canonical_rows_sha256": _sha256_bytes(first_bytes),
            "second_accounting_identical": accounting == second_accounting,
        }
        summary["timing"] = {
            "rollout_seconds": rollout_elapsed, "reward_scoring_seconds": reward_elapsed,
            "total_seconds": elapsed,
        }
        summary_path = output_dir / "summary.json"
        _write_json(summary_path, summary)
        alpha = summary["alpha_selection"]["selected_alpha"]
        alpha_freeze = {
            "schema_version": 1, "kind": "sc_idc_alpha_freeze", "status": "frozen",
            "selected_alpha_idc": alpha, "selection_rule": summary["alpha_selection"]["rule"],
            "signal_manifest_sha256": lineage["signal_manifest_sha256"],
            "rl_train_manifest_sha256": lineage["rl_train_manifest_sha256"],
            "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_tree_sha256": checkpoint_tree_sha256(checkpoint),
            "code_sha": git_sha, "decision": summary["decision"],
            "grpo_started": False, "final_evaluation_manifest_loaded": False,
        }
        alpha_path = output_dir / "alpha-freeze.json"
        _write_json(alpha_path, alpha_freeze)
        artifacts = {}
        for path in (rollout_path, audit_path, summary_path, alpha_path):
            artifacts[path.name] = {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
        manifest = {
            "schema_version": 1, "kind": "sc_idc_experiment_1_manifest",
            "status": "completed", "started_at": started_at, "finished_at": _utc_now(),
            "code": {"sha": git_sha, "dirty": False},
            "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
            "config": {"path": str(config.source_path), "semantic_hash": config.semantic_hash,
                       "data_hash": config.data_hash, "kg_hash": config.kg_hash},
            "parent": {"path": str(checkpoint), "tree_sha256": alpha_freeze["parent_checkpoint_tree_sha256"],
                       "selected_epoch": selection.get("stage_epoch")},
            "lineage": lineage, "reward_seed": int(args.reward_seed),
            "resource_budget": RESOURCE_BUDGET, "artifacts": artifacts,
            "decision": summary["decision"], "final_evaluation_manifest_loaded": False,
        }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_json(status_path, {
            "schema_version": 1, "status": "completed", "finished_at": manifest["finished_at"],
            "decision": summary["decision"], "manifest": manifest_path.name,
        })
        return manifest_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": 1, "status": "failed", "finished_at": _utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--phase2-data-manifest", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-signal-manifest-sha256")
    parser.add_argument("--reward-seed", type=int, default=42)
    args = parser.parse_args()
    print(run_experiment(args))


if __name__ == "__main__":
    main()

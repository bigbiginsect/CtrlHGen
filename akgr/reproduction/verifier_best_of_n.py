"""Formal verifier-guided best-of-N decoding experiments (P1 and P2).

P1 is validation-only and keeps the sealed final-evaluation manifest closed.
P2 can open that manifest exactly once, but only after verifying a passed P1
artifact produced by the same clean Git SHA and frozen method contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

# Required before torch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from akgr.abduction_model.experiment_runner import _datasets, _graph_samplers, _require_cuda
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.sc_idc import (
    QueryExecutor,
    canonical_query_sha256,
    denotation_sha256,
    extract_semantic_slots,
    jaccard_score,
    neutralize_value_branches,
    parse_action,
    value_occurrences,
)
from akgr.reproduction.sc_idc_phase2_data import (
    checkpoint_tree_sha256,
    load_frozen_condition_manifest,
    verify_frozen_validation_dataset,
)
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.reproduction.ss_csc_common import load_pair_artifact, mean, sha256_file
from akgr.tokenizer import build_generation_prompt
from akgr.utils.load_util import load_reproduction_checkpoint
from akgr.utils.parsing_util import ans_unshift_indices


SCHEMA_VERSION = 1
P1_K_VALUES = (1, 2, 4, 8)
P2_K = 4
DEFAULT_SEED = 314159
SELECTION_RULE = (
    "exact+branch_supported",
    "exact+nominal",
    "exact",
    "highest_semantic_average",
    "highest_model_mean_log_probability",
    "lowest_canonical_hypothesis_sha256",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state() -> dict[str, Any]:
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_manifest_rows(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifact_path = (manifest_path.parent / str(manifest["artifact"]["path"])).resolve()
    rows: dict[str, dict[str, Any]] = {}
    with artifact_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in rows:
                raise ValueError(f"Duplicate frozen condition at {artifact_path}:{line_number}")
            rows[record_id] = row
    if len(rows) != int(manifest["artifact"]["count"]):
        raise ValueError("Frozen condition artifact count mismatch")
    return rows


def _validate_preflight(preflight_path: Path, config) -> dict[str, Any]:
    preflight = _read_json(preflight_path)
    if preflight.get("kind") != "sc_idc_specific_relation_sft_preflight":
        raise ValueError("Not an SC-IDC specific-relation preflight")
    if preflight.get("status") != "ready":
        raise ValueError("SC-IDC preflight is not ready")
    if preflight.get("isolation", {}).get("final_evaluation_sealed") is not True:
        raise ValueError("Final evaluation is not sealed")
    expected = {
        "config_semantic_hash": config.semantic_hash,
        "data_hash": config.data_hash,
        "kg_hash": config.kg_hash,
        "condition": "specific_relation",
    }
    for key, value in expected.items():
        if preflight.get("target", {}).get(key) != value:
            raise ValueError(f"Preflight target {key} mismatch")
    return preflight


def _load_frozen_split(*, config, preflight_path: Path, purpose: str, split: str):
    preflight = _validate_preflight(preflight_path, config)
    reference = preflight["conditions"][purpose]
    expected_sealed = purpose == "final_evaluation"
    if bool(reference.get("sealed")) != expected_sealed:
        raise ValueError(f"Unexpected sealed flag for {purpose}")
    manifest_path = (preflight_path.parent / str(reference["manifest_path"])).resolve()
    conditions, manifest = load_frozen_condition_manifest(
        manifest_path,
        target_config=config,
        purpose=purpose,
        consumer="final_evaluation" if expected_sealed else "reporting",
        expected_manifest_sha256=str(reference["manifest_sha256"]),
    )
    rows = _read_manifest_rows(manifest_path, manifest)
    contracts = {
        record_id: {
            "condition_value": str(row["condition_value"]),
            "target_sha256": str(row["target_sha256"]),
        }
        for record_id, row in rows.items()
    }
    datasets, _, _ = _datasets(config, [split], train_variant="base")
    dataset = datasets[split]
    verify_frozen_validation_dataset(
        dataset, condition_values=conditions, record_contracts=contracts
    )
    if len(dataset) != int(reference["count"]):
        raise ValueError(f"Frozen {purpose} count mismatch")
    examples = []
    for item in dataset:
        record_id = str(item["record_id"])
        examples.append({
            "record_id": record_id,
            "source": str(item["source"]),
            "condition": str(conditions[record_id]),
            "group_id": record_id,
            "side": None,
            "alternative_condition": None,
        })
    return examples, {
        "purpose": purpose,
        "split": split,
        "count": len(examples),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "conditions_path": str((manifest_path.parent / manifest["artifact"]["path"]).resolve()),
        "conditions_sha256": str(manifest["artifact"]["sha256"]),
        "sealed": expected_sealed,
    }


def _load_pair_examples(pair_summary_path: Path):
    _, pairs, artifact_path = load_pair_artifact(pair_summary_path, "pair_validation")
    examples = []
    for pair in pairs:
        for side, alternative in (("left", "right"), ("right", "left")):
            examples.append({
                "record_id": f"{pair['pair_id']}:{side}",
                "source": str(pair["source"]),
                "condition": str(pair[side]["condition_value"]),
                "group_id": str(pair["pair_id"]),
                "side": side,
                "alternative_condition": str(pair[alternative]["condition_value"]),
            })
    return examples, {
        "pair_summary_path": str(pair_summary_path),
        "pair_summary_sha256": sha256_file(pair_summary_path),
        "pair_artifact_path": str(artifact_path),
        "pair_artifact_sha256": sha256_file(artifact_path),
        "pair_count": len(pairs),
        "directional_count": len(examples),
    }


def _verify_checkpoint(
    *, checkpoint: Path, selection_path: Path, expected_tree: str,
    expected_pair_summary_sha256: str, config,
):
    selection = _read_json(selection_path)
    if selection.get("kind") != "ss_csc_branch_selection":
        raise ValueError("Checkpoint selection is not an SS-CSC branch selection")
    if selection.get("branch") != "single_target":
        raise ValueError("Verifier proposal requires the single-target branch")
    if selection.get("sealed_final_evaluation_loaded") is not False:
        raise ValueError("Checkpoint selection did not preserve sealed evaluation isolation")
    if Path(str(selection.get("selected_checkpoint"))).expanduser().resolve() != checkpoint:
        raise ValueError("Checkpoint differs from the frozen branch selection")
    chosen = selection.get("selection", {})
    if int(chosen.get("epoch", -1)) != 5 or chosen.get("health_pass") is not True:
        raise ValueError("Verifier proposal requires healthy single-target epoch 5")
    actual_tree = checkpoint_tree_sha256(checkpoint)
    if actual_tree != expected_tree:
        raise ValueError("Checkpoint tree SHA256 mismatch")
    loaded = load_reproduction_checkpoint(
        checkpoint,
        mode="test",
        expected_stage="ss_csc_single_target",
        expected_condition="specific_relation",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=expected_pair_summary_sha256,
    )
    return loaded, {
        "path": str(checkpoint),
        "tree_sha256": actual_tree,
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selection_code_sha": str(selection["code_sha"]),
        "selected_epoch": 5,
        "reload_verified": True,
    }


def _observation(source: str) -> frozenset[int]:
    return frozenset(ans_unshift_indices([int(token) for token in source.split() if token]))


def _semantic_scores(prediction: Iterable[int], observation: Iterable[int]) -> dict[str, float]:
    left, right = frozenset(prediction), frozenset(observation)
    intersection = len(left & right)
    union = len(left | right)
    jaccard = intersection / union if union else 1.0
    dice_denominator = len(left) + len(right)
    dice = 2.0 * intersection / dice_denominator if dice_denominator else 1.0
    overlap_denominator = min(len(left), len(right)) + 1e-5
    overlap = intersection / overlap_denominator
    return {
        "jaccard": jaccard,
        "dice": dice,
        "overlap": overlap,
        "semantic_average": (jaccard + dice + overlap) / 3.0,
    }


def _audit_candidate(
    prediction: str,
    *,
    condition: str,
    observation: frozenset[int],
    executor: QueryExecutor,
) -> dict[str, Any]:
    executions = 0
    try:
        query = parse_action(prediction)
        executions += 1
        denotation = frozenset(executor.execute(query))
        occurrences = value_occurrences(query, "specific_relation", int(condition))
        nominal = bool(occurrences)
        branch_supported = False
        nonroot = False
        branch_delta = None
        if nominal:
            neutral, branches = neutralize_value_branches(
                query, "specific_relation", int(condition)
            )
            executions += 1
            neutral_denotation = frozenset(executor.execute(neutral))
            branch_delta = jaccard_score(denotation, observation) - jaccard_score(
                neutral_denotation, observation
            )
            branch_supported = branch_delta > 0.0
            nonroot = branch_supported and all(
                branch["branch_operator"] is not None for branch in branches
            )
        scores = _semantic_scores(denotation, observation)
        relation_values = sorted({
            int(slot.value)
            for slot in extract_semantic_slots(query, "specific_relation")
        })
        return {
            "parse_ok": True,
            "parse_error": None,
            "query_sha256": canonical_query_sha256(query),
            "canonical_hypothesis_sha256": canonical_query_sha256(query),
            "denotation_sha256": denotation_sha256(denotation),
            "denotation_cardinality": len(denotation),
            "exact": denotation == observation,
            "nominal": nominal,
            "branch_supported": branch_supported,
            "nonroot_branch_supported": nonroot,
            "branch_marginal_delta": branch_delta,
            "relation_values": relation_values,
            "graph_executions": executions,
            **scores,
        }
    except Exception as exc:
        raw_hash = hashlib.sha256(prediction.encode("utf-8")).hexdigest()
        return {
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "query_sha256": None,
            "canonical_hypothesis_sha256": raw_hash,
            "denotation_sha256": None,
            "denotation_cardinality": None,
            "exact": False,
            "nominal": False,
            "branch_supported": False,
            "nonroot_branch_supported": False,
            "branch_marginal_delta": None,
            "relation_values": [],
            "graph_executions": executions,
            "jaccard": 0.0,
            "dice": 0.0,
            "overlap": 0.0,
            "semantic_average": 0.0,
        }


def _candidate_tier(candidate: Mapping[str, Any]) -> int:
    if candidate["exact"] and candidate["branch_supported"]:
        return 3
    if candidate["exact"] and candidate["nominal"]:
        return 2
    if candidate["exact"]:
        return 1
    return 0


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    """Apply the frozen semantic-first lexicographic selection rule."""
    if not candidates:
        raise ValueError("Candidate selection requires a non-empty sequence")
    tiers = [_candidate_tier(candidate) for candidate in candidates]
    best_tier = max(tiers)
    eligible = [index for index, tier in enumerate(tiers) if tier == best_tier]
    reason = {
        3: "exact+branch_supported",
        2: "exact+nominal",
        1: "exact",
        0: "highest_semantic_average",
    }[best_tier]
    if best_tier == 0:
        best_semantic = max(float(candidates[index]["semantic_average"]) for index in eligible)
        eligible = [
            index for index in eligible
            if float(candidates[index]["semantic_average"]) == best_semantic
        ]
    best_logp = max(float(candidates[index]["model_mean_log_probability"]) for index in eligible)
    eligible = [
        index for index in eligible
        if float(candidates[index]["model_mean_log_probability"]) == best_logp
    ]
    selected = min(
        eligible,
        key=lambda index: str(candidates[index]["canonical_hypothesis_sha256"]),
    )
    return selected, reason


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), int(size)):
        yield values[start:start + int(size)]


def _generate_pass(
    *, model, tokenizer, prompts: Sequence[str], device, max_new_tokens: int,
    batch_size: int, do_sample: bool,
) -> list[dict[str, Any]]:
    generated_rows: list[dict[str, Any]] = []
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        model.eval()
        with torch.no_grad():
            for prompt_batch in _chunks(list(prompts), batch_size):
                encoded = tokenizer(
                    list(prompt_batch), padding="longest", add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                kwargs = {
                    "input_ids": encoded.input_ids,
                    "attention_mask": encoded.attention_mask,
                    "do_sample": bool(do_sample),
                    "max_new_tokens": int(max_new_tokens),
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }
                if do_sample:
                    kwargs.update({"temperature": 1.0, "top_k": 0, "top_p": 1.0})
                generated = model.generate(**kwargs)
                completion_ids = generated.sequences[:, encoded.input_ids.shape[1]:]
                transition = model.compute_transition_scores(
                    generated.sequences, generated.scores, normalize_logits=True
                )
                predictions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                for prediction, token_ids, token_logps in zip(
                    predictions, completion_ids.tolist(), transition.detach().cpu().tolist()
                ):
                    eos = tokenizer.eos_token_id in token_ids
                    length = token_ids.index(tokenizer.eos_token_id) + 1 if eos else len(token_ids)
                    used_logps = [float(value) for value in token_logps[:length]]
                    generated_rows.append({
                        "prediction": prediction,
                        "eos_emitted": eos,
                        "generated_token_count": length,
                        "hit_max_new_tokens": not eos and length >= int(max_new_tokens),
                        "model_mean_log_probability": (
                            math.fsum(used_logps) / len(used_logps) if used_logps else -math.inf
                        ),
                    })
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    return generated_rows


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _evaluate_examples(
    *, name: str, examples: Sequence[Mapping[str, Any]], model, tokenizer, device,
    graph_sampler, seed: int, batch_size: int, max_new_tokens: int,
    max_k: int, include_greedy: bool, condition_delimiter: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [
        build_generation_prompt(
            str(example["source"]),
            ConditionSpec("specific_relation", str(example["condition"])),
            tokenizer,
            condition_delimiter=condition_delimiter,
        )
        for example in examples
    ]
    observations = [_observation(str(example["source"])) for example in examples]
    executor = QueryExecutor(graph_sampler)
    rows = [{
        **dict(example),
        "observation_sha256": denotation_sha256(observation),
        "observation_cardinality": len(observation),
        "greedy": None,
        "sample_candidates": [],
        "selections": {},
    } for example, observation in zip(examples, observations)]
    costs: dict[str, Any] = {}

    if include_greedy:
        seed_everything(derive_seed(seed, name, "greedy"))
        _synchronize()
        started = time.perf_counter()
        generated = _generate_pass(
            model=model, tokenizer=tokenizer, prompts=prompts, device=device,
            max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=False,
        )
        _synchronize()
        generation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for row, example, observation, candidate in zip(rows, examples, observations, generated):
            audit = _audit_candidate(
                str(candidate["prediction"]), condition=str(example["condition"]),
                observation=observation, executor=executor,
            )
            row["greedy"] = {**candidate, **audit}
        verification_seconds = time.perf_counter() - started
        costs["greedy"] = {
            "generation_seconds": generation_seconds,
            "verification_seconds": verification_seconds,
            "wall_seconds": generation_seconds + verification_seconds,
            "graph_executions": sum(int(row["greedy"]["graph_executions"]) for row in rows),
            "generated_sequences": len(rows),
            "generated_tokens": sum(int(row["greedy"]["generated_token_count"]) for row in rows),
        }

    sample_generation_seconds = 0.0
    sample_verification_seconds = 0.0
    milestone_costs: dict[int, dict[str, Any]] = {}
    for candidate_index in range(max_k):
        seed_everything(derive_seed(seed, name, "sample", candidate_index))
        _synchronize()
        started = time.perf_counter()
        generated = _generate_pass(
            model=model, tokenizer=tokenizer, prompts=prompts, device=device,
            max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=True,
        )
        _synchronize()
        sample_generation_seconds += time.perf_counter() - started
        started = time.perf_counter()
        for row, example, observation, candidate in zip(rows, examples, observations, generated):
            audit = _audit_candidate(
                str(candidate["prediction"]), condition=str(example["condition"]),
                observation=observation, executor=executor,
            )
            row["sample_candidates"].append({
                "candidate_index": candidate_index,
                **candidate,
                **audit,
            })
        sample_verification_seconds += time.perf_counter() - started
        k = candidate_index + 1
        if k in set(P1_K_VALUES) or k == max_k:
            milestone_costs[k] = {
                "generation_seconds": sample_generation_seconds,
                "verification_seconds": sample_verification_seconds,
                "wall_seconds": sample_generation_seconds + sample_verification_seconds,
                "graph_executions": sum(
                    int(candidate["graph_executions"])
                    for row in rows for candidate in row["sample_candidates"][:k]
                ),
                "generated_sequences": len(rows) * k,
                "generated_tokens": sum(
                    int(candidate["generated_token_count"])
                    for row in rows for candidate in row["sample_candidates"][:k]
                ),
            }

    requested_k = [k for k in P1_K_VALUES if k <= max_k]
    if max_k not in requested_k:
        requested_k.append(max_k)
    for row in rows:
        for k in requested_k:
            index, reason = select_candidate(row["sample_candidates"][:k])
            row["selections"][f"sample_k{k}"] = {
                "selected_index": index,
                "selection_reason": reason,
            }
    for k, cost in milestone_costs.items():
        costs[f"sample_k{k}"] = cost
    return rows, costs


def _selected(row: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    if mode == "greedy":
        return row["greedy"]
    index = int(row["selections"][mode]["selected_index"])
    return row["sample_candidates"][index]


def _summarize_mode(rows: Sequence[Mapping[str, Any]], mode: str, cost: Mapping[str, Any]):
    selected = [_selected(row, mode) for row in rows]
    summary = {
        "count": len(rows),
        "semantic_average": mean([float(candidate["semantic_average"]) for candidate in selected]),
        "jaccard": mean([float(candidate["jaccard"]) for candidate in selected]),
        "dice": mean([float(candidate["dice"]) for candidate in selected]),
        "overlap": mean([float(candidate["overlap"]) for candidate in selected]),
        "exact_rate": mean([float(candidate["exact"]) for candidate in selected]),
        "nominal_rate": mean([float(candidate["nominal"]) for candidate in selected]),
        "branch_supported_rate": mean([
            float(candidate["branch_supported"]) for candidate in selected
        ]),
        "nonroot_branch_supported_rate": mean([
            float(candidate["nonroot_branch_supported"]) for candidate in selected
        ]),
        "parse_rate": mean([float(candidate["parse_ok"]) for candidate in selected]),
        "eos_rate": mean([float(candidate["eos_emitted"]) for candidate in selected]),
        "selected_mean_generated_tokens": mean([
            float(candidate["generated_token_count"]) for candidate in selected
        ]),
        "cost": {
            **dict(cost),
            "mean_generated_tokens_per_candidate": (
                float(cost["generated_tokens"]) / int(cost["generated_sequences"])
            ),
        },
    }
    if mode.startswith("sample_k"):
        k = int(mode.removeprefix("sample_k"))
        summary["mean_unique_ast"] = mean([
            float(len({
                candidate["query_sha256"]
                for candidate in row["sample_candidates"][:k]
                if candidate["query_sha256"] is not None
            }))
            for row in rows
        ])
    else:
        summary["mean_unique_ast"] = mean([float(candidate["parse_ok"]) for candidate in selected])
    return summary


def _pair_switch_metrics(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    ast_switches, condition_only_switches = [], []
    for group_id, pair_rows in grouped.items():
        if len(pair_rows) != 2 or {row["side"] for row in pair_rows} != {"left", "right"}:
            raise ValueError(f"Malformed directional pair group {group_id}")
        pair_rows = sorted(pair_rows, key=lambda row: str(row["side"]))
        candidates = [_selected(row, mode) for row in pair_rows]
        switch = (
            all(bool(candidate["exact"]) for candidate in candidates)
            and all(candidate["query_sha256"] is not None for candidate in candidates)
            and candidates[0]["query_sha256"] != candidates[1]["query_sha256"]
        )
        condition_only = switch and all(
            int(row["condition"]) in set(candidate["relation_values"])
            and int(row["alternative_condition"]) not in set(candidate["relation_values"])
            for row, candidate in zip(pair_rows, candidates)
        )
        ast_switches.append(float(switch))
        condition_only_switches.append(float(condition_only))
    return {
        "pair_count": len(grouped),
        "bilateral_exact_ast_switch_rate": mean(ast_switches),
        "bilateral_exact_condition_only_switch_rate": mean(condition_only_switches),
    }


def _summaries(rows, costs, *, pair: bool) -> dict[str, Any]:
    result = {}
    for mode, cost in costs.items():
        result[mode] = _summarize_mode(rows, mode, cost)
        if pair:
            result[mode]["pair_switch"] = _pair_switch_metrics(rows, mode)
    return result


def p1_gate(original: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    greedy_original, k4_original = original["greedy"], original["sample_k4"]
    greedy_pair, k4_pair = pair["greedy"], pair["sample_k4"]
    thresholds = {
        "paired_sft_bilateral_exact_ast_switch": 0.66766,
        "bilateral_tolerance": 0.01,
        "parse_eos_tolerance": 0.005,
    }
    checks = {
        "original_semantic_non_decrease": (
            float(k4_original["semantic_average"]) >= float(greedy_original["semantic_average"])
        ),
        "original_branch_non_decrease": (
            float(k4_original["branch_supported_rate"]) >= float(greedy_original["branch_supported_rate"])
        ),
        "pair_bilateral_exact_ast_switch": (
            float(k4_pair["pair_switch"]["bilateral_exact_ast_switch_rate"])
            >= thresholds["paired_sft_bilateral_exact_ast_switch"] - thresholds["bilateral_tolerance"]
        ),
        "original_parse": (
            float(k4_original["parse_rate"]) >= float(greedy_original["parse_rate"])
            - thresholds["parse_eos_tolerance"]
        ),
        "original_eos": (
            float(k4_original["eos_rate"]) >= float(greedy_original["eos_rate"])
            - thresholds["parse_eos_tolerance"]
        ),
        "pair_parse": (
            float(k4_pair["parse_rate"]) >= float(greedy_pair["parse_rate"])
            - thresholds["parse_eos_tolerance"]
        ),
        "pair_eos": (
            float(k4_pair["eos_rate"]) >= float(greedy_pair["eos_rate"])
            - thresholds["parse_eos_tolerance"]
        ),
        "k4_cost_recorded": (
            float(k4_original["cost"]["wall_seconds"]) > 0.0
            and int(k4_original["cost"]["graph_executions"]) > 0
            and float(k4_pair["cost"]["wall_seconds"]) > 0.0
            and int(k4_pair["cost"]["graph_executions"]) > 0
        ),
        "selection_reference_free": True,
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def _method_contract(*, seed: int, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "condition_kind": "specific_relation",
        "generator_frozen": True,
        "seed": int(seed),
        "sample_stream": "derive_seed(seed,dataset,sample,candidate_index)",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "p1_k_values": list(P1_K_VALUES),
        "p2_k": P2_K,
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "selection_rule": list(SELECTION_RULE),
        "selection_inputs": ["observation", "observable_graph", "condition", "candidate"],
        "reference_hypothesis_available_to_selection": False,
    }


def _prepare_output(output_dir: Path, kind: str, code: Mapping[str, Any]) -> Path:
    if code["dirty"]:
        raise ValueError("Formal verifier decoding requires a clean Git worktree")
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    _write_json(status_path, {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "status": "running",
        "started_at": _utc_now(),
        "code": dict(code),
    })
    return status_path


def run_p1(args: argparse.Namespace) -> Path:
    code = _git_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_path = _prepare_output(output_dir, "verifier_best_of_n_p1_status", code)
    try:
        started = time.perf_counter()
        config = load_experiment_config(args.experiment_config)
        preflight_path = Path(args.phase2_preflight).expanduser().resolve()
        original_examples, original_contract = _load_frozen_split(
            config=config, preflight_path=preflight_path, purpose="validation", split="valid"
        )
        pair_examples, pair_contract = _load_pair_examples(
            Path(args.pair_summary).expanduser().resolve()
        )
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        selection_path = Path(args.checkpoint_selection).expanduser().resolve()
        loaded, checkpoint_contract = _verify_checkpoint(
            checkpoint=checkpoint, selection_path=selection_path,
            expected_tree=str(args.checkpoint_tree_sha256),
            expected_pair_summary_sha256=pair_contract["pair_summary_sha256"],
            config=config,
        )
        device = _require_cuda("verifier best-of-N P1")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        model, tokenizer = loaded.model.to(device), loaded.tokenizer
        graph_samplers = _graph_samplers(config)
        max_new_tokens = int(config.raw["generation"]["max_new_tokens"])
        original_rows, original_costs = _evaluate_examples(
            name="original_validation", examples=original_examples,
            model=model, tokenizer=tokenizer, device=device,
            graph_sampler=graph_samplers["valid"], seed=int(args.seed),
            batch_size=int(args.batch_size), max_new_tokens=max_new_tokens,
            max_k=max(P1_K_VALUES), include_greedy=True,
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]),
        )
        pair_rows, pair_costs = _evaluate_examples(
            name="pair_validation", examples=pair_examples,
            model=model, tokenizer=tokenizer, device=device,
            graph_sampler=graph_samplers["train"], seed=int(args.seed),
            batch_size=int(args.batch_size), max_new_tokens=max_new_tokens,
            max_k=max(P1_K_VALUES), include_greedy=True,
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]),
        )
        original_summary = _summaries(original_rows, original_costs, pair=False)
        pair_summary = _summaries(pair_rows, pair_costs, pair=True)
        gate = p1_gate(original_summary, pair_summary)
        artifacts = {
            "original_validation_records": _write_jsonl(
                output_dir / "original-validation-records.jsonl", original_rows
            ),
            "pair_validation_records": _write_jsonl(
                output_dir / "pair-validation-records.jsonl", pair_rows
            ),
        }
        method = _method_contract(
            seed=int(args.seed), batch_size=int(args.batch_size),
            max_new_tokens=max_new_tokens,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p1",
            "status": "passed" if gate["passed"] else "failed_gate",
            "completed_at": _utc_now(),
            "code": code,
            "command": [sys.executable, "-m", __name__, "p1", *sys.argv[2:]],
            "method": method,
            "inputs": {
                "experiment_config": str(config.source_path),
                "experiment_config_sha256": sha256_file(config.source_path),
                "sampling_manifest": str(config.sampling_manifest_path),
                "sampling_manifest_sha256": sha256_file(config.sampling_manifest_path),
                "config_semantic_hash": config.semantic_hash,
                "data_hash": config.data_hash,
                "kg_hash": config.kg_hash,
                "checkpoint": checkpoint_contract,
                "original_validation": original_contract,
                "pair_validation": pair_contract,
                "sealed_final_evaluation_loaded": False,
            },
            "summaries": {
                "original_validation": original_summary,
                "pair_validation": pair_summary,
            },
            "decision_gate": gate,
            "artifacts": artifacts,
            "total_wall_seconds": time.perf_counter() - started,
            "limitations": [
                "WN18RR specific_relation only",
                "decoding improvement; generator parameters are unchanged",
                "observable-graph exactness is not full-KG logical equivalence",
            ],
        }
        report_path = output_dir / "summary.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p1_status",
            "status": report["status"],
            "completed_at": _utc_now(),
            "code": code,
            "summary_sha256": sha256_file(report_path),
            "sealed_final_evaluation_loaded": False,
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p1_status",
            "status": "failed",
            "failed_at": _utc_now(),
            "code": code,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "sealed_final_evaluation_loaded": False,
        })
        raise


def _verify_passed_p1(p1_path: Path, code: Mapping[str, Any]) -> dict[str, Any]:
    p1 = _read_json(p1_path)
    if p1.get("kind") != "verifier_best_of_n_p1" or p1.get("status") != "passed":
        raise ValueError("P2 requires a passed formal P1 summary")
    if p1.get("decision_gate", {}).get("passed") is not True:
        raise ValueError("P1 decision gate did not pass")
    if p1.get("code") != dict(code):
        raise ValueError("P2 must use the exact clean Git SHA used by P1")
    if p1.get("inputs", {}).get("sealed_final_evaluation_loaded") is not False:
        raise ValueError("P1 did not preserve sealed final-evaluation isolation")
    method = p1.get("method")
    if method != _method_contract(
        seed=int(method["seed"]), batch_size=int(method["batch_size"]),
        max_new_tokens=int(method["max_new_tokens"]),
    ):
        raise ValueError("P1 method contract is not the frozen verifier contract")
    for artifact in p1.get("artifacts", {}).values():
        artifact_path = (p1_path.parent / str(artifact["path"])).resolve()
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError("P1 record artifact hash mismatch")
    return p1


def run_p2(args: argparse.Namespace) -> Path:
    code = _git_state()
    output_dir = Path(args.output_dir).expanduser().resolve()
    status_path = _prepare_output(output_dir, "verifier_best_of_n_p2_status", code)
    final_manifest_loaded = False
    try:
        started = time.perf_counter()
        p1_path = Path(args.p1_summary).expanduser().resolve()
        p1 = _verify_passed_p1(p1_path, code)
        method = p1["method"]
        config_path = Path(p1["inputs"]["experiment_config"]).expanduser().resolve()
        if sha256_file(config_path) != p1["inputs"]["experiment_config_sha256"]:
            raise ValueError("P1 experiment config hash mismatch")
        config = load_experiment_config(config_path)
        preflight_path = Path(
            p1["inputs"]["original_validation"]["preflight_path"]
        ).expanduser().resolve()
        if sha256_file(preflight_path) != p1["inputs"]["original_validation"]["preflight_sha256"]:
            raise ValueError("P1 preflight hash mismatch")
        checkpoint_contract = p1["inputs"]["checkpoint"]
        loaded, verified_checkpoint = _verify_checkpoint(
            checkpoint=Path(checkpoint_contract["path"]).expanduser().resolve(),
            selection_path=Path(checkpoint_contract["selection_path"]).expanduser().resolve(),
            expected_tree=str(checkpoint_contract["tree_sha256"]),
            expected_pair_summary_sha256=p1["inputs"]["pair_validation"]["pair_summary_sha256"],
            config=config,
        )
        if verified_checkpoint != checkpoint_contract:
            raise ValueError("P2 checkpoint contract differs from P1")

        # This is the single point where the sealed artifact is opened.
        final_examples, final_contract = _load_frozen_split(
            config=config, preflight_path=preflight_path,
            purpose="final_evaluation", split="test",
        )
        final_manifest_loaded = True
        device = _require_cuda("verifier best-of-N P2")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)
        model, tokenizer = loaded.model.to(device), loaded.tokenizer
        graph_samplers = _graph_samplers(config)
        rows, costs = _evaluate_examples(
            name="final_evaluation", examples=final_examples,
            model=model, tokenizer=tokenizer, device=device,
            graph_sampler=graph_samplers["test"], seed=int(method["seed"]),
            batch_size=int(method["batch_size"]),
            max_new_tokens=int(method["max_new_tokens"]),
            max_k=P2_K, include_greedy=True,
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]),
        )
        summaries = _summaries(rows, costs, pair=False)
        artifact = _write_jsonl(output_dir / "final-evaluation-records.jsonl", rows)
        greedy, k4 = summaries["greedy"], summaries["sample_k4"]
        comparison = {
            key: float(k4[key]) - float(greedy[key])
            for key in (
                "semantic_average", "exact_rate", "nominal_rate",
                "branch_supported_rate", "nonroot_branch_supported_rate",
                "parse_rate", "eos_rate",
            )
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p2",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "command": [sys.executable, "-m", __name__, "p2", *sys.argv[2:]],
            "method": method,
            "p1": {
                "path": str(p1_path),
                "sha256": sha256_file(p1_path),
                "gate_passed": True,
            },
            "inputs": {
                "checkpoint": verified_checkpoint,
                "final_evaluation": final_contract,
                "sealed_final_evaluation_loaded": True,
            },
            "summaries": summaries,
            "sample_k4_minus_greedy": comparison,
            "artifact": artifact,
            "total_wall_seconds": time.perf_counter() - started,
            "post_evaluation_tuning_permitted": False,
            "limitations": p1["limitations"],
        }
        report_path = output_dir / "summary.json"
        _write_json(report_path, report)
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p2_status",
            "status": "completed",
            "completed_at": _utc_now(),
            "code": code,
            "summary_sha256": sha256_file(report_path),
            "sealed_final_evaluation_loaded": True,
        })
        return report_path
    except BaseException as exc:
        _write_json(status_path, {
            "schema_version": SCHEMA_VERSION,
            "kind": "verifier_best_of_n_p2_status",
            "status": "failed",
            "failed_at": _utc_now(),
            "code": code,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "sealed_final_evaluation_loaded": final_manifest_loaded,
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    p1 = subparsers.add_parser("p1")
    p1.add_argument("--experiment-config", required=True)
    p1.add_argument("--phase2-preflight", required=True)
    p1.add_argument("--pair-summary", required=True)
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--checkpoint-selection", required=True)
    p1.add_argument("--checkpoint-tree-sha256", required=True)
    p1.add_argument("--output-dir", required=True)
    p1.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p1.add_argument("--batch-size", type=int, default=32)
    p2 = subparsers.add_parser("p2")
    p2.add_argument("--p1-summary", required=True)
    p2.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("batch size must be positive")
    result = run_p1(args) if args.stage == "p1" else run_p2(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

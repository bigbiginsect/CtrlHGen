"""Pattern-conditioned verifier-guided Best-of-N experiments (P1/P2)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from akgr.abduction_model.experiment_runner import _datasets, _graph_samplers, _require_cuda
from akgr.evaluation import get_smatch_score
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.sc_idc import (
    QueryExecutor,
    action_string,
    canonical_query_sha256,
    denotation_sha256,
    parse_action,
    shifted_wordlist,
)
from akgr.reproduction.sc_idc_phase2_data import checkpoint_tree_sha256
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.tokenizer import build_generation_prompt, condition_value_from_target, number_to_pattern
from akgr.utils.load_util import load_reproduction_checkpoint
from akgr.utils.parsing_util import ans_unshift_indices, qry_wordlist_2_graph


SCHEMA_VERSION = 1
METHOD_VERSION = "pattern-best-of-n-v1"
P1_K_VALUES = (1, 2, 4, 8)
P2_K = 4
DEFAULT_SEED = 314159
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_MANIFEST_SHA256 = "244ef264ad538df30ad58d72127d7fe3c31f52480dc3ddac882978400c49e460"
EXPECTED_ARTIFACT_SHA256 = {
    "valid": "950dca912f2d602d32894562b6b17f856e350c0158d125bb01d784d51adeed75",
    "test": "38bf3b26367be27cf03689a1dd2b168a505024fde14ea065df4ec084c01929a3",
}
SELECTORS = ("first_sample", "likelihood_only", "semantic_only", "exact_semantic", "pattern_aware")
SELECTION_RULE = (
    "exact+pattern_match",
    "exact",
    "highest_semantic_average",
    "pattern_match_on_equal_semantic_average",
    "highest_model_mean_log_probability",
    "lowest_canonical_hypothesis_sha256",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_state() -> dict[str, Any]:
    return {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": _sha256_file(path), "bytes": path.stat().st_size}


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def canonical_pattern(query) -> str:
    """Return pattern from a parsed/canonical AST, never from malformed surface text."""
    return number_to_pattern(action_string(query))


def _observation(source: str) -> frozenset[int]:
    return frozenset(ans_unshift_indices([int(token) for token in source.split() if token]))


def _semantic_scores(prediction: Iterable[int], observation: Iterable[int]) -> dict[str, float]:
    left, right = frozenset(prediction), frozenset(observation)
    intersection = len(left & right)
    union = len(left | right)
    jaccard = intersection / union if union else 1.0
    denominator = len(left) + len(right)
    dice = 2.0 * intersection / denominator if denominator else 1.0
    overlap_denominator = min(len(left), len(right)) + 1e-5
    overlap = intersection / overlap_denominator
    return {"jaccard": jaccard, "dice": dice, "overlap": overlap,
            "semantic_average": (jaccard + dice + overlap) / 3.0}


def audit_candidate(prediction: str, *, condition: str, observation: frozenset[int], executor: QueryExecutor) -> dict[str, Any]:
    try:
        query = parse_action(prediction)
        denotation = frozenset(executor.execute(query))
        pattern = canonical_pattern(query)
        return {
            "parse_ok": True,
            "parse_error": None,
            "query_sha256": canonical_query_sha256(query),
            "canonical_hypothesis_sha256": canonical_query_sha256(query),
            "candidate_pattern": pattern,
            "pattern_match": pattern == condition,
            "denotation_sha256": denotation_sha256(denotation),
            "denotation_cardinality": len(denotation),
            "exact": denotation == observation,
            "graph_executions": 1,
            **_semantic_scores(denotation, observation),
        }
    except Exception as exc:
        return {
            "parse_ok": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "query_sha256": None,
            "canonical_hypothesis_sha256": _text_sha256(prediction),
            "candidate_pattern": None,
            "pattern_match": False,
            "denotation_sha256": None,
            "denotation_cardinality": None,
            "exact": False,
            "graph_executions": 0,
            "jaccard": 0.0,
            "dice": 0.0,
            "overlap": 0.0,
            "semantic_average": 0.0,
        }


def _break_tie(candidates: Sequence[Mapping[str, Any]], eligible: Sequence[int]) -> int:
    best_logp = max(float(candidates[index]["model_mean_log_probability"]) for index in eligible)
    eligible = [index for index in eligible if float(candidates[index]["model_mean_log_probability"]) == best_logp]
    return min(eligible, key=lambda index: str(candidates[index]["canonical_hypothesis_sha256"]))


def select_candidate(candidates: Sequence[Mapping[str, Any]], selector: str = "pattern_aware") -> tuple[int, str]:
    if not candidates:
        raise ValueError("Candidate selection requires a non-empty sequence")
    if selector == "first_sample":
        return 0, "candidate_0"
    eligible = list(range(len(candidates)))
    if selector == "likelihood_only":
        return _break_tie(candidates, eligible), "highest_model_mean_log_probability"
    if selector == "pattern_aware":
        joint = [index for index in eligible if candidates[index]["exact"] and candidates[index]["pattern_match"]]
        if joint:
            return _break_tie(candidates, joint), "exact+pattern_match"
    if selector in {"pattern_aware", "exact_semantic"}:
        exact = [index for index in eligible if candidates[index]["exact"]]
        if exact:
            eligible = exact
            reason = "exact"
        else:
            reason = "highest_semantic_average"
    elif selector == "semantic_only":
        reason = "highest_semantic_average"
    else:
        raise ValueError(f"Unknown selector: {selector}")
    best_semantic = max(float(candidates[index]["semantic_average"]) for index in eligible)
    eligible = [index for index in eligible if float(candidates[index]["semantic_average"]) == best_semantic]
    if selector == "pattern_aware":
        matched = [index for index in eligible if candidates[index]["pattern_match"]]
        if matched:
            eligible = matched
            reason += "+pattern_match_tiebreak"
    return _break_tie(candidates, eligible), reason


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), int(size)):
        yield values[start:start + int(size)]


def _generate_pass(*, model, tokenizer, prompts: Sequence[str], device, max_new_tokens: int,
                   batch_size: int, do_sample: bool) -> list[dict[str, Any]]:
    rows = []
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        model.eval()
        with torch.no_grad():
            for prompt_batch in _chunks(list(prompts), batch_size):
                encoded = tokenizer(list(prompt_batch), padding="longest", add_special_tokens=False, return_tensors="pt").to(device)
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
                transition = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)
                predictions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                for prediction, token_ids, token_logps in zip(predictions, completion_ids.tolist(), transition.detach().cpu().tolist()):
                    eos = tokenizer.eos_token_id in token_ids
                    length = token_ids.index(tokenizer.eos_token_id) + 1 if eos else len(token_ids)
                    used = [float(value) for value in token_logps[:length]]
                    rows.append({
                        "prediction": prediction,
                        "eos_emitted": eos,
                        "generated_token_count": length,
                        "hit_max_new_tokens": not eos and length >= int(max_new_tokens),
                        "model_mean_log_probability": math.fsum(used) / len(used) if used else -math.inf,
                    })
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    return rows


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _smatch(prediction: str, reference: str) -> float:
    try:
        pred = qry_wordlist_2_graph(shifted_wordlist(parse_action(prediction)))
        target = qry_wordlist_2_graph(shifted_wordlist(parse_action(reference)))
        if pred is None or target is None:
            return 0.0
        return float(get_smatch_score(pred, target))
    except Exception:
        return 0.0


def _load_split(config, split: str, limit: int | None = None):
    if config.condition != "pattern":
        raise ValueError("Pattern runner requires experiment condition=pattern")
    manifest_path = config.sampling_manifest_path
    if _sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Sampling manifest SHA256 mismatch")
    manifest = _read_json(manifest_path)
    artifact = manifest["artifacts"]["base"][split]
    if str(artifact["sha256"]) != EXPECTED_ARTIFACT_SHA256[split]:
        raise ValueError(f"Frozen {split} artifact SHA256 mismatch")
    datasets, _, _ = _datasets(config, [split], train_variant="base")
    dataset = datasets[split]
    expected_count = 1664
    if len(dataset) != expected_count:
        raise ValueError(f"Frozen {split} count mismatch: {len(dataset)} != {expected_count}")
    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")
    examples = []
    for index, item in enumerate(dataset):
        target = str(item["target"])
        condition = condition_value_from_target("pattern", target)
        pattern_id = int(item["pattern_id"])
        examples.append({
            "record_id": str(item.get("record_id", f"{split}:{index}")),
            "source": str(item["source"]),
            "target": target,
            "condition": condition,
            "pattern_id": pattern_id,
            "pattern_abbr": str(patterns.loc[pattern_id, "pattern_abbr"]),
        })
    if limit is not None:
        examples = examples[:int(limit)]
    artifact_path = (manifest_path.parent / str(artifact["path"])).resolve()
    return examples, {
        "split": split,
        "count": len(examples),
        "full_count": expected_count,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "limited": limit is not None,
    }


def _verify_checkpoint(*, checkpoint: Path, pointer_path: Path, expected_tree: str, config):
    pointer = _read_json(pointer_path)
    if pointer.get("kind") != "phase_d_parent" or pointer.get("stage") != "conditional":
        raise ValueError("Invalid Phase D parent pointer")
    if str(pointer.get("checkpoint")) != "conditional-epoch-45":
        raise ValueError("Pattern verifier requires conditional-epoch-45")
    resolved = checkpoint.resolve()
    expected = (pointer_path.parent / "conditional-epoch-45").resolve()
    if resolved != expected or checkpoint.name != "phase-d-parent":
        raise ValueError("Checkpoint/pointer does not resolve exactly through phase-d-parent")
    actual_tree = checkpoint_tree_sha256(resolved)
    if actual_tree != expected_tree:
        raise ValueError("Checkpoint tree SHA256 mismatch")
    loaded = load_reproduction_checkpoint(
        resolved,
        mode="test",
        expected_stage="conditional",
        expected_condition="pattern",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=EXPECTED_MANIFEST_SHA256,
    )
    if int(loaded.metadata.get("stage_epoch", -1)) != 45:
        raise ValueError("Checkpoint metadata epoch is not 45")
    return loaded, {
        "pointer_path": str(pointer_path),
        "pointer_sha256": _sha256_file(pointer_path),
        "path": str(checkpoint),
        "resolved_path": str(resolved),
        "resolved_name": resolved.name,
        "tree_sha256": actual_tree,
        "metadata_sha256": _sha256_file(resolved / "metadata.json"),
        "stage": loaded.metadata["stage"],
        "condition": loaded.metadata["condition"],
        "stage_epoch": int(loaded.metadata["stage_epoch"]),
    }


def _evaluate_examples(*, name: str, examples: Sequence[Mapping[str, Any]], model, tokenizer, device,
                       graph_sampler, seed: int, batch_size: int, max_new_tokens: int,
                       max_k: int, condition_delimiter: str, log: RunLog):
    prompts = [build_generation_prompt(str(item["source"]), ConditionSpec("pattern", str(item["condition"])), tokenizer,
                                       condition_delimiter=condition_delimiter) for item in examples]
    observations = [_observation(str(item["source"])) for item in examples]
    executor = QueryExecutor(graph_sampler)
    rows = [{
        **dict(item),
        "source_sha256": _text_sha256(str(item["source"])),
        "target_sha256": _text_sha256(str(item["target"])),
        "condition_sha256": _text_sha256(str(item["condition"])),
        "observation_sha256": denotation_sha256(observation),
        "observation_cardinality": len(observation),
        "greedy": None,
        "sample_candidates": [],
        "selections": {},
    } for item, observation in zip(examples, observations)]
    costs: dict[str, Any] = {}

    seed_everything(derive_seed(seed, name, "greedy"))
    _synchronize(); started = time.perf_counter()
    generated = _generate_pass(model=model, tokenizer=tokenizer, prompts=prompts, device=device,
                               max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=False)
    _synchronize(); generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for row, item, observation, candidate in zip(rows, examples, observations, generated):
        row["greedy"] = {**candidate, **audit_candidate(str(candidate["prediction"]), condition=str(item["condition"]),
                                                        observation=observation, executor=executor)}
    verification_seconds = time.perf_counter() - started
    costs["greedy"] = {"generation_seconds": generation_seconds, "verification_seconds": verification_seconds,
                       "graph_executions": sum(int(row["greedy"]["graph_executions"]) for row in rows),
                       "generated_sequences": len(rows),
                       "generated_tokens": sum(int(row["greedy"]["generated_token_count"]) for row in rows)}
    log.write(f"{name}: greedy generation complete ({generation_seconds:.1f}s)")

    cumulative_generation = cumulative_verification = 0.0
    for candidate_index in range(max_k):
        seed_everything(derive_seed(seed, name, "sample", candidate_index))
        _synchronize(); started = time.perf_counter()
        generated = _generate_pass(model=model, tokenizer=tokenizer, prompts=prompts, device=device,
                                   max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=True)
        _synchronize(); cumulative_generation += time.perf_counter() - started
        started = time.perf_counter()
        for row, item, observation, candidate in zip(rows, examples, observations, generated):
            row["sample_candidates"].append({"candidate_index": candidate_index, **candidate,
                **audit_candidate(str(candidate["prediction"]), condition=str(item["condition"]),
                                  observation=observation, executor=executor)})
        cumulative_verification += time.perf_counter() - started
        k = candidate_index + 1
        if k in P1_K_VALUES or k == max_k:
            costs[f"sample_k{k}"] = {
                "generation_seconds": cumulative_generation,
                "verification_seconds": cumulative_verification,
                "graph_executions": sum(int(candidate["graph_executions"]) for row in rows for candidate in row["sample_candidates"][:k]),
                "generated_sequences": len(rows) * k,
                "generated_tokens": sum(int(candidate["generated_token_count"]) for row in rows for candidate in row["sample_candidates"][:k]),
            }
            log.write(f"{name}: nested sample K={k} complete ({cumulative_generation:.1f}s generation cumulative)")

    for row in rows:
        for k in [value for value in P1_K_VALUES if value <= max_k]:
            index, reason = select_candidate(row["sample_candidates"][:k], "pattern_aware")
            row["selections"][f"sample_k{k}"] = {"selector": "pattern_aware", "selected_index": index, "selection_reason": reason}
        for selector in SELECTORS:
            index, reason = select_candidate(row["sample_candidates"][:P2_K], selector)
            row["selections"][selector] = {"selector": selector, "selected_index": index, "selection_reason": reason}

    # Reference hypotheses are first introduced here, after every selector decision is frozen.
    started = time.perf_counter()
    seed_everything(derive_seed(seed, name, "smatch_evaluation_only"))
    for row in rows:
        reference = str(row["target"])
        row["greedy"]["smatch"] = _smatch(str(row["greedy"]["prediction"]), reference)
        for candidate in row["sample_candidates"]:
            candidate["smatch"] = _smatch(str(candidate["prediction"]), reference)
    evaluation_seconds = time.perf_counter() - started
    for cost in costs.values():
        cost["evaluation_seconds"] = evaluation_seconds
        cost["wall_seconds"] = float(cost["generation_seconds"]) + float(cost["verification_seconds"]) + evaluation_seconds
    log.write(f"{name}: evaluation-only SMATCH complete ({evaluation_seconds:.1f}s)")
    return rows, costs


def _selected(row: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    if mode == "greedy":
        return row["greedy"]
    index = int(row["selections"][mode]["selected_index"])
    return row["sample_candidates"][index]


def _mode_summary(rows: Sequence[Mapping[str, Any]], mode: str, cost: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = [_selected(row, mode) for row in rows]
    result = {
        "count": len(rows),
        "jaccard": float(np.mean([item["jaccard"] for item in selected])),
        "dice": float(np.mean([item["dice"] for item in selected])),
        "overlap": float(np.mean([item["overlap"] for item in selected])),
        "semantic_average": float(np.mean([item["semantic_average"] for item in selected])),
        "exact_rate": float(np.mean([item["exact"] for item in selected])),
        "pattern_accuracy": float(np.mean([item["pattern_match"] for item in selected])),
        "smatch": float(np.mean([item["smatch"] for item in selected])),
        "parse_rate": float(np.mean([item["parse_ok"] for item in selected])),
        "eos_rate": float(np.mean([item["eos_emitted"] for item in selected])),
        "max_length_rate": float(np.mean([item["hit_max_new_tokens"] for item in selected])),
        "selected_mean_generated_tokens": float(np.mean([item["generated_token_count"] for item in selected])),
    }
    if mode.startswith("sample_k"):
        k = int(mode.removeprefix("sample_k"))
    elif mode in SELECTORS:
        k = P2_K
    else:
        k = 1
    result["mean_unique_ast"] = float(np.mean([
        len({candidate["query_sha256"] for candidate in row["sample_candidates"][:k] if candidate["query_sha256"]})
        if mode != "greedy" else int(bool(row["greedy"]["query_sha256"])) for row in rows
    ]))
    if cost is not None:
        result["cost"] = {**dict(cost), "mean_generated_tokens_per_candidate":
                          float(cost["generated_tokens"]) / int(cost["generated_sequences"])}
    return result


def _summaries(rows, costs) -> dict[str, Any]:
    result = {mode: _mode_summary(rows, mode, cost) for mode, cost in costs.items()}
    k4_cost = costs["sample_k4"]
    for selector in SELECTORS:
        result[selector] = _mode_summary(rows, selector, k4_cost)
    return result


def _coverage(rows: Sequence[Mapping[str, Any]], k: int = 4) -> dict[str, float]:
    return {
        "any_exact": float(np.mean([any(candidate["exact"] for candidate in row["sample_candidates"][:k]) for row in rows])),
        "any_pattern_match": float(np.mean([any(candidate["pattern_match"] for candidate in row["sample_candidates"][:k]) for row in rows])),
        "any_exact_and_pattern_match": float(np.mean([any(candidate["exact"] and candidate["pattern_match"] for candidate in row["sample_candidates"][:k]) for row in rows])),
    }


def _differences(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, float]:
    metrics = ("jaccard", "dice", "overlap", "semantic_average", "exact", "pattern_match", "smatch", "parse_ok", "eos_emitted")
    return {metric: float(np.mean([float(_selected(row, left)[metric]) - float(_selected(row, right)[metric]) for row in rows]))
            for metric in metrics}


def _paired_wtl(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, int]:
    values = [float(_selected(row, left)["semantic_average"]) - float(_selected(row, right)["semantic_average"]) for row in rows]
    return {"win": sum(value > 0 for value in values), "tie": sum(value == 0 for value in values), "loss": sum(value < 0 for value in values)}


def _bootstrap(rows: Sequence[Mapping[str, Any]], left: str, right: str, metrics: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray([[float(_selected(row, left)[metric]) - float(_selected(row, right)[metric]) for metric in metrics] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty((BOOTSTRAP_REPLICATES, len(metrics)), dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        size = min(250, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(rows), size=(size, len(rows)))
        means[start:start + size] = matrix[indices].mean(axis=1)
    return {metric: {"mean_delta": float(matrix[:, offset].mean()),
                     "ci95": [float(np.quantile(means[:, offset], 0.025)), float(np.quantile(means[:, offset], 0.975))]}
            for offset, metric in enumerate(metrics)}


def _by_pattern(rows: Sequence[Mapping[str, Any]], modes: Sequence[str]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['pattern_id']}:{row['pattern_abbr']}:{row['condition']}"] .append(row)
    return {pattern: {mode: _mode_summary(group, mode) for mode in modes} for pattern, group in sorted(grouped.items())}


def p1_gate(summaries: Mapping[str, Any], *, lineage_complete: bool = True) -> dict[str, Any]:
    greedy, pattern, exact = summaries["greedy"], summaries["pattern_aware"], summaries["exact_semantic"]
    checks = {
        "semantic_non_decrease": pattern["semantic_average"] >= greedy["semantic_average"],
        "exact_non_decrease": pattern["exact_rate"] >= greedy["exact_rate"],
        "pattern_accuracy_tolerance": pattern["pattern_accuracy"] >= greedy["pattern_accuracy"] - 0.005,
        "parse_tolerance": pattern["parse_rate"] >= greedy["parse_rate"] - 0.005,
        "eos_tolerance": pattern["eos_rate"] >= greedy["eos_rate"] - 0.005,
        "pattern_vs_exact_pattern_non_decrease": pattern["pattern_accuracy"] >= exact["pattern_accuracy"],
        "pattern_vs_exact_semantic_tolerance": pattern["semantic_average"] >= exact["semantic_average"] - 0.005,
        "k4_cost_complete": all(key in pattern["cost"] and pattern["cost"][key] is not None for key in
                                ("wall_seconds", "generated_sequences", "generated_tokens", "graph_executions")),
        "selection_reference_free": True,
        "lineage_and_hashes_complete": bool(lineage_complete),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "thresholds": {"pattern_parse_eos_tolerance": 0.005, "pattern_vs_exact_semantic_tolerance": 0.005}}


def _method_contract(*, seed: int, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "implementation_version": METHOD_VERSION,
        "condition_kind": "pattern",
        "generator_frozen": True,
        "seed": int(seed),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sample_stream": "derive_seed(seed,dataset,sample,candidate_index)",
        "temperature": 1.0, "top_k": 0, "top_p": 1.0,
        "p1_k_values": list(P1_K_VALUES), "p2_k": P2_K,
        "batch_size": int(batch_size), "max_new_tokens": int(max_new_tokens),
        "selection_rule": list(SELECTION_RULE), "selectors": list(SELECTORS),
        "selection_inputs": ["observation", "observable_graph", "pattern_condition", "candidate"],
        "reference_hypothesis_available_to_selection": False,
    }


def _prepare_output(output_dir: Path, kind: str, code: Mapping[str, Any], *, smoke: bool):
    if code["dirty"]:
        raise ValueError("Pattern verifier decoding requires a clean Git worktree")
    output_dir.mkdir(parents=True, exist_ok=False)
    log = RunLog(output_dir / "run.log")
    control = {"pid": os.getpid(), "hostname": socket.gethostname(), "platform": platform.platform(),
               "started_at": _utc_now(), "kind": kind, "smoke": smoke, "code": dict(code)}
    _write_json(output_dir / "control.json", control)
    status = output_dir / "status.json"
    _write_json(status, {"schema_version": SCHEMA_VERSION, "kind": kind + "_status", "status": "running", **control})
    return status, log


def _analysis(rows, summaries) -> dict[str, Any]:
    return {
        "coverage_k4": _coverage(rows, 4),
        "comparisons": {
            "pattern_aware_minus_greedy": _differences(rows, "pattern_aware", "greedy"),
            "pattern_aware_minus_first_sample": _differences(rows, "pattern_aware", "first_sample"),
            "pattern_aware_minus_exact_semantic": _differences(rows, "pattern_aware", "exact_semantic"),
            "pattern_aware_minus_likelihood_only": _differences(rows, "pattern_aware", "likelihood_only"),
        },
        "paired_win_tie_loss_semantic_average": _paired_wtl(rows, "pattern_aware", "greedy"),
        "bootstrap": {
            "pattern_aware_minus_greedy": _bootstrap(rows, "pattern_aware", "greedy", ("semantic_average", "exact", "pattern_match", "smatch")),
            "pattern_aware_minus_exact_semantic": _bootstrap(rows, "pattern_aware", "exact_semantic", ("semantic_average", "pattern_match")),
            "pattern_aware_minus_likelihood_only": _bootstrap(rows, "pattern_aware", "likelihood_only", ("semantic_average", "pattern_match")),
        },
        "by_pattern": _by_pattern(rows, ("greedy", "pattern_aware", "exact_semantic", "likelihood_only")),
    }


def run_p1(args: argparse.Namespace) -> Path:
    code = _git_state(); output_dir = Path(args.output_dir).expanduser().resolve(); smoke = args.limit is not None
    status_path, log = _prepare_output(output_dir, "pattern_verifier_best_of_n_p1", code, smoke=smoke)
    try:
        total_started = time.perf_counter(); log.write("P1 preflight started")
        config = load_experiment_config(args.experiment_config)
        examples, data_contract = _load_split(config, "valid", args.limit)
        loaded, checkpoint_contract = _verify_checkpoint(
            checkpoint=Path(args.checkpoint).expanduser().absolute(),
            pointer_path=Path(args.checkpoint_pointer).expanduser().resolve(),
            expected_tree=str(args.checkpoint_tree_sha256), config=config)
        device = _require_cuda("pattern verifier P1")
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
        model, tokenizer = loaded.model.to(device), loaded.tokenizer
        max_new_tokens = int(config.raw["generation"]["max_new_tokens"])
        rows, costs = _evaluate_examples(name="pattern_validation", examples=examples, model=model, tokenizer=tokenizer,
            device=device, graph_sampler=_graph_samplers(config)["valid"], seed=int(args.seed),
            batch_size=int(args.batch_size), max_new_tokens=max_new_tokens, max_k=max(P1_K_VALUES),
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]), log=log)
        summaries = _summaries(rows, costs)
        artifacts = {"records": _write_jsonl(output_dir / "records.jsonl", rows)}
        analysis = _analysis(rows, summaries)
        gate = {"passed": False, "not_applicable": "smoke run"} if smoke else p1_gate(summaries, lineage_complete=True)
        report = {
            "schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p1",
            "status": "smoke_completed" if smoke else ("passed" if gate["passed"] else "failed_gate"),
            "completed_at": _utc_now(), "code": code,
            "command": [sys.executable, "-m", __name__, "p1", *sys.argv[2:]],
            "method": _method_contract(seed=int(args.seed), batch_size=int(args.batch_size), max_new_tokens=max_new_tokens),
            "inputs": {"experiment_config": str(config.source_path), "experiment_config_sha256": _sha256_file(config.source_path),
                       "config_semantic_hash": config.semantic_hash, "data_hash": config.data_hash, "kg_hash": config.kg_hash,
                       "sampling_manifest_sha256": EXPECTED_MANIFEST_SHA256, "checkpoint": checkpoint_contract,
                       "validation": data_contract},
            "gpu": {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__},
            "summaries": summaries, "analysis": analysis, "decision_gate": gate, "artifacts": artifacts,
            "total_wall_seconds": time.perf_counter() - total_started,
            "limitations": ["WN18RR pattern condition only", "frozen decoding comparison; generator parameters unchanged",
                            "observable-graph exactness is not full-KG logical equivalence", "single seed and one dataset"],
        }
        report_path = output_dir / "summary.json"; _write_json(report_path, report)
        _write_json(status_path, {"schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p1_status",
            "status": report["status"], "completed_at": _utc_now(), "code": code,
            "summary_sha256": _sha256_file(report_path), "records_sha256": artifacts["records"]["sha256"]})
        log.write(f"P1 finished with status={report['status']}")
        return report_path
    except BaseException as exc:
        _write_json(status_path, {"schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p1_status",
            "status": "failed", "failed_at": _utc_now(), "code": code, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}); log.write(f"P1 failed: {type(exc).__name__}: {exc}"); raise


def _verify_p1(path: Path, code: Mapping[str, Any]) -> dict[str, Any]:
    p1 = _read_json(path)
    if p1.get("kind") != "pattern_verifier_best_of_n_p1" or p1.get("status") != "passed":
        raise ValueError("P2 requires a passed formal P1")
    if p1.get("decision_gate", {}).get("passed") is not True or p1.get("code") != dict(code):
        raise ValueError("P2 requires the exact clean P1 Git SHA and passed gate")
    method = p1["method"]
    if method != _method_contract(seed=method["seed"], batch_size=method["batch_size"], max_new_tokens=method["max_new_tokens"]):
        raise ValueError("P1 method contract changed")
    record_path = (path.parent / p1["artifacts"]["records"]["path"]).resolve()
    if _sha256_file(record_path) != p1["artifacts"]["records"]["sha256"]:
        raise ValueError("P1 records SHA256 mismatch")
    return p1


def _claim_p2_once(p1_path: Path, output_dir: Path) -> Path:
    marker = p1_path.parent / "p2-consumed.json"
    payload = json.dumps({"claimed_at": _utc_now(), "p1_summary_sha256": _sha256_file(p1_path),
                          "output_dir": str(output_dir)}, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return marker


def run_p2(args: argparse.Namespace) -> Path:
    code = _git_state(); output_dir = Path(args.output_dir).expanduser().resolve()
    status_path, log = _prepare_output(output_dir, "pattern_verifier_best_of_n_p2", code, smoke=False)
    try:
        total_started = time.perf_counter(); p1_path = Path(args.p1_summary).expanduser().resolve(); p1 = _verify_p1(p1_path, code)
        config_path = Path(p1["inputs"]["experiment_config"]).expanduser().resolve()
        if _sha256_file(config_path) != p1["inputs"]["experiment_config_sha256"]:
            raise ValueError("P1 config SHA256 mismatch")
        config = load_experiment_config(config_path); method = p1["method"]; checkpoint = p1["inputs"]["checkpoint"]
        loaded, verified_checkpoint = _verify_checkpoint(checkpoint=Path(checkpoint["path"]),
            pointer_path=Path(checkpoint["pointer_path"]), expected_tree=checkpoint["tree_sha256"], config=config)
        if verified_checkpoint != checkpoint:
            raise ValueError("P2 checkpoint lineage differs from P1")
        marker = _claim_p2_once(p1_path, output_dir)
        examples, data_contract = _load_split(config, "test", None)
        device = _require_cuda("pattern verifier P2")
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True); torch.use_deterministic_algorithms(True, warn_only=False)
        rows, costs = _evaluate_examples(name="pattern_historical_test", examples=examples, model=loaded.model.to(device),
            tokenizer=loaded.tokenizer, device=device, graph_sampler=_graph_samplers(config)["test"], seed=int(method["seed"]),
            batch_size=int(method["batch_size"]), max_new_tokens=int(method["max_new_tokens"]), max_k=P2_K,
            condition_delimiter=str(config.raw["tokenizer"]["condition_delimiter"]), log=log)
        summaries = _summaries(rows, costs); artifact = _write_jsonl(output_dir / "records.jsonl", rows)
        report = {
            "schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p2", "status": "completed",
            "completed_at": _utc_now(), "code": code,
            "command": [sys.executable, "-m", __name__, "p2", *sys.argv[2:]], "method": method,
            "p1": {"path": str(p1_path), "sha256": _sha256_file(p1_path), "records_sha256": p1["artifacts"]["records"]["sha256"],
                   "gate_passed": True, "consumption_marker": str(marker), "consumption_marker_sha256": _sha256_file(marker)},
            "inputs": {"checkpoint": verified_checkpoint, "historical_test": data_contract},
            "gpu": {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "torch": torch.__version__},
            "summaries": summaries, "analysis": _analysis(rows, summaries), "artifact": artifact,
            "total_wall_seconds": time.perf_counter() - total_started,
            "historical_test_previously_accessed": True, "post_evaluation_tuning_permitted": False,
            "limitations": p1["limitations"],
        }
        report_path = output_dir / "summary.json"; _write_json(report_path, report)
        _write_json(status_path, {"schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p2_status",
            "status": "completed", "completed_at": _utc_now(), "code": code,
            "summary_sha256": _sha256_file(report_path), "records_sha256": artifact["sha256"]})
        log.write("P2 completed"); return report_path
    except BaseException as exc:
        _write_json(status_path, {"schema_version": SCHEMA_VERSION, "kind": "pattern_verifier_best_of_n_p2_status",
            "status": "failed", "failed_at": _utc_now(), "code": code, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}); log.write(f"P2 failed: {type(exc).__name__}: {exc}"); raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("p1")
    p1.add_argument("--experiment-config", required=True); p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--checkpoint-pointer", required=True); p1.add_argument("--checkpoint-tree-sha256", required=True)
    p1.add_argument("--output-dir", required=True); p1.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p1.add_argument("--batch-size", type=int, default=32); p1.add_argument("--limit", type=int)
    p2 = sub.add_parser("p2"); p2.add_argument("--p1-summary", required=True); p2.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0 or getattr(args, "limit", 1) is not None and getattr(args, "limit", 1) <= 0:
        raise ValueError("batch size and limit must be positive")
    path = run_p1(args) if args.stage == "p1" else run_p2(args); print(path); return 0


if __name__ == "__main__":
    raise SystemExit(main())

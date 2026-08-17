"""Counterfactual condition-response audit for specific-relation SFT models.

For each frozen validation example, the audit compares the assigned in-target
relation with a matched relation that is absent from the reference hypothesis.
It measures both reference likelihood and greedy behavioral response without
claiming that the absent-condition prompt has a unique supervised target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as functional

from akgr.abduction_model.experiment_runner import _file_sha256, _graph_samplers, _prompt_length
from akgr.evaluation import scoring_input_act_batch_condition
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.sc_idc import canonical_query_sha256, denotation_sha256, parse_action
from akgr.reproduction.sc_idc_grpo_evaluate import _load_validation_contract
from akgr.reproduction.sc_idc_phase2_data import checkpoint_tree_sha256
from akgr.reproduction.sc_idc_phase2_reward import StaticRelationMatcher
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.tokenizer import (
    build_generation_prompt,
    build_prompt,
    unique_condition_values_from_target,
)
from akgr.utils.load_util import load_reproduction_checkpoint


SCHEMA_VERSION = 1
SEMANTIC_METRICS = ("jaccard", "dice", "overlap")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    return {"path": path.name, "count": count, "sha256": _file_sha256(path)}


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _rate(values: Iterable[bool]) -> float | None:
    materialized = [float(bool(value)) for value in values]
    return _mean(materialized)


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _counterfactual_condition(
    *, target: str, assigned: str, matcher: StaticRelationMatcher
) -> tuple[str, dict[str, Any]]:
    target_values = set(unique_condition_values_from_target("specific_relation", target))
    for candidate in matcher.ranked(int(assigned)):
        token = str(int(candidate["token"]))
        if token not in target_values:
            return token, dict(candidate)
    raise ValueError(f"No absent matched relation is available for target {target!r}")


def _generate(
    *, model, tokenizer, prompts: Sequence[str], device, max_new_tokens: int, batch_size: int
) -> tuple[list[str], list[dict[str, Any]]]:
    predictions: list[str] = []
    metadata: list[dict[str, Any]] = []
    old_padding_side = tokenizer.padding_side
    model.eval()
    try:
        tokenizer.padding_side = "left"
        with torch.no_grad():
            for prompt_batch in _chunks(list(prompts), batch_size):
                encoded = tokenizer(
                    prompt_batch,
                    padding="longest",
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    do_sample=False,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                completion_ids = generated[:, encoded.input_ids.shape[1]:]
                predictions.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
                for token_ids in completion_ids.tolist():
                    eos_emitted = tokenizer.eos_token_id in token_ids
                    length = (
                        token_ids.index(tokenizer.eos_token_id) + 1
                        if eos_emitted else len(token_ids)
                    )
                    metadata.append({
                        "eos_emitted": eos_emitted,
                        "generated_token_count": length,
                        "hit_max_new_tokens": not eos_emitted and length >= int(max_new_tokens),
                    })
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    return predictions, metadata


def _reference_mean_log_prob(
    *, model, tokenizer, prompts: Sequence[str], targets: Sequence[str], device,
    max_length: int, batch_size: int,
) -> list[float]:
    results: list[float] = []
    old_padding_side = tokenizer.padding_side
    model.eval()
    try:
        tokenizer.padding_side = "right"
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                prompt_batch = list(prompts[start:start + batch_size])
                target_batch = list(targets[start:start + batch_size])
                pair = tokenizer(
                    prompt_batch,
                    target_batch,
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                prompt_mask = tokenizer(
                    prompt_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=pair.input_ids.shape[-1],
                    return_tensors="pt",
                ).attention_mask.to(device)
                labels = pair.input_ids.clone()
                labels[prompt_mask == 1] = -100
                labels[labels == tokenizer.pad_token_id] = -100
                logits = model(
                    input_ids=pair.input_ids, attention_mask=pair.attention_mask
                ).logits
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                losses = functional.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]),
                    shifted_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view(shifted_labels.shape)
                counts = shifted_labels.ne(-100).sum(dim=1)
                if torch.any(counts == 0):
                    raise RuntimeError("Reference likelihood batch has an empty target")
                results.extend(
                    (-losses.sum(dim=1) / counts).detach().cpu().tolist()
                )
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    return [float(value) for value in results]


def _canonical_or_none(prediction: str) -> str | None:
    try:
        return canonical_query_sha256(parse_action(prediction))
    except Exception:
        return None


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margins = [float(row["reference_log_prob_margin_assigned_minus_absent"]) for row in rows]
    natural_semantic = [
        math.fsum(float(row["assigned_scores"][metric]) for metric in SEMANTIC_METRICS) / 3.0
        for row in rows
    ]
    counter_semantic = [
        math.fsum(float(row["absent_scores"][metric]) for metric in SEMANTIC_METRICS) / 3.0
        for row in rows
    ]
    return {
        "count": len(rows),
        "reference_log_prob": {
            "assigned_mean": _mean([float(row["assigned_reference_mean_log_prob"]) for row in rows]),
            "absent_mean": _mean([float(row["absent_reference_mean_log_prob"]) for row in rows]),
            "assigned_minus_absent_mean": _mean(margins),
            "assigned_preferred_rate": _rate(value > 0.0 for value in margins),
        },
        "greedy_condition_response": {
            "assigned_condition_adherence_rate": _rate(
                float(row["assigned_scores"]["spec"]) > 0.0 for row in rows
            ),
            "absent_condition_adherence_rate": _rate(
                float(row["absent_scores"]["spec"]) > 0.0 for row in rows
            ),
            "original_condition_retained_after_swap_rate": _rate(
                bool(row["original_condition_retained_after_swap"]) for row in rows
            ),
            "counterfactual_only_adherence_rate": _rate(
                float(row["absent_scores"]["spec"]) > 0.0
                and not bool(row["original_condition_retained_after_swap"])
                for row in rows
            ),
            "exact_prediction_change_rate": _rate(
                bool(row["exact_prediction_changed"]) for row in rows
            ),
            "parsed_ast_change_rate": _rate(
                bool(row["parsed_ast_changed"])
                for row in rows
                if row["assigned_query_sha256"] is not None
                and row["absent_query_sha256"] is not None
            ),
            "denotation_change_rate": _rate(
                bool(row["denotation_changed"]) for row in rows
            ),
        },
        "semantic_tradeoff": {
            "assigned_semantic_average": _mean(natural_semantic),
            "absent_semantic_average": _mean(counter_semantic),
            "absent_minus_assigned_semantic_average": _mean(
                [right - left for left, right in zip(natural_semantic, counter_semantic)]
            ),
        },
    }


def _audit_checkpoint(
    *, config, checkpoint: Path, dataset, conditions: Mapping[str, str],
    condition_rows: Mapping[str, Mapping[str, Any]], graph_samplers, output_dir: Path,
    batch_size: int, limit: int | None,
) -> dict[str, Any]:
    loaded = load_reproduction_checkpoint(
        checkpoint,
        mode="test",
        expected_stage="conditional",
        expected_condition="specific_relation",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    model, tokenizer = loaded.model, loaded.tokenizer
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    selected = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    records = [selected[index] for index in range(len(selected))]
    matcher = StaticRelationMatcher(graph_samplers["valid"])
    counterfactuals = [
        _counterfactual_condition(
            target=str(record["target"]),
            assigned=str(conditions[str(record["record_id"])]),
            matcher=matcher,
        )
        for record in records
    ]
    assigned_values = [str(conditions[str(record["record_id"])]) for record in records]
    absent_values = [value for value, _ in counterfactuals]
    delimiter = config.raw["tokenizer"]["condition_delimiter"]
    assigned_raw_prompts = [
        build_prompt(
            str(record["source"]), ConditionSpec("specific_relation", condition), tokenizer,
            condition_delimiter=delimiter,
        )
        for record, condition in zip(records, assigned_values)
    ]
    absent_raw_prompts = [
        build_prompt(
            str(record["source"]), ConditionSpec("specific_relation", condition), tokenizer,
            condition_delimiter=delimiter,
        )
        for record, condition in zip(records, absent_values)
    ]
    assigned_generation_prompts = [
        build_generation_prompt(
            str(record["source"]), ConditionSpec("specific_relation", condition), tokenizer,
            condition_delimiter=delimiter,
        )
        for record, condition in zip(records, assigned_values)
    ]
    absent_generation_prompts = [
        build_generation_prompt(
            str(record["source"]), ConditionSpec("specific_relation", condition), tokenizer,
            condition_delimiter=delimiter,
        )
        for record, condition in zip(records, absent_values)
    ]
    targets = [str(record["target"]) for record in records]
    max_new_tokens = int(config.raw["generation"]["max_new_tokens"])
    max_length = _prompt_length(config) + max_new_tokens
    seed_everything(derive_seed(config.seed, "condition-causality", checkpoint.name))
    assigned_logp = _reference_mean_log_prob(
        model=model, tokenizer=tokenizer, prompts=assigned_raw_prompts, targets=targets,
        device=device, max_length=max_length, batch_size=batch_size,
    )
    absent_logp = _reference_mean_log_prob(
        model=model, tokenizer=tokenizer, prompts=absent_raw_prompts, targets=targets,
        device=device, max_length=max_length, batch_size=batch_size,
    )
    assigned_predictions, assigned_generation = _generate(
        model=model, tokenizer=tokenizer, prompts=assigned_generation_prompts,
        device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    absent_predictions, absent_generation = _generate(
        model=model, tokenizer=tokenizer, prompts=absent_generation_prompts,
        device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    sources = [str(record["source"]) for record in records]
    assigned_scores, assigned_denotations = scoring_input_act_batch_condition(
        assigned_predictions, targets, sources, assigned_values,
        list(SEMANTIC_METRICS) + ["specific"], graph_samplers=graph_samplers,
        searching_split="valid", return_ans=True,
    )
    absent_scores, absent_denotations = scoring_input_act_batch_condition(
        absent_predictions, targets, sources, absent_values,
        list(SEMANTIC_METRICS) + ["specific"], graph_samplers=graph_samplers,
        searching_split="valid", return_ans=True,
    )
    output_rows = []
    for index, record in enumerate(records):
        assigned_sha = _canonical_or_none(assigned_predictions[index])
        absent_sha = _canonical_or_none(absent_predictions[index])
        assigned_denotation_sha = denotation_sha256(assigned_denotations[index])
        absent_denotation_sha = denotation_sha256(absent_denotations[index])
        output_rows.append({
            "record_id": str(record["record_id"]),
            "topology": str(condition_rows[str(record["record_id"])]["topology"]),
            "source": sources[index],
            "target": targets[index],
            "assigned_condition": assigned_values[index],
            "absent_condition": absent_values[index],
            "absent_match": counterfactuals[index][1],
            "assigned_reference_mean_log_prob": assigned_logp[index],
            "absent_reference_mean_log_prob": absent_logp[index],
            "reference_log_prob_margin_assigned_minus_absent": (
                assigned_logp[index] - absent_logp[index]
            ),
            "assigned_prediction": assigned_predictions[index],
            "absent_prediction": absent_predictions[index],
            "assigned_generation": assigned_generation[index],
            "absent_generation": absent_generation[index],
            "assigned_scores": assigned_scores[index],
            "absent_scores": absent_scores[index],
            "original_condition_retained_after_swap": (
                assigned_values[index] in set(absent_predictions[index].split())
            ),
            "exact_prediction_changed": (
                assigned_predictions[index] != absent_predictions[index]
            ),
            "assigned_query_sha256": assigned_sha,
            "absent_query_sha256": absent_sha,
            "parsed_ast_changed": assigned_sha != absent_sha,
            "assigned_denotation_sha256": assigned_denotation_sha,
            "absent_denotation_sha256": absent_denotation_sha,
            "denotation_changed": assigned_denotation_sha != absent_denotation_sha,
        })
    label = checkpoint.name
    artifact = _write_jsonl(output_dir / f"{label}.jsonl", output_rows)
    summary = _summarize(output_rows)
    summary["by_topology"] = {
        topology: _summarize([
            row for row in output_rows if str(row["topology"]) == topology
        ])
        for topology in sorted({str(row["topology"]) for row in output_rows})
    }
    result = {
        "checkpoint": {
            "label": label,
            "path": str(checkpoint),
            "tree_sha256": checkpoint_tree_sha256(checkpoint),
            "stage_epoch": int(loaded.metadata["stage_epoch"]),
            "global_step": int(loaded.metadata["global_step"]),
        },
        "artifact": artifact,
        "summary": summary,
    }
    model.to("cpu")
    del model, loaded
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--phase2-preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the condition causality audit")
    config = load_experiment_config(args.experiment_config)
    preflight = args.phase2_preflight.expanduser().resolve()
    dataset, conditions, condition_rows, validation_contract = _load_validation_contract(
        config, preflight
    )
    graph_samplers = _graph_samplers(config)
    output_dir = args.output_dir.expanduser().resolve()
    results = [
        _audit_checkpoint(
            config=config,
            checkpoint=checkpoint.expanduser().resolve(),
            dataset=dataset,
            conditions=conditions,
            condition_rows=condition_rows,
            graph_samplers=graph_samplers,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            limit=args.limit,
        )
        for checkpoint in args.checkpoint
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sc_idc_condition_causality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()),
        },
        "config": {
            "path": str(args.experiment_config.expanduser().resolve()),
            "semantic_hash": config.semantic_hash,
        },
        "validation_contract": validation_contract,
        "counterfactual": {
            "kind": "highest-ranked_static_match_absent_from_reference",
            "graph_split": "valid",
            "interpretation": (
                "Behavioral sensitivity diagnostic; the absent prompt has no unique supervised target."
            ),
        },
        "results": results,
        "final_evaluation_manifest_loaded": False,
    }
    _write_json(output_dir / "summary.json", payload)
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

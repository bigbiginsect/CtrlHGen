"""Run SS-CSC Experiment C and apply the frozen decision gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from akgr.abduction_model.experiment_runner import _graph_samplers, _prompt_length, _require_cuda
from akgr.evaluation import scoring_input_act_batch_condition
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc import (
    QueryExecutor,
    canonical_query_sha256,
    jaccard_score,
    neutralize_value_branches,
    parse_action,
    value_occurrences,
)
from akgr.reproduction.seed import seed_everything
from akgr.reproduction.ss_csc_common import (
    chunks,
    load_pair_artifact,
    mean,
    pair_likelihood_rows,
    read_json,
    sha256_file,
    summarize_pair_likelihood,
)
from akgr.tokenizer import build_generation_prompt
from akgr.reproduction.contracts import ConditionSpec
from akgr.utils.load_util import load_reproduction_checkpoint


REQUIRED_MODELS = {"single_target", "paired_sft", "ss_csc"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return {"path": path.name, "count": len(rows), "sha256": sha256_file(path)}


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate model name {name!r}")
        result[name] = Path(path).expanduser().resolve()
    if set(result) != REQUIRED_MODELS:
        raise ValueError(f"Models must be exactly {sorted(REQUIRED_MODELS)}")
    return result


def _generate(model, tokenizer, prompts, *, device, max_new_tokens: int, batch_size: int):
    predictions, metadata = [], []
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        model.eval()
        with torch.no_grad():
            for prompt_batch in chunks(list(prompts), batch_size):
                encoded = tokenizer(
                    list(prompt_batch), padding="longest", add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                    do_sample=False, max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
                completion_ids = generated[:, encoded.input_ids.shape[1]:]
                predictions.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
                for token_ids in completion_ids.tolist():
                    eos = tokenizer.eos_token_id in token_ids
                    length = token_ids.index(tokenizer.eos_token_id) + 1 if eos else len(token_ids)
                    metadata.append({
                        "eos_emitted": eos, "generated_token_count": length,
                        "hit_max_new_tokens": not eos and length >= int(max_new_tokens),
                    })
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.backend_tokenizer.no_padding()
    return predictions, metadata


def _ast_diagnostic(prediction: str, condition: str, observation, executor: QueryExecutor):
    try:
        query = parse_action(prediction)
        denotation = executor.execute(query)
        occurrences = value_occurrences(query, "specific_relation", int(condition))
        nominal = bool(occurrences)
        branch_supported = False
        nonroot_branch_supported = False
        branch_delta = None
        if nominal:
            neutral, branches = neutralize_value_branches(query, "specific_relation", int(condition))
            branch_delta = jaccard_score(denotation, observation) - jaccard_score(
                executor.execute(neutral), observation
            )
            branch_supported = branch_delta > 0.0
            nonroot_branch_supported = branch_supported and all(
                row["branch_operator"] is not None for row in branches
            )
        return {
            "parse_ok": True, "query_sha256": canonical_query_sha256(query),
            "denotation": sorted(denotation), "nominal_adherence": nominal,
            "branch_marginal_delta": branch_delta, "branch_supported": branch_supported,
            "nonroot_branch_supported": nonroot_branch_supported,
        }
    except Exception as exc:
        return {
            "parse_ok": False, "parse_error": f"{type(exc).__name__}: {exc}",
            "query_sha256": None, "denotation": None, "nominal_adherence": False,
            "branch_marginal_delta": None, "branch_supported": False,
            "nonroot_branch_supported": False,
        }


def _evaluate_model(
    *, model, tokenizer, pairs, config, pilot, device, graph_sampler,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    delimiter = config.raw["tokenizer"]["condition_delimiter"]
    max_length = _prompt_length(config) + int(config.raw["generation"]["max_new_tokens"])
    likelihood = pair_likelihood_rows(
        model=model, tokenizer=tokenizer, pairs=pairs, device=device, delimiter=delimiter,
        max_length=max_length, batch_size=int(pilot["training"]["micro_batch_pairs"]),
        margin=float(pilot["training"]["margin"]),
    )
    prompts = []
    for pair in pairs:
        for side in ("left", "right"):
            prompts.append(build_generation_prompt(
                pair["source"], ConditionSpec("specific_relation", pair[side]["condition_value"]),
                tokenizer, condition_delimiter=delimiter,
            ))
    predictions, generation_metadata = _generate(
        model, tokenizer, prompts, device=device,
        max_new_tokens=int(config.raw["generation"]["max_new_tokens"]),
        batch_size=max(1, int(pilot["training"]["micro_batch_examples"]) // 2),
    )
    executor = QueryExecutor(graph_sampler)
    rows = []
    for index, pair in enumerate(pairs):
        p1, p2 = predictions[2 * index:2 * index + 2]
        m1, m2 = generation_metadata[2 * index:2 * index + 2]
        c1, c2 = pair["left"]["condition_value"], pair["right"]["condition_value"]
        h1, h2 = pair["left"]["target"], pair["right"]["target"]
        scores, _ = scoring_input_act_batch_condition(
            [p1, p2], [h1, h2], [pair["source"], pair["source"]], [c1, c2],
            ["smatch", "jaccard", "dice", "overlap", "specific"],
            graph_samplers={"train": graph_sampler}, searching_split="train", return_failures=True,
        )
        a1 = _ast_diagnostic(p1, c1, pair["observation"], executor)
        a2 = _ast_diagnostic(p2, c2, pair["observation"], executor)
        c1_only = a1["nominal_adherence"] and not bool(value_occurrences(parse_action(p1), "specific_relation", int(c2))) if a1["parse_ok"] else False
        c2_only = a2["nominal_adherence"] and not bool(value_occurrences(parse_action(p2), "specific_relation", int(c1))) if a2["parse_ok"] else False
        denotation_changed = (
            a1["denotation"] != a2["denotation"]
            if a1["denotation"] is not None and a2["denotation"] is not None else None
        )
        semantic_average = sum(
            float(scores[side][metric]) for side in (0, 1) for metric in ("jaccard", "dice", "overlap")
        ) / 6.0
        rows.append({
            **likelihood[index], "observation": pair["observation"],
            "condition_1": c1, "condition_2": c2,
            "prediction_1": p1, "prediction_2": p2,
            "generation_1": m1, "generation_2": m2,
            "scores_1": scores[0], "scores_2": scores[1],
            "ast_1": a1, "ast_2": a2,
            "exact_prediction_changed": p1 != p2,
            "parsed_ast_changed": (
                a1["query_sha256"] != a2["query_sha256"]
                if a1["query_sha256"] is not None and a2["query_sha256"] is not None else None
            ),
            "denotation_changed": denotation_changed,
            "condition_1_only_response": c1_only,
            "condition_2_only_response": c2_only,
            "bilateral_paired_condition_only_response": c1_only and c2_only,
            "greedy_semantic_average": semantic_average,
        })
    likelihood_summary = summarize_pair_likelihood(rows)
    directional = [(row, side) for row in rows for side in (1, 2)]
    summary = {
        **likelihood_summary,
        "greedy": {
            "assigned_adherence_rate": mean([float(row[f"ast_{side}"]["nominal_adherence"]) for row, side in directional]),
            "paired_condition_only_response_rate": mean([
                (float(row["condition_1_only_response"]) + float(row["condition_2_only_response"])) / 2.0
                for row in rows
            ]),
            "bilateral_paired_condition_only_response_rate": mean([float(row["bilateral_paired_condition_only_response"]) for row in rows]),
            "exact_prediction_change_rate": mean([float(row["exact_prediction_changed"]) for row in rows]),
            "parsed_ast_change_rate": mean([float(row["parsed_ast_changed"]) for row in rows if row["parsed_ast_changed"] is not None]),
            "denotation_change_rate": mean([float(row["denotation_changed"]) for row in rows if row["denotation_changed"] is not None]),
            "semantic_average": mean([float(row["greedy_semantic_average"]) for row in rows]),
            "jaccard": mean([(float(row["scores_1"]["jaccard"]) + float(row["scores_2"]["jaccard"])) / 2.0 for row in rows]),
            "dice": mean([(float(row["scores_1"]["dice"]) + float(row["scores_2"]["dice"])) / 2.0 for row in rows]),
            "overlap": mean([(float(row["scores_1"]["overlap"]) + float(row["scores_2"]["overlap"])) / 2.0 for row in rows]),
            "parse_rate": mean([float(row[f"ast_{side}"]["parse_ok"]) for row, side in directional]),
            "eos_rate": mean([float(row[f"generation_{side}"]["eos_emitted"]) for row, side in directional]),
            "branch_supported_controlled_value_rate": mean([float(row[f"ast_{side}"]["branch_supported"]) for row, side in directional]),
            "nonroot_branch_supported_controlled_value_rate": mean([float(row[f"ast_{side}"]["nonroot_branch_supported"]) for row, side in directional]),
        },
    }
    return rows, summary


def _bootstrap_delta(left, right, *, samples: int, seed: int):
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap requires equal non-empty samples")
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(int(samples)):
        estimates.append(math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    estimates.sort()
    return {
        "delta": math.fsum(deltas) / len(deltas),
        "ci95": [estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]],
    }


def run_evaluation(args: argparse.Namespace) -> Path:
    seed_everything(int(args.seed))
    config = load_experiment_config(args.experiment_config)
    pilot_path = Path(args.pilot_config).expanduser().resolve()
    pilot = read_json(pilot_path)
    pair_summary_path = Path(args.pair_summary).expanduser().resolve()
    _, pairs, _ = load_pair_artifact(pair_summary_path, "pair_validation")
    models = _named_paths(args.model)
    selections = _named_paths(args.selection)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = _require_cuda("SS-CSC Experiment C")
    graph_sampler = _graph_samplers(config)["train"]
    all_rows, summaries, artifacts, selection_payloads = {}, {}, {}, {}
    for name in sorted(models):
        selection = read_json(selections[name])
        if Path(selection["selected_checkpoint"]).expanduser().resolve() != models[name]:
            raise ValueError(f"Selection/checkpoint mismatch for {name}")
        loaded = load_reproduction_checkpoint(
            models[name], mode="test", expected_stage=f"ss_csc_{name}",
            expected_condition="specific_relation", expected_config_hash=config.semantic_hash,
            expected_data_manifest_hash=sha256_file(pair_summary_path),
        )
        lineage = loaded.metadata.get("condition_lineage", {})
        if lineage.get("pair_summary_sha256") != sha256_file(pair_summary_path):
            raise ValueError(f"Pair lineage mismatch for {name}")
        if lineage.get("pilot_config_sha256") != sha256_file(pilot_path):
            raise ValueError(f"Pilot-config lineage mismatch for {name}")
        rows, summary = _evaluate_model(
            model=loaded.model.to(device), tokenizer=loaded.tokenizer, pairs=pairs,
            config=config, pilot=pilot, device=device, graph_sampler=graph_sampler,
        )
        all_rows[name] = rows
        summaries[name] = summary
        selection_payloads[name] = selection
        artifacts[name] = _write_jsonl(output / f"{name}.jsonl", rows)
        del loaded
        torch.cuda.empty_cache()

    gates = pilot["decision_gates"]
    bootstrap_samples = int(gates["bootstrap_samples"])
    ss_rows, paired_rows = all_rows["ss_csc"], all_rows["paired_sft"]
    selection_delta = _bootstrap_delta(
        [row["condition_consistent_selection_accuracy"] for row in ss_rows],
        [row["condition_consistent_selection_accuracy"] for row in paired_rows],
        samples=bootstrap_samples, seed=int(args.seed),
    )
    margin_delta = _bootstrap_delta(
        [float(row["symmetric_margin_satisfied"]) for row in ss_rows],
        [float(row["symmetric_margin_satisfied"]) for row in paired_rows],
        samples=bootstrap_samples, seed=int(args.seed) + 1,
    )
    original = {
        name: selection_payloads[name]["selection"]["original_validation"] for name in REQUIRED_MODELS
    }
    best_control_semantic = max(original[name]["semantic_average"] for name in ("single_target", "paired_sft"))
    best_control_parse = max(original[name]["parse_ok"] for name in ("single_target", "paired_sft"))
    best_control_eos = max(original[name]["eos_rate"] for name in ("single_target", "paired_sft"))
    checks = {
        "selection_absolute_gain": selection_delta["delta"] >= float(gates["paired_selection_min_gain_over_paired_sft"]),
        "selection_ci": selection_delta["ci95"][0] > float(gates["paired_bootstrap_ci_lower_must_exceed"]),
        "margin_absolute_gain": margin_delta["delta"] >= float(gates["symmetric_margin_min_gain_over_paired_sft"]),
        "margin_ci": margin_delta["ci95"][0] > float(gates["paired_bootstrap_ci_lower_must_exceed"]),
        "original_semantic": original["ss_csc"]["semantic_average"] >= best_control_semantic - float(gates["original_validation_semantic_tolerance"]),
        "original_parse": original["ss_csc"]["parse_ok"] >= best_control_parse - float(gates["parse_rate_tolerance"]),
        "original_eos": original["ss_csc"]["eos_rate"] >= best_control_eos - float(gates["eos_rate_tolerance"]),
        "greedy_nominal": summaries["ss_csc"]["greedy"]["assigned_adherence_rate"] >= summaries["paired_sft"]["greedy"]["assigned_adherence_rate"] - float(gates["parse_rate_tolerance"]),
    }
    result = {
        "schema_version": 1, "kind": "ss_csc_experiment_c", "status": "passed" if all(checks.values()) else "failed_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
        "inputs": {
            "pair_summary": str(pair_summary_path), "pair_summary_sha256": sha256_file(pair_summary_path),
            "pilot_config": str(pilot_path), "pilot_config_sha256": sha256_file(pilot_path),
            "models": {name: str(path) for name, path in models.items()},
            "sealed_final_evaluation_loaded": False,
        },
        "summaries": summaries, "original_validation": original,
        "comparisons": {
            "ss_csc_minus_paired_sft_selection": selection_delta,
            "ss_csc_minus_paired_sft_symmetric_margin": margin_delta,
        },
        "decision_gate": {"checks": checks, "passed": all(checks.values()), "thresholds": gates},
        "artifacts": artifacts,
    }
    path = output / "summary.json"
    _write_json(path, result)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--pilot-config", required=True)
    parser.add_argument("--pair-summary", required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--selection", action="append", required=True, help="NAME=BEST_JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(run_evaluation(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

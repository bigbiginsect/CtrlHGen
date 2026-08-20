"""Run one update-matched SS-CSC Experiment B branch."""

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

# Must be set before torch can initialize a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from akgr.abduction_model.experiment_runner import (
    _aggregate_evaluation_metrics,
    _datasets,
    _evaluation_records,
    _graph_samplers,
    _loader,
    _prompt_length,
    _require_cuda,
)
from akgr.abduction_model.reproduction import create_sft_optimizer_schedule
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.seed import derive_seed, seed_everything
from akgr.reproduction.sc_idc_phase2_data import (
    checkpoint_tree_sha256,
    load_frozen_condition_manifest,
    verify_parent_import_contract,
    verify_frozen_validation_dataset,
)
from akgr.reproduction.ss_csc_common import (
    condition_prompt,
    load_pair_artifact,
    pair_likelihood_rows,
    read_json,
    reference_mean_log_probs,
    sha256_file,
    summarize_pair_likelihood,
)
from akgr.tokenizer import dynamic_condition_value_from_target
from akgr.utils.load_util import load_reproduction_checkpoint, save_reproduction_checkpoint


BRANCHES = ("single_target", "paired_sft", "ss_csc")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _batch_indices(length: int, size: int, *, seed: int, epoch: int):
    indices = list(range(length))
    random.Random(derive_seed(seed, "ss-csc-order", epoch)).shuffle(indices)
    for start in range(0, length, size):
        yield indices[start:start + size]


def _flat_examples(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": pair["source"],
            "target": pair[side]["target"],
            "condition_value": pair[side]["condition_value"],
            "example_id": f'{pair["pair_id"]}:{side}',
        }
        for pair in pairs for side in ("left", "right")
    ]


def _assigned_loss(model, tokenizer, examples, *, device, delimiter, max_length):
    prompts = [condition_prompt(row["source"], row["condition_value"], tokenizer, delimiter) for row in examples]
    targets = [row["target"] for row in examples]
    return -reference_mean_log_probs(
        model=model, tokenizer=tokenizer, prompts=prompts, targets=targets,
        device=device, max_length=max_length,
    ).mean()


def _contrastive_loss(model, tokenizer, pairs, *, device, delimiter, max_length, margin):
    prompts, targets = [], []
    for pair in pairs:
        source = pair["source"]
        c1, c2 = pair["left"]["condition_value"], pair["right"]["condition_value"]
        h1, h2 = pair["left"]["target"], pair["right"]["target"]
        prompts.extend([
            condition_prompt(source, c1, tokenizer, delimiter),
            condition_prompt(source, c2, tokenizer, delimiter),
            condition_prompt(source, c2, tokenizer, delimiter),
            condition_prompt(source, c1, tokenizer, delimiter),
        ])
        targets.extend([h1, h2, h1, h2])
    scores = reference_mean_log_probs(
        model=model, tokenizer=tokenizer, prompts=prompts, targets=targets,
        device=device, max_length=max_length,
    ).view(len(pairs), 4)
    assigned = -scores[:, :2].mean()
    swap = (
        torch.relu(float(margin) - scores[:, 0] + scores[:, 2])
        + torch.relu(float(margin) - scores[:, 1] + scores[:, 3])
    ).mean()
    return assigned, swap


def _train_epoch(
    *, branch: str, model, tokenizer, rows, optimizer, scheduler, device,
    delimiter: str, max_length: int, batch_size: int, seed: int, epoch: int,
    margin: float, lambda_swap: float,
) -> dict[str, float | int]:
    model.train()
    losses, assigned_losses, swap_losses = [], [], []
    optimizer_steps = 0
    for indices in _batch_indices(len(rows), batch_size, seed=seed, epoch=epoch):
        batch = [rows[index] for index in indices]
        if branch == "single_target":
            batch = [
                {
                    **row,
                    "condition_value": dynamic_condition_value_from_target(
                        "specific_relation", row["target"], seed=int(seed),
                        record_id=str(row["record_id"]), epoch=int(epoch),
                    ),
                }
                for row in batch
            ]
        optimizer.zero_grad(set_to_none=True)
        if branch == "ss_csc":
            assigned, swap = _contrastive_loss(
                model, tokenizer, batch, device=device, delimiter=delimiter,
                max_length=max_length, margin=margin,
            )
            loss = assigned + float(lambda_swap) * swap
        else:
            assigned = _assigned_loss(
                model, tokenizer, batch, device=device, delimiter=delimiter,
                max_length=max_length,
            )
            swap = torch.zeros((), device=device)
            loss = assigned
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {branch} loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        losses.append(float(loss.detach().cpu()))
        assigned_losses.append(float(assigned.detach().cpu()))
        swap_losses.append(float(swap.detach().cpu()))
    return {
        "loss": math.fsum(losses) / len(losses),
        "assigned_loss": math.fsum(assigned_losses) / len(assigned_losses),
        "swap_loss": math.fsum(swap_losses) / len(swap_losses),
        "optimizer_steps": optimizer_steps,
    }


def _original_validation(
    *, model, tokenizer, config, dataset, conditions, graph_samplers, device,
) -> dict[str, Any]:
    batch_size = int(config.raw["training"]["validation"]["batch_size"])
    loader = _loader(dataset, batch_size, config.seed, False)
    loss_total = 0.0
    loss_count = 0
    model.eval()
    with torch.no_grad():
        for sample in loader:
            examples = [
                {
                    "source": str(source), "target": str(target),
                    "condition_value": str(conditions[str(record_id)]),
                }
                for source, target, record_id in zip(
                    sample["source"], sample["target"], sample["record_id"]
                )
            ]
            batch_loss = _assigned_loss(
                model, tokenizer, examples, device=device,
                delimiter=config.raw["tokenizer"]["condition_delimiter"],
                max_length=_prompt_length(config) + int(config.raw["generation"]["max_new_tokens"]),
            )
            loss_total += float(batch_loss.detach().cpu()) * len(examples)
            loss_count += len(examples)
    loss = loss_total / loss_count
    records = _evaluation_records(
        config=config, dataloader=loader, model=model, tokenizer=tokenizer,
        graph_samplers=graph_samplers, device=device, split="valid",
        condition_kind=config.condition, do_sample=False,
        condition_value_by_record_id=conditions,
    )
    metrics = _aggregate_evaluation_metrics(records)
    metrics["semantic_average"] = sum(float(metrics[key]) for key in ("jaccard", "dice", "overlap")) / 3.0
    return {"validation_loss": loss, **metrics}


def run_training(args: argparse.Namespace) -> Path:
    if args.branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    seed_everything(int(args.seed))
    config = load_experiment_config(args.experiment_config)
    pilot_path = Path(args.pilot_config).expanduser().resolve()
    pilot = read_json(pilot_path)
    train_name = "single_target_train" if args.branch == "single_target" else "pair_train"
    pair_summary, train_rows, _ = load_pair_artifact(Path(args.pair_summary), train_name)
    _, validation_pairs, _ = load_pair_artifact(Path(args.pair_summary), "pair_validation")
    if args.branch == "paired_sft":
        train_rows = _flat_examples(train_rows)
    pair_count = int(pair_summary["counts"]["pair_train"])
    expected_examples = 2 * pair_count
    if args.branch == "ss_csc":
        if len(train_rows) != pair_count:
            raise ValueError("SS-CSC pair count mismatch")
        batch_size = int(pilot["training"]["micro_batch_pairs"])
    else:
        if len(train_rows) != expected_examples:
            raise ValueError("Update-matched example count mismatch")
        batch_size = int(pilot["training"]["micro_batch_examples"])
    expected_steps = math.ceil(pair_count / int(pilot["training"]["micro_batch_pairs"]))
    if math.ceil(len(train_rows) / batch_size) != expected_steps:
        raise ValueError("Branches do not have update-matched batches")

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if output_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir} or {checkpoint_dir}")
    output_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    phase2_path = Path(args.phase2_preflight).expanduser().resolve()
    preliminary = read_json(phase2_path)
    parent_import_path = (phase2_path.parent / preliminary["parent_import"]["path"]).resolve()
    parent_import = read_json(parent_import_path)
    parent = Path(args.parent_checkpoint).expanduser().resolve()
    if parent != Path(parent_import["checkpoint"]["path"]).expanduser().resolve():
        raise ValueError("Parent differs from the frozen Phase 2 unconditional parent")
    if preliminary.get("kind") != "sc_idc_specific_relation_sft_preflight" or preliminary.get("status") != "ready":
        raise ValueError("Phase 2 preflight is not a ready frozen input contract")
    parent_ref = preliminary["parent_import"]
    if sha256_file(parent_import_path) != parent_ref["sha256"]:
        raise ValueError("Frozen parent-import hash mismatch")
    verify_parent_import_contract(
        parent_import, target_config=config, checkpoint=parent,
    )
    validation_ref = preliminary["conditions"]["validation"]
    validation_manifest_path = (phase2_path.parent / validation_ref["manifest_path"]).resolve()
    validation_conditions, validation_manifest = load_frozen_condition_manifest(
        validation_manifest_path, target_config=config, purpose="validation",
        consumer="conditional_sft", expected_manifest_sha256=validation_ref["manifest_sha256"],
    )
    validation_artifact_path = (
        validation_manifest_path.parent / validation_manifest["artifact"]["path"]
    ).resolve()
    validation_rows = [json.loads(line) for line in validation_artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validation_contracts = {
        str(row["record_id"]): {
            "condition_value": str(row["condition_value"]),
            "target_sha256": str(row["target_sha256"]),
        }
        for row in validation_rows
    }
    datasets, _, _ = _datasets(config, ["valid"], train_variant="base")
    verify_frozen_validation_dataset(
        datasets["valid"], condition_values=validation_conditions,
        record_contracts=validation_contracts,
    )
    loaded = load_reproduction_checkpoint(
        parent, mode="parent", expected_stage="unconditional",
        expected_condition="unconditional",
        expected_config_hash=parent_import["source"]["config_semantic_hash"],
        expected_data_manifest_hash=parent_import["source"]["sampling_manifest_sha256"],
    )
    device = _require_cuda(f"SS-CSC {args.branch}")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model, tokenizer = loaded.model.to(device), loaded.tokenizer
    training = pilot["training"]
    epochs = int(training["epochs"])
    schedule = create_sft_optimizer_schedule(
        model, learning_rate=float(training["learning_rate"]),
        num_batches=expected_steps, gradient_accumulation_steps=1, epochs=epochs,
        optimizer_config=training["optimizer"], scheduler_config=training["scheduler"],
    )
    graph_samplers = _graph_samplers(config)
    max_length = _prompt_length(config) + int(config.raw["generation"]["max_new_tokens"])
    history_path = output_dir / "history.jsonl"
    global_step = int(loaded.metadata["global_step"])
    selected = None
    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            branch=args.branch, model=model, tokenizer=tokenizer, rows=train_rows,
            optimizer=schedule.optimizer, scheduler=schedule.scheduler, device=device,
            delimiter=config.raw["tokenizer"]["condition_delimiter"], max_length=max_length,
            batch_size=batch_size, seed=int(args.seed), epoch=epoch,
            margin=float(training["margin"]), lambda_swap=float(training["lambda_swap"]),
        )
        if int(train_metrics["optimizer_steps"]) != expected_steps:
            raise RuntimeError("Optimizer-step contract failed")
        global_step += expected_steps
        record = {
            "branch": args.branch, "epoch": epoch, "global_step": global_step,
            "learning_rate": float(schedule.optimizer.param_groups[0]["lr"]),
            "train": train_metrics, "pair_validation": None,
            "original_validation": None, "checkpoint": None,
        }
        due = epoch % int(training["evaluate_every_epochs"]) == 0 or epoch == epochs
        if due:
            likelihood_rows = pair_likelihood_rows(
                model=model, tokenizer=tokenizer, pairs=validation_pairs, device=device,
                delimiter=config.raw["tokenizer"]["condition_delimiter"], max_length=max_length,
                batch_size=max(1, batch_size // 2), margin=float(training["margin"]),
            )
            pair_metrics = summarize_pair_likelihood(likelihood_rows)
            original_metrics = _original_validation(
                model=model, tokenizer=tokenizer, config=config, dataset=datasets["valid"],
                conditions=validation_conditions, graph_samplers=graph_samplers, device=device,
            )
            record["pair_validation"] = pair_metrics
            record["original_validation"] = original_metrics
            checkpoint = save_reproduction_checkpoint(
                checkpoint_dir / f"epoch-{epoch}", model=model, tokenizer=tokenizer,
                stage=f"ss_csc_{args.branch}", stage_epoch=epoch, global_step=global_step,
                condition="specific_relation", experiment_config=config.raw,
                experiment_config_hash=config.semantic_hash, seed=int(args.seed),
                parent_checkpoint=parent, optimizer=schedule.optimizer, scheduler=schedule.scheduler,
                data_manifest_hash=sha256_file(Path(args.pair_summary).expanduser().resolve()),
                condition_lineage={
                    "kind": "ss_csc_paired_supervision", "branch": args.branch,
                    "pair_summary": str(Path(args.pair_summary).expanduser().resolve()),
                    "pair_summary_sha256": sha256_file(Path(args.pair_summary).expanduser().resolve()),
                    "pilot_config_sha256": sha256_file(pilot_path),
                    "parent_tree_sha256": checkpoint_tree_sha256(parent),
                },
            )
            record["checkpoint"] = str(checkpoint)
            health = (
                float(original_metrics["parse_ok"] or 0.0) >= float(config.raw["training"]["validation"]["min_parse_ok"])
                and float(original_metrics["eos_rate"] or 0.0) >= float(config.raw["training"]["validation"]["min_eos_rate"])
            )
            record["health_pass"] = health
            key = (
                float(health), float(pair_metrics["condition_consistent_selection_accuracy"]),
                float(pair_metrics["symmetric_margin_satisfaction_rate"]),
                float(original_metrics["semantic_average"]), -epoch,
            )
            if selected is None or key > selected[0]:
                selected = (key, record.copy())
        _append_jsonl(history_path, record)
    if selected is None or selected[1].get("health_pass") is not True:
        raise RuntimeError(f"{args.branch} produced no healthy selectable checkpoint")
    best = {
        "schema_version": 1, "kind": "ss_csc_branch_selection", "branch": args.branch,
        "selected_checkpoint": selected[1]["checkpoint"],
        "selection": selected[1], "update_matching": {
            "pair_train": pair_count, "assigned_examples_per_epoch": expected_examples,
            "optimizer_steps_per_epoch": expected_steps, "epochs": epochs,
            "total_optimizer_steps": expected_steps * epochs,
        },
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sealed_final_evaluation_loaded": False,
        "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
    }
    path = output_dir / "best.json"
    _write_json(path, best)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--pilot-config", required=True)
    parser.add_argument("--pair-summary", required=True)
    parser.add_argument("--phase2-preflight", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(run_training(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

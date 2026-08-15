"""Run a no-update launch smoke for formal SC-IDC specific-relation SFT."""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from akgr.abduction_model.experiment_runner import (
    _datasets,
    _file_sha256,
    _loader,
    _prompt_length,
    _require_cuda,
)
from akgr.abduction_model.reproduction import create_sft_optimizer_schedule
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.sc_idc_phase2_data import (
    verify_frozen_validation_dataset,
    verify_sft_preflight,
)
from akgr.tokenizer import prepare_batch
from akgr.utils.load_util import load_reproduction_checkpoint


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_launch_preflight(args: argparse.Namespace) -> Path:
    config = load_experiment_config(args.target_config)
    if config.condition != "specific_relation":
        raise ValueError("Launch preflight requires a specific_relation target config")
    report_path = Path(args.output).expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.phase2_data_manifest).expanduser().resolve()
    preliminary = json.loads(summary_path.read_text(encoding="utf-8"))
    parent_import_path = (
        summary_path.parent / preliminary["parent_import"]["path"]
    ).resolve()
    parent_payload = json.loads(parent_import_path.read_text(encoding="utf-8"))
    parent = Path(parent_payload["checkpoint"]["path"]).expanduser().resolve()
    if Path(args.parent_checkpoint).expanduser().resolve() != parent:
        raise ValueError("Requested parent differs from the Phase 2 frozen parent")
    summary, validation_values, lineage, validation_contracts = verify_sft_preflight(
        summary_path, target_config=config, checkpoint=parent
    )

    stage_config = config.raw["training"]["conditional"]
    datasets, _, _ = _datasets(
        config, ["train", "valid"], train_variant=stage_config["data_variant"]
    )
    verify_frozen_validation_dataset(
        datasets["valid"],
        condition_values=validation_values,
        record_contracts=validation_contracts,
    )
    loaded = load_reproduction_checkpoint(
        parent,
        mode="parent",
        expected_stage="unconditional",
        expected_condition="unconditional",
        expected_config_hash=parent_payload["source"]["config_semantic_hash"],
        expected_data_manifest_hash=parent_payload["source"][
            "sampling_manifest_sha256"
        ],
    )
    device = _require_cuda("SC-IDC specific-relation SFT launch preflight")
    model = loaded.model.to(device)
    tokenizer = loaded.tokenizer
    smoke_batch_size = min(int(args.batch_size), len(datasets["train"]), len(datasets["valid"]))
    if smoke_batch_size <= 0:
        raise ValueError("Launch preflight datasets are empty")
    train_loader = _loader(datasets["train"], smoke_batch_size, config.seed, False)
    valid_loader = _loader(datasets["valid"], smoke_batch_size, config.seed, False)
    common = {
        "tokenizer": tokenizer,
        "is_gpt": True,
        "src_len": _prompt_length(config),
        "tgt_len": config.raw["generation"]["max_new_tokens"],
        "is_gen": False,
        "condition": config.condition,
        "condition_delimiter": config.raw["tokenizer"]["condition_delimiter"],
    }
    train_prepare = partial(
        prepare_batch,
        device,
        **common,
        condition_seed=config.seed,
        condition_epoch=1,
    )
    valid_prepare = partial(
        prepare_batch,
        device,
        **common,
        condition_value_by_record_id=validation_values,
    )
    train_batch = train_prepare(next(iter(train_loader)))
    valid_batch = valid_prepare(next(iter(valid_loader)))
    model.eval()
    with torch.no_grad():
        train_loss = model(
            input_ids=train_batch.input_ids,
            attention_mask=train_batch.attention_mask,
            labels=train_batch.labels,
        ).loss
        valid_loss = model(
            input_ids=valid_batch.input_ids,
            attention_mask=valid_batch.attention_mask,
            labels=valid_batch.labels,
        ).loss
    if not torch.isfinite(train_loss) or not torch.isfinite(valid_loss):
        raise FloatingPointError("Launch preflight produced a non-finite loss")

    formal_batch_size = int(stage_config["micro_batch_size"])
    formal_batches = math.ceil(len(datasets["train"]) / formal_batch_size)
    schedule_options = (
        {"warmup_epochs": stage_config["warmup_epochs"]}
        if "warmup_epochs" in stage_config
        else {
            "optimizer_config": stage_config["optimizer"],
            "scheduler_config": stage_config["scheduler"],
        }
    )
    schedule = create_sft_optimizer_schedule(
        model,
        learning_rate=stage_config["learning_rate"],
        num_batches=formal_batches,
        gradient_accumulation_steps=config.raw["training"][
            "gradient_accumulation_steps"
        ],
        epochs=stage_config["epochs"],
        **schedule_options,
    )
    if schedule.optimizer.state:
        raise AssertionError("Imported parent unexpectedly populated the new optimizer state")

    report = {
        "schema_version": 1,
        "kind": "sc_idc_specific_relation_sft_launch_preflight",
        "status": "ready",
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "command": [sys.executable, "-m", __name__, *sys.argv[1:]],
        "target": {
            "config_path": str(config.source_path),
            "config_semantic_hash": config.semantic_hash,
            "data_hash": config.data_hash,
            "kg_hash": config.kg_hash,
        },
        "parent": {
            "path": str(parent),
            "checkpoint_tree_sha256": parent_payload["checkpoint"]["tree_sha256"],
            "stage": loaded.metadata["stage"],
            "condition": loaded.metadata["condition"],
        },
        "phase2_data": {
            "preflight_path": str(summary_path),
            "preflight_sha256": _file_sha256(summary_path),
            "validation_manifest_sha256": lineage["validation_manifest_sha256"],
            "validation_conditions_sha256": lineage["validation_conditions_sha256"],
            "final_evaluation_manifest_loaded": False,
        },
        "data": {
            "variant": stage_config["data_variant"],
            "train_count": len(datasets["train"]),
            "validation_count": len(datasets["valid"]),
            "validation_condition_count": len(validation_values),
        },
        "smoke": {
            "batch_size": smoke_batch_size,
            "train_epoch": 1,
            "train_condition_values": [item.value for item in train_batch.conditions],
            "validation_condition_values": [
                item.value for item in valid_batch.conditions
            ],
            "train_loss": float(train_loss.detach().cpu()),
            "validation_loss": float(valid_loss.detach().cpu()),
            "forward_only": True,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
        },
        "formal_schedule": {
            **schedule.metadata,
            "micro_batch_size": formal_batch_size,
            "optimizer_state_entries_before_training": len(schedule.optimizer.state),
        },
        "formal_command": [
            "bash",
            "scripts/sc-idc/sft-specific-relation.sh",
            str(config.source_path),
            str(parent),
            str(summary_path),
        ],
        "checks": {
            "parent_reload": True,
            "dynamic_unique_value_epoch_1_batch": True,
            "frozen_validation_identity": True,
            "finite_train_forward_loss": True,
            "finite_validation_forward_loss": True,
            "optimizer_scheduler_fresh": True,
            "no_training_update": True,
            "sealed_final_evaluation_not_loaded": True,
        },
    }
    _write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--phase2-data-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(run_launch_preflight(args))


if __name__ == "__main__":
    main()

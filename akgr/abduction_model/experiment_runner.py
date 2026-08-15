"""Strict ``--experiment-config`` execution path for Phase A reproduction."""

from __future__ import annotations

from functools import partial
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from akgr.abduction_model.reproduction import (
    append_jsonl,
    configure_reproduction_logging,
    create_grpo_trainer,
    create_sft_optimizer_schedule,
    epoch_due,
    is_better_validation,
    load_best_checkpoint_record,
    prune_sft_checkpoints,
    sft_validation_loss,
    sft_train_epoch,
    train_grpo,
    write_best_checkpoint_pointer,
)
from akgr.abduction_model.transformer import create_reproduction_transformer
from akgr.dataloader import create_reproduction_dataset
from akgr.evaluation import (
    build_evaluation_record,
    scoring_input_act_batch,
    scoring_input_act_batch_condition,
    write_evaluation_jsonl,
)
from akgr.kgdata import load_kg
from akgr.reproduction.config import load_experiment_config
from akgr.reproduction.contracts import ConditionSpec
from akgr.reproduction.seed import derive_seed, make_generator, seed_everything
from akgr.reproduction.sc_idc_phase2_data import (
    verify_frozen_validation_dataset,
    verify_sft_preflight,
)
from akgr.tokenizer import (
    build_generation_prompt,
    condition_value_from_target,
    create_reproduction_tokenizer,
    prepare_batch,
)
from akgr.utils.load_util import load_reproduction_checkpoint, save_reproduction_checkpoint


def _require_cuda(stage: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for strict reproduction {stage}; CPU fallback is disabled")
    device = torch.device("cuda:0")
    print(f"# DEVICE: {device} ({torch.cuda.get_device_name(0)})")
    return device


def _datasets(config, splits, *, train_variant="base"):
    patterns = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")
    return create_reproduction_dataset(
        experiment_config=config,
        pattern_filtered=patterns,
        splits=splits,
        is_act=True,
        train_variant=train_variant,
    )


def _loader(dataset, batch_size, seed, shuffle):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=make_generator(seed),
    )


def _new_tokenizer(nentity, nrelation):
    return create_reproduction_tokenizer(nentity, nrelation)


def _prompt_length(config):
    return (
        int(config.raw["data"]["max_answers"])
        + int(config.raw["generation"]["max_new_tokens"])
        + 2
    )


def _graph_samplers(config):
    data = config.raw["data"]
    return load_kg(
        config.dataset,
        data_root=config.runtime_paths["data_root"],
        seed=config.seed,
        split_ratios=data["split_ratios"],
        reverse_edges_flag=data["reverse_edges"],
        semantic_hash=config.kg_hash,
        offline=True,
    ).graph_samplers


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_checkpoint_pointer(path, *, stage: str, pointer_stem: str) -> Path:
    """Require a checkpoint to match one healthy, recorded stage pointer."""
    checkpoint = Path(path).expanduser().resolve()
    pointer = checkpoint.parent / f"{pointer_stem}.json"
    if not pointer.is_file():
        raise ValueError(f"Missing {stage} checkpoint record: {pointer}")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    recorded_stage = payload.get("stage")
    if recorded_stage is not None and recorded_stage != stage:
        raise ValueError(
            f"Checkpoint record {pointer} has stage {recorded_stage!r}, expected {stage!r}"
        )
    selected = (checkpoint.parent / payload["checkpoint"]).resolve()
    if checkpoint != selected:
        raise ValueError(
            f"{stage} stage transition requires {pointer_stem} checkpoint "
            f"{selected}, got {checkpoint}"
        )
    validation = payload.get("validation", {})
    if validation.get("health_pass") is not True:
        raise ValueError(f"Recorded {stage} checkpoint did not pass parse/EOS health gates")
    return checkpoint


def _require_selected_healthy_checkpoint(path, *, stage: str) -> Path:
    """Require stage transitions to use the gated, training-selected SFT checkpoint."""
    return _require_checkpoint_pointer(path, stage=stage, pointer_stem=f"{stage}-best")


def _require_phase_d_parent_checkpoint(path) -> Path:
    """Prefer an audited Phase D parent decision without rewriting SFT history."""
    checkpoint = Path(path).expanduser().resolve()
    phase_d_pointer = checkpoint.parent / "phase-d-parent.json"
    if phase_d_pointer.is_file():
        return _require_checkpoint_pointer(
            checkpoint, stage="conditional", pointer_stem="phase-d-parent"
        )
    return _require_selected_healthy_checkpoint(checkpoint, stage="conditional")


def _evaluation_records(
    *, config, dataloader, model, tokenizer, graph_samplers, device, split, condition_kind,
    do_sample: bool, condition_value_by_record_id=None,
):
    model.eval()
    generation = config.raw["generation"]
    records = []
    record_index = 0
    with torch.no_grad():
        for sample in dataloader:
            batch = prepare_batch(
                device, sample, tokenizer, True, _prompt_length(config),
                generation["max_new_tokens"], True, condition_kind,
                condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
                condition_value_by_record_id=condition_value_by_record_id,
            )
            generation_kwargs = {
                "max_new_tokens": int(generation["max_new_tokens"]),
                "do_sample": bool(do_sample),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs.update(
                    top_k=int(generation["top_k"]),
                    top_p=float(generation["top_p"]),
                    temperature=float(generation["temperature"]),
                )
            generated = model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                **generation_kwargs,
            )
            completion_ids = generated[:, batch.input_ids.shape[1]:]
            predictions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            methods = ["smatch", "jaccard", "dice", "overlap"]
            if batch.conditions is None:
                scores, _ = scoring_input_act_batch(
                    predictions, batch.target, batch.source, methods,
                    graph_samplers=graph_samplers, searching_split=split, return_failures=True,
                )
            else:
                condition_values = [item.value for item in batch.conditions]
                condition_method = "specific" if config.condition.startswith("specific_") else "validity"
                scores, _ = scoring_input_act_batch_condition(
                    predictions, batch.target, batch.source, condition_values,
                    methods + [condition_method], graph_samplers=graph_samplers,
                    searching_split=split, return_failures=True,
                )
            for offset, (source, target, prediction, score) in enumerate(
                zip(batch.source, batch.target, predictions, scores)
            ):
                spec = None if batch.conditions is None else batch.conditions[offset]
                token_ids = completion_ids[offset].tolist()
                eos_emitted = tokenizer.eos_token_id in token_ids
                generated_token_count = (
                    token_ids.index(tokenizer.eos_token_id) + 1
                    if eos_emitted else len(token_ids)
                )
                records.append(build_evaluation_record(
                    record_id=(
                        str(sample["record_id"][offset])
                        if condition_value_by_record_id is not None
                        else f"{split}:{record_index}"
                    ), observation=source,
                    reference=target, prediction=prediction, scores=score, condition=spec,
                    eos_emitted=eos_emitted,
                    generated_token_count=generated_token_count,
                    hit_max_new_tokens=(
                        not eos_emitted
                        and generated_token_count >= int(generation["max_new_tokens"])
                    ),
                ))
                record_index += 1
    return records


def _aggregate_evaluation_metrics(records) -> dict[str, float | None]:
    metric_columns = ["jaccard", "dice", "overlap", "condition_accuracy", "smatch", "parse_ok"]
    metrics = {}
    for name in metric_columns:
        values = [float(record[name]) for record in records if record[name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
    for output_name, record_name in (
        ("eos_rate", "eos_emitted"),
        ("max_length_rate", "hit_max_new_tokens"),
    ):
        values = [
            float(record[record_name])
            for record in records
            if record.get(record_name) is not None
        ]
        metrics[output_name] = sum(values) / len(values) if values else None
    lengths = [
        float(record["generated_token_count"])
        for record in records
        if record.get("generated_token_count") is not None
    ]
    metrics["mean_generated_tokens"] = sum(lengths) / len(lengths) if lengths else None
    return metrics


def _write_validation_artifacts(output_root: Path, *, stage: str, stage_epoch: int, records, summary) -> None:
    validation_root = output_root / "validation"
    stem = f"{stage}-epoch-{stage_epoch}"
    write_evaluation_jsonl(validation_root / f"{stem}.jsonl", records)
    (validation_root / f"{stem}.metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _model_and_tokenizer(config, nentity, nrelation, args, stage, phase2_context=None):
    if args.resume_checkpoint:
        loaded = load_reproduction_checkpoint(
            args.resume_checkpoint,
            mode="test",
            expected_stage=stage,
            expected_condition="unconditional" if stage == "unconditional" else config.condition,
            expected_config_hash=config.semantic_hash,
            expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
        )
        return loaded.model, loaded.tokenizer, loaded
    if args.parent_checkpoint:
        if stage != "conditional":
            raise ValueError("--parent-checkpoint is only valid for conditional SFT or GRPO")
        if phase2_context is None:
            _require_selected_healthy_checkpoint(args.parent_checkpoint, stage="unconditional")
            loaded = load_reproduction_checkpoint(
                args.parent_checkpoint,
                mode="parent",
                expected_stage="unconditional",
                expected_condition="unconditional",
                expected_config_hash=config.semantic_hash,
                expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
            )
        else:
            parent_payload = phase2_context["parent_payload"]
            loaded = load_reproduction_checkpoint(
                args.parent_checkpoint,
                mode="parent",
                expected_stage="unconditional",
                expected_condition="unconditional",
                expected_config_hash=parent_payload["source"]["config_semantic_hash"],
                expected_data_manifest_hash=parent_payload["source"][
                    "sampling_manifest_sha256"
                ],
            )
        return loaded.model, loaded.tokenizer, loaded
    tokenizer = _new_tokenizer(nentity, nrelation)
    return create_reproduction_transformer(tokenizer, config.raw["model"]), tokenizer, None


def run_sft(config, args) -> Path:
    if args.stage not in {"unconditional", "conditional"}:
        raise ValueError("SFT requires --stage unconditional or conditional")
    if args.parent_checkpoint and args.resume_checkpoint:
        raise ValueError("--parent-checkpoint and --resume-checkpoint are mutually exclusive")
    if args.stage == "conditional" and not (args.parent_checkpoint or args.resume_checkpoint):
        raise ValueError("Conditional SFT requires --parent-checkpoint or --resume-checkpoint")

    phase2_context = None
    if config.condition == "specific_relation":
        if args.stage != "conditional":
            raise ValueError("The SC-IDC specific-relation config is conditional-SFT only")
        if not args.phase2_data_manifest:
            raise ValueError(
                "SC-IDC specific-relation SFT requires --phase2-data-manifest"
            )
        summary_path = Path(args.phase2_data_manifest).expanduser().resolve()
        preliminary = json.loads(summary_path.read_text(encoding="utf-8"))
        parent_import_path = (
            summary_path.parent / preliminary["parent_import"]["path"]
        ).resolve()
        parent_payload = json.loads(parent_import_path.read_text(encoding="utf-8"))
        frozen_parent = Path(parent_payload["checkpoint"]["path"]).expanduser().resolve()
        if args.parent_checkpoint and Path(args.parent_checkpoint).expanduser().resolve() != frozen_parent:
            raise ValueError("--parent-checkpoint differs from the Phase 2 frozen parent")
        summary, validation_conditions, condition_lineage, validation_contracts = (
            verify_sft_preflight(
                summary_path,
                target_config=config,
                checkpoint=frozen_parent,
            )
        )
        phase2_context = {
            "summary": summary,
            "parent_payload": parent_payload,
            "validation_conditions": validation_conditions,
            "validation_contracts": validation_contracts,
            "condition_lineage": condition_lineage,
        }
    elif args.phase2_data_manifest:
        raise ValueError(
            "--phase2-data-manifest is only valid for the specific_relation Phase 2 config"
        )

    stage_config = config.raw["training"][args.stage]
    dataset_dict, nentity, nrelation = _datasets(
        config, ["train", "valid"], train_variant=stage_config["data_variant"]
    )
    validation_config = config.raw["training"]["validation"]
    checkpoint_config = config.raw["training"]["checkpoint"]
    dataloader = _loader(dataset_dict["train"], stage_config["micro_batch_size"], config.seed, True)
    validation_loader = _loader(
        dataset_dict["valid"], validation_config["batch_size"], config.seed, False
    )
    if phase2_context is not None:
        verify_frozen_validation_dataset(
            dataset_dict["valid"],
            condition_values=phase2_context["validation_conditions"],
            record_contracts=phase2_context["validation_contracts"],
        )
    model, tokenizer, loaded = _model_and_tokenizer(
        config, nentity, nrelation, args, args.stage, phase2_context=phase2_context
    )
    device = _require_cuda("SFT")
    model.to(device)
    graph_samplers = _graph_samplers(config)
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
        num_batches=len(dataloader),
        gradient_accumulation_steps=config.raw["training"]["gradient_accumulation_steps"],
        epochs=stage_config["epochs"],
        **schedule_options,
    )
    start_epoch = 0
    inherited_global_step = 0
    if loaded is not None:
        inherited_global_step = int(loaded.metadata["global_step"])
    if args.resume_checkpoint:
        loaded = load_reproduction_checkpoint(
            args.resume_checkpoint,
            mode="resume",
            model=model,
            optimizer=schedule.optimizer,
            scheduler=schedule.scheduler,
            expected_stage=args.stage,
            expected_condition="unconditional" if args.stage == "unconditional" else config.condition,
            expected_config_hash=config.semantic_hash,
            expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
            restore_rng=True,
        )
        start_epoch = int(loaded.metadata["stage_epoch"])
        if phase2_context is not None and loaded.metadata.get("condition_lineage") != phase2_context[
            "condition_lineage"
        ]:
            raise ValueError("Resume checkpoint Phase 2 condition lineage mismatch")

    condition = "unconditional" if args.stage == "unconditional" else config.condition
    global_step = inherited_global_step
    final_epoch = int(stage_config["epochs"])
    if start_epoch >= final_epoch:
        raise ValueError(f"Checkpoint already reached configured {args.stage} epochs")
    root = config.runtime_paths["checkpoint_root"] / config.experiment["name"]
    output_root = config.runtime_paths["run_root"] / config.experiment["name"]
    history_path = output_root / f"{args.stage}-history.jsonl"
    best_pointer_path = root / f"{args.stage}-best.json"
    best_record = load_best_checkpoint_record(best_pointer_path)
    data_manifest_hash = _file_sha256(config.sampling_manifest_path)
    parent_reference = args.parent_checkpoint
    if parent_reference is None and loaded is not None:
        parent_reference = loaded.metadata.get("parent_checkpoint")
    output = None
    for stage_epoch in range(start_epoch + 1, final_epoch + 1):
        learning_rate_start = float(schedule.optimizer.param_groups[0]["lr"])
        epoch_loader = _loader(
            dataset_dict["train"],
            stage_config["micro_batch_size"],
            derive_seed(config.seed, args.stage, stage_epoch),
            True,
        )
        train_prepare = partial(
            prepare_batch,
            device,
            tokenizer=tokenizer,
            is_gpt=True,
            src_len=_prompt_length(config),
            tgt_len=config.raw["generation"]["max_new_tokens"],
            is_gen=False,
            condition=condition,
            condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
            condition_seed=(config.seed if phase2_context is not None else None),
            condition_epoch=(stage_epoch if phase2_context is not None else None),
        )
        train_loss, steps = sft_train_epoch(
            model=model,
            dataloader=epoch_loader,
            prepare=train_prepare,
            optimizer=schedule.optimizer,
            scheduler=schedule.scheduler,
            gradient_accumulation_steps=config.raw["training"]["gradient_accumulation_steps"],
        )
        if steps != schedule.optimizer_steps_per_epoch:
            raise RuntimeError(
                f"{args.stage} epoch {stage_epoch} produced {steps} optimizer steps; "
                f"expected {schedule.optimizer_steps_per_epoch}"
            )
        global_step += steps
        history_record = {
            "stage": args.stage,
            "stage_epoch": stage_epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "optimizer_steps": steps,
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": float(schedule.optimizer.param_groups[0]["lr"]),
            "optimizer_schedule": schedule.metadata,
            "validation": None,
            "checkpoint": None,
            "is_best": False,
        }
        validation_record = None
        improved = False
        if epoch_due(stage_epoch, final_epoch, validation_config["every_epochs"]):
            validation_prepare = partial(
                prepare_batch,
                device,
                tokenizer=tokenizer,
                is_gpt=True,
                src_len=_prompt_length(config),
                tgt_len=config.raw["generation"]["max_new_tokens"],
                is_gen=False,
                condition=condition,
                condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
                condition_value_by_record_id=(
                    phase2_context["validation_conditions"]
                    if phase2_context is not None
                    else None
                ),
            )
            validation_loss = sft_validation_loss(
                model=model, dataloader=validation_loader, prepare=validation_prepare
            )
            validation_records = _evaluation_records(
                config=config,
                dataloader=validation_loader,
                model=model,
                tokenizer=tokenizer,
                graph_samplers=graph_samplers,
                device=device,
                split="valid",
                condition_kind=condition,
                do_sample=False,
                condition_value_by_record_id=(
                    phase2_context["validation_conditions"]
                    if phase2_context is not None
                    else None
                ),
            )
            validation_record = {
                "stage": args.stage,
                "stage_epoch": stage_epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **_aggregate_evaluation_metrics(validation_records),
            }
            validation_record["health_pass"] = (
                float(validation_record["parse_ok"] or 0.0)
                >= float(validation_config["min_parse_ok"])
                and float(validation_record["eos_rate"] or 0.0)
                >= float(validation_config["min_eos_rate"])
            )
            improved = is_better_validation(args.stage, validation_record, best_record)
            validation_record["is_best"] = improved
            _write_validation_artifacts(
                output_root,
                stage=args.stage,
                stage_epoch=stage_epoch,
                records=validation_records,
                summary=validation_record,
            )
            history_record["validation"] = validation_record

        should_save = (
            improved
            or epoch_due(stage_epoch, final_epoch, checkpoint_config["every_epochs"])
        )
        if should_save:
            output = save_reproduction_checkpoint(
                root / f"{args.stage}-epoch-{stage_epoch}",
                model=model,
                tokenizer=tokenizer,
                stage=args.stage,
                stage_epoch=stage_epoch,
                global_step=global_step,
                condition=condition,
                experiment_config=config.raw,
                experiment_config_hash=config.semantic_hash,
                seed=config.seed,
                parent_checkpoint=parent_reference,
                optimizer=schedule.optimizer,
                scheduler=schedule.scheduler,
                data_manifest_hash=data_manifest_hash,
                condition_lineage=(
                    phase2_context["condition_lineage"]
                    if phase2_context is not None
                    else None
                ),
            )
            history_record["checkpoint"] = output.name
        if improved:
            best_record = validation_record
            history_record["is_best"] = True
            write_best_checkpoint_pointer(
                best_pointer_path, checkpoint=output, record=validation_record
            )
        history_record["pruned_checkpoints"] = []
        if should_save:
            removed = prune_sft_checkpoints(
                root,
                stage=args.stage,
                keep_last=checkpoint_config["keep_last"],
                best_epoch=None if best_record is None else best_record["stage_epoch"],
            )
            history_record["pruned_checkpoints"] = [path.name for path in removed]
        append_jsonl(history_path, history_record)
    if best_record is None:
        raise RuntimeError(
            f"{args.stage} SFT never passed the configured parse/EOS health gates; "
            "inspect validation artifacts before continuing"
        )
    return output


def run_evaluation(config, args) -> Path:
    if not args.checkpoint:
        raise ValueError("testing requires --checkpoint")
    loaded = load_reproduction_checkpoint(
        args.checkpoint,
        mode="test",
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    checkpoint_stage = loaded.metadata["stage"]
    train_variant = (
        config.raw["training"][checkpoint_stage]["data_variant"]
        if checkpoint_stage in {"unconditional", "conditional"}
        else config.raw["grpo"]["data_variant"]
    )
    dataset_dict, _, _ = _datasets(
        config, [args.test_split], train_variant=train_variant
    )
    dataloader = _loader(dataset_dict[args.test_split], args.overwrite_batchsize or 4, config.seed, False)
    graph_samplers = _graph_samplers(config)
    device = _require_cuda("evaluation")
    model, tokenizer = loaded.model.to(device), loaded.tokenizer
    condition_kind = loaded.metadata["condition"]
    records = _evaluation_records(
        config=config,
        dataloader=dataloader,
        model=model,
        tokenizer=tokenizer,
        graph_samplers=graph_samplers,
        device=device,
        split=args.test_split,
        condition_kind=condition_kind,
        do_sample=bool(config.raw["generation"]["do_sample"] and not args.greedy),
    )
    output_root = config.runtime_paths["run_root"] / config.experiment["name"]
    decode_kind = "greedy" if args.greedy or not config.raw["generation"]["do_sample"] else "sampled"
    jsonl_path = write_evaluation_jsonl(
        output_root / f"{args.test_split}-{decode_kind}.jsonl", records
    )
    pd.Series(_aggregate_evaluation_metrics(records), name="mean").to_csv(
        output_root / f"{args.test_split}-{decode_kind}.csv"
    )
    return jsonl_path


def run_grpo(config, args) -> Path:
    _require_cuda("GRPO")
    parent = args.parent_checkpoint or args.checkpoint
    if not parent:
        raise ValueError("GRPO requires --parent-checkpoint with the conditional SFT checkpoint")
    parent = _require_phase_d_parent_checkpoint(parent)
    loaded = load_reproduction_checkpoint(
        parent,
        mode="parent",
        expected_stage="conditional",
        expected_condition=config.condition,
        expected_config_hash=config.semantic_hash,
        expected_data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )
    dataset_dict, _, _ = _datasets(
        config, ["train"], train_variant=config.raw["grpo"]["data_variant"]
    )
    tokenizer = loaded.tokenizer

    def add_prompt(example):
        value = condition_value_from_target(config.condition, example["target"])
        spec = ConditionSpec(config.condition, value)
        return {
            "prompt": build_generation_prompt(
                example["source"], spec, tokenizer,
                condition_delimiter=config.raw["tokenizer"]["condition_delimiter"],
            ),
            "condition": value,
        }

    dataset = dataset_dict["train"].map(add_prompt)
    graph_samplers = _graph_samplers(config)
    output = config.runtime_paths["checkpoint_root"] / config.experiment["name"] / "grpo"
    if output.is_dir() and any(output.iterdir()) and not args.resume_checkpoint:
        raise FileExistsError(
            f"GRPO output is not empty; pass --resume-checkpoint explicitly or "
            f"preserve and move the existing run before starting: {output}"
        )
    trainer = create_grpo_trainer(
        model=loaded.model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        output_dir=output,
        graph_samplers=graph_samplers,
        max_steps=args.max_steps,
    )
    train_grpo(trainer, resume_checkpoint=args.resume_checkpoint)
    tokenizer.padding_side = "right"
    tokenizer.backend_tokenizer.no_padding()
    evaluation_checkpoint = output / f"evaluation-step-{int(trainer.state.global_step)}"
    return save_reproduction_checkpoint(
        evaluation_checkpoint,
        model=trainer.model,
        tokenizer=tokenizer,
        stage="grpo",
        stage_epoch=int(config.raw["grpo"]["epochs"]),
        global_step=int(trainer.state.global_step),
        condition=config.condition,
        experiment_config=config.raw,
        experiment_config_hash=config.semantic_hash,
        seed=config.seed,
        parent_checkpoint=parent,
        data_manifest_hash=_file_sha256(config.sampling_manifest_path),
    )


def run_experiment_config(args):
    config = load_experiment_config(args.experiment_config)
    seed_everything(config.seed)
    output_root = config.runtime_paths["run_root"] / config.experiment["name"]
    config.write_snapshots(output_root)
    configure_reproduction_logging(output_root / "reproduction.log")
    if args.mode == "training":
        return run_sft(config, args)
    if args.mode == "testing":
        return run_evaluation(config, args)
    if args.mode == "optimizing":
        return run_grpo(config, args)
    raise ValueError("Experiment config mode must be training, testing, or optimizing")

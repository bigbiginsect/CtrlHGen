"""Phase A training and GRPO helpers used by the strict experiment entry point."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
from typing import Callable

import torch
from transformers import get_linear_schedule_with_warmup

from akgr.evaluation import scoring_input_act_batch_condition
from akgr.reproduction.contracts import normalize_condition


LOGGER = logging.getLogger(__name__)


def configure_reproduction_logging(path) -> Path:
    """Persist strict-run training and reward metrics at INFO level."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers:
        if getattr(handler, "_ctrlhgen_destination", None) == destination:
            return destination
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler._ctrlhgen_destination = destination
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return destination


@dataclass(frozen=True)
class OptimizerSchedule:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer_steps_per_epoch: int
    warmup_steps: int
    total_steps: int


def epoch_due(stage_epoch: int, final_epoch: int, every_epochs: int) -> bool:
    """Run periodic work on schedule and always at the configured final epoch."""
    return int(stage_epoch) == int(final_epoch) or int(stage_epoch) % int(every_epochs) == 0


def validation_selection_key(stage: str, record: dict) -> tuple[float, ...]:
    """Return the fixed model-selection ordering for one SFT stage."""
    validation_loss = float(record["validation_loss"])
    parse_ok = float(record.get("parse_ok") or 0.0)
    jaccard = float(record.get("jaccard") or 0.0)
    if stage == "unconditional":
        return (parse_ok, jaccard, -validation_loss)
    if stage == "conditional":
        condition_accuracy = float(record.get("condition_accuracy") or 0.0)
        return (condition_accuracy, parse_ok, jaccard, -validation_loss)
    raise ValueError(f"Unknown SFT stage: {stage!r}")


def is_better_validation(stage: str, candidate: dict, current: dict | None) -> bool:
    """Select only healthy checkpoints, prioritizing generation-level quality."""
    if candidate.get("health_pass") is False:
        return False
    if current is None:
        return True
    return validation_selection_key(stage, candidate) > validation_selection_key(stage, current)


def append_jsonl(path, record: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return destination


def write_best_checkpoint_pointer(path, *, checkpoint: Path, record: dict) -> Path:
    """Atomically publish the one best-checkpoint pointer for an SFT stage."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    best_link = destination.with_suffix("")
    payload = {
        "schema_version": 1,
        "stage": record["stage"],
        "stage_epoch": int(record["stage_epoch"]),
        "checkpoint": checkpoint.name,
        "best_checkpoint": best_link.name,
        "selection": (
            ["parse_ok:max", "jaccard:max", "validation_loss:min"]
            if record["stage"] == "unconditional"
            else [
                "condition_accuracy:max", "parse_ok:max", "jaccard:max",
                "validation_loss:min",
            ]
        ),
        "validation": record,
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    temporary_link = best_link.with_name(f".{best_link.name}.tmp-{os.getpid()}")
    temporary_link.symlink_to(checkpoint.name, target_is_directory=True)
    os.replace(temporary_link, best_link)
    return destination


def load_best_checkpoint_record(path) -> dict | None:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return None
    return json.loads(source.read_text(encoding="utf-8"))["validation"]


def prune_sft_checkpoints(root, *, stage: str, keep_last: int, best_epoch: int | None) -> list[Path]:
    """Keep the latest periodic checkpoints plus the selected best checkpoint."""
    directory = Path(root).expanduser().resolve()
    pattern = re.compile(rf"^{re.escape(stage)}-epoch-(\d+)$")
    checkpoints: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return []
    for candidate in directory.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match and candidate.is_dir():
            checkpoints.append((int(match.group(1)), candidate))
    checkpoints.sort()
    protected = {epoch for epoch, _ in checkpoints[-int(keep_last):]}
    if best_epoch is not None:
        protected.add(int(best_epoch))
    removed = []
    for epoch, candidate in checkpoints:
        if epoch in protected:
            continue
        if candidate.parent.resolve() != directory:
            raise ValueError(f"Refusing to prune checkpoint outside {directory}: {candidate}")
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


def create_sft_optimizer_schedule(
    model,
    *,
    learning_rate: float,
    num_batches: int,
    gradient_accumulation_steps: int,
    epochs: int,
    warmup_epochs: int,
) -> OptimizerSchedule:
    """Create AdamW and translate epoch warm-up into optimizer-step units."""
    if num_batches <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("num_batches, gradient_accumulation_steps, and epochs must be positive")
    optimizer_steps_per_epoch = math.ceil(num_batches / gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = min(optimizer_steps_per_epoch * warmup_epochs, total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return OptimizerSchedule(optimizer, scheduler, optimizer_steps_per_epoch, warmup_steps, total_steps)


def sft_train_epoch(
    *,
    model,
    dataloader,
    prepare: Callable,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    accelerator=None,
) -> tuple[float, int]:
    """Train exactly one stage epoch and step the scheduler on optimizer steps."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0
    num_batches = len(dataloader)
    for batch_index, sample in enumerate(dataloader, start=1):
        batch = prepare(sample)
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        group_start = ((batch_index - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(gradient_accumulation_steps, num_batches - group_start)
        loss = output.loss / group_size
        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)
        total_loss += float(output.loss.detach().cpu())
        should_step = batch_index % gradient_accumulation_steps == 0 or batch_index == num_batches
        if should_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    return total_loss / num_batches, optimizer_steps


def sft_validation_loss(*, model, dataloader, prepare: Callable) -> float:
    """Compute a sample-weighted teacher-forced validation loss."""
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for sample in dataloader:
            batch = prepare(sample)
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            batch_size = int(batch.input_ids.shape[0])
            weighted_loss += float(output.loss.detach().cpu()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Validation dataloader is empty")
    return weighted_loss / sample_count


def _condition_scoring_method(condition: str) -> tuple[str, str]:
    condition = normalize_condition(condition)
    if condition == "pattern":
        return "validity", "validity"
    if condition == "entity_number":
        return "validity", "enumber"
    if condition == "relation_number":
        return "validity", "pnumber"
    if condition in {"specific_entity", "specific_relation"}:
        return "specific", "spec"
    raise ValueError("GRPO requires a conditional experiment")


def make_grpo_reward_functions(
    *, condition: str, graph_samplers, searching_split: str, reward_weights=None
):
    """Create TRL 0.16-compatible, named component reward callables."""
    condition = normalize_condition(condition)
    condition_method, condition_key = _condition_scoring_method(condition)
    names = ("jaccard", "dice", "overlap", "condition")
    if reward_weights is None:
        reward_weights = {"jaccard": 1.0, "dice": 0.5, "overlap": 0.5, "condition": 1.0}
    weights = {name: float(reward_weights[name]) for name in names}
    latest_components = {}

    def component_reward(component: str, scoring_method: str, score_key: str):
        def reward(prompts, completions, source, target, condition, **kwargs):
            scores, _ = scoring_input_act_batch_condition(
                pred_word_batch=completions,
                label_word_batch=target,
                ans_word_batch=source,
                condition_batch=condition,
                scoring_method=[scoring_method],
                graph_samplers=graph_samplers,
                searching_split=searching_split,
                return_failures=True,
            )
            values = [float(score[score_key]) for score in scores]
            latest_components[component] = values
            mean = sum(values) / len(values) if values else 0.0
            LOGGER.info("GRPO reward component %s mean=%.6f", component, mean)
            if all(name in latest_components for name in names):
                combined = [
                    sum(weights[name] * latest_components[name][index] for name in names)
                    for index in range(len(values))
                ]
                combined_mean = sum(combined) / len(combined) if combined else 0.0
                LOGGER.info("GRPO combined reward mean=%.6f", combined_mean)
                latest_components.clear()
            return values

        reward.__name__ = f"{component}_reward"
        return reward

    return [
        component_reward("jaccard", "jaccard", "jaccard"),
        component_reward("dice", "dice", "dice"),
        component_reward("overlap", "overlap", "overlap"),
        component_reward("condition", condition_method, condition_key),
    ]


def build_grpo_config(config, output_dir, *, max_steps: int = -1, save_steps: int | None = None):
    """Construct the pinned TRL 0.16 GRPO settings with no evaluation/W&B."""
    from trl import GRPOConfig

    grpo = config.raw["grpo"] if hasattr(config, "raw") else config["grpo"]
    if save_steps is None:
        save_steps = int(grpo["save_steps"])
    weights = grpo["reward_weights"]
    return GRPOConfig(
        output_dir=str(Path(output_dir)),
        seed=int(config.seed if hasattr(config, "seed") else config["experiment"]["seed"]),
        num_train_epochs=float(grpo["epochs"]),
        max_steps=int(max_steps),
        learning_rate=float(grpo["learning_rate"]),
        beta=float(grpo["beta"]),
        epsilon=float(grpo["epsilon"]),
        num_generations=4,
        per_device_train_batch_size=int(grpo["per_device_train_batch_size"]),
        max_completion_length=int(grpo["max_completion_length"]),
        reward_weights=[
            float(weights["jaccard"]),
            float(weights["dice"]),
            float(weights["overlap"]),
            float(weights["condition"]),
        ],
        remove_unused_columns=False,
        eval_strategy="no",
        report_to=[],
        save_strategy="steps",
        save_steps=int(save_steps),
        save_total_limit=int(grpo["save_total_limit"]),
    )


def create_grpo_trainer(
    *,
    model,
    tokenizer,
    dataset,
    config,
    output_dir,
    graph_samplers,
    searching_split="train",
    max_steps=-1,
):
    """Create a GRPOTrainer whose checkpoints support Trainer resume."""
    from trl import GRPOTrainer

    tokenizer.padding_side = "left"
    reward_functions = make_grpo_reward_functions(
        condition=config.condition,
        graph_samplers=graph_samplers,
        searching_split=searching_split,
        reward_weights=config.raw["grpo"]["reward_weights"],
    )
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=build_grpo_config(config, output_dir, max_steps=max_steps),
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def train_grpo(trainer, *, resume_checkpoint: str | None = None) -> Path:
    """Train/resume GRPO and save an evaluation-ready final model directory."""
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = Path(trainer.args.output_dir) / "final-model"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    return final_dir

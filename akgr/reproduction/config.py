"""Strict experiment configuration loading and reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import normalize_condition


ROOT_ENV = {
    "data_root": "CTRLHGEN_DATA_ROOT",
    "checkpoint_root": "CTRLHGEN_CHECKPOINT_ROOT",
    "run_root": "CTRLHGEN_RUN_ROOT",
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "tokenizer",
    "data",
    "sampling",
    "augmentation",
    "model",
    "training",
    "generation",
    "grpo",
}

SECTION_KEYS = {
    "experiment": {"name", "dataset", "profile", "seed", "condition"},
    "tokenizer": {"condition_delimiter"},
    "data": {
        "max_answers",
        "split_ratios",
        "reverse_edges",
        "max_attempts_per_record",
        "workers",
    },
    "sampling": {"train_per_pattern", "valid_per_pattern", "test_per_pattern"},
    "augmentation": {"enabled", "splits", "source_patterns"},
    "model": {
        "type",
        "initialization",
        "n_layer",
        "n_embd",
        "n_head",
        "n_positions",
        "n_ctx",
        "tie_word_embeddings",
    },
    "training": {
        "gradient_accumulation_steps",
        "validation",
        "checkpoint",
        "unconditional",
        "conditional",
    },
    "generation": {"do_sample", "top_k", "top_p", "temperature", "max_new_tokens"},
    "grpo": {
        "num_generations",
        "per_device_train_batch_size",
        "epochs",
        "learning_rate",
        "beta",
        "epsilon",
        "max_completion_length",
        "save_steps",
        "save_total_limit",
        "report_to",
        "data_variant",
        "reward_weights",
    },
}

STAGE_COMMON_KEYS = {
    "epochs",
    "learning_rate",
    "micro_batch_size",
    "effective_batch_size",
    "data_variant",
}
STAGE_LEGACY_KEYS = STAGE_COMMON_KEYS | {"warmup_epochs"}
STAGE_EXPLICIT_KEYS = STAGE_COMMON_KEYS | {"optimizer", "scheduler"}
OPTIMIZER_KEYS = {"name", "betas", "eps", "weight_decay"}
SCHEDULER_KEYS = {"name", "warmup_unit", "warmup_value", "start_factor"}
VALIDATION_KEYS = {"every_epochs", "batch_size", "min_parse_ok", "min_eos_rate"}
CHECKPOINT_KEYS = {"every_epochs", "keep_last"}
REWARD_KEYS = {"jaccard", "dice", "overlap", "condition"}


def _require_exact_keys(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing keys in {section}: {sorted(missing)}")


def _validate_stage_shape(stage: str, value: Mapping[str, Any]) -> str:
    """Validate one legacy or explicit SFT stage and return its config style."""
    section = f"training.{stage}"
    keys = set(value)
    explicit_fields = {"optimizer", "scheduler"}
    if "warmup_epochs" in keys and keys & explicit_fields:
        raise ValueError(
            f"{section} must use either legacy warmup_epochs or explicit "
            "optimizer/scheduler, not both"
        )
    if keys & explicit_fields:
        _require_exact_keys(section, value, STAGE_EXPLICIT_KEYS)
        optimizer = value["optimizer"]
        scheduler = value["scheduler"]
        if not isinstance(optimizer, dict):
            raise ValueError(f"{section}.optimizer must be a mapping")
        if not isinstance(scheduler, dict):
            raise ValueError(f"{section}.scheduler must be a mapping")
        _require_exact_keys(f"{section}.optimizer", optimizer, OPTIMIZER_KEYS)
        _require_exact_keys(f"{section}.scheduler", scheduler, SCHEDULER_KEYS)
        return "explicit"
    _require_exact_keys(section, value, STAGE_LEGACY_KEYS)
    return "legacy"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _data_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can change sampled examples or the KG split."""
    experiment = raw["experiment"]
    return {
        "schema_version": raw["schema_version"],
        "experiment": {
            "dataset": experiment["dataset"],
            "profile": experiment["profile"],
            "seed": experiment["seed"],
        },
        "data": raw["data"],
        "sampling": raw["sampling"],
        "augmentation": raw["augmentation"],
    }


def _kg_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    experiment = raw["experiment"]
    data = raw["data"]
    return {
        "schema_version": raw["schema_version"],
        "dataset": experiment["dataset"],
        "seed": experiment["seed"],
        "split_ratios": data["split_ratios"],
        "reverse_edges": data["reverse_edges"],
    }


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    raw: dict[str, Any]
    runtime_paths: dict[str, Path]
    semantic_hash: str
    data_hash: str
    kg_hash: str

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    @property
    def seed(self) -> int:
        return int(self.experiment["seed"])

    @property
    def dataset(self) -> str:
        return str(self.experiment["dataset"])

    @property
    def condition(self) -> str:
        return normalize_condition(str(self.experiment["condition"]))

    @property
    def artifact_dir(self) -> Path:
        return (
            self.runtime_paths["data_root"]
            / self.dataset
            / str(self.experiment["profile"])
            / f"seed-{self.seed}"
            / self.data_hash[:12]
        )

    @property
    def sampling_manifest_path(self) -> Path:
        return self.artifact_dir / "sampling-manifest.json"

    def resolved_dict(self) -> dict[str, Any]:
        resolved = dict(self.raw)
        resolved["runtime_paths"] = {key: str(path) for key, path in self.runtime_paths.items()}
        resolved["semantic_hash"] = self.semantic_hash
        resolved["data_hash"] = self.data_hash
        resolved["kg_hash"] = self.kg_hash
        resolved["source_path"] = str(self.source_path)
        return resolved

    def write_snapshots(self, output_dir: os.PathLike[str] | str) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "experiment.raw.yml").write_text(
            yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (target / "experiment.resolved.yml").write_text(
            yaml.safe_dump(self.resolved_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _validate(raw: dict[str, Any]) -> None:
    _require_exact_keys("root", raw, TOP_LEVEL_KEYS)
    if raw["schema_version"] != 1:
        raise ValueError("Only experiment schema_version=1 is supported")
    for section, allowed in SECTION_KEYS.items():
        value = raw.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a mapping")
        _require_exact_keys(section, value, allowed)
    stage_styles = {
        stage: _validate_stage_shape(stage, raw["training"][stage])
        for stage in ("unconditional", "conditional")
    }
    _require_exact_keys("training.validation", raw["training"]["validation"], VALIDATION_KEYS)
    _require_exact_keys("training.checkpoint", raw["training"]["checkpoint"], CHECKPOINT_KEYS)
    _require_exact_keys("grpo.reward_weights", raw["grpo"]["reward_weights"], REWARD_KEYS)

    exp = raw["experiment"]
    if exp["dataset"] != "WN18RR":
        raise ValueError("Phase A reproduce configs support WN18RR only")
    normalize_condition(str(exp["condition"]))
    if exp["profile"] not in {"tiny", "small", "full", "diagnostic"}:
        raise ValueError("profile must be tiny, small, full, or diagnostic")
    if not isinstance(exp["seed"], int) or exp["seed"] < 0:
        raise ValueError("experiment.seed must be a non-negative integer")

    if raw["tokenizer"]["condition_delimiter"] != "COND":
        raise ValueError("Strict reproduction configs must use the dedicated COND delimiter")

    data = raw["data"]
    if [float(x) for x in data["split_ratios"]] != [0.8, 0.1, 0.1]:
        raise ValueError("Phase A KG split_ratios must be [0.8, 0.1, 0.1]")
    if int(data["max_answers"]) <= 0 or int(data["max_attempts_per_record"]) <= 0:
        raise ValueError("max_answers and max_attempts_per_record must be positive")
    if int(data["workers"]) <= 0:
        raise ValueError("data.workers must be positive")
    for split_key in ("train_per_pattern", "valid_per_pattern", "test_per_pattern"):
        if int(raw["sampling"][split_key]) <= 0:
            raise ValueError(f"sampling.{split_key} must be positive")
    if set(raw["augmentation"]["splits"]) != {"train"}:
        raise ValueError("Sub-logic augmentation is train-only")
    if set(raw["augmentation"]["source_patterns"]) != {"up", "3in", "pni", "pin", "inp"}:
        raise ValueError("augmentation.source_patterns must match the five paper patterns")

    model = raw["model"]
    if model["type"] != "gpt2" or model["initialization"] != "random":
        raise ValueError("Phase A uses a randomly initialized GPT-2 model")
    if int(model["n_layer"]) not in {6, 12}:
        raise ValueError("Reproduction model n_layer must be one of the registered 6 or 12 layer variants")
    if (int(model["n_embd"]), int(model["n_head"])) != (768, 12):
        raise ValueError("Reproduction model must use the registered 768-hidden, 12-head GPT-2 structure")
    if int(model["n_positions"]) != 1024 or int(model["n_ctx"]) != 1024:
        raise ValueError("Reproduction model n_positions and n_ctx must both be 1024")
    if model["tie_word_embeddings"] is not True:
        raise ValueError("Reproduction model must use tied word embeddings")

    accumulation = int(raw["training"]["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    for stage in ("unconditional", "conditional"):
        stage_config = raw["training"][stage]
        if int(stage_config["epochs"]) <= 0 or int(stage_config["micro_batch_size"]) <= 0:
            raise ValueError(f"training.{stage} epochs and micro_batch_size must be positive")
        if float(stage_config["learning_rate"]) <= 0:
            raise ValueError(f"training.{stage}.learning_rate must be positive")
        if stage_styles[stage] == "legacy":
            if not 0 <= int(stage_config["warmup_epochs"]) <= int(stage_config["epochs"]):
                raise ValueError(f"training.{stage}.warmup_epochs must be between zero and epochs")
        else:
            optimizer = stage_config["optimizer"]
            scheduler = stage_config["scheduler"]
            if optimizer["name"] not in {"adam", "adamw"}:
                raise ValueError(f"training.{stage}.optimizer.name must be adam or adamw")
            betas = optimizer["betas"]
            if (
                not isinstance(betas, list)
                or len(betas) != 2
                or any(not 0.0 <= float(beta) < 1.0 for beta in betas)
            ):
                raise ValueError(f"training.{stage}.optimizer.betas must contain two values in [0, 1)")
            if float(optimizer["eps"]) <= 0 or float(optimizer["weight_decay"]) < 0:
                raise ValueError(
                    f"training.{stage}.optimizer eps must be positive and weight_decay non-negative"
                )
            if scheduler["name"] not in {"linear_warmup_decay", "linear_warmup_constant"}:
                raise ValueError(
                    f"training.{stage}.scheduler.name must be linear_warmup_decay "
                    "or linear_warmup_constant"
                )
            if scheduler["warmup_unit"] not in {"epoch", "optimizer_step"}:
                raise ValueError(
                    f"training.{stage}.scheduler.warmup_unit must be epoch or optimizer_step"
                )
            warmup_value = scheduler["warmup_value"]
            if isinstance(warmup_value, bool) or not isinstance(warmup_value, int) or warmup_value < 0:
                raise ValueError(f"training.{stage}.scheduler.warmup_value must be a non-negative integer")
            if scheduler["warmup_unit"] == "epoch" and warmup_value > int(stage_config["epochs"]):
                raise ValueError(
                    f"training.{stage}.scheduler warm-up cannot exceed configured epochs"
                )
            if not 0.0 <= float(scheduler["start_factor"]) <= 1.0:
                raise ValueError(f"training.{stage}.scheduler.start_factor must be between zero and one")
        if stage_config["data_variant"] not in {"base", "merged"}:
            raise ValueError(f"training.{stage}.data_variant must be base or merged")
        expected_effective = int(stage_config["micro_batch_size"]) * accumulation
        if int(stage_config["effective_batch_size"]) != expected_effective:
            raise ValueError(
                f"training.{stage}.effective_batch_size must equal "
                "micro_batch_size * gradient_accumulation_steps"
            )
    validation = raw["training"]["validation"]
    checkpoint = raw["training"]["checkpoint"]
    if int(validation["every_epochs"]) <= 0 or int(validation["batch_size"]) <= 0:
        raise ValueError("training.validation values must be positive")
    for metric in ("min_parse_ok", "min_eos_rate"):
        if not 0.0 <= float(validation[metric]) <= 1.0:
            raise ValueError(f"training.validation.{metric} must be between zero and one")
    if int(checkpoint["every_epochs"]) <= 0 or int(checkpoint["keep_last"]) <= 0:
        raise ValueError("training.checkpoint values must be positive")

    grpo = raw["grpo"]
    if grpo["data_variant"] not in {"base", "merged"}:
        raise ValueError("grpo.data_variant must be base or merged")
    if grpo["num_generations"] != 4:
        raise ValueError("grpo.num_generations must be 4")
    if grpo["per_device_train_batch_size"] % grpo["num_generations"]:
        raise ValueError("GRPO batch size must be divisible by num_generations")
    if int(grpo["save_steps"]) <= 0 or int(grpo["save_total_limit"]) <= 0:
        raise ValueError("GRPO save_steps and save_total_limit must be positive")
    if grpo["report_to"] not in ([], "none", None):
        raise ValueError("Phase A defaults to disabled external reporting")


def load_experiment_config(path: os.PathLike[str] | str, env: Mapping[str, str] | None = None) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment config root must be a mapping")
    _validate(raw)
    environment = os.environ if env is None else env
    missing = [name for name in ROOT_ENV.values() if not environment.get(name)]
    if missing:
        raise ValueError(f"Missing runtime path environment variables: {', '.join(missing)}")
    runtime_paths = {key: Path(environment[name]).expanduser().resolve() for key, name in ROOT_ENV.items()}
    return ExperimentConfig(
        source_path=source,
        raw=raw,
        runtime_paths=runtime_paths,
        semantic_hash=_canonical_hash(raw),
        data_hash=_canonical_hash(_data_identity(raw)),
        kg_hash=_canonical_hash(_kg_identity(raw)),
    )

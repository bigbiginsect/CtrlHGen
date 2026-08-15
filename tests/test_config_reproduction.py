from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from akgr.reproduction.config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "akgr" / "configs" / "reproduce"
CONFIG_PATHS = {
    profile: CONFIG_DIR / f"wn-pattern-{profile}.yml"
    for profile in ("tiny", "small", "full")
}
AUTHOR_CONFIG_PATH = CONFIG_DIR / "wn-pattern-small-author-aligned.yml"
AUTHOR_PILOT_PATH = (
    REPO_ROOT / "akgr" / "configs" / "diagnostics" / "wn-pattern-small-author-pilot.yml"
)
FULL_TRAIN_AUTHOR_CONFIG_PATH = (
    CONFIG_DIR / "wn-pattern-full-train-author-aligned.yml"
)
FULL_TRAIN_AUTHOR_PILOT_PATH = (
    REPO_ROOT / "akgr" / "configs" / "diagnostics" / "wn-pattern-full-train-author-pilot.yml"
)
SPECIFIC_RELATION_CONFIG_PATH = (
    CONFIG_DIR / "wn-specific-relation-full-train-author-aligned.yml"
)
RUNTIME_ENV = {
    "CTRLHGEN_DATA_ROOT": "/tmp/ctrlhgen-test/data",
    "CTRLHGEN_CHECKPOINT_ROOT": "/tmp/ctrlhgen-test/checkpoints",
    "CTRLHGEN_RUN_ROOT": "/tmp/ctrlhgen-test/runs",
}
LEGACY_SEMANTIC_HASHES = {
    "tiny": "c1ef8ed2f0a528d70d4f5fdb84496824dcd154848ef77d557a08460bcbaf0e4f",
    "small": "0ff4e8455ccf88e882840304ef8e031ae8e0fcb84967e6365bb1a2cf228d16d2",
    "full": "4b90540ab21c43586df932c5b7eb58a0a70ae9c44b18b29f4e5bce2c5d2c78b7",
}


@pytest.mark.parametrize(
    ("profile", "counts"),
    [
        ("tiny", (8, 2, 2)),
        ("small", (1024, 128, 128)),
        ("full", (8000, 1000, 1000)),
    ],
)
def test_wn_pattern_profiles_have_the_locked_contract(profile: str, counts: tuple[int, int, int]) -> None:
    assert set(CONFIG_PATHS) == {"tiny", "small", "full"}
    assert {path.name for path in CONFIG_DIR.glob("wn-pattern-*.yml")} == {
        "wn-pattern-tiny.yml",
        "wn-pattern-small.yml",
        "wn-pattern-full.yml",
        "wn-pattern-small-author-aligned.yml",
        "wn-pattern-full-train-author-aligned.yml",
    }
    config = load_experiment_config(CONFIG_PATHS[profile], env=RUNTIME_ENV)

    assert config.dataset == "WN18RR"
    assert config.condition == "pattern"
    assert config.experiment["profile"] == profile
    assert (
        config.raw["sampling"]["train_per_pattern"],
        config.raw["sampling"]["valid_per_pattern"],
        config.raw["sampling"]["test_per_pattern"],
    ) == counts
    assert config.raw["model"] == {
        "type": "gpt2",
        "initialization": "random",
        "n_layer": 12,
        "n_embd": 768,
        "n_head": 12,
        "n_positions": 1024,
        "n_ctx": 1024,
        "tie_word_embeddings": True,
    }
    assert config.raw["tokenizer"] == {"condition_delimiter": "COND"}
    assert config.raw["augmentation"] == {
        "enabled": True,
        "splits": ["train"],
        "source_patterns": ["up", "3in", "pni", "pin", "inp"],
    }
    assert config.raw["grpo"]["num_generations"] == 4
    assert config.raw["grpo"]["report_to"] == []
    assert config.raw["grpo"]["reward_weights"] == {
        "jaccard": 1.0,
        "dice": 0.5,
        "overlap": 0.5,
        "condition": 1.0,
    }
    expected_runtime_policy = {
        "tiny": {
            "validation": {
                "every_epochs": 1, "batch_size": 4,
                "min_parse_ok": 0.0, "min_eos_rate": 0.0,
            },
            "checkpoint": {"every_epochs": 1, "keep_last": 2},
            "grpo_save": (10, 2),
        },
        "small": {
            "validation": {
                "every_epochs": 10, "batch_size": 16,
                "min_parse_ok": 0.9, "min_eos_rate": 0.98,
            },
            "checkpoint": {"every_epochs": 25, "keep_last": 2},
            "grpo_save": (100, 2),
        },
        "full": {
            "validation": {
                "every_epochs": 10, "batch_size": 16,
                "min_parse_ok": 0.9, "min_eos_rate": 0.98,
            },
            "checkpoint": {"every_epochs": 25, "keep_last": 2},
            "grpo_save": (500, 2),
        },
    }[profile]
    assert config.raw["training"]["validation"] == expected_runtime_policy["validation"]
    assert config.raw["training"]["checkpoint"] == expected_runtime_policy["checkpoint"]
    assert (
        config.raw["grpo"]["save_steps"],
        config.raw["grpo"]["save_total_limit"],
    ) == expected_runtime_policy["grpo_save"]
    assert config.raw["training"]["unconditional"]["data_variant"] == "merged"
    assert config.raw["training"]["conditional"]["data_variant"] == "base"
    assert config.raw["grpo"]["data_variant"] == "base"
    if profile in {"small", "full"}:
        assert config.raw["training"]["gradient_accumulation_steps"] == 1
        assert config.raw["training"]["unconditional"]["micro_batch_size"] == 256
        assert config.raw["training"]["conditional"]["micro_batch_size"] == 256
        assert (
            config.raw["training"]["unconditional"]["epochs"],
            config.raw["training"]["unconditional"]["warmup_epochs"],
            config.raw["training"]["conditional"]["epochs"],
            config.raw["training"]["conditional"]["warmup_epochs"],
        ) == (400, 50, 50, 5)


@pytest.mark.parametrize("profile", ["tiny", "small", "full"])
def test_legacy_config_semantic_hashes_are_unchanged(profile: str) -> None:
    config = load_experiment_config(CONFIG_PATHS[profile], env=RUNTIME_ENV)
    assert config.semantic_hash == LEGACY_SEMANTIC_HASHES[profile]


def test_author_aligned_formal_and_pilot_reuse_small_data_with_isolated_runs() -> None:
    legacy = load_experiment_config(CONFIG_PATHS["small"], env=RUNTIME_ENV)
    formal = load_experiment_config(AUTHOR_CONFIG_PATH, env=RUNTIME_ENV)
    pilot = load_experiment_config(AUTHOR_PILOT_PATH, env=RUNTIME_ENV)

    assert formal.experiment["name"] == "repro-wn-pattern-small-author-aligned-v3"
    assert pilot.experiment["name"] == "diagnostic-wn-pattern-small-author-pilot-v1"
    assert formal.semantic_hash != pilot.semantic_hash != legacy.semantic_hash
    assert formal.data_hash == pilot.data_hash == legacy.data_hash
    assert formal.kg_hash == pilot.kg_hash == legacy.kg_hash
    assert formal.artifact_dir == pilot.artifact_dir == legacy.artifact_dir

    for config in (formal, pilot):
        assert config.raw["model"] == {
            "type": "gpt2",
            "initialization": "random",
            "n_layer": 6,
            "n_embd": 768,
            "n_head": 12,
            "n_positions": 1024,
            "n_ctx": 1024,
            "tie_word_embeddings": True,
        }
        for stage in ("unconditional", "conditional"):
            stage_config = config.raw["training"][stage]
            assert "warmup_epochs" not in stage_config
            assert stage_config["learning_rate"] == 5e-5
            assert stage_config["micro_batch_size"] == 160
            assert stage_config["effective_batch_size"] == 160
            assert stage_config["optimizer"] == {
                "name": "adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
            assert stage_config["scheduler"] == {
                "name": "linear_warmup_constant",
                "warmup_unit": "optimizer_step",
                "warmup_value": 5,
                "start_factor": 0.1,
            }

    assert formal.raw["training"]["unconditional"]["epochs"] == 50
    assert formal.raw["training"]["conditional"]["epochs"] == 50
    assert formal.raw["training"]["validation"] == {
        "every_epochs": 5,
        "batch_size": 16,
        "min_parse_ok": 0.9,
        "min_eos_rate": 0.98,
    }
    assert pilot.raw["training"]["unconditional"]["epochs"] == 10
    assert pilot.raw["training"]["conditional"]["epochs"] == 10
    assert pilot.raw["training"]["validation"] == {
        "every_epochs": 2,
        "batch_size": 16,
        "min_parse_ok": 0.1,
        "min_eos_rate": 0.9,
    }


def test_phase_c_iv_changes_only_train_scale_and_run_budget_from_v3() -> None:
    v3 = load_experiment_config(AUTHOR_CONFIG_PATH, env=RUNTIME_ENV)
    formal = load_experiment_config(FULL_TRAIN_AUTHOR_CONFIG_PATH, env=RUNTIME_ENV)
    pilot = load_experiment_config(FULL_TRAIN_AUTHOR_PILOT_PATH, env=RUNTIME_ENV)

    assert formal.experiment["name"] == "repro-wn-pattern-full-train-author-aligned-c4"
    assert pilot.experiment["name"] == "diagnostic-wn-pattern-full-train-author-pilot-c4"
    assert formal.data_hash == pilot.data_hash != v3.data_hash
    assert formal.kg_hash == pilot.kg_hash == v3.kg_hash
    for config in (formal, pilot):
        assert config.raw["sampling"] == {
            "train_per_pattern": 8000,
            "valid_per_pattern": 128,
            "test_per_pattern": 128,
            "deduplication": "full_supervision_across_splits",
        }
        assert config.raw["model"] == v3.raw["model"]
        assert config.raw["augmentation"] == v3.raw["augmentation"]
        assert config.raw["generation"] == v3.raw["generation"]
        assert config.raw["training"]["gradient_accumulation_steps"] == 1
        for stage in ("unconditional", "conditional"):
            current = dict(config.raw["training"][stage])
            reference = dict(v3.raw["training"][stage])
            current.pop("epochs")
            reference.pop("epochs")
            assert current == reference
    assert formal.raw["training"]["unconditional"]["epochs"] == 50
    assert formal.raw["training"]["conditional"]["epochs"] == 50
    assert formal.raw["training"]["validation"] == v3.raw["training"]["validation"]
    assert formal.raw["training"]["checkpoint"] == v3.raw["training"]["checkpoint"]
    assert pilot.raw["training"]["unconditional"]["epochs"] == 8
    assert pilot.raw["training"]["conditional"]["epochs"] == 8
    assert pilot.raw["training"]["validation"] == {
        "every_epochs": 2,
        "batch_size": 16,
        "min_parse_ok": 0.9,
        "min_eos_rate": 0.98,
    }


def test_sc_idc_specific_relation_config_reuses_data_model_and_sft_budget() -> None:
    source = load_experiment_config(FULL_TRAIN_AUTHOR_CONFIG_PATH, env=RUNTIME_ENV)
    target = load_experiment_config(SPECIFIC_RELATION_CONFIG_PATH, env=RUNTIME_ENV)

    assert target.experiment["name"] == "sc-idc-wn-specific-relation-full-sft-v1"
    assert target.condition == "specific_relation"
    assert target.semantic_hash != source.semantic_hash
    assert target.data_hash == source.data_hash
    assert target.kg_hash == source.kg_hash
    assert target.artifact_dir == source.artifact_dir
    for section in (
        "tokenizer", "data", "sampling", "augmentation", "model", "training",
        "generation", "grpo",
    ):
        assert target.raw[section] == source.raw[section]


def test_explicit_sft_config_rejects_legacy_mixing_and_invalid_nested_values(tmp_path: Path) -> None:
    raw = yaml.safe_load(AUTHOR_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["training"]["unconditional"]["warmup_epochs"] = 5
    mixed_path = tmp_path / "mixed.yml"
    mixed_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="either legacy warmup_epochs"):
        load_experiment_config(mixed_path, env=RUNTIME_ENV)

    raw = yaml.safe_load(AUTHOR_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["training"]["unconditional"]["optimizer"].pop("eps")
    missing_path = tmp_path / "missing.yml"
    missing_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing keys in training.unconditional.optimizer"):
        load_experiment_config(missing_path, env=RUNTIME_ENV)

    raw = yaml.safe_load(AUTHOR_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["training"]["unconditional"]["scheduler"]["warmup_unit"] = "batch"
    invalid_path = tmp_path / "invalid-scheduler.yml"
    invalid_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="warmup_unit"):
        load_experiment_config(invalid_path, env=RUNTIME_ENV)


def test_config_supports_only_registered_six_or_twelve_layer_models(tmp_path: Path) -> None:
    raw = yaml.safe_load(AUTHOR_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["model"]["n_layer"] = 5
    path = tmp_path / "invalid-model.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="6 or 12"):
        load_experiment_config(path, env=RUNTIME_ENV)


def test_real_data_overfit_diagnostic_is_isolated_from_reproduction_profiles() -> None:
    path = REPO_ROOT / "akgr" / "configs" / "diagnostics" / "wn-pattern-overfit.yml"
    config = load_experiment_config(path, env=RUNTIME_ENV)

    assert config.experiment["profile"] == "diagnostic"
    assert config.experiment["name"] == "diagnostic-wn-pattern-overfit-v7"
    assert config.raw["training"]["gradient_accumulation_steps"] == 1
    assert config.raw["generation"]["do_sample"] is False


def test_runtime_paths_do_not_change_the_semantic_hash() -> None:
    path = CONFIG_PATHS["tiny"]
    first = load_experiment_config(path, env=RUNTIME_ENV)
    second = load_experiment_config(
        path,
        env={
            "CTRLHGEN_DATA_ROOT": "/different/data",
            "CTRLHGEN_CHECKPOINT_ROOT": "/different/checkpoints",
            "CTRLHGEN_RUN_ROOT": "/different/runs",
        },
    )

    assert first.semantic_hash == second.semantic_hash
    assert first.data_hash == second.data_hash
    assert first.kg_hash == second.kg_hash
    assert first.runtime_paths != second.runtime_paths


def test_training_changes_do_not_change_sampled_data_identity(tmp_path: Path) -> None:
    original = load_experiment_config(CONFIG_PATHS["tiny"], env=RUNTIME_ENV)
    raw = yaml.safe_load(CONFIG_PATHS["tiny"].read_text(encoding="utf-8"))
    raw["experiment"]["name"] = "same-data-new-training"
    raw["training"]["unconditional"]["epochs"] = 2
    changed_path = tmp_path / "changed-training.yml"
    changed_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    changed = load_experiment_config(changed_path, env=RUNTIME_ENV)

    assert changed.semantic_hash != original.semantic_hash
    assert changed.data_hash == original.data_hash
    assert changed.kg_hash == original.kg_hash
    assert changed.artifact_dir == original.artifact_dir

    raw["sampling"]["train_per_pattern"] = 9
    data_changed_path = tmp_path / "changed-data.yml"
    data_changed_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    data_changed = load_experiment_config(data_changed_path, env=RUNTIME_ENV)
    assert data_changed.data_hash != original.data_hash
    assert data_changed.kg_hash == original.kg_hash


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATHS["tiny"].read_text(encoding="utf-8"))
    raw["data"]["silent_override"] = True
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown keys in data"):
        load_experiment_config(path, env=RUNTIME_ENV)


def test_config_requires_all_external_runtime_roots() -> None:
    with pytest.raises(ValueError, match="CTRLHGEN_CHECKPOINT_ROOT"):
        load_experiment_config(CONFIG_PATHS["tiny"], env={"CTRLHGEN_DATA_ROOT": "/tmp/data"})


def test_reproduction_requirements_keep_cuda_torch_install_separate() -> None:
    requirements = (REPO_ROOT / "requirements-repro.txt").read_text(encoding="utf-8")
    package_lines = [
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert all("==" in line for line in package_lines)
    assert not any(line.startswith(("torch==", "torchvision==", "torchaudio==")) for line in package_lines)

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cu124" in readme
    assert "torch==2.6.0" in readme
    assert "Python 3.11" in readme

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from akgr.reproduction.config import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "akgr" / "configs" / "reproduce"
CONFIG_PATHS = {
    path.stem.removeprefix("wn-pattern-"): path
    for path in sorted(CONFIG_DIR.glob("wn-pattern-*.yml"))
}
RUNTIME_ENV = {
    "CTRLHGEN_DATA_ROOT": "/tmp/ctrlhgen-test/data",
    "CTRLHGEN_CHECKPOINT_ROOT": "/tmp/ctrlhgen-test/checkpoints",
    "CTRLHGEN_RUN_ROOT": "/tmp/ctrlhgen-test/runs",
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
        assert (
            config.raw["training"]["unconditional"]["epochs"],
            config.raw["training"]["unconditional"]["warmup_epochs"],
            config.raw["training"]["conditional"]["epochs"],
            config.raw["training"]["conditional"]["warmup_epochs"],
        ) == (400, 50, 50, 5)


def test_real_data_overfit_diagnostic_is_isolated_from_reproduction_profiles() -> None:
    path = REPO_ROOT / "akgr" / "configs" / "diagnostics" / "wn-pattern-overfit.yml"
    config = load_experiment_config(path, env=RUNTIME_ENV)

    assert config.experiment["profile"] == "diagnostic"
    assert config.experiment["name"] == "diagnostic-wn-pattern-overfit-v3"
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
    assert first.runtime_paths != second.runtime_paths


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

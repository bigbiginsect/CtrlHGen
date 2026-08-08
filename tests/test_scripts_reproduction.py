from __future__ import annotations

import ast
from pathlib import Path
import re
import shlex
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "reproduce"
EXPECTED_COMMANDS = {
    "sample.sh": [
        "python", "-m", "akgr.sampling.sample_parallel",
        "--experiment-config", "$CONFIG",
    ],
    "sft-unconditional.sh": [
        "python", "-m", "akgr.abduction_model.main",
        "--experiment-config", "$CONFIG",
        "--mode", "training",
        "--stage", "unconditional",
    ],
    "sft-conditional.sh": [
        "python", "-m", "akgr.abduction_model.main",
        "--experiment-config", "$CONFIG",
        "--mode", "training",
        "--stage", "conditional",
        "--parent-checkpoint", "$CHECKPOINT",
    ],
    "evaluate.sh": [
        "python", "-m", "akgr.abduction_model.main",
        "--experiment-config", "$CONFIG",
        "--mode", "testing",
        "--checkpoint", "$CHECKPOINT",
    ],
    "grpo.sh": [
        "python", "-m", "akgr.abduction_model.main",
        "--experiment-config", "$CONFIG",
        "--mode", "optimizing",
        "--parent-checkpoint", "$CHECKPOINT",
    ],
}
GPU_SCRIPTS = set(EXPECTED_COMMANDS) - {"sample.sh"}


def _script_text(name: str) -> str:
    return (SCRIPT_DIR / name).read_text(encoding="utf-8")


def _module_command(text: str) -> list[str]:
    normalized = text.replace("\\\n", " ")
    match = re.search(r"^python -m [^\n]+", normalized, flags=re.MULTILINE)
    assert match, "script must contain exactly one Python module invocation"
    return shlex.split(match.group(0))


def _declared_cli_flags(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                if argument.value.startswith("--"):
                    flags.add(argument.value)
    return flags


def test_reproduction_script_set_and_exact_cli_contract() -> None:
    actual = {path.name for path in SCRIPT_DIR.glob("*.sh")}
    assert actual == set(EXPECTED_COMMANDS)

    for name, expected in EXPECTED_COMMANDS.items():
        text = _script_text(name)
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert text.count("python -m ") == 1
        assert _module_command(text) == expected
        for variable in (
            "CTRLHGEN_DATA_ROOT",
            "CTRLHGEN_CHECKPOINT_ROOT",
            "CTRLHGEN_RUN_ROOT",
        ):
            assert variable in text


def test_wrapper_flags_are_declared_by_the_target_entrypoints() -> None:
    entrypoints = {
        "akgr.sampling.sample_parallel": REPO_ROOT / "akgr" / "sampling" / "sample_parallel.py",
        "akgr.abduction_model.main": REPO_ROOT / "akgr" / "abduction_model" / "main.py",
    }
    declared = {module: _declared_cli_flags(path) for module, path in entrypoints.items()}

    for expected in EXPECTED_COMMANDS.values():
        module = expected[2]
        forwarded_flags = {token for token in expected[3:] if token.startswith("--")}
        assert forwarded_flags <= declared[module]


@pytest.mark.parametrize("name", sorted(EXPECTED_COMMANDS))
def test_reproduction_scripts_are_valid_bash(name: str) -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_DIR / name)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", sorted(EXPECTED_COMMANDS))
def test_reproduction_scripts_reject_wrong_arity_before_side_effects(name: str) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / name)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 64
    assert "Usage:" in result.stderr


def test_gpu_scripts_fail_fast_without_hard_coded_gpu_selection() -> None:
    for name in EXPECTED_COMMANDS:
        text = _script_text(name)
        assert "CUDA_VISIBLE_DEVICES" not in text
        assert not re.search(r"cuda:[1-9]", text, flags=re.IGNORECASE)
        if name in GPU_SCRIPTS:
            assert "torch.cuda.is_available()" in text
        else:
            assert "torch.cuda.is_available()" not in text


def test_readme_keeps_phase_and_legacy_boundaries_explicit() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Phase A" in readme and "does not download WN18RR" in readme
    assert "Phase B" in readme and "tiny end-to-end smoke" in readme
    assert "Scripts outside `scripts/reproduce/`" in readme
    for name in EXPECTED_COMMANDS:
        assert f"scripts/reproduce/{name}" in readme

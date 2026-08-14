#!/usr/bin/env python3
"""Rebuild the figures used by the CtrlHGen reproduction report.

The script consumes only compact, byte-for-byte copies of DSW JSON/JSONL
records under ``worklogs/reproduction/report-data``. It intentionally does not read model
checkpoints or per-example prediction files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_LABELS = {
    "c1": "C-I: initial small (12L)",
    "c2": "C-II: repaired small (12L)",
    "c3": "C-III: author-aligned small (6L)",
    "c4": "C-IV: full train (6L)",
}

COLORS = {
    "c1": "#9e9e9e",
    "c2": "#4472c4",
    "c3": "#ed7d31",
    "c4": "#2e8b57",
    "original": "#c44e52",
    "repaired-pilot": "#8172b3",
    "repaired-full": "#2e8b57",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def phase_c_records(data_root: Path, run: str, stage: str) -> list[dict[str, Any]]:
    return load_jsonl(data_root / "phase-c" / run / f"{stage}-history.jsonl")


def validation_series(
    rows: list[dict[str, Any]], metric: str
) -> tuple[list[int], list[float]]:
    points = [row for row in rows if row.get("validation")]
    epochs: list[int] = []
    values: list[float] = []
    for row in points:
        value = row["validation"].get(metric)
        if value is not None:
            epochs.append(int(row["stage_epoch"]))
            values.append(float(value))
    return epochs, values


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_c_losses(data_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, stage, title in zip(
        axes,
        ("unconditional", "conditional"),
        ("Unconditional SFT", "Conditional SFT"),
        strict=True,
    ):
        for run in RUN_LABELS:
            rows = phase_c_records(data_root, run, stage)
            ax.plot(
                [row["stage_epoch"] for row in rows],
                [row["train_loss"] for row in rows],
                label=RUN_LABELS[run],
                color=COLORS[run],
                linewidth=1.8,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (log scale)")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Phase C training loss across four runs", fontweight="bold")
    fig.subplots_adjust(bottom=0.23, top=0.86, wspace=0.22)
    save(fig, output_dir, "phase-c-loss-curves.png")


def plot_phase_c_validation(data_root: Path, output_dir: Path) -> None:
    metrics = (
        ("jaccard", "Jaccard"),
        ("condition_accuracy", "Pattern accuracy"),
        ("smatch", "Smatch"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for run in RUN_LABELS:
            rows = phase_c_records(data_root, run, "conditional")
            epochs, values = validation_series(rows, metric)
            if values:
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    markersize=3,
                    label=RUN_LABELS[run],
                    color=COLORS[run],
                    linewidth=1.7,
                )
        ax.set_title(label)
        ax.set_xlabel("Conditional SFT epoch")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Validation metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Phase C conditional validation trajectories", fontweight="bold")
    fig.subplots_adjust(bottom=0.24, top=0.84, wspace=0.12)
    save(fig, output_dir, "phase-c-conditional-validation.png")


def grpo_log(data_root: Path, run: str) -> list[dict[str, Any]]:
    state = load_json(data_root / "phase-d" / run / "trainer_state.json")
    return [row for row in state["log_history"] if "reward" in row and "step" in row]


def plot_phase_d_dynamics(data_root: Path, output_dir: Path) -> None:
    runs = (
        ("original", "Original Phase D"),
        ("repaired-pilot", "Repaired pilot"),
        ("repaired-full", "Repaired full"),
    )
    metrics = (
        ("reward", "Combined reward", False),
        ("reward_std", "Within-group reward std", False),
        ("kl", "Reverse-KL estimate", True),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (metric, label, log_scale) in zip(axes, metrics, strict=True):
        for run, run_label in runs:
            rows = grpo_log(data_root, run)
            points = [(row["step"], row.get(metric)) for row in rows]
            points = [(step, value) for step, value in points if value is not None]
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=run_label,
                color=COLORS[run],
            )
        ax.set_title(label)
        ax.set_xlabel("Optimizer step (one GRPO epoch)")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Logged 500-step aggregate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Phase D training dynamics before and after repair", fontweight="bold")
    fig.subplots_adjust(bottom=0.22, top=0.84, wspace=0.22)
    save(fig, output_dir, "phase-d-training-dynamics.png")


def plot_phase_d_semantic_gains(data_root: Path, output_dir: Path) -> None:
    sources = (
        ("repaired-pilot", "Pilot validation"),
        ("repaired-full", "Full validation"),
        ("repaired-test", "Frozen test"),
    )
    values: dict[str, list[float]] = {"greedy": [], "sampled": []}
    errors: dict[str, tuple[list[float], list[float]]] = {
        "greedy": ([], []),
        "sampled": ([], []),
    }
    for directory, _ in sources:
        filename = "comparison.json" if directory == "repaired-test" else "validation-comparison.json"
        comparison = load_json(data_root / "phase-d" / directory / filename)
        for decoding in ("greedy", "sampled"):
            stats = comparison[decoding]["paired_semantic_average"]
            mean = float(stats["mean"])
            values[decoding].append(mean)
            errors[decoding][0].append(mean - float(stats["ci95_low"]))
            errors[decoding][1].append(float(stats["ci95_high"]) - mean)

    x = np.arange(len(sources))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, decoding, color in (
        (-width / 2, "greedy", "#4472c4"),
        (width / 2, "sampled", "#ed7d31"),
    ):
        ax.bar(
            x + offset,
            values[decoding],
            width,
            yerr=np.array(errors[decoding]),
            capsize=4,
            label=decoding.capitalize(),
            color=color,
            alpha=0.9,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in sources])
    ax.set_ylabel("Delta of mean(Jaccard, Dice, Overlap)")
    ax.set_title("Repaired Phase D semantic gain with paired 95% bootstrap CI", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, output_dir, "phase-d-semantic-gains.png")


def plot_final_comparison(data_root: Path, output_dir: Path) -> None:
    comparison = load_json(data_root / "phase-d" / "repaired-test" / "comparison.json")
    metric_keys = ("jaccard", "dice", "overlap", "condition_accuracy", "smatch")
    metric_labels = ("Jaccard", "Dice", "Overlap", "Pattern acc.", "Smatch")
    series = {
        "Paper w/o RL": [0.715, 0.758, 0.837, 0.815, 0.790],
        "Our SFT parent": [comparison["greedy"]["baseline"][key] for key in metric_keys],
        "Our repaired GRPO": [comparison["greedy"]["candidate"][key] for key in metric_keys],
        "Paper CtrlHGen": [0.770, 0.808, 0.868, 0.935, 0.833],
    }
    colors = ("#b7b7b7", "#4472c4", "#2e8b57", "#222222")
    x = np.arange(len(metric_keys))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for index, ((label, values), color) in enumerate(zip(series.items(), colors, strict=True)):
        offset = (index - 1.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.9)
    ax.set_xticks(x, metric_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Greedy frozen-test metric")
    ax.set_title("Final reproduction result versus paper Table 3", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.25)
    save(fig, output_dir, "final-metric-comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("worklogs/reproduction/report-data"),
        help="Directory containing the compact DSW records.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("worklogs/reproduction/figures"),
        help="Directory in which PNG figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_phase_c_losses(args.data_root, args.output_dir)
    plot_phase_c_validation(args.data_root, args.output_dir)
    plot_phase_d_dynamics(args.data_root, args.output_dir)
    plot_phase_d_semantic_gains(args.data_root, args.output_dir)
    plot_final_comparison(args.data_root, args.output_dir)
    print(f"Wrote five figures to {args.output_dir}")


if __name__ == "__main__":
    main()

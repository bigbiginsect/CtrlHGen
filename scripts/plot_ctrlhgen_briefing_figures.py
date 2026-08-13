#!/usr/bin/env python3
"""Plot the restrained, presentation-specific CtrlHGen reproduction figures.

The existing report figures remain untouched.  These variants use the same
JSON/JSONL sources, Times New Roman, a grayscale/deep-blue palette, and honest
axes: every absolute metric chart is shown on [0, 1].
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ctrlhgen-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "worklogs/report-data"
OUTPUT = ROOT / "worklogs/figures/briefing-minimal"

BLACK = "#1A1A1A"
DARK_GRAY = "#666666"
MID_GRAY = "#A6A6A6"
LIGHT_GRAY = "#D9D9D9"
BLUE = "#17365D"
LIGHT_BLUE = "#8EA9DB"

RUNS = {
    "c1": ("C-I initial 12L", LIGHT_GRAY, ":"),
    "c2": ("C-II repaired 12L", MID_GRAY, "--"),
    "c3": ("C-III aligned 6L", DARK_GRAY, "-."),
    "c4": ("C-IV full 6L", BLUE, "-"),
}


def configure_style() -> None:
    for candidate in (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    ):
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.8,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "text.color": BLACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def phase_c_series(data_root: Path, run: str, metric: str) -> tuple[list[int], list[float]]:
    rows = load_jsonl(data_root / "phase-c" / run / "conditional-history.jsonl")
    points = [row for row in rows if row.get("validation") and metric in row["validation"]]
    return (
        [int(row["stage_epoch"]) for row in points],
        [float(row["validation"][metric]) for row in points],
    )


def plot_conditional_loss(data_root: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.15))
    for run, (label, color, linestyle) in RUNS.items():
        rows = load_jsonl(data_root / "phase-c" / run / "conditional-history.jsonl")
        ax.plot(
            [int(row["stage_epoch"]) for row in rows],
            [float(row["train_loss"]) for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0 if run == "c4" else 1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Conditional SFT epoch")
    ax.set_ylabel("Training loss (log scale)")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.43), ncol=2, fontsize=9.5)
    fig.subplots_adjust(bottom=0.37)
    save(fig, output_dir / "phase-c-conditional-loss.png")


def plot_phase_c_key(data_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("jaccard", "condition_accuracy"),
        ("Jaccard", "Pattern accuracy"),
        strict=True,
    ):
        for run, (label, color, linestyle) in RUNS.items():
            epochs, values = phase_c_series(data_root, run, metric)
            ax.plot(
                epochs,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2.0 if run == "c4" else 1.5,
                marker="o" if run == "c4" else None,
                markersize=3.2,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Conditional SFT epoch")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation metric (0–1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10)
    fig.subplots_adjust(bottom=0.22, wspace=0.12)
    save(fig, output_dir / "phase-c-jaccard-pattern-accuracy.png")


def grpo_rows(data_root: Path, run: str) -> list[dict[str, Any]]:
    state = load_json(data_root / "phase-d" / run / "trainer_state.json")
    return [row for row in state["log_history"] if "step" in row and "reward_std" in row]


def plot_grpo_signal(data_root: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25))
    for ax, metric, title, log_y in (
        (axes[0], "reward_std", "Within-group reward standard deviation", False),
        (axes[1], "kl", "Reverse-KL estimate", True),
    ):
        for run, label, color, linestyle in (
            ("original", "Original Phase D", DARK_GRAY, "--"),
            ("repaired-full", "Repaired full", BLUE, "-"),
        ):
            rows = grpo_rows(data_root, run)
            ax.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker="o" if run == "repaired-full" else "x",
                markersize=3.5,
                label=label,
            )
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Optimizer step")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Logged 500-step aggregate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.20, wspace=0.25)
    save(fig, output_dir / "phase-d-grpo-signal.png")


def plot_semantic_gains(data_root: Path, output_dir: Path) -> None:
    sources = (
        ("repaired-pilot", "Pilot validation", "validation-comparison.json"),
        ("repaired-full", "Full validation", "validation-comparison.json"),
        ("repaired-test", "Frozen test", "comparison.json"),
    )
    x = np.arange(len(sources))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.3, 4.15))
    for offset, decoding, color, hatch in (
        (-width / 2, "greedy", BLUE, None),
        (width / 2, "sampled", "white", "////"),
    ):
        means, low, high = [], [], []
        for directory, _, filename in sources:
            stats = load_json(data_root / "phase-d" / directory / filename)[decoding][
                "paired_semantic_average"
            ]
            mean = float(stats["mean"])
            means.append(mean)
            low.append(mean - float(stats["ci95_low"]))
            high.append(float(stats["ci95_high"]) - mean)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=np.array([low, high]),
            capsize=4,
            label=decoding.capitalize(),
            color=color,
            edgecolor=BLUE,
            hatch=hatch,
            linewidth=0.9,
        )
    ax.axhline(0, color=BLACK, linewidth=0.8)
    ax.set_xticks(x, [label for _, label, _ in sources])
    ax.set_ylabel("Δ mean(Jaccard, Dice, Overlap)")
    ax.set_ylim(-0.005, 0.065)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    save(fig, output_dir / "phase-d-semantic-gains.png")


def plot_final_comparison(data_root: Path, output_dir: Path) -> None:
    comparison = load_json(data_root / "phase-d" / "repaired-test" / "comparison.json")
    keys = ("jaccard", "dice", "overlap", "condition_accuracy", "smatch")
    labels = ("Jaccard", "Dice", "Overlap", "Pattern acc.", "Smatch")
    series = (
        ("Paper w/o RL", [0.715, 0.758, 0.837, 0.815, 0.790], "white", MID_GRAY, "////"),
        (
            "Our SFT parent",
            [comparison["greedy"]["baseline"][key] for key in keys],
            MID_GRAY,
            MID_GRAY,
            None,
        ),
        (
            "Our repaired GRPO",
            [comparison["greedy"]["candidate"][key] for key in keys],
            BLUE,
            BLUE,
            None,
        ),
        ("Paper CtrlHGen", [0.770, 0.808, 0.868, 0.935, 0.833], BLACK, BLACK, None),
    )
    x = np.arange(len(keys))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.7, 4.55))
    for index, (label, values, face, edge, hatch) in enumerate(series):
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=face,
            edgecolor=edge,
            hatch=hatch,
            linewidth=0.9,
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Greedy frozen-test metric (0–1)")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=4, fontsize=9.5)
    fig.subplots_adjust(bottom=0.24)
    save(fig, output_dir / "final-metric-comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_style()
    plot_conditional_loss(args.data_root, args.output_dir)
    plot_phase_c_key(args.data_root, args.output_dir)
    plot_grpo_signal(args.data_root, args.output_dir)
    plot_semantic_gains(args.data_root, args.output_dir)
    plot_final_comparison(args.data_root, args.output_dir)
    print(f"Wrote minimalist briefing figures to {args.output_dir}")


if __name__ == "__main__":
    main()

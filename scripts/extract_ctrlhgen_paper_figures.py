#!/usr/bin/env python3
"""Extract the paper visuals used by the minimalist teacher briefing.

This script intentionally keeps the original paper artwork and typography.  It
only rasterizes tightly specified regions of the repository PDF at 300 dpi.
PyMuPDF is required (``pip install pymupdf``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit("PyMuPDF is required: python -m pip install pymupdf") from exc


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/Controllable_Logical_Hypothesis_Generation.pdf"
OUTPUT = ROOT / "worklogs/reproduction/figures/paper"

# Coordinates are in PDF points (1/72 inch), after visual inspection of the
# repository copy.  Page indices are zero-based.
CROPS = {
    "paper-fig1-controllability.png": (1, (72, 62, 540, 255)),
    "paper-fig2-challenges.png": (2, (78, 52, 542, 200)),
    "paper-fig3-framework.png": (3, (88, 55, 542, 352)),
    "paper-fig4-patterns.png": (6, (300, 440, 548, 560)),
    "paper-table1-main-results.png": (7, (68, 45, 548, 350)),
    "paper-table3-reward-ablation.png": (9, (78, 58, 540, 182)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.paper)
    matrix = fitz.Matrix(300 / 72, 300 / 72)
    for name, (page_index, rectangle) in CROPS.items():
        pixmap = document[page_index].get_pixmap(
            matrix=matrix, clip=fitz.Rect(*rectangle), alpha=False
        )
        path = args.output_dir / name
        pixmap.save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

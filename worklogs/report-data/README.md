# Reproduction report data

This directory contains the compact, immutable training records copied from the
DSW persistent volume for rebuilding the figures in the comprehensive
reproduction report. Large checkpoints, per-example evaluation records, and
full stdout logs remain under `/mnt/workspace/ctrlhgen-*` on DSW.

The files are copied byte-for-byte and indexed by `sha256sums.txt`. Run
`python scripts/plot_reproduction_metrics.py` from the repository root to
rebuild the report figures.

Directory mapping:

| Local directory | DSW source |
|---|---|
| `phase-c/c1` | `.../repro-wn-pattern-phase-c-seed42-20260809-111557/repro-wn-pattern-small` |
| `phase-c/c2` | `.../repro-wn-pattern-small-v2` |
| `phase-c/c3` | `.../repro-wn-pattern-small-author-aligned-v3` |
| `phase-c/c4` | `.../repro-wn-pattern-full-train-author-aligned-c4` |
| `phase-d/original` | original epoch-45-parent GRPO checkpoint/run records |
| `phase-d/repaired-pilot` | `.../phase-d-repaired-pilot-20260811/control` |
| `phase-d/repaired-full` | `.../phase-d-repaired-full-20260811/control` |
| `phase-d/repaired-test` | `.../phase-d-repaired-frozen-test-20260812` |

The ellipsis denotes `/mnt/workspace/ctrlhgen-runs` except for the original
GRPO `trainer_state.json`, which comes from the experiment checkpoint root.

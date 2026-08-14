# Phase C Local Preflight Worklog

Date: 2026-08-09 (Asia/Shanghai)

## Scope

This change prepares the supported WN18RR pattern-conditioning path for Phase C
and D without starting either experiment.  Work was performed only in the
local checkout because the DSW instance was not available.  The live GPU
integration run remains pending until the user starts DSW.

The local base revision was `22df9d5` on
`codex/reproduction-pipeline`.  The delivery commit containing this worklog is
reported in the task handoff because a tracked file cannot contain its own Git
hash.

## Fixed policy

For `wn-pattern-small.yml`:

- SFT validation runs every 2 stage epochs and at the final epoch, with batch
  size 16 and greedy decoding.  Per-sample JSONL, aggregate JSON, and append-only
  stage histories are retained as the observation record.
- Unconditional SFT selects minimum teacher-forced validation loss, with
  Jaccard as a tie-break.  Conditional SFT selects maximum validation Pattern
  Accuracy, then Jaccard, then minimum validation loss.  Test data is not used
  for selection.
- Periodic SFT checkpoints are written every 5 stage epochs.  A new best and
  the final epoch are also saved.  Pruning keeps the latest 2 checkpoints plus
  the current best, bounding the normal steady state at 3 checkpoint
  directories per SFT stage.
- Each best checkpoint has a JSON selection record and a stable relative
  symlink.  The selection record contains the validation metrics, stage epoch,
  global step, checkpoint directory, and ordered selection rule.
- GRPO saves every 100 optimizer steps and keeps the latest 2 Trainer
  checkpoints.  This replaces the Phase B emergency save-every-step behavior
  while preserving a recent resume point and enough progress observations.

The tiny profile uses validation/checkpoint cadence 1/1 and GRPO cadence 10;
the full profile uses 10/25 and 500.  All profiles keep two recent SFT/GRPO
checkpoints, with SFT additionally protecting its selected best checkpoint.

## CUDA determinism decision

Phase B completed two SFT stages and all 204 GRPO optimizer steps despite the
CuBLAS and memory-efficient-attention deterministic-warn messages.  These
operations can introduce small run-to-run floating-point differences, so the
warning limits a claim of bitwise replay; it does not indicate invalid loss,
gradients, optimizer state, or checkpoint recovery.  Forcing alternative
attention kernels could materially change runtime and memory use and still
would not make the existing Phase B result bit-identical.  Per user direction,
the training kernel path is unchanged.  Phase C/D will retain exact config,
data hashes, seeds, validation histories, and checkpoints, and repeated-seed
comparisons remain the appropriate robustness check.

## Local validation

Targeted tests cover config validation, cadence including forced-final saves,
stage-specific best selection, JSON/symlink publication, bounded pruning,
sample-weighted validation loss, evaluation records, synthetic SFT, checkpoint
loading, and GRPO config construction.  The recorded local command is:

```bash
PYTHONPATH=/tmp/ctrlhgen-phase-c-pytest \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
  tests/test_config_reproduction.py \
  tests/test_sft_selection.py \
  tests/test_grpo_config.py \
  tests/test_evaluation_records.py \
  tests/test_model_sft_synthetic.py \
  tests/test_checkpoint_directory.py
```

Result: `33 passed, 1 skipped in 6.14s`.  The skip is the TRL integration test
because the local machine does not have the complete pinned reproduction
environment.  No replacement PyTorch installation was made.

The complete locally collectable suite passed as `78 passed, 4 skipped in
7.36s`, and the final post-edit repeat passed with the same counts in `5.45s`.
The four skips are dependency/GPU-gated tests.  The following checks also
exited successfully:

```bash
python3 -m compileall -q akgr tests
bash -n scripts/reproduce/*.sh
git diff --check
```

## DSW acceptance still required

After DSW becomes available, deploy the exact delivery SHA and verify the
pinned Python/PyTorch/TRL environment.  Before starting Phase C, run the
offline/synthetic suite and a bounded live check that confirms validation
artifacts, best-pointer recovery, retention after several save events, and the
GRPO `save_steps=100`/`save_total_limit=2` Trainer configuration.  This local
preflight is not a Phase C result and contains no paper-scale metrics.

# CtrlHGen Agent Working Agreement

## Scope and objective

This file applies to the entire repository.

The project goal is to repair the incomplete official implementation of
"Controllable Logical Hypothesis Generation for Abductive Reasoning in
Knowledge Graphs", reproduce its reported results under limited local compute,
and then develop controlled improvements.

The paper is available at:

`paper/Controllable_Logical_Hypothesis_Generation.pdf`

Read the paper and the relevant source/configuration files before making
algorithmic claims or changing experiment semantics.

## Source-of-truth model

Use the local checkout and the private GitHub repository as the source of truth.
Use the Alibaba Cloud DSW instance as a compute worker, not as an independent
development copy.

- `origin`: `git@github.com:bigbiginsect/CtrlHGen.git` (private working repo)
- `upstream`: `https://github.com/HKUST-KnowComp/CtrlHGen.git` (authors' repo)
- Local checkout: code editing, review, lightweight checks, commits, and pushes
- GitHub `origin`: durable collaboration and rollback history
- DSW: environment setup, data preparation, GPU smoke tests, training, and evaluation

Never push work to `upstream`. Do not casually edit code in both the local and
DSW checkouts. If an emergency edit is made on DSW, put it on a dedicated branch
and bring it back through Git rather than manually maintaining two versions.

## Git workflow

Before changing files:

1. Run `git status --short --branch` and preserve unrelated user changes.
2. Confirm `origin` and `upstream` with `git remote -v`.
3. Fetch when remote state matters; do not assume the checkout is current.
4. Create a focused branch named `codex/<topic>` for non-trivial repair or
   experiment work unless the user explicitly requests another strategy.

While working:

- Make small, reviewable commits with one purpose each.
- Stage explicit paths when the worktree contains mixed changes.
- Do not commit generated data, checkpoints, caches, credentials, or bulky logs.
- Do not use force-push, destructive reset, or history rewriting without the
  user's explicit approval.
- Keep `main` in a reproducible state. Push completed local commits to `origin`
  before using them on DSW.

The normal code handoff is:

```bash
# Local machine
git status --short --branch
git switch -c codex/<topic>
# edit and validate
git add <explicit-paths>
git commit -m "<focused change>"
git push -u origin codex/<topic>
git rev-parse HEAD
```

Record the resulting commit SHA. Experiments must identify an exact SHA, not
just a moving branch name.

## Local SSH access

The WSL user has dedicated SSH configuration outside this repository.

```bash
ssh ctrlhgen-dsw
ssh -T github.com
```

`ctrlhgen-dsw` currently resolves to:

- host: `47.93.100.200`
- port: `1024`
- user: `root`

The DSW endpoint may change when the instance is recreated. If the alias fails,
inspect `~/.ssh/config` and ask the user for the current DSW endpoint. Never
print, copy into the repository, or otherwise expose private keys or tokens.

## DSW operating rules

Treat `/mnt/workspace` as the persistent work area. Avoid placing important
artifacts only under `/root`, `/tmp`, or other container-layer paths.

Preferred layout (verify or create it when remote setup is in scope):

```text
/mnt/workspace/CtrlHGen/                 # Git checkout
/mnt/workspace/ctrlhgen-data/            # downloaded and sampled datasets
/mnt/workspace/ctrlhgen-checkpoints/     # model checkpoints
/mnt/workspace/ctrlhgen-runs/            # logs and experiment outputs
/mnt/workspace/envs/ctrlhgen/            # project environment
/mnt/workspace/cache/                    # package/model/dataset caches
```

Observed DSW baseline on 2026-08-08 (always re-check before relying on it):

- Ubuntu 22.04
- one NVIDIA L20 with about 48 GiB VRAM
- about 123 GiB system RAM
- NVIDIA driver 550.163.01
- system Python 3.11.11
- PyTorch 2.6.0+cu124 with CUDA available
- no Conda, micromamba, or uv initially installed
- about 195 GiB free under `/mnt/workspace` at inspection time

The private repository is not guaranteed to be cloneable from DSW yet. The
intended setup is a repository-scoped, read-only GitHub deploy key. Do not copy a
personal GitHub token or the local user's GitHub private key onto DSW. Until
read-only access is configured, report the blocker rather than weakening access
controls.

Once DSW can read `origin`, deploy an exact local commit as follows:

```bash
# Local machine
git push origin <branch>
git rev-parse HEAD

# DSW
ssh ctrlhgen-dsw
cd /mnt/workspace/CtrlHGen
git status --short --branch
git fetch origin --prune
git switch --detach <exact-commit-sha>
```

If the DSW checkout is dirty, stop and inspect it. Do not discard remote changes
silently. Detached-SHA experiment checkouts are preferred because they prevent a
later branch update from changing the meaning of a running experiment.

Do not install packages, download large datasets, start costly GPU jobs, or
delete DSW artifacts unless those actions are within the user's current request.
Before a long job, state the command, expected outputs, rough resource use, and
where logs/checkpoints will be written.

## Experiment record requirements

Every meaningful run should record at least:

- Git commit SHA and branch (if any)
- full command line
- configuration files or overrides
- dataset name, split, scale, and data version/checksum when practical
- random seeds
- Python/package/CUDA/PyTorch versions
- GPU model and GPU count
- start/end time and runtime
- checkpoint and log locations
- primary metrics and whether they match a paper table/figure
- failures, deviations from the paper, and rerun notes

Store large artifacts outside the Git checkout. Commit only lightweight configs,
manifests, scripts, result summaries, and documentation needed to understand or
repeat a run. Keep credentials out of shell history, logs, configs, W&B files,
and commits.

For long-running work, use a durable terminal/session mechanism available on
DSW and stream logs into `/mnt/workspace/ctrlhgen-runs/<run-id>/`. Make jobs
restartable and checkpointed where practical.

## Reproduction sequence

Prefer incremental evidence over launching the full paper pipeline immediately:

1. Audit paper-to-code mappings and list missing inputs/claims.
2. Repair and lock a reproducible environment.
3. Run import/static checks and unit tests locally where possible.
4. Establish a tiny/debug sampling run on DSW.
5. Establish a tiny supervised-training and evaluation smoke test.
6. Reproduce one unconditional baseline on one dataset.
7. Add conditional training and verify each condition independently.
8. Add reinforcement learning only after the supervised baseline is stable.
9. Reproduce remaining datasets/tables, then begin improvements and ablations.

Do not claim reproduction from a successful launch alone. Compare metrics,
dataset scale, seeds, training budget, and evaluation behavior with the paper.

## Known initial repository issues

These observations are starting points, not permanent truths; verify them before
repairing:

- `requirements.txt` is a Conda-style export despite README instructions to use
  `pip install -r requirements.txt`.
- The repository initially contains no sampled datasets or checkpoints.
- Sampling configuration for `full` and the WN18RR/FB15k-237 training scripts do
  not appear to line up.
- Several scripts contain fixed GPU IDs or machine-specific paths.
- Some optional imports/dependencies (for example PEFT/Optimum paths) are not
  fully declared.
- The checked-in environment description targets Python 3.9/CUDA 11.8, while
  the inspected DSW base image used Python 3.11/CUDA 12.4.
- README does not yet map commands and expected outputs to every paper result.

Avoid hiding these incompatibilities with ad hoc machine-only fixes. Prefer
portable configuration, pinned dependencies, documented commands, and tests.

## Repository hygiene and validation

Respect `.gitignore`. In particular, do not commit:

- `__pycache__`, virtual environments, or tool caches
- `sampled_data` contents
- checkpoints or model weight files
- experiment logs, W&B state, or large output directories
- `.env` files, SSH material, tokens, or cloud credentials

Before handing work back:

1. Run the most relevant proportional checks.
2. Run `git diff --check`.
3. Review `git status --short --branch` and the final diff.
4. State exactly which checks ran and which could not run.
5. Push the intended branch/commit to `origin` when requested or when continuing
   the established local-to-DSW workflow.
6. For DSW runs, report the exact SHA, run directory, process/job state, and
   artifact locations.
7. Leave the worktree understandable to the next agent; document active
   blockers instead of relying on chat history.

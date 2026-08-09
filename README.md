# CtrlHGen

This repository is the authors' implementation of **Controllable Logical
Hypothesis Generation for Abductive Reasoning in Knowledge Graphs**.  The
maintained reproduction path in this working copy is deliberately narrower:
**WN18RR with pattern conditioning on one CUDA GPU**.

The original author scripts remain available for reference, but they are not
the supported reproduction interface described below.

## Reproduction environment

The verified target is Linux, Python 3.11, PyTorch 2.6.0, and CUDA 12.4.  Create
a clean environment and install the CUDA build of PyTorch from its official
wheel index before installing the project dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-repro.txt
python -m pip check
```

`requirements-repro.txt` intentionally does not contain PyTorch, so a later
ordinary PyPI install cannot replace the CUDA 12.4 wheel.  The old
`requirements.txt` is a Conda export and is **not** a pip requirements file.
`requirements-legacy-optional.txt` lists packages used only by author-era
utilities outside the supported path.

All runtime locations come from environment variables.  Generated data,
checkpoints, caches, and large logs must remain outside the Git checkout:

```bash
export CTRLHGEN_DATA_ROOT=/mnt/workspace/ctrlhgen-data
export CTRLHGEN_CHECKPOINT_ROOT=/mnt/workspace/ctrlhgen-checkpoints
export CTRLHGEN_RUN_ROOT=/mnt/workspace/ctrlhgen-runs
export HF_HOME=/mnt/workspace/cache/huggingface
export PYKEEN_HOME=/mnt/workspace/cache/pykeen
export TRITON_CACHE_DIR=/mnt/workspace/cache/triton
export NLTK_DATA=/mnt/workspace/cache/nltk
```

The scripts do not select a physical GPU.  GPU-required stages check
`torch.cuda.is_available()` and exit instead of silently falling back to CPU.
Select GPU visibility in the calling environment when needed.

## Phase A: engineering baseline

Phase A validates configuration, deterministic components, checkpoint
contracts, and synthetic CPU/GPU smoke tests.  It does not download WN18RR,
sample the live graph, or claim that the end-to-end paper pipeline has run.
For a disposable Phase A validation, point the required runtime variables at
temporary directories:

```bash
export CTRLHGEN_DATA_ROOT=/tmp/ctrlhgen-phase-a/data
export CTRLHGEN_CHECKPOINT_ROOT=/tmp/ctrlhgen-phase-a/checkpoints
export CTRLHGEN_RUN_ROOT=/tmp/ctrlhgen-phase-a/runs

python -m akgr.abduction_model.main --help
python -m akgr.sampling.sample_parallel --help
pytest -m "not gpu and not live_data"
```

On the DSW L20, verify CUDA and then run the synthetic GPU layer separately:

```bash
python -c 'import torch; assert torch.__version__.startswith("2.6.0"); assert torch.version.cuda == "12.4"; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
pytest -m "gpu and not live_data"
```

Passing Phase A means the implementation baseline and offline/synthetic checks
are ready.  It does not mean the real WN18RR chain has passed.

## Phase B: WN18RR tiny end-to-end smoke

Phase B is the first live-data run.  It may download WN18RR and the NLTK
WordNet corpus, and it uses the persistent runtime directories shown above.
Download WordNet explicitly; library imports and Phase A tests must not perform
this download:

```bash
python -m nltk.downloader -d "$NLTK_DATA" wordnet
```

Use the tiny experiment config and the dedicated wrappers in order:

```bash
CONFIG=akgr/configs/reproduce/wn-pattern-tiny.yml

bash scripts/reproduce/sample.sh "$CONFIG"
bash scripts/reproduce/sft-unconditional.sh "$CONFIG"

UNCONDITIONAL_CHECKPOINT=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-tiny-v2/unconditional-best
bash scripts/reproduce/sft-conditional.sh \
  "$CONFIG" "$UNCONDITIONAL_CHECKPOINT"

CONDITIONAL_CHECKPOINT=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-tiny-v2/conditional-best
bash scripts/reproduce/evaluate.sh "$CONFIG" "$CONDITIONAL_CHECKPOINT"
bash scripts/reproduce/grpo.sh "$CONFIG" "$CONDITIONAL_CHECKPOINT"

GRPO_CHECKPOINT=/mnt/workspace/ctrlhgen-checkpoints/repro-wn-pattern-tiny-v2/grpo/evaluation-step-<global-step>
bash scripts/reproduce/evaluate.sh "$CONFIG" "$GRPO_CHECKPOINT"
```

Checkpoint paths are intentionally explicit.  Conditional SFT and GRPO accept
only the selected checkpoint that passed the configured parse/EOS health gate;
evaluation can load any compatible checkpoint.  The
tiny profile is a smoke test, not a paper-result reproduction.  Only after it
passes should `wn-pattern-small.yml` be considered for a scaled experiment.
The `full` profile reflects an author-code scale clue and must not be presented
as a fully disclosed paper setting.

## Phase C/D checkpoint and validation policy

The `small` profile keeps the paper SFT schedule (400 unconditional epochs with
50 warm-up epochs, then 50 conditional epochs with 5 warm-up epochs) while
scaling the per-pattern data to 1024/128/128.  It validates every ten stage
epochs with deterministic
greedy decoding.  It writes every validation prediction and aggregate metric
under `$CTRLHGEN_RUN_ROOT/repro-wn-pattern-small-v2/validation/`, plus an
append-only stage history JSONL.  A candidate must first reach 90% parseability
and 98% EOS emission.  Unconditional selection then prioritizes parseability,
Jaccard, and validation loss; conditional selection prioritizes Pattern
Accuracy, parseability, Jaccard, and validation loss.  Selection never reads
the test split.

Small-profile SFT makes a periodic checkpoint every 25 stage epochs and also saves a newly
selected best checkpoint.  Pruning retains the two newest checkpoints and the
current best; the configured final epoch is always saved.  The selected model
is exposed as `unconditional-best.json`/`conditional-best.json` and a matching
directory symlink in the experiment checkpoint root.  Phase D GRPO saves every
100 optimizer steps and retains the latest two resumable Trainer checkpoints.
The `tiny` and `full` profiles use the same policy with cadence values scaled
for smoke testing and longer training, respectively.

Conditional prompts have the fixed token contract
`answers COND condition SEP target END`; `SEP` is never reused as the condition
delimiter.  Training pair batches always use right padding, while left padding
is scoped only to generation.  Reproduction checkpoint format v2 records and
hashes this tokenizer contract, so checkpoints created before the padding fix
are rejected instead of silently training on prompt tokens with `END` masked.

For a read-only checkpoint audit that compares teacher-forced next-token
predictions, prompt-only logits, and greedy generation on the same examples:

```bash
python -m akgr.abduction_model.sft_diagnostics \
  --experiment-config akgr/configs/diagnostics/wn-pattern-overfit.yml \
  --checkpoint /path/to/checkpoint --split train
```

Configuration identity is separated into a full experiment `semantic_hash`, a
sampled-data `data_hash`, and a KG split `kg_hash`.  Training-budget changes no
longer rebuild an identical KG or resample identical examples, while checkpoint
compatibility still requires the exact full experiment configuration.

Seeded CUDA runs may warn that CuBLAS or memory-efficient attention is not
bitwise deterministic.  These warnings do not invalidate training, but exact
bit-for-bit replay is not claimed; formal comparisons retain fixed data splits,
configuration, checkpoints, and run seeds, and should use the planned repeated
seeds when budget permits.

## Supported and legacy boundaries

- Supported reproduction path: WN18RR, 13 query patterns, pattern condition,
  two-stage SFT, evaluation, and GRPO through `scripts/reproduce/`.
- Phase A: no live dataset download and no real training run.
- Phase B: tiny live-data smoke; still not a paper metric reproduction.
- Scripts outside `scripts/reproduce/`, T5 experiments, other datasets, and old
  PPO flags are legacy artifacts.  They may contain author-machine paths,
  physical GPU assignments, or obsolete arguments and are not validated here.
- External reporting is disabled by the reproduction configs.  Enabling W&B
  for a later formal run is an explicit, separately recorded decision.

## Citation

```bibtex
@article{gao2025controllable,
  title={Controllable Logical Hypothesis Generation for Abductive Reasoning in Knowledge Graphs},
  author={Gao, Yisen and Bai, Jiaxin and Zheng, Tianshi and Sun, Qingyun and Zhang, Ziwei and Li, Jianxin and Song, Yangqiu and Fu, Xingcheng},
  journal={arXiv preprint arXiv:2505.20948},
  year={2025}
}
```

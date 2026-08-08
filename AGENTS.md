# CtrlHGen Working Notes

CtrlHGen is the official implementation of "Controllable Logical Hypothesis
Generation for Abductive Reasoning in Knowledge Graphs". This private repository
is the working copy of the authors' open-source code. The paper is available at
`paper/Controllable_Logical_Hypothesis_Generation.pdf`.

This file applies to the whole repository and records the local/DSW working
setup. The current user request defines the task at hand.

## Source of truth

- Local checkout: `/mnt/d/yang_nankai/CtrlHGen`
- Private working repo: `git@github.com:bigbiginsect/CtrlHGen.git` (`origin`)
- Authors' repo: `https://github.com/HKUST-KnowComp/CtrlHGen.git` (`upstream`)
- DSW checkout: `/mnt/workspace/CtrlHGen`

Edit code locally. Commit and push it to `origin`, then make DSW use that exact
commit SHA. Do not keep separate manual edits on the local and DSW checkouts,
and never push to `upstream`.

## Normal local-to-DSW workflow

On the local machine:

```bash
cd /mnt/d/yang_nankai/CtrlHGen
git status --short --branch
# edit and validate
git add <paths>
git commit -m "<message>"
git push origin HEAD
git rev-parse HEAD
```

On DSW, deploy the exact SHA printed above:

```bash
ssh ctrlhgen-dsw
cd /mnt/workspace/CtrlHGen
git status --short --branch
git fetch origin --prune
git switch --detach <exact-sha>
git rev-parse HEAD
```

If the DSW checkout is dirty, inspect it instead of discarding changes. If code
must be edited on DSW in an emergency, commit it and bring it back through Git;
do not maintain two versions by copying files manually.

If the DSW checkout is missing:

```bash
GIT_SSH_COMMAND="ssh -F /mnt/workspace/.ssh/config" \
  git clone git@github.com:bigbiginsect/CtrlHGen.git /mnt/workspace/CtrlHGen
cd /mnt/workspace/CtrlHGen
git config core.sshCommand "ssh -F /mnt/workspace/.ssh/config"
```

## DSW access and environment

Connect with `ssh ctrlhgen-dsw`. The alias currently points to
`root@47.93.100.200` on port `1024`; if the instance changes, inspect the local
`~/.ssh/config` or ask the user for the new endpoint.

DSW uses a repository-scoped, read-only deploy key stored under
`/mnt/workspace/.ssh/`. Never print, copy, replace, or commit the private key.

Persistent paths:

```text
/mnt/workspace/CtrlHGen/                 # Git checkout
/mnt/workspace/ctrlhgen-data/            # datasets
/mnt/workspace/ctrlhgen-checkpoints/     # checkpoints
/mnt/workspace/ctrlhgen-runs/            # logs and results
/mnt/workspace/envs/ctrlhgen/            # Python environment
/mnt/workspace/cache/                    # caches
```

Activate the environment with:

```bash
source /mnt/workspace/envs/ctrlhgen/bin/activate
export HF_HOME=/mnt/workspace/cache/huggingface
export TRITON_CACHE_DIR=/mnt/workspace/cache/triton
```

Verified on 2026-08-08: Ubuntu 22.04, one NVIDIA L20, Python 3.11.11, and
PyTorch 2.6.0+cu124 with CUDA available.

## Minimal safeguards

- Keep datasets, checkpoints, caches, and large logs outside the Git checkout.
- Never commit credentials, SSH material, generated data, checkpoints, or logs.
- Run expensive downloads or GPU jobs only when they are part of the current
  user request.
- For a long DSW job, record its exact Git SHA, command, log directory, process
  state, and output locations under `/mnt/workspace/ctrlhgen-runs/<run-id>/`.

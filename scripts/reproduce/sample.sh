#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <experiment-config.yml>" >&2
    exit 64
fi

: "${CTRLHGEN_DATA_ROOT:?Set CTRLHGEN_DATA_ROOT to the external data directory}"
: "${CTRLHGEN_CHECKPOINT_ROOT:?Set CTRLHGEN_CHECKPOINT_ROOT to the external checkpoint directory}"
: "${CTRLHGEN_RUN_ROOT:?Set CTRLHGEN_RUN_ROOT to the external run directory}"

CONFIG=$1
if [[ ! -r "$CONFIG" ]]; then
    echo "Experiment config is not readable: $CONFIG" >&2
    exit 66
fi

python -m akgr.sampling.sample_parallel \
    --experiment-config "$CONFIG"

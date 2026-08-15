#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <source-config.yml> <target-config.yml> <unconditional-checkpoint> <fresh-rl-manifest.json> <output-dir>" >&2
    exit 64
fi

: "${CTRLHGEN_DATA_ROOT:?Set CTRLHGEN_DATA_ROOT to the external data directory}"
: "${CTRLHGEN_CHECKPOINT_ROOT:?Set CTRLHGEN_CHECKPOINT_ROOT to the external checkpoint directory}"
: "${CTRLHGEN_RUN_ROOT:?Set CTRLHGEN_RUN_ROOT to the external run directory}"

python -m akgr.reproduction.sc_idc_phase2_data \
    --source-config "$1" \
    --target-config "$2" \
    --parent-checkpoint "$3" \
    --fresh-manifest "$4" \
    --output-dir "$5" \
    --signal-per-pattern 16

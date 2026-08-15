#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <target-config.yml> <unconditional-checkpoint> <sft-preflight.json> <output-report.json>" >&2
    exit 64
fi

: "${CTRLHGEN_DATA_ROOT:?Set CTRLHGEN_DATA_ROOT to the external data directory}"
: "${CTRLHGEN_CHECKPOINT_ROOT:?Set CTRLHGEN_CHECKPOINT_ROOT to the external checkpoint directory}"
: "${CTRLHGEN_RUN_ROOT:?Set CTRLHGEN_RUN_ROOT to the external run directory}"

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is required for specific-relation SFT preflight")'
python -m akgr.reproduction.sc_idc_sft_launch_preflight \
    --target-config "$1" \
    --parent-checkpoint "$2" \
    --phase2-data-manifest "$3" \
    --output "$4" \
    --batch-size 2

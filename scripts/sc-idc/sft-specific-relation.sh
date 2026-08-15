#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <target-config.yml> <unconditional-checkpoint> <sft-preflight.json>" >&2
    exit 64
fi

: "${CTRLHGEN_DATA_ROOT:?Set CTRLHGEN_DATA_ROOT to the external data directory}"
: "${CTRLHGEN_CHECKPOINT_ROOT:?Set CTRLHGEN_CHECKPOINT_ROOT to the external checkpoint directory}"
: "${CTRLHGEN_RUN_ROOT:?Set CTRLHGEN_RUN_ROOT to the external run directory}"

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is required for specific-relation SFT")'
python -m akgr.abduction_model.main \
    --experiment-config "$1" \
    --mode training \
    --stage conditional \
    --parent-checkpoint "$2" \
    --phase2-data-manifest "$3"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <target-config> <phase2-preflight> <conditional-parent> <output-dir>" >&2
  exit 2
fi

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is required for Experiment 1")'
python -m akgr.reproduction.sc_idc_signal_audit \
  --experiment-config "$1" \
  --phase2-data-manifest "$2" \
  --parent-checkpoint "$3" \
  --output-dir "$4" \
  --expected-signal-manifest-sha256 47774daa4bf3a034af4ab58ef0a39b13159958d65d8f2ccfb8a0ce35213e8914 \
  --reward-seed 42

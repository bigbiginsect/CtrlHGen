#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "usage: $0 <target-config> <phase2-preflight> <alpha-freeze> <conditional-parent> <pair-run-dir> <pair-checkpoint-dir> <max-steps> [save-steps]" >&2
    exit 64
fi

: "${CTRLHGEN_DATA_ROOT:?Set CTRLHGEN_DATA_ROOT to the external data directory}"
: "${CTRLHGEN_CHECKPOINT_ROOT:?Set CTRLHGEN_CHECKPOINT_ROOT to the external checkpoint directory}"
: "${CTRLHGEN_RUN_ROOT:?Set CTRLHGEN_RUN_ROOT to the external run directory}"

CONFIG=$1
PREFLIGHT=$2
ALPHA_FREEZE=$3
PARENT=$4
PAIR_RUN_DIR=$5
PAIR_CHECKPOINT_DIR=$6
MAX_STEPS=$7
SAVE_STEPS=${8:-}

mkdir -p "$PAIR_RUN_DIR" "$PAIR_CHECKPOINT_DIR"
printf '%s\n' "$$" > "$PAIR_RUN_DIR/supervisor.pid"
printf '%s\n' "$(date --iso-8601=seconds)" > "$PAIR_RUN_DIR/started-at.txt"

run_branch() {
    local branch=$1
    local -a command=(
        python -m akgr.reproduction.sc_idc_grpo
        --branch "$branch"
        --experiment-config "$CONFIG"
        --phase2-preflight "$PREFLIGHT"
        --alpha-freeze "$ALPHA_FREEZE"
        --parent-checkpoint "$PARENT"
        --run-dir "$PAIR_RUN_DIR/$branch"
        --checkpoint-dir "$PAIR_CHECKPOINT_DIR/$branch"
        --max-steps "$MAX_STEPS"
    )
    if [[ -n "$SAVE_STEPS" ]]; then
        command+=(--save-steps "$SAVE_STEPS")
    fi
    "${command[@]}" 2>&1 | tee "$PAIR_RUN_DIR/$branch.console.log"
}

printf '%s\n' "baseline" > "$PAIR_RUN_DIR/active-branch.txt"
run_branch baseline
printf '%s\n' "sc_idc" > "$PAIR_RUN_DIR/active-branch.txt"
run_branch sc_idc
printf '%s\n' "completed" > "$PAIR_RUN_DIR/active-branch.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$PAIR_RUN_DIR/completed-at.txt"

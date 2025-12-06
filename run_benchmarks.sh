#!/usr/bin/env bash
set -euo pipefail

###############################################
# User-configurable section
###############################################

DATA="data/chunks_sparse_csv"   # <-- change if needed
EPOCHS=5
FUTURE_STEPS=3

# Whether to pass --load-checkpoint to Python scripts
LOAD_CHECKPOINT=false    # <-- set true if you want to resume

OUTFILE="benchmark_results.txt"

CKPT_FULL="checkpoint_full.pth"
CKPT_MEM="checkpoint_mem_aware.pth"
CHPT_MEM_GPU="checkpoint_mem_aware_gpu.pth"

###############################################
# Helper
###############################################

run_experiment() {
    local script_name="$1"   # train.py or train_memory_aware.py
    local label="$2"         # "full" or "memory_aware"
    local ckpt="$3"

    echo "========================================"
    echo "Running ${label} (${script_name})..."
    echo "========================================"

    local tmp_output="tmp_${label}_output.log"

    local start end elapsed
    start=$(date +%s)

    # Build Python args
    local args=(
        "${script_name}"
        --data "${DATA}"
        --epochs "${EPOCHS}"
        --future-steps "${FUTURE_STEPS}"
        --checkpoint "${ckpt}"
    )
    if [[ "${LOAD_CHECKPOINT}" == "true" ]]; then
        args+=(--load-checkpoint)
    fi

    # Capture stdout + stderr
    python3 "${args[@]}" 2>&1 | tee "${tmp_output}"

    end=$(date +%s)
    elapsed=$((end - start))

    # Look for the last line that has train=
    local last_line
    last_line=$(grep "train=" "${tmp_output}" | tail -n 1 || true)

    local train_loss="NA"
    local val_loss="NA"

    if [[ -z "${last_line}" ]]; then
        echo "WARNING: No 'train=' lines found in ${label} run output."
        echo "Check ${tmp_output} for errors or different log format."
    else
        # Grab the token after train= up to the next space
        train_loss=$(echo "${last_line}" | sed -nE 's/.*train=([^ ]*).*/\1/p')
        val_loss=$(echo "${last_line}"   | sed -nE 's/.*val=([^ ]*).*/\1/p')

        [[ -z "${train_loss}" ]] && train_loss="NA"
        [[ -z "${val_loss}"  ]] && val_loss="NA"
    fi

    {
        echo "Run: ${label}"
        echo "  Script:      ${script_name}"
        echo "  Data:        ${DATA}"
        echo "  Epochs:      ${EPOCHS}"
        echo "  FutureSteps: ${FUTURE_STEPS}"
        echo "  LoadCkpt:    ${LOAD_CHECKPOINT}"
        echo "  Time (s):    ${elapsed}"
        echo "  Train loss:  ${train_loss}"
        echo "  Val loss:    ${val_loss}"
        echo "----------------------------------------"
    } >> "${OUTFILE}"

    echo "Finished ${label}: time=${elapsed}s train=${train_loss} val=${val_loss}"
    echo
}

###############################################
# Main
###############################################

if [[ -f "${OUTFILE}" ]]; then
    echo "Removing old ${OUTFILE}"
    rm -f "${OUTFILE}"
fi

# run_experiment "train.py" "full_memory" "${CKPT_FULL}"
# run_experiment "train_memory_aware_one_window.py" "memory_aware" "${CKPT_MEM}"
run_experiment "train_memory_aware_gpu.py" "memory_aware_gpu" "${CHPT_MEM_GPU}"

echo "All runs complete. Summary written to ${OUTFILE}"

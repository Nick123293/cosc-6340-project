#!/usr/bin/env bash
set -euo pipefail

###############################################
# User-configurable section
###############################################

DATA="data/small_chunks_sparse_csv"
EPOCHS=5
FUTURE_STEPS=3

LOAD_CHECKPOINT=false
OUTFILE="benchmark_results.txt"

CKPT_FULL="checkpoint_full.pth"
CKPT_MEM="checkpoint_mem_aware.pth"
CHPT_MEM_GPU="checkpoint_mem_aware_gpu.pth"
CKPT_MEM_FULL_FILE="checkpoint_mem_aware_full_file.pth"

###############################################
# Run experiment N times and compute averages
###############################################

run_experiment_3x() {
    local script_name="$1"
    local label="$2"
    local ckpt="$3"

    echo "========================================"
    echo "Running ${label} (${script_name}) — 3 trials"
    echo "========================================"

    local train_losses=()
    local val_losses=()
    local runtimes=()

    # ------------------------------------------
    # Perform 3 runs
    # ------------------------------------------
    for trial in 1 2 3; do
        echo "--- ${label} Trial ${trial} ---"

        local tmp_log="tmp_${label}_trial_${trial}.log"

        local start end elapsed
        start=$(date +%s.%N)   # high precision timestamp

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

        python3 "${args[@]}" 2>&1 | tee "${tmp_log}"

        end=$(date +%s.%N)
        
        # Compute elapsed time with decimals
        elapsed=$(echo "scale=4; ${end} - ${start}" | bc -l)
        elapsed_fmt=$(printf "%.2f" "${elapsed}")

        # Extract final train/val losses
        local last_line
        last_line=$(grep "train=" "${tmp_log}" | tail -n 1 || true)

        local train_loss="NA"
        local val_loss="NA"

        if [[ -n "${last_line}" ]]; then
            train_loss=$(echo "${last_line}" | sed -nE 's/.*train=([^ ]*).*/\1/p')
            val_loss=$(echo "${last_line}"   | sed -nE 's/.*val=([^ ]*).*/\1/p')
        fi

        train_losses+=("${train_loss}")
        val_losses+=("${val_loss}")
        runtimes+=("${elapsed}")

        echo "Trial ${trial} result: time=${elapsed_fmt}s train=${train_loss} val=${val_loss}"
        echo
    done

    ###############################################
    # Compute averages
    ###############################################

    avg_float_list() {
        local arr=("$@")
        local sum=0
        local count=0
        for x in "${arr[@]}"; do
            [[ "${x}" == "NA" ]] && continue
            sum=$(echo "${sum} + ${x}" | bc -l)
            count=$((count + 1))
        done
        if [[ $count -eq 0 ]]; then echo "NA"; return; fi
        echo "scale=6; ${sum} / ${count}" | bc -l
    }

    avg_train=$(avg_float_list "${train_losses[@]}")
    avg_val=$(avg_float_list "${val_losses[@]}")

    # Runtime average (decimals ok)
    avg_rt=$(avg_float_list "${runtimes[@]}")
    avg_rt_fmt=$(printf "%.2f" "${avg_rt}")

    {
        echo "========================================"
        echo "Experiment: ${label} (${script_name}) — Averages over 3 runs"
        echo "----------------------------------------"
        echo "  Data:        ${DATA}"
        echo "  Epochs:      ${EPOCHS}"
        echo "  FutureSteps: ${FUTURE_STEPS}"
        echo "  Runtime avg: ${avg_rt_fmt} s"
        echo "  Train avg:   ${avg_train}"
        echo "  Val avg:     ${avg_val}"
        echo "----------------------------------------"
        echo
    } >> "${OUTFILE}"

    echo "Finished ${label}: avg_time=${avg_rt_fmt}s avg_train=${avg_train} avg_val=${avg_val}"
}

###############################################
# Main
###############################################

if [[ -f "${OUTFILE}" ]]; then
    echo "Removing old ${OUTFILE}"
    rm -f "${OUTFILE}"
fi

run_experiment_3x "train.py" "full_memory" "${CKPT_FULL}"
run_experiment_3x "train_memory_aware_one_window.py" "memory_aware" "${CKPT_MEM}"
run_experiment_3x "train_memory_aware_gpu.py" "memory_aware_gpu" "${CHPT_MEM_GPU}"
run_experiment_3x "train_memory_aware.py" "memory_aware_full_file" "${CKPT_MEM_FULL_FILE}"

echo "All experiments complete. Full summary in ${OUTFILE}"

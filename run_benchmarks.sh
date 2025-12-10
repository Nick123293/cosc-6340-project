#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# User configuration
# ---------------------------------------------------------

TRAIN_SCRIPT="train_no_checkpoint_no_DBMS.py"
EPOCHS=5

TRAIN_FILE="data/training_data.csv"
VAL_FILE="data/validation_data.csv"

OUTFILE="benchmark_results.txt"

# ---------------------------------------------------------
# Helper: run one configuration 3 times and aggregate stats
# ---------------------------------------------------------

run_config() {
    local config_id="$1"
    local config_label="$2"
    local extra_args="$3"

    echo "==================================================" | tee -a "$OUTFILE"
    echo "CONFIG $config_id: $config_label" | tee -a "$OUTFILE"
    echo "ARGS: $extra_args" | tee -a "$OUTFILE"
    echo "--------------------------------------------------" | tee -a "$OUTFILE"

    # Accumulators
    local sum_time="0"
    local sum_train="0"
    local sum_val="0"

    for run in 1 2 3; do
        echo "[CONFIG $config_id] Run $run..." | tee -a "$OUTFILE"

        # Call the training script.
        #
        # NOTE:
        #   This assumes your Python script supports:
        #     --train-csv and --val-csv
        #   If your interface is different (e.g. --data with internal 80/20 split),
        #   modify the PY_CMD line accordingly.
        #
        PY_CMD=(
            python "$TRAIN_SCRIPT"
            --train-csv "$TRAIN_FILE"
            --val-csv "$VAL_FILE"
            --epochs "$EPOCHS"
        )

        # Append the configuration-specific arguments
        # shellcheck disable=SC2206
        PY_CMD+=($extra_args)

        # Capture full stdout/stderr
        output="$("${PY_CMD[@]}" 2>&1)"
        echo "$output" >> "$OUTFILE"

        # Parse runtime, train loss, and validation loss from the output.
        # Assumes the Python script prints lines like:
        #   Total runtime: 12.34 seconds
        #   Training Loss: 0.123456
        #   Validation Loss: 0.234567
        #
        runtime=$(echo "$output" | awk -F': ' '/Total runtime/ {print $2}' | awk '{print $1}' | tail -n1)
        train_loss=$(echo "$output" | awk -F': ' '/Training Loss/ {print $2}' | awk '{print $1}' | tail -n1)
        val_loss=$(echo "$output" | awk -F': ' '/Validation Loss/ {print $2}' | awk '{print $1}' | tail -n1)

        # Fallbacks if parsing fails
        runtime="${runtime:-0}"
        train_loss="${train_loss:-0}"
        val_loss="${val_loss:-0}"

        echo "[CONFIG $config_id] Run $run results: time=${runtime}s, train_loss=${train_loss}, val_loss=${val_loss}" \
            | tee -a "$OUTFILE"

        # Accumulate using bc for floating point
        sum_time=$(printf '%s + %s\n' "$sum_time" "$runtime" | bc -l)
        sum_train=$(printf '%s + %s\n' "$sum_train" "$train_loss" | bc -l)
        sum_val=$(printf '%s + %s\n' "$sum_val" "$val_loss" | bc -l)

        # For memory-aware runs (RAM or VRAM constrained), capture effective params.
        if [[ "$extra_args" == *"--max-ram-bytes"* || "$extra_args" == *"--max-vram-bytes"* ]]; then
            # These patterns should match lines your script prints, e.g.:
            #   effective_seq_len_in=...
            #   effective_future_steps=...
            #   max_timesteps_in_memory=...
            #
            # Adjust the patterns if your log format is different.
            eff_seq=$(echo "$output" | awk -F'=' '/effective_seq_len_in/ {print $2}' | tail -n1)
            eff_future=$(echo "$output" | awk -F'=' '/effective_future_steps/ {print $2}' | tail -n1)
            max_ts=$(echo "$output" | awk -F'=' '/max_timesteps_in_memory/ {print $2}' | tail -n1)

            eff_seq="${eff_seq:-N/A}"
            eff_future="${eff_future:-N/A}"
            max_ts="${max_ts:-N/A}"

            echo "[CONFIG $config_id] Run $run memory-aware info:" | tee -a "$OUTFILE"
            echo "    effective_seq_len_in   = $eff_seq" | tee -a "$OUTFILE"
            echo "    effective_future_steps = $eff_future" | tee -a "$OUTFILE"
            echo "    max_timesteps_in_memory= $max_ts" | tee -a "$OUTFILE"
        fi

        echo "" >> "$OUTFILE"
    done

    # Compute averages over the 3 runs
    avg_time=$(printf '%s / 3\n' "$sum_time" | bc -l)
    avg_train=$(printf '%s / 3\n' "$sum_train" | bc -l)
    avg_val=$(printf '%s / 3\n' "$sum_val" | bc -l)

    # Format to 6 decimal places for losses, 2 for runtime
    avg_time_fmt=$(printf '%.2f' "$avg_time")
    avg_train_fmt=$(printf '%.6f' "$avg_train")
    avg_val_fmt=$(printf '%.6f' "$avg_val")

    echo ">>> AVERAGES for CONFIG $config_id over 3 runs:" | tee -a "$OUTFILE"
    echo "    avg_time_s   = $avg_time_fmt" | tee -a "$OUTFILE"
    echo "    avg_train    = $avg_train_fmt" | tee -a "$OUTFILE"
    echo "    avg_val      = $avg_val_fmt" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
}

# ---------------------------------------------------------
# Define the 18 configurations
# (Converted to the hyphenated argparse names)
# ---------------------------------------------------------

# Each config: ID | label | extra_args
CONFIGS=(
  "1|seq9_future3|--seq-len-in 9 --future-steps 3"
  "2|seq100_future10|--seq-len-in 100 --future-steps 10"
  "3|seq100_future100|--seq-len-in 100 --future-steps 100"

  "4|seq9_future3_ram1G|--seq-len-in 9 --future-steps 3 --max-ram-bytes 1073741824"
  "5|seq100_future10_ram1G|--seq-len-in 100 --future-steps 10 --max-ram-bytes 1073741824"
  "6|seq100_future100_ram1G|--seq-len-in 100 --future-steps 100 --max-ram-bytes 1073741824"

  "7|seq9_future3_vram1G|--seq-len-in 9 --future-steps 3 --max-vram-bytes 1073741824"
  "8|seq100_future10_vram1G|--seq-len-in 100 --future-steps 10 --max-vram-bytes 1073741824"
  "9|seq100_future100_vram1G|--seq-len-in 100 --future-steps 100 --max-vram-bytes 1073741824"

  "10|seq9_future3_ram512M|--seq-len-in 9 --future-steps 3 --max-ram-bytes 536870912"
  "11|seq100_future10_ram512M|--seq-len-in 100 --future-steps 10 --max-ram-bytes 536870912"
  "12|seq100_future100_ram512M|--seq-len-in 100 --future-steps 100 --max-ram-bytes 536870912"

  "13|seq9_future3_vram512M|--seq-len-in 9 --future-steps 3 --max-vram-bytes 536870912"
  "14|seq100_future10_vram512M|--seq-len-in 100 --future-steps 10 --max-vram-bytes 536870912"
  "15|seq100_future100_vram512M|--seq-len-in 100 --future-steps 100 --max-vram-bytes 536870912"
)

# ---------------------------------------------------------
# Main loop over configurations
# ---------------------------------------------------------

# Truncate output file at start
: > "$OUTFILE"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r cfg_id cfg_label cfg_args <<< "$cfg"
    run_config "$cfg_id" "$cfg_label" "$cfg_args"
done

echo "All configurations completed. Results saved to: $OUTFILE"

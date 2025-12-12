#!/usr/bin/env bash
set -euo pipefail

###############################################
# User configuration
###############################################

TRAIN_SCRIPT="train_no_checkpoint_no_DBMS.py"
VAL_FILE="data/validation_data.csv"
EPOCHS=1

# Directory to hold:
#   - per-run JSON logs
#   - benchmark_results.txt summary
LOG_DIR="results"
# mkdir -p "$LOG_DIR"

OUTFILE="${LOG_DIR}/benchmark_results_no_comp.txt"

###############################################
# RAM / VRAM configs
#
# Each entry: base_id|base_label|extra_args
###############################################

CONFIGS_BASE=(
  "1|seq9_future3|--seq-len-in 9 --future-steps 3"
  "2|seq9_future3_ram128M|--seq-len-in 9 --future-steps 3 --max-ram-bytes 134217728"
  "3|seq9_future3_vram128M|--seq-len-in 9 --future-steps 3 --max-vram-bytes 134217728"
#   "4|seq9_future3_ram32M|--seq-len-in 9 --future-steps 3 --max-ram-bytes 33554432"
#   "5|seq9_future3_vram32M|--seq-len-in 9 --future-steps 3 --max-vram-bytes 33554432"
)

###############################################
# Time ranges + pre-parsed CSVs
#
# Each entry: tag|parsed_csv|time_start|time_end
###############################################

RANGES=(
  "feb_march|data/training_data_parsed_feb_march.csv|2024-02-01 00:00:00|2024-03-01 00:00:00"
  "feb_april|data/training_data_parsed_feb_april.csv|2024-02-01 00:00:00|2024-04-01 00:00:00"
  "july_august|data/training_data_parsed_july_august.csv|2024-07-01 00:00:00|2024-08-01 00:00:00"
  "july_september|data/training_data_parsed_july_september.csv|2024-07-01 00:00:00|2024-09-01 00:00:00"
)

###############################################
# Helper: run one config 3 times and average
#
# Arguments:
#   $1 = config_id
#   $2 = config_label
#   $3 = train_csv
#   $4 = mode ("parsed" or "full")
#   $5 = base_args (seq_len, future_steps, RAM/VRAM flags)
#   $6 = range_tag
#   $7 = time_start
#   $8 = time_end
###############################################

run_config() {
    local config_id="$1"
    local config_label="$2"
    local train_csv="$3"
    local mode="$4"
    local base_args="$5"
    local range_tag="$6"
    local time_start="$7"
    local time_end="$8"

    echo "==================================================" | tee -a "$OUTFILE"
    echo "CONFIG $config_id: $config_label" | tee -a "$OUTFILE"
    echo "  range_tag   = $range_tag" | tee -a "$OUTFILE"
    echo "  mode        = $mode" | tee -a "$OUTFILE"
    echo "  train_csv   = $train_csv" | tee -a "$OUTFILE"
    echo "  base_args   = $base_args" | tee -a "$OUTFILE"
    echo "  time_start  = $time_start" | tee -a "$OUTFILE"
    echo "  time_end    = $time_end" | tee -a "$OUTFILE"
    echo "--------------------------------------------------" | tee -a "$OUTFILE"

    local sum_time="0"
    local sum_train="0"
    local sum_val="0"

    for run in 1 2; do
        echo "[CONFIG $config_id] Run $run..." | tee -a "$OUTFILE"

        # Per-run JSON log file
        local log_file="${LOG_DIR}/${config_label}_run${run}_json.json"
        local computation_dump="${LOG_DIR}/${config_label}_run${run}_json_COMP_DUMP.json"
        PY_CMD=(
            python "$TRAIN_SCRIPT"
            --train-csv "$train_csv"
            --val-csv "$VAL_FILE"
            --epochs "$EPOCHS"
            --log-file "$log_file"
            --save-computations "$computation_dump"
        )

        # Append base args (seq_len, future_steps, RAM/VRAM)
        # shellcheck disable=SC2206
        PY_CMD+=($base_args)

        # For "full" mode, add matching time window
        if [[ "$mode" == "full" ]]; then
            PY_CMD+=( --time-start "$time_start" --time-end "$time_end" )
        fi

        # Capture stdout/stderr and append to benchmark log
        output="$("${PY_CMD[@]}" 2>&1)"
        echo "$output" >> "$OUTFILE"

        # Extract metrics from JSON log
        local metrics
        metrics=$(python - <<EOF
import json, pathlib
p = pathlib.Path("$log_file")
if not p.exists():
    print("0 0 0")
else:
    with p.open() as f:
        d = json.load(f)
    rt = d.get("runtime_seconds", 0.0)
    tr = d.get("train_loss", 0.0)
    vl = d.get("val_loss", 0.0)
    print(rt, tr, vl)
EOF
)

        local runtime train_loss val_loss
        runtime=$(echo "$metrics"     | awk '{print $1}')
        train_loss=$(echo "$metrics" | awk '{print $2}')
        val_loss=$(echo "$metrics"   | awk '{print $3}')

        runtime="${runtime:-0}"
        train_loss="${train_loss:-0}"
        val_loss="${val_loss:-0}"

        sum_time=$(printf '%s + %s\n' "$sum_time" "$runtime" | bc -l)
        sum_train=$(printf '%s + %s\n' "$sum_train" "$train_loss" | bc -l)
        sum_val=$(printf '%s + %s\n' "$sum_val" "$val_loss" | bc -l)

        echo "  Run $run metrics:" | tee -a "$OUTFILE"
        echo "    runtime_s = $runtime" | tee -a "$OUTFILE"
        echo "    train     = $train_loss" | tee -a "$OUTFILE"
        echo "    val       = $val_loss" | tee -a "$OUTFILE"
        echo "" | tee -a "$OUTFILE"
    done

    # Averages
    local avg_time avg_train avg_val
    avg_time=$(printf '%s / 2\n' "$sum_time" | bc -l)
    avg_train=$(printf '%s / 2\n' "$sum_train" | bc -l)
    avg_val=$(printf '%s / 2\n' "$sum_val" | bc -l)

    local avg_time_fmt avg_train_fmt avg_val_fmt
    avg_time_fmt=$(printf '%.2f' "$avg_time")
    avg_train_fmt=$(printf '%.6f' "$avg_train")
    avg_val_fmt=$(printf '%.6f' "$avg_val")

    echo ">>> AVERAGES for CONFIG $config_id over 2 runs:" | tee -a "$OUTFILE"
    echo "    config_label = $config_label" | tee -a "$OUTFILE"
    echo "    base_args    = $base_args" | tee -a "$OUTFILE"
    echo "    avg_time_s   = $avg_time_fmt" | tee -a "$OUTFILE"
    echo "    avg_train    = $avg_train_fmt" | tee -a "$OUTFILE"
    echo "    avg_val      = $avg_val_fmt" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
}

###############################################
# Main: loop over RAM configs × ranges × modes
###############################################

: > "$OUTFILE"   # truncate summary at start

cfg_counter=1

for base in "${CONFIGS_BASE[@]}"; do
    IFS='|' read -r base_id base_label base_args <<< "$base"

    for r in "${RANGES[@]}"; do
        IFS='|' read -r range_tag parsed_csv time_start time_end <<< "$r"

        # 1) parsed subset
        parsed_label="${range_tag}_${base_label}_parsed"
        run_config \
          "$cfg_counter" \
          "$parsed_label" \
          "$parsed_csv" \
          "parsed" \
          "$base_args" \
          "$range_tag" \
          "$time_start" \
          "$time_end"
        cfg_counter=$((cfg_counter + 1))

        # 2) full training_data.csv with matching time window
        full_label="${range_tag}_${base_label}_full"
        run_config \
          "$cfg_counter" \
          "$full_label" \
          "data/training_data.csv" \
          "full" \
          "$base_args" \
          "$range_tag" \
          "$time_start" \
          "$time_end"
        cfg_counter=$((cfg_counter + 1))
    done
done

echo "All configurations completed."
echo "Per-run JSON logs and benchmark_results.txt are in: $LOG_DIR"

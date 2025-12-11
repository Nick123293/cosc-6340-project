import argparse
import os
import time
import json  # <--- Replaces psycopg2 for this script
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ConvLSTM import ConvLSTM2D


# =========================================================
# 0. Row-size + memory capacity estimation
# =========================================================

def estimate_row_sizes(csv_path, max_sample_rows=1000):
    """
    Use the first up to `max_sample_rows` data rows (excluding header)
    to estimate per-row size in bytes.
    """
    sizes = []
    with open(csv_path, "rb") as f:
        # Skip header
        _ = f.readline()
        for i, line in enumerate(f):
            if i >= max_sample_rows:
                break
            sizes.append(len(line))

    if not sizes:
        raise ValueError(f"[MEM] No data rows found when sampling CSV: {csv_path}")

    n = len(sizes)
    avg_bytes = float(sum(sizes)) / n
    min_bytes = min(sizes)
    max_bytes = max(sizes)

    print(
        f"[MEM] Row size stats over {n} sampled rows "
        f"(excluding header): avg={avg_bytes:.2f} bytes, "
        f"min={min_bytes} bytes, max={max_bytes} bytes"
    )
    return avg_bytes, min_bytes, max_bytes


def adjust_window_lengths(seq_len_in_arg, future_steps_arg, timestep_capacity):
    if timestep_capacity is None:
        return seq_len_in_arg, future_steps_arg

    # At least 3 timesteps (2 input + 1 prediction), per assumption.
    min_total = 3
    max_total = min(timestep_capacity, seq_len_in_arg + future_steps_arg)
    if max_total < min_total:
        max_total = min_total

    if future_steps_arg > 0:
        target_ratio = float(seq_len_in_arg) / float(future_steps_arg)
    else:
        target_ratio = float(seq_len_in_arg)

    best_pair = None
    best_total = -1
    best_ratio_diff = None

    # Search from largest feasible total timesteps downward.
    for total in range(max_total, min_total - 1, -1):
        local_best_pair = None
        local_best_ratio_diff = None

        max_future = min(future_steps_arg, total - 1)
        for new_future in range(1, max_future + 1):
            new_seq = total - new_future
            if new_seq < 1 or new_seq > seq_len_in_arg:
                continue

            ratio = float(new_seq) / float(new_future)
            diff = abs(ratio - target_ratio)
            if (local_best_ratio_diff is None) or (diff < local_best_ratio_diff):
                local_best_ratio_diff = diff
                local_best_pair = (new_seq, new_future)

        if local_best_pair is not None:
            best_pair = local_best_pair
            best_total = total
            best_ratio_diff = local_best_ratio_diff
            break

    if best_pair is None:
        # Fallback: simple clamping
        new_future = min(future_steps_arg, max_total - 1)
        if new_future < 1:
            new_future = 1
        new_seq = max_total - new_future
        if new_seq < 1:
            new_seq = 1
        best_pair = (new_seq, new_future)

    new_seq_len_in, new_future_steps = best_pair
    return new_seq_len_in, new_future_steps


def analyze_csv_and_memory(
    csv_path,
    seq_len_in_arg,
    future_steps_arg,
    max_ram_bytes=None,
    max_vram_bytes=None,
    time_filter_indices=None  # <--- NEW: Filter Arg
):
    """
    Scans CSV for full time range, then optionally filters it.
    Re-calculates memory usage based on the *filtered* duration.
    """
    avg_bytes, min_bytes, max_bytes = estimate_row_sizes(csv_path)

    # First pass: find t_min, t_max, global x/y sets, total_rows.
    total_rows_scanned = 0
    t_min = None
    t_max = None
    x_values = set()
    y_values = set()

    # Scan in moderate chunks
    scan_chunksize = 100000
    print(f"[MEM] Scanning CSV for global structure with chunksize={scan_chunksize} rows...")

    for chunk in pd.read_csv(csv_path, chunksize=scan_chunksize, usecols=[0, 1, 2]):
        if chunk.empty:
            continue

        total_rows_scanned += len(chunk)

        t_col = pd.to_datetime(chunk.iloc[:, 0])
        c_min = t_col.min()
        c_max = t_col.max()
        t_min = c_min if t_min is None else min(t_min, c_min)
        t_max = c_max if t_max is None else max(t_max, c_max)

        x_values.update(chunk.iloc[:, 1].astype(float).unique())
        y_values.update(chunk.iloc[:, 2].astype(float).unique())

    if total_rows_scanned == 0:
        raise ValueError(f"[DATA] CSV appears empty: {csv_path}")

    # Global hourly timeline (dense time axis)
    t_full = pd.date_range(start=t_min, end=t_max, freq="h")
    
    # === NEW: Apply Time Filter ===
    if time_filter_indices is not None:
        start_idx, end_idx = time_filter_indices
        start_idx = max(0, start_idx)
        end_idx = min(len(t_full), end_idx)
        
        t_full = t_full[start_idx:end_idx]
        if len(t_full) == 0:
            raise ValueError(f"Time filter {time_filter_indices} resulted in 0 timesteps.")
            
        print(f"[FILTER] Restricting training to timesteps {start_idx}-{end_idx} ({len(t_full)} hours).")
        
        # Adjust total_rows estimate based on the fraction of time kept
        fraction_kept = len(t_full) / ((t_max - t_min).total_seconds() / 3600 + 1)
        total_rows_estimated = int(total_rows_scanned * fraction_kept)
    else:
        total_rows_estimated = total_rows_scanned

    T_total = len(t_full)

    x_unique = np.array(sorted(x_values), dtype=float)
    y_unique = np.array(sorted(y_values), dtype=float)
    H = len(y_unique)
    W = len(x_unique)
    C = 4
    B = 1
    bytes_per_timestep_dense = float(B * C * H * W * 4)  # float32

    # Avoid division by zero if single timestep
    if T_total > 0:
        rows_per_timestep = float(total_rows_estimated) / float(T_total)
    else:
        rows_per_timestep = 0
        
    bytes_per_timestep_csv = avg_bytes * rows_per_timestep

    print(f"[MEM] Effective dense hourly timesteps: {T_total}")
    print(f"[MEM] Approx dense bytes per timestep: {bytes_per_timestep_dense:.2f}")

    timestep_limits = []

    # RAM limit
    if max_ram_bytes is not None and max_ram_bytes > 0:
        if bytes_per_timestep_csv > 0:
            t_ram_sparse = int(max_ram_bytes // bytes_per_timestep_csv)
        else:
            t_ram_sparse = None

        if bytes_per_timestep_dense > 0:
            t_ram_dense = int(max_ram_bytes // bytes_per_timestep_dense)
        else:
            t_ram_dense = None

        ram_candidates = []
        if t_ram_sparse is not None and t_ram_sparse > 0:
            ram_candidates.append(t_ram_sparse)
        if t_ram_dense is not None and t_ram_dense > 0:
            ram_candidates.append(t_ram_dense)

        if ram_candidates:
            t_ram = max(3, min(ram_candidates))
            timestep_limits.append(t_ram)

    # VRAM limit
    if max_vram_bytes is not None and max_vram_bytes > 0:
        if bytes_per_timestep_dense > 0:
            t_vram = int(max_vram_bytes // bytes_per_timestep_dense)
            t_vram = max(3, t_vram)
            timestep_limits.append(t_vram)

    if timestep_limits:
        timestep_capacity = min(timestep_limits)
        timestep_capacity = min(timestep_capacity, T_total)
    else:
        timestep_capacity = None

    # Adjust window lengths
    if timestep_capacity is not None:
        new_seq_len_in, new_future_steps = adjust_window_lengths(
            seq_len_in_arg,
            future_steps_arg,
            timestep_capacity
        )
    else:
        new_seq_len_in, new_future_steps = seq_len_in_arg, future_steps_arg

    # Rows per chunk for sparse reading
    if max_ram_bytes is not None and max_ram_bytes > 0 and avg_bytes > 0:
        approx_rows = int(max_ram_bytes // avg_bytes)
        rows_per_chunk_csv = max(1, min(approx_rows, 100000))
    else:
        rows_per_chunk_csv = 100000

    return (
        t_full,
        x_unique,
        y_unique,
        timestep_capacity,
        rows_per_chunk_csv,
        new_seq_len_in,
        new_future_steps,
    )


# =========================================================
# 1. Sparse DF → Dense 2D tensor on a fixed global grid
# =========================================================

def load_sparse_df_to_dense_2d_fixed_grid(
    df,
    t_start,
    t_end,
    x_unique,
    y_unique,
    device="cpu",
):
    """
    Convert a sparse DataFrame for a GIVEN local time range [t_start, t_end] 
    to a dense tensor.
    """
    if df.shape[1] < 5:
        raise ValueError("Chunk DataFrame must have at least 5 columns.")

    t_raw = pd.to_datetime(df.iloc[:, 0])
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)

    feats = df.iloc[:, -4:].astype(float).fillna(0.0)
    C = feats.shape[1]

    # Local hourly timeline for the block.
    t_block = pd.date_range(start=t_start, end=t_end, freq="h")
    t_block_vals = t_block.values
    T_block = len(t_block_vals)

    time_to_index = {ts: i for i, ts in enumerate(t_block_vals)}
    x_index = {val: i for i, val in enumerate(x_unique)}
    y_index = {val: i for i, val in enumerate(y_unique)}

    B = 1
    H = len(y_unique)
    W = len(x_unique)
    dense = torch.zeros((B, T_block, C, H, W), device=device, dtype=torch.float32)

    mask = (t_raw >= t_start) & (t_raw <= t_end)
    if not mask.any():
        return dense 

    df = df.loc[mask]
    t_raw = t_raw[mask]
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]
    feats = df.iloc[:, -4:].astype(float).fillna(0.0).values 

    t_idx = np.array([time_to_index[ts] for ts in t_raw.values], dtype=np.int64)
    x_idx = np.array([x_index[xv] for xv in x_raw.values], dtype=np.int64)
    y_idx = np.array([y_index[yv] for yv in y_raw.values], dtype=np.int64)

    t_idx_t = torch.from_numpy(t_idx).long().to(device)
    x_idx_t = torch.from_numpy(x_idx).long().to(device)
    y_idx_t = torch.from_numpy(y_idx).long().to(device)
    feats_t = torch.from_numpy(feats.astype(np.float32)).to(device)

    dense[0, t_idx_t, :, y_idx_t, x_idx_t] = feats_t
    return dense


def build_dense_block_from_csv(
    csv_path,
    t_start,
    t_end,
    x_unique,
    y_unique,
    device,
    rows_per_chunk_csv,
):
    """
    Build a dense tensor for [t_start, t_end] by reading the CSV in chunks.
    This respects the 'On-the-fly' philosophy by not loading the whole file.
    """
    collected = []
    chunksize = rows_per_chunk_csv or 100000

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue
        t_col = pd.to_datetime(chunk.iloc[:, 0])
        mask = (t_col >= t_start) & (t_col <= t_end)
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        collected.append(sub)

    if not collected:
        dense_empty = load_sparse_df_to_dense_2d_fixed_grid(
            pd.DataFrame(columns=["time", "x", "y", "c1", "c2", "c3", "c4"]),
            t_start,
            t_end,
            x_unique,
            y_unique,
            device=device,
        )
        return dense_empty

    df_block = pd.concat(collected, ignore_index=True)
    dense = load_sparse_df_to_dense_2d_fixed_grid(
        df_block,
        t_start,
        t_end,
        x_unique,
        y_unique,
        device=device,
    )
    return dense


# =========================================================
# 3. Core sliding-window processing over full timeline
# =========================================================

def process_csv_with_blocks(
    csv_path,
    convlstm,
    decoder,
    criterion,
    seq_len_in,
    future_steps,
    t_full,
    x_unique,
    y_unique,
    timestep_capacity,
    rows_per_chunk_csv,
    is_train,
    optimizer=None,
    epoch=0,
):
    device = next(convlstm.parameters()).device
    T_total = len(t_full)
    W_len = seq_len_in + future_steps

    if T_total < W_len:
        print(f"[WARN] Timeline has fewer timesteps ({T_total}) than window length ({W_len}).")
        return None

    if timestep_capacity is None or timestep_capacity >= T_total:
        T_block = T_total
    else:
        T_block = timestep_capacity

    if T_block < W_len:
        T_block = W_len

    block_stride = T_block - (W_len - 1) if T_block > W_len else 1

    total_loss = 0.0
    n_windows = 0
    global_last_processed_start = -1 

    # Iterate over blocks by global time index from the SLICED timeline
    for block_start_idx in range(0, T_total - W_len + 1, block_stride):
        block_end_idx = min(block_start_idx + T_block, T_total)
        t_start = t_full[block_start_idx]
        t_end = t_full[block_end_idx - 1]

        dense_block = build_dense_block_from_csv(
            csv_path,
            t_start,
            t_end,
            x_unique,
            y_unique,
            device,
            rows_per_chunk_csv,
        )
        B, T_block_actual, C, H, W = dense_block.shape

        if T_block_actual < W_len:
            del dense_block
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        local_max_start = T_block_actual - W_len

        for local_start in range(local_max_start + 1):
            global_start = block_start_idx + local_start
            if global_start <= global_last_processed_start:
                continue

            window = dense_block[:, local_start:local_start + W_len]   
            x_in = window[:, :seq_len_in]                              
            y_true = window[:, seq_len_in:]                            

            if is_train:
                optimizer.zero_grad()
                outputs, (h, c) = convlstm(x_in)
                preds = []
                h_t, c_t = h, c
                for _ in range(future_steps):
                    y_t = decoder(h_t) 
                    preds.append(y_t)
                    h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

                preds = torch.stack(preds, dim=1)  
                loss = criterion(preds, y_true)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs, (h, c) = convlstm(x_in)
                    preds = []
                    h_t, c_t = h, c
                    for _ in range(future_steps):
                        y_t = decoder(h_t)
                        preds.append(y_t)
                        h_t, c_t = convlstm.cell(y_t, (h_t, c_t))
                    preds = torch.stack(preds, dim=1)
                    loss = criterion(preds, y_true)

            total_loss += loss.item()
            n_windows += 1

        global_last_processed_start = block_start_idx + local_max_start

        del dense_block
        if device == "cuda":
            torch.cuda.empty_cache()

    if n_windows == 0:
        return None

    return total_loss / n_windows


# =========================================================
# 5. TRAIN LOOP
# =========================================================

def train(
    train_csv_path,
    val_csv_path,
    convlstm,
    decoder,
    num_epochs,
    seq_len_in,
    future_steps,
    t_full,
    x_unique,
    y_unique,
    timestep_capacity,
    rows_per_chunk_csv,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(convlstm.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    for epoch in range(num_epochs):
        convlstm.train()
        decoder.train()

        train_loss = process_csv_with_blocks(
            train_csv_path,
            convlstm,
            decoder,
            criterion,
            seq_len_in,
            future_steps,
            t_full,
            x_unique,
            y_unique,
            timestep_capacity,
            rows_per_chunk_csv,
            is_train=True,
            optimizer=optimizer,
            epoch=epoch,
        )
        if train_loss is None:
            train_loss = float("nan")

        convlstm.eval()
        decoder.eval()

        with torch.no_grad():
            val_loss = process_csv_with_blocks(
                val_csv_path,
                convlstm,
                decoder,
                criterion,
                seq_len_in,
                future_steps,
                t_full,
                x_unique,
                y_unique,
                timestep_capacity,
                rows_per_chunk_csv,
                is_train=False,
                optimizer=None,
                epoch=0,
            )
        if val_loss is None:
            val_loss = float("nan")

        print(
            f"[EPOCH {epoch+1}/{num_epochs}] "
            f"train={train_loss:.6f}  val={val_loss:.6f}"
        )

    return train_loss, val_loss


# =========================================================
# 6. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument("--seq-len-in", type=int, default=9)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pth")
    parser.add_argument("--max-ram-bytes", type=int, default=None)
    parser.add_argument("--max-vram-bytes", type=int, default=None)
    
    # NEW ARGUMENTS
    parser.add_argument("--time-start", type=int, default=None, help="Start timestep index")
    parser.add_argument("--time-end", type=int, default=None, help="End timestep index")
    parser.add_argument("--log-file", type=str, default="training_log.json", help="Path to save experiment metrics")

    start_time = time.time()  # <--- Start Timer
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Filter Tuple
    if args.time_start is not None and args.time_end is not None:
        time_filter_indices = (args.time_start, args.time_end)
    else:
        time_filter_indices = None

    # Analyze CSV with Filter
    (
        t_full,
        x_unique,
        y_unique,
        timestep_capacity,
        rows_per_chunk_csv,
        effective_seq_len_in,
        effective_future_steps,
    ) = analyze_csv_and_memory(
        args.train_csv,
        args.seq_len_in,
        args.future_steps,
        max_ram_bytes=args.max_ram_bytes,
        max_vram_bytes=args.max_vram_bytes,
        time_filter_indices=time_filter_indices # <--- Pass Filter
    )

    convlstm = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    decoder = nn.Conv2d(4, 4, kernel_size=1).to(device)

    train_loss, val_loss = train(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        convlstm=convlstm,
        decoder=decoder,
        num_epochs=args.epochs,
        seq_len_in=effective_seq_len_in,
        future_steps=effective_future_steps,
        t_full=t_full,
        x_unique=x_unique,
        y_unique=y_unique,
        timestep_capacity=timestep_capacity,
        rows_per_chunk_csv=rows_per_chunk_csv,
    )

    if args.checkpoint:
        torch.save(
            {
                "convlstm_state_dict": convlstm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            },
            args.checkpoint,
        )
        print("Model saved at: " + args.checkpoint)

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

    # === NEW: LOCAL JSON LOGGING (Replaces DB Logging) ===
    results = {
        "train_csv": args.train_csv,
        "epochs": args.epochs,
        "runtime_seconds": total_time,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "max_ram_bytes": args.max_ram_bytes,
        "max_vram_bytes": args.max_vram_bytes,
        "time_start": args.time_start,
        "time_end": args.time_end
    }
    
    with open(args.log_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Experiment metrics saved to {args.log_file}")
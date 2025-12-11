import argparse
import os
import time

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

    Prints and returns:
        avg_bytes, min_bytes, max_bytes
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
    """
    Given original seq_len_in and future_steps, and a maximum total number of
    timesteps we can hold at once (timestep_capacity), choose new
    (seq_len_in, future_steps) such that:

      - seq_len_in + future_steps <= timestep_capacity
      - seq_len_in <= seq_len_in_arg, future_steps <= future_steps_arg
      - seq_len_in >= 1, future_steps >= 1
      - The ratio seq_len_in : future_steps is as close as possible to the
        original ratio, preferring higher total timesteps and preferring
        seq_len_in to be higher when an exact ratio cannot be achieved.
    """
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
            # Because we iterate total from largest down, first valid is max.
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
):
    """
    - Estimates sparse row size (avg/min/max).
    - Streams through the full CSV in sparse chunks to estimate:
        * total_rows
        * global time range (t_min, t_max)
        * global unique x, y sets
    - From t_min..t_max (hourly), derives:
        * total_timesteps (T_total)
        * approximate rows_per_timestep
        * CSV bytes per timestep (sparse)
        * dense bytes per timestep (1, C=4, H, W, float32)
    - Uses these to compute a timestep_capacity from:
        * RAM limit based on BOTH sparse and dense
        * VRAM limit based on dense only
    - Adjusts seq_len_in and future_steps accordingly.

    Returns:
        t_full (DatetimeIndex),
        x_unique (np.array),
        y_unique (np.array),
        timestep_capacity (int or None),
        rows_per_chunk_csv (int or None),
        new_seq_len_in,
        new_future_steps
    """
    avg_bytes, min_bytes, max_bytes = estimate_row_sizes(csv_path)

    # First pass: find t_min, t_max, global x/y sets, total_rows.
    total_rows = 0
    t_min = None
    t_max = None
    x_values = set()
    y_values = set()

    # Scan in moderate chunks; RAM for this pass is bounded by chunk size.
    scan_chunksize = 100000
    print(f"[MEM] Scanning CSV for global structure with chunksize={scan_chunksize} rows...")

    for chunk in pd.read_csv(csv_path, chunksize=scan_chunksize, usecols=[0, 1, 2]):
        if chunk.empty:
            continue

        total_rows += len(chunk)

        t_col = pd.to_datetime(chunk.iloc[:, 0])
        c_min = t_col.min()
        c_max = t_col.max()
        t_min = c_min if t_min is None else min(t_min, c_min)
        t_max = c_max if t_max is None else max(t_max, c_max)

        x_values.update(chunk.iloc[:, 1].astype(float).unique())
        y_values.update(chunk.iloc[:, 2].astype(float).unique())

    if total_rows == 0:
        raise ValueError(f"[DATA] CSV appears empty: {csv_path}")

    # Global hourly timeline (dense time axis)
    t_full = pd.date_range(start=t_min, end=t_max, freq="h")
    T_total = len(t_full)

    x_unique = np.array(sorted(x_values), dtype=float)
    y_unique = np.array(sorted(y_values), dtype=float)
    H = len(y_unique)
    W = len(x_unique)
    C = 4  # assuming 4 feature channels (last 4 CSV columns)
    B = 1
    bytes_per_timestep_dense = float(B * C * H * W * 4)  # float32

    rows_per_timestep = float(total_rows) / float(T_total)
    bytes_per_timestep_csv = avg_bytes * rows_per_timestep

    print(f"[MEM] Total sparse rows in CSV: {total_rows}")
    print(f"[MEM] Dense hourly timesteps (t_min→t_max): {T_total}")
    print(f"[MEM] Unique X coords: {W}, unique Y coords: {H}")
    print(f"[MEM] Approx rows per timestep: {rows_per_timestep:.2f}")
    print(f"[MEM] Approx CSV bytes per timestep (sparse): {bytes_per_timestep_csv:.2f}")
    print(f"[MEM] Approx dense bytes per timestep: {bytes_per_timestep_dense:.2f} "
          f"(B=1, C={C}, H={H}, W={W})")

    timestep_limits = []

    # RAM limit: consider BOTH sparse CSV and dense tensor sizes.
    if max_ram_bytes is not None and max_ram_bytes > 0:
        if bytes_per_timestep_csv > 0:
            t_ram_sparse = int(max_ram_bytes // bytes_per_timestep_csv)
        else:
            t_ram_sparse = None

        if bytes_per_timestep_dense > 0:
            t_ram_dense = int(max_ram_bytes // bytes_per_timestep_dense)
        else:
            t_ram_dense = None

        # Combine: we need to satisfy both sparse + dense, so we take min.
        ram_candidates = []
        if t_ram_sparse is not None and t_ram_sparse > 0:
            ram_candidates.append(t_ram_sparse)
        if t_ram_dense is not None and t_ram_dense > 0:
            ram_candidates.append(t_ram_dense)

        if ram_candidates:
            t_ram = max(3, min(ram_candidates))
            timestep_limits.append(t_ram)
            print(f"[MEM] Based on max-ram-bytes={max_ram_bytes}, "
                  f"timestep capacity (sparse+dense) ≈ {t_ram}")
        else:
            print("[MEM] Unable to derive RAM-based timestep capacity (degenerate sizes).")

    # VRAM limit: dense tensor bytes only.
    if max_vram_bytes is not None and max_vram_bytes > 0:
        if bytes_per_timestep_dense > 0:
            t_vram = int(max_vram_bytes // bytes_per_timestep_dense)
            t_vram = max(3, t_vram)
            timestep_limits.append(t_vram)
            print(f"[MEM] Based on max-vram-bytes={max_vram_bytes}, "
                  f"timestep capacity (dense on GPU) ≈ {t_vram}")
        else:
            print("[MEM] Unable to derive VRAM-based timestep capacity (degenerate dense size).")

    if timestep_limits:
        timestep_capacity = min(timestep_limits)
        # Cap by total timesteps available
        timestep_capacity = min(timestep_capacity, T_total)
        print(
            f"[MEM] Effective timestep capacity (min over RAM/VRAM and T_total) = "
            f"{timestep_capacity}"
        )
    else:
        timestep_capacity = None
        print("[MEM] No RAM/VRAM limits provided; effective timestep capacity is unbounded.")

    # Adjust (seq_len_in, future_steps) based on timestep_capacity
    if timestep_capacity is not None:
        new_seq_len_in, new_future_steps = adjust_window_lengths(
            seq_len_in_arg,
            future_steps_arg,
            timestep_capacity
        )
        print(
            f"[MEM] Adjusting window lengths due to memory limits: "
            f"seq_len_in {seq_len_in_arg} → {new_seq_len_in}, "
            f"future_steps {future_steps_arg} → {new_future_steps}"
        )
    else:
        new_seq_len_in, new_future_steps = seq_len_in_arg, future_steps_arg
        print(
            "[MEM] Using original seq_len_in and future_steps "
            "because timestep capacity is unbounded."
        )

    # Explicitly print how many timesteps we can keep in memory
    if timestep_capacity is not None:
        print(
            f"[MEM] Final effective timestep capacity (timesteps in memory) = "
            f"{timestep_capacity}"
        )
    else:
        print(
            "[MEM] Final effective timestep capacity: no explicit limit; "
            "dense blocks may span full timeline."
        )

    # Choose rows_per_chunk for sparse CSV reading.
    # Approximate: keep chunk moderate. If no RAM limit, just use a default.
    if max_ram_bytes is not None and max_ram_bytes > 0 and avg_bytes > 0:
        approx_rows = int(max_ram_bytes // avg_bytes)
        # Keep this from being absurdly large
        rows_per_chunk_csv = max(1, min(approx_rows, 100000))
    else:
        rows_per_chunk_csv = 100000

    print(f"[MEM] Using rows_per_chunk_csv={rows_per_chunk_csv} "
          f"for sparse CSV reading.")

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
    Convert a sparse DataFrame (time, x, y, last-4 feature columns) for a
    GIVEN local time range [t_start, t_end] to a dense tensor:

        dense: (B=1, T_block, C=4, H=len(y_unique), W=len(x_unique))

    - The time axis is an hourly range from t_start to t_end (inclusive).
    - The spatial axes use the GLOBAL fixed x_unique / y_unique.
    """
    if df.shape[1] < 5:
        raise ValueError(
            "Chunk DataFrame must have at least 5 columns: "
            "time, x, y, and feature columns."
        )

    t_raw = pd.to_datetime(df.iloc[:, 0])
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)

    feats = df.iloc[:, -4:].astype(float).fillna(0.0)
    C = feats.shape[1]
    if C != 4:
        # Not strictly required, but we log a warning.
        print(f"[WARN] Expected 4 feature columns, found {C}. Proceeding anyway.")

    # Local hourly timeline for the block.
    t_block = pd.date_range(start=t_start, end=t_end, freq="h")
    t_block_vals = t_block.values
    T_block = len(t_block_vals)

    # Index maps
    time_to_index = {ts: i for i, ts in enumerate(t_block_vals)}
    x_index = {val: i for i, val in enumerate(x_unique)}
    y_index = {val: i for i, val in enumerate(y_unique)}

    # Allocate dense tensor
    B = 1
    H = len(y_unique)
    W = len(x_unique)
    dense = torch.zeros((B, T_block, C, H, W),
                        device=device, dtype=torch.float32)

    # Filter rows whose time really lies within [t_start, t_end]
    mask = (t_raw >= t_start) & (t_raw <= t_end)
    if not mask.any():
        return dense  # block is all zeros

    df = df.loc[mask]
    t_raw = t_raw[mask]
    x_raw = x_raw[mask]
    y_raw = y_raw[mask]
    feats = df.iloc[:, -4:].astype(float).fillna(0.0).values  # (N, C)

    # Convert to indices
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
    Build a dense (1, T_block, C, H, W) tensor for the time range
    [t_start, t_end], using the GLOBAL fixed grid (x_unique, y_unique).

    The CSV is read in sparse chunks of size rows_per_chunk_csv and filtered
    to only include rows within [t_start, t_end]. Multiple sparse chunks are
    loaded sequentially as needed to assemble the dense block.
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
        # No data rows in this time range; return all-zero dense block.
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
# 2. COMPRESSION (2D → vector) — kept for completeness
# =========================================================

def compress_2d_to_vector(tensor_5d):
    """
    tensor_5d: (B, T, C, H, W)
    Returns:   (B, T, C) = spatial mean over H, W.
    """
    return torch.mean(tensor_5d, dim=[3, 4])


# =========================================================
# 3. Core sliding-window processing over full timeline
#    with RAM/VRAM-limited dense blocks
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
    """
    Process ALL sliding windows over the FULL dense timeline, while
    respecting RAM/VRAM limits via dense time blocks.

    - t_full: global hourly timeline (DatetimeIndex).
    - x_unique, y_unique: global spatial grid.
    - timestep_capacity: maximum timesteps we allow in a dense block.
    - rows_per_chunk_csv: sparse CSV chunk size for building dense blocks.

    Every global sliding window of length (seq_len_in + future_steps) is
    processed exactly once, even if it spans sparse CSV chunk boundaries.
    """
    device = next(convlstm.parameters()).device
    T_total = len(t_full)
    W_len = seq_len_in + future_steps

    if T_total < W_len:
        print(
            f"[WARN] CSV {csv_path} has fewer timesteps ({T_total}) "
            f"than required window length ({W_len}); skipping."
        )
        return None

    # Determine dense-block size (in timesteps).
    if timestep_capacity is None or timestep_capacity >= T_total:
        T_block = T_total
    else:
        T_block = timestep_capacity

    # Ensure block can at least contain one full window.
    if T_block < W_len:
        T_block = W_len

    # Overlap blocks so that windows that cross block boundaries are seen.
    block_stride = T_block - (W_len - 1) if T_block > W_len else 1

    total_loss = 0.0
    n_windows = 0
    global_last_processed_start = -1  # global start index of last processed window

    # Iterate over blocks by global time index.
    # We only need to start blocks up to (T_total - W_len), because all windows
    # beyond that are shorter than required.
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
            # Not enough timesteps in this block to form a full window
            del dense_block
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        local_max_start = T_block_actual - W_len

        for local_start in range(local_max_start + 1):
            global_start = block_start_idx + local_start
            if global_start <= global_last_processed_start:
                # This window was already covered in a previous (overlapping) block.
                continue

            # Extract global window from this dense block.
            window = dense_block[:, local_start:local_start + W_len]   # (B, W_len, C, H, W)
            x_in = window[:, :seq_len_in]                              # (B, seq_len_in, C, H, W)
            y_true = window[:, seq_len_in:]                            # (B, future_steps, C, H, W)

            if is_train:
                optimizer.zero_grad()
                outputs, (h, c) = convlstm(x_in)

                # --- Autoregressive prediction for 'future_steps' ---
                preds = []
                h_t, c_t = h, c
                for _ in range(future_steps):
                    y_t = decoder(h_t)  # (B, C_out=4, H, W)
                    preds.append(y_t)
                    h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

                preds = torch.stack(preds, dim=1)   # (B, future_steps, C, H, W)
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

        # Free this block before moving on.
        del dense_block
        if device == "cuda":
            torch.cuda.empty_cache()

    if n_windows == 0:
        return None

    # Average loss per window across entire CSV.
    return total_loss / n_windows


# =========================================================
# 4. Train / Validate wrappers for a single CSV
# =========================================================

def train_single_csv_full(
    csv_path,
    convlstm,
    decoder,
    criterion,
    optimizer,
    epoch,
    seq_len_in,
    future_steps,
    t_full,
    x_unique,
    y_unique,
    timestep_capacity,
    rows_per_chunk_csv,
):
    return process_csv_with_blocks(
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
        is_train=True,
        optimizer=optimizer,
        epoch=epoch,
    )


def validate_single_csv_full(
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
):
    return process_csv_with_blocks(
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
        is_train=False,
        optimizer=None,
        epoch=0,
    )


# =========================================================
# 5. TRAIN LOOP (single train CSV, single val CSV)
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
    device = next(convlstm.parameters()).device

    # if load_checkpoint and os.path.exists(checkpoint_path):
    #     ckpt = torch.load(checkpoint_path, map_location=device)
    #     convlstm.load_state_dict(ckpt["convlstm_state_dict"])
    #     decoder.load_state_dict(ckpt["decoder_state_dict"])
    #     print(f"[CKPT] Loaded checkpoint {checkpoint_path}")
    # else:
    print("[CKPT] Starting from scratch.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(convlstm.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    # model_config = {
    #     "convlstm": {
    #         "class": convlstm.__class__.__name__,
    #         "input_dim": 4,
    #         "hidden_dim": 4,
    #         "kernel_size": 3,
    #     },
    #     "decoder": {
    #         "class": decoder.__class__.__name__,
    #         "in": 4,
    #         "out": 4,
    #         "kernel": 1,
    #     },
    #     "device": str(device),
    # }
    #
    # optimizer_config = {
    #     "type": "Adam",
    #     "lr": optimizer.param_groups[0]["lr"],
    #     "betas": list(optimizer.param_groups[0].get("betas", (0.9, 0.999))),
    #     "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    # }

    for epoch in range(num_epochs):
        convlstm.train()
        decoder.train()

        train_loss = train_single_csv_full(
            train_csv_path,
            convlstm,
            decoder,
            criterion,
            optimizer,
            epoch,
            seq_len_in,
            future_steps,
            t_full,
            x_unique,
            y_unique,
            timestep_capacity,
            rows_per_chunk_csv,
        )
        if train_loss is None:
            train_loss = float("nan")

        convlstm.eval()
        decoder.eval()

        with torch.no_grad():
            val_loss = validate_single_csv_full(
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
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to single CSV used for training.",
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Path to single CSV used for validation.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument("--seq-len-in", type=int, default=9)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint.pth",
    )
    # parser.add_argument("--load-checkpoint", action="store_true")

    # Memory-limit arguments (in bytes)
    parser.add_argument(
        "--max-ram-bytes",
        type=int,
        default=None,
        help=(
            "Maximum bytes of combined sparse+dense data to assume "
            "can reside in main memory at once."
        ),
    )
    parser.add_argument(
        "--max-vram-bytes",
        type=int,
        default=None,
        help=(
            "Maximum bytes of dense tensor data to assume can reside "
            "on GPU at once."
        ),
    )

    start_time = time.time()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Analyze training CSV + memory limits, adjust window lengths, and
    # determine global timeline and grid.
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
    )

    convlstm = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    # 2D decoder because ConvLSTM2D hidden state is (B, C, H, W)
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

    torch.save(
        {
            "convlstm_state_dict": convlstm.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        args.checkpoint,
    )

    total_time = time.time() - start_time
    print("Model saved at: " + args.checkpoint)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")

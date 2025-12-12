import argparse
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import psycopg2
from psycopg2.extras import execute_values

from ConvLSTM import ConvLSTM2D


# =========================================================
# GLOBAL DB CONFIG
# =========================================================
SEQ_LEN_IN = 9


# =========================================================
# 0. Row-size + memory capacity estimation (From No-DBMS script)
# =========================================================

def estimate_row_sizes(csv_path, max_sample_rows=1000):
    sizes = []
    with open(csv_path, "rb") as f:
        _ = f.readline()
        for i, line in enumerate(f):
            if i >= max_sample_rows:
                break
            sizes.append(len(line))

    if not sizes:
        raise ValueError(f"[MEM] No data rows found when sampling CSV: {csv_path}")

    n = len(sizes)
    avg_bytes = float(sum(sizes)) / n
    return avg_bytes


def adjust_window_lengths(seq_len_in_arg, future_steps_arg, timestep_capacity):
    if timestep_capacity is None:
        return seq_len_in_arg, future_steps_arg

    min_total = 3
    max_total = min(timestep_capacity, seq_len_in_arg + future_steps_arg)
    if max_total < min_total:
        max_total = min_total

    new_future = max(1, min(future_steps_arg, max_total - 1))
    new_seq = max(1, max_total - new_future)
    return new_seq, new_future


def analyze_csv_and_memory(
    csv_path,
    seq_len_in_arg,
    future_steps_arg,
    max_ram_bytes=None,
    max_vram_bytes=None,
    time_filter_range=None,
):
    """
    Analyze the CSV to:
      - infer the global time range and spatial grid
      - estimate how many timesteps we can fit into RAM/VRAM
      - optionally restrict the global timeline to a datetime range
        [t_start, t_end] (inclusive).

    time_filter_range: tuple (t_start, t_end) where each element is either
    a pandas.Timestamp / datetime / string or None.
    """
    avg_bytes = estimate_row_sizes(csv_path)

    # Scan for structure
    t_min, t_max = None, None
    x_values, y_values = set(), set()
    total_rows_scanned = 0

    scan_chunksize = 100000
    print(f"[MEM] Scanning CSV for global structure...")

    for chunk in pd.read_csv(csv_path, chunksize=scan_chunksize, usecols=[0, 1, 2]):
        if chunk.empty:
            continue
        total_rows_scanned += len(chunk)
        t_col = pd.to_datetime(chunk.iloc[:, 0])

        c_min, c_max = t_col.min(), t_col.max()
        t_min = c_min if t_min is None else min(t_min, c_min)
        t_max = c_max if t_max is None else max(t_max, c_max)

        x_values.update(chunk.iloc[:, 1].astype(float).unique())
        y_values.update(chunk.iloc[:, 2].astype(float).unique())

    if t_min is None or t_max is None:
        raise ValueError(f"[MEM] Could not infer time range from CSV: {csv_path}")

    # Build full hourly timeline over the entire dataset
    t_full = pd.date_range(start=t_min, end=t_max, freq="h")

    # Optional datetime filter
    if time_filter_range is not None:
        raw_start, raw_end = time_filter_range

        # Clamp to dataset range if partially outside
        if raw_start is None:
            t_start = t_min
        else:
            t_start = max(pd.to_datetime(raw_start), t_min)

        if raw_end is None:
            t_end = t_max
        else:
            t_end = min(pd.to_datetime(raw_end), t_max)

        if t_start > t_end:
            raise ValueError(
                f"[FILTER] Invalid time window: start {t_start} is after end {t_end}"
            )

        mask = (t_full >= t_start) & (t_full <= t_end)
        t_full = t_full[mask]

        if len(t_full) == 0:
            raise ValueError(
                f"[FILTER] Time filter resulted in 0 timesteps between "
                f"{t_start} and {t_end}."
            )

        print(
            f"[FILTER] Restricting to timestamps from {t_full[0]} to {t_full[-1]} "
            f"({len(t_full)} hours)."
        )

        # Update effective min/max for capacity estimation
        t_min, t_max = t_full[0], t_full[-1]

    T_total = len(t_full)
    x_unique = np.array(sorted(x_values), dtype=float)
    y_unique = np.array(sorted(y_values), dtype=float)

    # Memory calculation for a dense block
    H, W = len(y_unique), len(x_unique)
    bytes_per_timestep_dense = float(1 * 4 * H * W * 4)  # B=1, C=4, float32

    timestep_limits = []
    if max_ram_bytes:
        # Heuristic: RAM must hold the sparse CSV rows for a block + its dense tensor
        hours_total = max((t_max - t_min).total_seconds() / 3600.0, 1.0)
        rows_per_ts = total_rows_scanned / hours_total
        bytes_sparse = rows_per_ts * avg_bytes
        limit = int(max_ram_bytes // (bytes_per_timestep_dense + bytes_sparse))
        timestep_limits.append(max(3, limit))

    if max_vram_bytes:
        limit = int(max_vram_bytes // bytes_per_timestep_dense)
        timestep_limits.append(max(3, limit))

    timestep_capacity = min(timestep_limits) if timestep_limits else None
    if timestep_capacity:
        timestep_capacity = min(timestep_capacity, T_total)

    new_seq, new_fut = adjust_window_lengths(
        seq_len_in_arg, future_steps_arg, timestep_capacity
    )
    rows_per_chunk = 100000

    return t_full, x_unique, y_unique, timestep_capacity, rows_per_chunk, new_seq, new_fut


# =========================================================
# 1. DATABASE & COMPRESSION (From Memory-Aware Script)
# =========================================================

def init_db(reset_tables=True, seq_len=9):
    try:
        conn = psycopg2.connect(
            dbname="cosc6340_project_db",
            user="nanderson",
            password="",
            host="localhost",
            port="5432",
        )
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if reset_tables:
            cursor.execute("DROP TABLE IF EXISTS layer_computations CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS epoch_metrics;")
            cursor.execute("DROP TABLE IF EXISTS training_runs CASCADE;")
            conn.commit()

        # Metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                data_path TEXT NOT NULL,
                future_steps INT NOT NULL,
                ram_limit_bytes BIGINT,
                time_start TIMESTAMPTZ,
                time_end TIMESTAMPTZ,
                total_epochs INT,
                runtime_seconds DOUBLE PRECISION,
                model_config JSONB NOT NULL,
                optimizer_config JSONB NOT NULL,
                notes TEXT
            );
        """
        )

        # Metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id SERIAL PRIMARY KEY,
                run_id INT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
                epoch INT NOT NULL,
                train_loss DOUBLE PRECISION,
                val_loss DOUBLE PRECISION,
                learning_rate DOUBLE PRECISION,
                checkpoint_path TEXT
            );
        """
        )

        # Computations (Partitioned)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS layer_computations (
                id SERIAL,
                run_id INT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
                epoch INT NOT NULL,
                sample_path TEXT,
                time_step INT NOT NULL,
                embedding vector(4),
                notation JSONB
            ) PARTITION BY LIST (time_step);
        """
        )

        for t in range(seq_len):
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS layer_computations_t{t} "
                f"PARTITION OF layer_computations FOR VALUES IN ({t});"
            )

        conn.commit()
        print(f"[DB] Ready. Partitions created for seq_len={seq_len}.")
        return conn, cursor
    except Exception as e:
        print("[DB ERROR]", e)
        return None, None


def create_training_run(
    conn,
    cursor,
    data_path,
    future_steps,
    ram_limit,
    t_start_ts,
    t_end_ts,
    epochs,
    model_cfg,
    opt_cfg,
    notes,
):
    cursor.execute(
        """
        INSERT INTO training_runs (
            data_path,
            future_steps,
            ram_limit_bytes,
            time_start,
            time_end,
            total_epochs,
            model_config,
            optimizer_config,
            notes
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id;
    """,
        (
            data_path,
            future_steps,
            ram_limit,
            t_start_ts,
            t_end_ts,
            epochs,
            json.dumps(model_cfg),
            json.dumps(opt_cfg),
            notes,
        ),
    )
    run_id = cursor.fetchone()[0]
    conn.commit()
    return run_id


def update_run_runtime(conn, cursor, run_id, runtime):
    cursor.execute(
        "UPDATE training_runs SET runtime_seconds = %s WHERE id = %s;",
        (runtime, run_id),
    )
    conn.commit()


def log_epoch_metrics(conn, cursor, run_id, epoch, t_loss, v_loss, lr, ckpt):
    cursor.execute(
        """
        INSERT INTO epoch_metrics (
            run_id,
            epoch,
            train_loss,
            val_loss,
            learning_rate,
            checkpoint_path
        )
        VALUES (%s,%s,%s,%s,%s,%s);
    """,
        (run_id, epoch, t_loss, v_loss, lr, ckpt),
    )
    conn.commit()


def flush_layer_computations(conn, cursor, buffer):
    if not buffer:
        return
    execute_values(
        cursor,
        """
        INSERT INTO layer_computations (
            run_id,
            epoch,
            sample_path,
            time_step,
            embedding,
            notation
        )
        VALUES %s
    """,
        buffer,
    )
    conn.commit()
    buffer.clear()


def compress_2d_to_vector(tensor_5d):
    # Mean over H, W
    return torch.mean(tensor_5d, dim=[3, 4])


# =========================================================
# 2. Loading Dense Blocks (From No-DBMS script)
# =========================================================

def load_sparse_df_to_dense_2d_fixed_grid(
    df, t_start, t_end, x_unique, y_unique, device="cpu"
):
    t_block = pd.date_range(start=t_start, end=t_end, freq="h")
    T_block = len(t_block)

    time_to_idx = {ts: i for i, ts in enumerate(t_block)}
    x_to_idx = {v: i for i, v in enumerate(x_unique)}
    y_to_idx = {v: i for i, v in enumerate(y_unique)}

    dense = torch.zeros(
        (1, T_block, 4, len(y_unique), len(x_unique)),
        device=device,
        dtype=torch.float32,
    )

    t_col = pd.to_datetime(df.iloc[:, 0])
    mask = (t_col >= t_start) & (t_col <= t_end)
    df_sub = df.loc[mask]
    if df_sub.empty:
        return dense

    t_vals = pd.to_datetime(df_sub.iloc[:, 0])
    x_vals = df_sub.iloc[:, 1]
    y_vals = df_sub.iloc[:, 2]
    feats = df_sub.iloc[:, -4:].fillna(0.0).values

    t_indices = [time_to_idx[t] for t in t_vals]
    x_indices = [x_to_idx[x] for x in x_vals]
    y_indices = [y_to_idx[y] for y in y_vals]

    dense[0, t_indices, :, y_indices, x_indices] = torch.tensor(
        feats, dtype=torch.float32, device=device
    )
    return dense


def build_dense_block(csv_path, t_start, t_end, x_unique, y_unique, device, chunksize):
    collected = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        t_col = pd.to_datetime(chunk.iloc[:, 0])
        if (t_col >= t_start).any() and (t_col <= t_end).any():
            collected.append(chunk[(t_col >= t_start) & (t_col <= t_end)])

    if not collected:
        # Create an all-zero block with the right shape
        return torch.zeros(
            (1, len(pd.date_range(t_start, t_end, freq="h")), 4, len(y_unique), len(x_unique)),
            device=device,
            dtype=torch.float32,
        )

    return load_sparse_df_to_dense_2d_fixed_grid(
        pd.concat(collected), t_start, t_end, x_unique, y_unique, device
    )


# =========================================================
# 3. Processing with Buffering
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
    rows_per_chunk,
    is_train,
    optimizer=None,
    epoch=0,
    # DB Args
    run_id=None,
    conn=None,
    cursor=None,
    layer_buffer=None,
):
    device = next(convlstm.parameters()).device
    T_total = len(t_full)
    W_len = seq_len_in + future_steps

    if T_total < W_len:
        return None

    # Respect RAM/VRAM capacity when choosing a block length
    T_block = (
        timestep_capacity
        if (timestep_capacity and timestep_capacity < T_total)
        else T_total
    )
    T_block = max(T_block, W_len)
    block_stride = max(1, T_block - W_len + 1)

    total_loss, n_windows = 0.0, 0
    H, W = len(y_unique), len(x_unique)

    for start_idx in range(0, T_total - W_len + 1, block_stride):
        end_idx = min(start_idx + T_block, T_total)
        dense_block = build_dense_block(
            csv_path,
            t_full[start_idx],
            t_full[end_idx - 1],
            x_unique,
            y_unique,
            device,
            rows_per_chunk,
        )

        # Slide within block
        B_actual, T_actual, _, _, _ = dense_block.shape
        local_max = T_actual - W_len

        for local_start in range(local_max + 1):
            window = dense_block[:, local_start : local_start + W_len]
            x_in = window[:, :seq_len_in]
            y_true = window[:, seq_len_in:]

            if is_train:
                optimizer.zero_grad()
                outputs, (h, c) = convlstm(x_in)

                # --- DB BUFFERING ---
                if layer_buffer is not None:
                    compressed = compress_2d_to_vector(outputs).detach().cpu().numpy()
                    for t in range(seq_len_in):
                        note = json.dumps(
                            {
                                "layer": "ConvLSTM2D",
                                "op": "GlobalAvgPool",
                                "timestep": int(t),
                                "shape": [H, W],
                                "math": {"pool": "mean(h_t)", "conv": "matmul(im2col)"},
                            }
                        )
                        layer_buffer.append(
                            (
                                run_id,
                                epoch,
                                csv_path,
                                int(t),
                                compressed[0, t].tolist(),
                                note,
                            )
                        )

                preds = []
                h_t, c_t = h, c
                for _ in range(future_steps):
                    y_t = decoder(h_t)
                    preds.append(y_t)
                    h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

                loss = criterion(torch.stack(preds, dim=1), y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                with torch.no_grad():
                    outputs, (h, c) = convlstm(x_in)
                    preds = []
                    h_t, c_t = h, c
                    for _ in range(future_steps):
                        y_t = decoder(h_t)
                        preds.append(y_t)
                        h_t, c_t = convlstm.cell(y_t, (h_t, c_t))
                    total_loss += criterion(torch.stack(preds, dim=1), y_true).item()

            n_windows += 1

        del dense_block

        # Flush periodically (per block) to avoid huge RAM usage
        if layer_buffer and len(layer_buffer) > 1000:
            flush_layer_computations(conn, cursor, layer_buffer)

    return (total_loss / n_windows) if n_windows > 0 else None


# =========================================================
# 4. MAIN TRAIN LOOP
# =========================================================

def train(args):
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse optional datetime filters from CLI
    if args.time_start is not None or args.time_end is not None:
        t_start_ts = pd.to_datetime(args.time_start) if args.time_start is not None else None
        t_end_ts = pd.to_datetime(args.time_end) if args.time_end is not None else None
        time_filter = (t_start_ts, t_end_ts)
    else:
        t_start_ts = None
        t_end_ts = None
        time_filter = None

    # Analyze data and respect memory limits when deciding block sizes
    t_full, x_u, y_u, t_cap, rows_chunk, seq_len, future = analyze_csv_and_memory(
        args.train_csv,
        args.seq_len_in,
        args.future_steps,
        args.max_ram_bytes,
        args.max_vram_bytes,
        time_filter,
    )

    # Init DB
    conn, cursor = init_db(reset_tables=not args.load_checkpoint, seq_len=seq_len)

    # Init Model
    model = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    decoder = nn.Conv2d(4, 4, kernel_size=1).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=1e-3
    )

    if args.load_checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["convlstm"])
        decoder.load_state_dict(ckpt["decoder"])
        print("Loaded Checkpoint.")

    # Log Run with actual datetime filters (if any)
    run_id = create_training_run(
        conn,
        cursor,
        args.train_csv,
        future,
        args.max_ram_bytes,
        t_start_ts,
        t_end_ts,
        args.epochs,
        {"model": "ConvLSTM"},
        {"optim": "Adam"},
        "Sparse Disk + PgVector",
    )

    # Loop
    t_loss = float("nan")
    v_loss = float("nan")
    for epoch in range(args.epochs):
        model.train()
        decoder.train()
        buffer = []

        train_loss = process_csv_with_blocks(
            args.train_csv,
            model,
            decoder,
            nn.MSELoss(),
            seq_len,
            future,
            t_full,
            x_u,
            y_u,
            t_cap,
            rows_chunk,
            True,
            optimizer,
            epoch,
            run_id,
            conn,
            cursor,
            buffer,
        )

        # Flush remaining layer computations
        flush_layer_computations(conn, cursor, buffer)

        # Validation (uses same time window and grid)
        model.eval()
        decoder.eval()
        val_loss = process_csv_with_blocks(
            args.val_csv,
            model,
            decoder,
            nn.MSELoss(),
            seq_len,
            future,
            t_full,
            x_u,
            y_u,
            t_cap,
            rows_chunk,
            False,
        )

        t_loss = train_loss if train_loss is not None else float("nan")
        v_loss = val_loss if val_loss is not None else float("nan")
        print(f"[Epoch {epoch+1}] Train: {t_loss:.4f} | Val: {v_loss:.4f}")

        if args.checkpoint:
            torch.save(
                {"convlstm": model.state_dict(), "decoder": decoder.state_dict()},
                args.checkpoint,
            )

        log_epoch_metrics(conn, cursor, run_id, epoch, t_loss, v_loss, 0.001, args.checkpoint)

    # Finish
    runtime = time.time() - start_time
    update_run_runtime(conn, cursor, run_id, runtime)
    print(f"Done. Runtime: {runtime:.2f}s")

    cursor.close()
    conn.close()
    return runtime, t_loss, v_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument("--seq-len-in", type=int, default=9)
    parser.add_argument("--checkpoint", default="checkpoint.pth")
    parser.add_argument("--load-checkpoint", action="store_true")

    # Memory Limits
    parser.add_argument("--max-ram-bytes", type=int, default=None)
    parser.add_argument("--max-vram-bytes", type=int, default=None)

    # Datetime filters (NEW: strings like "YYYY-MM-DD HH:MM:SS")
    parser.add_argument(
        "--time-start",
        type=str,
        default=None,
        help="Optional start datetime (inclusive), format 'YYYY-MM-DD HH:MM:SS'",
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default=None,
        help="Optional end datetime (inclusive), format 'YYYY-MM-DD HH:MM:SS'",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="training_log.json",
        help="Path to save experiment metrics",
    )
    args = parser.parse_args()
    runtime, t_loss, v_loss = train(args)

    # === LOCAL JSON LOGGING ===
    results = {
        "train_csv": args.train_csv,
        "epochs": args.epochs,
        "runtime_seconds": runtime,
        "train_loss": t_loss,
        "val_loss": v_loss,
        "max_ram_bytes": args.max_ram_bytes,
        "max_vram_bytes": args.max_vram_bytes,
        # Keep the original string forms for reproducibility
        "time_start": args.time_start,
        "time_end": args.time_end,
    }

    with open(args.log_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Experiment metrics saved to {args.log_file}")

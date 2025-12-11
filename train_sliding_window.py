import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import psycopg2
from psycopg2.extras import execute_values

from ConvLSTM import ConvLSTM2D


# ---------------------------------------------------------
# 0. CSV → Dense 2D Tensor
# ---------------------------------------------------------

def load_sparse_csv_to_dense_2d(
    csv_path,
    T=None, X=None, Y=None, Z=None,   # Z kept for API compatibility, ignored
    device="cpu",
):
    """
    Load a sparse CSV (time, x, y, 4 channels) and produce a dense tensor:

        dense: (B=1, T_dim, C=4, H=Y_dim, W=X_dim)

    - Time axis:
        * Build full hourly timeline from min(t_raw) to max(t_raw).
        * Any missing hours are included and stay all-zero in the tensor.
    - Spatial axes:
        * Build full set of unique x and unique y from the file.
        * Any missing (t, x, y) combination stays all-zero in the tensor.
    """

    df = pd.read_csv(csv_path)

    # ---- 0. Basic sanity check ----
    # We expect at least: time, x, y, and 4 channels = 7 columns minimum
    if df.shape[1] < 5:
        raise ValueError(
            "CSV must have at least 5 columns: time, x, y, and feature columns."
        )

    # ---- 1. Parse columns ----
    # Assumed layout: [time, x, y, ..., last 4 cols = channels]
    t_raw = pd.to_datetime(df.iloc[:, 0])          # timestamps
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)

    # Take the last 4 columns as the 4 channels (more robust than fixed indices)
    feats = df.iloc[:, -4:].astype(float).fillna(0.0)
    C = feats.shape[1]  # should be 4, but we infer it

    # ---- 2. Build full hourly timeline (fills missing timesteps) ----
    t_min, t_max = t_raw.min(), t_raw.max()
    # Full timeline with hourly frequency
    t_full = pd.date_range(start=t_min, end=t_max, freq="h")
    t_full_values = t_full.values
    nT_full = len(t_full_values)

    # If T is provided and larger than nT_full, pad with extra time steps at the end
    T_dim = T if (T is not None and T >= nT_full) else nT_full

    # Map each timestamp in the CSV to an index in [0, nT_full)
    time_to_index = {ts: i for i, ts in enumerate(t_full_values)}
    t_idx_np = np.fromiter(
        (time_to_index[ts] for ts in t_raw.values),
        dtype=np.int64,
        count=len(t_raw),
    )

    # ---- 3. Spatial indices (fills missing x,y grid positions) ----
    # Build the full set of unique x and y values seen in the entire file
    x_unique, x_inv = np.unique(x_raw.values, return_inverse=True)
    y_unique, y_inv = np.unique(y_raw.values, return_inverse=True)

    nX, nY = len(x_unique), len(y_unique)

    # Allow optional overrides for X_dim, Y_dim if you ever use them
    X_dim = X if (X is not None and X >= nX) else nX
    Y_dim = Y if (Y is not None and Y >= nY) else nY

    B = 1  # batch size

    # ---- 4. Allocate dense tensor filled with zeros ----
    # Shape: (B, T, C, H, W) = (1, T_dim, C, Y_dim, X_dim)
    dense = torch.zeros((B, T_dim, C, Y_dim, X_dim), device=device, dtype=torch.float32)

    # ---- 5. Scatter the observed values into the dense grid ----
    t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=device)
    x_idx = torch.tensor(x_inv, dtype=torch.long, device=device)
    y_idx = torch.tensor(y_inv, dtype=torch.long, device=device)

    feats_t = torch.tensor(feats.values, dtype=torch.float32, device=device)

    # Fill only the positions present in the CSV; everything else stays zero
    dense[0, t_idx, :, y_idx, x_idx] = feats_t

    # We return a dummy z_unique to keep the same return signature shape-wise.
    z_unique = np.array([0], dtype=int)

    return dense, t_full, x_unique, y_unique, z_unique


# ---------------------------------------------------------
# 1. DATABASE
# ---------------------------------------------------------

def init_db(reset_tables=True):
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

        # ALWAYS drop & recreate for now to fix schema mismatch
        if reset_tables:
            cursor.execute("DROP TABLE IF EXISTS epoch_metrics;")
            cursor.execute("DROP TABLE IF EXISTS layer_computations;")
            cursor.execute("DROP TABLE IF EXISTS training_runs;")
            conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                data_path TEXT NOT NULL,
                future_steps INT NOT NULL,
                model_config JSONB NOT NULL,
                optimizer_config JSONB NOT NULL,
                notes TEXT
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id SERIAL PRIMARY KEY,
                run_id INT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
                epoch INT NOT NULL,
                train_loss DOUBLE PRECISION,
                val_loss DOUBLE PRECISION,
                n_train_samples INT,
                n_val_samples INT,
                learning_rate DOUBLE PRECISION,
                checkpoint_path TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_computations (
                id SERIAL PRIMARY KEY,
                run_id INT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
                epoch INT NOT NULL,
                sample_path TEXT,
                time_step INT,
                embedding vector(4),
                notation JSONB
            );
        """)

        conn.commit()
        print("[DB] Ready.")
        return conn, cursor

    except Exception as e:
        print("[DB ERROR]", e)
        return None, None


def create_training_run(conn, cursor, data_path, future_steps, model_config, optimizer_config, notes=None):
    cursor.execute(
        """
        INSERT INTO training_runs (data_path, future_steps, model_config, optimizer_config, notes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (data_path, future_steps, json.dumps(model_config), json.dumps(optimizer_config), notes),
    )
    run_id = cursor.fetchone()[0]
    conn.commit()
    print(f"[DB] Created training_run id={run_id}")
    return run_id


def log_epoch_metrics(
    conn, cursor, run_id, epoch,
    train_loss, val_loss, n_train_samples, n_val_samples,
    learning_rate, checkpoint_path,
):
    cursor.execute(
        """
        INSERT INTO epoch_metrics (
            run_id, epoch, train_loss, val_loss,
            n_train_samples, n_val_samples,
            learning_rate, checkpoint_path, created_at
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW());
        """,
        (
            run_id, epoch, train_loss, val_loss,
            n_train_samples, n_val_samples, learning_rate, checkpoint_path
        )
    )
    conn.commit()


# ---------------------------------------------------------
# 2. COMPRESSION (2D → vector)
# ---------------------------------------------------------

def compress_2d_to_vector(tensor_5d):
    """
    tensor_5d: (B, T, C, H, W)
    Returns:   (B, T, C) = spatial mean over H, W.
    """
    return torch.mean(tensor_5d, dim=[3, 4])


# ---------------------------------------------------------
# 3. TRAINING ON ONE CSV (SLIDING WINDOWS)
# ---------------------------------------------------------

def train_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    optimizer,
    epoch,
    run_id,
    layer_buffer,
    seq_len_in=9,
    future_steps=3,
    T=None, X=None, Y=None, Z=None,
):
    """
    Train on ALL possible time windows of length SEQ_LEN_IN from a single CSV.
    Each window produces one (x_in, y_true) pair and one optimizer step.
    Embeddings are buffered in `layer_buffer` with GLOBAL time_step indices.
    """
    device = next(convlstm.parameters()).device

    dense, _, _, _, _ = load_sparse_csv_to_dense_2d(csv_path, T, X, Y, Z, device=device)
    # dense: (B, T_total, C, H, W)
    B, T_total, C, H, W = dense.shape

    min_required = seq_len_in + future_steps
    if T_total < min_required:
        print(f"[WARN] File {csv_path} skipped (not enough time steps: T={T_total}, need ≥ {min_required}).")
        return None

    # Number of sliding windows: start indices 0..(T_total - min_required)
    max_start = T_total - min_required
    if max_start < 0:
        print(f"[WARN] File {csv_path} skipped (no valid windows).")
        return None

    total_loss = 0.0
    n_windows = 0

    for start_t in range(max_start + 1):  # inclusive
        end_t = start_t + min_required  # exclusive
        window = dense[:, start_t:end_t]   # (B, seq_len_in + future_steps, C, H, W)

        x_in = window[:, :seq_len_in]                  # (B, seq_len_in, C, H, W)
        y_true = window[:, seq_len_in:]               # (B, future_steps, C, H, W)

        optimizer.zero_grad()
        outputs, (h, c) = convlstm(x_in)              # outputs: (B, seq_len_in, hidden_dim, H, W)

        # --- Buffer embeddings (per window, per timestep) ---
        if layer_buffer is not None:
            compressed = compress_2d_to_vector(outputs).detach().cpu().numpy()  # (B, seq_len_in, hidden_dim)

            for t_local in range(seq_len_in):
                t_global = start_t + t_local
                notation_json = json.dumps({
                    "layer": "ConvLSTM2D",
                    "operation": "GlobalAvgPooling",
                    "source_op": "Conv2dGEMM",
                    "timestep_local": t_local,
                    "timestep_global": t_global,
                    "input_shape": [int(H), int(W)],
                    "math": {"pool": "mean(h_t)", "conv": "matmul(im2col(...))"}
                })
                layer_buffer.append(
                    (run_id, epoch, csv_path, t_global, compressed[0, t_local].tolist(), notation_json)
                )

        # --- Autoregressive prediction for 'future_steps' ---
        preds = []
        h_t, c_t = h, c
        for _ in range(future_steps):
            y_t = decoder(h_t)          # (B, C_out=4, H, W)
            preds.append(y_t)
            # feed prediction back into ConvLSTM cell
            h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

        preds = torch.stack(preds, dim=1)   # (B, future_steps, C, H, W)
        loss = criterion(preds, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_windows += 1

    if n_windows == 0:
        return None

    return total_loss / n_windows


# ---------------------------------------------------------
# 4. VALIDATION (SLIDING WINDOWS)
# ---------------------------------------------------------

def validate_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    seq_len_in,
    future_steps=3,
    T=None, X=None, Y=None, Z=None,
):
    """
    Validation over ALL sliding windows of length SEQ_LEN_IN.
    Returns average loss over windows for this file.
    """
    device = next(convlstm.parameters()).device

    dense, _, _, _, _ = load_sparse_csv_to_dense_2d(csv_path, T, X, Y, Z, device=device)
    B, T_total, C, H, W = dense.shape

    min_required = seq_len_in + future_steps
    if T_total < min_required:
        return None

    max_start = T_total - min_required
    if max_start < 0:
        return None

    total_loss = 0.0
    n_windows = 0

    with torch.no_grad():
        for start_t in range(max_start + 1):
            end_t = start_t + min_required
            window = dense[:, start_t:end_t]

            x_in = window[:, :seq_len_in]
            y_true = window[:, seq_len_in:]

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

    if n_windows == 0:
        return None

    return total_loss / n_windows


# ---------------------------------------------------------
# 5. TRAIN LOOP (BULK INSERT PER EPOCH)
# ---------------------------------------------------------

def train(
    data_path,
    convlstm,
    decoder,
    num_epochs,
    seq_len_in,
    future_steps,
    checkpoint_path,
    load_checkpoint=False,
):
    device = next(convlstm.parameters()).device

    # Load checkpoint if necessary
    if load_checkpoint and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        convlstm.load_state_dict(ckpt["convlstm_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        print(f"[CKPT] Loaded checkpoint {checkpoint_path}")
    else:
        print("[CKPT] Starting from scratch.")

    conn, cursor = init_db(reset_tables=not load_checkpoint)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(convlstm.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    # Determine CSVs
    if os.path.isdir(data_path):
        all_csvs = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".csv")
        )
    else:
        all_csvs = [data_path]

    n_total = len(all_csvs)
    n_train = int(0.8 * n_total)
    train_files = all_csvs[:n_train]
    val_files = all_csvs[n_train:]
    n_val = len(val_files)

    print(f"[DATA] {n_total} samples → {n_train} train + {n_val} val")

    # Register training run in DB
    model_config = {
        "convlstm": {
            "class": convlstm.__class__.__name__,
            "input_dim": 4,
            "hidden_dim": 4,
            "kernel_size": 3,
        },
        "decoder": {
            "class": decoder.__class__.__name__,
            "in": 4,
            "out": 4,
            "kernel": 1,
        },
        "device": str(device),
    }

    optimizer_config = {
        "type": "Adam",
        "lr": optimizer.param_groups[0]["lr"],
        "betas": list(optimizer.param_groups[0].get("betas", (0.9, 0.999))),
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    }

    run_id = create_training_run(
        conn, cursor,
        data_path=data_path,
        future_steps=future_steps,
        model_config=model_config,
        optimizer_config=optimizer_config,
        notes=f"load_checkpoint={load_checkpoint}",
    )

    # -------------------------------
    # Main Loop
    # -------------------------------
    for epoch in range(num_epochs):
        convlstm.train()
        decoder.train()
        train_losses = []

        # fresh buffer for this epoch
        layer_buffer = []

        # ---- TRAIN ----
        for csv in train_files:
            loss = train_single_csv(
                csv, convlstm, decoder,
                criterion, optimizer,
                epoch, run_id,
                layer_buffer=layer_buffer,
                seq_len_in=seq_len_in,
                future_steps=future_steps
            )
            if loss is not None:
                train_losses.append(loss)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ---- VALIDATION ----
        convlstm.eval()
        decoder.eval()
        val_losses = []

        with torch.no_grad():
            for csv in val_files:
                loss = validate_single_csv(csv, convlstm, decoder, criterion, seq_len_in, future_steps)
                if loss is not None:
                    val_losses.append(loss)

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"[EPOCH {epoch+1}/{num_epochs}] train={train_loss:.6f}  val={val_loss:.6f}")

        # ---- Save checkpoint ----
        if checkpoint_path:
            torch.save({
                "convlstm_state_dict": convlstm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            }, checkpoint_path)

        # ---- Log epoch metrics (single row) ----
        lr = optimizer.param_groups[0]["lr"]
        log_epoch_metrics(
            conn, cursor, run_id, epoch,
            train_loss, val_loss,
            len(train_files), len(val_files),
            lr,
            os.path.abspath(checkpoint_path) if checkpoint_path else None,
        )

        # ---- Bulk-insert buffered layer_computations for this epoch ----
        if layer_buffer:
            execute_values(
                cursor,
                """
                INSERT INTO layer_computations
                    (run_id, epoch, sample_path, time_step, embedding, notation)
                VALUES %s
                """,
                layer_buffer
            )
            conn.commit()
            print(f"[DB] Inserted {len(layer_buffer)} layer_computations rows for epoch {epoch}")

    # optional: close connection
    if conn is not None:
        cursor.close()
        conn.close()


# ---------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument("--seq-len-in", type=int, default=9)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    parser.add_argument("--load-checkpoint", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    convlstm = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    # 2D decoder because ConvLSTM2D hidden state is (B, C, H, W)
    decoder = nn.Conv2d(4, 4, kernel_size=1).to(device)

    train(
        data_path=args.data,
        convlstm=convlstm,
        decoder=decoder,
        num_epochs=args.epochs,
        future_steps=args.future_steps,
        seq_len_in=args.seq_len_in,
        checkpoint_path=args.checkpoint,
        load_checkpoint=args.load_checkpoint,
    )

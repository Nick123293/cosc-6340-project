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


# =========================================================
# GLOBAL "RAM LIMIT" SIMULATION PARAMETERS
# =========================================================

# Cap for how much CSV data we read into RAM at once
CSV_MAX_BYTES_PER_CHUNK = 3 * 1024 * 1024  # ~3MB

# Cap for how large the *dense* tensor can be (in bytes).
# This directly limits how many time steps we materialize.
DENSE_MAX_BYTES = 3 * 1024 * 1024  # ~3MB

SEQ_LEN_IN = 9  # input sequence length

def estimate_rows_per_chunk(csv_path, max_bytes=CSV_MAX_BYTES_PER_CHUNK, sample_rows=1000):
    """
    Estimate how many rows we can read per chunk so that the in-memory
    DataFrame is ~max_bytes in size.

    This is a heuristic to simulate a RAM cap on CSV loading.
    """
    sample = pd.read_csv(csv_path, nrows=sample_rows)
    if len(sample) == 0:
        return 1

    bytes_total = sample.memory_usage(deep=True).sum()
    bytes_per_row = bytes_total / len(sample)
    if bytes_per_row <= 0:
        return 1

    rows_per_chunk = int(max_bytes // bytes_per_row)
    return max(rows_per_chunk, 1)

def stream_dense_windows_from_csv(
    csv_path,
    seq_len_in,
    future_steps,
    T=None, X=None, Y=None, Z=None,
    device="cpu",
    dense_max_bytes=DENSE_MAX_BYTES,
    window_stride=None,
):
    rows_per_chunk = estimate_rows_per_chunk(csv_path)

    t_min = None
    t_max = None
    x_vals = set()
    y_vals = set()

    for chunk in pd.read_csv(csv_path, usecols=[0,1,2], chunksize=rows_per_chunk):
        t_chunk = pd.to_datetime(chunk.iloc[:,0])
        x_chunk = chunk.iloc[:,1].astype(float)
        y_chunk = chunk.iloc[:,2].astype(float)

        cmin = t_chunk.min()
        cmax = t_chunk.max()
        t_min = cmin if t_min is None else min(t_min, cmin)
        t_max = cmax if t_max is None else max(t_max, cmax)

        x_vals.update(x_chunk.unique())
        y_vals.update(y_chunk.unique())

    if t_min is None:
        return  # no data

    x_unique = np.array(sorted(x_vals), dtype=float)
    y_unique = np.array(sorted(y_vals), dtype=float)

    X_dim = X if (X and X >= len(x_unique)) else len(x_unique)
    Y_dim = Y if (Y and Y >= len(y_unique)) else len(y_unique)

    x_to_index = {v:i for i,v in enumerate(x_unique)}
    y_to_index = {v:i for i,v in enumerate(y_unique)}

    t_full = pd.date_range(start=t_min, end=t_max, freq="h")
    nT_full = len(t_full)
    time_to_idx = {ts:i for i,ts in enumerate(t_full)}
    B = 1
    C = 4
    bytes_per_element = 4
    bytes_per_timestep = B * C * Y_dim * X_dim * bytes_per_element

    max_T_fit = max(int(dense_max_bytes // bytes_per_timestep), 1)
    T_window = max_T_fit

    if window_stride is None:
        window_stride = T_window

    # n_windows = int(np.ceil(nT_full / window_stride)) FOR TESTING
    # print(f"[INFO] nT_full={nT_full}, T_window={T_window}, total_windows={n_windows}")

    buckets = [[] for _ in range(nT_full)]

    for chunk in pd.read_csv(csv_path, chunksize=rows_per_chunk):
        t_raw  = pd.to_datetime(chunk.iloc[:,0])
        x_raw  = chunk.iloc[:,1].astype(float)
        y_raw  = chunk.iloc[:,2].astype(float)
        feats  = chunk.iloc[:,-4:].astype(float).fillna(0.0).values

        for ts, x, y, f in zip(t_raw, x_raw, y_raw, feats):
            idx = time_to_idx.get(ts)
            if idx is None:
                continue
            buckets[idx].append((x_to_index[x], y_to_index[y], f))

    # ---------------------------------------------------------
    # PASS 3 — Yield dense windows populated from buckets
    # ---------------------------------------------------------
    device = torch.device(device)

    for start in range(0, nT_full, window_stride):
        end = min(start + T_window, nT_full)
        T_win = end - start

        if T_win < seq_len_in + 1:
            continue

        dense = torch.zeros((1, T_win, C, Y_dim, X_dim),
                            device=device, dtype=torch.float32)

        # populate
        for t in range(T_win):
            for (xi, yi, f) in buckets[start + t]:
                dense[0, t, :, yi, xi] = torch.tensor(f, device=device)

        seq_len_eff = seq_len_in
        future_steps_eff = min(future_steps, T_win - seq_len_eff)
        if future_steps_eff <= 0:
            continue

        yield dense, t_full[start:end], x_unique, y_unique, np.array([0]), seq_len_eff, future_steps_eff



# ---------------------------------------------------------
# 2. DATABASE
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
            learning_rate, checkpoint_path
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
        """,
        (
            run_id, epoch, train_loss, val_loss,
            n_train_samples, n_val_samples, learning_rate, checkpoint_path
        )
    )
    conn.commit()


# ---------------------------------------------------------
# 3. COMPRESSION (2D → vector)
# ---------------------------------------------------------

def compress_2d_to_vector(tensor_5d):
    """
    tensor_5d: (B, T, C, H, W)
    Returns:  (B, T, C) = spatial mean over H, W.
    """
    return torch.mean(tensor_5d, dim=[3, 4])


# ---------------------------------------------------------
# 4. TRAINING ON ONE CSV (ALL WINDOWS)
# ---------------------------------------------------------

def train_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    optimizer,
    epoch,
    run_id,
    conn,
    cursor,
    future_steps=3,
    T=None, X=None, Y=None, Z=None,
):
    device = next(convlstm.parameters()).device

    total_loss = 0.0
    n_windows = 0

    for (dense, t_window, x_unique, y_unique, z_unique,
         seq_len_eff, future_steps_eff) in stream_dense_windows_from_csv(
        csv_path,
        seq_len_in=SEQ_LEN_IN,
        future_steps=future_steps,
        T=T, X=X, Y=Y, Z=Z,
        device=device,
        dense_max_bytes=DENSE_MAX_BYTES,
    ):
        B, T_total, C, H, W = dense.shape

        if T_total < seq_len_eff + future_steps_eff:
            continue

        x_in = dense[:, :seq_len_eff]   # (B, seq_len_eff, C, H, W)
        y_true = dense[:, seq_len_eff:seq_len_eff + future_steps_eff]  # (B, future_steps_eff, C, H, W)

        optimizer.zero_grad()
        outputs, (h, c) = convlstm(x_in)  # (B, seq_len_eff, hidden_dim, H, W)

        # --- Store embeddings for this window ---
        if conn is not None and run_id is not None:
            compressed = compress_2d_to_vector(outputs).detach().cpu().numpy()

            rows = []
            for t in range(seq_len_eff):
                notation_json = json.dumps({
                    "layer": "ConvLSTM2D",
                    "operation": "GlobalAvgPooling",
                    "source_op": "Conv2dGEMM",
                    "timestep": int(t),
                    "input_shape": [int(H), int(W)],
                    "math": {"pool": "mean(h_t)", "conv": "matmul(im2col(...))"}
                })
                rows.append((run_id, epoch, csv_path, int(t), compressed[0, t].tolist(), notation_json))

            execute_values(cursor,
                """
                INSERT INTO layer_computations
                    (run_id, epoch, sample_path, time_step, embedding, notation)
                VALUES %s
                """,
                rows
            )
            conn.commit()

        # --- Predict autoregressively for future_steps_eff ---
        preds = []
        h_t, c_t = h, c
        for _ in range(future_steps_eff):
            y_t = decoder(h_t)          # (B, C_out=4, H, W)
            preds.append(y_t)
            h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

        preds = torch.stack(preds, dim=1)  # (B, future_steps_eff, C, H, W)
        loss = criterion(preds, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_windows += 1

    if n_windows == 0:
        print(f"[WARN] File {csv_path} produced no usable windows.")
        return None

    return total_loss / n_windows


# ---------------------------------------------------------
# 5. VALIDATION (ALL WINDOWS)
# ---------------------------------------------------------

def validate_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    future_steps=3,
    T=None, X=None, Y=None, Z=None,
):
    device = next(convlstm.parameters()).device

    total_loss = 0.0
    n_windows = 0

    with torch.no_grad():
        for (dense, t_window, x_unique, y_unique, z_unique,
             seq_len_eff, future_steps_eff) in stream_dense_windows_from_csv(
            csv_path,
            seq_len_in=SEQ_LEN_IN,
            future_steps=future_steps,
            T=T, X=X, Y=Y, Z=Z,
            device=device,
            dense_max_bytes=DENSE_MAX_BYTES,
        ):
            B, T_total, C, H, W = dense.shape

            if T_total < seq_len_eff + future_steps_eff:
                continue

            x_in = dense[:, :seq_len_eff]
            y_true = dense[:, seq_len_eff:seq_len_eff + future_steps_eff]

            outputs, (h, c) = convlstm(x_in)

            preds = []
            h_t, c_t = h, c
            for _ in range(future_steps_eff):
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
# 6. TRAIN LOOP
# ---------------------------------------------------------

def train(
    data_path,
    convlstm,
    decoder,
    num_epochs,
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

        for csv in train_files:
            loss = train_single_csv(
                csv, convlstm, decoder,
                criterion, optimizer,
                epoch, run_id,
                conn, cursor,
                future_steps=future_steps
            )
            if loss is not None:
                train_losses.append(loss)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validation
        convlstm.eval()
        decoder.eval()
        val_losses = []

        for csv in val_files:
            loss = validate_single_csv(csv, convlstm, decoder, criterion, future_steps)
            if loss is not None:
                val_losses.append(loss)

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        print(f"[EPOCH {epoch+1}/{num_epochs}] train={train_loss:.6f}  val={val_loss:.6f}")

        # Save checkpoint
        if checkpoint_path:
            torch.save({
                "convlstm_state_dict": convlstm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            }, checkpoint_path)

        # Log epoch metrics
        lr = optimizer.param_groups[0]["lr"]
        log_epoch_metrics(
            conn, cursor, run_id, epoch,
            train_loss, val_loss,
            len(train_files), len(val_files),
            lr,
            os.path.abspath(checkpoint_path) if checkpoint_path else None,
        )


# ---------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=3)
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
        checkpoint_path=args.checkpoint,
        load_checkpoint=args.load_checkpoint,
    )

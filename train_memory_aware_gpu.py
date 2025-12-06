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
# GLOBAL "MEMORY LIMIT" SIMULATION PARAMETERS
# =========================================================

# We now assume:
#   - CPU RAM is "large enough" to hold the full dense tensor.
#   - GPU memory is limited to ~3 MB worth of (B,T,C,H,W) at a time.
GPU_MAX_BYTES = 3 * 1024 * 1024  # ~3MB on GPU per window

SEQ_LEN_IN = 9  # input sequence length (same as full memory version)


# ---------------------------------------------------------
# 0. CSV → Dense 2D Tensor ON CPU (load once per file)
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

    NOTE: This function always builds the tensor on CPU; GPU windowing happens later.
    """

    # Force CPU here; we only want this dense tensor in host RAM.
    device = torch.device("cpu")

    df = pd.read_csv(csv_path)

    # ---- 0. Basic sanity check ----
    if df.shape[1] < 5:
        raise ValueError(
            "CSV must have at least 5 columns: time, x, y, and feature columns."
        )

    # ---- 1. Parse columns ----
    t_raw = pd.to_datetime(df.iloc[:, 0])          # timestamps
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)

    # Take the last 4 columns as the 4 channels
    feats = df.iloc[:, -4:].astype(float).fillna(0.0)
    C = feats.shape[1]  # should be 4, but we infer it

    # ---- 2. Build full hourly timeline (fills missing timesteps) ----
    t_min, t_max = t_raw.min(), t_raw.max()
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
    x_unique, x_inv = np.unique(x_raw.values, return_inverse=True)
    y_unique, y_inv = np.unique(y_raw.values, return_inverse=True)

    nX, nY = len(x_unique), len(y_unique)

    X_dim = X if (X is not None and X >= nX) else nX
    Y_dim = Y if (Y is not None and Y >= nY) else nY

    B = 1  # batch size

    # ---- 4. Allocate dense tensor on CPU ----
    dense = torch.zeros(
        (B, T_dim, C, Y_dim, X_dim),
        device=device,
        dtype=torch.float32,
    )

    # ---- 5. Scatter observed values into dense grid ----
    t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=device)
    x_idx = torch.tensor(x_inv, dtype=torch.long, device=device)
    y_idx = torch.tensor(y_inv, dtype=torch.long, device=device)
    feats_t = torch.tensor(feats.values, dtype=torch.float32, device=device)

    dense[0, t_idx, :, y_idx, x_idx] = feats_t

    z_unique = np.array([0], dtype=int)
    return dense, t_full, x_unique, y_unique, z_unique


# ---------------------------------------------------------
# 1. GPU WINDOW SIZE (simulate GPU memory limit)
# ---------------------------------------------------------

def compute_T_window(
    dense_cpu,
    gpu_max_bytes=GPU_MAX_BYTES,
    seq_len_in=SEQ_LEN_IN,
    future_steps=3,
):
    """
    Given a dense CPU tensor (B, T_total, C, H, W), compute:
      - T_window: number of timesteps we can fit on GPU at once, under gpu_max_bytes
      - window_stride: stride for non-overlapping windows (we use T_window here)

    This version tries to use as *much* of the GPU memory budget as possible,
    not just the minimal seq_len_in + future_steps.
    """
    B, T_total, C, H, W = dense_cpu.shape

    bytes_per_element = dense_cpu.element_size()  # 4 for float32
    bytes_per_timestep = B * C * H * W * bytes_per_element

    if bytes_per_timestep <= 0:
        raise ValueError("Computed zero-sized spatial grid.")

    max_T_fit = max(int(gpu_max_bytes // bytes_per_timestep), 1)
    base_T_needed = seq_len_in + future_steps

    # We want to use as many timesteps as we can *while staying under the VRAM cap*.
    # - At minimum we need base_T_needed to have a full (seq_len_in, future_steps) pair.
    # - If the cap doesn't allow that, we fall back to the old behavior.
    if max_T_fit < base_T_needed:
        # Very tight GPU limit, fall back to the smaller window (but still >= 2)
        T_window = max(max_T_fit, 2)
        print(
            f"[WARN] Very tight GPU limit: bytes_per_timestep={bytes_per_timestep}, "
            f"max_T_fit={max_T_fit}, base_T_needed={base_T_needed}, using T_window={T_window}"
        )
    else:
        # Use as much as the cap will allow (but not more than the sequence length).
        T_window = min(max_T_fit, T_total)

    window_stride = T_window
    return T_window, window_stride


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


def flush_layer_computations(conn, cursor, layer_buffer):
    """
    Flush buffered rows into layer_computations in one batched INSERT.
    layer_buffer is a list of tuples:
        (run_id, epoch, sample_path, time_step, embedding, notation)
    """
    if not layer_buffer:
        return

    execute_values(
        cursor,
        """
        INSERT INTO layer_computations
            (run_id, epoch, sample_path, time_step, embedding, notation)
        VALUES %s
        """,
        layer_buffer,
    )
    conn.commit()
    layer_buffer.clear()


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
# 4. TRAINING ON ONE CSV (GPU-WINDOWED, BUFFERED DB LOGGING)
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
    layer_buffer=None,
):
    """
    Load full CSV into a dense CPU tensor once, then process it on the GPU
    in time windows that respect a simulated GPU memory cap.

    Instead of writing to the DBMS per window, we append rows to layer_buffer,
    which is then flushed once per epoch in train().
    """
    device = next(convlstm.parameters()).device

    # 1) Load full dense tensor on CPU
    dense_cpu, _, _, _, _ = load_sparse_csv_to_dense_2d(
        csv_path, T, X, Y, Z, device="cpu"
    )
    B, T_total, C, H, W = dense_cpu.shape

    if T_total < SEQ_LEN_IN + future_steps:
        print(f"[WARN] File {csv_path} skipped (not enough time steps).")
        return None

    # 2) Compute GPU window size / stride
    T_window, window_stride = compute_T_window(
        dense_cpu,
        gpu_max_bytes=GPU_MAX_BYTES,
        seq_len_in=SEQ_LEN_IN,
        future_steps=future_steps,
    )

    total_loss = 0.0
    n_windows = 0

    # 3) Slide over full time axis in non-overlapping windows
    for start_t in range(0, T_total, window_stride):
        end_t = min(start_t + T_window, T_total)
        T_win = end_t - start_t
        if T_win < 2:
            continue

        # Effective seq_len and future_steps within this window
        seq_len_eff = min(SEQ_LEN_IN, T_win - 1)
        future_steps_eff = min(future_steps, T_win - seq_len_eff)
        if future_steps_eff <= 0:
            continue

        # CPU slices
        x_cpu = dense_cpu[:, start_t : start_t + seq_len_eff]  # (B, seq_len_eff, C, H, W)
        y_cpu = dense_cpu[:, start_t + seq_len_eff : start_t + seq_len_eff + future_steps_eff]

        # Move to GPU
        x_in = x_cpu.to(device, non_blocking=True)
        y_true = y_cpu.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs, (h, c) = convlstm(x_in)  # (B, seq_len_eff, hidden_dim, H, W)

        # --- Buffer embeddings for this window ---
        if (
            conn is not None
            and run_id is not None
            and layer_buffer is not None
        ):
            compressed = compress_2d_to_vector(outputs).detach().cpu().numpy()
            # compressed: (B, T_out, hidden_dim) = (1, seq_len_eff, hidden_dim)
            for t in range(seq_len_eff):
                notation_json = json.dumps({
                    "layer": "ConvLSTM2D",
                    "operation": "GlobalAvgPooling",
                    "source_op": "Conv2dGEMM",
                    "timestep": int(t),
                    "input_shape": [int(H), int(W)],
                    "math": {"pool": "mean(h_t)", "conv": "matmul(im2col(...))"}
                })
                # Note: time_step is relative to the *window* (0..seq_len_eff-1),
                # same as your previous code. If you want global time index, you
                # could use start_t + t instead.
                layer_buffer.append(
                    (run_id, epoch, csv_path, int(t), compressed[0, t].tolist(), notation_json)
                )

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
# 5. VALIDATION (GPU-WINDOWED, NO DB LOGGING)
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

    dense_cpu, _, _, _, _ = load_sparse_csv_to_dense_2d(
        csv_path, T, X, Y, Z, device="cpu"
    )
    B, T_total, C, H, W = dense_cpu.shape

    if T_total < SEQ_LEN_IN + future_steps:
        return None

    T_window, window_stride = compute_T_window(
        dense_cpu,
        gpu_max_bytes=GPU_MAX_BYTES,
        seq_len_in=SEQ_LEN_IN,
        future_steps=future_steps,
    )

    total_loss = 0.0
    n_windows = 0

    with torch.no_grad():
        for start_t in range(0, T_total, window_stride):
            end_t = min(start_t + T_window, T_total)
            T_win = end_t - start_t
            if T_win < 2:
                continue

            seq_len_eff = min(SEQ_LEN_IN, T_win - 1)
            future_steps_eff = min(future_steps, T_win - seq_len_eff)
            if future_steps_eff <= 0:
                continue

            x_cpu = dense_cpu[:, start_t : start_t + seq_len_eff]
            y_cpu = dense_cpu[:, start_t + seq_len_eff : start_t + seq_len_eff + future_steps_eff]

            x_in = x_cpu.to(device, non_blocking=True)
            y_true = y_cpu.to(device, non_blocking=True)

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

        # buffer for this epoch's layer_computations rows
        layer_rows_buffer = []

        for csv in train_files:
            loss = train_single_csv(
                csv, convlstm, decoder,
                criterion, optimizer,
                epoch, run_id,
                conn, cursor,
                future_steps=future_steps,
                layer_buffer=layer_rows_buffer,
            )
            if loss is not None:
                train_losses.append(loss)

        # Flush all buffered layer computations for this epoch in one batch
        flush_layer_computations(conn, cursor, layer_rows_buffer)

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

        # NOTE: this format is what your bash parser expects
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

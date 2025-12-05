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

from ConvLSTM import ConvLSTM3D


# ---------------------------------------------------------
# 0. CSV → Dense 3D Tensor
# ---------------------------------------------------------
def load_sparse_csv_to_dense_3d(
    csv_path,
    T=None, X=None, Y=None, Z=None,
    device="cpu",
):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 8:
        raise ValueError("CSV must have at least 8 columns.")

    t_raw = pd.to_datetime(df.iloc[:, 0])
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)
    z_raw = df.iloc[:, 3].astype(float)
    feats = df.iloc[:, 4:8].astype(float).fillna(0.0)

    t_vals = t_raw.values
    t_unique, t_inv = np.unique(t_vals, return_inverse=True)
    x_unique, x_inv = np.unique(x_raw.values, return_inverse=True)
    y_unique, y_inv = np.unique(y_raw.values, return_inverse=True)
    z_unique, z_inv = np.unique(z_raw.values, return_inverse=True)

    nT, nX, nY, nZ = len(t_unique), len(x_unique), len(y_unique), len(z_unique)

    T_dim = T if (T and T >= nT) else nT
    X_dim = X if (X and X >= nX) else nX
    Y_dim = Y if (Y and Y >= nY) else nY
    Z_dim = Z if (Z and Z >= nZ) else nZ

    B, C = 1, 4
    dense = torch.zeros((B, T_dim, C, Z_dim, Y_dim, X_dim), device=device)

    t_idx = torch.tensor(t_inv, device=device)
    x_idx = torch.tensor(x_inv, device=device)
    y_idx = torch.tensor(y_inv, device=device)
    z_idx = torch.tensor(z_inv, device=device)
    feats_t = torch.tensor(feats.values, dtype=torch.float32, device=device)


    dense[0, t_idx, :, z_idx, y_idx, x_idx] = feats_t
    return dense


# ---------------------------------------------------------
# 1. DATABASE HANDLING
# ---------------------------------------------------------
def init_db(reset_table=True):
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

        if reset_table:
            cursor.execute("DROP TABLE IF EXISTS layer_computations;")
            cursor.execute("""
                CREATE TABLE layer_computations (
                    id SERIAL PRIMARY KEY,
                    epoch INT,
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


# ---------------------------------------------------------
# 2. COMPRESSION
# ---------------------------------------------------------
def compress_3d_to_vector(tensor_5d):
    return torch.mean(tensor_5d, dim=[3, 4, 5])


# ---------------------------------------------------------
# 3. TRAINING FUNCTION
# ---------------------------------------------------------
def train_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    optimizer,
    epoch,
    conn,
    cursor,
    future_steps=1,
    T=None, X=None, Y=None, Z=None,
):
    device = next(convlstm.parameters()).device

    dense = load_sparse_csv_to_dense_3d(csv_path, T, X, Y, Z, device=device)
    B, T_total, C, D, H, W = dense.shape

    seq_len_in = 9
    if T_total < seq_len_in + future_steps:
        print(f"[WARN] File {csv_path} skipped: too few time steps.")
        return None

    x_input = dense[:, :seq_len_in]
    y_target = dense[:, seq_len_in:seq_len_in + future_steps]

    optimizer.zero_grad()
    outputs, (h, c) = convlstm(x_input)

    # --- Store embeddings in DB ---
    if conn:
        compressed = compress_3d_to_vector(outputs).detach().cpu().numpy()

        rows = []
        for t in range(seq_len_in):
            notation_json = json.dumps({
                "layer": "ConvLSTM",
                "operation": "GlobalAvgPooling",
                "source_op": "Conv3dGEMM",
                "timestep": t,
                "input_shape": [int(D), int(H), int(W)],
                "math": {
                    "pool": "mean(h_t, dim=[D,H,W])",
                    "conv": "Y = matmul(im2col([x_t,h_{t-1}]), W^T) + b",
                }
            })
            rows.append((epoch, t, compressed[0, t].tolist(), notation_json))

        execute_values(
            cursor,
            """INSERT INTO layer_computations (epoch,time_step,embedding,notation)
               VALUES %s""",
            rows
        )
        conn.commit()

    # --- Predict future ---
    preds = []
    h_t, c_t = h, c
    for _ in range(future_steps):
        y_t = decoder(h_t)
        preds.append(y_t)
        h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

    preds = torch.stack(preds, dim=1)

    loss = criterion(preds, y_target)
    loss.backward()
    optimizer.step()

    return loss.item()


### VALIDATION
def validate_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    future_steps=1,
    T=None, X=None, Y=None, Z=None,
):
    device = next(convlstm.parameters()).device

    dense = load_sparse_csv_to_dense_3d(csv_path, T, X, Y, Z, device=device)
    B, T_total, C, D, H, W = dense.shape

    seq_len_in = 9
    if T_total < seq_len_in + future_steps:
        return None

    x_input = dense[:, :seq_len_in]
    y_target = dense[:, seq_len_in:seq_len_in + future_steps]

    outputs, (h, c) = convlstm(x_input)

    preds = []
    h_t, c_t = h, c
    for _ in range(future_steps):
        y_t = decoder(h_t)
        preds.append(y_t)
        h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

    preds = torch.stack(preds, dim=1)

    loss = criterion(preds, y_target)
    return loss.item()

# ---------------------------------------------------------
# 4. TRAIN OVER DIRECTORY OR FILE
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

    # Load checkpoint if requested
    if load_checkpoint and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        convlstm.load_state_dict(ckpt["convlstm_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        print(f"[CKPT] Loaded checkpoint from {checkpoint_path}")
    else:
        print("[CKPT] Starting model from scratch.")

    conn, cursor = init_db(reset_table=not load_checkpoint)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(convlstm.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    # -------------------------------
    # Build training + validation sets
    # -------------------------------
    if os.path.isdir(data_path):
        all_csvs = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".csv")
        ])
    else:
        all_csvs = [data_path]

    n_total = len(all_csvs)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_files = all_csvs[:n_train]
    val_files = all_csvs[n_train:]

    print(f"[DATA] {n_total} samples found → {n_train} train + {n_val} val")

    # -------------------------------
    # Main Training Loop
    # -------------------------------
    for epoch in range(num_epochs):

        # ----- TRAIN -----
        convlstm.train()
        decoder.train()
        train_losses = []

        for csv_file in train_files:
            loss = train_single_csv(
                csv_file,
                convlstm,
                decoder,
                criterion,
                optimizer,
                epoch,
                conn,
                cursor,
                future_steps=future_steps,
            )
            if loss is not None:
                train_losses.append(loss)

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # ----- VALIDATION -----
        convlstm.eval()
        decoder.eval()
        val_losses = []

        with torch.no_grad():
            for csv_file in val_files:
                loss = validate_single_csv(
                    csv_file,
                    convlstm,
                    decoder,
                    criterion,
                    future_steps=future_steps,
                )
                if loss is not None:
                    val_losses.append(loss)

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        # Log results
        print(
            f"[EPOCH {epoch+1}/{num_epochs}] "
            f"train_loss={mean_train_loss:.6f}   val_loss={mean_val_loss:.6f}"
        )

        # Save checkpoint each epoch
        if checkpoint_path:
            torch.save({
                "convlstm_state_dict": convlstm.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
            }, checkpoint_path)
            print("[CKPT] Saved checkpoint.")



# ---------------------------------------------------------
# 5. COMMAND-LINE ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="CSV file or directory of CSVs.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth")
    parser.add_argument("--load-checkpoint", action="store_true",
                        help="Load checkpoint instead of training from scratch.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    convlstm = ConvLSTM3D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    decoder = nn.Conv3d(4, 4, kernel_size=1).to(device)

    train(
        data_path=args.data,
        convlstm=convlstm,
        decoder=decoder,
        num_epochs=args.epochs,
        future_steps=args.future_steps,
        checkpoint_path=args.checkpoint,
        load_checkpoint=args.load_checkpoint,
    )

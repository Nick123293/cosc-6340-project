#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ConvLSTM import ConvLSTM2D


# ---------------------------------------------------------
# 0. CSV → Dense 2D Tensor (same as in your train script)
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
    if df.shape[1] < 5:
        raise ValueError(
            "CSV must have at least 5 columns: time, x, y, and feature columns."
        )

    # ---- 1. Parse columns ----
    # Assumed layout: [time, x, y, ..., last 4 cols = channels]
    t_raw = pd.to_datetime(df.iloc[:, 0])          # timestamps
    x_raw = df.iloc[:, 1].astype(float)
    y_raw = df.iloc[:, 2].astype(float)

    # Take the last 4 columns as the 4 channels
    feats = df.iloc[:, -4:].astype(float).fillna(0.0)
    C = feats.shape[1]  # should be 4, but we infer it

    # ---- 2. Build full hourly timeline ----
    t_min, t_max = t_raw.min(), t_raw.max()
    t_full = pd.date_range(start=t_min, end=t_max, freq="h")
    t_full_values = t_full.values
    nT_full = len(t_full_values)

    # If T is provided and larger than nT_full, pad with extra time steps at the end
    T_dim = T if (T is not None and T >= nT_full) else nT_full

    # Map each timestamp to an index in [0, nT_full)
    time_to_index = {ts: i for i, ts in enumerate(t_full_values)}
    t_idx_np = np.fromiter(
        (time_to_index[ts] for ts in t_raw.values),
        dtype=np.int64,
        count=len(t_raw),
    )

    # ---- 3. Spatial indices ----
    x_unique, x_inv = np.unique(x_raw.values, return_inverse=True)
    y_unique, y_inv = np.unique(y_raw.values, return_inverse=True)

    nX, nY = len(x_unique), len(y_unique)

    X_dim = X if (X is not None and X >= nX) else nX
    Y_dim = Y if (Y is not None and Y >= nY) else nY

    B = 1  # batch size

    # ---- 4. Allocate dense tensor ----
    dense = torch.zeros((B, T_dim, C, Y_dim, X_dim), device=device, dtype=torch.float32)

    # ---- 5. Scatter observed values ----
    t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=device)
    x_idx = torch.tensor(x_inv, dtype=torch.long, device=device)
    y_idx = torch.tensor(y_inv, dtype=torch.long, device=device)

    feats_t = torch.tensor(feats.values, dtype=torch.float32, device=device)

    dense[0, t_idx, :, y_idx, x_idx] = feats_t

    z_unique = np.array([0], dtype=int)
    return dense, t_full, x_unique, y_unique, z_unique


# ---------------------------------------------------------
# 1. EVALUATE ONE CSV (mirrors validate_single_csv)
# ---------------------------------------------------------

def evaluate_single_csv(
    csv_path,
    convlstm,
    decoder,
    criterion,
    future_steps=3,
    seq_len_in=9,
    T=None, X=None, Y=None, Z=None,
):
    """
    Runs a forward pass on a single CSV and returns the MSE loss.
    """
    device = next(convlstm.parameters()).device

    dense, _, _, _, _ = load_sparse_csv_to_dense_2d(csv_path, T, X, Y, Z, device=device)
    B, T_total, C, H, W = dense.shape

    if T_total < seq_len_in + future_steps:
        print(f"[WARN] {csv_path} skipped (not enough timesteps: {T_total})")
        return None

    x_in = dense[:, :seq_len_in]
    y_true = dense[:, seq_len_in:seq_len_in + future_steps]

    with torch.no_grad():
        outputs, (h, c) = convlstm(x_in)

        preds = []
        h_t, c_t = h, c
        for _ in range(future_steps):
            y_t = decoder(h_t)      # (B, C_out=4, H, W)
            preds.append(y_t)
            h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

        preds = torch.stack(preds, dim=1)  # (B, future_steps, C, H, W)
        loss = criterion(preds, y_true)

    return loss.item()


# ---------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------------------------------

def load_trained_model(checkpoint_path, device="cpu"):
    """
    Load ConvLSTM2D + 1x1 Conv decoder from a .pth checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    convlstm = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
    decoder = nn.Conv2d(4, 4, kernel_size=1).to(device)

    convlstm.load_state_dict(ckpt["convlstm_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    convlstm.eval()
    decoder.eval()

    print(f"[TEST] Loaded checkpoint from {checkpoint_path}")
    return convlstm, decoder


# ---------------------------------------------------------
# 3. MAIN TEST LOOP
# ---------------------------------------------------------

def test_model(
    data_path,
    checkpoint_path,
    future_steps=3,
    seq_len_in=9,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    convlstm, decoder = load_trained_model(checkpoint_path, device=device)

    criterion = nn.MSELoss()

    # Decide what to test on
    if os.path.isdir(data_path):
        csv_files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".csv")
        )
    else:
        csv_files = [data_path]

    if not csv_files:
        print("[TEST] No CSV files found.")
        return

    print(f"[TEST] Evaluating on {len(csv_files)} file(s).")

    losses = []

    for csv in csv_files:
        loss = evaluate_single_csv(
            csv,
            convlstm,
            decoder,
            criterion,
            future_steps=future_steps,
            seq_len_in=seq_len_in,
        )
        if loss is not None:
            losses.append(loss)
            print(f"[FILE] {os.path.basename(csv)} → loss={loss:.6f}")

    if losses:
        mean_loss = float(np.mean(losses))
        print("\n===============================")
        print(f"[TEST RESULT] Mean loss = {mean_loss:.6f}")
        print("===============================\n")
    else:
        print("[TEST] No valid samples were evaluated (all too short?).")


# ---------------------------------------------------------
# 4. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained ConvLSTM2D model.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a CSV file or a directory of CSVs."
    )
    parser.add_argument(
        "--checkpoint", "--model",
        required=True,
        help="Path to the model checkpoint (.pth)."
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=3,
        help="Number of future timesteps the model predicts."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=9,
        help="Input sequence length (must match training)."
    )
    args = parser.parse_args()

    test_model(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        future_steps=args.future_steps,
        seq_len_in=args.seq_len,
    )

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ConvLSTM import ConvLSTM
def run_test(csv_path):
  device = "cpu"

  # --- Load dense tensor from CSV ---
  dense = load_sparse_coo_csv_to_dense(csv_path, device=device)
  # dense shape: (1, T=10, 1, H=100, W=10000)

  B, T, C, H, W = dense.shape

  # Use first 9 steps as input, last one as target
  seq_len_in   = 9
  future_steps = 1   # <<< CHANGE THIS to predict more future timesteps

  x_input   = dense[:, :seq_len_in]     # (1, 9, 1, H, W)
  y_target_full = dense[:, seq_len_in:] # (1, 1, 1, H, W) here (only t=9)

  # If future_steps > 1, you'd want y_target_full to have that many steps.

  # --- Build model ---
  in_channels = 1      # temperature only
  hidden_dim  = 8

  convlstm = ConvLSTM(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=3).to(device)
  decoder  = nn.Conv2d(hidden_dim, in_channels, kernel_size=1).to(device)

  # --- Simple forward pass (no real training here) ---
  convlstm.eval()
  decoder.eval()

  with torch.no_grad():
    outputs, (h, c) = convlstm(x_input)  # h: (1, hidden_dim, H, W)

    preds = []
    h_t, c_t = h, c

    for t in range(future_steps):
      y_t = decoder(h_t)         # (1, 1, H, W)
      preds.append(y_t)
      # autoregressive: feed prediction back in
      h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

    y_pred_seq = torch.stack(preds, dim=1)  # (1, future_steps, 1, H, W)

  print("Input shape:     ", x_input.shape)
  print("Target shape:    ", y_target_full.shape)
  print("Predicted shape: ", y_pred_seq.shape)

  return y_pred_seq, y_target_full

def load_sparse_coo_csv_to_dense(
  csv_path,
  T=10,
  X=100,
  Y=100,
  Z=100,
  H=100,           # flattened spatial H
  device="cpu",
):
  """
  Loads COO csv: columns t, x, y, z, value
  Returns dense tensor of shape (1, T, 1, H, W)
  where W = (X*Y*Z) // H
  """
  df = pd.read_csv(csv_path)
  W = (X * Y * Z) // H

  dense = torch.zeros(1, T, 1, H, W, device=device)

  for _, row in df.iterrows():
    t = int(row["t"])
    x = int(row["x"])
    y = int(row["y"])
    z = int(row["z"])
    val = float(row["value"])

    # Flatten (x,y,z) -> linear index
    idx = x * (Y * Z) + y * Z + z  # in [0, X*Y*Z - 1]
    h = idx // W
    w = idx % W

    dense[0, t, 0, h, w] = val

  return dense  # (1, T, 1, H, W)

if __name__ == "__main__":
    csv_path = "../data/weather_sparse_coo_100x100x100_t10.csv"  # adjust to your path
    y_pred, y_target = run_test(csv_path)



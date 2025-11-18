import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ConvLSTM import ConvLSTM3D
def train_on_csv_3d(csv_path):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Using device:", device)

  # --- Load dense tensor from sparse CSV ---
  T = 10
  X = Y = Z = 100
  dense = load_sparse_coo_csv_to_dense_3d(csv_path, T=T, X=X, Y=Y, Z=Z, device=device)
  # dense: (1, 10, 1, 100, 100, 100)

  B, T, C, D, H, W = dense.shape

  # Use first 9 steps as input, 10th step as target
  seq_len_in   = 9
  future_steps = 1   # <<< change this later if you want to predict more steps

  x_input = dense[:, :seq_len_in]           # (1, 9, 1, D, H, W)
  y_target_full = dense[:, seq_len_in:seq_len_in+future_steps]  # (1, 1, 1, D, H, W)

  # --- Build model ---
  in_channels = 1      # temperature only
  hidden_dim  = 4      # keep small; 3D convs on 100^3 are big!

  convlstm = ConvLSTM3D(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=3).to(device)
  decoder  = nn.Conv3d(hidden_dim, in_channels, kernel_size=1).to(device)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(list(convlstm.parameters()) + list(decoder.parameters()), lr=1e-3)

  num_epochs = 10   # bump up later if you want

  for epoch in range(num_epochs):
      convlstm.train()
      decoder.train()

      optimizer.zero_grad()

      # Forward pass through ConvLSTM on the past 9 timesteps
      outputs, (h, c) = convlstm(x_input)   # h: (1, hidden_dim, D, H, W)

      # Autoregressive prediction of future_steps timesteps
      preds = []
      h_t, c_t = h, c

      for t in range(future_steps):
          # predict one 3D temperature field
          y_t = decoder(h_t)               # (1, 1, D, H, W)
          preds.append(y_t)

          # feed prediction back into the ConvLSTM cell if you want multi-step prediction
          h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

      # Stack predictions over time: (B, future_steps, 1, D, H, W)
      y_pred_seq = torch.stack(preds, dim=1)

      loss = criterion(y_pred_seq, y_target_full)
      loss.backward()
      optimizer.step()

      print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}")

  # After training, do a final forward in eval mode
  convlstm.eval()
  decoder.eval()
  with torch.no_grad():
      outputs, (h, c) = convlstm(x_input)
      preds = []
      h_t, c_t = h, c
      for t in range(future_steps):
          y_t = decoder(h_t)
          preds.append(y_t)
          h_t, c_t = convlstm.cell(y_t, (h_t, c_t))
      y_pred_seq = torch.stack(preds, dim=1)

  print("Final predicted shape:", y_pred_seq.shape)   # (1, 1, 1, D, H, W)
  return y_pred_seq, y_target_full

def load_sparse_coo_csv_to_dense_3d(
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

  dense = torch.zeros(1, T, 1, X, Y, Z, device=device)

  for _, row in df.iterrows():
    t = int(row["t"])
    x = int(row["x"])
    y = int(row["y"])
    z = int(row["z"])
    val = float(row["value"])

    dense[0, t, 0, x, y, z] = val

  return dense  # (1, T, 1, X, Y, Z)

if __name__ == "__main__":
  csv_path = "../data/weather_sparse_coo_100x100x100_t10.csv" 
  y_pred, y_target = train_on_csv_3d(csv_path)



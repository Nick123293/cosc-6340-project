import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ConvLSTM import ConvLSTM3D
import tensorFormatTransformation as tFT

def train_on_csv_3d(
  csv_path,
  convlstm,
  decoder,
  num_epochs=10,
  future_steps=1,
  T=10,
  X=100,
  Y=100,
  Z=100,
  checkpoint_path=None,):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Using device:", device)

  #Load dense tensor from sparse CSV
  dense = tFT.load_sparse_coo_csv_to_dense_3d(csv_path, T=T, X=X, Y=Y, Z=Z, device=device)

  B, T, C, D, H, W = dense.shape
  assert T>=9+future_steps, "Not enough Timestamps in data."
  seq_len_in   = 9
  x_input = dense[:, :seq_len_in]           # (1, 9, 1, D, H, W)
  y_target_full = dense[:, seq_len_in:seq_len_in+future_steps]  # (1, 1, 1, D, H, W)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(
    list(convlstm.parameters()) + 
    list(decoder.parameters()), lr=1e-3
    )
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
  if checkpoint_path is not None:
    torch.save(
      {
        "convlstm_state_dict": convlstm.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "hidden_dim": convlstm.cell.hidden_dim,
        "input_channels": 1,   # temperature only for now
      },
      checkpoint_path,
  ) 
  print(f"[TRAIN] Saved checkpoint to {checkpoint_path}")
  return

def build_model(input_channels=1, hidden_dim=4, device="cpu"):
  convlstm = ConvLSTM3D(input_dim=input_channels, hidden_dim=hidden_dim, kernel_size=3).to(device)
  decoder = nn.Conv3d(hidden_dim, input_channels, kernel_size=1).to(device)
  return convlstm, decoder

if __name__ == "__main__":
  csv_path = "../data/weather_sparse_coo_100x100x100_t10.csv"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  convlstm, decoder = build_model(input_channels=1, hidden_dim=4, device=device)

  checkpoint_path = "../data/weather_convlstm3d_checkpoint.pth"

  train_on_csv_3d(
    csv_path,
    convlstm,
    decoder,
    num_epochs=5,
    future_steps=1,
    T=10,
    X=100,
    Y=100,
    Z=100,
    checkpoint_path=checkpoint_path,
  )




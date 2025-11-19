import tensorFormatTransformation as tFT
from ConvLSTM import ConvLSTM3D
import torch
def predict_next_on_csv_3d(
  csv_path,
  convlstm,
  decoder,
  future_steps=1,
  T=10,
  X=100,
  Y=100,
  Z=100,
):
  """
  Uses a *trained* convlstm+decoder to predict `future_steps` 3D fields
  given the first 9 timesteps from the CSV.
  """
  device = next(convlstm.parameters()).device

  dense = tFT.load_sparse_coo_csv_to_dense_3d(
    csv_path, T=T, X=X, Y=Y, Z=Z, device=device
  )
  B, T_real, C, D, H, W = dense.shape
  seq_len_in = 9
  assert T_real >= seq_len_in, "Not enough timesteps in data."

  x_input = dense[:, :seq_len_in]                       # (1, 9, 1, D, H, W)
  y_target_full = dense[:, seq_len_in:seq_len_in+future_steps]  # might be smaller if T_real=10 and future_steps>1

  convlstm.eval()
  decoder.eval()
  with torch.no_grad():
    outputs, (h, c) = convlstm(x_input)

    preds = []
    h_t, c_t = h, c
    for t in range(future_steps):
      y_t = decoder(h_t)                # (1, 1, D, H, W)
      preds.append(y_t)
      h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

    y_pred_seq = torch.stack(preds, dim=1)  # (1, future_steps, 1, D, H, W)

  print("[TEST] Predicted shape:", y_pred_seq.shape)
  if y_target_full.numel() > 0:
    print("[TEST] Target shape:   ", y_target_full.shape)

  return y_pred_seq, y_target_full

def build_model_from_checkpoint(checkpoint_path, device="cpu"):
  ckpt = torch.load(checkpoint_path, map_location=device)

  input_channels = ckpt.get("input_channels", 1)
  hidden_dim = ckpt.get("hidden_dim", 4)

  convlstm = ConvLSTM3D(
      input_dim=input_channels,
      hidden_dim=hidden_dim,
      kernel_size=3,
  ).to(device)

  decoder = torch.nn.Conv3d(hidden_dim, input_channels, kernel_size=1).to(device)

  convlstm.load_state_dict(ckpt["convlstm_state_dict"])
  decoder.load_state_dict(ckpt["decoder_state_dict"])

  print("[TEST] Loaded model from", checkpoint_path)
  return convlstm, decoder


if __name__ == "__main__":
  csv_path = "../data/weather_sparse_coo_100x100x100_t10.csv"
  checkpoint_path = "../data/weather_convlstm3d_checkpoint.pth"

  device = "cuda" if torch.cuda.is_available() else "cpu"

  convlstm, decoder = build_model_from_checkpoint(checkpoint_path, device=device)

  # Use the *trained* model for prediction only
  y_pred, y_target = predict_next_on_csv_3d(
    csv_path,
    convlstm,
    decoder,
    future_steps=1,
    T=10,
    X=100,
    Y=100,
    Z=100,
  )
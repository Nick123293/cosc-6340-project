import torch
import torch.nn as nn
import torch.optim as optim

class ConvLSTM3DCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, padding):
    """
    3D ConvLSTM cell (D, H, W spatial dims).

    input_dim:  # of input channels (e.g., 1 for temperature)
    hidden_dim: # of hidden channels (feature maps)
    kernel_size: int or tuple, e.g. 3
    padding: same-type padding, e.g. 1 to keep size
    """
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim

    self.conv = nn.Conv3d(
      in_channels=input_dim + hidden_dim,
      out_channels=4 * hidden_dim,
      kernel_size=kernel_size,
      padding=padding,
      bias=True,
    )

  def forward(self, x, state):
    """
    x:      (B, C_in, D, H, W)
    state:  (h_cur, c_cur), each (B, C_hidden, D, H, W)
    """
    h_cur, c_cur = state   # each (B, hidden_dim, D, H, W)

    # concat along channels
    combined = torch.cat([x, h_cur], dim=1)  # (B, C_in + C_hidden, D, H, W)
    combined_conv = self.conv(combined)      # (B, 4*C_hidden, D, H, W)

    cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)

    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)

    c_next = f * c_cur + i * g
    h_next = o * torch.tanh(c_next)

    return h_next, c_next

  def init_hidden(self, batch_size, spatial_size, device=None):
    D, H, W = spatial_size
    if device is None:
        device = next(self.parameters()).device
    h = torch.zeros(batch_size, self.hidden_dim, D, H, W, device=device)
    c = torch.zeros(batch_size, self.hidden_dim, D, H, W, device=device)
    return h, c


class ConvLSTM3D(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size=3):
    """
    Single-layer ConvLSTM over time for 3D volumes.
    """
    super().__init__()
    padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
    self.cell = ConvLSTM3DCell(input_dim, hidden_dim, kernel_size, padding)

  def forward(self, x):
    """
    x: (B, T, C_in, D, H, W)

    Returns:
      outputs: (B, T, C_hidden, D, H, W)  # hidden states for each time step
      (h_last, c_last)
    """
    B, T, C, D, H, W = x.shape
    device = x.device
    h, c = self.cell.init_hidden(B, (D, H, W), device=device)

    outputs = []
    for t in range(T):
      h, c = self.cell(x[:, t], (h, c))  # x[:, t]: (B, C_in, D, H, W)
      outputs.append(h)

    outputs = torch.stack(outputs, dim=1)   # (B, T, C_hidden, D, H, W)
    return outputs, (h, c)
  


# if __name__ == "__main__":
#   # ----- Hyperparameters -----
#   batch_size   = 2
#   seq_len      = 4          # number of PAST timesteps you feed in
#   in_channels  = 1          # e.g., 1 weather variable; make this >1 if needed
#   height, width = 16, 16
#   hidden_dim   = 8
#   future_steps = 1          # <<< CHANGE THIS to predict more future timesteps

#   # ----- Fake dense training data (for demo) -----
#   # x_dense: past sequence, y_future_dense: future ground truth frames
#   # In a real setup, you'd have real weather grids here.
#   x_dense = torch.randn(batch_size, seq_len, in_channels, height, width)
#   y_future_dense = torch.randn(batch_size, future_steps, in_channels, height, width)

#   # ----- Convert dense -> sparse COO to mimic your input format -----
#   # Shape is still conceptually (B, T, C, H, W), but storage is sparse.
#   x_sparse = x_dense.to_sparse_coo()

#   # ----- Model -----
#   convlstm = ConvLSTM(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=3)
#   decoder  = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)  # map hidden -> frame

#   criterion = nn.MSELoss()
#   optimizer = optim.Adam(
#     list(convlstm.parameters()) + list(decoder.parameters()),
#     lr=1e-3,
#   )

#   for epoch in range(10):
#     optimizer.zero_grad()

#     # ============================================
#     # 1) Convert sparse COO -> dense for ConvLSTM
#     # ============================================
#     # Your real code: x_sparse will come from your data loader.
#     x_dense_again = x_sparse.to_dense()  # (B, T, C, H, W), now dense

#     # --------------------------------------------
#     # 2) Encode past sequence with ConvLSTM
#     # --------------------------------------------
#     outputs, (h, c) = convlstm(x_dense_again)  # h: (B, hidden_dim, H, W)

#     # --------------------------------------------
#     # 3) Autoregressive prediction of future steps
#     # --------------------------------------------
#     preds = []
#     # (Optional) you can start from last real frame if you want,
#     # but here we only use hidden state h and c.
#     for t in range(future_steps):
#       # Predict a frame from the current hidden state
#       y_t = decoder(h)        # (B, C_in, H, W)
#       preds.append(y_t)

#       # Feed the predicted frame back into the ConvLSTM cell
#       # to update (h, c) for the next future step.
#       h, c = convlstm.cell(y_t, (h, c))

#     # Stack predictions: (B, future_steps, C_in, H, W)
#     y_pred_seq = torch.stack(preds, dim=1)

#     # For future_steps = 1 this is just (B, 1, C, H, W)
#     loss = criterion(y_pred_seq, y_future_dense)

#     loss.backward()
#     optimizer.step()

#     print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")


import torch
import torch.nn as nn

class ConvLSTM3DCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, padding):
    """
    3D ConvLSTM cell (D, H, W spatial dims).
    """
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
      padding = (padding, padding, padding)

    self.conv = nn.Conv3d(
      in_channels=input_dim + hidden_dim,
      out_channels=4 * hidden_dim,
      kernel_size=kernel_size,
      padding=padding,
      bias=True,
    )

  def forward(self, x, state):
    h_cur, c_cur = state
    combined = torch.cat([x, h_cur], dim=1)
    combined_conv = self.conv(combined)
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
    super().__init__()
    padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
    self.cell = ConvLSTM3DCell(input_dim, hidden_dim, kernel_size, padding)

  def forward(self, x, hidden_state=None):
    """
    x: (B, T, C, D, H, W)
    hidden_state: Tuple (h, c) from previous chunk.
    """
    B, T, C, D, H, W = x.shape
    device = x.device
    
    # SCENARIO 1 & 2 SUPPORT:
    # If hidden_state is None -> Start fresh (Scenario 1 or Start of Scen 2)
    # If hidden_state provided -> Continue memory (Scenario 2)
    if hidden_state is None:
        h, c = self.cell.init_hidden(B, (D, H, W), device=device)
    else:
        h, c = hidden_state

    outputs = []
    for t in range(T):
      h, c = self.cell(x[:, t], (h, c))
      outputs.append(h)
    outputs = torch.stack(outputs, dim=1)
    
    return outputs, (h, c)
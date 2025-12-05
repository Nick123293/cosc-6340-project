import torch
import torch.nn as nn
import torch.nn.functional as F


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

class Conv3dGEMM(nn.Module):
    """
    3D convolution implemented explicitly as:
      - im2col via as_strided
      - matrix multiplication

    Supports:
      - stride=1 (you can generalize if needed)
      - symmetric padding for each spatial dim
      - dilation=1 (can generalize similarly)
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (Kd, Kh, Kw)
        self.padding = padding          # (Pd, Ph, Pw)
        self.stride = (1, 1, 1)
        self.dilation = (1, 1, 1)

        Kd, Kh, Kw = kernel_size

        # Weight: (out_channels, in_channels, Kd, Kh, Kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, Kd, Kh, Kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize similar to Conv3d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * Kd * Kh * Kw
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: (B, C_in, D, H, W)
        returns: (B, C_out, D_out, H_out, W_out)
        """
        B, C_in, D, H, W = x.shape
        Kd, Kh, Kw = self.kernel_size
        Pd, Ph, Pw = self.padding
        Sd, Sh, Sw = self.stride
        Dd, Dh, Dw = self.dilation  # currently all 1

        # ---- 1) Pad input ----
        # F.pad 3D order: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        x_padded = F.pad(x, (Pw, Pw, Ph, Ph, Pd, Pd))

        Bp, Cp, Dp, Hp, Wp = x_padded.shape
        assert Cp == C_in

        # ---- 2) Compute output sizes (same as Conv3d) ----
        D_out = (Dp - Dd*(Kd-1) - 1) // Sd + 1
        H_out = (Hp - Dh*(Kh-1) - 1) // Sh + 1
        W_out = (Wp - Dw*(Kw-1) - 1) // Sw + 1

        # ---- 3) Use as_strided to create a view of all sliding 3D patches ----
        # x_padded strides
        sB, sC, sD, sH, sW = x_padded.stride()

        # We want a view of shape:
        # (B, C_in, D_out, H_out, W_out, Kd, Kh, Kw)
        shape = (B, C_in, D_out, H_out, W_out, Kd, Kh, Kw)
        strides = (
            sB,
            sC,
            sD * Sd,
            sH * Sh,
            sW * Sw,
            sD * Dd,
            sH * Dh,
            sW * Dw,
        )

        patches = x_padded.as_strided(size=shape, stride=strides)
        # patches: (B, C_in, D_out, H_out, W_out, Kd, Kh, Kw)

        # ---- 4) Flatten patches into im2col-style matrix ----
        # Move (D_out, H_out, W_out) to one dim and flatten kernel + channel dims:
        # (B, C_in * Kd * Kh * Kw, D_out * H_out * W_out)
        cols = patches.reshape(
            B,
            C_in * Kd * Kh * Kw,
            D_out * H_out * W_out
        )

        # ---- 5) Reshape weight and perform GEMM ----
        # weight: (C_out, C_in * Kd * Kh * Kw)
        weight_mat = self.weight.view(self.out_channels, -1)
        # We want: (B, L, C_out) = (B, L, K) @ (K, C_out)
        # where L = D_out*H_out*W_out, K = C_in*Kd*Kh*Kw
        cols_T = cols.permute(0, 2, 1)   # (B, L, K)
        out = torch.matmul(cols_T, weight_mat.T)  # (B, L, C_out)

        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)

        # ---- 6) Reshape back to 5D conv output ----
        out = out.permute(0, 2, 1)  # (B, C_out, L)
        out = out.view(B, self.out_channels, D_out, H_out, W_out)
        return out

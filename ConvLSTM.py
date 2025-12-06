import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 2D ConvLSTM
# ============================================================

class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        """
        2D ConvLSTM cell (H, W spatial dims).

        Args:
            input_dim:   number of input channels C_in
            hidden_dim:  number of hidden channels C_hidden
            kernel_size: int or (Kh, Kw)
            padding:     int or (Ph, Pw)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.padding = padding

        # Conv2d over concatenated [x_t, h_{t-1}] along channel dim
        self.conv = Conv2dGEMM(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, state):
        """
        x:      (B, C_in, H, W)
        state:  (h_cur, c_cur), each (B, C_hidden, H, W)
        """
        h_cur, c_cur = state

        # Concatenate along channels
        combined = torch.cat([x, h_cur], dim=1)  # (B, C_in + C_hidden, H, W)
        combined_conv = self.conv(combined)      # (B, 4*C_hidden, H, W)

        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device=None):
        """
        spatial_size: (H, W)
        Returns h, c each of shape (B, C_hidden, H, W)
        """
        H, W = spatial_size
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        """
        Single-layer 2D ConvLSTM over sequences.

        Args:
            input_dim:   channels of input x_t
            hidden_dim:  channels of hidden state h_t
            kernel_size: int or (Kh, Kw)
        """
        super().__init__()

        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            # for e.g. (Kh, Kw) → (Kh//2, Kw//2)
            padding = tuple(k // 2 for k in kernel_size)

        self.cell = ConvLSTM2DCell(input_dim, hidden_dim, kernel_size, padding)

    def forward(self, x, hidden_state=None):
        """
        x: (B, T, C, H, W)
        hidden_state: optional (h, c) from previous chunk
            h, c: (B, C_hidden, H, W)

        Returns:
            outputs: (B, T, C_hidden, H, W)
            (h_T, c_T): final states
        """
        B, T, C, H, W = x.shape
        device = x.device

        # If hidden_state is None -> start fresh
        if hidden_state is None:
            h, c = self.cell.init_hidden(B, (H, W), device=device)
        else:
            h, c = hidden_state

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (B, T, C_hidden, H, W)
        return outputs, (h, c)


# ============================================================
# 2D Conv implemented via GEMM (im2col + matmul)
# ============================================================

class Conv2dGEMM(nn.Module):
    """
    2D convolution implemented explicitly as:
      - im2col via as_strided
      - matrix multiplication

    Supports:
      - stride=1
      - symmetric padding for H and W
      - dilation=1
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # (Kh, Kw)
        self.padding = padding          # (Ph, Pw)
        self.stride = (1, 1)
        self.dilation = (1, 1)

        Kh, Kw = kernel_size

        # Weight: (out_channels, in_channels, Kh, Kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, Kh, Kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize similar to Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * Kh * Kw
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H_out, W_out)
        """
        B, C_in, H, W = x.shape
        Kh, Kw = self.kernel_size
        Ph, Pw = self.padding
        Sh, Sw = self.stride
        Dh, Dw = self.dilation  # currently all 1

        # ---- 1) Pad input ----
        # F.pad order for 4D: (pad_w_left, pad_w_right, pad_h_left, pad_h_right)
        x_padded = F.pad(x, (Pw, Pw, Ph, Ph))

        Bp, Cp, Hp, Wp = x_padded.shape
        assert Cp == C_in

        # ---- 2) Compute output sizes (same as Conv2d) ----
        H_out = (Hp - Dh * (Kh - 1) - 1) // Sh + 1
        W_out = (Wp - Dw * (Kw - 1) - 1) // Sw + 1

        # ---- 3) Use as_strided to create view of all sliding 2D patches ----
        sB, sC, sH, sW = x_padded.stride()

        # Desired view: (B, C_in, H_out, W_out, Kh, Kw)
        shape = (B, C_in, H_out, W_out, Kh, Kw)
        strides = (
            sB,            # step for batch
            sC,            # step for channel
            sH * Sh,       # step when moving in output H
            sW * Sw,       # step when moving in output W
            sH * Dh,       # step within kernel H
            sW * Dw,       # step within kernel W
        )

        patches = x_padded.as_strided(size=shape, stride=strides)
        # patches: (B, C_in, H_out, W_out, Kh, Kw)

        # ---- 4) Flatten patches into im2col-style matrix ----
        # (B, C_in * Kh * Kw, H_out * W_out)
        cols = patches.reshape(
            B,
            C_in * Kh * Kw,
            H_out * W_out
        )

        # ---- 5) Reshape weight and perform GEMM ----
        weight_mat = self.weight.view(self.out_channels, -1)  # (C_out, C_in * Kh * Kw)

        # cols_T: (B, L, K) where L = H_out * W_out, K = C_in * Kh * Kw
        cols_T = cols.permute(0, 2, 1)
        out = torch.matmul(cols_T, weight_mat.T)  # (B, L, C_out)

        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)

        # ---- 6) Reshape back to standard conv output ----
        out = out.permute(0, 2, 1)  # (B, C_out, L)
        out = out.view(B, self.out_channels, H_out, W_out)
        return out

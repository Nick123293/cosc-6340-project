import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from ConvLSTM import ConvLSTM2D


# ---------------------------------------------------------
# 0. Row-size + timestep capacity estimation
# ---------------------------------------------------------

def estimate_row_sizes(csv_path, max_sample_rows=1000):
  """
  Use the first up to `max_sample_rows` data rows (excluding header)
  to estimate per-row size in bytes.

  Prints and returns:
      avg_bytes, min_bytes, max_bytes
  """
  sizes = []
  with open(csv_path, "rb") as f:
    # skip header
    header = f.readline()
    for i, line in enumerate(f):
      if i >= max_sample_rows:
        break
      sizes.append(len(line))

  if not sizes:
    raise ValueError(f"[MEM] No data rows found when sampling CSV: {csv_path}")

  n = len(sizes)
  avg_bytes = float(sum(sizes)) / n
  min_bytes = min(sizes)
  max_bytes = max(sizes)

  print(
    f"[MEM] Row size stats over {n} sampled rows "
    f"(excluding header): avg={avg_bytes:.2f} bytes, "
    f"min={min_bytes} bytes, max={max_bytes} bytes"
  )
  return avg_bytes, min_bytes, max_bytes


def adjust_window_lengths(seq_len_in_arg, future_steps_arg, timestep_capacity):
  """
  Given original seq_len_in and future_steps, and a maximum total number of
  timesteps we can hold (timestep_capacity), choose new (seq_len_in, future_steps)
  such that:

    - seq_len_in + future_steps <= timestep_capacity
    - seq_len_in <= seq_len_in_arg, future_steps <= future_steps_arg
    - seq_len_in >= 1, future_steps >= 1
    - The ratio seq_len_in : future_steps is as close as possible to the
      original ratio, preferring higher total timesteps and preferring
      seq_len_in to be higher when an exact ratio cannot be achieved.

  If timestep_capacity is None, we simply return the original pair.
  """
  if timestep_capacity is None:
    return seq_len_in_arg, future_steps_arg

  # At least 3 timesteps (2 input + 1 prediction), as per your assumption.
  min_total = 3
  max_total = min(timestep_capacity, seq_len_in_arg + future_steps_arg)
  if max_total < min_total:
    max_total = min_total

  if future_steps_arg > 0:
    target_ratio = float(seq_len_in_arg) / float(future_steps_arg)
  else:
    target_ratio = float(seq_len_in_arg)

  best_pair = None
  best_total = -1
  best_ratio_diff = None

  # Search from largest feasible total timesteps downward.
  for total in range(max_total, min_total - 1, -1):
    local_best_pair = None
    local_best_ratio_diff = None

    max_future = min(future_steps_arg, total - 1)
    for new_future in range(1, max_future + 1):
      new_seq = total - new_future
      if new_seq < 1 or new_seq > seq_len_in_arg:
        continue

      ratio = float(new_seq) / float(new_future)
      diff = abs(ratio - target_ratio)
      if (local_best_ratio_diff is None) or (diff < local_best_ratio_diff):
        local_best_ratio_diff = diff
        local_best_pair = (new_seq, new_future)

    if local_best_pair is not None:
      best_pair = local_best_pair
      best_total = total
      best_ratio_diff = local_best_ratio_diff
      # Because we're iterating total from largest down, first valid is max.
      break

  if best_pair is None:
    # Fallback: simple clamping
    new_future = min(future_steps_arg, max_total - 1)
    if new_future < 1:
      new_future = 1
    new_seq = max_total - new_future
    if new_seq < 1:
      new_seq = 1
    best_pair = (new_seq, new_future)

  new_seq_len_in, new_future_steps = best_pair
  return new_seq_len_in, new_future_steps


def analyze_csv_and_memory(
  csv_path,
  seq_len_in_arg,
  future_steps_arg,
  max_ram_bytes=None,
  max_vram_bytes=None,
):
  """
  - Estimates row size (avg/min/max).
  - Streams through the full CSV (in RAM-limited chunks) to estimate:
      * total_rows
      * number of unique timesteps
      * approximate rows_per_timestep
      * approximate CSV bytes per timestep
      * approximate dense tensor bytes per timestep (for VRAM)
  - Computes timestep_capacity from RAM + VRAM limits.
  - Adjusts seq_len_in, future_steps accordingly.

  Returns:
      new_seq_len_in, new_future_steps, timestep_capacity,
      rows_per_chunk (for RAM-based chunking), avg_row_bytes
  """
  avg_bytes, min_bytes, max_bytes = estimate_row_sizes(csv_path)

  # For RAM-limited scanning of the *full* file, pick a chunk size by rows.
  # We keep this approximate: chunk_rows * avg_bytes ≈ max_ram_bytes.
  if max_ram_bytes is not None and avg_bytes > 0:
    rows_per_chunk = max(1, int(max_ram_bytes // avg_bytes))
  else:
    rows_per_chunk = None  # no explicit RAM limit for reading

  # ---- First pass over entire CSV to estimate structure ----
  # We keep memory usage bounded by chunk size for the main CSV data,
  # but we do maintain small sets/dicts for unique times / coords.
  total_rows = 0
  time_values = set()
  x_values = set()
  y_values = set()
  t_min = None
  t_max = None

  # Use only first 3 columns for this scan to reduce per-row RAM footprint.
  scan_chunksize = rows_per_chunk if rows_per_chunk is not None else 100000

  print(f"[MEM] Analyzing CSV structure with chunksize={scan_chunksize} rows...")

  for chunk in pd.read_csv(csv_path, chunksize=scan_chunksize, usecols=[0, 1, 2]):
    total_rows += len(chunk)
    if total_rows == 0:
      continue

    t_col = pd.to_datetime(chunk.iloc[:, 0])
    if not t_col.empty:
      c_min = t_col.min()
      c_max = t_col.max()
      t_min = c_min if t_min is None else min(t_min, c_min)
      t_max = c_max if t_max is None else max(t_max, c_max)
      time_values.update(t_col.unique())

    x_values.update(chunk.iloc[:, 1].astype(float).unique())
    y_values.update(chunk.iloc[:, 2].astype(float).unique())

  if total_rows == 0:
    raise ValueError(f"[DATA] CSV appears empty: {csv_path}")

  n_unique_ts = len(time_values) if time_values else 1
  rows_per_timestep = float(total_rows) / float(n_unique_ts)

  bytes_per_timestep_csv = avg_bytes * rows_per_timestep

  # For dense tensor, approximate C=4 (last 4 columns), B=1, float32=4 bytes.
  C = 4
  H = max(1, len(y_values))
  W = max(1, len(x_values))
  bytes_per_timestep_dense = 4.0 * 1.0 * float(C) * float(H) * float(W)

  print(f"[MEM] Total rows in CSV: {total_rows}")
  print(f"[MEM] Unique timesteps (by timestamp): {n_unique_ts}")
  print(f"[MEM] Approx rows per timestep: {rows_per_timestep:.2f}")
  print(f"[MEM] Approx CSV bytes per timestep: {bytes_per_timestep_csv:.2f}")
  print(f"[MEM] Approx dense bytes per timestep: {bytes_per_timestep_dense:.2f} "
        f"(H={H}, W={W}, C={C})")

  timestep_limits = []

  if max_ram_bytes is not None and bytes_per_timestep_csv > 0:
    t_ram = int(max_ram_bytes // bytes_per_timestep_csv)
    if t_ram < 3:
      t_ram = 3
    timestep_limits.append(t_ram)
    print(f"[MEM] Based on max-ram-bytes={max_ram_bytes}, "
          f"timestep capacity ≈ {t_ram}")

  if max_vram_bytes is not None and bytes_per_timestep_dense > 0:
    t_vram = int(max_vram_bytes // bytes_per_timestep_dense)
    if t_vram < 3:
      t_vram = 3
    timestep_limits.append(t_vram)
    print(f"[MEM] Based on max-vram-bytes={max_vram_bytes}, "
          f"timestep capacity ≈ {t_vram}")

  if timestep_limits:
    timestep_capacity = min(timestep_limits)
    print(f"[MEM] Effective timestep capacity (min over limits) = "
          f"{timestep_capacity}")
  else:
    timestep_capacity = None
    print("[MEM] No memory limits provided; using original "
          "seq_len_in and future_steps.")

  # Adjust seq_len_in and future_steps based on capacity (if any).
  new_seq_len_in, new_future_steps = adjust_window_lengths(
    seq_len_in_arg, future_steps_arg, timestep_capacity if timestep_capacity is not None
    else (seq_len_in_arg + future_steps_arg)
  )

  if (new_seq_len_in, new_future_steps) != (seq_len_in_arg, future_steps_arg):
    print(
      f"[MEM] Adjusting window lengths due to memory limits: "
      f"seq_len_in {seq_len_in_arg} → {new_seq_len_in}, "
      f"future_steps {future_steps_arg} → {new_future_steps}"
    )
  else:
    print("[MEM] Memory limits do not constrain window lengths; "
          "using original seq_len_in and future_steps.")

  # Explicitly print how many timesteps we are able to keep in memory.
  if timestep_capacity is not None:
    print(f"[MEM] Final effective timestep capacity (timesteps in memory) = "
          f"{timestep_capacity}")
  else:
    print("[MEM] Effective timestep capacity is unbounded by given limits; "
          "using full requested window sizes.")

  # For chunked reading during training/validation, use rows_per_chunk from RAM.
  if max_ram_bytes is not None and avg_bytes > 0:
    rows_per_chunk = max(1, int(max_ram_bytes // avg_bytes))
  else:
    rows_per_chunk = None

  if rows_per_chunk is not None:
    print(f"[MEM] Using rows_per_chunk={rows_per_chunk} for RAM-limited "
          f"CSV reading.")
  else:
    print("[MEM] No RAM limit specified for chunking; reading CSV in "
          "a single chunk.")

  return new_seq_len_in, new_future_steps, timestep_capacity, rows_per_chunk, avg_bytes


# ---------------------------------------------------------
# 1. CSV / DF → Dense 2D Tensor
# ---------------------------------------------------------

def load_sparse_df_to_dense_2d(df, device="cpu"):
  """
  Load a sparse DataFrame (time, x, y, 4 channels) and produce a dense tensor:

      dense: (B=1, T_dim, C=4, H=Y_dim, W=X_dim)

  - Time axis:
      * Build full hourly timeline from min(t_raw) to max(t_raw) *within chunk*.
      * Any missing hours in that local range are included and stay all-zero.
  - Spatial axes:
      * Build full set of unique x and y values seen in the chunk.
      * Any missing (t, x, y) combination stays all-zero.
  """

  # ---- 0. Basic sanity check ----
  if df.shape[1] < 5:
    raise ValueError(
      "Chunk DataFrame must have at least 5 columns: time, x, y, and feature columns."
    )

  # ---- 1. Parse columns ----
  t_raw = pd.to_datetime(df.iloc[:, 0])          # timestamps
  x_raw = df.iloc[:, 1].astype(float)
  y_raw = df.iloc[:, 2].astype(float)

  # Take the last 4 columns as the 4 channels (robust to extra cols).
  feats = df.iloc[:, -4:].astype(float).fillna(0.0)
  C = feats.shape[1]

  # ---- 2. Build full hourly timeline (local) ----
  t_min, t_max = t_raw.min(), t_raw.max()
  t_full = pd.date_range(start=t_min, end=t_max, freq="h")
  t_full_values = t_full.values
  nT_full = len(t_full_values)
  T_dim = nT_full

  time_to_index = {ts: i for i, ts in enumerate(t_full_values)}
  t_idx_np = np.fromiter(
    (time_to_index[ts] for ts in t_raw.values),
    dtype=np.int64,
    count=len(t_raw),
  )

  # ---- 3. Spatial indices (within chunk) ----
  x_unique, x_inv = np.unique(x_raw.values, return_inverse=True)
  y_unique, y_inv = np.unique(y_raw.values, return_inverse=True)

  nX, nY = len(x_unique), len(y_unique)
  X_dim = nX
  Y_dim = nY

  B = 1  # batch size

  # ---- 4. Allocate dense tensor ----
  dense = torch.zeros((B, T_dim, C, Y_dim, X_dim),
                      device=device, dtype=torch.float32)

  # ---- 5. Scatter observed values ----
  t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=device)
  x_idx = torch.tensor(x_inv, dtype=torch.long, device=device)
  y_idx = torch.tensor(y_inv, dtype=torch.long, device=device)
  feats_t = torch.tensor(feats.values, dtype=torch.float32, device=device)

  dense[0, t_idx, :, y_idx, x_idx] = feats_t

  # Dummy z_unique to keep signature symmetric with original if needed.
  z_unique = np.array([0], dtype=int)

  return dense, t_full, x_unique, y_unique, z_unique


def load_sparse_csv_to_dense_2d(
  csv_path,
  T=None, X=None, Y=None, Z=None,   # Z kept for API compatibility, ignored
  device="cpu",
):
  """
  Original "full CSV" loader retained for compatibility. This loads the
  entire CSV into memory. For RAM-limited training we instead chunk the
  CSV and call `load_sparse_df_to_dense_2d` on each chunk.
  """
  df = pd.read_csv(csv_path)
  return load_sparse_df_to_dense_2d(df, device=device)


# ---------------------------------------------------------
# 2. COMPRESSION (2D → vector)
# ---------------------------------------------------------

def compress_2d_to_vector(tensor_5d):
  """
  tensor_5d: (B, T, C, H, W)
  Returns:   (B, T, C) = spatial mean over H, W.
  """
  return torch.mean(tensor_5d, dim=[3, 4])


# ---------------------------------------------------------
# 3. TRAINING ON ONE CSV (SLIDING WINDOWS, CHUNKED)
# ---------------------------------------------------------

def train_on_chunk_df(
  df_chunk,
  convlstm,
  decoder,
  criterion,
  optimizer,
  epoch,
  seq_len_in=9,
  future_steps=3,
):
  """
  Train over ALL sliding windows within a single DataFrame chunk.

  This keeps the logic of the original train_single_csv, but restricted
  to the local time range of this chunk.
  """
  device = next(convlstm.parameters()).device
  dense, _, _, _, _ = load_sparse_df_to_dense_2d(df_chunk, device=device)

  B, T_total, C, H, W = dense.shape
  min_required = seq_len_in + future_steps
  if T_total < min_required:
    return None

  max_start = T_total - min_required
  if max_start < 0:
    return None

  total_loss = 0.0
  n_windows = 0

  for start_t in range(max_start + 1):  # inclusive
    end_t = start_t + min_required
    window = dense[:, start_t:end_t]   # (B, seq_len_in+future_steps, C, H, W)

    x_in = window[:, :seq_len_in]      # (B, seq_len_in, C, H, W)
    y_true = window[:, seq_len_in:]    # (B, future_steps, C, H, W)

    optimizer.zero_grad()
    outputs, (h, c) = convlstm(x_in)   # outputs: (B, seq_len_in, hidden_dim, H, W)

    # --- Buffer embeddings (per window, per timestep) ---
    # if layer_buffer is not None:
    #   compressed = compress_2d_to_vector(outputs).detach().cpu().numpy()  # (B, seq_len_in, hidden_dim)
    #
    #   for t_local in range(seq_len_in):
    #     t_global = start_t + t_local
    #     notation_json = json.dumps({
    #       "layer": "ConvLSTM2D",
    #       "operation": "GlobalAvgPooling",
    #       "source_op": "Conv2dGEMM",
    #       "timestep_local": t_local,
    #       "timestep_global": t_global,
    #       "input_shape": [int(H), int(W)],
    #       "math": {"pool": "mean(h_t)", "conv": "matmul(im2col(...))"}
    #     })
    #     layer_buffer.append(
    #       (run_id, epoch, csv_path, t_global, compressed[0, t_local].tolist(), notation_json)
    #     )

    # --- Autoregressive prediction for 'future_steps' ---
    preds = []
    h_t, c_t = h, c
    for _ in range(future_steps):
      y_t = decoder(h_t)          # (B, C_out=4, H, W)
      preds.append(y_t)
      # feed prediction back into ConvLSTM cell
      h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

    preds = torch.stack(preds, dim=1)   # (B, future_steps, C, H, W)
    loss = criterion(preds, y_true)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    n_windows += 1

  if n_windows == 0:
    return None

  return total_loss / n_windows


def train_single_csv(
  csv_path,
  convlstm,
  decoder,
  criterion,
  optimizer,
  epoch,
  seq_len_in=9,
  future_steps=3,
  rows_per_chunk=None,
):
  """
  Train over a single CSV, potentially using RAM-limited chunks.

  - If rows_per_chunk is None: loads entire CSV into memory once.
  - If rows_per_chunk is set: iterates over `pd.read_csv(..., chunksize=rows_per_chunk)`
    and applies `train_on_chunk_df` to each chunk, freeing it before
    loading the next chunk.

  In either case, all rows in the CSV are eventually seen. With chunked
  reading, sliding windows are local to each chunk (no windows that span
  across chunk boundaries).
  """
  device = next(convlstm.parameters()).device

  total_loss = 0.0
  n_chunks = 0

  if rows_per_chunk is None:
    df = pd.read_csv(csv_path)
    loss = train_on_chunk_df(
      df,
      convlstm,
      decoder,
      criterion,
      optimizer,
      epoch,
      seq_len_in=seq_len_in,
      future_steps=future_steps,
    )
    if loss is not None:
      total_loss += loss
      n_chunks += 1
  else:
    for df_chunk in pd.read_csv(csv_path, chunksize=rows_per_chunk):
      loss = train_on_chunk_df(
        df_chunk,
        convlstm,
        decoder,
        criterion,
        optimizer,
        epoch,
        seq_len_in=seq_len_in,
        future_steps=future_steps,
      )
      if loss is not None:
        total_loss += loss
        n_chunks += 1

      # Free chunk and VRAM occupied by its dense tensor before next chunk.
      del df_chunk
      if device == "cuda":
        torch.cuda.empty_cache()

  if n_chunks == 0:
    return None

  return total_loss / n_chunks


# ---------------------------------------------------------
# 4. VALIDATION ON ONE CSV (SLIDING WINDOWS, CHUNKED)
# ---------------------------------------------------------

def validate_on_chunk_df(
  df_chunk,
  convlstm,
  decoder,
  criterion,
  seq_len_in,
  future_steps=3,
):
  """
  Validation over ALL sliding windows within a single DataFrame chunk.
  """
  device = next(convlstm.parameters()).device
  dense, _, _, _, _ = load_sparse_df_to_dense_2d(df_chunk, device=device)

  B, T_total, C, H, W = dense.shape
  min_required = seq_len_in + future_steps
  if T_total < min_required:
    return None

  max_start = T_total - min_required
  if max_start < 0:
    return None

  total_loss = 0.0
  n_windows = 0

  with torch.no_grad():
    for start_t in range(max_start + 1):
      end_t = start_t + min_required
      window = dense[:, start_t:end_t]

      x_in = window[:, :seq_len_in]
      y_true = window[:, seq_len_in:]

      outputs, (h, c) = convlstm(x_in)

      preds = []
      h_t, c_t = h, c
      for _ in range(future_steps):
        y_t = decoder(h_t)
        preds.append(y_t)
        h_t, c_t = convlstm.cell(y_t, (h_t, c_t))

      preds = torch.stack(preds, dim=1)
      loss = criterion(preds, y_true)

      total_loss += loss.item()
      n_windows += 1

  if n_windows == 0:
    return None

  return total_loss / n_windows


def validate_single_csv(
  csv_path,
  convlstm,
  decoder,
  criterion,
  seq_len_in,
  future_steps=3,
  rows_per_chunk=None,
):
  """
  Validation over a single CSV, potentially using RAM-limited chunks.

  Same chunking semantics as train_single_csv, but without optimizer steps.
  """
  device = next(convlstm.parameters()).device

  total_loss = 0.0
  n_chunks = 0

  if rows_per_chunk is None:
    df = pd.read_csv(csv_path)
    loss = validate_on_chunk_df(
      df,
      convlstm,
      decoder,
      criterion,
      seq_len_in,
      future_steps=future_steps,
    )
    if loss is not None:
      total_loss += loss
      n_chunks += 1
  else:
    for df_chunk in pd.read_csv(csv_path, chunksize=rows_per_chunk):
      loss = validate_on_chunk_df(
        df_chunk,
        convlstm,
        decoder,
        criterion,
        seq_len_in,
        future_steps=future_steps,
      )
      if loss is not None:
        total_loss += loss
        n_chunks += 1

      del df_chunk
      if device == "cuda":
        torch.cuda.empty_cache()

  if n_chunks == 0:
    return None

  return total_loss / n_chunks


# ---------------------------------------------------------
# 5. TRAIN LOOP
# ---------------------------------------------------------

def train(
  train_csv_path,
  val_csv_path,
  convlstm,
  decoder,
  num_epochs,
  seq_len_in,
  future_steps,
  rows_per_chunk=None,
):
  device = next(convlstm.parameters()).device

  # Load checkpoint if necessary
  # if load_checkpoint and os.path.exists(checkpoint_path):
  #   ckpt = torch.load(checkpoint_path, map_location=device)
  #   convlstm.load_state_dict(ckpt["convlstm_state_dict"])
  #   decoder.load_state_dict(ckpt["decoder_state_dict"])
  #   print(f"[CKPT] Loaded checkpoint {checkpoint_path}")
  # else:
  print("[CKPT] Starting from scratch.")

  # conn, cursor = init_db(reset_tables=not load_checkpoint)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(
    list(convlstm.parameters()) + list(decoder.parameters()),
    lr=1e-3,
  )

  # Register training run in DB
  # model_config = {
  #   "convlstm": {
  #     "class": convlstm.__class__.__name__,
  #     "input_dim": 4,
  #     "hidden_dim": 4,
  #     "kernel_size": 3,
  #   },
  #   "decoder": {
  #     "class": decoder.__class__.__name__,
  #     "in": 4,
  #     "out": 4,
  #     "kernel": 1,
  #   },
  #   "device": str(device),
  # }
  #
  # optimizer_config = {
  #   "type": "Adam",
  #   "lr": optimizer.param_groups[0]["lr"],
  #   "betas": list(optimizer.param_groups[0].get("betas", (0.9, 0.999))),
  #   "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
  # }

  for epoch in range(num_epochs):
    convlstm.train()
    decoder.train()
    train_losses = []

    # ---- TRAIN ----
    loss = train_single_csv(
      train_csv_path,
      convlstm,
      decoder,
      criterion,
      optimizer,
      epoch,
      seq_len_in=seq_len_in,
      future_steps=future_steps,
      rows_per_chunk=rows_per_chunk,
    )
    if loss is not None:
      train_losses.append(loss)

    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

    # ---- VALIDATION ----
    convlstm.eval()
    decoder.eval()
    val_losses = []

    with torch.no_grad():
      loss = validate_single_csv(
        val_csv_path,
        convlstm,
        decoder,
        criterion,
        seq_len_in,
        future_steps=future_steps,
        rows_per_chunk=rows_per_chunk,
      )
      if loss is not None:
        val_losses.append(loss)

    val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

    print(
      f"[EPOCH {epoch+1}/{num_epochs}] "
      f"train={train_loss:.6f}  val={val_loss:.6f}"
    )

  return train_loss, val_loss


# ---------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-csv", required=True,
                      help="Path to single CSV used for training.")
  parser.add_argument("--val-csv", required=True,
                      help="Path to single CSV used for validation.")
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--future-steps", type=int, default=3)
  parser.add_argument("--seq-len-in", type=int, default=9)
  parser.add_argument("--checkpoint", type=str,
                      default="checkpoints/checkpoint.pth")
  # parser.add_argument("--load-checkpoint", action="store_true")

  # New optional memory-limit arguments (in bytes)
  parser.add_argument(
    "--max-ram-bytes",
    type=int,
    default=None,
    help="Maximum bytes of CSV data to assume can reside in main memory at once."
  )
  parser.add_argument(
    "--max-vram-bytes",
    type=int,
    default=None,
    help="Maximum bytes of dense tensor data to assume can reside on GPU at once."
  )

  start_time = time.time()
  args = parser.parse_args()
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Analyze training CSV + memory limits, adjust window lengths, and determine
  # chunk size for RAM-limited reading.
  (
    effective_seq_len_in,
    effective_future_steps,
    timestep_capacity,
    rows_per_chunk,
    avg_row_bytes,
  ) = analyze_csv_and_memory(
    args.train_csv,
    args.seq_len_in,
    args.future_steps,
    max_ram_bytes=args.max_ram_bytes,
    max_vram_bytes=args.max_vram_bytes,
  )

  convlstm = ConvLSTM2D(input_dim=4, hidden_dim=4, kernel_size=3).to(device)
  # 2D decoder because ConvLSTM2D hidden state is (B, C, H, W)
  decoder = nn.Conv2d(4, 4, kernel_size=1).to(device)

  train_loss, val_loss = train(
    train_csv_path=args.train_csv,
    val_csv_path=args.val_csv,
    convlstm=convlstm,
    decoder=decoder,
    num_epochs=args.epochs,
    future_steps=effective_future_steps,
    seq_len_in=effective_seq_len_in,
    rows_per_chunk=rows_per_chunk,
  )

  torch.save(
    {
      "convlstm_state_dict": convlstm.state_dict(),
      "decoder_state_dict": decoder.state_dict(),
    },
    args.checkpoint,
  )
  total_time = time.time() - start_time
  print("Model saved at: " + args.checkpoint)
  print(f"Total runtime: {total_time:.2f} seconds")
  print(f"Training Loss: {train_loss:.6f}")
  print(f"Validation Loss: {val_loss:.6f}")

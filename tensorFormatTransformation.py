import torch
import pandas as pd

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

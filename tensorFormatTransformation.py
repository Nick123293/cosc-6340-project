import torch
import pandas as pd

def load_sparse_coo_csv_to_dense_3d(csv_path, T=10, X=100, Y=100, Z=100, H=100, device="cpu"):
  """
  SCENARIO 1 LOADER: Loads EVERYTHING at once.
  Use this for small datasets (fits in RAM).
  """
  df = pd.read_csv(csv_path)
  # H parameter was for flattening logic in original snippets, keeping signature same
  dense = torch.zeros(1, T, 1, X, Y, Z, device=device)

  for _, row in df.iterrows():
    t = int(row["t"])
    x = int(row["x"])
    y = int(row["y"])
    z = int(row["z"])
    val = float(row["value"])
    
    # Boundary check to be safe
    if t < T:
        dense[0, t, 0, x, y, z] = val

  return dense 

def load_dense_slice(df, t_start, t_end, X=100, Y=100, Z=100, device="cpu"):
    """
    SCENARIO 2 LOADER: Loads a specific TIME CHUNK.
    Use this for massive datasets (Chunking).
    """
    # 1. Calculate duration of just this small slice
    duration = t_end - t_start
    
    # 2. Allocate memory ONLY for this small duration
    dense = torch.zeros(1, duration, 1, X, Y, Z, device=device)
    
    # 3. Filter the DataFrame (cheap operation)
    mask = (df['t'] >= t_start) & (df['t'] < t_end)
    subset = df.loc[mask]
    
    for _, row in subset.iterrows():
        # 4. Shift the index. 
        # Global Time (e.g., 50) becomes Local Tensor Index (e.g., 0)
        local_t = int(row["t"]) - t_start 
        
        x, y, z = int(row["x"]), int(row["y"]), int(row["z"])
        val = float(row["value"])
        dense[0, local_t, 0, x, y, z] = val
        
    return dense
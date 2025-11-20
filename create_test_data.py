import numpy as np
import pandas as pd
import os  # <--- 1. Import the os module

# Parameters
T = 10
X = Y = Z = 100
num_nonzero = 5000

rng = np.random.default_rng(0)

t_idx = rng.integers(0, T, size=num_nonzero, dtype=int)
x_idx = rng.integers(0, X, size=num_nonzero, dtype=int)
y_idx = rng.integers(0, Y, size=num_nonzero, dtype=int)
z_idx = rng.integers(0, Z, size=num_nonzero, dtype=int)

values = rng.normal(loc=20.0, scale=5.0, size=num_nonzero)

df = pd.DataFrame({
    "t": t_idx,
    "x": x_idx,
    "y": y_idx,
    "z": z_idx,
    "value": values
})

# Define the directory and filename separately
output_dir = "../data"
filename = "weather_sparse_coo_100x100x100_t10.csv"
full_path = os.path.join(output_dir, filename)

# <--- 2. Check if directory exists, if not, create it
os.makedirs(output_dir, exist_ok=True) 

df.to_csv(full_path, index=False)

print(f"File successfully saved to: {os.path.abspath(full_path)}")
import numpy as np
import pandas as pd

# Parameters
T = 10
X = Y = Z = 100
num_nonzero = 5000  # number of non-zero entries for sparsity

rng = np.random.default_rng(0)

t_idx = rng.integers(0, T, size=num_nonzero, dtype=int)
x_idx = rng.integers(0, X, size=num_nonzero, dtype=int)
y_idx = rng.integers(0, Y, size=num_nonzero, dtype=int)
z_idx = rng.integers(0, Z, size=num_nonzero, dtype=int)

values = rng.normal(loc=20.0, scale=5.0, size=num_nonzero)  # e.g., temperatures around 20C

df = pd.DataFrame({
    "t": t_idx,
    "x": x_idx,
    "y": y_idx,
    "z": z_idx,
    "value": values
})

csv_path = "../data/weather_sparse_coo_100x100x100_t10.csv"
df.to_csv(csv_path, index=False)

csv_path

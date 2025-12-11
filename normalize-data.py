#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ------------------------------
# User parameters
# ------------------------------

INPUT_CSV = "data/dense_data.csv"
OUTPUT_CSV = "data/sparse_data.csv"

TIME_DROP_FRAC   = 0.10  # Drop 10% of timesteps
COORD_DROP_FRAC  = 0.60  # Drop 60% of (lat,lon) locations per timestep
VALUE_DROP_FRAC  = 0.60  # Drop 60% of remaining values per variable

RANDOM_SEED = 42

# Expected variable names from your existing CSV
rename_map = {
    "r": "humidity (r)",
    "t": "temperature (t)",
    "u": "u-component of wind (u)",
    "v": "v-component of wind (v)",
}

desired_cols = [
    "time",
    "latitude",
    "longitude",
    "humidity (r)",
    "temperature (t)",
    "u-component of wind (u)",
    "v-component of wind (v)",
]


# ------------------------------
# Running (causal) normalization
# ------------------------------
def running_normalize_time_first(arr, eps=1e-8):
    """
    Causal (running) normalization for arr: shape (T, H, W).
    NaNs stay NaN and are ignored in mean/std.
    """
    T, H, W = arr.shape
    arr = arr.astype(np.float64, copy=False)

    means = np.zeros(T, dtype=np.float64)
    stds  = np.ones(T, dtype=np.float64)

    cum_count = 0.0
    cum_sum   = 0.0
    cum_sumsq = 0.0

    for t in range(T):
        slice_ = arr[t]
        mask = ~np.isnan(slice_)
        if mask.any():
            vals = slice_[mask]
            cum_count += vals.size
            cum_sum   += float(np.sum(vals))
            cum_sumsq += float(np.sum(vals * vals))

        if cum_count > 0:
            mean = cum_sum / cum_count
            var  = cum_sumsq / cum_count - mean * mean
            var = max(var, eps)
            std = np.sqrt(var)
        else:
            mean, std = 0.0, 1.0

        means[t] = mean
        stds[t]  = std

    means_3d = means.reshape(T, 1, 1)
    stds_3d  = stds.reshape(T, 1, 1)

    return (arr - means_3d) / stds_3d, means, stds


# ==========================================================
# 1. LOAD CSV
# ==========================================================
df = pd.read_csv(INPUT_CSV)

# Standardize names if CSV uses r/t/u/v
df = df.rename(columns=rename_map)

for col in desired_cols:
    if col not in df.columns:
        raise ValueError(f"Input CSV missing required column: {col}")

df = df[desired_cols]
df["time"] = pd.to_datetime(df["time"], format="mixed")

# ==========================================================
# 2. Convert to dense 3D arrays (T, H, W)
# ==========================================================

# Sort by time to establish ordering
df = df.sort_values("time")

times_unique = df["time"].unique()
lats_unique  = np.sort(df["latitude"].unique())
lons_unique  = np.sort(df["longitude"].unique())

T = len(times_unique)
H = len(lats_unique)
W = len(lons_unique)

print(f"Loaded CSV: T={T}, H={H}, W={W}")

# Build dense arrays filled with NaN
r_arr = np.full((T, H, W), np.nan)
t_arr = np.full((T, H, W), np.nan)
u_arr = np.full((T, H, W), np.nan)
v_arr = np.full((T, H, W), np.nan)

# Index mapping
time_index = {t: i for i, t in enumerate(times_unique)}
lat_index  = {lat: i for i, lat in enumerate(lats_unique)}
lon_index  = {lon: i for i, lon in enumerate(lons_unique)}

for row in df.itertuples():
    ti = time_index[row.time]
    la = lat_index[row.latitude]
    lo = lon_index[row.longitude]

    r_arr[ti, la, lo] = row._4
    t_arr[ti, la, lo] = row._5
    u_arr[ti, la, lo] = row._6
    v_arr[ti, la, lo] = row._7


# ==========================================================
# 3. DROP TIMESTEPS / COORDS / VALUES
# ==========================================================

rng = np.random.default_rng(RANDOM_SEED)

# A) Drop 10% timesteps
n_drop = int(TIME_DROP_FRAC * T)
keep_mask = np.ones(T, dtype=bool)

if n_drop > 0:
    drop_idx = rng.choice(T, n_drop, replace=False)
    keep_mask[drop_idx] = False

times_keep = times_unique[keep_mask]
r_arr = r_arr[keep_mask]
t_arr = t_arr[keep_mask]
u_arr = u_arr[keep_mask]
v_arr = v_arr[keep_mask]

T_keep = len(times_keep)

# B) Drop 60% of (lat,lon) per timestep
coord_drop = rng.random((T_keep, H, W)) < COORD_DROP_FRAC
r_arr[coord_drop] = np.nan
t_arr[coord_drop] = np.nan
u_arr[coord_drop] = np.nan
v_arr[coord_drop] = np.nan

# C) Drop 60% of remaining values per variable
if VALUE_DROP_FRAC > 0:
    r_arr[rng.random(r_arr.shape) < VALUE_DROP_FRAC] = np.nan
    t_arr[rng.random(t_arr.shape) < VALUE_DROP_FRAC] = np.nan
    u_arr[rng.random(u_arr.shape) < VALUE_DROP_FRAC] = np.nan
    v_arr[rng.random(v_arr.shape) < VALUE_DROP_FRAC] = np.nan


# ==========================================================
# 4. CAUSAL NORMALIZATION
# ==========================================================

r_arr, _, _ = running_normalize_time_first(r_arr)
t_arr, _, _ = running_normalize_time_first(t_arr)
u_arr, _, _ = running_normalize_time_first(u_arr)
v_arr, _, _ = running_normalize_time_first(v_arr)

# ==========================================================
# 5. FLATTEN + DROP ALL-NaN rows
# ==========================================================

# Create coordinate grids
lat_grid, lon_grid = np.meshgrid(lats_unique, lons_unique, indexing="ij")

rows = []
for ti in range(T_keep):
    for la in range(H):
        for lo in range(W):

            vals = [
                r_arr[ti, la, lo],
                t_arr[ti, la, lo],
                u_arr[ti, la, lo],
                v_arr[ti, la, lo],
            ]

            if all(np.isnan(v) for v in vals):
                continue

            rows.append([
                times_keep[ti],
                lats_unique[la],
                lons_unique[lo],
                *vals,
            ])

df_out = pd.DataFrame(rows, columns=desired_cols)

# Replace NaN with empty strings in variables
for col in desired_cols[3:]:
    df_out[col] = df_out[col].astype(object)
    df_out.loc[df_out[col].isna(), col] = ""

# ==========================================================
# 6. SAVE SINGLE CSV
# ==========================================================

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved output CSV: {OUTPUT_CSV} ({len(df_out)} rows)")

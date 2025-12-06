#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# User parameters
# ------------------------------
INPUT_GRIB = "data/data.grib"
OUTPUT_DIR = "data/chunks_sparse_csv"

BLOCK_LAT = 8   # latitude chunk size
BLOCK_LON = 8   # longitude chunk size

TIME_DROP_FRAC   = 0.10  # Drop 10% of timesteps
COORD_DROP_FRAC  = 0.60  # Drop 60% of (lat,lon) locations per timestep
VALUE_DROP_FRAC  = 0.60  # Drop 60% of remaining values per variable

RANDOM_SEED = 42         # For reproducibility
MAX_WORKERS = 4          # Start with 1–2; you can try 4 later if RAM looks good
# ------------------------------

# 1. Open dataset from GRIB
ds = xr.open_dataset(INPUT_GRIB, engine="cfgrib")

# Only keep the variables we care about
vars_to_save = ["r", "t", "u", "v"]
ds = ds[vars_to_save]  # coords (time, latitude, longitude, isobaricInhPa, etc.) still attached

# 2. Prepare output directory
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(exist_ok=True)

# 3. Get dimension sizes
n_time = ds.sizes["time"]
n_lat = ds.sizes["latitude"]
n_lon = ds.sizes["longitude"]

print(f"time: {n_time}, latitude: {n_lat}, longitude: {n_lon}")

# We no longer care about isobaricInhPa at all.
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


def running_normalize_time_first(arr, eps=1e-8):
    """
    Causal (running) normalization over time for a 3D array:

        arr: shape (T, H, W) with NaNs for missing values.

    For each time t:
      - compute mean/std using ALL valid entries from arr[0..t, :, :]
      - normalize arr[t] using those μ_t, σ_t

    NaNs stay NaN.
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
            if var < eps:
                var = eps
            std = np.sqrt(var)
        else:
            # No data yet → arbitrary but stable defaults
            mean = 0.0
            std  = 1.0

        means[t] = mean
        stds[t]  = std

    # Broadcast means/stds over spatial dims
    means_3d = means.reshape(T, 1, 1)
    stds_3d  = stds.reshape(T, 1, 1)

    normed = (arr - means_3d) / stds_3d

    return normed, means, stds


def process_chunk(lat_start, lat_end, lon_start, lon_end):
    """
    Low-memory processing of a single (lat, lon) block.

    Steps:
      - slice xarray dataset in lat/lon
      - pull only the needed data as NumPy arrays
      - randomly drop 10% of timesteps
      - randomly drop 60% of (lat,lon) locations per remaining timestep
      - randomly drop 60% of remaining values per variable
      - perform causal (running) normalization per variable over time
      - drop all-NaN rows
      - write a CSV with time/lat/lon + 4 normalized variables

    Deleted values are kept as missing internally (NaN),
    but written as EMPTY cells in the CSV (not the literal 'NaN' string).
    """

    # Make a reproducible RNG for this chunk
    local_seed = RANDOM_SEED + lat_start * 1000 + lon_start
    rng = np.random.default_rng(local_seed)

    # Slice coords (we'll work in NumPy from here)
    times = ds["time"].values          # shape (n_time,)
    lats_block = ds["latitude"].values[lat_start:lat_end]   # shape (block_lat,)
    lons_block = ds["longitude"].values[lon_start:lon_end]  # shape (block_lon,)

    if lats_block.size == 0 or lons_block.size == 0:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: empty lat/lon, skipped."

    # Load each variable for this block as a NumPy array: (time, block_lat, block_lon)
    # This is the main memory footprint for this chunk.
    try:
        r_data = ds["r"].isel(latitude=slice(lat_start, lat_end),
                              longitude=slice(lon_start, lon_end)).values
        t_data = ds["t"].isel(latitude=slice(lat_start, lat_end),
                              longitude=slice(lon_start, lon_end)).values
        u_data = ds["u"].isel(latitude=slice(lat_start, lat_end),
                              longitude=slice(lon_start, lon_end)).values
        v_data = ds["v"].isel(latitude=slice(lat_start, lat_end),
                              longitude=slice(lon_start, lon_end)).values
    except KeyError:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: missing vars, skipped."

    # Sanity: all have same shape
    T, BLat, BLon = r_data.shape

    # ------------------------------
    # A) Drop 10% of timesteps
    # ------------------------------
    n_times = T
    n_drop = int(np.floor(TIME_DROP_FRAC * n_times))
    keep_time_mask = np.ones(n_times, dtype=bool)

    if n_drop > 0 and n_times > 0:
        drop_idx = rng.choice(n_times, size=n_drop, replace=False)
        keep_time_mask[drop_idx] = False

    if not keep_time_mask.any():
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: all timesteps dropped, skipped."

    # Apply time mask
    times_keep = times[keep_time_mask]
    r_data = r_data[keep_time_mask, :, :]
    t_data = t_data[keep_time_mask, :, :]
    u_data = u_data[keep_time_mask, :, :]
    v_data = v_data[keep_time_mask, :, :]

    T_keep = times_keep.shape[0]

    # ------------------------------
    # B) Drop 60% of (lat,lon) points per remaining timestep
    # ------------------------------
    # coord_mask_drop[t, i, j] = True -> drop this coordinate (set all vars NaN)
    coord_rand = rng.random((T_keep, BLat, BLon))
    coord_drop_mask = coord_rand < COORD_DROP_FRAC

    r_data[coord_drop_mask] = np.nan
    t_data[coord_drop_mask] = np.nan
    u_data[coord_drop_mask] = np.nan
    v_data[coord_drop_mask] = np.nan

    # ------------------------------
    # C) Drop 60% of remaining values per variable (independent masks)
    # ------------------------------
    if VALUE_DROP_FRAC > 0.0:
        r_mask = rng.random(r_data.shape) < VALUE_DROP_FRAC
        t_mask = rng.random(t_data.shape) < VALUE_DROP_FRAC
        u_mask = rng.random(u_data.shape) < VALUE_DROP_FRAC
        v_mask = rng.random(v_data.shape) < VALUE_DROP_FRAC

        r_data[r_mask] = np.nan
        t_data[t_mask] = np.nan
        u_data[u_mask] = np.nan
        v_data[v_mask] = np.nan

    # ------------------------------
    # D) Causal (running) normalization per variable over time
    #     NaNs remain NaN and are ignored when computing stats.
    # ------------------------------
    r_data, r_means, r_stds = running_normalize_time_first(r_data)
    t_data, t_means, t_stds = running_normalize_time_first(t_data)
    u_data, u_means, u_stds = running_normalize_time_first(u_data)
    v_data, v_means, v_stds = running_normalize_time_first(v_data)

    # ------------------------------
    # E) Flatten to 1D arrays and drop all-NaN rows
    # ------------------------------
    # Stack variables into one array: (T_keep, BLat, BLon, 4)
    feats_stack = np.stack([r_data, t_data, u_data, v_data], axis=-1)
    # Flatten to (N, 4)
    feats_flat = feats_stack.reshape(-1, 4)

    # Build coordinate grids
    # time_flat: repeat each time for all lat*lon
    time_flat = np.repeat(times_keep, BLat * BLon)

    # lat/lon grids:
    lat_grid, lon_grid = np.meshgrid(lats_block, lons_block, indexing="ij")
    lat_flat = np.tile(lat_grid.reshape(-1), T_keep)
    lon_flat = np.tile(lon_grid.reshape(-1), T_keep)

    # Mask: keep rows where not all 4 vars are NaN
    valid_mask = ~np.isnan(feats_flat).all(axis=1)

    if not valid_mask.any():
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: all rows NaN, skipped."

    time_flat = time_flat[valid_mask]
    lat_flat = lat_flat[valid_mask]
    lon_flat = lon_flat[valid_mask]
    feats_flat = feats_flat[valid_mask, :]

    # ------------------------------
    # F) Build DataFrame (small)
    # ------------------------------
    df = pd.DataFrame({
        "time": pd.to_datetime(time_flat),
        "latitude": lat_flat,
        "longitude": lon_flat,
        "humidity (r)": feats_flat[:, 0],
        "temperature (t)": feats_flat[:, 1],
        "u-component of wind (u)": feats_flat[:, 2],
        "v-component of wind (v)": feats_flat[:, 3],
    })

    # Enforce column order & presence
    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[desired_cols]

    if df.empty:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: empty after filtering, skipped."

    # ------------------------------
    # G) Convert NaNs in channel columns to empty cells for CSV
    #     (they remain "missing" logically; your loader will map blanks → NaN → 0)
    # ------------------------------
    channel_cols = [
        "humidity (r)",
        "temperature (t)",
        "u-component of wind (u)",
        "v-component of wind (v)",
    ]
    for col in channel_cols:
        df[col] = df[col].astype(object)
        df.loc[df[col].isna(), col] = ""  # empty string in CSV

    # ------------------------------
    # H) Save CSV
    # ------------------------------
    fname = out_dir / (
        f"chunk_lat{lat_start:03d}-{lat_end-1:03d}"
        f"_lon{lon_start:03d}-{lon_end-1:03d}.csv"
    )
    df.to_csv(fname, index=False)

    return f"Saved {fname} with {len(df)} rows"


# Build list of all chunks
tasks = []
for lat_start in range(0, n_lat, BLOCK_LAT):
    lat_end = min(lat_start + BLOCK_LAT, n_lat)
    for lon_start in range(0, n_lon, BLOCK_LON):
        lon_end = min(lon_start + BLOCK_LON, n_lon)
        tasks.append((lat_start, lat_end, lon_start, lon_end))

print(f"Processing {len(tasks)} chunks with up to {MAX_WORKERS} workers...")

# Run in parallel with *small* worker count
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_chunk, *args): args for args in tasks}
    for fut in as_completed(futures):
        msg = fut.result()
        print(msg)

print("Done.")

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

TIME_DROP_FRAC = 0.10  # Drop 10% of timesteps
VALUE_DROP_FRAC = 0.30 # Drop 30% of values in each variable

RANDOM_SEED = 42       # For reproducibility
MAX_WORKERS = 16        # Adjust to number of CPU cores
# ------------------------------

# 1. Open dataset from GRIB
ds = xr.open_dataset(INPUT_GRIB, engine="cfgrib")

# Variables you care about (short names in the GRIB)
vars_to_save = ["r", "t", "u", "v"]
ds = ds[vars_to_save]  # coordinates (time, lat, lon, isobaricInhPa, etc.) are preserved

# 2. Prepare output directory
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(exist_ok=True)

# 3. Get dimension sizes
n_time = ds.sizes["time"]
n_lat = ds.sizes["latitude"]
n_lon = ds.sizes["longitude"]

print(f"time: {n_time}, latitude: {n_lat}, longitude: {n_lon}")

# Shared rename + columns
rename_map = {
    "isobaricInhPa": "isobaricInhPa",
    "r": "humidity (r)",
    "t": "temperature (t)",
    "u": "u-component of wind (u)",
    "v": "v-component of wind (v)",
}
desired_cols = [
    "time",
    "latitude",
    "longitude",
    "isobaricInhPa",
    "humidity (r)",
    "temperature (t)",
    "u-component of wind (u)",
    "v-component of wind (v)",
]

def process_chunk(lat_start, lat_end, lon_start, lon_end):
    """
    Process a single (lat, lon) block:
      - subset xarray dataset
      - convert to DataFrame
      - drop 10% timesteps
      - drop 30% of values per variable
      - drop all-NaN rows
      - rename + reorder columns
      - save CSV
    This function is meant to run in parallel threads.
    """

    # Make a reproducible RNG for this chunk (deterministic given coordinates)
    local_seed = RANDOM_SEED + lat_start * 1000 + lon_start
    rng = np.random.default_rng(local_seed)

    # Slice the chunk: (time, lat_block, lon_block)
    ds_chunk = ds.isel(
        latitude=slice(lat_start, lat_end),
        longitude=slice(lon_start, lon_end)
    )

    # Convert to DataFrame: one row per (time, lat, lon)
    df = ds_chunk.to_dataframe().reset_index()

    if df.empty:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: empty, skipped."

    # ------------------------------
    # A) Drop 10% of timesteps (uniform over time)
    # ------------------------------
    unique_times = df["time"].unique()
    n_times = len(unique_times)
    n_drop = int(np.floor(TIME_DROP_FRAC * n_times))

    if n_drop > 0 and n_times > 0:
        times_to_drop = rng.choice(unique_times, size=n_drop, replace=False)
        df = df[~df["time"].isin(times_to_drop)]

    if df.empty:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: all rows dropped after time filtering."

    # ------------------------------
    # B) Randomly delete 30% of values per variable
    # ------------------------------
    for var in vars_to_save:
        if var in df.columns:
            mask = rng.random(len(df)) < VALUE_DROP_FRAC
            df.loc[mask, var] = np.nan

    # ------------------------------
    # C) Drop any row where *all* variables are NaN
    # ------------------------------
    df = df.dropna(subset=vars_to_save, how="all")

    if df.empty:
        return f"Chunk lat{lat_start}-{lat_end-1} lon{lon_start}-{lon_end-1}: all rows NaN, skipped."

    # ------------------------------
    # D) Rename and select columns in your desired format
    # ------------------------------
    df = df.rename(columns=rename_map)

    for col in desired_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[desired_cols]

    # ------------------------------
    # E) Save chunk to CSV
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

# Run in parallel
print(f"Processing {len(tasks)} chunks in parallel with {MAX_WORKERS} workers...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_chunk, *args): args for args in tasks
    }
    for fut in as_completed(futures):
        msg = fut.result()
        print(msg)

print("Done.")

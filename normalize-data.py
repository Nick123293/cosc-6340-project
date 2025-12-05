#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


def normalize_local_and_get_stats(df, num_last_cols=4):
    """
    Sparse normalization:
      - Compute column means/stds ignoring NaN
      - Normalize only observed entries
      - Leave missing values as NaN (remain sparse)
    """
    target = df.iloc[:, -num_last_cols:].apply(pd.to_numeric, errors="coerce")

    # Compute stats ignoring NaNs
    means = target.mean(axis=0)
    stds = target.std(axis=0, ddof=0)

    # Handle all-NaN or zero-std cases
    means = means.fillna(0.0)
    stds = stds.fillna(0.0)
    
    # Avoid division by zero → use NaN so normalization result stays NaN where needed
    stds_safe = stds.replace(0, np.nan)

    # Normalize only real values
    normalized = (target - means) / stds_safe

    # DO NOT modify NaNs — leave them as NaN
    # DO NOT .fillna() at all

    df.iloc[:, -num_last_cols:] = normalized

    return df, means.to_list(), stds.to_list(), list(target.columns)



def process_file(csv_path, output_dir, num_last_cols=4):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {
            "file": csv_path.name,
            "status": f"[ERROR] Could not read {csv_path}: {e}",
            "means": None,
            "stds": None,
            "columns": None,
        }

    if df.shape[1] < num_last_cols:
        return {
            "file": csv_path.name,
            "status": f"[SKIP] not enough columns",
            "means": None,
            "stds": None,
            "columns": None,
        }

    df_norm, means, stds, cols = normalize_local_and_get_stats(
        df, num_last_cols=num_last_cols
    )

    out_path = output_dir / csv_path.name
    try:
        df_norm.to_csv(out_path, index=False)
    except Exception as e:
        return {
            "file": csv_path.name,
            "status": f"[ERROR] Could not write {out_path}: {e}",
            "means": None,
            "stds": None,
            "columns": None,
        }

    return {
        "file": csv_path.name,
        "status": f"[OK] {csv_path.name} → {out_path.name}",
        "means": means,
        "stds": stds,
        "columns": cols,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Locally z-score normalize the last N columns of sparse CSV files, preserving NaN sparsity."
    )
    parser.add_argument("input_dir", help="Directory containing CSVs")
    parser.add_argument("output_dir", help="Where to save sparse-normalized CSVs")
    parser.add_argument("--num-last-cols", type=int, default=4)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--metadata-out", default="normalization_metadata.json")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV files")
    print("Performing sparse normalization (NaNs preserved)\n")

    metadata = {}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_file, f, output_dir, args.num_last_cols): f
            for f in csv_files
        }

        for fut in as_completed(futures):
            result = fut.result()
            print(result["status"])
            if result["means"] is not None:
                metadata[result["file"]] = {
                    "means": result["means"],
                    "stds": result["stds"],
                    "columns": result["columns"],
                }

    with open(args.metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata for {len(metadata)} files → {args.metadata_out}")


if __name__ == "__main__":
    main()

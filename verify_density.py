#!/usr/bin/env python3
import pandas as pd

# -----------------------------
# User parameters
# -----------------------------
INPUT_CSV = "data/dense_data.csv"
TIME_COL  = "time"
LAT_COL   = "latitude"
LON_COL   = "longitude"

# How many missing (time, lat, lon) combos to print as examples
MAX_MISSING_SAMPLES = 20


def main():
    print(f"Loading CSV: {INPUT_CSV}")
    # Read without forcing dtypes; let pandas infer numeric for lat/lon
    df = pd.read_csv(INPUT_CSV)

    # ---- Basic sanity checks on columns ----
    for col in [TIME_COL, LAT_COL, LON_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ---- Parse time column as datetime (robust to date + datetime) ----
    print("Parsing time column as datetime...")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="raise")

    # ---- Drop exact duplicate (time, lat, lon) rows and report ----
    before = len(df)
    df_unique = df.drop_duplicates(subset=[TIME_COL, LAT_COL, LON_COL])
    duplicates = before - len(df_unique)
    if duplicates > 0:
        print(f"Found and ignored {duplicates} duplicate (time, lat, lon) rows.")
    else:
        print("No duplicate (time, lat, lon) rows found.")
    df = df_unique

    # ---- Sort by time ----
    df = df.sort_values(TIME_COL)

    # ----------------------------------------------------------
    # 1) Check time-step completeness (contiguous time axis)
    # ----------------------------------------------------------
    # Force times into a DatetimeIndex so difference() works consistently
    times = pd.DatetimeIndex(df[TIME_COL].drop_duplicates().sort_values().values)
    print(f"Number of unique time steps: {len(times)}")

    if len(times) < 2:
        print("Not enough time steps to infer frequency.")
    else:
        # Convert to Series before diff() so .mode() is available
        diffs = pd.Series(times).diff().dropna()

        freq_mode = diffs.mode()
        if freq_mode.empty:
            print("Could not infer a dominant time-step difference.")
            full_time_range = None
        else:
            inferred_freq = freq_mode.iloc[0]
            print(f"Inferred dominant time-step: {inferred_freq}")

            # Build the expected full time range from min to max
            full_time_range = pd.date_range(
                start=times[0],
                end=times[-1],
                freq=inferred_freq
            )

            # Correct difference operations (Index vs Index)
            missing_times = full_time_range.difference(times)
            extra_times   = times.difference(full_time_range)

            print("\n--- Time-step completeness check ---")
            print(f"Expected number of steps: {len(full_time_range)}")
            print(f"Actual number of steps:   {len(times)}")
            print(f"Missing time steps:       {len(missing_times)}")
            print(f"Extra time steps:         {len(extra_times)}")

            if len(missing_times) > 0:
                print("Example missing time steps (up to 20):")
                for ts in missing_times[:20]:
                    print(f"  {ts}")

            if len(extra_times) > 0:
                print("Example extra time steps (up to 20):")
                for ts in extra_times[:20]:
                    print(f"  {ts}")

    # ----------------------------------------------------------
    # 2) Check full lat/lon coverage for each time step
    # ----------------------------------------------------------
    print("\nChecking lat/lon grid completeness at each time step...")

    lats = df[LAT_COL].drop_duplicates().sort_values()
    lons = df[LON_COL].drop_duplicates().sort_values()
    n_lats = len(lats)
    n_lons = len(lons)

    print(f"Number of unique latitudes:  {n_lats}")
    print(f"Number of unique longitudes: {n_lons}")

    # Use the same unique times (ensure DatetimeIndex)
    times = pd.DatetimeIndex(df[TIME_COL].drop_duplicates().sort_values().values)

    # Expected number of (time, lat, lon) combos if full dense grid
    expected_total = len(times) * n_lats * n_lons
    actual_total = len(df)

    print(f"\n--- Global (time, lat, lon) coverage ---")
    print(f"Expected rows if full dense grid: {expected_total}")
    print(f"Actual rows:                      {actual_total}")
    print(f"Missing (time, lat, lon) combinations (global): {expected_total - actual_total}")

    # Build full index for (time, lat, lon) and compare with existing
    print("Building full grid index (this may take a bit for very large datasets)...")
    full_index = pd.MultiIndex.from_product(
        [times, lats.values, lons.values],
        names=[TIME_COL, LAT_COL, LON_COL]
    )

    actual_index = pd.MultiIndex.from_arrays(
        [df[TIME_COL].values, df[LAT_COL].values, df[LON_COL].values],
        names=[TIME_COL, LAT_COL, LON_COL]
    )

    missing_index = full_index.difference(actual_index)
    missing_count = len(missing_index)

    print(f"\n--- Detailed grid completeness ---")
    print(f"Total missing (time, lat, lon) points: {missing_count}")

    if missing_count > 0:
        # Count how many time steps are incomplete
        missing_times_for_grid = missing_index.get_level_values(TIME_COL).unique()
        print(f"Number of time steps with incomplete lat/lon coverage: {len(missing_times_for_grid)}")

        print(f"\nExample missing (time, lat, lon) entries (up to {MAX_MISSING_SAMPLES}):")
        for t, lat, lon in list(missing_index)[:MAX_MISSING_SAMPLES]:
            print(f"  time={t}, lat={lat}, lon={lon}")
    else:
        print("All (time, lat, lon) combinations present. The grid is complete.")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def split_csv_by_months(input_csv: str, dt_format: str = None) -> None:
    """
    Split a CSV whose first column is a datetime into:
      - months 1–10 (Jan–Oct)
      - month 11 (Nov)
      - month 12 (Dec)

    Parameters
    ----------
    input_csv : str
        Path to the input CSV.
    dt_format : str, optional
        Explicit datetime format for parsing, e.g. '%Y-%m-%d %H-%M-%S'.
        If None, pandas will infer the format.
    """

    # Read the CSV
    df = pd.read_csv(input_csv)

    # Assume first column is the datetime column
    time_col = df.columns[0]

    # Parse the datetime column
    if dt_format is not None:
        df[time_col] = pd.to_datetime(df[time_col], format=dt_format)
    else:
        df[time_col] = pd.to_datetime(df[time_col])

    # Extract month
    months = df[time_col].dt.month

    # Split into three dataframes
    df_months_1_10 = df[months.between(1, 10)]
    df_month_11 = df[months == 11]
    df_month_12 = df[months == 12]

    # Build output filenames
    out_1_10 = "data/training_data.csv"
    out_11 = "data/validation_data.csv"
    out_12 = "data/testing_data.csv"

    # Write to CSV
    df_months_1_10.to_csv(out_1_10, index=False)
    df_month_11.to_csv(out_11, index=False)
    df_month_12.to_csv(out_12, index=False)

    print(f"Wrote:")
    print(f"  Months  1–10 -> {out_1_10}")
    print(f"  Month      11 -> {out_11}")
    print(f"  Month      12 -> {out_12}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV with datetime in the first column into:\n"
            "  - months 1–10 (Jan–Oct)\n"
            "  - month 11 (Nov)\n"
            "  - month 12 (Dec)"
        )
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--dt-format",
        default=None,
        help=(
            "Optional explicit datetime format, e.g. '%Y-%m-%d %H-%M-%S'. "
            "If omitted, pandas will infer the format."
        ),
    )

    args = parser.parse_args()
    split_csv_by_months(args.input_csv, args.dt_format)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def copy_time_window(
    input_csv: str,
    start_str: str = None,
    end_str: str = None,
    dt_format: str = None,
    output_csv: str = None,
) -> None:
    """
    Filter a CSV by a datetime window (inclusive) on its first column.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV.
    start_str : str, optional
        Start of the window as a datetime string (inclusive).
        If None, the window starts from the earliest timestamp.
    end_str : str, optional
        End of the window as a datetime string (inclusive).
        If None, the window ends at the latest timestamp.
    dt_format : str, optional
        Explicit datetime format for parsing, e.g. '%Y-%m-%d %H-%M-%S'.
        If None, pandas will infer the format.
    output_csv : str, optional
        Path to the output CSV. If None, defaults to
        '<input_basename>_window.csv' in the same directory.
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

    # Build start and end timestamps
    if start_str is not None:
        if dt_format is not None:
            start_ts = pd.to_datetime(start_str, format=dt_format)
        else:
            start_ts = pd.to_datetime(start_str)
    else:
        start_ts = df[time_col].min()

    if end_str is not None:
        if dt_format is not None:
            end_ts = pd.to_datetime(end_str, format=dt_format)
        else:
            end_ts = pd.to_datetime(end_str)
    else:
        end_ts = df[time_col].max()

    # Filter by inclusive window
    mask = (df[time_col] >= start_ts) & (df[time_col] <= end_ts)
    df_window = df[mask]

    # Determine output filename if not provided
    if output_csv is None:
        base, ext = os.path.splitext(os.path.basename(input_csv))
        output_csv = os.path.join(
            os.path.dirname(input_csv),
            f"{base}_window{ext or '.csv'}"
        )

    # Write filtered data
    df_window.to_csv(output_csv, index=False)

    print("Time window selection complete.")
    print(f"Input file:  {input_csv}")
    print(f"Time column: {time_col}")
    print(f"Start time:  {start_ts}")
    print(f"End time:    {end_ts}")
    print(f"Rows kept:   {len(df_window)}")
    print(f"Output file: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy rows from a CSV whose first column is a datetime, "
            "keeping only those within a specified time window."
        )
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start of the time window (inclusive). "
            "If omitted, uses earliest timestamp in the file."
        ),
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=(
            "End of the time window (inclusive). "
            "If omitted, uses latest timestamp in the file."
        ),
    )
    parser.add_argument(
        "--dt-format",
        type=str,
        default=None,
        help=(
            "Explicit datetime format for parsing, e.g. '%Y-%m-%d %H-%M-%S'. "
            "If omitted, pandas will infer the format."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output CSV path. If omitted, uses '<input_basename>_window.csv' "
            "in the same directory."
        ),
    )

    args = parser.parse_args()

    copy_time_window(
        input_csv=args.input_csv,
        start_str=args.start,
        end_str=args.end,
        dt_format=args.dt_format,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

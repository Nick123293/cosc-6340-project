#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def process_file(input_path: Path, output_path: Path, delimiter: str = ","):
    # 1-based columns: 1,2,3,8,9,10,11
    # -> 0-based indices:
    keep_indices = [0, 1, 2, 7, 8, 9, 10]

    with input_path.open("r", newline="") as fin, \
         output_path.open("w", newline="") as fout:

        reader = csv.reader(fin, delimiter=delimiter)
        writer = csv.writer(fout, delimiter=delimiter)

        for row in reader:
            # Guard in case of malformed rows
            new_row = [row[i] for i in keep_indices if i < len(row)]
            writer.writerow(new_row)


def main():
    parser = argparse.ArgumentParser(
        description="Drop columns 4–7 (1-based) from an 11-column CSV."
    )
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("output", help="Path to output CSV")
    parser.add_argument(
        "--delimiter", "-d",
        default=",",
        help="Field delimiter (default: ',')"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    process_file(input_path, output_path, delimiter=args.delimiter)


if __name__ == "__main__":
    main()

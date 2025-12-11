#!/usr/bin/env python3
import pandas as pd
import re

INPUT_CSV  = "data/stripped_data.csv"
OUTPUT_CSV = "data/dense_data.csv"

# Regex for strict date-only format: YYYY-MM-DD
DATE_ONLY_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

df = pd.read_csv(INPUT_CSV, dtype=str)  # keep everything as strings

def fix_time(value):
    """Convert YYYY-MM-DD → YYYY-MM-DD 00:00:00, else leave unchanged."""
    if value is None:
        return value
    v = str(value).strip()
    if DATE_ONLY_REGEX.match(v):
        return v + " 00:00:00"
    return v

# Apply the fix
df["time"] = df["time"].apply(fix_time)

# Save cleaned CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved cleaned CSV to: {OUTPUT_CSV}")
print("Done.")

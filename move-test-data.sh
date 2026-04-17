#!/usr/bin/env bash

# -------------------------------
# User parameters
# -------------------------------
SRC_DIR="data/chunks_sparse_csv"          # directory containing your 273 datasets
DEST_DIR="data/chunk_sparse_csv_testing"  # directory to copy 50 random datasets into
NUM_FILES=50

# -------------------------------
# Prep destination directory
# -------------------------------
mkdir -p "$DEST_DIR"

# -------------------------------
# Randomly select files
# -------------------------------
# Shuffles file list, selects 50, and copies each to destination
shuf -e "$SRC_DIR"/* | head -n $NUM_FILES | while read -r file; do
    mv "$file" "$DEST_DIR"/
done

echo "✔ Done! Moved $NUM_FILES random files to $DEST_DIR."

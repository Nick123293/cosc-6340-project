#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-}"

if [[ -z "$DIR" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Ensure directory exists
if [[ ! -d "$DIR" ]]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Process files
for f in "$DIR"/*; do
    # Skip if no files match (e.g. literal "*")
    [[ -e "$f" ]] || continue

    basename_f="$(basename "$f")"
    newname="$basename_f"

    # Replace 1G → 128M
    newname="${newname//1G/128M}"

    # Replace 512MB → 32M
    newname="${newname//256M/32M}"

    # Only rename if something changed
    if [[ "$newname" != "$basename_f" ]]; then
        echo "Renaming: $basename_f → $newname"
        mv "$DIR/$basename_f" "$DIR/$newname"
    fi
done

echo "Done."

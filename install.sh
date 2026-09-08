#!/usr/bin/env bash

set -euo pipefail

PYTHON="${PYTHON:-/usr/local/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python executable not found at $PYTHON" >&2
    echo "Set PYTHON=/path/to/python and run this script again." >&2
    exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Dependencies installed successfully."
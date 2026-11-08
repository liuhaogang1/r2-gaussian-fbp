#!/usr/bin/env bash
# Default: repair H12/H15/H20; supply all paths explicitly (run_shape_grid.md).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$SCRIPT_DIR/run_shape_grid.py" "$@"

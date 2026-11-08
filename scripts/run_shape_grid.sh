#!/usr/bin/env bash
# Default: nine auto filter pairs + one strong trial; see run_shape_grid.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$SCRIPT_DIR/run_shape_grid.py" "$@"

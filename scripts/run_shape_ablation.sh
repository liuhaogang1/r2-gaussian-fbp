#!/usr/bin/env bash
# Required paths, thresholds and weights are supplied on the command line.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -u "$SCRIPT_DIR/run_shape_ablation.py" "$@"

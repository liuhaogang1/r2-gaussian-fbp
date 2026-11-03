#!/usr/bin/env bash
# Activate the server Conda environment first. All arguments go to the runner.
# The runner continues independent stages after failures and exits nonzero only
# after printing failed/skipped stages and writing summary.json/tsv + report.txt.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
cd -- "$PROJECT_ROOT"
exec "${PYTHON:-python}" -u scripts/server_parallel_validation.py "$@"

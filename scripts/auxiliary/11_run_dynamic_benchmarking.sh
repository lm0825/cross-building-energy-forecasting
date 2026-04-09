#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment3_dynamic_benchmarking" \
  src/experiment3.py \
  --cpu-fraction "$CPU_FRACTION"

echo "[done] auxiliary dynamic benchmarking finished"

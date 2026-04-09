#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment5_heew_pair_enumeration" \
  src/experiment5_heew_pair_enumeration.py \
  --device "$HEEW_DEVICE" \
  --cpu-fraction "$CPU_FRACTION"

run_logged \
  "experiment5_heew_lag_ablation" \
  src/experiment5_heew_lag_ablation.py \
  --max-cpu-threads 4

echo "[done] auxiliary HEEW deep checks finished"

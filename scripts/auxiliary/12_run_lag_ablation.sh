#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment6_lag_ablation" \
  src/experiment6_lag_ablation.py \
  --cpu-fraction "$CPU_FRACTION"

echo "[done] auxiliary lag ablation finished"

#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment2_grouping" \
  src/experiment2.py \
  --models lgbm lgbm_lag lstm patchtst \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

echo "[done] auxiliary grouping analysis finished"

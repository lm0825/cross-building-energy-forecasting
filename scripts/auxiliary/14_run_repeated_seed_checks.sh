#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "repeated_main_metrics" \
  src/repeated_main_metrics.py \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

run_logged \
  "repeated_exp2_metrics" \
  src/repeated_exp2_metrics.py \
  --models lgbm lgbm_lag lstm patchtst \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

run_logged \
  "repeated_exp3_sensitivity" \
  src/repeated_exp3_sensitivity.py \
  --cpu-fraction "$CPU_FRACTION"

echo "[done] auxiliary repeated-seed checks finished"

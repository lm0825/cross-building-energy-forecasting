#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment5_heew" \
  src/experiment5_heew.py \
  --models lgbm lgbm_lag lstm patchtst \
  --device "$HEEW_DEVICE" \
  --cpu-fraction "$CPU_FRACTION"

run_logged \
  "experiment7_information_budget" \
  src/experiment7_information_budget.py \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

run_logged \
  "experiment8_strict_cold_start" \
  src/experiment8_strict_cold_start.py \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

echo "[done] extended paper analyses finished"

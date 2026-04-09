#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "experiment1_main" \
  src/experiment1.py \
  --models lgbm lgbm_lag lstm patchtst \
  --splits t_split b_split s_split \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

run_logged \
  "exp1_supplementary_baselines" \
  src/exp1_supplementary_baselines.py \
  --cpu-fraction "$CPU_FRACTION"

run_logged \
  "experiment4_gepiii" \
  src/experiment4_gepiii.py \
  --models lgbm lgbm_lag lstm patchtst \
  --cpu-fraction "$CPU_FRACTION" \
  --device "$DEVICE"

echo "[done] main paper results finished"

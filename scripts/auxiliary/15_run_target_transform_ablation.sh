#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged \
  "ablation_target_transforms_bdg2" \
  src/ablation_target_transforms.py \
  --dataset bdg2 \
  --models lgbm lstm patchtst \
  --splits t_split b_split \
  --target-transforms minmax log1p \
  --device "$DEVICE" \
  --cpu-fraction "$CPU_FRACTION"

run_logged \
  "ablation_target_transforms_gepiii" \
  src/ablation_target_transforms.py \
  --dataset gepiii \
  --models lgbm lstm patchtst \
  --splits t_split b_split \
  --target-transforms minmax log1p \
  --device "$DEVICE" \
  --cpu-fraction "$CPU_FRACTION"

echo "[done] auxiliary target-transform ablation finished"

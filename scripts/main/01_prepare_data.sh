#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged "stage1_bdg2" src/stage1_bdg2.py
run_logged "stage1_gepiii" src/stage1_gepiii.py

echo "[done] preprocessing finished"

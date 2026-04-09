#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python
cd "$REPO_ROOT"

run_logged "render_paper_figures" src/render_paper_figures.py

echo "[done] figure rendering finished"

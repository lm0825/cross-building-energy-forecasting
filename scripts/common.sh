#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CPU_FRACTION="${CPU_FRACTION:-0.7}"
DEVICE="${DEVICE:-cuda:0}"
HEEW_DEVICE="${HEEW_DEVICE:-cpu}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

ensure_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Run 'uv sync' first, or set PYTHON_BIN explicitly." >&2
    exit 1
  fi
}

timestamp() {
  date '+%Y%m%d_%H%M%S'
}

run_logged() {
  local name="$1"
  shift
  mkdir -p "$LOG_DIR"
  local log_file="$LOG_DIR/${name}_$(timestamp).log"
  echo "[run] $name"
  echo "[log] $log_file"
  "$PYTHON_BIN" -u "$@" 2>&1 | tee "$log_file"
}

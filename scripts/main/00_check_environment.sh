#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

ensure_python

echo "[python] $PYTHON_BIN"
"$PYTHON_BIN" -V

"$PYTHON_BIN" - <<'PY'
import importlib

packages = [
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "scikit_learn",
    "scipy",
    "torch",
]

for package in packages:
    module_name = "sklearn" if package == "scikit_learn" else package
    importlib.import_module(module_name)

print("Environment check passed.")
PY

echo "[status] environment looks ready"

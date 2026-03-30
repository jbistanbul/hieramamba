#!/bin/bash
set -euo pipefail

echo "Installing HieraMamba..."

if [ ! -d ".git" ]; then
    echo "Error: run this script from the repository root."
    exit 1
fi

if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "Error: Python is not installed."
    exit 1
fi

PYTHON_CMD="python"
if ! command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python3"
fi

if [ -f .gitmodules ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

echo "Installing Python dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt

echo "Building 1D NMS extension..."
(
    cd libs/nms
    $PYTHON_CMD setup_nms.py build_ext --inplace
)

echo "Running import sanity checks..."
$PYTHON_CMD - <<'PY'
import sys
import torch
sys.path.append('hydra')
import libs.nms.nms
print('NMS import OK')
if torch.cuda.is_available():
    from hydra.modules.hydra import Hydra
    print('Hydra import OK')
else:
    print('CUDA unavailable: skipped Hydra runtime import check')
PY

echo "Installation complete!"

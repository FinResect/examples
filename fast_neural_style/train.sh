#!/usr/bin/env bash
# ==============================================================================
# fast_neural_style 通用模型训练包装脚本
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/fin-resect/miniconda3/envs/yolo/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || echo "python")"
fi

exec "$PYTHON_BIN" train.py "$@"

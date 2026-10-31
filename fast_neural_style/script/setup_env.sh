#!/usr/bin/env bash
# ==============================================================================
# fast_neural_style 训练与模型量化转换环境一键配置脚本
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv-convert"

echo "============================================================"
echo "    Fast Neural Style: Ti60 神经网络训练与量化转换环境配置"
echo "============================================================"

# 1. 查找基础 Python 环境（优先寻找支持 CUDA 的 PyTorch 环境）
BASE_PYTHON=""
if [[ -x "/home/fin-resect/miniconda3/envs/yolo/bin/python" ]]; then
    BASE_PYTHON="/home/fin-resect/miniconda3/envs/yolo/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
fi

if [[ -z "$BASE_PYTHON" ]]; then
    echo "❌ 错误: 未找到可用的 Python 解释器" >&2
    exit 1
fi

echo "==> [1/3] 检查训练运行环境 (PyTorch + CUDA)..."
echo "    使用解释器: $BASE_PYTHON"

TORCH_STATUS="$("$BASE_PYTHON" -c "
try:
    import torch, torchvision
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else 'CPU only'
    print(f'PyTorch {torch.__version__} | CUDA: {cuda} ({device})')
except Exception as e:
    print(f'ERROR: {e}')
")"

if [[ "$TORCH_STATUS" == ERROR* ]]; then
    echo "    ⚠️ PyTorch 未就绪: $TORCH_STATUS"
    echo "    正在安装基础 requirements.txt..."
    "$BASE_PYTHON" -m pip install -r "$REPO_DIR/requirements.txt"
else
    echo "    ✅ $TORCH_STATUS"
fi

# 2. 检查或构建专用于 ONNX -> TFLite 转换的隔离环境
echo "==> [2/3] 配置 ONNX/TensorFlow 量化转换环境 (.venv-convert)..."
if [[ ! -d "$VENV_DIR" ]]; then
    echo "    创建独立量化转换虚拟环境 (继承系统/Conda基础包): $VENV_DIR"
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

CONVERT_PYTHON="$VENV_DIR/bin/python"

# 检查转换核心库
NEED_INSTALL=0
"$CONVERT_PYTHON" -c "import onnx, onnxruntime, tensorflow, onnx2tf, onnx_graphsurgeon, tflite, flatbuffers" >/dev/null 2>&1 || NEED_INSTALL=1

if [[ $NEED_INSTALL -eq 1 ]]; then
    echo "    正在安装量化转换依赖包..."
    UV_BIN="/home/fin-resect/miniconda3/envs/yolo/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then UV_BIN="$(command -v uv || echo "")"; fi

    PKGS=(
        "onnx>=1.16,<1.18"
        "onnxruntime>=1.18,<1.21"
        "tensorflow>=2.16,<2.19"
        "tf-keras>=2.16,<2.19"
        "onnx2tf>=1.26,<1.27"
        "onnxsim>=0.4.33,<0.5"
        "sng4onnx>=1.0.1,<2"
        "onnx-graphsurgeon"
    )

    if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
        "$UV_BIN" pip install --python "$CONVERT_PYTHON" "${PKGS[@]}"
    else
        "$VENV_DIR/bin/pip" install "${PKGS[@]}"
    fi
fi

# 3. 验证转换环境
echo "==> [3/3] 校验模型转换链路工具..."
"$CONVERT_PYTHON" -c "
import onnx, tensorflow as tf, onnx2tf, tflite, flatbuffers
print('    ✅ 核心依赖校验通过:')
print(f'       - ONNX: {onnx.__version__}')
print(f'       - TensorFlow: {tf.__version__}')
print(f'       - TFLite: {tflite.__version__}')
"

echo "============================================================"
echo "🎉 环境配置完成！"
echo "  - 模型训练: $BASE_PYTHON neural_style/neural_style.py train ..."
echo "  - 一键导出: $CONVERT_PYTHON export_rain_model.py ..."
echo "============================================================"

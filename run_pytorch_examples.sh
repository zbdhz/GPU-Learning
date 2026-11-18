#!/usr/bin/env bash
# 一键运行《图解入门Pytorch》文档对应的测试用例
# 用法：./run_pytorch_examples.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定位包含 torch 的 python 解释器
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in \
        /home/gtx/miniforge3/envs/gpu-dev/bin/python \
        /home/gtx/miniconda3/envs/gpu-dev/bin/python \
        python3; do
        if command -v "$cand" >/dev/null 2>&1 \
           && "$cand" -c "import torch" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi

if [ -z "$PY" ]; then
    echo "错误：未找到可用的 Python（需安装 torch）。可用 PYTHON=xxx 指定解释器。" >&2
    exit 1
fi

echo "使用解释器: $PY"
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())"
echo "----------------------------------------"

"$PY" "$SCRIPT_DIR/examples/test_pytorch_tutorial.py" -v

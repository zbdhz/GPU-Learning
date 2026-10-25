#!/usr/bin/env bash
# 先激活已配置好的 GPU 开发环境。
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
: "${CUDA_HOME:?请先激活 GPU 开发环境，参见 doc/test-environment.md}"
: "${NCCL_HOME:?请先配置 NCCL_HOME，参见 doc/test-environment.md}"
cuda_arch="${CUDA_ARCH:-sm_75}"
mkdir -p build
python verify_env.py
nvcc -O2 -lineinfo -arch="$cuda_arch" examples/vector_add.cu -o build/vector_add
./build/vector_add
python examples/pytorch_cuda_op.py
torchrun --standalone --nproc_per_node=2 examples/nccl_collectives.py
nvcc -O2 -arch="$cuda_arch" examples/nccl_allreduce.cu -I"$NCCL_HOME/include" \
  -L"$NCCL_HOME/lib" -Xlinker -rpath -Xlinker "$NCCL_HOME/lib" -lnccl -o build/nccl_allreduce
./build/nccl_allreduce
echo 'All environment checks: PASS'

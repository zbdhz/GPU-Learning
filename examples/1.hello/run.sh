#!/bin/bash
# ============================================================================
# 1.hello 一键编译运行脚本
# 用法: ./run.sh          —— 编译并运行
#       ./run.sh clean    —— 清理编译产物
# 依赖: CUDA Toolkit（自动探测 /usr/local/cuda*/bin/nvcc）
# ============================================================================
set -e  # 任何一步失败立即退出

# 切换到脚本所在目录，保证从任意路径调用都能找到源文件
cd "$(dirname "$0")"

# ---------- 1. 定位 nvcc 编译器 ----------
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
    for d in /usr/local/cuda*/bin/nvcc; do
        [ -x "$d" ] && NVCC="$d" && break
    done
fi
if [ -z "$NVCC" ]; then
    echo "[错误] 未找到 nvcc，请先安装 CUDA Toolkit"; exit 1
fi
echo "使用编译器: $NVCC ($($NVCC --version | grep -o 'release [0-9.]*' | head -1))"

# ---------- 2. 自动探测本机 GPU 架构（sm_XX），探测失败则用 sm_75 兜底 ----------
ARCH="sm_75"
if command -v nvidia-smi >/dev/null 2>&1; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    [ -n "$CC" ] && ARCH="sm_${CC}"
fi
echo "目标架构  : $ARCH"

# ---------- 3. 编译 ----------
# -O2           开启优化
# -std=c++17    使用 C++17 标准
# --ptxas-options=-v 顺便打印 kernel 的寄存器/栈用量（学习期观察资源占用）
mkdir -p build
echo "---------------------------------------- 编译 ----------------------------------------"
"$NVCC" -O2 -std=c++17 -arch="$ARCH" --ptxas-options=-v \
        hello_cuda.cu -o build/hello_cuda

# ---------- 4. 运行 ----------
echo "---------------------------------------- 运行 ----------------------------------------"
./build/hello_cuda

# ---------- 5. 可选清理 ----------
if [ "$1" = "clean" ]; then
    rm -rf build
    echo "已清理 build/ 目录"
fi

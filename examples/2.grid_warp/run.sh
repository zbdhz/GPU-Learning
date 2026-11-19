#!/bin/bash
# ============================================================================
# 2.grid_warp 一键编译运行脚本（CMake 版）
# 用法: ./run.sh          —— 配置 + 编译 + 运行
#       ./run.sh clean    —— 清理构建目录
#       ./run.sh rebuild  —— 完全重新配置构建
#
# CMake 的典型工作方式（本脚本自动完成这三步）：
#   1) cmake -B build          配置：分析 CMakeLists.txt，生成 build/Makefile
#   2) cmake --build build     构建：执行 Makefile，调用 nvcc 编译
#   3) ./build/grid_warp_demo      运行产物
# ============================================================================
set -e  # 任何一步失败立即退出

# 切换到脚本所在目录，保证从任意路径调用都能找到 CMakeLists.txt
cd "$(dirname "$0")"

# 本机环境的 LD_LIBRARY_PATH 被部分软件指向旧版 libstdc++，会让 cmake 启动失败，
# 这里清掉该变量再启动 cmake（对构建无影响，链接路径由 CMake 自己管理）
unset LD_LIBRARY_PATH

# ---------- 0. 清理模式 ----------
if [ "$1" = "clean" ]; then
    rm -rf build
    echo "已清理 build/ 目录"
    exit 0
fi

# ---------- 1. 定位工具 ----------
command -v cmake  >/dev/null || { echo "[错误] 未找到 cmake";  exit 1; }

NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && for d in /usr/local/cuda*/bin/nvcc; do
    [ -x "$d" ] && NVCC="$d" && break
done
[ -z "$NVCC" ] && { echo "[错误] 未找到 nvcc，请先安装 CUDA Toolkit"; exit 1; }

# 告诉 CMake 去哪里找 nvcc（CMAKE_CUDA_COMPILER 环境变量形式）
export CUDACXX="$NVCC"

# rebuild 模式：删掉 build 目录从头配置（CMake 会缓存配置结果，
# 改了 CMakeLists.txt 后如果行为诡异，rebuild 是第一选择）
[ "$1" = "rebuild" ] && rm -rf build

# ---------- 2. 配置 + 构建 ----------
echo "---------------------------------------- 配置 ----------------------------------------"
cmake -B build -S .

echo "---------------------------------------- 构建 ----------------------------------------"
cmake --build build -j"$(nproc)"

# ---------- 3. 运行 ----------
echo "---------------------------------------- 运行 ----------------------------------------"
./build/grid_warp_demo

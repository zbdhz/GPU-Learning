# ============================================================================
# 3.vector_add — CMakeLists.txt（与 2.grid_warp 同构，讲解见 1.hello）
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(3_vector_add LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动探测本机 GPU 架构（不能加 OUTPUT_QUIET，否则 OUTPUT_VARIABLE 为空）
execute_process(
    COMMAND /usr/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_CAP_RAW
    RESULT_VARIABLE GPU_CAP_RC
    ERROR_VARIABLE GPU_CAP_ERR
)
if(GPU_CAP_RC EQUAL 0 AND GPU_CAP_RAW)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP ${GPU_CAP_RAW})
endif()
if(GPU_COMPUTE_CAP)
    string(REPLACE "." "" GPU_ARCH_NUM ${GPU_COMPUTE_CAP})
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_NUM})
    message(STATUS "检测到 GPU 计算能力 ${GPU_COMPUTE_CAP} -> 架构 ${GPU_ARCH_NUM}")
else()
    set(CMAKE_CUDA_ARCHITECTURES 75)
    message(STATUS "未检测到 GPU，使用默认架构 75")
endif()

add_executable(vector_add vector_add.cu)

target_compile_options(vector_add PRIVATE
    -O3                       # 教程第 6 节用的 -O3（性能基准要开满优化）
    --ptxas-options=-v
    -lineinfo
)

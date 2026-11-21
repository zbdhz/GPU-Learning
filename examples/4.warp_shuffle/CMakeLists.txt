# ============================================================================
# 4.warp_shuffle — CMakeLists.txt（与 3.vector_add 同构）
# ============================================================================
cmake_minimum_required(VERSION 3.18)
project(4_warp_shuffle LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 自动探测本机 GPU 架构
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

add_executable(warp_shuffle_demo warp_shuffle_demo.cu)

target_compile_options(warp_shuffle_demo PRIVATE
    -O3
    --ptxas-options=-v
    -lineinfo
)

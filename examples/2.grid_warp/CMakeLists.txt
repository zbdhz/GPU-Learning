# ============================================================================
# 2.grid_warp — CMakeLists.txt
# 与 1.hello 相比只多了两处新知识点：
#   1) CMAKE_CUDA_FLAGS 追加 -lineinfo（生成行号信息，nsys/ncu 分析用）
#   2) 同样的 CMAKE_CUDA_ARCHITECTURES 自动探测逻辑
# 具体逐行讲解见 ../1.hello/CMakeLists.txt
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(2_grid_warp LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 检测本机 GPU 架构（踩坑：execute_process 不能加 OUTPUT_QUIET，否则拿空串）
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

add_executable(grid_warp_demo grid_warp_demo.cu)

target_compile_options(grid_warp_demo PRIVATE
    -O2
    --ptxas-options=-v
    -lineinfo          # 带源码行号编译，供 ncu/nsys 性能分析定位热点行
)

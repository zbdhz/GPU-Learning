# ============================================================================
# 1.hello — CMakeLists.txt（CUDA 学习版，带逐行注释）
#
# CMake 核心思想：不直接写编译命令，而是"声明式"地描述项目：
#   项目叫什么、要生成哪个可执行文件、由哪些源文件组成、用什么编译选项。
#   CMake 负责生成具体的构建系统（Makefile），后者再调用 nvcc 真正编译。
#
# 构建流程（三步，run.sh 已自动完成）：
#   cmake -B build        配置：分析本文件，在 build/ 生成 Makefile
#   cmake --build build   构建：执行 Makefile，调用 nvcc 编译
#   ./build/hello_cuda    运行
# ============================================================================

# 声明本工程需要的 CMake 最低版本，低于此版本直接报错（防止语法不兼容）
cmake_minimum_required(VERSION 3.18)

# project(<项目名> <语言...>)：定义项目与使用的语言
# LANGUAGES CUDA CXX 让 CMake 自动寻找 nvcc（CUDA 编译器）和 C++ 编译器
project(1_hello LANGUAGES CUDA CXX)

# set(变量 值)：定义变量，后续用 ${变量名} 引用
# CMAKE_CUDA_STANDARD 相当于 nvcc 的 -std=c++17（CUDA 12.x 最低要求 C++17）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)   # 编译器不支持时报错而非静默降级

# ----------------------------------------------------------------------------
# 检测 GPU 架构：在 CMake 里执行外部命令 nvidia-smi
#
# execute_process 各参数：
#   COMMAND            要执行的命令（这里是 nvidia-smi 的绝对路径）
#   OUTPUT_VARIABLE    命令 stdout 存入此变量
#   RESULT_VARIABLE    命令退出码存入此变量（0 = 成功）
#   ERROR_VARIABLE     命令 stderr 存入此变量（无 GPU 时的报错就在这里）
#
# 踩坑记录（CMake 3.22 实测）：
#   不要加 OUTPUT_QUIET —— 加了它 OUTPUT_VARIABLE 不会被回填，拿到空串
#
# 本机装了 4 张卡，输出是多行 "7.5\n7.5\n..."，所以下面用正则只取第一行
# ----------------------------------------------------------------------------
execute_process(
    COMMAND /usr/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_CAP_RAW
    RESULT_VARIABLE GPU_CAP_RC
    ERROR_VARIABLE GPU_CAP_ERR
)

if(GPU_CAP_RC EQUAL 0 AND GPU_CAP_RAW)
    # string(REGEX MATCH <正则> <输出变量> <输入>)：从字符串中提取第一个匹配
    # ^[0-9]+\\.[0-9]+ 匹配行首的 "7.5" 这样的版本号（只取第一张卡）
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" GPU_COMPUTE_CAP ${GPU_CAP_RAW})
endif()

if(GPU_COMPUTE_CAP)
    # string(REPLACE 被替换 替换为 输出 输入)："7.5" -> "75"
    string(REPLACE "." "" GPU_ARCH_NUM ${GPU_COMPUTE_CAP})
    # CMAKE_CUDA_ARCHITECTURES 是 CMake 管理架构的"正规"变量：
    # 设为 75 后 CMake 自动给 nvcc 传 -gencode arch=compute_75,code=sm_75
    # 【注意】不要再手动加 -arch=sm_xx，否则会和这个机制叠加导致双份编译
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_NUM})
    message(STATUS "检测到 GPU 计算能力 ${GPU_COMPUTE_CAP} -> CMAKE_CUDA_ARCHITECTURES=${GPU_ARCH_NUM}")
else()
    # 兜底架构：无 nvidia-smi 时用 sm_75（Turing，向下兼容大部分卡）
    set(CMAKE_CUDA_ARCHITECTURES 75)
    message(STATUS "未检测到 GPU（${GPU_CAP_ERR}），使用默认架构 75")
endif()

# ----------------------------------------------------------------------------
# add_executable(<目标名> <源文件...>)：声明要生成的可执行文件
# CMake 看到 .cu 后缀会自动用 nvcc 编译（因为 project() 声明了 CUDA 语言）
# ----------------------------------------------------------------------------
add_executable(hello_cuda hello_cuda.cu)

# ----------------------------------------------------------------------------
# target_compile_options(<目标> PRIVATE <选项...>)：只给 hello_cuda 追加选项
# PRIVATE = 这些选项仅用于编译本目标自身，不传播给依赖它的目标
#   -O2                 开启 O2 优化
#   --ptxas-options=-v  打印 kernel 的寄存器/栈用量（学习期观察资源占用）
# （架构选项交给上面的 CMAKE_CUDA_ARCHITECTURES，不在这里重复指定）
# ----------------------------------------------------------------------------
target_compile_options(hello_cuda PRIVATE
    -O2
    --ptxas-options=-v
)

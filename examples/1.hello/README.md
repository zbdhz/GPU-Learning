# 1.hello — 第一个 CUDA 程序

> 教程对应：AIInfraGuide《[1.1 CUDA 开发环境搭建](file:///home/gtx/data/AIInfraGuide/docs/guides/模块二-CUDA编程与算子优化/1.1-CUDA开发环境搭建.md)》第 7 节 "Hello World：第一个 CUDA 程序"

## 这个例子做什么

1. 打印 GPU 设备信息（SM 数量 / 计算能力 / 显存）
2. 用 `<<<2, 4>>>` 启动 8 个 GPU 线程，每线程打印自己的 `threadIdx` 和 `blockIdx`
3. 演示**全局线程编号**的标准计算式：`blockIdx.x * blockDim.x + threadIdx.x`
4. 演示完整错误检查流程：`cudaGetLastError()` + `cudaDeviceSynchronize()`

## 一键运行

```bash
cd examples/1.hello
./run.sh           # CMake 配置 + 构建 + 运行
./run.sh rebuild   # 删掉 build/ 完全重新配置（改了 CMakeLists 后用）
./run.sh clean     # 清理构建产物
```

脚本内部就是 CMake 的标准三步曲，理解了它就理解了 CMake 工作流：

```bash
cmake -B build          # ① 配置：分析 CMakeLists.txt，生成 build/Makefile
cmake --build build -j8 # ② 构建：执行 Makefile，调用 nvcc 编译
./build/hello_cuda      # ③ 运行
```

## CMakeLists.txt 讲解（配套学习）

本例 [CMakeLists.txt](file:///home/gtx/dev/GPU-Learning/examples/1.hello/CMakeLists.txt) 只用了 5 个最核心的指令，每个都有详细注释：

| 指令 | 作用 | 类比手写 nvcc |
|---|---|---|
| `cmake_minimum_required(VERSION 3.18)` | 声明最低 CMake 版本 | — |
| `project(1_hello LANGUAGES CUDA CXX)` | 项目名 + 启用 CUDA 语言（自动找 nvcc） | 找 nvcc 路径 |
| `set(CMAKE_CUDA_STANDARD 17)` | CUDA 代码用 C++17 标准 | `-std=c++17` |
| `add_executable(hello_cuda hello_cuda.cu)` | 声明目标和源文件 | 编译哪个文件 |
| `target_compile_options(... PRIVATE -O2)` | 给目标追加编译选项 | `-O2` 等 |

**三个新手易踩的坑（都已在文件注释中标注）**：

1. **架构要用 `CMAKE_CUDA_ARCHITECTURES`，不要手写 `-arch`**。CMake 有自己的架构管理机制（设为 `75` 会自动转成 `-gencode arch=compute_75,code=sm_75`）；如果再手动加 `-arch=sm_75`，两套机制叠加，同一个 kernel 会被编译两遍（曾在本机实际复现：`sm_52` + `sm_75` 双份编译）
2. **CMake 3.22 的 `execute_process` 加 `OUTPUT_QUIET` 时 `OUTPUT_VARIABLE` 拿到空串**。想在配置期探测 `nvidia-smi`，就不加 QUIET、改用 `RESULT_VARIABLE` 判断成功
3. **多卡机器 `nvidia-smi` 输出多行**。用 `string(REGEX MATCH "^[0-9]+\\.[0-9]+" ...)` 只取第一行

**为什么值得学 CMake**（而不是继续写 nvcc 命令）：

- **递归可扩展**：再复杂的项目（多目标、多目录）都是同一套指令，2.vector_add 起后续示例全用 CMake
- **自动管理架构/标准**：`CMAKE_CUDA_ARCHITECTURES` 一行搞定多架构分发（如 `80-real;90-virtual` 可同时出 SASS+PTX）
- **IDE/CI 友好**：VSCode、CLion、Jenkins 都原生识别 CMake 项目

## 预期输出

```
===== 1. 设备信息 =====
设备名称       : NVIDIA GeForce RTX 4090
SM 数量       : 128
计算能力      : 8.9
显存容量      : 24564 MB
每 Block 最大线程数: 1024

===== 2. Hello Kernel（<<<2, 4>>>：8 个线程）=====
Hello from GPU thread 0 in block 0!
Hello from GPU thread 1 in block 0!
Hello from GPU thread 2 in block 0!
Hello from GPU thread 3 in block 0!
Hello from GPU thread 0 in block 1!      <- 注意：两个 Block 的
Hello from GPU thread 1 in block 1!         输出顺序不保证！
Hello from GPU thread 2 in block 1!
Hello from GPU thread 3 in block 1!

===== 3. 全局线程编号（同一 Grid：编号 0~7）=====
  block 0, thread 0 -> 全局编号 0
  ...
  block 1, thread 3 -> 全局编号 7

Done!
```

## 核心知识点

### 1. 函数修饰符（谁能调用谁）

| 修饰符 | 在哪执行 | 谁能调用 | 典型用途 |
|---|---|---|---|
| `__global__` | GPU | CPU（或动态并行时 GPU） | kernel 入口，返回值必须 void |
| `__device__` | GPU | 只能被 GPU 侧代码调用 | kernel 的辅助工具函数 |
| `__host__` | CPU | CPU | 普通 C++ 函数（默认，可省略） |

### 2. 启动配置 `<<<Grid, Block>>>`

```
helloKernel<<<2, 4>>>();
              │  └── 每个 Block 4 个线程（blockDim.x = 4）
              └────── Grid 含 2 个 Block（gridDim.x = 2）
```

总线程数 = Grid × Block = 8。上限：每 Block ≤ 1024 线程（见设备信息输出）。

### 3. 线程索引：每个线程"我是谁"

| 变量 | 含义 | 本例取值 |
|---|---|---|
| `threadIdx.x` | 线程在 Block 内编号 | 0~3 |
| `blockIdx.x` | Block 在 Grid 内编号 | 0~1 |
| `blockDim.x` | 每 Block 线程数 | 4 |
| `gridDim.x` | Grid 内 Block 数 | 2 |

**全局编号**（"一线程一数据"的标准写法）：

```cuda
int gid = blockIdx.x * blockDim.x + threadIdx.x;  // 0~7 唯一
```

### 4. 异步与同步

- kernel 启动是**异步**的：CPU 提交任务后立即往下跑，不等 GPU
- `cudaDeviceSynchronize()`：CPU 阻塞等 GPU 做完——没有它，`main` 可能在打印前就退出
- GPU 的 `printf` 输出会缓冲到同步点才刷到终端

### 5. 错误检查（工程必备习惯）

```cuda
helloKernel<<<2, 4>>>();
cudaError_t err = cudaGetLastError();   // 捕获启动阶段的配置错误
if (err != cudaSuccess) { /* 处理 */ }
cudaDeviceSynchronize();                // 捕获执行阶段的运行时错误
```

几乎所有 CUDA API 都返回 `cudaError_t`，教程第 10 节的排查表就靠这些错误码定位问题。

## 编译选项说明（见 CMakeLists.txt）

| CMake 写法 | 对应 nvcc 选项 | 作用 |
|---|---|---|
| `set(CMAKE_CUDA_ARCHITECTURES 75)` | `-gencode arch=compute_75,code=sm_75` | 目标 GPU 架构（自动探测本机） |
| `set(CMAKE_CUDA_STANDARD 17)` | `-std=c++17` | C++ 标准 |
| `target_compile_options(... -O2)` | `-O2` | 开启优化 |
| `--ptxas-options=-v` | `--ptxas-options=-v` | 打印 kernel 寄存器/共享内存占用 |

## 常见问题

- **输出顺序和预期不一样？** 正常。Block 间并行，先后由硬件调度器决定（教程 7.2 节关键点）。
- **`nvcc: command not found`？** 见教程第 4 节，把 `/usr/local/cuda/bin` 加入 PATH；本脚本也会自动探测全路径。
- **`no CUDA-capable device`？** 驱动问题，见教程第 10 节排查表。

## 目录结构

```
1.hello/
├── CMakeLists.txt  # CMake 构建脚本（带逐行注释）
├── hello_cuda.cu   # 源码（含详细注释）
├── run.sh          # 一键编译运行脚本（封装 CMake 三步曲）
├── README.md       # 本文档
└── build/          # CMake 构建目录（run.sh 生成，可 clean）
```

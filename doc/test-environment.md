# 测试环境配置与验证

## 已验证的环境

以下配置来自原项目 2026-10-23 的硬件记录和实测日志。
这是当时通过验证的版本组合；迁移本身不安装、升级或移动本机已有 Conda 环境。

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，Linux x86_64 |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti，标称每卡 11 GiB |
| GPU 架构 | Turing，计算能力 7.5，编译目标 `sm_75` |
| 双卡拓扑 | PCIe，`nvidia-smi topo -m` 显示 NODE |
| NVIDIA 驱动 | 590.48.01 |
| 环境管理 | Miniforge / Conda；原环境名 `gpu-dev` |
| Python | 3.11 |
| PyTorch | 2.14.1+cu126，官方 wheel |
| CUDA Toolkit | 12.6.3，nvcc 12.6.85，NVIDIA Conda 包 |
| NCCL | 2.29.3，使用 PyTorch 依赖提供的头文件和动态库 |
| C/C++ 编译器 | Conda GCC/G++ 12.4.0 |
| 构建工具 | CMake、Ninja、Make |
| 调试及分析工具 | cuda-gdb、Compute Sanitizer、Nsight Compute |

原机 `nvidia-smi` 显示的 CUDA 13.1 是驱动支持版本；这里实际编译使用环境内的 CUDA 12.6。
原机还安装了 `/usr/local/cuda-12.5`，激活本环境后以 `$CUDA_HOME` 和 `which nvcc` 为准。
测试需要宿主机驱动可用、两张 GPU 对进程可见，以及进程间通信所需的本地端口和共享内存。

## 安装与精确复现

先安装 Miniforge（Linux x86_64），安装器从 [Miniforge 官方发布页](https://github.com/conda-forge/miniforge/releases) 获取。
驱动由宿主机提供，仓库不安装驱动或携带安装器。以下假设 Miniforge 安装在 `~/miniforge3`；其他位置请调整首行路径。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/dev/GPU-Learning
bash setup/recreate.sh gpu-dev-copy
conda activate gpu-dev-copy
```

如果复用本机已有环境，只需 `conda activate gpu-dev`，无需重建。
`setup/recreate.sh` 从当前 Conda 安装定位初始化脚本，可以从任意工作目录调用。
环境名参数应使用普通 Conda 名称；已存在的环境会被拒绝覆盖。
新建过程需要访问锁定文件中的 Conda 和 pip 下载地址。

重建脚本执行以下配置：

1. 用 `setup/conda-linux-64.lock.txt` 安装精确的 Linux x86_64 Conda 包，包括 CUDA 工具链和编译器。
2. 用 `setup/requirements-pip.lock.txt` 安装原测试环境的 pip 版本快照，包括 PyTorch 官方 wheel。
3. 用 `setup/link_nccl.py` 将同一份 NCCL 头文件和动态库链接到环境内的 `opt/nccl`，供 C++ 示例使用。
4. 将 `setup/conda-pinned.txt` 放到环境的 `conda-meta/pinned`，约束 Python 与 CUDA 版本。
5. 为新环境保存以下变量，激活后生效：

| 变量 | 值 |
| --- | --- |
| `CUDA_HOME`、`CUDA_PATH` | 新环境的 `$CONDA_PREFIX` |
| `CUDACXX` | `$CONDA_PREFIX/bin/nvcc` |
| `NCCL_HOME` | `$CONDA_PREFIX/opt/nccl` |
| `TORCH_CUDA_ARCH_LIST` | `7.5` |
| `MAX_JOBS` | `4` |

Conda 编译器包在激活时配置 `CC`、`CXX` 等编译变量。
NCCL 链接脚本仅创建环境内的符号链接，不另装一套 NCCL。
`environment.yml` 记录主要 Conda 依赖，单独创建它还不包含 PyTorch 安装和上述环境配置；完整复现请用重建脚本。
锁定文件保留了原环境的 Notebook 等辅助依赖，便于复现；这些依赖不是运行每个示例的前提。

若已有环境缺少开发配置，可以激活它后在仓库根目录执行：

```bash
python setup/link_nccl.py
cp setup/conda-pinned.txt "$CONDA_PREFIX/conda-meta/pinned"
conda env config vars set \
  CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  CUDACXX="$CONDA_PREFIX/bin/nvcc" NCCL_HOME="$CONDA_PREFIX/opt/nccl" \
  TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4
```

然后重新激活环境。VS Code 选择该环境的 Python。
需要 Notebook 时可在激活环境后执行 `python -m ipykernel install --user --name gpu-dev --display-name "Python (gpu-dev)"`；
从已激活的终端启动 Notebook 服务，使内核继承 CUDA 和编译器环境变量。

## 基础检查

以下命令在激活开发环境后执行。示例中的相对路径均以仓库根目录为基准：

```bash
cd ~/dev/GPU-Learning
nvidia-smi
nvidia-smi topo -m
which python nvcc
nvcc --version
./run_checks.sh
```

脚本依次检查双卡矩阵乘法、原生 CUDA 加法、PyTorch CUDA 扩展、两进程 NCCL 四种集合通信，
以及单进程双卡 NCCL C API AllReduce。任一步失败即退出，全部成功输出 `All environment checks: PASS`。
编译产物写入被忽略的 `build/`；PyTorch 扩展使用 PyTorch 自身的编译缓存。
GPU 不可见或设备数不足两张时，检查预期会失败。

单项测试命令：

```bash
python verify_env.py

mkdir -p build
nvcc -O2 -lineinfo -arch=sm_75 examples/vector_add.cu -o build/vector_add
./build/vector_add
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/vector_add

python examples/pytorch_cuda_op.py

torchrun --standalone --nproc_per_node=2 examples/nccl_collectives.py
# 查看通信初始化与传输选择：
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 examples/nccl_collectives.py

nvcc -O2 -arch=sm_75 examples/nccl_allreduce.cu \
  -I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" \
  -Xlinker -rpath -Xlinker "$NCCL_HOME/lib" -lnccl -o build/nccl_allreduce
./build/nccl_allreduce
```

`run_checks.sh` 不包含 Compute Sanitizer、Nsight Compute 或官方 nccl-tests，它们按文档单独运行。
换用其他架构的 GPU 时，需要同时调整 `TORCH_CUDA_ARCH_LIST` 和 nvcc 的 `-arch`；
检查脚本的 nvcc 目标可通过 `CUDA_ARCH` 覆盖，默认 `sm_75`。当前示例及版本组合仅在上述硬件验证过。

## 官方 nccl-tests

第三方测试源码按固定提交下载到 `external/`，无需拷入本仓库，也不需要 MPI。
在已激活开发环境的仓库根目录执行（目标目录应尚不存在）：

```bash
mkdir -p external
git clone https://github.com/NVIDIA/nccl-tests.git external/nccl-tests
git -C external/nccl-tests checkout --detach "$(cat setup/nccl-tests-commit.txt)"

NVLDFLAGS="-Xlinker -rpath -Xlinker $NCCL_HOME/lib -Xlinker -rpath -Xlinker $CUDA_HOME/targets/x86_64-linux/lib" \
make -C external/nccl-tests -j4 MPI=0 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
  CUDA_INC="$CUDA_HOME/targets/x86_64-linux/include" \
  CUDA_LIB="$CUDA_HOME/targets/x86_64-linux/lib" \
  NVCC_GENCODE='-gencode=arch=compute_75,code=sm_75'

./external/nccl-tests/build/all_reduce_perf -b 8 -e 32M -f 2 -g 2 -w 2 -n 10 -c 1
```

固定提交为 `afd59abab774a6a4b492eb344bd0c845a0b7bb4d`，对应原测试日志中的 nccl-tests 2.21.1。
该测试覆盖 8 B 到 32 MiB，启用结果校验。小规模运行用于确认通信可用性，不能据此断言完整硬件性能。
多机 MPI/RDMA 不在已验证范围内。

## 已有测试结果与限制

原目录 2026-10-23 保存的日志记录：

| 检查 | 结果 |
| --- | --- |
| 两张 GPU 的 PyTorch 矩阵乘法、架构与版本 | PASS |
| 原生 CUDA 加法与 Compute Sanitizer | PASS，0 errors |
| PyTorch CUDA 扩展，双卡与独立 stream | PASS |
| 两进程 NCCL 四种集合通信 | PASS |
| NCCL C API 双卡 AllReduce | PASS |
| 官方 nccl-tests 双卡 AllReduce | PASS，0 个错误 |
| Nsight Compute 性能计数器 | `ERR_NVGPUCTRPERM`，需要管理员开放权限 |

性能计数器权限限制不影响这里的 CUDA 运算与显存检查。
普通 PyTorch CUDA 运算、CUDA C++ 扩展和 NCCL 是本项目的验证范围。
原环境虽然随 PyTorch 安装了 Triton，但未验证 Triton 示例。

这些是迁移前的测试记录。迁移会话中 `nvidia-smi` 无法连接驱动，且没有 `/dev/nvidia*`，未能重跑 GPU 测试。
原始日志保留在原机旧目录中；仓库仅保存配置和结果摘要。新机器应按上述命令重新验证。

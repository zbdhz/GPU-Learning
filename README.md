# GPU-Learning

面向单机双 GPU 的 CUDA、PyTorch 自定义算子与 NCCL 集合通信学习示例。
测试硬件为两张 RTX 2080 Ti（`sm_75`）。

## 开始使用

先按 [测试环境配置](doc/test-environment.md) 安装环境，或复用本机已有的 `gpu-dev`：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gpu-dev
cd ~/dev/GPU-Learning
./run_checks.sh
```

脚本基于自身位置定位仓库，可以从任意工作目录调用。

## 示例

| 文件 | 内容 |
| --- | --- |
| `verify_env.py` | 版本、GPU 与双卡矩阵乘法正确性检查 |
| `examples/vector_add.cu` | 原生 CUDA 向量加法、设备内存与边界检查 |
| `examples/add_binding.cpp`、`examples/add_kernel.cu` | PyTorch 的 C++/CUDA 扩展 |
| `examples/pytorch_cuda_op.py` | 编译扩展，验证双卡、空输入和独立 stream |
| `examples/nccl_collectives.py` | 每卡一个进程，验证四种集合通信 |
| `examples/nccl_allreduce.cu` | 单进程管理双卡，直接调用 NCCL C API |

PyTorch 扩展演示 float32 前向计算。自动求导、多机 MPI/RDMA 和进阶性能优化尚未覆盖。

## 仓库内容

- `examples/`：学习示例源码。
- `setup/`：环境版本锁定、重建脚本和 NCCL 开发文件链接脚本。
- `doc/`：测试环境配置、单项测试命令和已知限制。
- `environment.yml`：主要 Conda 依赖；精确复现使用 `setup/` 中的锁定文件。

安装器、Conda 环境、编译产物、缓存、原始日志与第三方源码不纳入版本管理。
官方 `nccl-tests` 按文档下载到被忽略的 `external/`，固定使用已测试的提交。

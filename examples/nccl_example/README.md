# NCCL 文档配套实验

每篇原文对应一个独立目录，每个目录均含代码、README.md 和 Makefile，不依赖其他实验目录的源码。
先按 [测试环境配置](../../doc/test-environment.md) 激活开发环境，再运行对应目录的 `make`。

## 新手阅读顺序

源码的有效语句前已添加中文注释，Makefile 也说明了变量、目标和命令参数。
不必先理解所有工具代码，可以按下面的顺序跟着数据看：

1. [mini_tp.py](01_stack_overview/mini_tp.py)：先看如何选 CPU/GPU，再看进程组如何初始化。`rank` 是全局进程编号，`LOCAL_RANK` 是本机进程编号，`world` 是进程总数。
2. [all_reduce.py](01_stack_overview/all_reduce.py)：跟踪两个进程各自的 Tensor，从 1、2 变成各自的 3；两份代码运行在独立进程中，各有自己的变量。
3. [exp2_primitives.py](02_process_groups/exp2_primitives.py)：比较广播、求和、收集、分片和点对点的结果差异。
4. [exp5_p2p.cu](04_p2p_ipc/exp5_p2p.cu) 和 [exp6_ipc.cu](04_p2p_ipc/exp6_ipc.cu)：分清 CPU 指针变量、GPU 显存以及跨进程的 IPC 句柄。
5. [rdma_write.cpp](05_rdma_verbs/rdma_write.cpp)：先读 `experiment()` 的九步流程，再回头看 TCP 握手和资源释放辅助函数。

CUDA 的 kernel 是在 GPU 上运行的函数；“提交 kernel”与“GPU 已执行完”是两件事，代码中的同步注释解释了何时需要等待。
RDMA 的 QP 是工作请求队列对，CQ 是完成记录队列，MR 是网卡已登记的内存区域；三者各自负责不同的步骤。

## 目录与运行

| 原文章节 | 目录 | 默认 make 执行内容 |
| --- | --- | --- |
| 01 通信栈全景 | [01_stack_overview](01_stack_overview/README.md) | 两进程 AllReduce 自检 |
| 02 进程组、rank 与集合通信 | [02_process_groups](02_process_groups/README.md) | 自检、四 rank 子组、通信原语 |
| 03 Python 到 CUDA | [03_python_to_cuda](03_python_to_cuda/README.md) | 首轮计时、预热后的 profiler |
| 04 机内通路 | [04_p2p_ipc](04_p2p_ipc/README.md) | 编译并运行 P2P、双进程 CUDA IPC |
| 05 RDMA Verbs | [05_rdma_verbs](05_rdma_verbs/README.md) | 自动运行本机 ib_write_bw 双方；make verbs 运行完整九步代码 |

```bash
# 仓库根目录运行指定章节：
make -C examples/nccl_example/02_process_groups DEVICE=cpu
# 一次运行前三篇的 CPU 实验：
make -C examples/nccl_example cpu
# 有双 GPU/P2P 条件时：
make -C examples/nccl_example/04_p2p_ipc
# 已配置 Soft-RoCE 时：
make -C examples/nccl_example/05_rdma_verbs RDMA_DEVICE=rxe0
```

前三篇默认自动选择 CUDA/NCCL 或 CPU/Gloo；第 2 篇的四 rank 子组默认使用 CPU。
所有 Python Makefile 可通过 `PYTHON=/path/to/python` 指定解释器。
P2P 不可用时第 4 篇打印 SKIP；无 GPU/驱动等错误会失败。第 5 篇需要单独安装系统 RDMA 依赖。
本机原记录为双 RTX 2080 Ti、PCIe NODE，不能由此声称具备 NVLink 或 GPUDirect。

2026-10-26 已验证前三篇 CPU/Gloo 运行、CUDA 示例编译和 RDMA 主存/显存版本编译。
GPU、真实 RDMA 与多机路径仍需相应硬件验证；详情见各目录说明。

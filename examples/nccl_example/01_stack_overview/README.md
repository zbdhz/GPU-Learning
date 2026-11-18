# 01 从一行 dist.all_reduce 到 NVLink 与 RDMA：大模型通信栈全景

对应原文：[阅读文档](<../../../doc/nccl_example/01 从一行 dist.all_reduce 到 NVLink 与 RDMA：大模型通信栈全景.md>)。

## 一键运行

先按仓库 [测试环境配置](../../../doc/test-environment.md) 激活 `gpu-dev`，然后：

```bash
cd examples/nccl_example/01_stack_overview
make
# 无 GPU 时可显式选择：
make DEVICE=cpu
# 有两张可见 GPU 时：
make DEVICE=cuda
```

本篇为全景介绍，原文只有一行 `dist.all_reduce(tensor)`。`all_reduce.py` 将它补成完整的多进程自检：
每个 rank 贡献 `rank+1`，最终所有 rank 都应得到 `world*(world+1)/2`，两进程时为 3。
每个 rank 都断言结果，失败会以非零状态退出。`mini_tp.py` 在本目录内提供入口，不依赖其他示例目录。

`DEVICE=auto` 默认在 CUDA 可用时选择 NCCL，否则选择 CPU/Gloo；显式 `DEVICE=cuda` 不会退回 CPU。
`NPROC` 控制进程数，CUDA 模式每个进程使用一张 GPU；`PYTHON` 可指定环境解释器。
`make debug DEVICE=cuda` 开启 NCCL 初始化日志。

本例只验证语义。AllReduce 成功不代表底层一定走 NVLink 或 RDMA，传输方式需结合拓扑和日志确认。

## 验证状态

- 2026-10-26：双进程 CPU/Gloo 实际运行通过。
- 2026-11-01：用户提供 NJUEE 主机的 `make debug` 日志，确认双进程 CUDA/NCCL AllReduce 校验通过；实际卡间连接为 `SHM/direct/direct`。该日志不代表 NVLink、GPU P2P 或 RDMA 通路验证通过。


## 一次 `make debug` 的完整过程解析

以下解读对应 2026-11-01 用户提供的 NJUEE 运行日志。此次实验成功：GPU 0 提供 1，GPU 1 提供 2，最终两边都得到 3。
实际卡间连接使用共享主存 SHM；日志没有证明本次数据通过 NVLink 或 RDMA 传输。

### 1. Makefile 启动两个独立进程

`make debug` 展开后的命令相当于：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run \
  --standalone --nproc_per_node=2 all_reduce.py --device auto
```

| 部分 | 作用 |
| --- | --- |
| `NCCL_DEBUG=INFO` | 为本次运行开启 NCCL 信息日志，不是执行额外一次实验 |
| `python -m torch.distributed.run` | 通过 Python 模块启动 torchrun |
| `--standalone` | 自动配置本次单机运行的进程集合入口 |
| `--nproc_per_node=2` | 在本机启动两个独立的 Python 进程 |
| `all_reduce.py` | 两个进程执行同一份脚本，但各自拥有自己的变量和 Tensor |
| `--device auto` | CUDA 可用时使用 GPU/NCCL，否则使用 CPU/Gloo |

最先出现的 OpenMP 提示：

```text
Setting OMP_NUM_THREADS environment variable for each process to be 1
```

表示启动器为每个进程设置 CPU 侧的 OpenMP 线程数，避免多个进程各自创建过多 CPU 工作线程。
它不表示 GPU 只使用一个线程，也不是报错。

随后两个进程分别打印：

```text
rank=1/2 local_rank=1 device=cuda:1 backend=nccl
rank=0/2 local_rank=0 device=cuda:0 backend=nccl
```

| 进程 | 全局编号 rank | 本机编号 local_rank | 使用设备 |
| --- | --- | --- | --- |
| 第一个参与者 | 0 | 0 | GPU 0，即 cuda:0 |
| 第二个参与者 | 1 | 1 | GPU 1，即 cuda:1 |

`/2` 表示组内共有两个进程。rank 1 的日志先显示很正常：进程编号不决定操作系统先调度谁。
多进程、多线程的日志会交错，不能把终端显示顺序当作所有操作的严格全局顺序。

### 2. 创建输入，首次通信触发 NCCL 初始化

脚本的核心代码是：

```python
t = torch.tensor([float(rank + 1)], device=dev)
dist.all_reduce(t)
```

因此，通信前两个进程分别持有：

```text
进程 0：GPU 0 上的 t = [1.0]
进程 1：GPU 1 上的 t = [2.0]
```

`mini_tp.py` 已调用 `init_process_group()` 建立默认进程组，但 NCCL 设备通信器可能延迟到首次通信时才创建。
在本次输出中，rank 信息打印后出现了大量 NCCL 初始化日志，与首次 AllReduce 触发设备通信器初始化的顺序吻合。

```text
ncclCommInitRankConfig ... rank 0 nranks 2 cudaDev 0 ... Init START
```

这表示为 rank 0 初始化一个包含两个参与者、使用 GPU 0 的 NCCL 通信器。
通信器保存参与者、连接、队列等运行通信所需的状态。
两个 rank 日志中相同的 `commId`，说明它们在加入同一次 NCCL 通信。
`comm` 后面的十六进制地址则是各自进程中的本地对象地址，不要求相同。

日志前缀例如：

```text
NJUEE:3240634:3240688 [0] NCCL INFO ...
```

依次是主机名、进程 ID、线程 ID、当前 CUDA 设备编号、日志级别。
方括号里的数字不能一概当作全局 rank，尤其是后台线程输出的记录。

版本日志中，`cudaDriverVersion 13010` 表示驱动报告支持的 CUDA API 版本为 13.1；
`NCCL version 2.29.3+cuda12.9` 是该 NCCL 库报告的版本与构建标记。
这些信息不能代替 `nvcc --version` 判断当前 CUDA 编译工具链，也不能代替 `torch.version.cuda` 判断 PyTorch 的构建版本。

### 3. Bootstrap：先交换“联系方式”

```text
Bootstrap: Using enp181s0:114.212.115.112<0>
```

Bootstrap 是 NCCL 通信器初始化时的联络阶段，用来让参与者交换后续建立连接所需的信息。
这里选用了 `enp181s0` 网络接口。

需要分清三件事：

- torchrun 的 rendezvous：让 Python 分布式进程找到彼此。
- NCCL 的 bootstrap：让 NCCL 参与者交换连接信息。
- 实际 Tensor 数据传输：根据最终选中的 P2P、SHM 或 NET 等通路进行。

它们处于不同层次。日志出现网卡 IP，不表示本次 GPU Tensor 一定经由外部网络传输。
`Bootstrap timings` 记录的是这一联络阶段的耗时，不是 AllReduce 的纯数据传输耗时。

### 4. 探测网络后端：IB 不可用，准备 Socket 后端

```text
Failed to open libibverbs.so[.1]
Failed to initialize NET plugin IB
Initialized NET plugin Socket
Assigned NET plugin Socket to comm
Using network Socket
```

这几行表示 NCCL 尝试准备 IB/RDMA 网络后端，但无法加载 Verbs 动态库，随后成功准备并选择了 Socket 网络后端。
`libibverbs` 是 Linux 上应用程序访问 RDMA 设备所需的用户态库之一。

本次能确定的是“IB 后端初始化失败”，不能仅凭加载库失败就断言“机器没有 RDMA 网卡”。
是否存在 HCA、设备是否可用，还需要单独检查。

同样，`Using network Socket` 描述的是选择了哪个网络后端；单机卡间通信仍可以选择 SHM。
判断 Tensor 实际怎么走，要继续看后面的逐连接 `via ...` 日志。
日志列出的 `enp181s0`、`tailscale0` 是 Socket 后端识别到的接口，不证明本例数据走过每个接口。

### 5. 识别单机双卡，规划逻辑通道

```text
nRanks 2 nNodes 1 localRanks 2
```

表示共有两个 rank，分布在一台机器上，本机有两个参与者。
CPU affinity 的 `0-19` 表示该阶段报告的 CPU 亲和性范围，不是说启动了 20 个 rank。

```text
Channel 00/02 : 0 1
Channel 01/02 : 0 1
```

表示准备了两个逻辑通信通道，参与者排列均为 rank 0、rank 1。
channel 是 NCCL 内部组织通信工作的逻辑通路，不是新的 Python 进程，也不等于独立物理网线。
这里每个 Tensor 只有一个 float32，不能根据“两个 channel”推断实际数据被均分为两份或带宽翻倍。

`Trees ...` 则描述准备的树形拓扑。日志同时出现 ring 和 tree 的规划信息，
不表示本次数据必然完整执行了一次 ring 算法、又执行了一次 tree 算法。
仅靠这些 INFO 初始化日志，不能准确还原每个数据块的 kernel 执行过程。

### 6. 关键证据：实际连接走 SHM

本次最关键的逐连接日志是：

```text
Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
Channel 00 : 1[1] -> 0[0] via SHM/direct/direct
Channel 01 : 0[0] -> 1[1] via SHM/direct/direct
Channel 01 : 1[1] -> 0[0] via SHM/direct/direct
```

两个通道的双向连接都选中了 SHM，即共享主存传输。可以按下面的概念路径理解：

```text
GPU 0 显存 ⇄ 主机共享内存缓冲区 ⇄ GPU 1 显存
```

SHM 使用主机内存作为设备间通信媒介，GPU P2P 则属于另一类直接设备访问通路。
具体传输机制见 [NCCL 官方环境变量说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2243/user-guide/docs/env.html)。

`direct/direct` 是 SHM 通路内部实现的标记，不能解读成“GPU 直接通过 NVLink 写入另一张卡”。
同样，经过主存不等于 CPU 必须用普通循环逐字节搬数据；主存访问还可能由 GPU 或 DMA 等机制执行。

前面的：

```text
Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0
```

与本次没有建立全 P2P 连接的情况相符，但仅凭这些日志无法确定没有选择 P2P 的具体原因。
硬件拓扑、驱动支持和环境配置等都可能影响选择。

另一个容易误读的输出是：

```text
Connected all rings, use ring PXN 0 GDR 1
```

不能单凭其中的 `GDR 1` 宣称本次使用 GPUDirect RDMA。
本次实际连接明确显示为 SHM，而且 IB 后端初始化失败，没有对应的 NET/IB 数据通路证据。
连接 ring 完成的记录，也不等于一份逐操作的算法执行轨迹。

### 7. 初始化完成，执行求和并校验

```text
Init timings - ncclCommInitRankConfig: ... total 0.16 (kernels 0.14, ...)
```

该日志的时间单位为秒：总初始化耗时约 0.16 秒，即 160 毫秒，其中 `kernels` 初始化项约 140 毫秒。
这是通信器初始化开销，不能当作“一个浮点数相加需要 160 毫秒”，也不是测得的纯 AllReduce 延迟。
两个 rank 各自记录约 160 毫秒，并不能相加成程序总耗时，因为它们会并行工作。

通信完成后：

```text
rank=0: all_reduce=3.0 expected=3.0 PASS
rank=1: all_reduce=3.0 expected=3.0 PASS
```

对应数据变化：

| 参与者 | 原输入 | 求和后的本地 Tensor | 校验 |
| --- | --- | --- | --- |
| rank 0 / GPU 0 | `[1.0]` | `[3.0]` | PASS |
| rank 1 / GPU 1 | `[2.0]` | `[3.0]` | PASS |

AllReduce 逐对应位置求和，完整结果每个参与者都拿到一份，并原地写回各自的 `t`。
脚本用 `world*(world+1)/2` 算正确答案，通过 `torch.testing.assert_close` 比较。
GPU Tensor 的 `.item()` 将数值取回 Python，过程中可能等待 GPU 工作完成。

### 8. 销毁通信器，正常退出

```text
Destroy COMPLETE
ENV/Plugin: Closing env plugin ncclEnvDefault
```

对应脚本的 `finally` 中调用 `dist.destroy_process_group()`，释放本进程持有的通信资源。
双方都输出 PASS，随后都完成销毁，说明这次功能实验正常结束。

### 其他 INFO 提示怎么看

| 日志 | 本次的含义 |
| --- | --- |
| 缺少 `libnccl-env.so` | 未找到外部环境插件；后续仍使用默认实现 |
| 缺少 `libnccl-net.so` | 未找到外部网络插件；不代表 NCCL 没有内置网络后端 |
| 缺少 `libnccl-profiler.so`、`libnccl-tuner.so` | 没有加载对应外部分析、调优插件，不妨碍本例完成求和 |
| `NVLS multicast support is not available` | 未启用 NVLink SHARP 相关能力；不能据此否定所有通信，也不能单独据此判断有无普通 NVLink |
| `Symmetric memory is not supported` | 该高级内存机制未启用；普通 AllReduce 仍然可以工作 |
| `RAS client listening socket` | NCCL 的可靠性、可用性和可维护性诊断服务监听信息，不是又启动了一个训练 rank |
| `Proxy Service`、`Proxy Service UDS` | NCCL 的后台服务线程及相关控制服务信息；UDS 是 Unix domain socket，不等于外部网卡数据通路 |
| `P2P Chunksize set to 131072` | 内部配置项的输出，单独出现不能证明本次选用了 GPU P2P 传输 |

本次过程可以按这条线索阅读：

```text
启动两个进程
  → 各自绑定 GPU
  → 创建输入 1 和 2
  → 首次通信触发 NCCL 初始化
  → 交换连接信息并探测通路
  → 为单机卡间连接选择 SHM
  → 两边得到 3 并校验通过
  → 释放通信资源
```

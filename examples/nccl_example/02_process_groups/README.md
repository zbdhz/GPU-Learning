# 02 谁和谁通信：进程组、rank 与集合通信

对应原文：[阅读文档](<../../../doc/nccl_example/02 谁和谁通信：进程组、rank 与集合通信.md>)。

## 一键运行

激活仓库的 `gpu-dev` 环境后，在本目录执行：

```bash
make
# 全部以 CPU 复现（包含四进程子组）：
make DEVICE=cpu
```

默认按顺序运行自检、子组、原语实验；每步失败立即停止，即使 `make -j` 也不会把这些实验并发启动。

| 文件 / 目标 | 对应实验与校验 |
| --- | --- |
| `mini_tp.py` | 根据 torchrun 环境变量建组，绑定 LOCAL_RANK 对应 GPU；进程组超时为 60 秒 |
| `exp0_hello.py` / `make hello` | rank、local_rank、world 与求和，双进程各得 3 |
| `exp1_groups.py` / `make groups` | 四 rank 同序建立 `[0,1]` 和 `[2,3]`，分别得到 3、7 |
| `exp2_primitives.py` / `make primitives` | broadcast=99、all_reduce=3、reduce 的 dst=3、all_gather=[1,2]、reduce_scatter=[3,30]、send/recv=777（双进程） |
| `make debug DEVICE=cuda` | 在自检中开启 NCCL 日志 |

子组实验严格保留原文的四 rank 设计。默认 `GROUP_DEVICE=cpu`，所以双 GPU 原机也能运行它。
有四张 GPU 时可用 `make groups GROUP_DEVICE=cuda`；若要全部实验使用 GPU，执行 `make DEVICE=cuda GROUP_DEVICE=cuda`。
其他两个实验默认 `DEVICE=auto`、`NPROC=2`。CPU 模式可用 `NPROC=4` 等更多进程；子组实验的进程数固定为 4。

`reduce` 只检查目标 rank 的输出，其余 rank 的 buffer 不具有结果语义。
`send/recv` 只在 rank 0、1 之间执行；全部 rank 最终经过 barrier 再销毁进程组。
所有原语都有正确性断言，进程组在退出时释放。

多机自检也可直接启动：两台机器分别设置 `NODE_RANK=0` 和 `1`，以及相同的主节点地址：

```bash
# 两台机器分别执行；把地址和本机节点编号替换为实际值。
torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=2 \
  --master_addr=$MASTER_ADDR --master_port=29500 exp0_hello.py --device cuda
```

主节点端口及后端通信需要网络可达。多机运行不会自动保证使用 RDMA。
环境安装见 [测试环境配置](../../../doc/test-environment.md)。

## 验证状态

2026-10-26：自检、四进程子组和六项原语的 CPU/Gloo 路径全部实际运行通过。
`reduce_scatter_tensor` 按原文保留；原机 PyTorch 2.14.1 会给出建议使用 `reduce_scatter_single` 的弃用提示。
本会话未验证 GPU 或多机路径。

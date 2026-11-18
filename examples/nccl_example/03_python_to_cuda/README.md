# 03 从Python到CUDA：一行 all_reduce 是怎么变成 GPU kernel 的

对应原文：[阅读文档](<../../../doc/nccl_example/03 从Python到CUDA：一行 all_reduce 是怎么变成 GPU kernel 的.md>)。

## 一键运行

激活仓库的开发环境后，在本目录执行：

```bash
make
make DEVICE=cpu       # 没有 GPU 时观察 Gloo
make DEVICE=cuda      # 两张可见 GPU，观察 NCCL/CUDA kernel
```

默认先计时再录制 profiler。`PYTHON` 可指定解释器，`NPROC` 默认 2，`DEVICE` 默认 auto。
目录自带 `mini_tp.py` 与日志自检 `exp0_hello.py`，无需从第 2 篇目录复制文件。

## 实验与输出

- `make timing`：运行 `exp3_timing.py`，对 1024×1024 float32 Tensor 连续进行六轮 AllReduce。每轮前恢复输入，在 CUDA 模式下于计时两端同步 GPU，再检查结果为 world。
- `make profiler`：运行 `exp4_profiler.py`，预热三轮后只采集一次通信，在每个 rank 生成 `build/rankN.trace.json`。可用 Chrome tracing 或 Perfetto 打开；rank 0 同时打印耗时表。
- `make debug DEVICE=cuda`：用自检触发 NCCL 初始化并输出 INIT/GRAPH/P2P/NET 日志。实际输出随硬件和 NCCL 版本变化。
- `make clean`：清理本目录的 trace。

第一次通信可能包含初始化开销，但本例不对“首轮一定最慢”作断言；调度与系统负载也影响耗时。
CPU/Gloo 模式录制 CPU 活动，没有 NCCL GPU kernel；CUDA 模式记录 CPU 与 GPU 活动，
需要 PyTorch profiler/CUPTI 可用。不要将 CPU trace 当作 GPU 通信证据。

原文中的 `ProcessGroupNCCL::collective()` 是 PyTorch 内部流程的删节示意，不是可单独编译的接口。
本例通过真实的 `dist.all_reduce` 和 profiler 观察该调用链，不另写一个同名函数模拟 PyTorch。

## 验证状态

2026-10-26：CPU/Gloo 的六轮计时、profiler 及两个 rank 的 trace 导出均实际通过。
本会话没有可用 GPU，未验证 CUDA profiler 或 NCCL kernel。

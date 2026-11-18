# 04 机内通路：NVLink 上的一次显存直写

对应原文：[阅读文档](<../../../doc/nccl_example/04 机内通路：NVLink 上的一次显存直写.md>)。

## 环境与一键运行

需要 Linux、CUDA Toolkit 的 `nvcc`，以及两张允许 GPU 0 访问 GPU 1 的 GPU。
先按 [测试环境配置](../../../doc/test-environment.md) 激活 `gpu-dev`，在本目录执行：

```bash
make                         # 编译，先运行 P2P，再自动运行 IPC 的两个进程
make build                   # 只编译
make p2p                     # 单进程跨 GPU 直写
make ipc PORT=29777           # 自动启动 holder/writer
make CUDA_ARCH=sm_75          # 原机 RTX 2080 Ti 的目标架构
```

`NVCC`、`NVCCFLAGS`、`PYTHON` 可覆盖。编译产物与 holder 日志放在本目录的 `build/`，`make clean` 清理。
无 nvcc、无 GPU 或 CUDA 初始化错误会使命令失败。
两张卡可见但 `cudaDeviceCanAccessPeer(0,1)=0` 时，程序返回 77，Makefile 明确显示 SKIP；这不代表实验通过。

## 对应代码

- `exp5_p2p.cu`：在 GPU 1 分配并清零 buffer，同步初始化，在 GPU 0 开启 peer access，然后由 GPU 0 的 kernel 写入 42。回到 GPU 1 拷回并断言，释放 buffer 和 peer access。
- `exp6_ipc.cu`：holder 在 GPU 1 分配显存，通过 TCP 发送 IPC handle；writer 在 GPU 0 导入，执行 kernel 并同步，关闭导入映射，然后回复确认。holder 收到确认后回读并断言为 42，最后释放 allocation。
- `run_ipc.py`：等待 holder 的 READY 信号再启动 writer，处理超时并回收子进程；holder 日志在 `build/holder.log`。
- `cuda_check.h`：错误处理及设备/P2P 能力检查。

TCP 控制通道只使用本机 loopback，只传 IPC 句柄和完成通知，不发送数值 42。
收发均处理短读写；只有 writer 关闭映射后，holder 才释放显存。

保留原文的双终端操作方式：

```bash
# 终端 A，先执行：
make holder PORT=29777
# 终端 B，使用相同环境和设备可见性，再执行：
make writer PORT=29777
```

手动启动的 holder 等待连接；自动模式会在 30 秒内等待就绪，并限制 writer 的运行时长。
CUDA IPC 只能在同一台主机使用，不是跨机 RDMA。

## 链路说明与原文修正

`P2P/IPC` 不等于 NVLink：PCIe P2P 也能运行同一段代码，实际链路请通过 `nvidia-smi topo -m` 检查。
原机记录为 NODE，不能把结果宣称为 NVLink 带宽或连通性验证。
原文“无 P2P 时 exp6 照做”不适用于跨卡直写：IPC 导入无法创造硬件/驱动不支持的 P2P 能力。
本例在两种实验中都检查这一前提。

接口参考：[NVIDIA CUDA IPC 内存管理](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)。

## 验证状态

2026-10-26：两个程序均已使用 CUDA 12.6 nvcc、`sm_75` 编译通过。
本会话缺少 GPU 设备，P2P、IPC 的实际运行和 NVLink 通路未验证。

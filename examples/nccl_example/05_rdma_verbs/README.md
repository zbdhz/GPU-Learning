# 05 跨机通路：RDMA 的一串动词（Verbs）

对应原文：[阅读文档](<../../../doc/nccl_example/05 跨机通路：RDMA 的一串动词（Verbs）.md>)。

## 环境与安装

需要 Linux 上可用的 RDMA 设备：真实 IB/RoCE HCA，或管理员预先配置的 Soft-RoCE/RXE。
主存 RDMA 实验不需要 GPU。以 Ubuntu 为例，由管理员安装：

```bash
sudo apt install build-essential pkg-config libibverbs-dev ibverbs-providers ibverbs-utils rdma-core perftest
ibv_devices
ibv_devinfo
```

程序使用普通用户权限运行。`ibv_reg_mr` 失败时检查 `ulimit -l` 与设备访问权限。
`RDMA_DEVICE` 是 HCA 名称，`IB_PORT` 是 HCA 端口；`GID_INDEX` 应按实际路由/GID 表选择，
默认 0 仅是起点，RoCE v2 常需其他索引。IB 使用 LID 路由时可设 `GID_INDEX=-1`。
两端需使用匹配的配置，内存注册、QP 状态切换或网络不通会明确失败。

## 一键运行原文 perftest 实验

在本目录执行（真实 HCA 的本机回环或已配置的 RXE）：

```bash
make RDMA_DEVICE=rxe0
# 也可使用真实 HCA：
make RDMA_DEVICE=mlx5_0 GID_INDEX=3
```

默认目标运行 `ib_write_bw` 的 server 和 client，自动等 server 监听后再启动 client。
使用 64 KiB、100 次迭代，输出带宽；这是功能实验，不能作为完整性能评估。
`run_pair.py` 不会通过探测连接消耗 perftest 的一次握手，有启动和运行超时，并回收 server。
原始输出保存在本地 `build/perftest-server.log`。

双机运行，在两台机器的本目录分别执行：

```bash
# 机器 0：
make server RDMA_DEVICE=mlx5_0 GID_INDEX=3
# 机器 1：
make client RDMA_DEVICE=mlx5_0 GID_INDEX=3 SERVER=192.168.1.10
```

两台机器的 HCA 名称和 GID 索引可能不同，按各自配置填写；`PORT` 默认 29877。

## Verbs 九步的完整代码

原文的骨架带省略号，`rdma_write.cpp` 将其补为完整的 RC 连接与一次 RDMA WRITE：

1. `ibv_get_device_list` / `ibv_open_device` 打开网卡。
2. `ibv_alloc_pd` 建立保护域。
3. `ibv_create_cq` 建立完成队列。
4. `ibv_create_qp` 建立 RC 队列对。
5. `ibv_reg_mr` 注册 4 KiB 主存（可选显存），仅传输前 4 字节。
6. TCP 交换 qpn、lid、gid、mtu、远端地址与 rkey。
7. `ibv_modify_qp` 将 QP 推至 INIT、RTR、RTS，双方确认 ready。
8. client 用 `ibv_post_send(IBV_WR_RDMA_WRITE)` 写入网络端序的 42。
9. client 用 `ibv_poll_cq` 检查 CQE，然后通过 TCP 发完成通知；server 回读并断言为 42，再回复校验确认。

数值 42 不通过 TCP 传输；TCP 只交换控制信息。
普通 RDMA WRITE 不会给 server 自动生成接收 CQE，因此本例用 TCP 通知控制 server 的回读时机。
示例包含错误检查、CQ 超时、socket 收发循环和退出时资源释放。

```bash
make build                         # 只编译主存版，链接 libibverbs
make verbs RDMA_DEVICE=rxe0         # 本机启动双方并验证值为 42
# 双机可分别执行：
make verbs-server RDMA_DEVICE=mlx5_0 GID_INDEX=3
make verbs-client RDMA_DEVICE=mlx5_0 GID_INDEX=3 SERVER=192.168.1.10
```

## Soft-RoCE 配置

以下操作由管理员在支持 `rdma_rxe` 的内核上执行；`eth0` 替换为实际网卡名。
Makefile 不会自动加载模块或改动网卡配置。

```bash
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
ibv_devices
make RDMA_DEVICE=rxe0
make verbs RDMA_DEVICE=rxe0
# 完成后按需清理：
sudo rdma link delete rxe0
```

若内核没有 RXE 模块或网络设备不支持，应换有 RDMA 条件的机器；不以普通 TCP 模拟 RDMA 数据面。
RXE 的结果只用于熟悉 Verbs 流程，不代表真实 HCA 的带宽或延迟。

## 主存注册与 GPUDirect RDMA

`rdma_write.cpp` 用条件编译保留相同的 `ibv_reg_mr` 调用。
主存版分配页对齐 host buffer；显存版使用 `cudaMalloc`，初始数据先拷入显存，然后注册显存地址。
完成通知后，server 才调用 `cudaMemcpy` 回读 GPU buffer 并校验，最后先注销 MR、再释放 allocation。

```bash
# 激活 CUDA 开发环境，编译显存版；真实 GPU/HCA/驱动必须支持 GPUDirect：
make build-cuda CUDA_ARCH=sm_75
make verbs-gpu RDMA_DEVICE=mlx5_0 GID_INDEX=3 GPU=0
# 原文 perftest 的 GPUDirect 双机实验，两机分别执行：
make gdr-server RDMA_DEVICE=mlx5_0 GID_INDEX=3 GPU=0
make gdr-client RDMA_DEVICE=mlx5_0 GID_INDEX=3 GPU=0 SERVER=192.168.1.10
```

perftest 需要编译时包含 CUDA 支持，`ib_write_bw --help` 应包含 `--use_cuda`。
CUDA 构建通过 pkg-config 定位系统 RDMA 库目录，帮助 Conda 链接器查找 libnl 等间接依赖。
若 pkg-config 没有给出目录，可显式指定 `make build-cuda RDMA_LIBDIR=/usr/lib/x86_64-linux-gnu`（Ubuntu x86_64）。
显存直接注册需要硬件、拓扑及驱动配合；本例使用传统 peer-memory 注册方式，
环境通常还需相应的 `nvidia-peermem` 支持。Soft-RoCE 不作为 GPUDirect 验证环境。
缺少支持时会在分配/注册阶段报错，不会悄悄改用主存。

## 多机 NCCL 的 RDMA 日志

目录自带 AllReduce 自检，保留原文两机各两进程的实验：

```bash
# 主节点：
make nccl MASTER_ADDR=192.168.1.10 NODE_RANK=0
# 另一节点：
make nccl MASTER_ADDR=192.168.1.10 NODE_RANK=1
```

`NNODES` 默认 2，`NPROC` 默认每机 2，`MASTER_PORT` 默认 29500。
命令开启 NCCL 的 INIT/NET/GRAPH 日志。必须检查实际的 NET/IB/传输日志；
AllReduce 校验通过本身不能证明使用 RDMA，NCCL 也可能选择 Socket。

## 验证状态与参考

2026-10-26：主存版和 CUDA 显存版均编译通过（测试用 libibverbs 39.0 依赖仅在 `/tmp` 解包）。
本会话没有 RDMA 设备和 GPU；RDMA 数据传输、perftest、GPUDirect 与多机 NCCL 未实测。
编译产物、日志及测试用依赖不进入仓库。

- [rdma-core 的 RC 示例](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/examples/rc_pingpong.c)
- [ibv_modify_qp 状态与属性](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_modify_qp.3)
- [NVIDIA GPUDirect RDMA 同步说明](https://docs.nvidia.com/cuda/gpudirect-rdma/)

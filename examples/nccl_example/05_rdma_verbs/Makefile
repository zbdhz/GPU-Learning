# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 默认执行 run，也就是原文的 perftest 功能实验。
.DEFAULT_GOAL := run
# Python 负责自动启动 server/client 并管理日志和超时。
PYTHON ?= python
# 主存 Verbs 程序使用 C++ 编译器；也可指定已配置环境中的编译器路径。
CXX ?= g++
# 显存版本需要 nvcc；纯主存实验不要求 CUDA。
NVCC ?= nvcc
# -O2 优化，-Wall/-Wextra 开启常见警告，-std=c++17 选择语言标准；已有环境值会被保留。
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17
# 预处理选项默认留空；可用来增加 -I/path/to/include 指定额外头文件目录。
CPPFLAGS ?=
# 链接选项默认留空；可通过 -L 指定库目录或通过链接器选项设置运行库路径。
LDFLAGS ?=
# -l 后面是库名，-libverbs 的含义是链接名为 ibverbs 的库，而不是名为 libverbs 的库。
LDLIBS ?= -libverbs
# Conda nvcc 的链接器也需找到系统 libibverbs 的 libnl 间接依赖。
# 从 libibverbs 的 pkg-config 配置查询库目录；屏蔽查询错误，未找到时得到空值，可手动覆盖。
RDMA_LIBDIR ?= $(shell pkg-config --variable=libdir libibverbs 2>/dev/null)
# 显存版本的默认 GPU 架构是原机使用的 sm_75。
CUDA_ARCH ?= sm_75
# RDMA_DEVICE 默认不填，让工具选择设备；可指定 mlx5_0 或 rxe0 等 HCA 名称。
RDMA_DEVICE ?=
# HCA 端口号默认 1，与 TCP 控制端口不是同一个概念。
IB_PORT ?= 1
# GID 表索引默认 0，但必须按实际网络选择；-1 表示本例的 IB/LID 路径。
GID_INDEX ?= 0
# RDMA 示例和 perftest 双方使用的 TCP 控制端口默认 29877。
PORT ?= 29877
# 双机 client 的 server 地址默认留空，调用 client 时必须显式填写。
SERVER ?=
# 显存实验默认选择本进程可见的 GPU 0；主存实验不使用此变量。
GPU ?= 0
# make 的 if 函数：设备名非空时生成 --device 选项，否则不传，让程序自行选择。
DEVICE_ARG = $(if $(RDMA_DEVICE),--device $(RDMA_DEVICE),)
# 组合自写 Verbs 程序和 Python 启动器共用的设备、端口与 GID 参数。
VERBS_ARGS = $(DEVICE_ARG) --port $(PORT) --ib-port $(IB_PORT) --gid-index $(GID_INDEX)
# 同一个设备名在 perftest 中用 -d 传入，与自写程序的 --device 语法不同。
PERF_DEVICE = $(if $(RDMA_DEVICE),-d $(RDMA_DEVICE),)
# 如果 GID_INDEX=-1 就省略 -x，否则生成 perftest 的 GID 选择参数。
PERF_GID = $(if $(filter -1,$(GID_INDEX)),,-x $(GID_INDEX))
# 组合 perftest 参数：HCA 端口、TCP 端口、忽略频率检查、Gbit/s 单位、100 轮和 64 KiB 数据。
PERF_ARGS = $(PERF_DEVICE) -i $(IB_PORT) $(PERF_GID) -p $(PORT) -F --report_gbits -n 100 -s 65536
# 多机 NCCL 默认每台机器启动 2 个进程。
NPROC ?= 2
# 多机 NCCL 默认共有 2 台机器。
NNODES ?= 2
# 本机节点编号默认 0；另一台机器应指定 NODE_RANK=1。
NODE_RANK ?= 0
# NCCL rendezvous 的主节点地址没有默认值，需要运行时指定。
MASTER_ADDR ?=
# 主节点的 rendezvous TCP 端口默认 29500，各节点应一致。
MASTER_PORT ?= 29500
# 把所有入口标记为动作目标，不需要生成同名文件。
.PHONY: run build build-cuda verbs verbs-server verbs-client verbs-gpu perftest server client gdr-server gdr-client nccl check clean help
# 单机 real HCA loopback 或已配置的 Soft-RoCE/RXE。
# run 把默认行为交给 perftest 目标，不能在没有 RDMA/RXE 条件时期待成功。
run: perftest
# 定义本机双进程带宽工具实验入口。
perftest:
# 自动启动 perftest 的 server/client；run_pair.py 管理就绪等待和退出清理。
	$(PYTHON) run_pair.py --tool perftest $(VERBS_ARGS)
# build 只编译主存 Verbs 程序，不发起网络实验。
build: build/rdma_write
# 定义主存可执行文件的依赖，源文件更新后 make 会重新编译。
build/rdma_write: rdma_write.cpp
# 创建本地 build 输出目录，已存在也可继续。
	mkdir -p build
# 编译并链接 C++ 源码；$< 是源文件，$@ 是可执行文件路径，各选项变量由 make 展开。
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
# build-cuda 只编译带 CUDA 支持的版本。
build-cuda: build/rdma_write_cuda
# 显存版依赖同一份 rdma_write.cpp，通过宏选择是否包含 CUDA 分支。
build/rdma_write_cuda: rdma_write.cpp
# 确保输出目录存在。
	mkdir -p build
# -DUSE_CUDA 让 #ifdef 分支生效；如查询到系统库目录，再传入 -L 和 -rpath-link 帮链接器找到 libnl 间接依赖。
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) -DUSE_CUDA $(CPPFLAGS) $< $(LDFLAGS) $(if $(RDMA_LIBDIR),-L$(RDMA_LIBDIR) -Xlinker -rpath-link -Xlinker $(RDMA_LIBDIR),) $(LDLIBS) -o $@
# verbs 在运行前先准备主存版程序。
verbs: build
# 调用自动配对脚本，使用自写 Verbs 程序完成一次真实 RDMA WRITE 校验。
	$(PYTHON) run_pair.py --tool verbs $(VERBS_ARGS)
# verbs-gpu 先准备带 CUDA 支持的程序，需要硬件/驱动允许显存直接注册。
verbs-gpu: build-cuda
# 在自动配对启动时指定 GPU，双方都用相应显存 buffer；不支持时会失败。
	$(PYTHON) run_pair.py --tool verbs $(VERBS_ARGS) --cuda $(GPU)
# verbs-server 是手动双机 Verbs 实验的服务端入口。
verbs-server: build
# 运行自己的 server，打开设备、注册内存并等待 client 交换配置。
	./build/rdma_write --server $(VERBS_ARGS)
# verbs-client 是手动双机 Verbs 实验的客户端入口。
verbs-client: build
# test -n 检查 SERVER 非空；未指定地址就输出提示并以 1 退出，避免连接错误主机。
	@test -n "$(SERVER)" || { echo '请设置 SERVER=<服务端IP>'; exit 1; }
# 连接指定 server，双方就绪后由 client 投递一次 RDMA WRITE。
	./build/rdma_write --connect $(SERVER) $(VERBS_ARGS)
# server 是原文 perftest 的服务端入口，不依赖自写 C++ 程序。
server:
# ib_write_bw 没有最后的 IP 参数时作为 server 运行。
	ib_write_bw $(PERF_ARGS)
# client 是 perftest 的客户端入口。
client:
# 先检查用户是否明确指定了 server 地址。
	@test -n "$(SERVER)" || { echo '请设置 SERVER=<服务端IP>'; exit 1; }
# 命令最后加 SERVER IP，即让 ib_write_bw 以 client 角色发起测试。
	ib_write_bw $(PERF_ARGS) $(SERVER)
# gdr-server 用于 GPUDirect RDMA 的 perftest 显存服务端。
gdr-server:
# --use_cuda 指定 GPU 显存 buffer；这还要求 perftest 编译了 CUDA 支持。
	ib_write_bw $(PERF_ARGS) --use_cuda=$(GPU)
# gdr-client 是 GPUDirect 的 perftest 显存客户端。
gdr-client:
# 同样要求显式填写服务端 IP，避免误运行。
	@test -n "$(SERVER)" || { echo '请设置 SERVER=<服务端IP>'; exit 1; }
# 客户端同时指定显存 GPU 和服务端 IP；这不是自动回退到主存的接口。
	ib_write_bw $(PERF_ARGS) --use_cuda=$(GPU) $(SERVER)
# nccl 入口复现两机的 PyTorch AllReduce，并开启网络选择日志。
nccl:
# 先检查主节点地址存在，两个节点必须使用相同 MASTER_ADDR 和不同 NODE_RANK。
	@test -n "$(MASTER_ADDR)" || { echo '请设置 MASTER_ADDR=<主节点IP>，两台机器分别设置 NODE_RANK=0/1'; exit 1; }
# 下面三行是一条 shell 命令，末尾反斜杠用于续行，中间不能插注释。
# 第一行开启 NCCL 的初始化/网络/拓扑日志，并使用 Python 启动 torchrun。
# 第二行传入节点总数、本机节点编号和每机进程数。
# 第三行传入 rendezvous 地址端口和自检脚本，并显式使用 CUDA/NCCL。
	NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH $(PYTHON) -m torch.distributed.run \
	  --nnodes=$(NNODES) --node_rank=$(NODE_RANK) --nproc_per_node=$(NPROC) \
	  --master_addr=$(MASTER_ADDR) --master_port=$(MASTER_PORT) all_reduce.py --device cuda
# check 入口只检查工具和 RDMA 设备信息，不执行数据传输。
check:
# command -v 检查 PATH 中是否有 ib_write_bw；缺少就提示安装 perftest。
	@command -v ib_write_bw || { echo '缺少 perftest'; exit 1; }
# 列出系统识别到的 RDMA 设备名。
	ibv_devices
# 显示各 RDMA 设备与端口的详细属性，用来核对 ACTIVE 状态等前提。
	ibv_devinfo
# clean 入口清理本实验生成文件。
clean:
# 删除本目录 build，包括两种可执行文件和自动运行日志，不删除源码。
	rm -rf build
# help 入口只显示常用目标和参数。
help:
# 提示本机默认工具实验、自写主存 Verbs 和显存注册入口。
	@echo 'make: 单机 ib_write_bw；make verbs: 九步 Verbs 实验；make verbs-gpu: 显存注册'
# 提示两类程序的双机 server/client 命令，以及客户端必需的地址参数。
	@echo '双机：server/client 或 verbs-server/verbs-client；client 需 SERVER=<IP>'
# 提示 GPUDirect 和两机 NCCL 的启动入口。
	@echo 'GPUDirect：gdr-server/gdr-client；NCCL：nccl MASTER_ADDR=<IP> NODE_RANK=0/1'
# 提示可调整的设备、端口、GPU 等参数，并指向本目录 README。
	@echo '参数：RDMA_DEVICE、IB_PORT、GID_INDEX、PORT、GPU；硬件和 RXE 配置见 README.md'

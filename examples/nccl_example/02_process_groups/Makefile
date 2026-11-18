# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 不写目标名时执行 run，一次运行本篇所有实验。
.DEFAULT_GOAL := run
# 默认使用已激活环境的 python，可用 PYTHON=/path/to/python 覆盖。
PYTHON ?= python
# 普通实验默认自动选设备；显式 cuda 时不回退 CPU。
DEVICE ?= auto
# 自检和原语实验默认两进程；四 rank 子组的数量在其命令里单独固定。
NPROC ?= 2
# 准备单机 torchrun 命令；同一脚本会在每个进程中分别执行。
LAUNCH = $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=$(NPROC)

# 两张 GPU 的原机也能完整运行四 rank 子组实验；CUDA 子组需四卡。
# 子组实验默认四个 CPU 进程，因此双 GPU 原机也能跑；四卡时可指定 cuda。
GROUP_DEVICE ?= cpu
# 将这些入口声明为操作目标，不按同名文件是否存在决定跳过。
.PHONY: run hello groups primitives debug help
# 定义总入口，下面三条子 make 按顺序执行。
run:
# 先进行设备、rank 和 AllReduce 自检。
	$(MAKE) hello
# 再进行两个互相独立子组的实验。
	$(MAKE) groups
# 最后依次验证各项通信原语。
	$(MAKE) primitives
# hello 入口只运行环境自检。
hello:
# 启动 NPROC 个进程进行自检，并传入普通实验使用的 DEVICE。
	$(LAUNCH) exp0_hello.py --device $(DEVICE)
# groups 入口运行原文要求的四 rank 子组。
groups:
# 固定启动四个进程，GROUP_DEVICE 控制它们用 CPU 还是四张 GPU。
	$(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=4 exp1_groups.py --device $(GROUP_DEVICE)
# primitives 入口只运行集合通信与点对点实验。
primitives:
# 用 NPROC 和 DEVICE 启动六项通信原语验证。
	$(LAUNCH) exp2_primitives.py --device $(DEVICE)
# debug 入口帮助观察 NCCL 在自检时如何初始化。
debug:
# 仅对这次自检开启 NCCL INFO 日志；不改变永久环境设置。
	NCCL_DEBUG=INFO $(LAUNCH) exp0_hello.py --device $(DEVICE)
# help 入口只显示可用命令。
help:
# 提示单项实验以及四卡子组用法，避免把 NPROC=2 误用于固定四 rank 的实验。
	@echo 'make [DEVICE=auto|cpu|cuda]；单项：hello groups primitives debug；四卡子组：make groups GROUP_DEVICE=cuda'

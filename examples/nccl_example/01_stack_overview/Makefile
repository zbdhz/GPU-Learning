# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 直接执行 make 时默认选 run 目标。
.DEFAULT_GOAL := run
# 设置 Python 解释器默认值，可通过命令行指定完整解释器路径。
PYTHON ?= python
# 默认让脚本自动选择 CUDA/NCCL 或 CPU/Gloo；可覆盖为 cpu 或 cuda。
DEVICE ?= auto
# 默认启动两个本机进程，CUDA 模式要求每个进程各有一张可见 GPU。
NPROC ?= 2
# 拼好 torchrun 启动命令；--standalone 自动建立单机 rendezvous，nproc 指定进程数。
LAUNCH = $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=$(NPROC)

# run、debug、help 都是动作名称，不需要生成同名文件。
.PHONY: run debug help
# 定义默认运行目标。
run:
# 启动 NPROC 个进程执行相同脚本，并把设备选项传给每个进程。
	$(LAUNCH) all_reduce.py --device $(DEVICE)
# 定义带 NCCL 日志的运行目标。
debug:
# 仅为本条命令设置 NCCL_DEBUG=INFO；CPU/Gloo 模式不会因此变成 NCCL。
	NCCL_DEBUG=INFO $(LAUNCH) all_reduce.py --device $(DEVICE)
# 定义帮助目标，不进行任何通信。
help:
# 打印可以覆盖的变量和 debug 入口；@ 使输出只保留帮助内容。
	@echo 'make [DEVICE=auto|cpu|cuda] [NPROC=2] [PYTHON=python]；make debug 查看 NCCL 日志'

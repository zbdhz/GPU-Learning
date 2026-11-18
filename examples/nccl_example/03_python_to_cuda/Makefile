# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 默认运行 run，依次完成计时和 profiler。
.DEFAULT_GOAL := run
# Python 默认取当前环境的解释器，也能用命令行变量覆盖。
PYTHON ?= python
# 设备默认 auto，脚本会根据 CUDA 可用性选择通信后端。
DEVICE ?= auto
# 默认两进程；CUDA 模式下每个进程需要一张独立 GPU。
NPROC ?= 2
# 拼接单机多进程命令，--standalone 自动选择 rendezvous 配置。
LAUNCH = $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=$(NPROC)

# 这些名字都表示操作，防止同名文件干扰 make 的判断。
.PHONY: run timing profiler debug clean help
# run 是本篇的一键入口，按顺序调用两个实验。
run:
# 先做首轮与后续轮次耗时测量。
	$(MAKE) timing
# 计时成功后，再做预热和 profiler 录制。
	$(MAKE) profiler
# 定义单独的计时入口。
timing:
# 启动 exp3_timing.py，六轮计时由脚本内部控制。
	$(LAUNCH) exp3_timing.py --device $(DEVICE)
# 定义单独的 profiler 入口。
profiler:
# 启动 exp4_profiler.py，每个 rank 生成自己的 trace 文件。
	$(LAUNCH) exp4_profiler.py --device $(DEVICE)
# 定义日志观察入口，用小型自检触发真实 NCCL 初始化。
debug:
# 开启 INFO 日志，并只选择初始化、拓扑、P2P、网络相关子系统输出；不做日志过滤或伪造。
	NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET $(LAUNCH) exp0_hello.py --device $(DEVICE)
# 定义清理入口，执行前可先确认本目录 build 内容。
clean:
# 递归删除本目录生成的 build；只包含 trace 等输出，不是源码目录。
	rm -rf build
# 定义帮助入口。
help:
# 打印子目标名称和 trace 的输出位置。
	@echo 'make [DEVICE=auto|cpu|cuda]；单项：timing profiler debug；trace 在 build/'

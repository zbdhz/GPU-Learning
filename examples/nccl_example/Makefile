# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 不指定目标直接执行 make 时，默认只运行前三篇的 CPU 实验。
.DEFAULT_GOAL := cpu
# 默认使用当前环境的 python；请先激活含 PyTorch 的开发环境。
PYTHON ?= python
# 声明这些名字是操作目标，而不是同名文件，避免已有文件导致 make 跳过命令。
.PHONY: cpu gpu rdma clean help
# cpu 目标把三个实验按下面的顺序执行；前一步失败就不会继续。
cpu:
# 进入第 1 篇目录执行 make，明确选择 CPU/Gloo 模式。
	$(MAKE) -C 01_stack_overview DEVICE=cpu
# 进入第 2 篇目录，普通实验和四 rank 子组都用 CPU。
	$(MAKE) -C 02_process_groups DEVICE=cpu GROUP_DEVICE=cpu
# 进入第 3 篇目录，以 CPU 模式运行计时和 profiler。
	$(MAKE) -C 03_python_to_cuda DEVICE=cpu
# gpu 目标运行 GPU 版实验，需要对应 GPU 和驱动条件。
gpu:
# 第 1 篇明确选择 CUDA；不可用时应失败，不自动退回 CPU。
	$(MAKE) -C 01_stack_overview DEVICE=cuda
# 第 2 篇普通实验使用 CUDA；子组仍采用该目录的默认 CPU，四卡时可另传 GROUP_DEVICE=cuda。
	$(MAKE) -C 02_process_groups DEVICE=cuda
# 第 3 篇用 CUDA 计时并录制 GPU 活动。
	$(MAKE) -C 03_python_to_cuda DEVICE=cuda
# 第 4 篇编译和执行 P2P、IPC，额外需要两张卡之间有 P2P 能力。
	$(MAKE) -C 04_p2p_ipc
# rdma 目标只负责转入第 5 篇，硬件或 RXE 环境应提前准备好。
rdma:
# 第 5 篇默认运行本机 ib_write_bw 的双方，输出功能测试带宽。
	$(MAKE) -C 05_rdma_verbs
# clean 目标清理生成文件，不删除任何示例源码。
clean:
# 清理第 3 篇导出的 profiler trace。
	$(MAKE) -C 03_python_to_cuda clean
# 清理第 4 篇 CUDA 可执行文件和 holder 日志。
	$(MAKE) -C 04_p2p_ipc clean
# 清理第 5 篇 RDMA 可执行文件和 server 日志。
	$(MAKE) -C 05_rdma_verbs clean
# help 目标只打印入口说明，不启动实验。
help:
# @ 隐藏 echo 命令本身，只显示后面的中文帮助文字。
	@echo 'make cpu：前三篇 CPU 实验；make gpu：前三篇 CUDA+第四篇；make rdma：第五篇 perftest'

# 阅读提示：target: dependency 表示运行目标前，先准备其依赖；下一行以 Tab 开头的是 shell 命令。
# ?= 表示尚未设置时才提供默认值；命令行可用 make 变量名=值 覆盖，例如 make PYTHON=python3。
# $(变量名) 由 make 展开；$(MAKE) 用于调用子 make，能继承命令行指定的变量。
# 说明注释放在命令前；命令本身的 Tab 和续行反斜杠都保留，不能随意改成空格。
# 默认进入 run，先编译再按顺序执行两个 CUDA 实验。
.DEFAULT_GOAL := run
# 使用当前开发环境中的 nvcc 编译器；找不到时应先配置 CUDA 环境。
NVCC ?= nvcc
# Python 用来运行自动启动 holder/writer 的辅助脚本。
PYTHON ?= python
# sm_75 对应原机 RTX 2080 Ti；换 GPU 架构时可以覆盖这个变量。
CUDA_ARCH ?= sm_75
# IPC 使用的本机 TCP 控制端口，双方必须一致；它不传输实际显存数值。
PORT ?= 29777
# -O2 开启优化，-lineinfo 保留源码行信息，-std=c++17 选择 C++17 语言标准。
NVCCFLAGS ?= -O2 -lineinfo -std=c++17
# 声明操作目标；build 虽然也是目录名，但此目标应每次检查依赖。
.PHONY: run build p2p ipc holder writer clean help
# run 先依赖 build；编译成功后才执行下面的运行命令。
run: build
# 先运行单进程跨 GPU P2P 实验。
	$(MAKE) p2p
# 再运行需要 holder/writer 两个进程配合的 IPC 实验。
	$(MAKE) ipc
# build 总目标要求两个可执行文件都已准备好。
build: build/exp5_p2p build/exp6_ipc
# % 是模式匹配，例如 build/exp5_p2p 由 exp5_p2p.cu 和公共头文件生成。
build/%: %.cu cuda_check.h
# 创建输出目录；-p 表示目录已存在也不报错。
	mkdir -p build
# $< 表示第一个依赖（.cu 源文件），$@ 表示目标路径；-arch 选择 GPU 指令架构，-o 指定输出文件。
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $< -o $@
# p2p 在运行前会先检查对应程序是否需要重新编译。
p2p: build/exp5_p2p
# 执行程序并保存退出码；$$ 让美元符号传给 shell，$$? 取上一命令状态；77 显示 SKIP，其他状态原样返回。
	@./build/exp5_p2p; rc=$$?; if [ $$rc -eq 77 ]; then echo 'P2P: SKIP'; else exit $$rc; fi
# ipc 同样先依赖编译好的 exp6_ipc。
ipc: build/exp6_ipc
# 用 Python 自动启动双方并设置端口；退出码 77 显示跳过，真实失败仍使 make 失败。
	@$(PYTHON) run_ipc.py --port $(PORT); rc=$$?; if [ $$rc -eq 77 ]; then echo 'IPC: SKIP'; else exit $$rc; fi
# holder 入口用于手动双终端模式，先保证程序已经编译。
holder: build/exp6_ipc
# 只运行持有方，让它分配显存、导出 IPC 句柄并等待 writer。
	./build/exp6_ipc holder $(PORT)
# writer 入口用于第二个终端，也先检查是否需要编译。
writer: build/exp6_ipc
# 只运行写入方，连接到同一端口的 holder 后导入句柄并写入。
	./build/exp6_ipc writer $(PORT)
# clean 入口用于移除本实验生成的文件。
clean:
# 删除本目录的 build，包含可执行文件与自动运行的 holder 日志。
	rm -rf build
# help 入口用于查看操作方法。
help:
# 打印自动运行、手动双方和可覆盖的构建参数。
	@echo 'make；单项：p2p ipc；手动双终端：holder / writer；可设置 CUDA_ARCH、PORT、NVCC'

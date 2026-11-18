"""本目录独立使用的 torchrun 入口；CPU/Gloo 与 CUDA/NCCL 共用语义。"""
# argparse 用来读取命令行选项，例如 --device cpu。
import argparse
# os 用来读取 torchrun 注入的环境变量，告诉当前进程自己是谁。
import os
# timedelta 表示时间长度，这里用来设置分布式操作的超时时间。
from datetime import timedelta
# torch 提供 Tensor（多维数值数组）和 CUDA 设备接口。
import torch
# 把 PyTorch 的分布式模块简称为 dist，后面通过它建组和通信。
import torch.distributed as dist


# 定义统一初始化函数；desc 是运行 --help 时显示的实验说明。
def lab_init(desc):
    # 创建命令行参数解析器，把实验说明传进去。
    parser = argparse.ArgumentParser(description=desc)
    # 只接受 auto、cpu、cuda 三种设备选项；不填写时使用 auto。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 读取实际命令行，把解析结果放进 args 对象。
    args = parser.parse_args()
    # 取出用户选择的设备模式，暂存为 kind。
    kind = args.device
    # 如果用户让程序自动选设备，就检查当前环境是否能用 GPU。
    if kind == "auto":
        # CUDA 可用时选 GPU，否则选 CPU；这是当前进程的设备选择。
        kind = "cuda" if torch.cuda.is_available() else "cpu"
    # LOCAL_RANK 是本机内的进程编号；缺少它通常表示没有用 torchrun 启动。
    if "LOCAL_RANK" not in os.environ:
        # 打印正确启动方式并退出，而不是继续运行一个不完整的分布式程序。
        parser.error("请通过 torchrun 或本目录 Makefile 启动")
    # 进入 GPU 模式的初始化分支。
    if kind == "cuda":
        # 环境变量原本是字符串，转为整数后才能用作 GPU 编号。
        local_rank = int(os.environ["LOCAL_RANK"])
        # 检查 CUDA 可用性和编号范围，避免多个进程误用同一块卡或访问不存在的卡。
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            # 报告设备条件不足；显式选择 cuda 时不会偷偷改成 CPU。
            parser.error("CUDA 不可用或进程数超过可见 GPU 数；每个进程需要独立 GPU")
        # 设置当前进程默认使用的 GPU；本机 rank 0 用 cuda:0，rank 1 用 cuda:1。
        torch.cuda.set_device(local_rank)
        # 构造设备对象供后续 Tensor 使用；这一步本身不申请显存。
        device = torch.device("cuda", local_rank)
    # 如果不是 GPU 模式，就走 CPU 分支。
    else:
        # 构造 CPU 设备对象，后续 Tensor 会分配在主存里。
        device = torch.device("cpu")
    # NCCL 是本例的 GPU 通信后端；Gloo 是本例的 CPU 通信后端。
    backend = "nccl" if kind == "cuda" else "gloo"
    # 读取 RANK、WORLD_SIZE、MASTER_ADDR 等环境变量建立默认进程组；超时设为 60 秒。
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    # 返回设备、当前进程的全局编号 rank、组内进程总数 world；rank 从 0 开始。
    return device, dist.get_rank(), dist.get_world_size()

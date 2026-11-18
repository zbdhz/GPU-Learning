"""首轮与后续轮次的端到端耗时；不预先调用集合通信。"""
# time 提供高精度计时器，记录一次通信的起止时刻。
import time
# 导入 PyTorch，创建测试用 Tensor 并同步 GPU。
import torch
# 导入分布式模块，用它执行真实的 AllReduce。
import torch.distributed as dist
# 导入当前目录的进程组和设备初始化函数。
from mini_tp import lab_init


# 定义实验入口，测量首轮和后续五轮通信。
def main():
    # 建立默认组；这里不提前调用 AllReduce，让第一轮保留可能的初始化开销。
    dev, rank, world = lab_init("第一次 AllReduce 为什么慢")
    # 用 finally 保证退出时清理通信资源。
    try:
        # 分配 1024×1024 个 float32 的全 1 Tensor，共 4 MiB。
        t = torch.ones(1024, 1024, device=dev)
        # 循环六次，i 从 0 到 5；i=0 是第一轮。
        for i in range(6):
            # 每轮都把输入数值恢复成 1，避免上一轮的和再次参与求和；Tensor 形状不变。
            t.fill_(1)  # 每轮恢复输入，避免上一轮结果参与下一轮求和。
            # CUDA 工作通常异步提交，计时前需要单独处理 GPU。
            if dev.type == "cuda":
                # 等当前设备此前排队的工作全部完成，避免把前面的填充时间算进本轮。
                torch.cuda.synchronize(dev)
            # 读取单调高精度时钟，记为本轮开始时间。
            start = time.perf_counter()
            # 发起逐元素求和；所有 rank 都必须参与这一轮操作。
            dist.all_reduce(t)
            # CUDA 下，Python 接口返回不等于 GPU 已经完成通信。
            if dev.type == "cuda":
                # 等 GPU 完成后再停止计时，使结果包含实际设备执行时间。
                torch.cuda.synchronize(dev)
            # 用结束时刻减开始时刻得到秒数，再乘 1000 换成毫秒。
            elapsed_ms = (time.perf_counter() - start) * 1000
            # 所有输入都是 1，所以归约后每个元素都应为 world；检查不计入已记录耗时。
            torch.testing.assert_close(t, torch.full_like(t, world))
            # 只用 rank 0 的观测值输出六行计时，避免日志重复。
            if rank == 0:
                # 打印轮次、保留三位小数的耗时及使用的后端；首轮不保证一定最慢。
                print(f"round={i}: {elapsed_ms:.3f} ms backend={dist.get_backend()} PASS", flush=True)
    # 无论计时或校验是否出错，都进入清理阶段。
    finally:
        # 释放默认进程组的通信资源。
        dist.destroy_process_group()


# 直接执行脚本时才开始实验。
if __name__ == "__main__":
    # 调用入口函数。
    main()

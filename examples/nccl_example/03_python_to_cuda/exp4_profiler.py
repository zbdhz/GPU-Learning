"""预热后采集一次通信；CUDA 模式录制 GPU 活动，CPU 模式只观察 Gloo。"""
# Path 用来生成与当前脚本位置相关的输出路径，不依赖启动时的工作目录。
from pathlib import Path
# torch 提供 Tensor 和 CUDA 同步接口。
import torch
# dist 提供要被观察的真实 AllReduce 操作。
import torch.distributed as dist
# profile 是录制器；ProfilerActivity 指定观察 CPU/GPU；record_function 添加自定义标记。
from torch.profiler import profile, ProfilerActivity, record_function
# 导入本目录的分布式实验初始化函数。
from mini_tp import lab_init


# 定义 profiler 实验的主入口。
def main():
    # 建立进程组并选择设备，返回当前 rank 和进程总数。
    dev, rank, world = lab_init("用 profiler 观察通信")
    # 用 finally 保证最后释放通信组，即使 profiler 或校验失败。
    try:
        # 创建一个 4 MiB 的 float32 Tensor，作为通信输入。
        t = torch.ones(1024, 1024, device=dev)
        # 先运行三轮预热；下划线表示不需要使用循环编号。
        for _ in range(3):
            # 预热时实际执行 AllReduce，让可能的建组和首次运行开销发生在录制之前。
            dist.all_reduce(t)
        # 预热结束后恢复输入为 1，方便后面验证一次归约的结果。
        t.fill_(1)
        # 默认录制 CPU 活动，例如 Python 下层的 c10d 调用。
        activities = [ProfilerActivity.CPU]
        # 如果当前运行在 GPU 上，再额外开启设备端录制。
        if dev.type == "cuda":
            # 确保预热和 Tensor 填充在正式录制前完成。
            torch.cuda.synchronize(dev)
            # 把 CUDA 活动加入录制范围，才有机会看到 NCCL 的 GPU kernel。
            activities.append(ProfilerActivity.CUDA)
        # 进入 with 时开始录制，离开 with 时结束；prof 保存此次采集结果。
        with profile(activities=activities) as prof:
            # 给下面这段通信起名 lab_all_reduce，方便在时间线上找到它。
            with record_function("lab_all_reduce"):
                # 正式录制的一次 AllReduce；所有 rank 的输入都为 1。
                dist.all_reduce(t)
                # GPU 模式下，还需要等待设备真正完成。
                if dev.type == "cuda":
                    # 同步放在录制范围内，避免 GPU kernel 还没执行完就停录。
                    torch.cuda.synchronize(dev)
        # 录制结束后检查所有元素是否为 world；检查本身不混入录制片段。
        torch.testing.assert_close(t, torch.full_like(t, world))
        # 生成当前实验目录下的 build 路径，用来保存 trace 文件。
        out = Path(__file__).resolve().parent / 'build'
        # 创建输出目录；exist_ok=True 表示目录已存在也不报错。
        out.mkdir(exist_ok=True)
        # 每个 rank 使用不同文件名，避免多个进程覆盖彼此的录制结果。
        trace = out / f"rank{rank}.trace.json"
        # 导出 Chrome tracing 格式的 JSON，可用 Perfetto 等工具查看时间线。
        prof.export_chrome_trace(str(trace))
        # 只让 rank 0 在终端显示统计表，各 rank 的 trace 仍分别保留。
        if rank == 0:
            # CUDA 模式按设备总耗时排序，CPU 模式按 CPU 总耗时排序。
            sort_key = "device_time_total" if dev.type == "cuda" else "cpu_time_total"
            # 把同名事件聚合并显示前 15 行，观察通信调用的耗时分布。
            print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
            # GPU 模式提示检查 trace 内的真实 NCCL kernel，名称可能因版本变化。
            print("CUDA 模式请在 trace 中检查 NCCL kernel（名称随版本变化）。" if dev.type == "cuda"
                  # CPU 模式明确提示没有 NCCL/GPU kernel；flush=True 使提示立即输出。
                  else "CPU/Gloo 模式：不含 NCCL 或 GPU kernel。", flush=True)
        # 每个 rank 输出自己的 trace 路径，让你知道该看哪个文件。
        print(f"rank={rank}: profiler PASS; trace={trace}", flush=True)
    # 无论录制成功还是发生异常，都执行清理。
    finally:
        # 销毁默认进程组，结束通信环境。
        dist.destroy_process_group()


# 只有直接执行此文件时才运行实验。
if __name__ == "__main__":
    # 调用 main，开始预热和录制。
    main()

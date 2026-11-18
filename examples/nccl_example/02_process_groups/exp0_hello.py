"""每个 rank 贡献 rank+1；所有 rank 都验证最终求和结果。"""
# 导入 os，以便显示 torchrun 传入的本机进程编号。
import os
# 导入 PyTorch，用来创建数值 Tensor 并检查结果。
import torch
# 导入分布式通信模块，简写为 dist。
import torch.distributed as dist
# 从当前目录的 mini_tp.py 导入设备选择和进程组初始化函数。
from mini_tp import lab_init


# 把实验主体放入 main，避免文件被 import 时直接启动通信。
def main():
    # 建立进程组；dev 是计算设备，rank 是全局进程编号，world 是总进程数。
    dev, rank, world = lab_init("rank 与 AllReduce 自检")
    # 把通信包在 try/finally 内，保证即使校验失败也会执行清理。
    try:
        # 打印全局编号、总人数和本机编号；这里的 f 字符串会自动填入变量值。
        print(f"rank={rank}/{world} local_rank={os.environ['LOCAL_RANK']} "
              # 接着打印设备和通信后端；flush=True 让多进程日志及时显示。
              f"device={dev} backend={dist.get_backend()}", flush=True)
        # 创建只有一个浮点数的 Tensor；两进程时，rank 0 放 1，rank 1 放 2。
        t = torch.tensor([float(rank + 1)], device=dev)
        # 所有 rank 按相同顺序调用求和操作；结果原地写回各自的 t，即每人都得到 3。
        dist.all_reduce(t)
        # 用等差数列求和公式算出 1+2+...+world，作为正确答案。
        expected = world * (world + 1) / 2
        # 创建与 t 同形状、同设备的期望 Tensor 并比较；不一致时抛异常，让运行失败。
        torch.testing.assert_close(t, torch.full_like(t, expected))
        # t.item() 将单个 Tensor 数值取为 Python 数字；GPU 模式下取值可能需要等待 GPU。
        print(f"rank={rank}: all_reduce={t.item()} expected={expected} PASS", flush=True)
    # 无论 try 内成功还是抛异常，finally 块都会执行。
    finally:
        # 销毁默认进程组，释放本进程持有的通信资源。
        dist.destroy_process_group()


# 只有直接运行这个脚本时才进入下面的代码；被其他文件 import 时不会运行。
if __name__ == "__main__":
    # 调用 main，开始本次 AllReduce 自检。
    main()

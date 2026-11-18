"""四个 rank 以相同顺序建组，组内分别求和 1+2 与 3+4。"""
# 导入 PyTorch，负责创建输入 Tensor 和比较输出。
import torch
# 导入分布式通信模块，负责建子组、求和和释放资源。
import torch.distributed as dist
# 导入当前目录的公共初始化函数。
from mini_tp import lab_init


# 定义本实验入口；先初始化，再演示两个互不混合的子组。
def main():
    # 取得设备 dev、全局进程编号 rank 和总进程数 world。
    dev, rank, world = lab_init("两个独立子组")
    # 先建空列表保存子组；后面即使中途出错，清理代码也有列表可遍历。
    groups = []
    # 使用 try/finally，让子组和默认组都能得到清理。
    try:
        # 实验设计是把四个进程分成两对，因此严格要求总进程数为 4。
        if world != 4:
            # 人数不符就报错，并给出可以直接使用的启动命令。
            raise ValueError("本实验需要四个进程：make groups，或 torchrun --standalone --nproc_per_node=4 exp1_groups.py --device cpu")
        # 所有 rank 必须按相同顺序调用，即使自己不是子组成员。
        # 四个 rank 都依次调用两次 new_group；返回值里会标明当前进程是否属于该组。
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        # rank 0、1 选择第一个子组；rank 2、3 选择第二个子组。
        group = groups[0 if rank < 2 else 1]
        # 每个 rank 创建一个不同的输入：1、2、3、4。
        t = torch.tensor([float(rank + 1)], device=dev)
        # 只在所选子组内求和；第一组算 1+2，第二组算 3+4。
        dist.all_reduce(t, group=group)
        # 按本进程属于哪个子组，选择正确答案 3 或 7。
        expected = 3.0 if rank < 2 else 7.0
        # 比较本进程拿到的结果和期望值，保证每个组的结果都正确。
        torch.testing.assert_close(t, torch.full_like(t, expected))
        # 打印当前 rank 的组内求和结果，方便你把四个进程的日志对应起来。
        print(f"rank={rank}: subgroup_sum={t.item()} expected={expected} PASS", flush=True)
    # 无论实验成功还是失败，都进入资源清理阶段。
    finally:
        # 逐个检查此前创建的子组，释放本进程确实加入的那些组。
        for group in groups:
            # NON_GROUP_MEMBER 表示“我不在这个组里”；不能把它当作真实组销毁。
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                # 销毁本进程加入的子组，释放该组的本地通信资源。
                dist.destroy_process_group(group)
        # 子组清理后，最后销毁默认的全员进程组。
        dist.destroy_process_group()


# 只有直接运行此文件时才启动实验。
if __name__ == "__main__":
    # 进入 main，开始子组实验。
    main()

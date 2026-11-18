"""原文六项实验：broadcast/all_reduce/reduce/all_gather/reduce_scatter/send-recv。"""
# torch 提供 Tensor 的创建、分配和比较接口。
import torch
# dist 提供集合通信和点对点通信接口。
import torch.distributed as dist
# 从本目录导入设备选择和默认进程组初始化函数。
from mini_tp import lab_init


# 定义本实验的入口函数，各项通信按下面的固定顺序执行。
def main():
    # 初始化通信环境，获得设备、本进程的 rank 和总人数 world。
    dev, rank, world = lab_init("集合通信原语与点对点通信")
    # 确保执行结束时，即使发生异常，也会销毁进程组。
    try:
        # 最后的点对点实验需要发送方 rank 0 和接收方 rank 1。
        if world < 2:
            # 人数不足时给出明确错误，避免发送给不存在的进程。
            raise ValueError("send/recv 实验至少需要两个进程")
        # 给每个 rank 一个“身份数”；两进程时分别是 1.0 和 2.0。
        me = float(rank + 1)
        # 计算所有身份数之和，作为后面归约操作的正确答案。
        total = world * (world + 1) / 2

        # 定义显示辅助函数：把每个 rank 的单元素结果汇总后统一打印。
        def report(name, value):
            # 给每个来源 rank 分配一个接收位置；empty_like 只分配空间，初始值不保证为零。
            outs = [torch.empty_like(value) for _ in range(world)]
            # 所有 rank 都拿到大家的 value；调用返回后 outs 按来源 rank 排列。
            dist.all_gather(outs, value)
            # 只由 rank 0 打印全局视图，避免同样的表输出 world 份。
            if rank == 0:
                # enumerate 同时取来源编号和 Tensor，item() 取单个数值；拼成可读的结果行。
                print(f"{name}: " + ", ".join(f"r{i}={x.item():g}" for i, x in enumerate(outs)), flush=True)

        # 广播前，rank 0 放 99，其他 rank 放 0；dev 决定 Tensor 在主存还是显存。
        t = torch.tensor([99.0 if rank == 0 else 0.0], device=dev)
        # broadcast 把 src=0 的值覆盖到全部 rank 的 t 上。
        dist.broadcast(t, src=0)
        # 验证广播后，本进程的 t 确实变成 99。
        torch.testing.assert_close(t, torch.full_like(t, 99))
        # 汇总显示广播结果；report 本身也会进行一次 all_gather。
        report("broadcast", t)

        # 重新创建输入，每个 rank 放回自己的身份数。
        t = torch.tensor([me], device=dev)
        # all_reduce 默认逐元素求和，并把完整结果发给全部 rank，原地覆盖 t。
        dist.all_reduce(t)
        # 验证每个 rank 都得到身份数总和；两进程时是 3。
        torch.testing.assert_close(t, torch.full_like(t, total))
        # 显示所有 rank 的 AllReduce 结果。
        report("all_reduce", t)

        # 再次恢复不同的身份数，为普通 reduce 准备输入。
        t = torch.tensor([me], device=dev)
        # reduce 也求和，但只承诺 dst=0 的 Tensor 中有最终结果。
        dist.reduce(t, dst=0)
        # 只有目标进程可以把自己的 t 当作有效的归约结果检查。
        if rank == 0:
            # rank 0 比较收到的结果是否为 total。
            torch.testing.assert_close(t, torch.full_like(t, total))
            # 提示其他 rank 的 reduce 输出没有结果保证，不能据其值推断语义。
            print(f"reduce: 仅 dst=0 的结果有效：{t.item()}", flush=True)

        # 准备收集实验的输入，每个 rank 放自己的身份数。
        t = torch.tensor([me], device=dev)
        # 分配 world 个单元素 Tensor，分别存放来自每个 rank 的值。
        outs = [torch.empty_like(t) for _ in range(world)]
        # all_gather 收集而不相加；每个 rank 的 outs 都会包含完整列表。
        dist.all_gather(outs, t)
        # 遍历收集结果，peer 是来源 rank，value 是它贡献的 Tensor。
        for peer, value in enumerate(outs):
            # 每个来源 rank 应贡献 peer+1，从而验证列表的内容及顺序。
            torch.testing.assert_close(value, torch.full_like(value, peer + 1))
        # 让 rank 0 作为唯一的汇总打印者。
        if rank == 0:
            # 把各 Tensor 转成 Python 数字并显示；两进程时列表为 [1, 2]。
            print(f"all_gather: {[x.item() for x in outs]}", flush=True)

        # 构造 world 个数；两进程时 rank 0 放 [1,10]，rank 1 放 [2,20]。
        t = torch.tensor([me * 10 ** i for i in range(world)], device=dev)
        # 每个 rank 只分配一个数的输出空间，准备接收归约后的一个分片。
        out = torch.empty(1, device=dev)
        # 先逐位置求和为 [3,30]，再按 rank 分发：rank 0 得 3，rank 1 得 30。
        dist.reduce_scatter_tensor(out, t)
        # 用 total×10的rank次方计算本进程应收到的分片，检查结果。
        torch.testing.assert_close(out, torch.full_like(out, total * 10 ** rank))
        # 把各 rank 分别持有的分片汇总显示。
        report("reduce_scatter", out)

        # 下面进入点对点通信分支：只有 rank 0 负责发送。
        if rank == 0:
            # 把 777 发给全局 rank 1；send 是阻塞接口，需要接收方配合调用 recv。
            dist.send(torch.tensor([777.0], device=dev), dst=1)
        # rank 1 进入接收分支；如果总人数大于 2，其他 rank 跳过这段收发。
        elif rank == 1:
            # 分配一个数的空间用来接收消息；调用 recv 前内容尚未确定。
            value = torch.empty(1, device=dev)
            # 从全局 rank 0 接收数值并写入 value。
            dist.recv(value, src=0)
            # 检查实际收到的消息是否等于约定值 777。
            torch.testing.assert_close(value, torch.full_like(value, 777))
            # 由接收方打印收到的数，直观看到消息的方向。
            print(f"send/recv: rank 1 收到 {value.item()}", flush=True)
        # 所有 rank 到达这里后才继续，避免部分进程过早退出通信环境。
        dist.barrier()
        # 每个 rank 打印通过标记；此前任意检查失败都会抛出异常。
        print(f"rank={rank}: primitives PASS", flush=True)
    # 无论上面是否出错，最后都清理默认进程组。
    finally:
        # 释放本进程的分布式通信资源。
        dist.destroy_process_group()


# 判断当前文件是否被直接执行。
if __name__ == "__main__":
    # 调用 main，按顺序运行全部原语实验。
    main()

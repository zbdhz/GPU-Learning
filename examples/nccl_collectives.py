"""运行：torchrun --standalone --nproc_per_node=2 examples/nccl_collectives.py"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", local_rank)
try:
    x = torch.full((16,), float(rank + 1), device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(x, torch.full_like(x, world * (world + 1) / 2))

    x = torch.full((16,), float(rank + 7), device=device)
    dist.broadcast(x, src=0)
    torch.testing.assert_close(x, torch.full_like(x, 7))

    x = torch.full((16,), float(rank), device=device)
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    for peer, result in enumerate(gathered):
        torch.testing.assert_close(result, torch.full_like(x, peer))

    source = torch.full((world * 16,), float(rank + 1), device=device)
    result = torch.empty(16, device=device)
    dist.reduce_scatter_single(result, source, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(result, torch.full_like(result, world * (world + 1) / 2))
    torch.cuda.synchronize()
    print(f"rank={rank} GPU={local_rank}: all_reduce, broadcast, "
          "all_gather, reduce_scatter: PASS", flush=True)
finally:
    dist.destroy_process_group()

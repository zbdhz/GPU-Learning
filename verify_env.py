"""检查 PyTorch 的实际 GPU 运算；在激活 gpu-dev 后运行。"""
import os
import subprocess

import torch
import torch.distributed as dist

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("NCCL_HOME:", os.environ.get("NCCL_HOME"))
print("Built architectures:", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA 不可用，请先检查 nvidia-smi 和 GPU 设备访问权限"
assert dist.is_nccl_available(), "PyTorch 缺少 NCCL 后端"
assert torch.cuda.device_count() >= 2, "本机应有两张 GPU；请检查 CUDA_VISIBLE_DEVICES"
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {prop.name}, {prop.total_memory / 2**30:.1f} GiB, "
          f"sm_{prop.major}{prop.minor}")
    with torch.cuda.device(i):
        torch.manual_seed(42)
        a = torch.randn(256, 256, device=f"cuda:{i}")
        b = torch.randn(256, 256, device=f"cuda:{i}")
        actual = (a @ b).cpu()
        expected = a.cpu() @ b.cpu()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        torch.cuda.synchronize()
    print(f"GPU {i} matrix multiplication: PASS")
subprocess.run(["nvcc", "--version"], check=True)
print("Environment: PASS")

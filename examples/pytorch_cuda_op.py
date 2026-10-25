"""首次运行会使用 nvcc + Ninja 编译，再与 PyTorch 原生加法比较。"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

here = Path(__file__).resolve().parent
extension = load(
    name="learning_vector_add",
    sources=[str(here / "add_binding.cpp"), str(here / "add_kernel.cu")],
    extra_cflags=["-O2"], extra_cuda_cflags=["-O2", "-lineinfo"],
    verbose=True,
)
for device in range(torch.cuda.device_count()):
    for n in (0, 17, 1024, 1048593):
        a = torch.randn(n, device=f"cuda:{device}")
        b = torch.randn_like(a)
        torch.testing.assert_close(extension.vector_add(a, b), a + b)
    # 自定义 kernel 必须使用 PyTorch 当前 stream。
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        a = torch.randn(4097, device=f"cuda:{device}")
        b = torch.randn_like(a)
        actual = extension.vector_add(a, b)
        torch.testing.assert_close(actual, a + b)
    stream.synchronize()
    print(f"GPU {device}: PyTorch CUDA extension: PASS")
print("本示例演示前向计算；训练用算子还需注册 autograd 和 dispatcher。")

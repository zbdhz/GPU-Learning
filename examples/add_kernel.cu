#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device(), "inputs must be on the same GPU");
  TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
              "this learning example supports float32 only");
  TORCH_CHECK(a.sizes() == b.sizes(), "input shapes must match");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  const c10::cuda::CUDAGuard guard(a.device());
  auto out = torch::empty_like(a);
  if (a.numel() > 0) {
    add_kernel<<<(a.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

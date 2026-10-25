// 编译：nvcc -O2 -arch=sm_75 vector_add.cu -o vector_add
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main() {
  constexpr int n = (1 << 20) + 17;  // 非整 block 大小，检查边界处理。
  std::vector<float> a(n), b(n), out(n);
  for (int i = 0; i < n; ++i) { a[i] = (i % 97) * .25f; b[i] = (i % 31) * .5f; }
  float *da, *db, *dout;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  vector_add<<<(n + 255) / 256, 256>>>(da, db, dout, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (std::fabs(out[i] - (a[i] + b[i])) > 1e-6f) {
      std::fprintf(stderr, "Mismatch at %d\n", i); return 1;
    }
  }
  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dout));
  std::printf("CUDA vector_add: PASS (%d elements)\n", n);
}

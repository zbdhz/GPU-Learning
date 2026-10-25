// 单进程管理两张卡，直接调用 NCCL C API。
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d CUDA: %s\n", \
      __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } \
} while (0)
#define NCCL_CHECK(call) do { \
  ncclResult_t e = (call); \
  if (e != ncclSuccess) { std::fprintf(stderr, "%s:%d NCCL: %s\n", \
      __FILE__, __LINE__, ncclGetErrorString(e)); std::exit(1); } \
} while (0)

int main() {
  constexpr int devices = 2, n = 1024;
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count < devices) { std::fprintf(stderr, "Need two GPUs\n"); return 1; }
  ncclComm_t comm[devices];
  cudaStream_t stream[devices];
  float* data[devices];
  NCCL_CHECK(ncclCommInitAll(comm, devices, nullptr));
  for (int d = 0; d < devices; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaStreamCreate(&stream[d]));
    CUDA_CHECK(cudaMalloc(&data[d], n * sizeof(float)));
    std::vector<float> input(n, float(d + 1));
    CUDA_CHECK(cudaMemcpy(data[d], input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  }
  NCCL_CHECK(ncclGroupStart());
  for (int d = 0; d < devices; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    NCCL_CHECK(ncclAllReduce(data[d], data[d], n, ncclFloat, ncclSum, comm[d], stream[d]));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (int d = 0; d < devices; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaStreamSynchronize(stream[d]));
    std::vector<float> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), data[d], n * sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : output) {
      if (v != 3.0f) { std::fprintf(stderr, "GPU %d wrong result: %f\n", d, v); return 1; }
    }
    std::printf("GPU %d: NCCL C API all_reduce: PASS\n", d);
    CUDA_CHECK(cudaFree(data[d]));
    CUDA_CHECK(cudaStreamDestroy(stream[d]));
    NCCL_CHECK(ncclCommDestroy(comm[d]));
  }
}

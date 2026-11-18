// 头文件只处理一次；多个源文件重复包含它时，避免重复定义。
#pragma once
// 引入 CUDA Runtime API，获得设备查询、显存分配及 cudaError_t 等声明。
#include <cuda_runtime.h>
// 引入标准异常类，用来报告无法继续执行的错误。
#include <stdexcept>
// 引入 std::string，用来拼接错误信息。
#include <string>

// inline 允许此函数放在头文件里；status 是 CUDA 返回码，call 是原调用的文字。
inline void cuda_check(cudaError_t status, const char* call) {
  // cudaSuccess 表示成功；如果返回的不是它，就进入错误分支。
  if (status != cudaSuccess)
    // 把“哪个调用失败”和 CUDA 的可读错误描述一起抛出，由 main 中的 catch 打印。
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
// 宏同时执行调用并检查结果；#call 把调用表达式转成文字，便于报错时定位。
#define CUDA_CHECK(call) cuda_check((call), #call)

// 定义查询函数：本实验需要 GPU 0 能直接访问 GPU 1 的显存。
inline bool has_peer_path() {
  // count 存可见 GPU 数量，can 存能否 P2P 访问；先初始化为 0。
  int count = 0, can = 0;
  // 查询可见 GPU 数；&count 是把变量地址交给 CUDA，让它写入查询结果。
  CUDA_CHECK(cudaGetDeviceCount(&count));
  // 连两张 GPU 都没有时直接报错，不能执行本例的跨卡实验。
  if (count < 2) throw std::runtime_error("需要两张可见的 GPU");
  // 查询访问方向 0→1；P2P 能力有方向性，因此不能用反向结果代替。
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
  // can 不为 0 就返回 true；这个查询只说明 P2P 能力，不说明一定有 NVLink。
  return can != 0;
}

// ============================================================================
// 1.hello — 第一个 CUDA 程序：Hello World
// 参考：AIInfraGuide《1.1 CUDA 开发环境搭建》第 7 节
//
// 学习目标：
//   1. 认识 kernel 函数的三种修饰符（__global__ / __device__ / __host__）
//   2. 理解 <<<Grid, Block>>> 启动配置
//   3. 掌握 threadIdx / blockIdx 内置索引变量
//   4. 学会用 cudaGetLastError / cudaDeviceSynchronize 做错误检查
// ============================================================================

#include <cstdio>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// 设备查询辅助函数：打印当前 GPU 的基本信息
// 在写任何 CUDA 程序前，先了解硬件（SM 数量、计算能力）是好习惯
// ----------------------------------------------------------------------------
static void printDeviceInfo() {
    int deviceCount = 0;
    // cudaGetDeviceCount：查询机器上有几块可用 GPU
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("[错误] 未检测到可用 GPU: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    // cudaGetDeviceProperties：获取 0 号 GPU 的详细属性
    cudaGetDeviceProperties(&prop, 0);
    printf("设备名称       : %s\n", prop.name);
    printf("SM 数量       : %d\n", prop.multiProcessorCount);
    printf("计算能力      : %d.%d\n", prop.major, prop.minor);
    printf("显存容量      : %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("每 Block 最大线程数: %d\n\n", prop.maxThreadsPerBlock);
}

// ----------------------------------------------------------------------------
// Kernel 函数：在 GPU 上执行的代码
//
// __global__ 修饰符的三个含义：
//   1. 函数运行在 GPU（设备）上
//   2. 只能从 CPU（主机）侧调用（即"启动"一个 kernel）
//   3. 返回类型必须是 void
//
// 内置索引变量（每个线程各自持有一份，值互不相同）：
//   threadIdx.x —— 当前线程在其所在 Block 内的编号
//   blockIdx.x  —— 当前 Block 在整个 Grid 内的编号
//   blockDim.x  —— 一个 Block 里有多少线程（本例 = 4）
//   gridDim.x   —— Grid 里一共有多少 Block（本例 = 2）
// ----------------------------------------------------------------------------
__global__ void helloKernel() {
    // GPU 上的 printf 辇输出到主机终端。
    // 注意：输出顺序不保证！2 个 Block 并行执行，谁先完成谁先打印
    printf("Hello from GPU thread %d in block %d!\n",
           threadIdx.x, blockIdx.x);
}

// 一个演示 __device__ 函数的小例子：
// __device__ 表示"只能在 GPU 上被调用"的工具函数（不能加 <<<>>> 启动）
__device__ int globalThreadIdx() {
    // 全局线程编号 = Block 编号 × 每 Block 线程数 + Block 内编号
    // 这是"每个线程处理一个数据元素"的标准写法
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// 第二个 kernel：演示全局线程编号的计算（配合上面的 __device__ 函数）
__global__ void globalIdKernel() {
    int gid = globalThreadIdx();
    printf("  block %d, thread %d -> 全局编号 %d\n",
           blockIdx.x, threadIdx.x, gid);
}

int main() {
    printf("===== 1. 设备信息 =====\n");
    printDeviceInfo();

    // ------------------------------------------------------------------
    // 启动 kernel：helloKernel<<<Grid 大小, Block 大小>>>()
    //   第 1 个参数：Grid 里的 Block 数量（2 个）
    //   第 2 个参数：每个 Block 的线程数量（4 个）
    // 本例共启动 2 × 4 = 8 个 GPU 线程
    // ------------------------------------------------------------------
    printf("===== 2. Hello Kernel（<<<2, 4>>>：8 个线程）=====\n");
    helloKernel<<<2, 4>>>();

    // 错误检查（重要习惯）：
    // kernel 启动是异步的，启动失败（如配置非法）不会立刻报错，
    // 必须显式查询
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[错误] kernel 启动失败: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // CPU 在此阻塞，等 GPU 把队列里的所有任务（含 printf）执行完
    // 没有这一行，main 可能在 GPU 打印前就退出了
    cudaDeviceSynchronize();

    printf("\n===== 3. 全局线程编号（同一 Grid：编号 0~7）=====\n");
    globalIdKernel<<<2, 4>>>();
    cudaDeviceSynchronize();  // 同样需要同步等待

    printf("\n===== 4. 完成提示（对照教程 7.2 节预期输出）=====\n");
    printf("Done!\n");
    return 0;
}

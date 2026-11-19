// ============================================================================
// 3.vector_add — 第一个实用 Kernel：向量加法全流程
// 教程对应：AIInfraGuide《1.4 第一个实用 Kernel：向量加法》
//
// 覆盖教程全部内容：
//   Part 1  五步标准流程（cudaMalloc → H2D → Kernel → D2H → cudaFree）
//           + CPU vs GPU 性能对比 + CUDA Event 计时 + 有效带宽计算
//   Part 2  进阶练习 3：Grid-stride Loop 版本（固定线程数处理任意 N）
//   Part 3  进阶练习 2：SAXPY（Y = a*X + Y，BLAS 基础操作）
//   Part 4  进阶练习 1：向量点积（Block 内共享内存规约 + atomicAdd）
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// 错误检查宏（教程 4.1 节）：生产代码必须检查每个 CUDA 调用的返回值
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// ============================================================================
// Part 1: 基础版向量加法（教程 2.1 节）
// ============================================================================

// 每个线程独立处理一个元素：天然并行、无需通信
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 全局线程索引 = Block 编号 × Block 大小 + Block 内编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查（教程 2.2 节）：N=1000、block=256 时 grid=4 共 1024 线程，
    // 多出的 24 个线程 idx >= N，不检查就会越界写显存（未定义行为）
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU 基线（教程 5.1 节）：单核循环，用于对比加速比
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

// ============================================================================
// Part 2: Grid-stride Loop 版（教程练习 8.3）
// ============================================================================

// 固定线程数处理任意大的 N：每个线程以"总线程数"为步长循环跳跃
__global__ void vectorAddStride(const float* A, const float* B,
                                float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;      // 步长 = 总线程数
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
    // 好处：①任意 N 都正确 ②Grid 大小可独立调优（不与 N 绑定）
    // ③线程复用率高（长 N 时比"一线程一元素"省调度开销）
}

// ============================================================================
// Part 3: SAXPY（教程练习 8.2）：Y[i] = alpha * X[i] + Y[i]
// ============================================================================

// 比向量加法多一个标量乘法。alpha 是统一参数（所有线程相同），
// 标量放寄存器里零成本；X/Y 是向量必须访存 —— 依旧内存密集型
__global__ void saxpy(float alpha, const float* X, float* Y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = alpha * X[idx] + Y[idx];    // 注意：Y 既是输入又是输出（原地更新）
    }
}

// ============================================================================
// Part 4: 向量点积（教程练习 8.1）：result = Σ A[i]*B[i]
// ============================================================================

// 点积无法"一线程一元素"：结果只有一个数，必须把所有部分和累加起来。
// 标准做法是两级规约（Reduction）：
//   第一级：Block 内 256 个线程的部分和 → 共享内存树形规约 → 1 个数
//   第二级：每个 Block 的结果 atomicAdd 到全局累加器（原子操作防竞争）
__global__ void dotProduct(const float* A, const float* B,
                           float* result, int N) {
    extern __shared__ float sdata[];         // 动态共享内存（大小启动时指定）

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;

    // 步骤 1：每个线程先算自己的部分和（Grid-stride 累加多个元素）
    float partial = 0.0f;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        partial += A[i] * B[i];
    }
    sdata[tid] = partial;
    __syncthreads();                          // 等 Block 内所有线程写完

    // 步骤 2：树形规约 —— 每轮活跃线程数减半（256→128→...→1）
    // 例（8 线程）：[a b c d e f g h] → [a+c b+d e+g f+h] → ... → [总和]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();                      // 每轮都要同步，防止读写竞态
    }

    // 步骤 3：tid==0 的线程代表本 Block，把结果原子累加到全局变量
    // atomicAdd：读-改-写三步不可分割，多个 Block 同时加也不会丢更新
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ----------------------------------------------------------------------------
// CUDA Event 计时封装（教程 5.2 节）：GPU 侧打点，微秒精度
// 用法：eventTimerBegin(...) → 要测的 kernel → eventTimerEnd() 返回毫秒
// ----------------------------------------------------------------------------
static cudaEvent_t g_evBeg, g_evEnd;
static void eventTimerInit() {
    CUDA_CHECK(cudaEventCreate(&g_evBeg));
    CUDA_CHECK(cudaEventCreate(&g_evEnd));
}
static float eventTimerEnd() {
    float ms;
    CUDA_CHECK(cudaEventRecord(g_evEnd));
    CUDA_CHECK(cudaEventSynchronize(g_evEnd));
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_evBeg, g_evEnd));
    return ms;
}

int main() {
    // N = 2^24 ≈ 1600 万元素（与教程 5.4 节参考值同规模）
    const int N = 1 << 24;
    const size_t size = N * sizeof(float);
    printf("向量长度: %d 元素 (%.1f MB / 个向量)\n", N, size / (1024.0 * 1024.0));

    // ========== 主机内存分配与初始化 ==========
    float* h_A     = (float*)malloc(size);
    float* h_B     = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i * 0.001f);            // 与教程一致：sin/cos 波形数据
        h_B[i] = cosf(i * 0.001f);
    }

    // ========== 设备内存分配（五步流程第 1 步）==========
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    eventTimerInit();

    // ========================================================================
    printf("\n========== Part 1: 基础版 + CPU vs GPU 对比 ==========\n");
    // ========================================================================
    {
        // ---- CPU 计时（std::chrono，CPU 侧没有 Event 可用）----
        auto t0 = std::chrono::high_resolution_clock::now();
        vectorAddCPU(h_A, h_B, h_C_cpu, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // ---- 端到端计时（教程 5.3 节）：含 H2D + kernel + D2H ----
        CUDA_CHECK(cudaEventRecord(g_evBeg));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));   // 第 2 步：H2D
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        int blockSize = 256;
        int gridSize  = (N + blockSize - 1) / blockSize;   // 向上取整：16384 个 Block

        // ---- 仅 kernel 计时（教程 5.2 节）----
        cudaEvent_t kBeg, kEnd;
        CUDA_CHECK(cudaEventCreate(&kBeg));
        CUDA_CHECK(cudaEventCreate(&kEnd));
        CUDA_CHECK(cudaEventRecord(kBeg));
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);            // 第 3 步：启动
        CUDA_CHECK(cudaEventRecord(kEnd));
        // kernel 启动的错误检查需要两步（教程 4.2 节）：
        CUDA_CHECK(cudaGetLastError());      // ①启动参数是否合法
        CUDA_CHECK(cudaEventSynchronize(kEnd));

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost)); // 第 4 步：D2H
        float total_ms = eventTimerEnd();

        float kernel_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kBeg, kEnd));

        // ---- 验证（每元素与 CPU 结果比对，容差 1e-5）----
        int errors = 0;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-5f) errors++;
        }
        printf("验证: %s (%d 处不一致)\n", errors == 0 ? "通过" : "失败", errors);

        // ---- 性能报表 ----
        printf("CPU 单核        : %8.3f ms\n", cpu_ms);
        printf("GPU 仅 Kernel   : %8.3f ms   (加速 %.1fx)\n", kernel_ms, cpu_ms / kernel_ms);
        printf("GPU 端到端含传输: %8.3f ms   (加速 %.1fx)\n", total_ms, cpu_ms / total_ms);

        // 有效带宽（教程 7.2 节）：读 A + 读 B + 写 C = 12 字节/元素
        float bw = (3.0f * size) / (kernel_ms * 1e-3f) / 1e9f;
        printf("Kernel 有效带宽 : %8.1f GB/s  (RTX 2080Ti 理论峰值 616 GB/s)\n", bw);
        printf("-> 端到端耗时是纯 Kernel 的 %.0f 倍：PCIe 传输主导，数据应尽量留在 GPU\n",
               total_ms / kernel_ms);
        CUDA_CHECK(cudaEventDestroy(kBeg));
        CUDA_CHECK(cudaEventDestroy(kEnd));
    }

    // ========================================================================
    printf("\n========== Part 2: Grid-stride Loop 版 ==========\n");
    // ========================================================================
    {
        // 只启动 128 Block × 256 线程 = 3.3 万线程（远小于 1600 万），
        // 每个线程循环处理 ~512 个元素
        vectorAddStride<<<128, 256>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < N; i++)
            if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-5f) errors++;
        printf("验证: %s  (128 个 Block 也覆盖了全部 %d 元素)\n",
               errors == 0 ? "通过" : "失败", N);
    }

    // ========================================================================
    printf("\n========== Part 3: SAXPY (Y = a*X + Y) ==========\n");
    // ========================================================================
    {
        const float alpha = 2.5f;
        // 重新初始化：Y 用 h_B 的副本，验证时同样计算
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        saxpy<<<(N + 255) / 256, 256>>>(alpha, d_A, d_B, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_B, size, cudaMemcpyDeviceToHost));
        // CPU 参考实现比对
        int errors = 0;
        for (int i = 0; i < N; i++) {
            float expect = alpha * h_A[i] + h_B[i];
            if (fabsf(h_C_gpu[i] - expect) > 1e-4f) errors++;
        }
        printf("验证: %s  (Y 原地更新为 %.1f*X+Y)\n", errors == 0 ? "通过" : "失败", alpha);
    }

    // ========================================================================
    printf("\n========== Part 4: 向量点积 (两级规约) ==========\n");
    // ========================================================================
    {
        // 注意：Part 3 的 SAXPY 把 d_B 原地改写成了 2.5X+Y（脏数据），
        // 点积要用原始的 A、B —— 重新上传。这本身就是"数据生命周期"的教训：
        // 原地更新省了显存，但下游使用者必须知道数据已变
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

        float* d_result;
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));   // 累加器清零

        int blockSize = 256;
        int gridSize  = 1024;
        // 动态共享内存大小在第三个启动参数指定：256 线程 × 4 字节
        dotProduct<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_A, d_B, d_result, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpu_result = 0.f;
        CUDA_CHECK(cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

        // CPU 参考实现
        double cpu_result = 0.0;
        for (int i = 0; i < N; i++) cpu_result += (double)h_A[i] * h_B[i];

        // 浮点累加顺序不同会有微小差异，用相对误差判断（1e-4 相对容差）
        double rel_err = fabs(gpu_result - cpu_result) / fabs(cpu_result);
        printf("GPU 点积 = %.4f, CPU 点积 = %.4f, 相对误差 = %.2e\n",
               gpu_result, cpu_result, rel_err);
        printf("验证: %s\n", rel_err < 1e-4 ? "通过" : "失败");
        CUDA_CHECK(cudaFree(d_result));
    }

    // ========== 清理（五步流程第 5 步）==========
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);

    printf("\n全部演示完成！\n");
    return 0;
}

// ============================================================================
// 2.grid_warp — Grid 多维度分组 + Warp 分歧性能实验
// 教程对应：AIInfraGuide《1.2 CUDA 编程模型》第 2/3/6/7 节
//
// 本程序演示四部分内容：
//   Part 1  1D Grid/Block：向量处理的基础索引公式
//   Part 2  2D Grid/Block：矩阵/图像的行列映射（含边界检查）
//   Part 3  3D Grid：批处理维度（z 维 = batch）
//   Part 4  Warp 分歧实验：
//             kernel A —— 同一 Warp 内奇偶线程走不同分支（严重分歧）
//             kernel B —— 以 Warp 为边界分支（无分歧）
//           用 CUDA Event 计时对比两者性能差距
// ============================================================================

#include <cstdio>
#include <cuda_runtime.h>

// Part 4 分支体的循环次数：几百次 FMA 让分支指令数成为耗时主导，
// Warp 分歧"串行执行两份分支"的代价才能在计时中显现
#define BRANCH_ITERS 512

// ----------------------------------------------------------------------------
// 错误检查宏：教程 4.4 节的模板。任何 CUDA 调用后都应检查返回值
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            printf("[CUDA 错误] %s @ %s:%d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================================
// Part 1: 1D 索引 —— 教程 3.2 节
// ============================================================================

// 数据规模远大于线程总数时，用 Grid-stride Loop（教程 3.4 节最佳实践）
// 好处：任意 N 都正确；且 Grid 大小可自由选择（不必等于 N/块大小）
__global__ void doubleElements1D(float* data, int n) {
    // 全局线程 ID = Block 编号 × Block 大小 + Block 内编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 步长 = 总线程数（gridDim.x * blockDim.x）
    int stride = gridDim.x * blockDim.x;

    // 每个线程以 stride 为步长遍历数组，覆盖所有元素
    for (int i = idx; i < n; i += stride) {
        data[i] *= 2.0f;
    }
}

// ============================================================================
// Part 2: 2D 索引 —— 教程 3.3 / 7.1 节（矩阵加法）
// ============================================================================

// 2D Block(16,16)=256 线程，2D Grid 覆盖整幅矩阵
// x 维映射列 col，y 维映射行 row（教程惯例：x=横向，y=纵向）
__global__ void matrixAdd2D(const float* A, const float* B, float* C,
                            int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：矩阵宽高不一定是 16 的整数倍时，越界线程直接跳过
    if (row < rows && col < cols) {
        int idx = row * cols + col;   // 行主序线性化
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================================
// Part 3: 3D Grid 批处理 —— 教程 7.3 节
// ============================================================================

// z 维（blockIdx.z）天然对应 batch 编号，一份 kernel 处理所有批次
__global__ void batchScale3D(float* data, int batchSize,
                             int height, int width) {
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;   // 第几个样本

    if (col < width && row < height && batch < batchSize) {
        int idx = batch * height * width + row * width + col;
        data[idx] *= 2.0f;
    }
}

// ============================================================================
// Part 4: Warp 分歧实验 —— 教程 6.2 节
// ============================================================================

// ----------------------------------------------------------------------------
// 分支体负载：纯寄存器计算密集循环
//
// 设计要点（前两版踩坑记录）：
//   v1 踩坑：分支体只有 8 次 FMA，0.004ms 跑完，分歧开销被淹没（1.01 倍）
//   v2 踩坑：两个分支写成一模一样 → nvcc 直接把 if/else 合并成
//            predication（谓词执行），硬件层面根本没发生分歧（1.00 倍）
//   正确做法：
//   1. 分支体要有几百条指令（512 次 FMA），让指令数成为耗时主导
//   2. 两条路径的代码必须真的不同（乘数不同），逼编译器保留分支结构
//      —— 同时保证指令条数、类型完全一致，成本才可比
//   3. 数据只读一次进寄存器，循环全在寄存器上，纯计算不访存
// ----------------------------------------------------------------------------
__device__ __forceinline__ float branchWorkA(float seed, int iters) {
    float r = seed;
    for (int k = 0; k < iters; ++k) r = fmaf(r, 1.0000001f, 1e-7f);
    return r;
}
__device__ __forceinline__ float branchWorkB(float seed, int iters) {
    float r = seed;
    for (int k = 0; k < iters; ++k) r = fmaf(r, 1.0000002f, 1e-7f);
    return r;
}

// ----- kernel A：Warp 内分歧（错误示范）-----
// 条件 threadIdx.x % 2 == 0：同一个 Warp 的 32 个线程里，
// 偶数 lane 走 pathA、奇数 lane 走 pathB —— 各占一半，分歧代价最大
// GPU 只能让整个 Warp 先串行执行 pathA 的完整循环（奇数线程空转），
// 再串行执行 pathB 的完整循环（偶数线程空转）—— 指令数翻倍
__global__ void divergentKernel(const float* __restrict__ data,
                                float* __restrict__ out, int n, int iters) {
    // Grid-stride：保证覆盖全部 N 个元素（本例 N=400万 > 线程总数 26万）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float r;
        if (threadIdx.x % 2 == 0) {
            r = branchWorkA(data[i], iters);    // pathA：偶数线程
        } else {
            r = branchWorkB(data[i], iters);    // pathB：奇数线程（工作量相同）
        }
        out[i] = r;
    }
}

// ----- kernel B：以 Warp 为边界分支（正确示范）-----
// 条件 (threadIdx.x / 32) % 2 == 0：Warp 0 的 32 个线程整体走同一分支，
// Warp 1 整体走另一分支 —— Warp 内部无分歧，每个 Warp 只需执行一份循环
// （教程原话：分歧发生在 Warp 内部，Warp 之间走不同分支完全无代价）
__global__ void warpAlignedKernel(const float* __restrict__ data,
                                  float* __restrict__ out, int n, int iters) {
    // 同样 Grid-stride 覆盖全部元素（与 kernel A 完全公平的对比条件）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float r;
        if ((threadIdx.x / 32) % 2 == 0) {
            r = branchWorkA(data[i], iters);    // pathA：整个 Warp 一起走
        } else {
            r = branchWorkB(data[i], iters);    // pathB：下一个 Warp 整体走
        }
        out[i] = r;
    }
}

// ----------------------------------------------------------------------------
// CUDA Event 计时：GPU 侧计时的标准做法（比 CPU 侧 clock() 准确，
// 因为 kernel 启动是异步的，CPU 必须用 event 才能量到纯 GPU 时间）
// ----------------------------------------------------------------------------
static float timeKernel(void (*launch)(float*, float*, int),
                        float* d_data, float* d_out, int n, int repeat) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(beg));            // 记录起始事件
    for (int r = 0; r < repeat; ++r) launch(d_data, d_out, n);
    CUDA_CHECK(cudaEventRecord(end));            // 记录结束事件
    CUDA_CHECK(cudaEventSynchronize(end));       // 等 GPU 真正做完

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));  // 两次事件间毫秒数
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms / repeat;   // 返回平均单次耗时
}

// 两种 kernel 的启动封装（timeKernel 通过函数指针调用）
static void launchDivergent(float* d, float* o, int n)  { divergentKernel  <<<1024, 256>>>(d, o, n, BRANCH_ITERS); }
static void launchWarpAligned(float* d, float* o, int n){ warpAlignedKernel<<<1024, 256>>>(d, o, n, BRANCH_ITERS); }

// ----------------------------------------------------------------------------
// 打印 2D 索引映射小样：直观展示 col/row 如何由 threadIdx/blockIdx 算出
// ----------------------------------------------------------------------------
__global__ void print2DMapping(int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4 && threadIdx.y < 4)
        printf("  thread(%2d,%2d) -> 矩阵坐标 [row=%d, col=%d]\n",
               threadIdx.x, threadIdx.y, row, col);
}

int main() {
    printf("========== Part 1: 1D Grid-stride Loop ==========\n");
    {
        const int N = 1000000;
        float* h = new float[N];
        for (int i = 0; i < N; ++i) h[i] = 1.0f;

        float* d;
        CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice));

        // 数据 100 万，但只启动 1024 Block × 256 线程 = 26 万线程，
        // 每个线程靠 stride 循环处理约 4 个元素
        doubleElements1D<<<1024, 256>>>(d, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("N=%d, 配置<<<1024,256>>>(总线程=%d), 验证 h[0]=%.1f h[N-1]=%.1f (期望 2.0)\n",
               N, 1024 * 256, h[0], h[N - 1]);
        CUDA_CHECK(cudaFree(d));
        delete[] h;
    }

    printf("\n========== Part 2: 2D Grid/Block 矩阵加法 ==========\n");
    {
        // 故意用非 16 倍数的尺寸（61×37），演示边界检查的必要性
        const int ROWS = 37, COLS = 61;
        const int N = ROWS * COLS;
        float *hA = new float[N], *hB = new float[N], *hC = new float[N];
        for (int i = 0; i < N; ++i) { hA[i] = 1.f; hB[i] = 2.f; }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);                                    // 256 线程/Block
        //要让一个block中的访存尽量连续
        dim3 grid((COLS + 15) / 16, (ROWS + 15) / 16);         // 向上取整覆盖全矩阵
        printf("矩阵 %d×%d, block(16,16), grid(%d,%d)\n",
               ROWS, COLS, grid.x, grid.y);

        print2DMapping<<<dim3(1,1), dim3(4,4)>>>(COLS);        // 打印索引映射示例
        CUDA_CHECK(cudaDeviceSynchronize());

        matrixAdd2D<<<grid, block>>>(dA, dB, dC, ROWS, COLS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < N; ++i) if (hC[i] != 3.f) { ok = false; break; }
        printf("验证 C = A + B: %s（含边界角落元素 row=%d,col=%d -> %.1f）\n",
               ok ? "通过" : "失败", ROWS - 1, COLS - 1, hC[N - 1]);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        delete[] hA; delete[] hB; delete[] hC;
    }

    printf("\n========== Part 3: 3D Grid 批处理 ==========\n");
    {
        const int BATCH = 4, H = 32, W = 32, N = BATCH * H * W;
        float* h = new float[N];
        for (int i = 0; i < N; ++i) h[i] = 1.0f;

        float* d;
        CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 8, 1);
        // z 维直接 = batch 数，xy 维覆盖单张图像
        dim3 grid((W + 15) / 16, (H + 7) / 8, BATCH);
        printf("批 %d 张 %d×%d 图, block(16,8,1), grid(%d,%d,%d)\n",
               BATCH, H, W, grid.x, grid.y, grid.z);

        batchScale3D<<<grid, block>>>(d, BATCH, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost));
        // 验证每个 batch 的首尾元素（确认 z 维映射正确）
        printf("验证: batch0[0]=%.1f, batch3[末]=%.1f (期望 2.0)\n",
               h[0], h[N - 1]);
        CUDA_CHECK(cudaFree(d));
        delete[] h;
    }

    printf("\n========== Part 4: Warp 分歧性能实验 ==========\n");
    {
        const int N = 1 << 22;             // 400 万元素
        const int REPEAT = 50;             // 每种 kernel 重复 50 次取平均
        float* h = new float[N];
        for (int i = 0; i < N; ++i) h[i] = 1.5f;

        float *dData, *dOut;
        CUDA_CHECK(cudaMalloc(&dData, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut,   N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dData, h, N * sizeof(float), cudaMemcpyHostToDevice));

        printf("配置 <<<1024, 256>>>，N=%d，重复 %d 次取平均\n", N, REPEAT);

        // 先各跑一次预热（首次启动有初始化开销，不计时）
        launchDivergent(dData, dOut, N);
        launchWarpAligned(dData, dOut, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        float msA = timeKernel(launchDivergent,   dData, dOut, N, REPEAT);
        float msB = timeKernel(launchWarpAligned, dData, dOut, N, REPEAT);

        printf("kernel A (Warp 内奇偶分歧): %8.3f ms\n", msA);
        printf("kernel B (Warp 对齐无分歧) : %8.3f ms\n", msB);
        printf("分歧版本慢 %.2f 倍 —— 同一 Warp 的 32 线程被迫串行执行两个分支\n",
               msA / msB);
        printf("注: 两版分支工作量相同，差距完全来自 Warp 分歧调度开销\n");

        CUDA_CHECK(cudaFree(dData)); CUDA_CHECK(cudaFree(dOut));
        delete[] h;
    }

    printf("\n全部演示完成！\n");
    return 0;
}

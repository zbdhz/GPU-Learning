// ============================================================================
// 4.warp_shuffle — Warp Shuffle 全家福测试
// 教程对应：AIInfraGuide《2.1 Warp 与执行模型》第 4~6 节
//
// 覆盖内容：
//   Part 1  四种 Shuffle 指令行为验证（shfl / shfl_up / shfl_down / shfl_xor）
//   Part 2  Warp 内广播（教程 4.4）与 Warp 内求和（教程 4.5 蝶形归约）
//   Part 3  width 参数：子 Warp 独立归约（教程 5.4，8 线程一组）
//   Part 4  Warp Vote 三兄弟：__all_sync / __any_sync / __ballot_sync（教程 5.1）
//   Part 5  实战：Block 级归约（Shuffle + 共享内存，教程 6.1）
//           vs 纯共享内存树形归约 —— 性能对比（教程 6.2）
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

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
// Part 1: 四种 Shuffle 指令的行为验证
// ============================================================================
// 每个 kernel 只用一个 Warp（32 线程），laneId = threadIdx.x（0~31）
// 每个线程初始 val = laneId，Shuffle 后把"我拿到的值"写回输出数组，
// 主机端检查"lane i 拿到的值"是否符合教程 4.2/4.3 表格的定义

// __shfl_sync：直接读取指定 lane 的值（广播语义）
__global__ void test_shfl(int* out, int srcLane) {
    int lane = threadIdx.x;
    int v = __shfl_sync(0xFFFFFFFF, lane, srcLane);   // 人人去读 srcLane 的 laneId
    out[lane] = v;                                    // 所有线程都应拿到 srcLane
}

// __shfl_up_sync：从低编号 lane 取（lane - delta），低边界线程取不到就返回自己
__global__ void test_shfl_up(int* out, int delta) {
    int lane = threadIdx.x;
    int v = __shfl_up_sync(0xFFFFFFFF, lane, delta);  // 我读 lane-delta 的值
    out[lane] = v;                                    // lane<delta 的拿不到 → 保持原值
}

// __shfl_down_sync：从高编号 lane 取（lane + delta），高边界同理
__global__ void test_shfl_down(int* out, int delta) {
    int lane = threadIdx.x;
    int v = __shfl_down_sync(0xFFFFFFFF, lane, delta);
    out[lane] = v;
}

// __shfl_xor_sync：与 lane ^ laneMask 配对交换（蝶形连接，归约的主力）
__global__ void test_shfl_xor(int* out, int laneMask) {
    int lane = threadIdx.x;
    int v = __shfl_xor_sync(0xFFFFFFFF, lane, laneMask);
    out[lane] = v;                                    // lane i 拿到 (i^mask) 的值
}

// ============================================================================
// Part 2: Warp 内广播 + Warp 内求和（教程 4.4 / 4.5）
// ============================================================================

// 广播：所有线程获取 srcLane 线程持有的 val
__device__ float warp_broadcast(float val, int srcLane) {
    return __shfl_sync(0xFFFFFFFF, val, srcLane);
}

// 蝶形归约求和：log2(32)=5 步，lane 0 持有最终结果（教程 4.5）
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 验证 kernel：一个 Warp，每个 lane 持有一个值，验证广播与求和
__global__ void test_broadcast_reduce(const float* in, float* bcast,
                                      float* sumOut) {
    int lane = threadIdx.x;
    float val = in[lane];                 // lane i 持有 in[i]

    // 广播 lane 7 的值给所有人
    float b = warp_broadcast(val, 7);
    if (lane != 7) bcast[lane] = b;       // 写回检查（lane7 本来就是自己）

    // Warp 求和（down 版本：lane 0 拿到总和）
    float s = warp_reduce_sum(val);
    if (lane == 0) *sumOut = s;
}

// ============================================================================
// Part 3: width 参数 —— 子 Warp 独立归约（教程 5.4）
// ============================================================================
// width=8：Warp 被视为 4 个独立 8 线程子组，归约只发生在组内。
// 理论结果：Lane 0/8/16/24 分别持有各自子组的和（down 版本）

__global__ void test_subwarp(const float* in, float* out) {
    int lane = threadIdx.x;
    float val = in[lane];

    // offset 从 width/2=4 开始，只在 8 线程子组内折半
    for (int offset = 4; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, 8);
    }
    // 每个子组的 lane 0（即 lane 0/8/16/24）写出子组和
    if ((lane & 7) == 0) {
        out[lane / 8] = val;
    }
}

// ============================================================================
// Part 4: Warp Vote —— __all_sync / __any_sync / __ballot_sync（教程 5.1）
// ============================================================================

// pred = (lane >= threshold)：随 threshold 变化观察三种投票结果
//
// ⚠️ mask 契约（本例首跑就踩过的坑）：
//    __all/__any/__ballot_sync 是 Warp 集体操作，mask（0xFFFFFFFF）指定的
//    全部 32 个线程【必须都执行到这条指令】。绝不能写成
//        if (lane == 0) *out = __any_sync(0xFFFFFFFF, pred);   // ❌
//    ——那样只有 lane 0 执行 vote，其余 31 人缺席，返回值未定义（实测全 0）。
//    正确姿势：全员参与投票，结果先落寄存器，lane 0 再负责写回
__global__ void test_vote(int threshold, int* allResult, int* anyResult,
                          unsigned* ballotResult) {
    int lane = threadIdx.x;
    int pred = (lane >= threshold);       // 后半段（从 threshold 起）为真

    // 全体 32 线程一起投票（集体操作，一个都不能少）
    int allR      = __all_sync(0xFFFFFFFF, pred);     // 全体真才真
    int anyR      = __any_sync(0xFFFFFFFF, pred);     // 任一真即真
    unsigned bal  = __ballot_sync(0xFFFFFFFF, pred);  // 32bit 位图

    // 投票完毕，只有 lane 0 负责写回全局内存（省 31 次冗余写）
    if (lane == 0) {
        *allResult    = allR;
        *anyResult    = anyR;
        *ballotResult = bal;
    }
}

// ============================================================================
// Part 5: 实战 —— 两种 Block 级归约性能对比（教程 6.1 / 6.2）
// ============================================================================

// ---------- 版本 A：纯共享内存树形归约（示例 3 Part 4 的做法）----------
__global__ void blockReduceShared(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 每线程先做 Grid-stride 部分和（这里每线程只管一个元素，简化对照）
    float v = (idx < N) ? in[idx] : 0.0f;
    sdata[tid] = v;
    __syncthreads();

    // 256 → 128 → ... → 1：每轮折半，8 轮，每轮一个 __syncthreads()
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- 版本 B：Warp Shuffle + 共享内存（教程 6.1 推荐写法）----------
__global__ void blockReduceShuffle(const float* in, float* out, int N) {
    __shared__ float warp_sums[32];       // 最多 32 个 Warp（1024 线程 Block）
    int tid  = threadIdx.x;
    int lane = tid % 32;                   // Warp 内编号
    int warp = tid / 32;                   // Block 内 Warp 编号
    int idx  = blockIdx.x * blockDim.x + tid;

    float v = (idx < N) ? in[idx] : 0.0f;

    // 第一步：Warp 内 Shuffle 归约（5 步，无共享内存无同步）
    v = warp_reduce_sum(v);

    // 第二步：每个 Warp 的 lane 0 写共享内存，只需 1 次 __syncthreads()
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    // 第三步：第一个 Warp 把 8 个 warp 和（256/32=8）再做一次 Shuffle 归约
    if (warp == 0) {
        v = (tid < blockDim.x / 32) ? warp_sums[tid] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

// ----------------------------------------------------------------------------
// CUDA Event 计时封装：预热 + REPEAT 次取最小（bench 惯例，最小值=无干扰状态）
// ----------------------------------------------------------------------------
static float benchKernel(void (*launch)(void*), void* ctx, int repeat) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    launch(ctx);                                    // 预热 1 次（模块加载等一次性开销）
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < repeat; ++i) launch(ctx);   // 一次 Event 对包 repeat 次启动
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&best, beg, end));
    return best / repeat;                           // 平均每次的毫秒数
}

// 启动器上下文：把 kernel 参数打包传给 benchKernel
struct ReduceCtx {
    const float* in; float* out; int n; int blocks; int threads;
};
static void launchShared(void* p) {
    ReduceCtx* c = (ReduceCtx*)p;
    blockReduceShared<<<c->blocks, c->threads, c->threads * sizeof(float)>>>(
        c->in, c->out, c->n);
}
static void launchShuffle(void* p) {
    ReduceCtx* c = (ReduceCtx*)p;
    blockReduceShuffle<<<c->blocks, c->threads>>>(c->in, c->out, c->n);
}

int main() {
    printf("Warp Shuffle 全家福测试（RTX GPU，教程 2.1 节）\n");
    printf("====================================================\n");

    // ========================================================================
    printf("\n========== Part 1: 四种 Shuffle 指令行为 ==========\n");
    // ========================================================================
    {
        int* d_out; int* h_out = (int*)malloc(32 * sizeof(int));
        CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(int)));

        // --- __shfl_sync：广播 lane 5 的 laneId ---
        test_shfl<<<1, 32>>>(d_out, 5);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < 32; i++) if (h_out[i] != 5) ok = false;
        printf("__shfl_sync(5)     : %s  (全部 lane 都拿到 5)\n", ok ? "通过" : "失败");

        // --- __shfl_up_sync(delta=4)：lane i 拿 max(i-4, i) ---
        test_shfl_up<<<1, 32>>>(d_out, 4);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost));
        ok = true;
        for (int i = 0; i < 32; i++) {
            int expect = (i >= 4) ? (i - 4) : i;    // 低边界拿不到 → 原值
            if (h_out[i] != expect) ok = false;
        }
        printf("__shfl_up_sync(4)  : %s  (lane i 得 i-4；lane<4 保持原值)\n",
               ok ? "通过" : "失败");

        // --- __shfl_down_sync(delta=4)：lane i 拿 min(i+4, i) ---
        test_shfl_down<<<1, 32>>>(d_out, 4);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost));
        ok = true;
        for (int i = 0; i < 32; i++) {
            int expect = (i < 28) ? (i + 4) : i;    // 高边界拿不到 → 原值
            if (h_out[i] != expect) ok = false;
        }
        printf("__shfl_down_sync(4): %s  (lane i 得 i+4；lane>27 保持原值)\n",
               ok ? "通过" : "失败");

        // --- __shfl_xor_sync(mask=1)：两两互换 0↔1, 2↔3, ...
        test_shfl_xor<<<1, 32>>>(d_out, 1);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost));
        ok = true;
        for (int i = 0; i < 32; i++) if (h_out[i] != (i ^ 1)) ok = false;
        printf("__shfl_xor_sync(1) : %s  (lane i 与 i^1 互换：0<->1, 2<->3...)\n",
               ok ? "通过" : "失败");

        // --- __shfl_xor_sync(mask=16)：前后两半交叉配对 ---
        test_shfl_xor<<<1, 32>>>(d_out, 16);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost));
        ok = true;
        for (int i = 0; i < 32; i++) if (h_out[i] != (i ^ 16)) ok = false;
        printf("__shfl_xor_sync(16): %s  (lane i 与 i^16 配对：前后半交叉)\n",
               ok ? "通过" : "失败");

        CUDA_CHECK(cudaFree(d_out));
        free(h_out);
    }

    // ========================================================================
    printf("\n========== Part 2: Warp 广播 + Warp 求和 ==========\n");
    // ========================================================================
    {
        float* d_in; float* d_b; float* d_sum;
        float* h_in = (float*)malloc(32 * sizeof(float));
        for (int i = 0; i < 32; i++) h_in[i] = (float)(i + 1);   // 1~32
        CUDA_CHECK(cudaMalloc(&d_in, 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b,  32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

        test_broadcast_reduce<<<1, 32>>>(d_in, d_b, d_sum);

        float* h_b = (float*)malloc(32 * sizeof(float));
        float h_sum = 0;
        CUDA_CHECK(cudaMemcpy(h_b, d_b, 32 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < 32; i++)
            if (i != 7 && fabsf(h_b[i] - 8.0f) > 1e-6f) ok = false;  // lane7 值=8
        printf("广播 lane7 的值     : %s  (全员得到 %.0f)\n",
               ok ? "通过" : "失败", h_b[0]);

        float expect = 32.0f * 33.0f / 2.0f;          // 1+2+...+32 = 528
        printf("Warp 求和 1..32    : %s  (lane0 得 %.0f, 期望 %.0f)\n",
               fabsf(h_sum - expect) < 1e-3f ? "通过" : "失败", h_sum, expect);

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_sum));
        free(h_in); free(h_b);
    }

    // ========================================================================
    printf("\n========== Part 3: width=8 子 Warp 归约 ==========\n");
    // ========================================================================
    {
        float* d_in; float* d_out;
        float* h_in = (float*)malloc(32 * sizeof(float));
        for (int i = 0; i < 32; i++) h_in[i] = (float)(i % 8) + 1.0f;  // 每组都是 1~8
        CUDA_CHECK(cudaMalloc(&d_in, 32 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

        test_subwarp<<<1, 32>>>(d_in, d_out);

        float h_out[4];
        CUDA_CHECK(cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        // 每个子组 1+2+...+8 = 36；且组间互不干扰（若 width 失效会串成别的数）
        bool ok = true;
        for (int g = 0; g < 4; g++)
            if (fabsf(h_out[g] - 36.0f) > 1e-3f) ok = false;
        printf("4 个子组各自求和    : %s  (结果 [%g %g %g %g]，期望全 36)\n",
               ok ? "通过" : "失败", h_out[0], h_out[1], h_out[2], h_out[3]);

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        free(h_in);
    }

    // ========================================================================
    printf("\n========== Part 4: Warp Vote 投票 ==========\n");
    // ========================================================================
    {
        int *d_all, *d_any; unsigned* d_ballot;
        int h_all, h_any; unsigned h_ballot;
        CUDA_CHECK(cudaMalloc(&d_all, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_any, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ballot, sizeof(unsigned)));

        // threshold=20：lane 0~19 为假，20~31 为真（12 个真）
        test_vote<<<1, 32>>>(20, d_all, d_any, d_ballot);
        CUDA_CHECK(cudaMemcpy(&h_all, d_all, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_any, d_any, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_ballot, d_ballot, 4, cudaMemcpyDeviceToHost));

        unsigned expectBallot = 0xFFF00000u;         // 高 12 位为 1
        printf("threshold=20:\n");
        printf("  __all_sync  = %d (期望 0：并非全员为真)\n", h_all);
        printf("  __any_sync  = %d (期望 1：有人为真)\n", h_any);
        printf("  __ballot    = 0x%08X (期望 0x%08X：%s)\n", h_ballot, expectBallot,
               h_ballot == expectBallot ? "通过" : "失败");

        // threshold=0：全员为真
        test_vote<<<1, 32>>>(0, d_all, d_any, d_ballot);
        CUDA_CHECK(cudaMemcpy(&h_all, d_all, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_any, d_any, 4, cudaMemcpyDeviceToHost));
        printf("threshold=0 : __all_sync = %d (期望 1：全员为真)\n", h_all);

        // threshold=32：全员为假
        test_vote<<<1, 32>>>(32, d_all, d_any, d_ballot);
        CUDA_CHECK(cudaMemcpy(&h_any, d_any, 4, cudaMemcpyDeviceToHost));
        printf("threshold=32: __any_sync = %d (期望 0：全员为假)\n", h_any);

        CUDA_CHECK(cudaFree(d_all)); CUDA_CHECK(cudaFree(d_any)); CUDA_CHECK(cudaFree(d_ballot));
    }

    // ========================================================================
    printf("\n========== Part 5: Block 归约性能对比（1 亿元素）==========\n");
    // ========================================================================
    {
        const int N = 1 << 27;                       // 1.34 亿元素，512MB
        const size_t size = N * sizeof(float);
        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;   // 524288 个 Block

        float* h_in = (float*)malloc(size);
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;  // 全 1：期望和 = N（易验证）

        float *d_in, *d_outA, *d_outB;
        CUDA_CHECK(cudaMalloc(&d_in, size));
        CUDA_CHECK(cudaMalloc(&d_outA, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outB, blocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

        ReduceCtx ctx = { d_in, d_outA, N, blocks, threads };
        const int REPEAT = 50;

        // 预热 + 计时（两者同规模：每 Block 输出 1 个部分和）
        float msA = benchKernel(launchShared,  &ctx, REPEAT);
        ctx.out = d_outB;
        float msB = benchKernel(launchShuffle, &ctx, REPEAT);

        // 正确性：每 Block 全 1×256=256，两种版本都检查前 1024 个 Block
        float* h_out = (float*)malloc(1024 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_outA, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
        bool okA = true;
        for (int i = 0; i < 1024; i++) if (h_out[i] != 256.0f) okA = false;
        CUDA_CHECK(cudaMemcpy(h_out, d_outB, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
        bool okB = true;
        for (int i = 0; i < 1024; i++) if (h_out[i] != 256.0f) okB = false;

        // 有效带宽：每元素只读 4B（写出量极小忽略）
        float bwA = size / (msA * 1e-3f) / 1e9f;
        float bwB = size / (msB * 1e-3f) / 1e9f;

        printf("纯共享内存树形归约  : %8.3f ms  %7.1f GB/s  验证:%s\n",
               msA, bwA, okA ? "通过" : "失败");
        printf("Warp Shuffle 归约   : %8.3f ms  %7.1f GB/s  验证:%s\n",
               msB, bwB, okB ? "通过" : "失败");
        printf("Shuffle 版加速      : %.2fx\n", msA / msB);
        printf("-> Shuffle 省下 7 次 __syncthreads 与大块共享内存，归约阶段延迟更低\n");

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_outA)); CUDA_CHECK(cudaFree(d_outB));
        free(h_in); free(h_out);
    }

    printf("\n全部测试完成！\n");
    return 0;
}

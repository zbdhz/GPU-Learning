// ============================================================================
// 5.mem_access — 内存访问优化全家桶
// 教程对应：AIInfraGuide《2.2 内存访问优化》第 2~5 节全部例子
//
// 覆盖内容：
//   Part 1  合并访问 vs 跨步访问：stride=1/2/4/32 带宽实测（教程 2.2）
//   Part 2  AoS vs SoA：粒子系统"只读 x"场景对比（教程 2.4）
//   Part 3  Bank Conflict：32[32] 列访问 32-way 冲突 + Padding 修复（教程 3.3/3.4）
//   Part 4  向量化加载：标量 vs float4 拷贝带宽对比（教程 4.1~4.3）
//   Part 5  综合实战：矩阵转置三版本——朴素 / 共享内存+Padding / float4（教程 5.1）
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

// ----------------------------------------------------------------------------
// 计时工具：CUDA Event，预热 1 次 + REPEAT 次取平均
// ----------------------------------------------------------------------------
static cudaEvent_t g_beg, g_end;
static void timerInit()  { CUDA_CHECK(cudaEventCreate(&g_beg)); CUDA_CHECK(cudaEventCreate(&g_end)); }
static void timerFree()  { CUDA_CHECK(cudaEventDestroy(g_beg)); CUDA_CHECK(cudaEventDestroy(g_end)); }
static float timerAvg(void (*launch)(void*), void* ctx, int repeat) {
    launch(ctx);                                   // 预热：排除首跑模块加载开销
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(g_beg));
    for (int i = 0; i < repeat; ++i) launch(ctx);
    CUDA_CHECK(cudaEventRecord(g_end));
    CUDA_CHECK(cudaEventSynchronize(g_end));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_beg, g_end));
    return ms / repeat;
}

// ============================================================================
// Part 1: 合并访问 vs 跨步访问（教程 2.2）
// ============================================================================
// 预期：stride=1 接近峰值带宽；stride=2 半速；stride=4 四分之一；
//       stride=32 时每线程独占一条 128B line，带宽利用率 ~3%
// 跨步读取：out[i] = data[i*stride]。注意 stride>1 时最大下标是 N*stride-1，
// 调用方必须保证 data 缓冲区足够大，否则越界（本文件 Part 1 已处理）
__global__ void strided_read(const float* data, float* out, long long N,
                             int stride) {
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    long long gstride = (long long)blockDim.x * gridDim.x;
    for (long long i = tid; i < N; i += gstride) {
        out[i] = data[i * stride];      // stride=1 连续；stride>1 跳跃
    }
}

struct StrideCtx {
    const float* d_data; float* d_out; long long n; int stride;
};
static void launchStride(void* p) {
    StrideCtx* c = (StrideCtx*)p;
    strided_read<<<2048, 256>>>(c->d_data, c->d_out, c->n, c->stride);
}

// ============================================================================
// Part 2: AoS vs SoA（教程 2.4）
// ============================================================================
// 粒子系统：每个粒子有 x,y,z,w 四个字段。Kernel 只需要把所有 x 翻倍。
//   AoS：内存排布 x0 y0 z0 w0 x1 y1 z1 w1 ... → 相邻线程地址差 16B（stride=4）
//   SoA：内存排布 x0 x1 x2 ... y0 y1 ...        → 相邻线程地址差 4B（合并）

struct ParticleAoS { float x, y, z, w; };           // 16B

__global__ void scaleX_aos(ParticleAoS* p, long long N) {
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    if (tid < N) p[tid].x *= 2.0f;                  // stride=4，只用 1/4 数据
}

__global__ void scaleX_soa(float* x, long long N) {
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    if (tid < N) x[tid] *= 2.0f;                    // stride=1，全合并
}

// Part 2 的计时上下文与启动包装（timerAvg 需要统一的 void(*)(void*) 签名）
struct AoSCtx { ParticleAoS* a; long long n; };
struct SoACtx { float* x; long long n; };
static void launchAoS(void* p) {
    AoSCtx* c = (AoSCtx*)p;
    long long blocks = (c->n + 255) / 256;
    scaleX_aos<<<(unsigned)blocks, 256>>>(c->a, c->n);
}
static void launchSoA(void* p) {
    SoACtx* c = (SoACtx*)p;
    long long blocks = (c->n + 255) / 256;
    scaleX_soa<<<(unsigned)blocks, 256>>>(c->x, c->n);
}

// ============================================================================
// Part 3: Bank Conflict 与 Padding（教程 3.3 / 3.4）
// ============================================================================
// 共享内存 32 Bank × 4B。tile[32][32] 按列读 = 32 线程地址同 Bank → 32-way；
// tile[32][33] 行距 33 float，与 32 互质 → 同列地址落在 32 个不同 Bank → 无冲突

#define TILE 32

__global__ void read_col_conflict(const float* in, float* out) {
    __shared__ float tile[TILE][TILE];              // ❌ 无 Padding：列读 32-way
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = in[ty * TILE + tx];
    __syncthreads();
    // 列访问：32 个线程(tx=0..31)读 tile[tx][ty]，地址间隔 32 float = 同 Bank
    out[ty * TILE + tx] = tile[tx][ty];
}

__global__ void read_col_padded(const float* in, float* out) {
    __shared__ float tile[TILE][TILE + 1];          // ✅ Padding：行距 33，互质
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = in[ty * TILE + tx];
    __syncthreads();
    out[ty * TILE + tx] = tile[tx][ty];             // 32 线程落 32 个 Bank
}

struct BCCtx { const float* d_in; float* d_out; };
static void launchConflict(void* p) {
    BCCtx* c = (BCCtx*)p;
    read_col_conflict<<<1, dim3(TILE, TILE)>>>(c->d_in, c->d_out);
}
static void launchPadded(void* p) {
    BCCtx* c = (BCCtx*)p;
    read_col_padded<<<1, dim3(TILE, TILE)>>>(c->d_in, c->d_out);
}

// ============================================================================
// Part 4: 向量化加载 float4（教程 4.1~4.3）
// ============================================================================
// 标量：每线程 1 个 float（4B）→ Warp 128B / 1 条 line，但 LDG.32 指令多
// float4：每线程 16B → 一条 LDG.128，指令数减 75%，带宽通常提升 10~30%

__global__ void copy_scalar(const float* src, float* dst, long long N) {
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    long long gstride = (long long)blockDim.x * gridDim.x;
    for (long long i = tid; i < N; i += gstride)
        dst[i] = src[i];
}

__global__ void copy_float4(const float4* src, float4* dst, long long nVec) {
    long long tid = (long long)threadIdx.x + (long long)blockIdx.x * blockDim.x;
    long long gstride = (long long)blockDim.x * gridDim.x;
    for (long long i = tid; i < nVec; i += gstride) {
        float4 v = src[i];                          // 一条 16B 加载
        dst[i] = v;
    }
}

struct CopyCtx {
    const float* d_src; float* d_dst; long long n; int pad;
};
static void launchScalar(void* p) {
    CopyCtx* c = (CopyCtx*)p;
    copy_scalar<<<2048, 256>>>(c->d_src, c->d_dst, c->n);
}
static void launchVec(void* p) {
    CopyCtx* c = (CopyCtx*)p;
    copy_float4<<<2048, 256>>>(
        reinterpret_cast<const float4*>(c->d_src),
        reinterpret_cast<float4*>(c->d_dst), c->n / 4);
}

// ============================================================================
// Part 5: 矩阵转置三版本（教程 5.1）
// ============================================================================
// 矩阵 8192x8192。转置零计算，纯访存，是检验优化链的最佳标本：
//   v1 朴素：        读合并 ✅ 写跨步 ❌
//   v2 共享+Padding：读写全合并，tile 列读经 +1 padding 消冲突
//   v3 float4：      在 v2 基础上每个线程一次搬 4 列（tile[33][TILE+1] 改为按
//                    float4 行存：tile 行距 = TILE+4，同样与 32 互质保无冲突）

#define MAT_N 8192                                   // 矩阵边长（float）
#define ROWS_PER_THREAD 8                            // 每线程负责 8 行（教程 BLOCK_ROWS=8）

// ---------- v1: 朴素转置 ----------
__global__ void transpose_naive(const float* in, float* out, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < width && row < width) {
        out[col * width + row] = in[row * width + col];
        // 读 in[row*width+col]：tx 变化 → col 连续 → 合并
        // 写 out[col*width+row]：tx 变化 → 行号 stride=width → 非合并
    }
}

// ---------- v2: 共享内存 + Padding ----------
__global__ void transpose_shared(const float* in, float* out, int width) {
    __shared__ float tile[TILE][TILE + 1];           // +1 Padding（教程 5.1 原版）

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // 合并读入 tile：warp 内 tx 连续 → in 地址连续
    for (int j = 0; j < TILE; j += ROWS_PER_THREAD) {
        if (x < width && (y + j) < width)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
    __syncthreads();

    // 写出坐标做转置交换（block 级）：读 tile 列（padding 防冲突），写 out 合并
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS_PER_THREAD) {
        if (x < width && (y + j) < width)
            out[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ---------- v3: 共享内存 + float4 ----------
// 每线程一次处理 4 个连续列元素：in 行内 float4 合并读；tile 按
// [TILE][TILE+4] 存（行距 32+4=36 float，与 32 互质 → 列读无 Bank 冲突）
__global__ void transpose_float4(const float* in, float* out, int width) {
    __shared__ float4 tile[TILE][TILE / 4 + 1];      // 每行 8 个 float4，+1 padding

    int x4 = (blockIdx.x * TILE + threadIdx.x) / 4;  // float4 列索引（0..7）
    int tx4 = threadIdx.x / 4;                        // 线程组编号
    int laneInGrp = threadIdx.x % 4;                  // 组内位置
    int y = blockIdx.y * TILE + threadIdx.y;

    // 合并读：一个 warp 一次读 128 个连续 float（4 个 tile 行 × 32 float）
    // 注意：threadIdx.y 每次跨 ROWS_PER_THREAD=8 行，同一 warp 的 32 个线程
    // 按 tx4（0..7）分 8 组，每组 4 线程读同一个 float4 地址 → 硬件广播，无冲突
    for (int j = 0; j < TILE; j += ROWS_PER_THREAD) {
        if ((x4 * 4 + 3) < width && (y + j) < width) {
            float4 v = reinterpret_cast<const float4*>(
                in + (y + j) * width + x4 * 4)[0];
            // 拆成 4 个 float 存入 tile（保持列语义，为写出做准备）
            tile[threadIdx.y + j][tx4].x = v.x;
            tile[threadIdx.y + j][tx4].y = v.y;
            tile[threadIdx.y + j][tx4].z = v.z;
            tile[threadIdx.y + j][tx4].w = v.w;
        }
    }
    __syncthreads();

    // 转置写出：读 tile 列（行距 TILE/4+1 个 float4 = 9×16B，无冲突），合并写
    // outRow/outCol 交换了 blockIdx.x/y —— 这就是"转置"发生在 block 级的体现
    int outRow = blockIdx.x * TILE + threadIdx.y;
    int outCol = blockIdx.y * TILE + threadIdx.x;
    for (int j = 0; j < TILE; j += ROWS_PER_THREAD) {
        if (outRow + j < width && outCol < width) {
            // threadIdx.y 每 8 行一轮，j 也是 8 步进 —— 二者组合覆盖 0..31 行。
            // 每个线程从 tile[threadIdx.x][ ... ] 取一个 float 写到转置位置：
            //   行选择 = threadIdx.y%4（决定取 .x/.y/.z/.w 哪个分量）
            //   组选择 = threadIdx.y/4 + j/4（决定 tile 内的 float4 列）
            float4 t = tile[threadIdx.x][threadIdx.y / 4 + j / 4];
            float sel = (threadIdx.y % 4 == 0) ? t.x :
                        (threadIdx.y % 4 == 1) ? t.y :
                        (threadIdx.y % 4 == 2) ? t.z : t.w;
            out[(outRow + j) * width + outCol] = sel;
        }
    }
    (void)laneInGrp;
}

struct TransCtx {
    const float* d_in; float* d_out; int n;
};
static void launchNaive(void* p) {
    TransCtx* c = (TransCtx*)p;
    dim3 block(TILE, TILE / 4);                      // 32x8 = 256 线程
    dim3 grid(c->n / TILE, c->n / TILE);
    transpose_naive<<<grid, block>>>(c->d_in, c->d_out, c->n);
}
static void launchShared(void* p) {
    TransCtx* c = (TransCtx*)p;
    dim3 block(TILE, ROWS_PER_THREAD);               // 教程 32x8
    dim3 grid(c->n / TILE, c->n / TILE);
    transpose_shared<<<grid, block>>>(c->d_in, c->d_out, c->n);
}
static void launchF4(void* p) {
    TransCtx* c = (TransCtx*)p;
    dim3 block(TILE, ROWS_PER_THREAD);
    dim3 grid(c->n / TILE, c->n / TILE);
    transpose_float4<<<grid, block>>>(c->d_in, c->d_out, c->n);
}

// ============================================================================
// 主函数：跑全部 Part，打印带宽与加速比
// ============================================================================
int main() {
    int devCnt = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCnt));
    if (devCnt == 0) { printf("未找到 GPU\n"); return 1; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SM)\n", prop.name, prop.multiProcessorCount);
    // 理论带宽 = 显存等效频率(Hz) × 总线字节宽度
    // memoryClockRate 是实际频率，DDR 显存等效带宽 ×2
    double peakBW = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    printf("理论带宽: %.1f GB/s\n\n", peakBW);

    timerInit();
    const int REPEAT = 20;
    // Part 1 用 256M 元素（1GB 有效数据）；data 按 32 倍stride余量分配会爆显存，
    // 故 data 实际分配 2GB（覆盖 stride=8），stride=32 单独用小 N 跑（见下）
    const long long M = (long long)256 * 1024 * 1024;    // 256M floats = 1GB

    // ========================================================================
    printf("========== Part 1: 合并 vs 跨步（1GB float）==========\n");
    // ========================================================================
    {
        // data 分配 M*8（2GB，覆盖到 stride=8 的最大下标）；8GB 显存 2GB+1GB 放得下。
        // stride=32 会访问到 data[M*32] 越界，所以 stride=32 用 N=M/32 的小规模
        float *d_data, *d_out;
        CUDA_CHECK(cudaMalloc(&d_data, M * 8 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_data, 0, M * 8 * sizeof(float)));

        StrideCtx ctx = { d_data, d_out, M, 1 };
        float bw1 = 0;                               // stride=1 的带宽基准
        int strides[3] = {1, 2, 4};
        for (int s = 0; s < 3; s++) {
            ctx.stride = strides[s];
            float ms = timerAvg(launchStride, &ctx, REPEAT);
            double gb = M * 4.0 / 1e9;               // 有效读出的字节
            double bw = gb / (ms * 1e-3);
            if (s == 0) bw1 = bw;
            printf("stride=%-2d : %7.3f ms  %7.1f GB/s  (%.1fx vs stride=1)\n",
                   strides[s], ms, bw, bw1 / bw);
        }

        // stride=32：缩小 N 保证 data[i*32] 最大下标 N*32 <= M*8
        ctx.n = M / 32;
        {
            float ms = timerAvg(launchStride, &ctx, REPEAT);
            double gb = ctx.n * 4.0 / 1e9;
            double bw = gb / (ms * 1e-3);
            printf("stride=32 : %7.3f ms  %7.1f GB/s  (%.1fx vs stride=1)  [N/32 规模]\n",
                   ms, bw, bw1 / bw);
        }
        CUDA_CHECK(cudaFree(d_data)); CUDA_CHECK(cudaFree(d_out));
    }

    // ========================================================================
    printf("\n========== Part 2: AoS vs SoA（1024 万粒子，只操作 x）==========\n");
    // ========================================================================
    {
        const long long PN = (long long)1024 * 1024 * 10;   // 1024 万粒子
        ParticleAoS* d_aos;
        float* d_soa;
        CUDA_CHECK(cudaMalloc(&d_aos, PN * sizeof(ParticleAoS)));
        CUDA_CHECK(cudaMalloc(&d_soa, PN * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_aos, 0, PN * sizeof(ParticleAoS)));
        CUDA_CHECK(cudaMemset(d_soa, 0, PN * sizeof(float)));

        AoSCtx ctxA = { d_aos, PN };
        SoACtx ctxS = { d_soa, PN };

        float msA = timerAvg(launchAoS, &ctxA, REPEAT);
        float msS = timerAvg(launchSoA, &ctxS, REPEAT);
        double gb = PN * 4.0 / 1e9;                  // 有效字节：只有 x 被用
        printf("AoS (stride=4): %7.3f ms  %7.1f GB/s 有效带宽\n", msA, gb / (msA * 1e-3));
        printf("SoA (stride=1): %7.3f ms  %7.1f GB/s 有效带宽  → %.2fx\n",
               msS, gb / (msS * 1e-3), msA / msS);

        CUDA_CHECK(cudaFree(d_aos)); CUDA_CHECK(cudaFree(d_soa));
    }

    // ========================================================================
    printf("\n========== Part 3: Bank Conflict vs Padding（1024x1024 列读）==========\n");
    // ========================================================================
    {
        const int SZ = 1024;                          // 32x32 tile × 1024 个 block
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, SZ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, SZ * sizeof(float)));
        float* h_in = (float*)malloc(SZ * sizeof(float));
        for (int i = 0; i < SZ; i++) h_in[i] = (float)i;
        CUDA_CHECK(cudaMemcpy(d_in, h_in, SZ * sizeof(float), cudaMemcpyHostToDevice));

        BCCtx ctx = { d_in, d_out };
        float msC = timerAvg(launchConflict, &ctx, 200);   // 小 kernel 多跑几轮
        float msP = timerAvg(launchPadded,   &ctx, 200);

        // 验证两版结果一致
        float* h_out = (float*)malloc(SZ * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_out, SZ * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int ty = 0; ty < TILE; ty++)
            for (int tx = 0; tx < TILE; tx++)
                if (h_out[ty * TILE + tx] != h_in[tx * TILE + ty]) ok = false;

        printf("无 Padding（32-way 冲突）: %7.4f ms\n", msC);
        printf("+1 Padding（无冲突）     : %7.4f ms  → %.2fx\n", msP, msC / msP);
        printf("结果一致性               : %s\n", ok ? "通过" : "失败");

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        free(h_in); free(h_out);
    }

    // ========================================================================
    printf("\n========== Part 4: 标量 vs float4 拷贝（1GB）==========\n");
    // ========================================================================
    {
        float *d_src, *d_dst;
        CUDA_CHECK(cudaMalloc(&d_src, M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dst, M * sizeof(float)));
        // cudaMemset 按字节填充（memset(p,1) 会得到 0x01010101 而非 1.0f），
        // 校验目的用"源模式=元素下标"更直观：拷贝后 dst[i] 应等于 i
        {
            float* h_init = (float*)malloc(M * sizeof(float));
            for (long long i = 0; i < M; i++) h_init[i] = (float)i;
            CUDA_CHECK(cudaMemcpy(d_src, h_init, M * sizeof(float),
                                  cudaMemcpyHostToDevice));
            free(h_init);
        }

        CopyCtx ctx = { d_src, d_dst, M, 0 };
        float msS = timerAvg(launchScalar, &ctx, REPEAT);
        float msV = timerAvg(launchVec, &ctx, REPEAT);

        double gb = M * 4.0 * 2 / 1e9;               // 读+写
        printf("标量 float  (LDG.32 x1): %7.3f ms  %7.1f GB/s\n", msS, gb / (msS * 1e-3));
        printf("float4      (LDG.128 x1): %7.3f ms  %7.1f GB/s  → %.2fx\n",
               msV, gb / (msV * 1e-3), msS / msV);

        // 正确性抽查：dst[999997..1000000] 应等于自身下标（float4 跑的是最后一次）
        float* h = (float*)malloc(4 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h, d_dst + 999997, 16, cudaMemcpyDeviceToHost));
        bool ok = (h[0] == 999997.0f && h[1] == 999998.0f &&
                   h[2] == 999999.0f && h[3] == 1000000.0f);
        printf("float4 数据校验           : %s\n", ok ? "通过" : "失败");

        CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_dst)); free(h);
    }

    // ========================================================================
    printf("\n========== Part 5: 矩阵转置三版本（8192x8192 = 256MB）==========\n");
    // ========================================================================
    {
        const int W = MAT_N;
        size_t bytes = (size_t)W * W * sizeof(float);
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));

        // 初始化：in[r][c] = r*W+c（转置后 out[c][r] 应等于 r*W+c，好验证）
        float* h_in = (float*)malloc(bytes);
        for (int r = 0; r < W; r++)
            for (int c = 0; c < W; c++)
                h_in[(size_t)r * W + c] = (float)(r * W + c);
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        TransCtx ctx = { d_in, d_out, W };

        // 拷贝基线（不转置只拷贝，代表理想带宽上限）
        CopyCtx cctx = { d_in, d_out, (long long)W * W, 0 };
        float msCopy = timerAvg(launchScalar, &cctx, REPEAT);

        float msN = timerAvg(launchNaive,  &ctx, REPEAT);
        float msS = timerAvg(launchShared, &ctx, REPEAT);
        float msF = timerAvg(launchF4,     &ctx, REPEAT);

        // 正确性验证：转置性质 out[c][r] == in[r][c] == r*W+c。
        // 三个探针读 out 的（行,列）：(5,0)、(0,5)、(W-1,W-1)
        //   out[5][0]     = in[0][5]     = 0*W+5       = 5
        //   out[0][5]     = in[5][0]     = 5*W+0       = 40960
        //   out[W-1][W-1] = in[W-1][W-1] = (W-1)*W+W-1 = 67108863（主对角线不变）
        // 注意最后一个不是 (W-1)^2 —— 线性下标 r*W+c 在主对角线上等于 (W-1)*W+(W-1)
        float* h_out = (float*)malloc(3 * sizeof(float));
        bool okN = true, okS = true, okF = true;
        // probes[k] = 行*W + 列（读 out）；expect[k] = 对应 in 的线性下标值
        size_t probes[3] = {(size_t)5 * W + 0, (size_t)0 * W + 5, (size_t)(W - 1) * W + (W - 1)};
        float expect[3] = {5.0f, 5.0f * W, (float)((size_t)(W - 1) * W + (W - 1))};

        launchNaive(&ctx); CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(h_out + i, d_out + probes[i], 4, cudaMemcpyDeviceToHost));
            if (fabsf(h_out[i] - expect[i]) > 0.5f) okN = false;
        }
        printf("v1 探针值: %.0f %.0f %.0f（期望 %.0f %.0f %.0f）\n",
               h_out[0], h_out[1], h_out[2], expect[0], expect[1], expect[2]);
        launchShared(&ctx); CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(h_out + i, d_out + probes[i], 4, cudaMemcpyDeviceToHost));
            if (fabsf(h_out[i] - expect[i]) > 0.5f) okS = false;
        }
        printf("v2 探针值: %.0f %.0f %.0f\n", h_out[0], h_out[1], h_out[2]);
        launchF4(&ctx); CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; i++) {
            CUDA_CHECK(cudaMemcpy(h_out + i, d_out + probes[i], 4, cudaMemcpyDeviceToHost));
            if (fabsf(h_out[i] - expect[i]) > 0.5f) okF = false;
        }
        printf("v3 探针值: %.0f %.0f %.0f\n", h_out[0], h_out[1], h_out[2]);

        double gb = bytes * 2 / 1e9;
        printf("纯拷贝基线（理想带宽）    : %7.3f ms  %7.1f GB/s\n", msCopy, gb / (msCopy * 1e-3));
        printf("v1 朴素转置               : %7.3f ms  %7.1f GB/s  验证:%s\n",
               msN, gb / (msN * 1e-3), okN ? "通过" : "失败");
        printf("v2 共享内存+Padding       : %7.3f ms  %7.1f GB/s  (%.2fx)  验证:%s\n",
               msS, gb / (msS * 1e-3), msN / msS, okS ? "通过" : "失败");
        printf("v3 共享内存+float4        : %7.3f ms  %7.1f GB/s  (%.2fx)  验证:%s\n",
               msF, gb / (msF * 1e-3), msN / msF, okF ? "通过" : "失败");

        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        free(h_in); free(h_out);
    }

    timerFree();
    printf("\n全部测试完成！\n");
    return 0;
}

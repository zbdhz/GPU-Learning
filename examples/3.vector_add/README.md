# 3.vector_add — 第一个实用 Kernel：向量加法全流程

> 教程对应：AIInfraGuide《[1.4 第一个实用 Kernel：向量加法](file:///home/gtx/data/AIInfraGuide/docs/guides/模块二-CUDA编程与算子优化/1.4-第一个实用Kernel.md)》

## 覆盖内容

| Part | 主题 | 教程章节 |
|---|---|---|
| 1 | 五步标准流程 + CPU/GPU 对比 + CUDA Event 计时 + 有效带宽 | 1、3、5 |
| 2 | Grid-stride Loop 版本（固定线程数处理任意 N） | 练习 8.3 |
| 3 | SAXPY：`Y = α·X + Y`（BLAS 基础操作，原地更新） | 练习 8.2 |
| 4 | 向量点积：Block 内树形规约 + `atomicAdd` 全局累加 | 练习 8.1 |
| 5 | 串行传输 vs 分块多 Stream 重叠传输（`cudaMemcpyAsync` + pinned 内存） | 7.3 |

## 一键运行

```bash
cd examples/3.vector_add
./run.sh            # 配置 + 构建 + 运行
./run.sh rebuild    # 清缓存重新配置
./run.sh clean      # 清理
```

## 核心概念

### 五步标准流程（Part 1 的骨架）

```
cudaMalloc ──► cudaMemcpy(H2D) ──► kernel<<<grid,block>>> ──► cudaMemcpy(D2H) ──► cudaFree
  分配显存       数据上卡           并行计算                  结果回传          释放
```

### 边界检查为什么必须有

N=1600 万、block=256 → grid=65536，恰好整除。但若 N=1000：grid=4 → 1024 线程，多出的 24 个线程 `idx >= N`，不检查就越界写显存。**任何 N 都写 `if (idx < N)` 是肌肉记忆**。

### 算术强度：为什么加法“快”却不是真快

每个元素：1 次加法 / 12 字节访存（读 A + 读 B + 写 C）→ **0.083 FLOP/Byte**，纯内存密集型。Kernel 跑得再快也卡在显存带宽上；而含 PCIe 传输的端到端时间更是纯 Kernel 的几十倍——**数据应尽量留在 GPU 上串联多个 Kernel**。

### 点积的两级规约（Part 4，本例最有含金量的部分）

```
16384 线程的部分和
   │ ①每线程 Grid-stride 累加自己的部分和
   ▼
共享内存 sdata[256]
   │ ②树形规约：256→128→64→…→1（每轮活跃线程减半，__syncthreads 同步）
   ▼
每 Block 1 个结果 ──③──► atomicAdd(result, …) ──► 最终点积
```

- `__syncthreads()`：Block 内栅栏，每轮规约后必须等所有人，否则读到旧值
- `atomicAdd`：读-改-写不可分割。多个 Block 并发累加同一地址，没有原子性就会丢失更新
- 浮点加法不满足结合律 → GPU 树形顺序 vs CPU 顺序结果有微小差异，验证用**相对误差**（1e-4）而非绝对

## 实测结果（RTX 2080 Ti，N = 2^24，2026-11-19 实跑）

```
CPU 单核        :   31 ms
GPU 仅 Kernel   :  0.41 ms   (加速 128.5x)
GPU 端到端含传输:   60 ms     (加速 0.9x —— PCIe 主导，甚至比 CPU 慢！)
Kernel 有效带宽 :  487 GB/s  (理论峰值 616 GB/s 的 79%)
点积相对误差    :  3.4e-07   (树形规约 vs 顺序累加，远小于 1e-4 容差)
--- Part 5 传输优化 ---
串行（memcpy 同步）: 24.70 ms
重叠（8 流 Async） : 18.52 ms  (快 1.33x)
```

对照教程 5.4 节 RTX 4090 参考值（CPU 25ms / Kernel 0.3ms / 含传输 8ms）：本机 2080 Ti 的 Kernel 加速 128 倍很可观，但 PCIe 3.0 x16 拖到端到端 0.9x——**单次搬运场景 GPU 甚至不敌 CPU，这就是教程 7.3 节"传输占比 96%"的现身说法**。

### Part 5：传输与计算重叠（多 Stream）

Part 1 里端到端被 PCIe 拖垮（0.9x），Part 5 演示教程 7.3 节的解法：

```
串行:  [上传0][算0][回传0][上传1][算1][回传1]...   PCIe 和 SM 轮流闲着
重叠:  stream0: [上传0][算0][回传0]
       stream1:          [上传1][算1][回传1]       上传块1 与 计算块0 并行
```

三个必备要素（缺一不可）：

1. **`cudaMemcpyAsync`**：立即返回的异步拷贝，DMA 引擎与 SM 并行工作
2. **Pinned 内存（`cudaMallocHost`）**：异步拷贝的**硬性要求**——pageable 内存会退化为同步行为，重叠完全失效
3. **多个 Stream**：流内按序、流间无依赖，硬件才能交叉调度

**为什么只有 1.33 倍而不是理论 2 倍**：本例 kernel 仅占 6% 时间（0.4ms/24.7ms），能藏进传输缝隙的只有这部分——Amdahl 定律的又一次现身。计算占比越大，重叠收益越接近 2 倍；同时 8 块切分让每块传输的 DMA 效率略降（小块开销），这是粒度权衡。

## 验证策略

| Part | 参考实现 | 容差 |
|---|---|---|
| 1/2 向量加法 | CPU 循环逐元素比对 | 绝对 1e-5（同序加法，应严格相等） |
| 3 SAXPY | CPU 计算 `α·X+Y` | 绝对 1e-4 |
| 4 点积 | CPU double 累加 | **相对** 1e-4（规约顺序不同） |
| 5 串行/重叠传输 | 与 Part 1 相同的 CPU 结果逐元素比对 | 绝对 1e-5 |

## 开发中实际踩的坑

**Part 4 首次运行结果错得离谱**（GPU 2097 万 vs CPU 401）：Part 3 的 SAXPY 把 `d_B` **原地更新**成了 `2.5X+Y`，Part 4 点积直接复用了这块脏显存。教训：**原地更新（in-place）省显存，但同一 buffer 的后续使用者必须知道数据已变**——多 kernel 串联时要么各自准备输入，要么明确数据流。已修复（Part 4 前重新上传原始 A/B，源码有注释）。

## 目录结构

```
3.vector_add/
├── CMakeLists.txt   # -O3（性能基准开满优化），其余同前例
├── vector_add.cu    # 4 个 kernel + CPU 基线（详细注释）
├── run.sh           # 一键编译运行
├── README.md        # 本文档
└── build/           # 构建产物（已 gitignore）
```

## 下一步

- 多 Stream 重叠在"统一内存/Pinned Memory"示例中继续深挖（如双缓冲 Double Buffering）
- 规约的进阶（warp shuffle `__shfl_down_sync` 替代共享内存树形规约）见第 3 章专题示例

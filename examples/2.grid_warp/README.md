# 2.grid_warp — Grid 多维度分组与 Warp 分歧实验

> 教程对应：AIInfraGuide《[1.2 CUDA 编程模型](file:///home/gtx/data/AIInfraGuide/docs/guides/模块二-CUDA编程与算子优化/1.2-CUDA编程模型.md)》第 2、3、6、7 节

## 这个例子演示什么

| Part | 主题 | 核心知识点 |
|---|---|---|
| 1 | 1D Grid-stride Loop | 数据量 > 线程总数时的标准遍历模式（教程 3.4） |
| 2 | 2D Grid/Block 矩阵加法 | 行列映射 + 边界检查（教程 3.3/7.1） |
| 3 | 3D Grid 批处理 | z 维 = batch（教程 7.3） |
| 4 | **Warp 分歧性能对比** | 同一分支条件，Warp 内分歧 vs Warp 对齐，CUDA Event 计时（教程 6.2） |

## 一键运行

```bash
cd examples/2.grid_warp
./run.sh            # CMake 配置 + 构建 + 运行
./run.sh rebuild    # 清缓存重新配置
./run.sh clean      # 清理构建产物
```

## Part 4：Warp 分歧为什么慢（本例核心）

**Warp 是 SM 调度的最小单位：32 个连续线程锁定执行同一条指令（SIMT）。**

### 分歧的代价

```cuda
// kernel A：同一 Warp 的 32 个线程，偶数 lane 走 A、奇数 lane 走 B
if (threadIdx.x % 2 == 0) { pathA(); }   // 16 个线程干活，16 个空转
else                      { pathB(); }   // 再换 16 个干活，另外 16 个空转
```

硬件没有“半个 Warp”的概念，只能让整个 Warp **先串行执行 pathA、再串行执行 pathB**——两个分支的指令时间全部要付，理论减速接近 2 倍。

### 无分歧的写法

```cuda
// kernel B：分支条件以 32 为边界切
if ((threadIdx.x / 32) % 2 == 0) { pathA(); }  // Warp 0 整体走 A
else                            { pathB(); }  // Warp 1 整体走 B
```

只要**同一 Warp 内的 32 个线程走相同分支**就没有惩罚；分歧发生在 Warp 内部，不同 Warp 之间各走各的完全免费。

### 实测结果（RTX 2080 Ti）

```
kernel A (Warp 内奇偶分歧): 0.806 ms
kernel B (Warp 对齐无分歧) : 0.410 ms
分歧版本慢 1.97 倍 —— 同一 Warp 的 32 线程被迫串行执行两个分支
```

1.97 倍几乎打满理论上限（2 倍）：两个分支各占 Warp 的一半线程，硬件先串行跑完 pathA 全部 512 条 FMA、再串行跑 pathB，指令吞吐直接减半。

### 设计这个实验踩的三次坑（重要！）

想真正测出 Warp 分歧，分支体设计比想象的讲究：

| 版本 | 结果 | 原因 |
|---|---|---|
| v1：分支体 8 次 FMA | 1.01 倍 | 计算太轻，0.004ms 完成，分歧开销被启动/调度噪声淹没 |
| v2：分支体 512 FMA 但两分支**代码相同** | 1.00 倍 | nvcc 发现 A/B 路径等价，把 if/else 优化成 **predication（谓词执行）**，硬件层面根本没分支了 |
| v3：512 FMA 且两分支乘数不同 | **1.97 倍** | 编译器被迫保留真实分支，分歧代价显现 |

顺带的教训：**“测不到分歧”≠“没有分歧”，可能只是被编译器谓词化或被噪声掩盖**。写 benchmark 前先看 SASS（`cuobjdump -sass`）确认分支结构还在。

另一个公平性细节：两个 kernel 都用 **Grid-stride 循环**覆盖全部 400 万元素（`<<<1024,256>>>` 只有 26 万线程，若直接 `if (i>=n) return` 会漏掉 94% 的数据，两边却都"看起来跑完了"）。

## 分歧机制图解

一个 Warp 的 32 个 lane 被"奇偶分支"劈成两半时，硬件的执行时间线：

```
kernel B（无分歧）—— 两个 Warp 并行，各走各的：
  Warp 0 (lane 0~31 全走A):  [==== pathA 512 FMA ====]
  Warp 1 (lane 32~63 全走B): [==== pathB 512 FMA ====]   ← 同时执行，互不干扰

kernel A（Warp 内分歧）—— 单个 Warp 被串行化：
  Warp 0 (lane 0~31 一半走A一半走B):
       [==== pathA 512 FMA ====][==== pathB 512 FMA ====]
        前半: 走B的lane全部空转 ↗    后半: 走A的lane全部空转
```

kernel A 的每个 Warp 花了 2 倍时间，kernel B 的两个分支在两个 Warp 里**同时**跑——这就是实测 1.97 倍差距的来源。

## 多维索引速查

| 维度 | 索引公式 | 本例 |
|---|---|---|
| 1D | `idx = blockIdx.x*blockDim.x + threadIdx.x` | `<<<1024,256>>>` + stride 循环 |
| 2D | `col = bx*dx+tx; row = by*dy+ty; idx = row*cols+col` | block(16,16) grid(4,3) 覆盖 61×37 |
| 3D | `batch = blockIdx.z` | grid(2,4,4) 处理 4 张 32×32 |

**边界检查必须写**：Part 2 故意用 61×37（非 16 倍数），`grid((61+15)/16, (37+15)/16) = (4,3)` 会多出 3×11 个越界线程，靠 `if (row < rows && col < cols)` 跳过。

## 计时工具：CUDA Event

```cuda
cudaEventRecord(beg);                    // GPU 时间轴打点
for (...) launch(...);                   // 提交 N 次 kernel
cudaEventRecord(end);
cudaEventSynchronize(end);               // 等 GPU 真做完
cudaEventElapsedTime(&ms, beg, end);     // 两事件间毫秒数
```

为什么不用 CPU 的 `clock()`：kernel 启动是异步的，CPU 计时会把“提交时间”和“执行时间”混在一起；Event 打在 GPU 时间轴上才准。

## 目录结构

```
2.grid_warp/
├── CMakeLists.txt      # 与 1.hello 相同结构，新增 -lineinfo 选项
├── grid_warp_demo.cu   # 四部分演示源码（详细注释）
├── run.sh              # 一键编译运行
├── README.md           # 本文档
└── build/              # 构建产物（已 gitignore）
```

## 下一步

- 用 `ncu --set full ./build/grid_warp_demo` 观察两个 kernel 的 `smsp__thread_inst_executed_per_inst_executed`（分支指令串行度指标）
- 教程 6.3 节的 `__shfl_sync` Warp 级原语将在后续规约求和示例中演示

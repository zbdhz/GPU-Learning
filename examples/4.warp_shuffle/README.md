# 4.warp_shuffle — Warp Shuffle 全家福测试

> 教程对应：AIInfraGuide《[2.1 Warp 与执行模型](file:///home/gtx/data/AIInfraGuide/docs/guides/模块二-CUDA编程与算子优化/2.1-Warp与执行模型.md)》第 4~6 节

## 覆盖内容

| Part | 主题 | 教程章节 |
|---|---|---|
| 1 | 四种 Shuffle 指令行为验证（含边界行为与 XOR 蝶形配对） | 4.2 / 4.3 |
| 2 | Warp 内广播 + Warp 蝶形求和（1~32 加出 528） | 4.4 / 4.5 |
| 3 | `width=8` 子 Warp 独立归约（一个 Warp 当 4 组用） | 5.4 |
| 4 | Warp Vote：`__all_sync` / `__any_sync` / `__ballot_sync` | 5.1 |
| 5 | Block 级归约实战：纯共享内存 vs Shuffle 版性能对比 | 6.1 / 6.2 |

## 一键运行

```bash
cd examples/4.warp_shuffle
./run.sh            # 配置 + 构建 + 运行
./run.sh rebuild    # 清缓存重新配置
./run.sh clean
```

## 四种 Shuffle 速查（Part 1 实测验证）

| 指令 | lane i 拿到谁 | 实测行为 |
|---|---|---|
| `__shfl_sync(mask, v, 5)` | lane 5 的值 | 全员拿到 5 ✅ |
| `__shfl_up_sync(mask, v, 4)` | `i-4` | 边界 lane<4 保持原值 ✅ |
| `__shfl_down_sync(mask, v, 4)` | `i+4` | 边界 lane>27 保持原值 ✅ |
| `__shfl_xor_sync(mask, v, 1)` | `i^1` | 0↔1, 2↔3 两两互换 ✅ |
| `__shfl_xor_sync(mask, v, 16)` | `i^16` | 前后半交叉配对 ✅ |

记忆点：**down 配 offset=16,8,4,2,1 做"折叠"归约；xor 配同序列做"蝶形"归约**（所有人最终都有份 vs 只有 lane 0 有结果）。

## ⚠️ 实测踩坑：mask 契约（Part 4 首跑全错）

**错误写法**（首跑 `__any_sync` 得 0、`__ballot` 得 0x00000000）：

```cpp
if (lane == 0) *anyResult = __any_sync(0xFFFFFFFF, pred);   // ❌
```

只有 lane 0 执行了 vote，mask 却要求全体 32 线程参与——**缺席者的行为未定义，实测返回值全 0**。这正是 mask 参数存在的意义：集体操作必须"人到齐"。

**正确写法**：全员先投票落寄存器，lane 0 再写回：

```cpp
int anyR = __any_sync(0xFFFFFFFF, pred);    // ✅ 32 人都执行
if (lane == 0) *anyResult = anyR;
```

修复后 `__any_sync=1`、`__ballot=0xFFF00000` 全部符合预期。**这条规则同样适用于所有 `__shfl_*_sync`**——永远不要把 Shuffle/Vote 放进只有部分线程进入的分支里。

## 实测结果（RTX 2080 Ti，2026-11-20 实跑，Part 5 为 50 次平均）

```
纯共享内存树形归约  :  3.34 ms   160.8 GB/s
Warp Shuffle 归约   :  1.39 ms   386.2 GB/s   → 加速 2.40x
```

对照教程 6.2 表格（共享内存 1024B/8 次同步 → 128B/1 次同步）：

- 纯共享内存版：256 线程树形折半要 **8 轮 × `__syncthreads()`**，后半程活跃线程越来越少，同步开销占比急增，带宽只剩 160GB/s
- Shuffle 版：8 个 Warp **各自 5 步寄存器级归约**（零同步零共享内存），只有"Warp 和汇总"那一步碰共享内存（1 次同步），带宽 386GB/s（峰值的 63%）

归约是典型的"尾部长尾"问题——Shuffle 把最后 8 轮同步砍成 1 轮，正好砍在长尾上，所以 2.4 倍的加速比教程估算的还大。

## Part 5 两版实现的关键差异

```
纯共享内存:  256值 → shared → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1     (8 次 syncthreads)
Shuffle 版:  256值 → 8 个 Warp 各自 shuffle 5 步 → 8 个和 → 1 个 Warp 再 shuffle → 1
                      ↑ 无同步                                ↑ 仅 1 次 syncthreads
```

源码 [warp_shuffle_demo.cu](file:///home/gtx/dev/GPU-Learning/examples/4.warp_shuffle/warp_shuffle_demo.cu) 中两个 kernel 相邻摆放，逐行注释可直接对照。

## 计时方法说明

Part 5 使用 `benchKernel()`：**预热 1 次 + 50 次取平均**（一对 Event 包住循环），比单次测量稳定；这与示例 2/3 的单次口径不同，教程 6.2 的量化也建议这么做。

## 目录结构

```
4.warp_shuffle/
├── CMakeLists.txt        # -O3，同前例自动架构探测
├── warp_shuffle_demo.cu  # 5 个 Part（详细注释，含 mask 契约反面教材）
├── run.sh                # 一键编译运行
├── README.md             # 本文档
└── build/                # 构建产物（已 gitignore）
```

## 下一步

- `__shfl_xor_sync` 的蝶形特性（全员持有结果）是 AllReduce/FFT/扫描（Scan）的基础
- 第 3 章 Reduce 专题将把 Part 5 的 Shuffle 归约扩展成多 Block 两阶段完整版

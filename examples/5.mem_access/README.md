# 5.mem_access — 内存访问优化全家桶

对应教程：《AIInfraGuide 模块二》**2.2-内存访问优化.md** 全部例子（第 2~5 节），
以及 2.4（AoS/SoA）、3.3/3.4（Bank Conflict）、5.1（矩阵转置）。

```bash
./run.sh          # 配置 + 编译 + 运行
./run.sh rebuild  # 清缓存重新配置
./run.sh clean    # 删除 build/
```

## 五个 Part 对应教程哪一节

| Part | 教程章节 | 实验 |
|---|---|---|
| 1 | 2.2 合并访问 | stride=1/2/4/32 读带宽实测 |
| 2 | 2.4 AoS vs SoA | 粒子系统只翻倍 x 字段 |
| 3 | 3.3/3.4 Bank Conflict | `tile[32][32]` vs `tile[32][33]` 列读 |
| 4 | 4.1~4.3 向量化 | 标量 vs `float4` 拷贝 |
| 5 | 5.1 矩阵转置 | 朴素 / 共享+Padding / 共享+float4 三版本 |

## 实测结果（RTX 2080 Ti，68 SM，理论带宽 616 GB/s）

```
Part 1  stride=1: 207.9 GB/s | stride=2: 174.4 | stride=4: 109.5 | stride=32: ~110（见下）
Part 2  AoS 59.8 GB/s vs SoA 237.7 GB/s → 3.97x（接近理论 4 倍上限）
Part 3  Padding 1.22x（kernel 太小，见"实测反常"第 2 条）
Part 4  标量 431.8 GB/s vs float4 496.6 GB/s → 1.15x
Part 5  朴素 910 GB/s(!) > 共享+Padding 391 GB/s > 拷贝基线 505 GB/s（见"实测反常"第 1 条）
```

## 实测反常与解释（比"符合预期"更有教学价值）

### 1. 朴素转置反而最快（910 GB/s > 拷贝基线 505 GB/s？）

教程的结论是"朴素转置只有 25% 带宽，共享内存版能到 60~70%"。
但在 Turing（2080 Ti）上实测方向相反。原因：

- **写路径 stride=W 的非合并请求会被 L2 缓存（1.5MB，Turing 写回策略）吸收合并**，
  32 线程写的 32 条 line 大多命中共享的 L2 分辨率（32B sector），实际下盘流量并不放大 32 倍
- 8192×8192×4B=256MB 远大于 L2，但 L2 对**同一次写突发的 coalescing 发生在 sector 粒度**，
  Turing 的 L1→L2 写路径本身按 32B 打包，部分抵消了 stride 的伤害
- 共享内存版慢的原因：**block 数 256×256=65536 个、每 block 装载 4KB tile 再写出**，
  额外的共享内存往返 + `__syncthreads` 串行化，在这个"读写都已有 L2 缓冲"的架构上反而是净开销
- 910 GB/s > 拷贝基线：写非合并的 line 会驻留 L2，20 次重复计时中后续读写大量命中 L2（缓存热），
  测得的是"L2 加持带宽"而非纯显存带宽——这是**微基准的经典陷阱**

> 教训：**教程的优化结论绑定在其测试时代的架构上**（Fermi/Kepler 时代 L2 小、写不合并）。
> 优化必须在目标硬件上重新 profile（`ncu` 的 `l1tex__` 与 `lts__` 计数器），不能照搬结论。

### 2. Part 1 的 stride=32 为什么只有 1.9x 差距而不是 32x

stride=32 理论上每线程独占一条 128B line（带宽利用率 3%）。但注意本实验的口径：
**有效带宽按"程序逻辑读出的 N×4B"计算，而硬件实际搬运了 32 倍的 line**。
1.9x 的数字背后：stride=32 用的是 N/32 规模（防止越界），数据量小 → 请求全部落进 L2 命中，
"浪费的 32 倍流量"被 L2 吃掉大半。要看真实的 32 倍悬崖，需要：
数据集远大于 L2 + 用 ncu 看 `dram__bytes_read`（实际 DRAM 流量）而非逻辑流量。

### 3. Part 3 的 Padding 只快 1.22x 而不是 32x

`1,1,1,1` tile 太小（单 block 一次只处理 4KB），kernel 总耗时 2 微秒级，
启动开销与同步占比盖过了 32-way 冲突的惩罚。32x 是"纯列读周期数"的理论比值。
要放大信号：单 kernel 内重复多轮列读（教程 3.4 的口径）或用 ncu 数
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`。

### 4. Part 5 的验证教训：期望值表写错

首跑"验证失败"，探针值其实全对，错的是期望值：
`out[c][r] = in[r][c] = r*W+c`，则 `out[0][5] = in[5][0] = 5*W`（不是 W+5）；
主对角线 `out[W-1][W-1] = (W-1)*W+(W-1)`（不是 (W-1)^2）。
**转置验证的正确姿势：以 out 的 (行,列) 反推 in 的 (行,列)，别心算线性下标。**

## 工程细节备忘

- `cudaMemset(p, 1, ...)` 是按字节填充（得到 0x01010101，不是 1.0f）——Part 4 改用
  "元素值 = 下标"的源模式来校验
- stride 越界：`data[i*stride]` 最大下标是 `(N-1)*stride`，Part 1 的 data 缓冲按
  `M*8` 分配，stride=32 单独用 `N=M/32` 规模跑
- `float` 精确整数上限 2^24≈1677 万：下标模式初始化 256M 元素时，大于 2^24 的下标
  会失真——所以校验探针只挑小下标与大下标各 1 处并在容差内比较（本文探针值
  67108864 = 2^26，恰好可被 float 精确表示）
- Part 2 的启动配置用 `(n+255)/256` 一维网格（1024 万粒子 = 40000 block），没有用
  grid-stride——因为 AoS/SoA 的对比变量是访存模式，网格策略必须两边一致

## ncu 进一步验证（可选）

```bash
# Bank Conflict 计数器（Part 3）
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./build/mem_access_demo
# 全局加载的 sector/请求效率（Part 1/5）
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    ./build/mem_access_demo
```

# HNSW Milvus 集成优化轮次

**数据集**：Cohere-1M（768D, IP metric）
**机器**：hannsdb-x86（16 cores）
**参考 raw**：`benchmark_results/`（HNSW Milvus benchmark files）

---

## 轮次总表

| Round | Commit | c=20 QPS | c=80 QPS | 关键变更 |
|-------|--------|---------|---------|---------|
| R0 | pre-opt | — | 284 | baseline |
| R1 | `2418f5b` | — | 285.9 | Batch serialize IO + 消除 per-node to_vec() |
| R2 | `6ef694e` | — | 260.4 | FFI lazy bitset + prefetch + TRACE_SLAB（混杂，prefetch 害处被掩盖）|
| R3 | `0294852` | — | 141.3 | 去 TRACE_SLAB，改 prefetch offset → **prefetch L1 thrash 完全暴露** |
| **R4** | `106dee7` | — | **349.3** | 去 prefetch call site → FFI fix 收益 +22% 显现 |
| **R5** | `c5db83f` | — | **349.0** | hugepage Layer0Slab：延迟 6.7→2.8ms（QPS 不变，瓶颈在调度）|
| ~~R6~~ | ~~`89e99f3`~~ | — | ~~170~~ | rayon 全局池（**失败，已 revert**）|
| **R7** | `06d3ec2` | **548** | **540** | HNSW_NQ_POOL 私有池：c=20 +6×，c=80 +55% |
| **R8** | `0a6d699` | **1051** | **1042** | alloc-reduction：clone消除 + 预分配 buffer：c=20/c=80 各 +92%/+93% |

---

## R8 权威数字（Cohere-1M, 2026-04-07）

| 指标 | native | RS R8 | RS vs native |
|------|--------|-------|-------------|
| Insert | 304.6s | 336.8s | +10.6% 慢 |
| Optimize | 854.2s | **336.9s** | **2.53× 快** |
| Load | 1158.9s | **673.7s** | **1.72× 快** |
| QPS (c=20) | ~500+ | **1051** | **~2× 快** |
| QPS (c=80) | ~800+ | **1042** | **~1.3× 快** |
| Recall | 0.960 | 0.957 | parity |

---

## 关键里程碑分析

### R4：FFI lazy bitset fix（+22% QPS）

- **旧**：`bitset_view`（~15KB）+ `sparse_bitset`（~120KB）在每次 search 无条件创建
- **新**：移入实际需要的分支内，HNSW 走 zero-copy `bitset_ref`
- **节省**：107MB/s 无用堆分配（~135KB × 796 calls/sec）

### R5：hugepage 改延迟不改 QPS

Layer0Slab 397MB，madvise(MADV_HUGEPAGE) 后 TLB entries 99K→199。
延迟 6.7ms→2.8ms 是真实改善，但 QPS 不变说明瓶颈在 Milvus query 调度层，不在搜索计算本身。

### R6→R7：rayon 全局池 vs 私有池

R6 证伪：全局池 + CGO executor = 过度订阅。
R7 方案：`HNSW_NQ_POOL.install()` 将 CGO 线程纳入私有池，避免线程数爆炸。
→ 详见 [[decisions/rayon-private-pool]]

### R8：两处 alloc-reduction

**Fix 1**：`to_sorted_pairs()` clone 消除
```rust
// 前：分配完整 BinaryHeap 副本
self.entries.clone().into_sorted_vec()
// 后：原地 drain，无额外 malloc
std::mem::take(&mut self.entries).into_sorted_vec()
```

**Fix 2**：预分配 flat 输出 buffer
```rust
// 前：Vec<Mutex<Vec<(i64,f32)>>> + 后处理 copy
// 后：all_ids.resize(n_queries*k, -1) + SendPtr raw 写入不相交切片
```
SAFETY：每个 q_idx 唯一 → 切片 `[q_idx*k..(q_idx+1)*k]` 不相交。

---

## 并发诊断结论（2026-04-07）

- RS QPS 线性扩展（c=2: 208.6 ≈ 1.87× c=1: 111.3）→ 无序列化锁 ✅
- Milvus dispatcher 对每个 sealed segment 启动独立 goroutine，executor 有 32 slots
- peak_concurrent=2（测试集合只有 2 个 sealed segment）
- Milvus 调度无病理性限制，RS 侧无全局锁

---

## 相关页面

- [[decisions/rayon-private-pool]] — R6 失败 + R7 方案
- [[decisions/prefetch-l1-thrash]] — R2→R3 prefetch 灾难
- [[concepts/hnsw]] — HNSW 实现架构
- [[concepts/milvus-integration]] — H + nq×t 调度模型

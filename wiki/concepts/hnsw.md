# HNSW

**一句话定义**：Hierarchical Navigable Small World 图，RS 实现在 `src/faiss/hnsw.rs`（9255+ 行），Milvus 集成路径为 `hnsw_rust_node.cpp` → `knowhere_search_with_bitset()`。

---

## RS 实现特点

- Layer 0 用 **Layer0Slab**：flat 连续内存（hugepage 友好），`madvise(MADV_HUGEPAGE)` 后 TLB entries 99K→199
- 搜索并发：**HNSW_NQ_POOL**（私有 rayon ThreadPool，大小 `(hw-32).max(4)`）— 不用全局池
- 输出 buffer：预分配 `n_queries×k` flat array + `SendPtr` 不相交切片写入（zero Mutex contention）
- `VisitedList`：thread-local，Vec\<u16\> + generation counter，O(1) reset

---

## Milvus 集成优化历史

→ 详见 [[benchmarks/hnsw-milvus-rounds]]

| Round | Commit | 关键变更 | c=80 QPS |
|-------|--------|---------|---------|
| R0 | — | baseline | 284 |
| R1 | `2418f5b` | Batch serialize IO + 消除 per-node to_vec() | 285.9 |
| R2 | `6ef694e` | FFI lazy bitset + prefetch + TRACE_SLAB（混杂）| 260.4 |
| R3 | `0294852` | 去 TRACE_SLAB，改 prefetch offset | 141.3 ← 最差 |
| R4 | `106dee7` | 去 prefetch call site → FFI fix 收益显现 | 349.3 |
| R5 | `c5db83f` | hugepage Layer0Slab | 349.0（延迟 6.7→2.8ms） |
| R6 | `89e99f3` | rayon nq 并行（**已 revert**）| 170 ← 失败 |
| R7 | `06d3ec2` | **HNSW_NQ_POOL** 私有池 | c=20: 548, c=80: 540 |
| R8 | `0a6d699` | alloc-reduction（clone消除 + 预分配buffer） | **c=20: 1051, c=80: 1042** |

**R8 结论**：c=20/c=80 QPS 超越 native ~800 QPS（~1.3×）。

---

## 关键教训

### Prefetch 导致 L1 cache thrash（R2→R3 教训）

64 邻居 × 4 prefetch lines × 128 候选 = 32768 L1 cache fills/search。
L1 仅 32KB（512 cache lines）→ 2 个候选后完全蒸发。
→ **不要在 HNSW beam search 中做 prefetch**。详见 [[decisions/prefetch-l1-thrash]]

### rayon 全局池过度订阅（R6 教训）

CGO executor 线程 JOIN 全局 rayon 池 → 32 CGO线程 + N rayon 线程 → 过度订阅。
解法：私有池 + `install()` 模式。详见 [[decisions/rayon-private-pool]]

### Instant::now() 无条件调用（2026-03-22 修复）

`search_layer_idx_shared` 中 22.57% 时间在 `clock_gettime`（profile 代码未 gate）。
修复：44 处 `Instant::now()` 改为 `profile_enabled.then(Instant::now)`。
结果：15,043 → 17,319 QPS (+15.1%)，从 parity 变为 LEADING。

---

## Standalone 权威数字（SIFT-1M, x86, M=16, ef_c=200）

| ef | recall@10 | 单线程 QPS | batch parallel QPS |
|----|-----------|------------|-------------------|
| 60 | 0.9720 | 7,482 | **17,319** |
| 138 | 0.9945 | 3,578 | **17,517** |

**native 8T (ef=138, recall=0.952): 15,918 QPS**
- ef=60: +8.8% leading；ef=138: +10.1% leading

## Milvus 权威数字（Cohere-1M, hannsdb-x86, 2026-04-07 R8）

| 指标 | native | RS R8 | RS vs native |
|------|--------|-------|-------------|
| Insert | 304.6s | 336.8s | +10.6% 慢 |
| Optimize | 854.2s | 336.9s | **2.53× 更快** |
| Load | 1158.9s | 673.7s | **1.72× 更快** |
| QPS (c=80) | ~800+ | **1042** | **~1.3× 更快** |
| QPS (c=20) | ~500+ | **1051** | **~2× 更快** |
| Recall | 0.960 | 0.957 | parity |

---

## 相关页面

- [[decisions/rayon-private-pool]] — 为什么不能用全局 rayon pool
- [[decisions/prefetch-l1-thrash]] — prefetch 为什么适得其反
- [[decisions/optimization-log]] — 完整优化决策记录
- [[concepts/milvus-integration]] — Milvus 调度链分析
- [[machines/hannsdb-x86]] — 权威 benchmark 环境

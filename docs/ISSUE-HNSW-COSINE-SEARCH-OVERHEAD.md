# ISSUE: HNSW Cosine Search Path Overhead

**提交来源**: HannsDB 集成测试 (2026-03-23)
**场景**: 50K 向量 / 1536 维 / cosine / ef_search=32 / serial 搜索
**当前现象**: HannsDB (via knowhere-rs) p99=110ms vs zvec (C++ HNSW) p99=0.6ms，同算法差距 ~185x

---

## 问题 1: HNSW-COSINE-SCRATCH-001 ✅ FIXED (commit 94ca617)

**标题**: Cosine 搜索路径缺少 TLS scratch 复用

**优先级**: P1 → 已修复

**文件**: `src/faiss/hnsw.rs:5267–5369`

### 现象

`search_single()` 在 Cosine/IP 路径每次 query 都执行 `SearchScratch::new()`（第 5299 行），触发堆分配。L2 unfiltered 路径已有对应的快速路径 `search_single_l2_unfiltered_with_scratch`，接收外部 scratch 参数，由调用方（`search()` 入口）通过 TLS 复用。Cosine 路径没有对应实现。

```rust
// hnsw.rs:5278 — L2 有快速路径
if self.metric_type == MetricType::L2 && filter.is_none() {
    return self.search_single_l2_unfiltered(query, ef, k);
    // ↑ 内部走 TLS scratch 复用
}

// hnsw.rs:5299 — Cosine 路径：每次 query 重新分配
let mut scratch = SearchScratch::new();  // ← 问题所在
```

### 影响

在 VectorDBBench serial search（1000 次串行 query）场景下，每次查询都触发 `SearchScratch` 内部优先队列和候选集的堆分配与释放。高频调用时累积开销显著。

### 修复方向

1. 实现 `search_single_cosine_unfiltered_with_scratch(scratch: &mut SearchScratch)` 快速路径，与 `search_single_l2_unfiltered_with_scratch` 对称
2. 在 `search()` 入口对 Cosine + 无 filter 情况走 TLS scratch 复用路径：
   ```rust
   thread_local! {
       static COSINE_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
   }
   ```
3. 注意与 `brute_force_search` 小集合退化路径（第 5293 行）的边界兼容

### 验证

- `cargo test --lib hnsw` 全部通过
- x86 `bench_hnsw_cosine_build_hotpath_smoke` 搜索阶段 QPS 提升 > 5%

---

## 问题 2: HNSW-COSINE-QNORM-001 ✅ FIXED (commits 94ca617, f13b9ee)

**标题**: `distance_to_idx_cosine_dispatch` 每次调用重算 query norm

**优先级**: P1

**文件**: `src/faiss/hnsw.rs:1280–1295`

### 现象

```rust
fn distance_to_idx_cosine_dispatch(index: &HnswIndex, query: &[f32], idx: usize) -> f32 {
    let stored = &index.vectors[start..start + index.dim];
    let ip = simd::inner_product(query, stored);
    let q_norm_sq = simd::inner_product(query, query);  // ← 问题：每次调用都算
    let v_norm_sq = simd::inner_product(stored, stored);
    let q_norm = q_norm_sq.sqrt();
    let v_norm = v_norm_sq.sqrt();
    1.0 - ip / (q_norm * v_norm)
}
```

同一次 search 中，`query` 固定不变，但 `distance_to_idx_cosine_dispatch` 被调用 O(ef × graph_degree) 次，每次都对 1536-dim query 重复执行 `inner_product(query, query)`（约 1536 次 FP32 乘加）。

已有的 `cosine_vector_norm_for_idx_hot` 缓存了**向量**的 norm（`v_norm`），但**query norm**（`q_norm`）没有对应缓存。

### 影响

50K/1536/cosine，ef=32 场景下单次 query 的图搜索路径约访问数百个节点，`q_norm` 被重算数百次，每次 1536-dim dot product。这是纯冗余计算。

### 修复方向

1. 在 `search_single` 入口预算一次 query norm：
   ```rust
   let q_norm = simd::inner_product(query, query).sqrt();
   ```
2. 将 `q_norm` 携带进搜索层：通过参数传递，或放入 `SearchScratch`
3. `distance_to_idx_cosine_dispatch` 接收预算的 `q_norm`，跳过重算：
   ```rust
   1.0 - ip / (q_norm * v_norm)  // q_norm 由调用方传入
   ```
4. 注意 `q_norm == 0` 的边界处理保持不变

### 验证

- `test_hnsw_cosine_metric` 通过（语义不变）
- `test_cosine_distance_with_query_norm_matches_dispatch_path` 通过
- x86 `10K/1536/cosine` 搜索延迟下降可测

---

## 问题 3: HNSW-ALLOC-001

**标题**: `search()` 入口每次调用分配 `all_ids`/`all_dists` Vec

**优先级**: P2（次于上两项）

**文件**: `src/faiss/hnsw.rs:3623–3624`

### 现象

```rust
pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
    // ...
    let mut all_ids = vec![-1; n_queries * k];        // ← 每次分配
    let mut all_dists = vec![f32::MAX; n_queries * k]; // ← 每次分配
```

serial search 场景（n_queries=1, k=10）下每次搜索都分配两个小 Vec（10 个元素），并在函数返回时释放。

### 影响

单次分配开销小（10×i64 + 10×f32），但 1000 次串行 query 累积 2000 次堆分配。相比上两项是次要开销。

### 修复方向

提供接收外部 output buffer 的变体：
```rust
pub fn search_into(
    &self,
    query: &[f32],
    req: &SearchRequest,
    ids: &mut [i64],
    dists: &mut [f32],
) -> Result<()>
```
调用方（HannsDB adapter 或 benchmark）复用 buffer，避免重复分配。

### 前置条件

建议在 HNSW-COSINE-SCRATCH-001 和 HNSW-COSINE-QNORM-001 完成并取得 x86 数据后，再评估此项的实际收益。

---

## 背景：HannsDB 集成调用链

```
VectorDBBench Python
  └─ hannsdb_py::PyCollection::query()
       └─ hannsdb_core::db::search_with_ef()
            └─ hannsdb_index::KnowhereHnswIndex::search()
                 └─ knowhere_rs::HnswIndex::search()          ← 入口分配 (ALLOC-001)
                      └─ search_single()                       ← scratch 分配 (SCRATCH-001)
                           └─ distance_to_idx_cosine_dispatch() ← q_norm 重算 (QNORM-001)
```

当前 HannsDB p99=110ms 的估算构成（无 profiler，代码推断）：
- HNSW 图搜索本体（knowhere-rs cosine）：~70–90ms
- Python binding / PyDoc 构造：~8–15ms
- wrapper/mapping/lock：~8–20ms

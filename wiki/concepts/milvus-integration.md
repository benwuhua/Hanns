# Milvus 集成架构

**一句话定义**：knowhere-rs 通过 C++ shim 集成进 Milvus，替换 knowhere C++ 共享库，暴露相同 `IndexNode` 接口。

---

## 调用链

```
Milvus Go → SearchTask.Execute
  → segments.searchSegments (errgroup, per-segment goroutine)
    → cgo.Async → C.AsyncSearch
      → milvus::futures::getGlobalCPUExecutor()  (32 slots)
        → knowhere IndexNode::Search()
          → [RS shim] knowhere_search_with_bitset()
            → Rust PQFlashIndex / HnswIndex
```

---

## CGO Executor 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `maxReadConcurrentRatio` | 1.0 | milvus.yaml:550 |
| `MaxReadConcurrency` | 16 × 1 = 16 | hw_cores × ratio |
| `cgoPoolSizeRatio` | 2.0 | → AsyncSearch executor |
| **CGO_EXECUTOR_SLOTS** | **ceil(16×2.0) = 32** | 并发 FFI 调用上限 |

→ 理论上 N 个 segment 可全并发，每个 goroutine 分配一个 slot。

---

## Segment 模型

Milvus **per sealed segment** 构建和搜索索引，不是 per collection：

- 1M 向量 / BATCH=10K → ~25 个 sealed segment（各 ~40K 向量）
- 每次 search：所有 25 segment 被 goroutine 并发发起，但 CGO executor 侧实际并发度取决于 segment 状态

**关键诊断（2026-04-07）**：
- `peak_concurrent=2`（当时的 collection 实际只有 2 个 sealed segment）
- RS QPS 确实线性扩展：c=2 时 208.6 ≈ 1.87× c=1 的 111.3 — 无序列化锁

---

## 搜索时序模型（H + nq×t）

对于 HNSW（1M，26 segments）：

```
SearchTask_time = H + nq × t
H = 固定开销（dispatch 所有 segment + reduce）
t = 每个 query 的摊还搜索时间

实测（R4, c=80）：
  H + 80t = 6100ms
  H + t   = 417ms
  → H ≈ 345ms, t ≈ 72ms

High concurrency 优势：H 被 nq=80 摊还 → 每个 query 有效延迟 ≈ t = 72ms
Serial QPS 瓶颈：1 / (H + t) = 1/0.417 ≈ 2.4 QPS（DiskANN R1 的根因）
```

---

## Shim ABI 边界

### CIndexConfig（`cabi_bridge.hpp` ↔ `src/ffi.rs`）

```c
struct CIndexConfig {
    CIndexType  index_type;
    CMetricType metric_type;
    size_t dim;
    size_t ef_construction;   // DiskANN: max_degree; HNSW: ef_construction
    size_t ef_search;         // DiskANN: search_list_size; HNSW: ef_search
    size_t num_partitions;
    size_t num_centroids;
    size_t reorder_k;
    size_t prq_nsplits, prq_msub, prq_nbits;  // HNSW-PRQ
    size_t num_clusters, nprobe;              // IVF
    int32_t data_type;
    // DiskANN-specific (added 2026-04-08):
    float  pq_code_budget_gb;
    float  build_dram_budget_gb;
    size_t disk_pq_dims;
    size_t beamwidth;
};
```

Rust 侧必须与 C++ 侧字段顺序完全一致（`#[repr(C)]`）。

### 动态搜索参数

`search_list_size` 在每次搜索前通过 `knowhere_set_ef_search()` 动态更新（不通过 CIndexConfig）。

---

## HNSW vs DiskANN Shim 差异

| 特性 | HNSW | DiskANN |
|------|------|---------|
| Shim 文件 | `hnsw_rust_node.cpp` | `diskann_rust_node.cpp` |
| Milvus Index class | `VectorMemIndex` | `VectorMemIndex`（非 DiskAnnIndex）|
| `UseDiskLoad()` | false | false（→ 不走 VectorDiskAnnIndex） |
| Deserialize | 直接 load binary | MakeTempDir → load → RemoveAll |
| search 函数 | `knowhere_search_with_bitset` | `knowhere_search_with_bitset` |

---

## 相关页面

- [[concepts/hnsw]] — HNSW 实现细节
- [[concepts/diskann]] — DiskANN / PQFlashIndex
- [[decisions/rayon-private-pool]] — CGO executor 与 rayon 交互
- [[machines/hannsdb-x86]] — Milvus 部署环境

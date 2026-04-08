# DiskANN / PQFlashIndex

**一句话定义**：基于 Vamana 图的近似近邻搜索算法，支持内存模式（NoPQ）和磁盘模式（PQ 压缩），RS 实现在 `src/faiss/diskann_aisaq.rs`（~3300 行）。

---

## 两种实现

| 文件 | 用途 |
|------|------|
| `src/faiss/diskann_aisaq.rs` | **主实现**（PQFlashIndex）— 生产用，含 PQ/NoPQ/disk 路径 |
| `src/faiss/diskann.rs` | 简化版 Vamana graph — 独立 benchmark 用 |

---

## 核心结构：PQFlashIndex

### 关键字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `vectors` | `Vec<f32>` | materialize 后的节点原始向量（NoPQ 热路径） |
| `node_neighbor_ids` | `Vec<u32>` | materialize 后的邻居 ID flat array |
| `storage` | `Option<DiskStorage>` | 磁盘存储；`None` = 已 materialize（快路径） |
| `pq_code_size` | `usize` | 0 = NoPQ；>0 = PQ 字节数 |
| `medoid` | `u32` | 入口节点（图中心） |

### 两条搜索路径

```
storage = None (materialize 后)     → 直接索引 self.vectors[node*dim..]  ← 快
storage = Some(DiskStorage)         → PageCache::read() → Arc alloc + memcpy ← 慢
```

`search_internal` 中的判断：
```rust
let prefetch_enabled = self.storage.is_some();
let use_sector_batch = self.storage.is_some();
```

---

## 生命周期

```
train() → build Vamana graph
add()   → insert_point() 逐节点插入 + materialize_storage() 最后调用
save()  → 序列化到文件
load()  → 反序列化 + ⚠️ 必须 materialize_storage()（见下）
search_batch() / search_batch_with_bitset()
```

### ⚠️ 关键：load() 后必须 materialize

`load()` 默认创建 `storage: Some(DiskStorage)`（磁盘路径），即使 NoPQ in-memory 模式。
修复（commit `72e1cd8`）：`load()` 末尾加：

```rust
if index.pq_code_size == 0 {
    index.materialize_storage()?;
}
```

不加这行 → Milvus DiskANN serial QPS 2.4（与 FLAT 相同），加了 → 11.2（+4.7×）。

→ 详见 [[decisions/materialize-storage]]

---

## 参数说明

### 构建参数（通过 Milvus `create_index` 传入）

| Milvus 参数 | 映射到 CIndexConfig | AisaqConfig 字段 | 含义 |
|-------------|--------------------|--------------------|------|
| `max_degree` | `ef_construction` | `max_degree` | Vamana 图最大出度（native 默认 48） |
| `search_list_size` | `ef_search` | `search_list_size` | build beam size（native 默认 128） |
| `pq_code_budget_gb` | `pq_code_budget_gb` | `pq_code_budget_gb` | 0.0 = NoPQ；>0 = PQ disk 模式 |
| `build_dram_budget_gb` | `build_dram_budget_gb` | `build_dram_budget_gb` | build 内存预算 |
| `beamwidth` | `beamwidth` | `beamwidth` | 搜索 IO 并行宽度（默认 8） |

> 参数对齐修复（commits `a1e49ba` + `e231450` + `d6eb0f2`，2026-04-08）——之前这四个参数无法从 Milvus 侧传入 RS。

### Shim Fallback 值（diskann_rust_node.cpp）

| 参数 | 修复前 fallback | 修复后 fallback | native 默认 |
|------|----------------|----------------|------------|
| `max_degree` | 56 | 48 | 48 |
| `search_list_size` | 100 | 128 | 128 |
| `beamwidth` | 条件赋值（零初始化） | `value_or(8)` | 8 |

---

## Milvus 集成架构

```
Milvus IndexFactory
  └─ INDEX_DISKANN → MakeDiskAnnRustNode()   (index_factory.h)
       └─ DiskAnnRustNode                     (diskann_rust_node.cpp)
            ├─ EnsureIndex() → knowhere_create_index(CIndexConfig)
            ├─ Train()       → knowhere_train_index()
            ├─ Search()      → knowhere_set_ef_search() + knowhere_search_with_bitset()
            └─ Deserialize() → MakeTempDir → knowhere_load_index(tmpdir) → RemoveAll(tmpdir)
```

**注意**：Milvus 通过 `VectorMemIndex`（不是 `VectorDiskAnnIndex`）管理 RS DiskANN，因为 `UseDiskLoad()` 在 RS shim 中永远返回 `false`。

### Segment 模型影响 QPS

1M 向量以 BATCH=10K 插入 → ~25 个 segment，每个 ~40K 向量，各自有一个 PQFlashIndex。

```
serial QPS ≈ 1 / (25 × per_segment_ms + outer_ms)
           ≈ 1 / (25 × 14ms + 10ms) = 2.78 QPS  (修复后)
           ≈ 1 / (25 × 16ms + 10ms) = 2.44 QPS  (修复前)
```

→ 用更少、更大的 segment（bulk-load）可接近 standalone 6,062 QPS。

---

## Standalone vs Milvus 性能对比

| 模式 | 规模 | QPS | 说明 |
|------|------|-----|------|
| Standalone NoPQ 内存 | SIFT-1M | 6,062 | 单图，无 segment 路由 |
| Milvus RS DiskANN serial (R2) | 1M×768D | 11.2 | 25 segments |
| Milvus RS DiskANN c=80 (R2) | 1M×768D | 12.6 | 25 segments 并发 |
| Milvus native DiskANN serial | 1M×768D | 11.4 | native C++ |
| disk/mmap 冷启动 | SIFT-1M | 401 | — |
| disk PQ32 io_uring V3 | SIFT-1M | 1,063 | recall 0.9114 |

→ 详见 [[benchmarks/diskann-milvus-rounds]] | [[benchmarks/authority-numbers]]

---

## 相关页面

- [[concepts/milvus-integration]] — segment dispatch、CGO executor
- [[decisions/materialize-storage]] — load() 路径 bug 根因分析
- [[machines/hannsdb-x86]] — 权威 benchmark 环境

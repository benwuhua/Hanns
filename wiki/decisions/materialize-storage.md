# load() 后必须 materialize_storage()

**结论**：`PQFlashIndex::load()` 默认创建磁盘存储路径（`storage: Some(DiskStorage)`），即使 NoPQ in-memory 模式。NoPQ 模式下必须在 `load()` 末尾调用 `materialize_storage()`，否则所有搜索走 PageCache 路径，QPS 损失 4.7×。

---

## 背景

**症状**：Milvus RS DiskANN serial QPS = 2.4，与 FLAT 暴力搜索相同（~2.4）。预期应与 native DiskANN serial（11.4）持平。

**诊断过程**（2026-04-08 session）：

1. 建立 H + nq×t 模型：H_RS=345ms，H_native≈10ms，t_RS≈72ms，t_native≈77ms
2. 差距完全在 H（fixed per-SearchTask overhead），不在 t（per-query search）
3. 定位到 `load()` 函数：始终创建 `DiskStorage`

---

## 根因分析

### `PQFlashIndex::load()`（`diskann_aisaq.rs:2502`）

```rust
// 修复前 — 始终创建磁盘路径
storage: Some(DiskStorage {
    file_group,
    page_cache,
    vectors_mmap,
    ...
}),
```

### `search_internal` 路径选择

```rust
let prefetch_enabled = self.storage.is_some();  // true → 慢路径
let use_sector_batch = self.storage.is_some();  // true → PageCache 读取
```

**PageCache 路径代价**：每次节点访问 = `Arc::new(mmap[range].to_vec())` = 堆分配 + memcpy。
100 候选 × 每次分配 → 显著开销 vs 直接数组索引。

### `materialize_storage()` 的作用（`diskann_aisaq.rs:5151`）

读取所有节点 → 填入 `self.vectors` / `self.node_neighbor_ids` flat array → 设置 `self.storage = None`。
之后 `search_internal` 走：`self.vectors[node_id * dim..(node_id+1) * dim]`（zero-copy 直接索引）。

`add()` 之后会自动调用 `materialize_storage()`，所以 `train→add→search` 路径从未有问题。
`load()` 之后没有调用，所以 Milvus 场景（序列化/反序列化）始终走慢路径。

---

## 修复

```rust
// src/faiss/diskann_aisaq.rs — load() 末尾，Ok(index) 之前
if index.pq_code_size == 0 {
    // NoPQ in-memory 模式：转换为直接数组访问
    index.materialize_storage()?;
}
Ok(index)
```

**Commit**：`72e1cd8`（2026-04-08）

---

## 效果

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| Milvus serial QPS (1M) | 2.4 | **11.2** | +4.7× |
| Milvus c=1 QPS (1M) | 2.5 | **10.8** | +4.3× |
| 100K per-segment QPS | 38.1 | **41.3** | +8% |
| vs native serial (1M) | −4.75× | **−2%** | parity |

c=20/c=80 QPS 变化不大（批处理模式摊还了 H 开销）。

---

## 参数对齐修复（同 session，2026-04-08）

发现 `load()` 问题后，审计了整个参数传递链，发现 4 个参数缺少通道：

| 参数 | 问题 | 修复 |
|------|------|------|
| `pq_code_budget_gb` | 无法从 Milvus 传入 RS | 加入 `CIndexConfig` |
| `build_dram_budget_gb` | 同上 | 同上 |
| `disk_pq_dims` | 同上 | 同上 |
| `beamwidth` | 同上，且 fallback 为零初始化 | 同上，改为 `value_or(8)` |
| `max_degree` fallback | shim 用 56，native 默认 48 | 改为 48 |
| `search_list_size` fallback | shim 用 100，native 默认 128 | 改为 128 |

**Commits**：`a1e49ba`（Rust）+ `e231450` + `d6eb0f2`（C++ shim）

---

## 教训

1. **load/add 路径不对称**是隐性 bug 温床：add 后自动 materialize，load 后没有，行为静默偏离。
2. **第一性原理诊断法**：症状是"QPS 与 FLAT 相同"→ 不问"怎么调参"→ 问"每次 node access 做了什么" → 发现 Arc alloc + memcpy。
3. **搜索参数审计**：每次 wiring 新 FFI 后，应检查所有参数的完整传递链。

---

## 相关页面

- [[concepts/diskann]] — PQFlashIndex 完整架构
- [[concepts/milvus-integration]] — H + nq×t 模型
- [[benchmarks/diskann-milvus-rounds]] — R1/R2 数字对比

# Plan: HNSW/DiskANN × PCA × SQ/USQ 组合索引

**日期**: 2026-04-08
**目标**: 实现 PCA 降维 + 图索引 + 量化的组合，覆盖四种变体：
- `HNSW-PCA-SQ` — HNSW + PCA 降维 + SQ8 量化
- `HNSW-PCA-USQ` — HNSW + PCA 降维 + ExRaBitQ/USQ 量化
- `DiskANN-PCA-SQ` — **已有**（`diskann_sq.rs` 含 `pca: Option<PcaTransform>`）
- `DiskANN-PCA-USQ` — DiskANN + PCA + USQ（新增，基于 diskann_sq.rs）

---

## 动机 & 收益分析

### 为什么 PCA + 图索引有意义？

**标准 HNSW 痛点**：graph traversal 的 bottleneck 是 distance computation。768D → 计算一次 IP 需要 768 multiply-add。

**PCA-SQ 路径**：
1. PCA 降到 64D（信息保留 ~95%）
2. SQ8 量化（64 bytes → 64 bytes，但每步 L2 只需 64 MAC vs 768 MAC）
3. HNSW 图结构不变，只是存储/计算在低维量化空间

**预期收益**（类比 knowhere C++ HNSW_SQ benchmark）：
- QPS 提升 3-5×（distance 计算量 768→64 = 12× 降低，被量化噪音和 recall 补偿抵消）
- 内存 768×4B → 64×1B = 48× 压缩（每向量）
- recall 需要用高 ef_search 补偿，目标 ≥0.95

### USQ 优势

USQ (ExRaBitQ) 的 1-bit/4-bit 路径配合 AVX512 fastscan，比 SQ8 更快：
- 1-bit: 64D → 8 bytes，SIMD 64-way parallel
- 4-bit: 64D → 32 bytes，SIMD 32-way parallel
- recall 略低于 SQ8，但 QPS 大幅领先

---

## 已有基础

| 组件 | 文件 | 状态 |
|------|------|------|
| `PcaTransform` | `src/quantization/pca.rs` | ✅ train/apply/apply_one |
| `ScalarQuantizer` (SQ8) | `src/quantization/scalar_quantizer.rs` | ✅ |
| `UsqQuantizer` (ExRaBitQ) | `src/quantization/usq.rs` | ✅ 1/4/8-bit + fastscan |
| `HnswSqIndex` | `src/faiss/hnsw_quantized.rs` | ✅ HNSW+SQ8，无 PCA |
| `DiskAnnSqIndex` | `src/faiss/diskann_sq.rs` | ✅ 含 PCA+SQ8 |
| `IvfUsqIndex` | `src/faiss/ivf_usq.rs` | ✅ USQ 实现参考 |

---

## 架构设计

### 统一数据流

```
训练阶段:
  raw vectors [N×D]
    → PcaTransform::train(data, D, D')   # SVD，保留 top D' 主成分
    → projected [N×D']
    → Quantizer::train(projected)         # SQ8 范围 or USQ codebook
    → Graph::build(projected_quantized)   # HNSW or DiskANN

搜索阶段:
  query [D]
    → pca.apply_one(query) → [D']
    → quantizer.encode(query_proj) [可选，某些路径用浮点查询]
    → graph.search(query_proj, ef)
    → rerank with original distance? [可选]
```

### 参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `pca_dim` | 降维目标维度 | 32/64/128 |
| `sq_bits` | USQ 量化位数 | 1/4/8 |
| `ef_construction` | HNSW 建图 ef | 200 |
| `M` | HNSW 连接数 | 16 |
| `ef_search` | 搜索 ef（需调高补偿 recall 损失）| 100-300 |

---

## 实现计划

### Task 1: `HnswPcaSqIndex`（HNSW-PCA-SQ8）

**文件**: 新建 `src/faiss/hnsw_pca_sq.rs` 或扩展 `hnsw_quantized.rs`

**结构**:
```rust
pub struct HnswPcaSqConfig {
    pub dim: usize,
    pub pca_dim: usize,      // 降维目标，0 = 不降维
    pub M: usize,
    pub ef_construction: usize,
    pub metric_type: MetricType,
}

pub struct HnswPcaSqIndex {
    pca: Option<PcaTransform>,     // None 时跳过 PCA
    hnsw_sq: HnswSqIndex,          // 内部用降维后的 dim 构建
    orig_dim: usize,
    proj_dim: usize,
}
```

**build()**:
1. 若 pca_dim < dim：`pca = Some(PcaTransform::train(data, N, dim, pca_dim))`
2. `projected = pca.apply(data, N)`
3. `hnsw_sq.build(projected)` — 内部处理 SQ8 训练 + HNSW 建图

**search()**:
1. `query_proj = pca.apply_one(query)` 或直接传入（若无 PCA）
2. `hnsw_sq.search(query_proj, k, ef)`

**save/load**: PCA 矩阵 + hnsw_sq 各自序列化（参考 `diskann_sq.rs` 的 `write_pca_transform`）

### Task 2: `HnswPcaUsqIndex`（HNSW-PCA-USQ）

**文件**: `src/faiss/hnsw_pca_usq.rs`

设计类似 Task 1，但图节点存储用 USQ 编码（参考 `ivf_usq.rs` 的 `UsqQuantizer`）。

关键差异：
- `UsqQuantizer` 支持 1/4/8-bit，search 路径分两种：
  - **全精度 path**：HNSW 用 float projected 做 graph nav，USQ 只用于存储
  - **fastscan path**：beam 内 batch 距离用 AVX512 fastscan
- 推荐先实现全精度 path，fastscan 作 Phase 2

### Task 3: `DiskAnnPcaUsqIndex`（DiskANN-PCA-USQ）

**文件**: 扩展 `diskann_sq.rs` 或新建 `diskann_pca_usq.rs`

`diskann_sq.rs` 已有 `pca: Option<PcaTransform>` + `Sq8Quantizer`，只需：
1. 把 `Sq8Quantizer` 换成 `UsqQuantizer`
2. search 路径改用 USQ decode/fastscan

### Task 4: Benchmark

测试集：Cohere-1M (768D, IP) 或 SIFT-1M (128D, L2)

关键指标：
- recall@10 vs ef_search（recall-speed trade-off curve）
- QPS @ recall=0.95
- 内存占用（MB per 1M vectors）
- Build time

对比基准：
- HNSW（无 PCA/量化）
- HNSW-SQ（无 PCA）
- native FAISS HNSW-PQ

---

## 优先级

```
Task 1 (HNSW-PCA-SQ) > Task 2 (HNSW-PCA-USQ) > Task 3 (DiskANN-PCA-USQ)
```

Task 3 代码量最小（diskann_sq.rs 改造），但 Task 1 先验证 PCA+图的 recall 可行性。

---

## 风险

1. **PCA recall 损失**：高维语义向量（Cohere 768D）的前 64 个主成分可能只保留 ~70% variance，recall 损失需实测
2. **重排序是否必要**：若 recall 不足 0.95，可加 rerank（用原始 float 向量精排 top-k×rerank_factor）
3. **HnswSqIndex 内部 dim 改变**：需确认 HnswSqIndex 可以用 pca_dim 初始化（而非原始 dim）

# Knowhere-RS 详细开发计划

**版本**: 0.3.6 → 1.0.0
**更新日期**: 2026-02-26
**当前状态**: 183 tests passed, 90% 功能覆盖
**目标**: 95% 功能覆盖, 95% 性能

---

## 里程碑概览

| 里程碑 | 目标 | 状态 | 关键交付物 |
|-------|------|------|-----------|
| **M1** | SCANN 索引 | ✅ 完成 | Google ScaNN |
| **M2** | JNI 绑定 | ⚠️ 骨架 | Java 绑定 |
| **M3** | 优化完善 | 🔄 进行中 | 性能, 文档 |

---

## 当前完成状态

### 已完成 ✅ (P0)

| 功能 | 状态 | 文件 |
|-----|------|------|
| SIMD L2/IP (SSE/AVX2/AVX512/NEON) | ✅ | `src/simd.rs` |
| PQ SIMD 优化 | ✅ | `src/faiss/pq_simd.rs` |
| RaBitQ 量化 (32x) | ✅ | `src/quantization/rabitq.rs` |
| GetVectorByIds | ✅ | `src/index.rs`, `src/faiss/mem_index.rs` |
| CalcDistByIDs | ✅ | `src/faiss/mem_index.rs` |
| BinarySet 序列化 | ✅ | `src/faiss/mem_index.rs` |
| DiskANN 序列化 | ✅ | `src/faiss/diskann.rs` |
| K-means SIMD | ✅ | `src/quantization/kmeans.rs` |
| **AnnIterator** | ✅ | `src/api/search.rs` |
| **FP16/BF16** | ✅ | `src/half.rs` |
| **FP16 SIMD** | ✅ | `src/half.rs` (AVX2 IP) |
| **Federation Info** | ✅ | `src/federation.rs` |
| **HNSW 参数** | ✅ | `src/faiss/hnsw.rs` |
| **SCANN 索引** | ✅ | `src/faiss/scann.rs` |
| **JNI 骨架** | ✅ | `src/jni/mod.rs` |
| **L1/Linf SIMD** | ✅ | `src/simd.rs` |
| **IVF-SQ8 并行** | ✅ | `src/faiss/ivf_sq8.rs` |
| **Serializable Trait** | ✅ | `src/faiss/hnsw.rs`, `mem_index.rs` |

### 索引实现状态

| 索引 | 状态 | 质量 | 最新更新 |
|-----|------|------|---------|
| Flat | ✅ | ⭐⭐⭐⭐⭐ | Serializable |
| HNSW | ✅ | ⭐⭐⭐⭐⭐ | M/ef_search/ef_construction, Filter |
| HNSW-SQ/PQ | ✅ | ⭐⭐⭐⭐ | |
| IVF-Flat | ✅ | ⭐⭐⭐⭐ | |
| IVF-PQ | ✅ | ⭐⭐⭐⭐⭐ | PQ SIMD |
| IVF-SQ8 | ✅ | ⭐⭐⭐⭐ | 并行搜索/添加 |
| DiskANN | ✅ | ⭐⭐⭐ | |
| ANNOY | ✅ | ⭐⭐⭐⭐ | |
| Binary | ✅ | ⭐⭐⭐⭐ | |
| Sparse | ✅ | ⭐⭐⭐ | |
| RaBitQ | ✅ | ⭐⭐⭐⭐ | 32x 压缩 |
| **SCANN** | ✅ | ⭐⭐⭐⭐ | Anisotropic Quantization |

---

## M1: SCANN 索引 ✅ 完成

### 已实现功能

| 功能 | 状态 | 说明 |
|-----|------|------|
| AnisotropicQuantizer | ✅ | 各向异性量化器 |
| ScaNNConfig | ✅ | 配置参数 |
| K-means++ 初始化 | ✅ | 质心初始化 |
| 加权 K-means | ✅ | 各向异性权重 |
| encode/decode | ✅ | 向量编解码 |
| ADC 距离计算 | ✅ | 非对称距离 |
| 粗排 + 精排 | ✅ | 两阶段搜索 |
| save/load | ✅ | 序列化 |

### 测试覆盖

- `test_scann_basic` - 基础功能测试
- `test_scann_with_query_sample` - 带查询样本训练
- `test_scann_save_load` - 序列化测试
- `test_scann_with_ids` - ID 管理
- `test_scann_empty_search` - 空搜索
- `test_anisotropic_quantizer` - 量化器测试

**文件**: `src/faiss/scann.rs` (907 行)

---

## M2: JNI 绑定 ⚠️ 进行中

### 已实现功能

| 功能 | 状态 | 说明 |
|-----|------|------|
| 索引注册表 | ✅ | 全局 HashMap |
| createIndex | ✅ | Flat, HNSW, IVF-PQ, DiskANN |
| freeIndex | ✅ | 释放索引 |
| addIndex | ✅ | 添加向量 |
| search | ✅ | 搜索 |
| getResultIds | ✅ | 获取结果 ID |
| getResultDistances | ✅ | 获取结果距离 |
| freeResult | ✅ | 释放结果 |
| serializeIndex | ✅ | 序列化到字节数组 |
| deserializeIndex | ✅ | 从字节数组反序列化 |

### 待完成

1. ~~**序列化 API**: 实现 serializeIndex/deserializeIndex~~ ✅
2. **Java 类**: 创建 KnowhereIndex.java 包装类
3. **单元测试**: JNI 单元测试
4. **性能测试**: JNI 调用开销测试

**文件**: `src/jni/mod.rs` (366 行)

---

## M3: 优化完善 🔄 进行中

### 3.1 性能优化

| 优化项 | 状态 | 说明 |
|-------|------|------|
| SIMD L2/IP | ✅ | SSE/AVX2/AVX512/NEON |
| FP16 SIMD | ✅ | AVX2 内积 |
| L1/Linf SIMD | ✅ | 新增 |
| PQ ADC SIMD | ✅ | 4x 展开 |

### 3.2 待完成优化

| 优化项 | 优先级 | 工作量 |
|-------|-------|--------|
| SCANN SIMD | P1 | 2 天 |
| ~~JNI 序列化~~ | P1 | ~~1 天~~ ✅ |
| 内存池优化 | P2 | 2 天 |

### 3.3 文档

- [x] GAP_ANALYSIS.md 更新
- [x] DEV_ROADMAP.md 更新
- [ ] API 文档完善
- [ ] 性能基准测试

---

## 交付物汇总

| 里程碑 | 交付物 | 状态 | 文件 |
|-------|-------|------|------|
| M1 | SCANN 索引 | ✅ 完成 | `src/faiss/scann.rs` |
| M2 | JNI 绑定 | ⚠️ 骨架 | `src/jni/mod.rs` |
| M3 | 性能基准 | 🔄 进行中 | `benches/` |

---

## 剩余工作

### P1 - 重要功能

| 功能 | 工作量 | 说明 |
|-----|-------|------|
| JNI 序列化 | 2 天 | serializeIndex/deserializeIndex |
| Python 绑定 | 3 天 | PyO3 |

### P2 - 增强功能

| 功能 | 工作量 | 说明 |
|-----|-------|------|
| PRQ 量化 | 5 天 | 渐进残差量化 |
| 动态删除完善 | 3 天 | 部分索引 |
| 异步构建 | 3 天 | async/await |

### P3 - 长期目标

| 功能 | 工作量 | 说明 |
|-----|-------|------|
| GPU 支持 (wgpu) | 长期 | 需要 GPU 基础 |
| 混合搜索 | 5 天 | 多模态 |
| MinHash-LSH | 3 天 | LSH 近似 |

---

## 成功标准

### 当前状态 vs 目标

| 指标 | 当前 | 目标 | 差距 |
|-----|------|------|------|
| 索引类型 | 14 | 15 | -1 |
| 功能覆盖 | 90% | 95% | -5% |
| 测试覆盖 | 183 | 250+ | -67 |
| Recall@10 | 95%+ | 95%+ | ✅ |
| QPS (vs C++) | 90% | 95% | -5% |
| API 完整度 | 95% | 98% | -3% |

### 预计完成时间

- **95% 覆盖**: 2-3 周
- **1.0.0 发布**: 4-5 周

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|-----|------|------|---------|
| JNI 序列化复杂 | 中 | 中 | 参考 C++ 实现 |
| 性能目标未达成 | 高 | 低 | 增加优化迭代 |
| Python 绑定延期 | 低 | 中 | 可选功能 |

---

## 技术细节

### SCANN 实现

```rust
// 各向异性量化核心
pub struct AnisotropicQuantizer {
    codebook: Vec<f32>,      // [num_partitions * num_centroids * sub_dim]
    weights: Vec<f32>,        // 各向异性权重
    centroid_norms: Vec<f32>, // 质心范数
}

// 加权 L2 距离
fn weighted_l2_squared(&self, a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .zip(weights.iter())
        .map(|((&x, &y), &w)| w * (x - y) * (x - y))
        .sum()
}
```

### JNI 绑定架构

```rust
// 全局索引注册表
static INDEX_REGISTRY: Mutex<Option<HashMap<jlong, Box<dyn Index + Send + Sync>>>>;

// 创建索引
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_createIndex(
    index_type: jint,
    dim: jint,
    metric_type: jint,
    ...
) -> jlong;
```

---

## 更新日志

### 2026-02-26
- ✅ SCANN 索引实现 (907 行)
- ✅ JNI 绑定骨架 (366 行)
- ✅ FP16 内积 AVX2 SIMD
- ✅ L1/Linf SIMD 优化
- ✅ IVF-SQ8 并行搜索/添加
- ✅ Serializable trait (HNSW, MemIndex)
- ✅ HNSW filter 支持
- ✅ FFI C API 完善
- 📝 文档更新 (GAP_ANALYSIS.md, DEV_ROADMAP.md)
- 🧪 测试: 160 → 183

### 2026-02-25
- ✅ AnnIterator 迭代器
- ✅ FP16/BF16 支持
- ✅ Federation Info
- ✅ HNSW 参数增强
- ✅ RaBitQ 量化

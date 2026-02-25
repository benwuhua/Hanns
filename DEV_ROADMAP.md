# Knowhere-RS 详细开发计划

**版本**: 0.3.5 → 1.0.0
**更新日期**: 2026-02-25
**当前状态**: 160 tests passed, 85% 功能覆盖
**目标**: 95% 功能覆盖, 95% 性能

---

## 里程碑概览

| 里程碑 | 目标 | 时间 | 关键交付物 |
|-------|------|------|-----------|
| **M1** | SCANN 索引 | 2 周 | Google ScaNN |
| **M2** | JNI 绑定 | 1.5 周 | Java 绑定 |
| **M3** | 优化完善 | 1 周 | 性能, 文档 |

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
| **Federation Info** | ✅ | `src/federation.rs` |
| **HNSW 参数** | ✅ | `src/faiss/hnsw.rs` |

### 索引实现状态

| 索引 | 状态 | 质量 | 最新更新 |
|-----|------|------|---------|
| Flat | ✅ | ⭐⭐⭐⭐⭐ | |
| HNSW | ✅ | ⭐⭐⭐⭐⭐ | M/ef_search/ef_construction |
| HNSW-SQ/PQ | ✅ | ⭐⭐⭐⭐ | |
| IVF-Flat | ✅ | ⭐⭐⭐⭐ | |
| IVF-PQ | ✅ | ⭐⭐⭐⭐⭐ | PQ SIMD |
| IVF-SQ8 | ✅ | ⭐⭐⭐⭐ | |
| DiskANN | ✅ | ⭐⭐⭐ | |
| ANNOY | ✅ | ⭐⭐⭐⭐ | |
| Binary | ✅ | ⭐⭐⭐⭐ | |
| Sparse | ✅ | ⭐⭐⭐ | |
| RaBitQ | ✅ | ⭐⭐⭐⭐ | |

---

## M1: SCANN 索引 (Week 1-2)

### 目标
- 各向异性向量量化 (AVQ)
- 高召回 + 高吞吐
- 对齐 C++ 性能

### 1.1 核心结构

**文件**: `src/faiss/scann.rs`

```rust
//! SCANN: Scalable Nearest Neighbors
//!
//! Google Research 的各向异性向量量化
//! 论文: https://arxiv.org/abs/1908.10396

use std::collections::HashMap;
use crate::api::{IndexConfig, MetricType, Result, SearchRequest, SearchResult};

/// SCANN 配置
#[derive(Clone, Debug)]
pub struct ScaNNConfig {
    /// 子空间数量 (默认 16)
    pub num_partitions: usize,
    /// 每个子空间的质心数 (默认 256)
    pub num_centroids: usize,
    /// 重排序候选数 (默认 100)
    pub reorder_k: usize,
    /// 各向异性参数 (默认 0.2)
    pub anisotropic_alpha: f32,
    /// 是否启用 SIMD
    pub use_simd: bool,
}

impl Default for ScaNNConfig {
    fn default() -> Self {
        Self {
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            anisotropic_alpha: 0.2,
            use_simd: true,
        }
    }
}

/// 各向异性量化器
pub struct AnisotropicQuantizer {
    dim: usize,
    sub_dim: usize,
    config: ScaNNConfig,

    /// 码书: [num_partitions * num_centroids * sub_dim]
    codebooks: Vec<f32>,

    /// 各向异性权重
    weights: Vec<f32>,

    /// 质心范数 (用于快速距离计算)
    centroid_norms: Vec<f32>,

    /// 距离表 (SIMD 优化)
    distance_tables: Vec<Vec<f32>>,
}

impl AnisotropicQuantizer {
    pub fn new(dim: usize, config: ScaNNConfig) -> Self {
        let sub_dim = dim / config.num_partitions;

        Self {
            dim,
            sub_dim,
            config,
            codebooks: Vec::new(),
            weights: Vec::new(),
            centroid_norms: Vec::new(),
            distance_tables: Vec::new(),
        }
    }

    /// 各向异性训练
    ///
    /// 目标: 最小化 Σ w(θ) * ||x - c(x)||²
    /// 其中 w(θ) = 1 / (1 + α * |cos(θ)|)
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) {
        // 1. 计算各向异性权重
        self.compute_anisotropic_weights(query_sample);

        // 2. 对每个子空间训练加权 K-means
        for p in 0..self.config.num_partitions {
            let sub_vectors = self.extract_subspace(data, p);
            self.train_subspace_weighted(p, &sub_vectors);
        }
    }

    /// 编码向量
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        (0..self.config.num_partitions)
            .map(|p| {
                let sub_vec = self.get_subvector(vector, p);
                self.find_nearest_centroid(p, sub_vec) as u8
            })
            .collect()
    }

    /// 预计算距离表 (查询优化)
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = vec![0.0f32; self.config.num_partitions * self.config.num_centroids];

        for p in 0..self.config.num_partitions {
            let sub_query = self.get_subvector(query, p);

            for c in 0..self.config.num_centroids {
                let centroid = self.get_centroid(p, c);
                table[p * self.config.num_centroids + c] = self.l2_distance(sub_query, centroid);
            }
        }

        table
    }

    /// ADC 距离 (非对称距离计算)
    #[inline]
    pub fn adc_distance(&self, table: &[f32], codes: &[u8]) -> f32 {
        codes.iter()
            .enumerate()
            .map(|(p, &code)| table[p * self.config.num_centroids + code as usize])
            .sum()
    }

    /// SIMD 优化的 ADC 距离
    #[cfg(feature = "simd")]
    pub fn adc_distance_simd(&self, table: &[f32], codes: &[u8]) -> f32 {
        use crate::simd;

        // 4x 展开
        let mut sum = 0.0f32;
        let chunks = codes.len() / 4;

        for i in 0..chunks {
            let base = i * 4;
            for j in 0..4 {
                let code = codes[base + j] as usize;
                sum += table[(base + j) * self.config.num_centroids + code];
            }
        }

        // 处理剩余
        for i in (chunks * 4)..codes.len() {
            sum += table[i * self.config.num_centroids + codes[i] as usize];
        }

        sum
    }
}

/// SCANN 索引
pub struct ScaNNIndex {
    dim: usize,
    config: ScaNNConfig,

    /// 量化器
    quantizer: AnisotropicQuantizer,

    /// 倒排索引: partition -> [(id, codes)]
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,

    /// 原始向量 (用于重排序)
    vectors: Vec<f32>,
    ids: Vec<i64>,
    id_to_pos: HashMap<i64, usize>,

    trained: bool,
}

impl ScaNNIndex {
    pub fn new(dim: usize, config: ScaNNConfig) -> Self {
        Self {
            dim,
            quantizer: AnisotropicQuantizer::new(dim, config.clone()),
            config,
            inverted_lists: HashMap::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            id_to_pos: HashMap::new(),
            trained: false,
        }
    }

    /// 训练
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) -> Result<()> {
        self.quantizer.train(data, query_sample);
        self.trained = true;
        Ok(())
    }

    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".into(),
            ));
        }

        let n = vectors.len() / self.dim;

        for i in 0..n {
            let vector = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.quantizer.encode(vector);

            // 分配到分区
            let partition = self.assign_partition(&codes);

            let id = ids.map(|ids| ids[i]).unwrap_or(self.ids.len() as i64);
            let pos = self.ids.len();

            self.inverted_lists
                .entry(partition)
                .or_default()
                .push((id, codes));

            self.vectors.extend_from_slice(vector);
            self.ids.push(id);
            self.id_to_pos.insert(id, pos);
        }

        Ok(n)
    }

    /// 搜索
    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        // 1. 预计算距离表
        let table = self.quantizer.compute_distance_table(query);

        // 2. 粗排: ADC 距离
        let candidates = self.coarse_search(&table, self.config.reorder_k);

        // 3. 精排: 原始向量重排序
        let results = self.rerank(query, candidates, k);

        Ok(SearchResult::new(
            results.iter().map(|(id, _)| *id).collect(),
            results.iter().map(|(_, dist)| *dist).collect(),
            0.0,
        ))
    }

    /// 粗排
    fn coarse_search(&self, table: &[f32], k: usize) -> Vec<(i64, f32)> {
        let mut candidates = Vec::new();

        for list in self.inverted_lists.values() {
            for &(id, ref codes) in list {
                let dist = self.quantizer.adc_distance(table, codes);
                candidates.push((id, dist));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        candidates
    }

    /// 精排
    fn rerank(&self, query: &[f32], candidates: Vec<(i64, f32)>, k: usize) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = candidates
            .iter()
            .filter_map(|&(id, _)| {
                let pos = *self.id_to_pos.get(&id)?;
                let vector = &self.vectors[pos * self.dim..(pos + 1) * self.dim];
                Some((id, self.l2_distance(query, vector)))
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        results
    }

    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scann_basic() {
        let dim = 128;
        let config = ScaNNConfig::default();
        let mut index = ScaNNIndex::new(dim, config);

        // 生成数据
        let n = 1000;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();

        // 训练 + 添加
        index.train(&data, None).unwrap();
        index.add(&data, None).unwrap();

        // 搜索
        let query = &data[0..dim];
        let results = index.search(query, 10).unwrap();

        assert_eq!(results.ids.len(), 10);
        assert_eq!(results.ids[0], 0); // 第一个应该是自己
    }
}
```

**工作量**: 7 天
**验收标准**:
- [ ] 各向异性量化
- [ ] 加权 K-means
- [ ] ADC 距离
- [ ] SIMD 优化
- [ ] Recall@10 ≥ 95%
- [ ] QPS ≥ IVF-PQ 1.2x

---

## M2: JNI 绑定 (Week 3-4)

### 目标
- 完整 Java 绑定
- 对齐 C++ Knowhere JNI

### 2.1 JNI 模块

**文件**: `src/jni/mod.rs`, `src/jni/index.rs`

```rust
//! JNI 绑定
//!
//! Java Native Interface for Knowhere-RS

use jni::JNIEnv;
use jni::objects::{JClass, JObject, JLongArray, JFloatArray, JIntArray, JString, JByteArray};
use jni::sys::{jint, jlong, jfloat, jboolean, jbyteArray};

mod index;
mod search;
mod bitset;

/// 初始化 JNI
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _reserved: *mut std::ffi::c_void) -> jint {
    // 全局初始化
    jni::JNIVersion::V1_8.into()
}

/// 释放 JNI
#[no_mangle]
pub extern "system" fn JNI_OnUnload(_vm: jni::JavaVM, _reserved: *mut std::ffi::c_void) {
    // 清理
}

// ===== Index 操作 =====

/// 创建索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    index_type: jint,
    metric_type: jint,
    dim: jint,
) -> jlong {
    match create_index_from_type(index_type, metric_type, dim as usize) {
        Ok(index) => Box::into_raw(Box::new(index)) as jlong,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", &e.to_string());
            0
        }
    }
}

/// 释放索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut Box<dyn Index>)); }
    }
}

/// 训练
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeTrain(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    data: JFloatArray,
) {
    let index = unsafe { &mut *(handle as *mut Box<dyn Index>) };

    let data_vec: Vec<f32> = env.get_float_array_elements(&data, Default::default())
        .unwrap()
        .iter()
        .map(|&f| f)
        .collect();

    let dataset = Dataset::from_vec(data_vec, index.dim());

    if let Err(e) = index.train(&dataset) {
        let _ = env.throw_new("java/lang/RuntimeException", &e.to_string());
    }
}

/// 添加向量
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeAdd(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    data: JFloatArray,
    ids: JLongArray,
) {
    let index = unsafe { &mut *(handle as *mut Box<dyn Index>) };

    let data_vec: Vec<f32> = env.get_float_array_elements(&data, Default::default())
        .unwrap()
        .iter()
        .map(|&f| f)
        .collect();

    let ids_vec: Vec<i64> = env.get_long_array_elements(&ids, Default::default())
        .unwrap()
        .iter()
        .map(|&l| l)
        .collect();

    let dataset = Dataset::from_vec_with_ids(data_vec, ids_vec, index.dim());

    if let Err(e) = index.add(&dataset) {
        let _ = env.throw_new("java/lang/RuntimeException", &e.to_string());
    }
}

/// 搜索
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeSearch(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    queries: JFloatArray,
    k: jint,
) -> JObject {
    let index = unsafe { &*(handle as *mut Box<dyn Index>) };

    let query_vec: Vec<f32> = env.get_float_array_elements(&queries, Default::default())
        .unwrap()
        .iter()
        .map(|&f| f)
        .collect();

    let dataset = Dataset::from_vec(query_vec, index.dim());

    match index.search(&dataset, k as usize) {
        Ok(result) => create_java_search_result(&mut env, result),
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", &e.to_string());
            JObject::null()
        }
    }
}

/// GetVectorByIds
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeGetVectorByIds(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    ids: JLongArray,
) -> JFloatArray {
    let index = unsafe { &*(handle as *mut Box<dyn Index>) };

    let ids_vec: Vec<i64> = env.get_long_array_elements(&ids, Default::default())
        .unwrap()
        .iter()
        .map(|&l| l)
        .collect();

    match index.get_vector_by_ids(&ids_vec) {
        Ok(vectors) => {
            let arr = env.new_float_array(vectors.len() as jint).unwrap();
            env.set_float_array_region(&arr, 0, &vectors).unwrap();
            arr
        }
        Err(_) => JFloatArray::default(),
    }
}

// ===== 辅助函数 =====

fn create_java_search_result(env: &mut JNIEnv, result: SearchResult) -> JObject {
    // 创建 SearchResult Java 对象
    let class = env.find_class("io/milvus/knowhere/SearchResult").unwrap();

    let ids_arr = env.new_long_array(result.ids.len() as jint).unwrap();
    env.set_long_array_region(&ids_arr, 0, &result.ids).unwrap();

    let dists_arr = env.new_float_array(result.distances.len() as jint).unwrap();
    env.set_float_array_region(&dists_arr, 0, &result.distances).unwrap();

    env.new_object(
        class,
        "([J[FD)V",
        &[ids_arr.into(), dists_arr.into(), result.elapsed_ms.into()],
    ).unwrap()
}
```

### 2.2 Java 类

**文件**: `src/java/io/milvus/knowhere/KnowhereIndex.java`

```java
package io.milvus.knowhere;

import java.io.Closeable;

public class KnowhereIndex implements Closeable {
    static {
        System.loadLibrary("knowhere_rs");
    }

    private long handle;
    private boolean closed = false;

    private KnowhereIndex(long handle) {
        this.handle = handle;
    }

    // ===== 工厂方法 =====

    public static KnowhereIndex createFlat(int dim, MetricType metric) {
        return create(IndexType.FLAT, metric, dim);
    }

    public static KnowhereIndex createHnsw(int dim, MetricType metric, int m, int efConstruction) {
        return create(IndexType.HNSW, metric, dim);
    }

    public static KnowhereIndex createIvfPq(int dim, MetricType metric, int nlist, int m) {
        return create(IndexType.IVF_PQ, metric, dim);
    }

    public static KnowhereIndex create(IndexType type, MetricType metric, int dim) {
        long handle = nativeCreate(type.getValue(), metric.getValue(), dim);
        return new KnowhereIndex(handle);
    }

    // ===== 核心操作 =====

    public void train(float[] data) {
        checkNotClosed();
        nativeTrain(handle, data);
    }

    public void add(float[] data, long[] ids) {
        checkNotClosed();
        nativeAdd(handle, data, ids);
    }

    public SearchResult search(float[] queries, int k) {
        checkNotClosed();
        return nativeSearch(handle, queries, k);
    }

    public float[] getVectorByIds(long[] ids) {
        checkNotClosed();
        return nativeGetVectorByIds(handle, ids);
    }

    public void save(String path) {
        checkNotClosed();
        nativeSave(handle, path);
    }

    public static KnowhereIndex load(String path) {
        long handle = nativeLoad(path);
        return new KnowhereIndex(handle);
    }

    // ===== 资源管理 =====

    @Override
    public void close() {
        if (!closed) {
            nativeFree(handle);
            handle = 0;
            closed = true;
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Index has been closed");
        }
    }

    // ===== Native 方法 =====

    private static native long nativeCreate(int type, int metric, int dim);
    private native void nativeTrain(long handle, float[] data);
    private native void nativeAdd(long handle, float[] data, long[] ids);
    private native SearchResult nativeSearch(long handle, float[] queries, int k);
    private native float[] nativeGetVectorByIds(long handle, long[] ids);
    private native void nativeSave(long handle, String path);
    private static native long nativeLoad(String path);
    private native void nativeFree(long handle);
}
```

**工作量**: 5 天
**验收标准**:
- [ ] JNI 模块编译
- [ ] Java 类可用
- [ ] 单元测试通过
- [ ] 示例程序运行

---

## M3: 优化完善 (Week 5)

### 目标
- 性能基准测试
- 文档完善
- CI/CD 优化

### 3.1 性能基准

**文件**: `benches/comparison_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_all_indices(c: &mut Criterion) {
    let dim = 128;
    let n = 1_000_000;
    let data = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Flat
    let mut group = c.benchmark_group("search_1m");
    group.bench_function("flat", |b| {
        let index = FlatIndex::new(dim);
        // ...
    });

    // HNSW
    group.bench_function("hnsw", |b| {
        // ...
    });

    // IVF-PQ
    group.bench_function("ivf_pq", |b| {
        // ...
    });

    // SCANN
    group.bench_function("scann", |b| {
        // ...
    });

    group.finish();
}

criterion_group!(benches, bench_all_indices);
criterion_main!(benches);
```

**工作量**: 2 天

### 3.2 文档

- [ ] API 文档
- [ ] README 更新
- [ ] 迁移指南
- [ ] 性能对比

**工作量**: 2 天

---

## 交付物汇总

| 里程碑 | 交付物 | 文件 |
|-------|-------|------|
| M1 | SCANN 索引 | `src/faiss/scann.rs` |
| M2 | JNI 绑定 | `src/jni/`, `src/java/` |
| M3 | 性能基准 | `benches/` |

---

## 成功标准

### 5 周目标

| 指标 | 当前 | 目标 |
|-----|------|------|
| 索引类型 | 13 | 14 |
| 功能覆盖 | 85% | 95% |
| 测试覆盖 | 160 | 250+ |
| Recall@10 | 95%+ | 95%+ |
| QPS (vs C++) | 90% | 95% |
| API 完整度 | 90% | 98% |

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|-----|------|------|---------|
| SCANN 算法复杂 | 高 | 中 | 参考官方实现 |
| JNI 内存管理 | 中 | 中 | 充分测试 |
| 性能目标未达成 | 高 | 低 | 增加优化迭代 |
| 时间延期 | 中 | 低 | 预留缓冲 |

---

## 长期目标 (P3)

| 功能 | 时间 | 说明 |
|-----|------|------|
| GPU 支持 (wgpu) | 长期 | 需要 GPU 基础 |
| Python 绑定 | 1 周 | PyO3 |
| 混合搜索 | 1 周 | 多模态 |
| PRQ 量化 | 1 周 | 残差量化 |

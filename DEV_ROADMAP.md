# Knowhere-RS 详细开发计划

**版本**: 0.2.0 → 1.0.0
**更新日期**: 2026-02-25
**当前状态**: 133 tests passed, 70% 功能覆盖
**目标**: 85% 功能覆盖, 80% 性能

---

## 里程碑概览

| 里程碑 | 目标 | 时间 | 关键交付物 |
|-------|------|------|-----------|
| **M1** | API 补全 | 1 周 | GetVectorByIds, BinarySet |
| **M2** | RaBitQ 量化 | 2 周 | 32x 压缩量化 |
| **M3** | SCANN 索引 | 2 周 | Google ScaNN |
| **M4** | FFI/JNI | 2 周 | C API, Java 绑定 |
| **M5** | 优化完善 | 1 周 | 性能调优, 文档 |

---

## M1: API 补全 (Week 1)

### 目标
- GetVectorByIds 功能
- BinarySet 内存序列化
- 迭代器接口设计

### 1.1 GetVectorByIds

**文件**: `src/index.rs`, 各索引实现

```rust
// src/index.rs - Index trait 扩展
pub trait Index: Send + Sync {
    // ... 现有方法 ...

    /// 按ID批量获取向量
    fn get_vectors_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>>;
}

// src/faiss/mem_index.rs - 实现
impl Index for MemIndex {
    fn get_vectors_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(ids.len() * self.dim);

        for &id in ids {
            if let Some(&pos) = self.id_to_pos.get(&id) {
                let start = pos * self.dim;
                result.extend_from_slice(&self.vectors[start..start + self.dim]);
            } else {
                // 填充零向量或返回错误
                result.extend(std::iter::repeat(0.0f32).take(self.dim));
            }
        }

        Ok(result)
    }
}
```

**工作量**: 2 天
**验收标准**:
- [ ] Index trait 添加方法
- [ ] Flat, HNSW, IVF 索引实现
- [ ] 单元测试覆盖

### 1.2 BinarySet 内存序列化

**文件**: `src/codec/binary_set.rs`

```rust
/// 二进制集合 (内存序列化)
pub struct BinarySet {
    data: HashMap<String, Binary>,
}

pub struct Binary {
    data: Vec<u8>,
    size: usize,
}

impl BinarySet {
    pub fn new() -> Self {
        Self { data: HashMap::new() }
    }

    /// 添加二进制数据
    pub fn append(&mut self, name: &str, data: &[u8]) {
        self.data.insert(name.to_string(), Binary {
            data: data.to_vec(),
            size: data.len(),
        });
    }

    /// 获取二进制数据
    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.data.get(name).map(|b| b.data.as_slice())
    }

    /// 序列化到字节
    pub fn to_bytes(&self) -> Vec<u8> {
        // 格式: [count:u32, [(name_len:u32, name:bytes, data_len:u64, data:bytes), ...]]
    }

    /// 从字节反序列化
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // 解析格式
    }
}

// Index trait 扩展
pub trait Index {
    /// 序列化到 BinarySet
    fn serialize(&self) -> Result<BinarySet>;

    /// 从 BinarySet 反序列化
    fn deserialize(&mut self, binary_set: &BinarySet) -> Result<()>;
}
```

**工作量**: 3 天
**验收标准**:
- [ ] BinarySet 结构实现
- [ ] Flat, IVF, HNSW 序列化
- [ ] 零拷贝支持 (可选)

### 1.3 迭代器接口设计

**文件**: `src/iterator.rs`

```rust
/// ANN 迭代器 trait
pub trait AnnIterator: Iterator<Item = (i64, f32)> {
    /// 获取当前最佳距离
    fn distance(&self) -> f32;

    /// 是否已耗尽
    fn is_exhausted(&self) -> bool;
}

// HNSW 迭代器实现
pub struct HnswIterator<'a> {
    index: &'a HnswIndex,
    query: Vec<f32>,
    visited: HashSet<i64>,
    candidates: BinaryHeap<OrderedCandidate>,
    current_best: f32,
}

impl<'a> Iterator for HnswIterator<'a> {
    type Item = (i64, f32);

    fn next(&mut self) -> Option<Self::Item> {
        // 增量式返回最近邻
    }
}
```

**工作量**: 2 天

---

## M2: RaBitQ 量化 (Week 2-3)

### 目标
- 实现 RaBitQ 1-bit 量化
- 32x 压缩比
- 与 IVF 结合

### 2.1 RaBitQ 核心实现

**文件**: `src/quantization/rabitq.rs`

```rust
//! RaBitQ: 1-bit Random Quantization
//!
//! 参考: RaBitQ: Quantizing High-Dimensional Vectors with a 1-Bit Code
//! 论文: https://arxiv.org/abs/2405.12497

use rand::Rng;

/// RaBitQ 量化器
pub struct RaBitQ {
    dim: usize,

    // 随机旋转矩阵 (dim x dim)
    rotation: Vec<f32>,

    // 二值码书
    binary_centroids: Vec<u64>,

    // 残差信息 (用于距离校正)
    norms: Vec<f32>,
}

impl RaBitQ {
    pub fn new(dim: usize) -> Self {
        // 生成随机旋转矩阵
        let mut rng = rand::thread_rng();
        let mut rotation = vec![0.0f32; dim * dim];

        // 使用随机正交矩阵
        for i in 0..dim {
            for j in 0..dim {
                rotation[i * dim + j] = rng.gen::<f32>() * 2.0 - 1.0;
            }
        }

        // Gram-Schmidt 正交化
        orthogonalize(&mut rotation, dim);

        Self {
            dim,
            rotation,
            binary_centroids: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// 训练: 计算二值码书
    pub fn train(&mut self, data: &[f32], n_clusters: usize) {
        let n = data.len() / self.dim;

        // 1. 应用随机旋转
        let rotated = self.apply_rotation(data);

        // 2. 对每个聚类，计算二值质心
        for c in 0..n_clusters {
            let centroid = compute_cluster_centroid(&rotated, c, self.dim);
            let binary = self.binarize(&centroid);

            self.binary_centroids.push(binary);
            self.norms.push(centroid.iter().map(|x| x * x).sum::<f32>().sqrt());
        }
    }

    /// 编码: 向量 -> 64-bit 二值码 (32x 压缩)
    ///
    /// 原始: dim * 4 bytes
    /// 编码后: ceil(dim / 64) * 8 bytes
    /// 压缩比: dim / ceil(dim/64) ≈ 32x (for dim=128-960)
    pub fn encode(&self, vector: &[f32]) -> Vec<u64> {
        let rotated = self.rotate(vector);
        let n_words = (self.dim + 63) / 64;

        let mut codes = vec![0u64; n_words];

        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                codes[i / 64] |= 1u64 << (i % 64);
            }
        }

        codes
    }

    /// 距离计算: 使用 popcount
    ///
    /// Hamming(x, y) ≈ angle(x, y) * dim
    /// 通过查表校正
    pub fn distance(&self, query_codes: &[u64], db_codes: &[u64]) -> f32 {
        let mut hamming = 0usize;

        for (&q, &d) in query_codes.iter().zip(db_codes.iter()) {
            hamming += (q ^ d).count_ones() as usize;
        }

        // 转换为近似距离
        let angle = hamming as f32 / self.dim as f32 * std::f32::consts::PI;
        angle.cos().abs() // 近似余弦距离
    }

    /// 非对称距离 (查询向量未量化)
    pub fn asymmetric_distance(&self, query: &[f32], db_codes: &[u64], db_norm: f32) -> f32 {
        let rotated_q = self.rotate(query);

        // 计算查询的二值码
        let query_codes = self.encode(&rotated_q);

        // 计算近似距离
        let hamming = self.hamming(&query_codes, db_codes);

        // 校正距离
        let query_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let correction = query_norm * db_norm;

        correction * (hamming as f32 / self.dim as f32)
    }

    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.dim];

        for i in 0..self.dim {
            for j in 0..self.dim {
                result[i] += vector[j] * self.rotation[j * self.dim + i];
            }
        }

        result
    }

    fn hamming(&self, a: &[u64], b: &[u64]) -> usize {
        a.iter().zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as usize)
            .sum()
    }
}
```

**工作量**: 5 天
**验收标准**:
- [ ] 32x 压缩比验证
- [ ] Recall@10 ≥ 90%
- [ ] AVX-512 VPOPCNTDQ 优化 (可选)

### 2.2 IVF-RaBitQ 索引

**文件**: `src/faiss/ivf_rabitq.rs`

```rust
/// IVF + RaBitQ 索引
pub struct IvfRabitQIndex {
    // IVF 结构
    nlist: usize,
    centroids: Vec<f32>,

    // RaBitQ 量化器
    rabitq: RaBitQ,

    // 倒排列表: cluster_id -> [(id, codes, norm)]
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u64>, f32)>>,
}

impl IvfRabitQIndex {
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(i64, f32)> {
        // 1. 找最近的 nprobe 个聚类
        let clusters = self.find_nearest_clusters(query, nprobe);

        // 2. 粗排: 使用 RaBitQ 距离
        let mut candidates = Vec::new();
        for &cluster in &clusters {
            for &(id, ref codes, norm) in &self.inverted_lists[&cluster] {
                let dist = self.rabitq.asymmetric_distance(query, codes, norm);
                candidates.push((id, dist));
            }
        }

        // 3. 排序取 top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        candidates
    }
}
```

**工作量**: 3 天

---

## M3: SCANN 索引 (Week 4-5)

### 目标
- 实现 Google ScaNN 算法
- 高召回率 + 高吞吐

### 3.1 SCANN 核心结构

**文件**: `src/faiss/scann.rs`

```rust
//! SCANN: Scalable Nearest Neighbors
//!
//! 参考: Google Research ScaNN
//! 论文: Accelerating Large-Scale Inference with Anisotropic Vector Quantization

/// SCANN 索引
pub struct ScaNNIndex {
    dim: usize,

    // 量化器 (各向异性)
    quantizer: AnisotropicQuantizer,

    // 倒排索引
    inverted_index: InvertedIndex,

    // 重排序参数
    reorder_k: usize,
}

/// 各向异性量化器
pub struct AnisotropicQuantizer {
    dim: usize,
    n_partitions: usize,  // 子空间数
    n_centroids: usize,   // 每个子空间的质心数

    // 码书: [n_partitions * n_centroids * sub_dim]
    codebooks: Vec<f32>,

    // 各向异性权重
    weights: Vec<f32>,
}

impl AnisotropicQuantizer {
    /// 各向异性 K-means 训练
    ///
    /// 目标: 最小化 Σ w_i * ||x_i - c(x_i)||^2
    /// 其中 w_i 根据向量的重要性加权
    pub fn train(&mut self, data: &[f32]) {
        // 1. 计算各向异性权重
        self.compute_weights(data);

        // 2. 对每个子空间训练加权 K-means
        for p in 0..self.n_partitions {
            let sub_vectors = self.extract_subspace(data, p);
            self.train_subspace(p, &sub_vectors);
        }
    }

    fn compute_weights(&mut self, data: &[f32]) {
        // 根据向量与查询的预期角度分布计算权重
        // 权重 = f(angle), angle 越小权重越大
    }

    /// 编码
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.n_partitions);

        for p in 0..self.n_partitions {
            let sub = self.get_subvector(vector, p);
            let centroid = self.find_nearest_centroid(p, &sub);
            codes.push(centroid as u8);
        }

        codes
    }

    /// 非对称距离计算
    pub fn adc_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // 预计算距离表
        let table = self.compute_distance_table(query);

        // 查表求和
        codes.iter().enumerate()
            .map(|(p, &code)| table[p * self.n_centroids + code as usize])
            .sum()
    }
}

/// 倒排索引
pub struct InvertedIndex {
    // 质心 -> [(id, codes)]
    lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,

    // 分区 -> 质心列表
    partition_to_centroids: Vec<Vec<usize>>,
}

impl ScaNNIndex {
    pub fn new(dim: usize, n_partitions: usize, n_centroids: usize) -> Self {
        Self {
            dim,
            quantizer: AnisotropicQuantizer::new(dim, n_partitions, n_centroids),
            inverted_index: InvertedIndex::new(),
            reorder_k: 100,
        }
    }

    /// 训练
    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        self.quantizer.train(data);
        Ok(())
    }

    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        let n = vectors.len() / self.dim;

        for i in 0..n {
            let vector = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.quantizer.encode(vector);

            // 找到最近的质心
            let centroid = self.quantizer.find_partition_centroid(vector);

            let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
            self.inverted_index.add(centroid, id, codes);
        }

        Ok(n)
    }

    /// 搜索
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        // 1. 找到查询所属的分区
        let partition = self.quantizer.find_partition(query);

        // 2. 在分区内搜索
        let candidates = self.search_partition(query, partition, self.reorder_k);

        // 3. 重排序 (使用精确距离)
        let reranked = self.rerank(query, candidates, k);

        reranked
    }

    fn search_partition(&self, query: &[f32], partition: usize, k: usize) -> Vec<(i64, f32)> {
        let mut candidates = Vec::new();

        for &centroid in &self.inverted_index.partition_to_centroids[partition] {
            for &(id, ref codes) in &self.inverted_index.lists[&centroid] {
                let dist = self.quantizer.adc_distance(query, codes);
                candidates.push((id, dist));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        candidates
    }

    fn rerank(&self, query: &[f32], candidates: Vec<(i64, f32)>, k: usize) -> Vec<(i64, f32)> {
        // 使用原始向量重新计算精确距离
        // 需要 GetVectorByIds 支持
        candidates
    }
}
```

**工作量**: 7 天
**验收标准**:
- [ ] 各向异性量化实现
- [ ] Recall@10 ≥ 95%
- [ ] QPS ≥ IVF-PQ 1.5x

---

## M4: FFI/JNI 完善 (Week 6-7)

### 目标
- 完整 C API
- JNI 绑定
- Python 绑定 (可选)

### 4.1 C FFI 完善

**文件**: `src/ffi.rs`, `include/knowhere.h`

```c
// include/knowhere.h

#ifndef KNOWHERE_RS_H
#define KNOWHERE_RS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 类型定义
typedef struct CIndex CIndex;
typedef struct CBitset CBitset;
typedef struct CSearchResult {
    int64_t* ids;
    float* distances;
    size_t num_results;
    float elapsed_ms;
} CSearchResult;

typedef struct CBinarySet CBinarySet;

// 索引类型
typedef enum {
    INDEX_FLAT = 0,
    INDEX_IVF_FLAT = 1,
    INDEX_IVF_PQ = 2,
    INDEX_IVF_SQ8 = 3,
    INDEX_HNSW = 4,
    INDEX_HNSW_SQ = 5,
    INDEX_HNSW_PQ = 6,
    INDEX_DISKANN = 7,
    INDEX_ANNOY = 8,
    INDEX_SCANN = 9,
    INDEX_BINARY = 10,
    INDEX_SPARSE = 11,
} IndexType;

// 度量类型
typedef enum {
    METRIC_L2 = 0,
    METRIC_IP = 1,
    METRIC_COSINE = 2,
    METRIC_HAMMING = 3,
    METRIC_JACCARD = 4,
} MetricType;

// 生命周期
CIndex* knowhere_index_create(IndexType type, MetricType metric, size_t dim);
void knowhere_index_free(CIndex* index);

// 训练和添加
int knowhere_index_train(CIndex* index, const float* data, size_t n);
int knowhere_index_add(CIndex* index, const float* data, const int64_t* ids, size_t n);

// 搜索
int knowhere_index_search(
    CIndex* index,
    const float* queries,
    size_t n_queries,
    size_t k,
    CSearchResult* result
);

// 范围搜索
int knowhere_index_range_search(
    CIndex* index,
    const float* queries,
    size_t n_queries,
    float radius,
    CSearchResult* result
);

// 按ID获取向量
int knowhere_index_get_vectors(
    CIndex* index,
    const int64_t* ids,
    size_t n_ids,
    float* vectors
);

// 序列化
int knowhere_index_save(CIndex* index, const char* path);
CIndex* knowhere_index_load(const char* path);

// 内存序列化
CBinarySet* knowhere_index_serialize(CIndex* index);
CIndex* knowhere_index_deserialize(CBinarySet* binary_set);
void knowhere_binary_set_free(CBinarySet* set);

// Bitset
CBitset* knowhere_bitset_create(size_t n);
void knowhere_bitset_free(CBitset* bitset);
void knowhere_bitset_set(CBitset* bitset, size_t i, bool value);
bool knowhere_bitset_get(CBitset* bitset, size_t i);
size_t knowhere_bitset_count(CBitset* bitset);

// 释放搜索结果
void knowhere_free_search_result(CSearchResult* result);

// 错误信息
const char* knowhere_get_last_error();

#ifdef __cplusplus
}
#endif

#endif // KNOWHERE_RS_H
```

**工作量**: 4 天

### 4.2 JNI 绑定

**文件**: `src/jni/mod.rs`, `src/java/`

```rust
// src/jni/mod.rs
use jni::JNIEnv;
use jni::objects::{JClass, JLongArray, JFloatArray, JIntArray, JObject, JString};
use jni::sys::{jint, jlong, jfloat, jboolean};

mod index;
mod bitset;
mod search_result;

/// 初始化 JNI
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: JavaVM, _reserved: *mut c_void) -> jint {
    // 初始化
    jni::JNIVersion::V1_8.into()
}

// Index JNI
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    index_type: jint,
    metric_type: jint,
    dim: jint,
) -> jlong {
    match create_index(index_type, metric_type, dim as usize) {
        Ok(index) => Box::into_raw(Box::new(index)) as jlong,
        Err(e) => {
            env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeTrain(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    data: JFloatArray,
) {
    let index = unsafe { &mut *(handle as *mut Box<dyn Index>) };
    let data_vec: Vec<f32> = env.get_float_array_elements(&data, ReleaseMode::NoCopyBack)
        .unwrap().iter().map(|&x| x).collect();

    if let Err(e) = index.train(&data_vec) {
        env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
    }
}

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeSearch(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    queries: JFloatArray,
    k: jint,
) -> JObject {
    let index = unsafe { &*(handle as *mut Box<dyn Index>) };
    let query_vec: Vec<f32> = env.get_float_array_elements(&queries, ReleaseMode::NoCopyBack)
        .unwrap().iter().map(|&x| x).collect();

    match index.search(&query_vec, k as usize) {
        Ok(result) => {
            // 返回 SearchResult Java 对象
            create_java_search_result(&mut env, result)
        }
        Err(e) => {
            env.throw_new("java/lang/RuntimeException", &e.to_string()).unwrap();
            JObject::null()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { Box::from_raw(handle as *mut Box<dyn Index>) };
    }
}
```

**Java 包装类**:

```java
// src/java/io/milvus/knowhere/KnowhereIndex.java
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

    public static KnowhereIndex create(IndexType type, MetricType metric, int dim) {
        long handle = nativeCreate(type.getValue(), metric.getValue(), dim);
        return new KnowhereIndex(handle);
    }

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

    public float[] getVectors(long[] ids) {
        checkNotClosed();
        return nativeGetVectors(handle, ids);
    }

    public void save(String path) {
        checkNotClosed();
        nativeSave(handle, path);
    }

    public static KnowhereIndex load(String path) {
        long handle = nativeLoad(path);
        return new KnowhereIndex(handle);
    }

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

    // Native methods
    private static native long nativeCreate(int type, int metric, int dim);
    private native void nativeTrain(long handle, float[] data);
    private native void nativeAdd(long handle, float[] data, long[] ids);
    private native SearchResult nativeSearch(long handle, float[] queries, int k);
    private native float[] nativeGetVectors(long handle, long[] ids);
    private native void nativeSave(long handle, String path);
    private static native long nativeLoad(String path);
    private native void nativeFree(long handle);
}
```

**工作量**: 5 天
**验收标准**:
- [ ] C 头文件完整
- [ ] Java 绑定可用
- [ ] 示例代码运行

---

## M5: 优化完善 (Week 8)

### 目标
- 性能基准测试
- 文档完善
- CI/CD 优化

### 5.1 性能基准

**文件**: `benches/overall_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_sift_1m(c: &mut Criterion) {
    // 加载 SIFT-1M 数据
    let (base, query, gt) = load_sift_1m();

    // Flat
    let mut flat = MemIndex::new(&IndexConfig::new(IndexType::Flat, MetricType::L2, 128)).unwrap();
    flat.add(&base, None);

    c.bench_function("flat_search_1m", |b| {
        b.iter(|| flat.search(&query[..128], 10))
    });

    // HNSW
    let mut hnsw = HnswIndex::new(&IndexConfig::hnsw(128, 16, 200)).unwrap();
    hnsw.train(&base).unwrap();
    hnsw.add(&base, None);

    c.bench_function("hnsw_search_1m", |b| {
        b.iter(|| hnsw.search(&query[..128], 10))
    });

    // IVF-PQ
    let mut ivf_pq = IvfPqIndex::new(&IndexConfig::ivf_pq(128, 100, 8)).unwrap();
    ivf_pq.train(&base).unwrap();
    ivf_pq.add(&base, None);

    c.bench_function("ivf_pq_search_1m", |b| {
        b.iter(|| ivf_pq.search(&query[..128], 10))
    });
}

criterion_group!(benches, bench_sift_1m);
criterion_main!(benches);
```

**工作量**: 2 天

### 5.2 文档

- [ ] API 文档 (rustdoc)
- [ ] README 更新
- [ ] 性能对比报告
- [ ] 使用指南

**工作量**: 2 天

---

## 交付物汇总

| 里程碑 | 交付物 | 文件 |
|-------|-------|------|
| M1 | GetVectorByIds | `src/index.rs`, 各索引 |
| M1 | BinarySet | `src/codec/binary_set.rs` |
| M2 | RaBitQ | `src/quantization/rabitq.rs` |
| M2 | IVF-RaBitQ | `src/faiss/ivf_rabitq.rs` |
| M3 | SCANN | `src/faiss/scann.rs` |
| M4 | C FFI | `src/ffi.rs`, `include/knowhere.h` |
| M4 | JNI | `src/jni/`, `src/java/` |
| M5 | Benchmark | `benches/overall_bench.rs` |

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|-----|------|------|---------|
| RaBitQ 算法复杂 | 高 | 中 | 参考官方实现 |
| JNI 内存管理 | 中 | 中 | 充分测试 |
| 性能目标未达成 | 高 | 低 | 增加优化迭代 |
| 时间延期 | 中 | 中 | 预留缓冲 |

---

## 成功标准

### 8 周目标

| 指标 | 当前 | 目标 |
|-----|------|------|
| 索引类型 | 12 | 15 |
| 功能覆盖 | 70% | 85% |
| 测试覆盖 | 133 | 200+ |
| Recall@10 | 90%+ | 95%+ |
| QPS (vs C++) | 70% | 80% |

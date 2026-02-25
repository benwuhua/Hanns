# Knowhere-RS 详细开发计划

**版本**: 0.3.0 → 1.0.0
**更新日期**: 2026-02-25
**当前状态**: 144 tests passed, 80% 功能覆盖
**目标**: 90% 功能覆盖, 90% 性能

---

## 里程碑概览

| 里程碑 | 目标 | 时间 | 关键交付物 |
|-------|------|------|-----------|
| **M1** | SCANN 索引 | 2 周 | Google ScaNN 实现 |
| **M2** | 迭代器接口 | 1 周 | AnnIterator |
| **M3** | FFI/JNI 完善 | 2 周 | C API, Java 绑定 |
| **M4** | 优化完善 | 1 周 | 性能调优, 文档 |

---

## 当前完成状态

### 已完成 ✅ (P0)

| 功能 | 状态 | 文件 |
|-----|------|------|
| SIMD L2/IP (SSE/AVX2/AVX512/NEON) | ✅ | `src/simd.rs` |
| PQ SIMD 优化 | ✅ | `src/faiss/pq_simd.rs` |
| RaBitQ 量化 (32x) | ✅ | `src/quantization/rabitq.rs` |
| GetVectorByIds | ✅ | `src/index.rs`, `src/faiss/mem_index.rs` |
| BinarySet 序列化 | ✅ | `src/faiss/mem_index.rs` |
| DiskANN 序列化 | ✅ | `src/faiss/diskann.rs` |
| K-means SIMD | ✅ | `src/quantization/kmeans.rs` |

### 索引实现状态

| 索引 | 状态 | 质量 |
|-----|------|------|
| Flat | ✅ | ⭐⭐⭐⭐⭐ |
| HNSW | ✅ | ⭐⭐⭐⭐ |
| HNSW-SQ/PQ | ✅ | ⭐⭐⭐⭐ |
| IVF-Flat | ✅ | ⭐⭐⭐⭐ |
| IVF-PQ | ✅ | ⭐⭐⭐⭐ |
| IVF-SQ8 | ✅ | ⭐⭐⭐⭐ |
| DiskANN | ✅ | ⭐⭐⭐ |
| ANNOY | ✅ | ⭐⭐⭐⭐ |
| Binary | ✅ | ⭐⭐⭐⭐ |
| Sparse | ✅ | ⭐⭐⭐ |

---

## M1: SCANN 索引 (Week 1-2)

### 目标
- 实现 Google ScaNN 算法
- 各向异性向量量化
- 高召回 + 高吞吐

### 1.1 各向异性量化器

**文件**: `src/faiss/scann.rs`

```rust
//! SCANN: Scalable Nearest Neighbors
//!
//! 参考: Google Research ScaNN
//! 论文: Accelerating Large-Scale Inference with Anisotropic Vector Quantization

use std::collections::HashMap;

/// SCANN 配置
pub struct ScaNNConfig {
    /// 子空间数量
    pub num_partitions: usize,
    /// 每个子空间的质心数
    pub num_centroids: usize,
    /// 重排序候选数
    pub reorder_k: usize,
    /// 各向异性权重参数
    pub anisotropic_alpha: f32,
}

impl Default for ScaNNConfig {
    fn default() -> Self {
        Self {
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            anisotropic_alpha: 0.2,
        }
    }
}

/// 各向异性量化器
pub struct AnisotropicQuantizer {
    dim: usize,
    config: ScaNNConfig,

    /// 子空间维度
    sub_dim: usize,

    /// 码书: [num_partitions * num_centroids * sub_dim]
    codebooks: Vec<f32>,

    /// 各向异性权重: [num_partitions * sub_dim]
    weights: Vec<f32>,

    /// 质心归一化因子
    centroid_norms: Vec<f32>,
}

impl AnisotropicQuantizer {
    pub fn new(dim: usize, config: ScaNNConfig) -> Self {
        let sub_dim = dim / config.num_partitions;

        Self {
            dim,
            config,
            sub_dim,
            codebooks: Vec::new(),
            weights: Vec::new(),
            centroid_norms: Vec::new(),
        }
    }

    /// 各向异性 K-means 训练
    ///
    /// 目标: 最小化 Σ w(angle) * ||x - c(x)||²
    /// 其中权重根据向量与查询角度分布计算
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) {
        let n = data.len() / self.dim;

        // 1. 计算各向异性权重
        if let Some(queries) = query_sample {
            self.compute_anisotropic_weights(queries);
        } else {
            // 使用默认权重
            self.weights = vec![1.0; self.config.num_partitions * self.sub_dim];
        }

        // 2. 对每个子空间训练加权 K-means
        for p in 0..self.config.num_partitions {
            let sub_vectors = self.extract_subspace(data, p);
            self.train_subspace_weighted(p, &sub_vectors);
        }
    }

    /// 计算各向异性权重
    ///
    /// 对于每个维度 d，计算权重:
    /// w(d) = 1 / (1 + α * |cos(θ_d)|)
    /// 其中 θ_d 是该维度与查询方向的角度
    fn compute_anisotropic_weights(&mut self, queries: &[f32]) {
        let n_queries = queries.len() / self.dim;
        let alpha = self.config.anisotropic_alpha;

        // 计算查询的主方向
        let mut query_direction = vec![0.0f32; self.dim];
        for q in queries.chunks(self.dim) {
            for (i, &v) in q.iter().enumerate() {
                query_direction[i] += v;
            }
        }
        for v in query_direction.iter_mut() {
            *v /= n_queries as f32;
        }

        // 计算每个子空间的权重
        self.weights.clear();
        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            let end = (start + self.sub_dim).min(self.dim);

            for d in start..end {
                let cos_theta = query_direction[d].abs();
                let w = 1.0 / (1.0 + alpha * cos_theta);
                self.weights.push(w);
            }
        }
    }

    /// 加权 K-means 训练单个子空间
    fn train_subspace_weighted(&mut self, partition: usize, vectors: &[f32]) {
        let n = vectors.len() / self.sub_dim;
        let k = self.config.num_centroids;

        // 初始化质心 (K-means++)
        let mut centroids = self.kmeans_plusplus_init(vectors, k);

        // 迭代优化
        for _iter in 0..50 {
            let mut assignments = vec![0usize; n];
            let mut new_centroids = vec![0.0f32; k * self.sub_dim];
            let mut counts = vec![0usize; k];

            // 分配阶段 (加权距离)
            for i in 0..n {
                let vec = &vectors[i * self.sub_dim..(i + 1) * self.sub_dim];
                let weights = &self.weights[partition * self.sub_dim..(partition + 1) * self.sub_dim];

                let mut min_dist = f32::MAX;
                let mut best = 0;

                for c in 0..k {
                    let centroid = &centroids[c * self.sub_dim..(c + 1) * self.sub_dim];
                    let dist = self.weighted_l2(vec, centroid, weights);

                    if dist < min_dist {
                        min_dist = dist;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // 更新阶段
            new_centroids.fill(0.0);
            counts.fill(0);

            for i in 0..n {
                let c = assignments[i];
                let vec = &vectors[i * self.sub_dim..(i + 1) * self.sub_dim];

                for j in 0..self.sub_dim {
                    new_centroids[c * self.sub_dim + j] += vec[j];
                }
                counts[c] += 1;
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..self.sub_dim {
                        centroids[c * self.sub_dim + j] =
                            new_centroids[c * self.sub_dim + j] / counts[c] as f32;
                    }
                }
            }
        }

        // 存储码书
        let offset = partition * k * self.sub_dim;
        if self.codebooks.len() < offset + k * self.sub_dim {
            self.codebooks.resize(offset + k * self.sub_dim, 0.0);
        }
        self.codebooks[offset..offset + k * self.sub_dim].copy_from_slice(&centroids);

        // 计算质心范数
        for c in 0..k {
            let centroid = &centroids[c * self.sub_dim..(c + 1) * self.sub_dim];
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.centroid_norms.push(norm);
        }
    }

    /// 加权 L2 距离
    fn weighted_l2(&self, a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .zip(weights.iter())
            .map(|((&x, &y), &w)| w * (x - y).powi(2))
            .sum()
    }

    /// 编码向量
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.num_partitions);

        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            let end = (start + self.sub_dim).min(self.dim);
            let sub_vec = &vector[start..end];

            let code = self.find_nearest_centroid(p, sub_vec);
            codes.push(code as u8);
        }

        codes
    }

    /// 非对称距离计算 (ADC)
    pub fn adc_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut distance = 0.0f32;

        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            let end = (start + self.sub_dim).min(self.dim);
            let sub_query = &query[start..end];

            let code = codes[p] as usize;
            let centroid = self.get_centroid(p, code);

            distance += self.l2_distance(sub_query, centroid);
        }

        distance
    }

    fn get_centroid(&self, partition: usize, code: usize) -> &[f32] {
        let offset = partition * self.config.num_centroids * self.sub_dim + code * self.sub_dim;
        &self.codebooks[offset..offset + self.sub_dim]
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

    trained: bool,
}

impl ScaNNIndex {
    pub fn new(dim: usize, config: ScaNNConfig) -> Self {
        Self {
            dim,
            config,
            quantizer: AnisotropicQuantizer::new(dim, config.clone()),
            inverted_lists: HashMap::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            trained: false,
        }
    }

    /// 训练索引
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) {
        self.quantizer.train(data, query_sample);
        self.trained = true;
    }

    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> usize {
        let n = vectors.len() / self.dim;

        for i in 0..n {
            let vector = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.quantizer.encode(vector);

            // 确定分区 (使用第一个子空间的质心)
            let partition = codes[0] as usize % self.config.num_partitions;

            let id = ids.map(|ids| ids[i]).unwrap_or(self.ids.len() as i64);

            self.inverted_lists
                .entry(partition)
                .or_insert_with(Vec::new)
                .push((id, codes));

            self.vectors.extend_from_slice(vector);
            self.ids.push(id);
        }

        n
    }

    /// 搜索
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        // 1. 粗排: 使用 ADC 距离
        let candidates = self.coarse_search(query, self.config.reorder_k);

        // 2. 精排: 使用原始向量重排序
        let reranked = self.rerank(query, candidates, k);

        reranked
    }

    /// 粗排: ADC 距离
    fn coarse_search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let mut candidates = Vec::new();

        for (_, list) in &self.inverted_lists {
            for &(id, ref codes) in list {
                let dist = self.quantizer.adc_distance(query, codes);
                candidates.push((id, dist));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        candidates
    }

    /// 精排: 原始向量重排序
    fn rerank(&self, query: &[f32], candidates: Vec<(i64, f32)>, k: usize) -> Vec<(i64, f32)> {
        let mut reranked: Vec<(i64, f32)> = candidates
            .iter()
            .filter_map(|&(id, _)| {
                // 查找向量
                let pos = self.ids.iter().position(|&x| x == id)?;
                let vector = &self.vectors[pos * self.dim..(pos + 1) * self.dim];
                let dist = self.l2_distance(query, vector);
                Some((id, dist))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked.truncate(k);

        reranked
    }

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
        let config = ScaNNConfig {
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            anisotropic_alpha: 0.2,
        };

        let mut index = ScaNNIndex::new(dim, config);

        // 生成测试数据
        let n = 1000;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.01).sin();
        }

        // 训练
        index.train(&data, None);
        assert!(index.trained);

        // 添加
        let added = index.add(&data, None);
        assert_eq!(added, n);

        // 搜索
        let query = &data[0..dim];
        let results = index.search(query, 10);
        assert_eq!(results.len(), 10);

        // 第一个结果应该是自己 (距离最小)
        assert_eq!(results[0].0, 0);
    }
}
```

**工作量**: 7 天
**验收标准**:
- [ ] 各向异性量化实现
- [ ] 加权 K-means
- [ ] ADC 距离计算
- [ ] Recall@10 ≥ 95%
- [ ] QPS ≥ IVF-PQ 1.2x

---

## M2: 迭代器接口 (Week 3)

### 目标
- 实现 AnnIterator trait
- 支持 HNSW, IVF 等索引
- 增量式返回结果

### 2.1 AnnIterator Trait

**文件**: `src/iterator.rs`

```rust
//! ANN 迭代器接口
//!
//! 支持增量式搜索结果返回

use std::collections::{BinaryHeap, HashSet, HashMap};

/// 有序候选 (用于堆)
#[derive(Clone, Copy, Debug)]
struct OrderedCandidate {
    distance: f32,
    id: i64,
}

impl PartialEq for OrderedCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for OrderedCandidate {}

impl PartialOrd for OrderedCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 最小堆 (距离越小越好)
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}

/// ANN 迭代器 trait
pub trait AnnIterator: Iterator<Item = (i64, f32)> + Send {
    /// 获取当前最佳距离
    fn current_distance(&self) -> Option<f32>;

    /// 是否已耗尽
    fn is_exhausted(&self) -> bool;

    /// 获取已访问的节点数
    fn visited_count(&self) -> usize;
}

/// HNSW 迭代器
pub struct HnswIterator<'a> {
    index: &'a crate::faiss::HnswIndex,
    query: Vec<f32>,

    /// 已访问节点
    visited: HashSet<i64>,

    /// 候选堆 (最小堆)
    candidates: BinaryHeap<OrderedCandidate>,

    /// 已返回结果数
    returned: usize,

    /// 最大返回数
    max_results: usize,

    /// 当前最佳距离
    current_best: Option<f32>,
}

impl<'a> HnswIterator<'a> {
    pub fn new(index: &'a crate::faiss::HnswIndex, query: Vec<f32>, max_results: usize) -> Self {
        let mut iter = Self {
            index,
            query,
            visited: HashSet::new(),
            candidates: BinaryHeap::new(),
            returned: 0,
            max_results,
            current_best: None,
        };

        // 从入口点开始
        if let Some(ep) = iter.index.entry_point {
            let dist = iter.compute_distance(ep);
            iter.candidates.push(OrderedCandidate { distance: dist, id: ep });
        }

        iter
    }

    fn compute_distance(&self, id: i64) -> f32 {
        // 获取向量并计算距离
        let vectors = self.index.get_vectors_by_ids(&[id]).unwrap_or_default();
        if vectors.is_empty() {
            return f32::MAX;
        }

        self.query
            .iter()
            .zip(vectors.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// 扩展候选 (访问邻居)
    fn expand(&mut self, id: i64) {
        // 获取邻居并添加到候选堆
        // 需要索引支持获取邻居列表
    }
}

impl<'a> Iterator for HnswIterator<'a> {
    type Item = (i64, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.returned >= self.max_results {
            return None;
        }

        loop {
            let candidate = self.candidates.pop()?;

            if self.visited.contains(&candidate.id) {
                continue;
            }

            self.visited.insert(candidate.id);

            // 扩展邻居
            self.expand(candidate.id);

            self.current_best = Some(candidate.distance);
            self.returned += 1;

            return Some((candidate.id, candidate.distance));
        }
    }
}

impl<'a> AnnIterator for HnswIterator<'a> {
    fn current_distance(&self) -> Option<f32> {
        self.current_best
    }

    fn is_exhausted(&self) -> bool {
        self.candidates.is_empty() || self.returned >= self.max_results
    }

    fn visited_count(&self) -> usize {
        self.visited.len()
    }
}

/// IVF 迭代器
pub struct IvfIterator<'a> {
    index: &'a crate::faiss::IvfIndex,
    query: Vec<f32>,

    /// 待搜索的聚类列表
    cluster_order: Vec<usize>,
    current_cluster_idx: usize,

    /// 当前聚类的候选
    current_candidates: Vec<(i64, f32)>,
    current_candidate_idx: usize,

    /// 已返回数
    returned: usize,
    max_results: usize,

    current_best: Option<f32>,
}

impl<'a> IvfIterator<'a> {
    pub fn new(index: &'a crate::faiss::IvfIndex, query: Vec<f32>, nprobe: usize, max_results: usize) -> Self {
        // 找最近的 nprobe 个聚类
        let cluster_order = index.find_nearest_clusters(&query, nprobe);

        Self {
            index,
            query,
            cluster_order,
            current_cluster_idx: 0,
            current_candidates: Vec::new(),
            current_candidate_idx: 0,
            returned: 0,
            max_results,
            current_best: None,
        }
    }

    fn load_next_cluster(&mut self) -> bool {
        while self.current_cluster_idx < self.cluster_order.len() {
            let cluster = self.cluster_order[self.current_cluster_idx];
            self.current_cluster_idx += 1;

            // 获取该聚类的所有向量
            if let Some(list) = self.index.get_inverted_list(cluster) {
                let mut candidates: Vec<(i64, f32)> = list
                    .iter()
                    .map(|&id| {
                        let dist = self.compute_distance_to_id(id);
                        (id, dist)
                    })
                    .collect();

                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                self.current_candidates = candidates;
                self.current_candidate_idx = 0;
                return true;
            }
        }
        false
    }

    fn compute_distance_to_id(&self, _id: i64) -> f32 {
        // 实现距离计算
        f32::MAX
    }
}

impl<'a> Iterator for IvfIterator<'a> {
    type Item = (i64, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.returned >= self.max_results {
            return None;
        }

        loop {
            if self.current_candidate_idx < self.current_candidates.len() {
                let result = self.current_candidates[self.current_candidate_idx];
                self.current_candidate_idx += 1;
                self.returned += 1;
                self.current_best = Some(result.1);
                return Some(result);
            }

            if !self.load_next_cluster() {
                return None;
            }
        }
    }
}

impl<'a> AnnIterator for IvfIterator<'a> {
    fn current_distance(&self) -> Option<f32> {
        self.current_best
    }

    fn is_exhausted(&self) -> bool {
        self.returned >= self.max_results &&
            self.current_candidate_idx >= self.current_candidates.len() &&
            self.current_cluster_idx >= self.cluster_order.len()
    }

    fn visited_count(&self) -> usize {
        self.returned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordered_candidate() {
        let a = OrderedCandidate { distance: 0.5, id: 1 };
        let b = OrderedCandidate { distance: 0.3, id: 2 };

        // 最小堆: 距离小的优先
        assert!(b < a);
    }
}
```

**工作量**: 3 天

---

## M3: FFI/JNI 完善 (Week 4-5)

### 目标
- 完善 C FFI
- 实现 JNI 绑定
- Python 绑定 (可选)

### 3.1 C FFI 完善

**文件**: `src/ffi.rs`, `include/knowhere.h`

```c
// include/knowhere.h - 更新

// 新增迭代器接口
typedef struct CAnnIterator CAnnIterator;

CAnnIterator* knowhere_iterator_create(CIndex* index, const float* query, size_t max_results);
bool knowhere_iterator_next(CAnnIterator* iter, int64_t* id, float* distance);
void knowhere_iterator_free(CAnnIterator* iter);

// 新增 GetVectorByIds
int knowhere_index_get_vector_by_ids(
    CIndex* index,
    const int64_t* ids,
    size_t n_ids,
    float* vectors_out
);

// 新增 BinarySet 序列化
typedef struct CBinarySet CBinarySet;

CBinarySet* knowhere_index_serialize(CIndex* index);
CIndex* knowhere_index_deserialize(CBinarySet* set);
void knowhere_binary_set_free(CBinarySet* set);
size_t knowhere_binary_set_size(CBinarySet* set);
int knowhere_binary_set_get(CBinarySet* set, const char* name, void* data_out, size_t* size);
```

**工作量**: 3 天

### 3.2 JNI 绑定

**文件**: `src/jni/mod.rs`

```rust
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JLongArray, JFloatArray};
use jni::sys::{jint, jlong, jfloat, jbooleanArray};

mod index;
mod iterator;
mod binary_set;

/// 初始化
#[no_mangle]
pub extern "system" fn JNI_OnLoad(vm: jni::JavaVM, _reserved: *mut std::ffi::c_void) -> jint {
    jni::JNIVersion::V1_8.into()
}

// Index 方法
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    index_type: jint,
    metric_type: jint,
    dim: jint,
) -> jlong;

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_nativeGetVectorByIds(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    ids: JLongArray,
) -> JFloatArray;

// Iterator 方法
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIterator_nativeCreate(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    query: JFloatArray,
    max_results: jint,
) -> jlong;

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIterator_nativeNext(
    mut env: JNIEnv,
    _class: JClass,
    iter_handle: jlong,
) -> JObject; // Returns Optional<SearchResult>

// BinarySet 方法
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereBinarySet_nativeSerialize(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jbyteArray;

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereBinarySet_nativeDeserialize(
    mut env: JNIEnv,
    _class: JClass,
    data: jbyteArray,
) -> jlong;
```

**Java 类**:

```java
// KnowhereIterator.java
package io.milvus.knowhere;

import java.util.Iterator;
import java.util.Optional;

public class KnowhereIterator implements Iterator<SearchResult>, AutoCloseable {
    private long handle;
    private SearchResult next;
    private boolean closed = false;

    KnowhereIterator(long handle) {
        this.handle = handle;
        advance();
    }

    @Override
    public boolean hasNext() {
        return next != null && !closed;
    }

    @Override
    public SearchResult next() {
        if (closed || next == null) {
            throw new IllegalStateException("Iterator exhausted or closed");
        }
        SearchResult result = next;
        advance();
        return result;
    }

    private void advance() {
        Optional<SearchResult> opt = nativeNext(handle);
        next = opt.orElse(null);
    }

    @Override
    public void close() {
        if (!closed) {
            nativeFree(handle);
            handle = 0;
            closed = true;
        }
    }

    private native Optional<SearchResult> nativeNext(long handle);
    private native void nativeFree(long handle);
}
```

**工作量**: 5 天

---

## M4: 优化完善 (Week 6)

### 目标
- 性能基准测试
- 文档完善
- CI/CD 优化

### 4.1 性能基准

**文件**: `benches/comparison_bench.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn bench_index_comparison(c: &mut Criterion) {
    let dim = 128;
    let n = 1_000_000;

    // 生成数据
    let data = generate_random_vectors(n, dim);
    let query = generate_random_vectors(100, dim);

    // Flat
    let mut flat = MemIndex::new(&IndexConfig::flat(dim, MetricType::L2)).unwrap();
    flat.add(&data, None);

    c.bench_function("flat_search_1m", |b| {
        b.iter(|| {
            for q in query.chunks(dim) {
                flat.search(q, 10);
            }
        })
    });

    // HNSW
    let mut hnsw = HnswIndex::new(&IndexConfig::hnsw(dim, 16, 200)).unwrap();
    hnsw.train(&data).unwrap();
    hnsw.add(&data, None);

    c.bench_function("hnsw_search_1m", |b| {
        b.iter(|| {
            for q in query.chunks(dim) {
                hnsw.search(q, 10);
            }
        })
    });

    // IVF-PQ
    let mut ivf_pq = IvfPqIndex::new(&IndexConfig::ivf_pq(dim, 100, 8)).unwrap();
    ivf_pq.train(&data).unwrap();
    ivf_pq.add(&data, None);

    c.bench_function("ivf_pq_search_1m", |b| {
        b.iter(|| {
            for q in query.chunks(dim) {
                ivf_pq.search(q, 10);
            }
        })
    });

    // SCANN
    let mut scann = ScaNNIndex::new(dim, ScaNNConfig::default());
    scann.train(&data, None);
    scann.add(&data, None);

    c.bench_function("scann_search_1m", |b| {
        b.iter(|| {
            for q in query.chunks(dim) {
                scann.search(q, 10);
            }
        })
    });
}

criterion_group!(benches, bench_index_comparison);
criterion_main!(benches);
```

**工作量**: 2 天

### 4.2 文档更新

- [ ] API 文档 (rustdoc)
- [ ] README 更新
- [ ] 性能对比报告
- [ ] 迁移指南 (从 C++ Knowhere)

**工作量**: 2 天

---

## 交付物汇总

| 里程碑 | 交付物 | 文件 |
|-------|-------|------|
| M1 | SCANN 索引 | `src/faiss/scann.rs` |
| M1 | 各向异性量化器 | `src/faiss/scann.rs` |
| M2 | AnnIterator trait | `src/iterator.rs` |
| M2 | HNSW/IVF 迭代器 | `src/iterator.rs` |
| M3 | C FFI 完善 | `src/ffi.rs`, `include/knowhere.h` |
| M3 | JNI 绑定 | `src/jni/`, `src/java/` |
| M4 | 性能基准 | `benches/comparison_bench.rs` |

---

## 成功标准

### 6 周目标

| 指标 | 当前 | 目标 |
|-----|------|------|
| 索引类型 | 13 | 15 |
| 功能覆盖 | 80% | 90% |
| 测试覆盖 | 144 | 200+ |
| Recall@10 | 95%+ | 95%+ |
| QPS (vs C++) | 85% | 90% |
| API 完整度 | 85% | 95% |

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|-----|------|------|---------|
| SCANN 算法复杂 | 高 | 中 | 参考官方实现 |
| JNI 内存管理 | 中 | 中 | 充分测试 |
| 性能目标未达成 | 高 | 低 | 增加优化迭代 |
| 时间延期 | 中 | 中 | 预留缓冲 |

---

## 长期目标 (P3)

| 功能 | 时间 | 说明 |
|-----|------|------|
| GPU 支持 (wgpu) | 长期 | 需要 GPU 计算基础 |
| 混合搜索 | 1-2 周 | 多模态搜索 |
| PRQ 量化 | 1 周 | 残差量化 |
| Python 绑定 | 1 周 | PyO3 |

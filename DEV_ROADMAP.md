# Knowhere-RS 详细开发计划

**版本**: 0.2.0 → 1.0.0
**时间范围**: 2026-02 - 2026-08
**目标**: 达到 C++ Knowhere 85% 功能覆盖，70% 性能

---

## 里程碑概览

| 里程碑 | 目标 | 时间 | 关键交付物 |
|-------|------|------|-----------|
| **M1** | SIMD 完善 | 2 周 | 内积 SIMD, AVX-512, 批量优化 |
| **M2** | DiskANN 重构 | 3 周 | Vamana 算法, 磁盘存储 |
| **M3** | HNSW 重构 | 2 周 | 多层结构, 动态删除 |
| **M4** | 量化完善 | 2 周 | IVF-SQ8, RaBitQ |
| **M5** | FFI/JNI | 3 周 | C API, Java 绑定 |
| **M6** | 质量保证 | 2 周 | 测试, 文档, 基准 |

---

## M1: SIMD 完善 (Week 1-2)

### 目标
- 内积 SIMD 实现
- AVX-512 支持
- 批量距离优化
- 性能提升 3-5x

### 任务清单

#### 1.1 内积 SIMD (`src/simd.rs`)

```rust
// 添加以下函数:

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.2")]
pub unsafe fn ip_sse(a: &[f32], b: &[f32]) -> f32 {
    // SSE 4.2 内积实现
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn ip_avx2(a: &[f32], b: &[f32]) -> f32 {
    // AVX2 内积实现 (8x f32 并行)
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub unsafe fn ip_neon(a: &[f32], b: &[f32]) -> f32 {
    // NEON 内积实现
}
```

**工作量**: 2 天
**验收标准**:
- [ ] SSE 内积性能 ≥ 标量 3x
- [ ] AVX2 内积性能 ≥ 标量 6x
- [ ] NEON 内积性能 ≥ 标量 3x
- [ ] 测试覆盖所有路径

#### 1.2 AVX-512 支持

```rust
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn l2_avx512(a: &[f32], b: &[f32]) -> f32 {
    // AVX-512 实现 (16x f32 并行)
}

// 批量距离计算 (矩阵乘法风格)
pub fn l2_batch_avx512(queries: &[f32], database: &[f32], dim: usize) -> Vec<f32> {
    // 优化的批量距离计算
}
```

**工作量**: 3 天
**验收标准**:
- [ ] AVX-512 L2 性能 ≥ AVX2 1.5x
- [ ] 批量计算性能 ≥ 逐个计算 5x
- [ ] 支持 VPOPCNTDQ (如果可用)

#### 1.3 余弦距离 SIMD

**工作量**: 1 天

#### 1.4 基准测试

```rust
// benches/simd_bench.rs
#[bench]
fn bench_l2_128_scalar(b: &mut Bencher) { ... }
#[bench]
fn bench_l2_128_avx2(b: &mut Bencher) { ... }
#[bench]
fn bench_ip_128_avx2(b: &mut Bencher) { ... }
```

**工作量**: 1 天

### 交付物

- [ ] `src/simd.rs` 完整 SIMD 实现
- [ ] `benches/simd_bench.rs` 性能基准
- [ ] 性能报告文档

---

## M2: DiskANN 重构 (Week 3-5)

### 目标
- 实现标准 Vamana 算法
- 磁盘存储优化
- 支持 PQ 编码
- Recall@10 ≥ 95%

### 任务清单

#### 2.1 Vamana 图构建算法

**文件**: `src/faiss/diskann_vamana.rs`

```rust
pub struct VamanaIndex {
    // 图结构
    graph: Vec<Vec<Neighbor>>,  // 邻接表
    data: Vec<f32>,             // 原始向量
    pq_codes: Option<Vec<u8>>,  // PQ 编码 (可选)

    // 参数
    r: usize,      // 最大出度 (默认 64)
    l: usize,      // 搜索列表大小 (默认 100)
    alpha: f32,    // 贪婪参数 (默认 1.2)

    // 入口点
    medoid: usize,
}

impl VamanaIndex {
    /// Vamana 图构建
    pub fn build(&mut self, data: &[f32]) {
        // 1. 随机初始化 R-regular 图
        self.init_random_graph();

        // 2. 计算中位点
        self.medoid = self.find_medoid();

        // 3. 迭代优化 (2 轮)
        for iter in 0..2 {
            let order = if iter == 0 {
                // 第一轮: 随机顺序
                random_permutation(self.n)
            } else {
                // 第二轮: 按距离排序
                distance_ordered_permutation(self.n)
            };

            for &i in &order {
                let candidates = self.search_with_visit(self.medoid, &data[i], self.l);
                self.robust_prune(i, &candidates, self.alpha, self.r);

                // 反向连接
                self.add_reverse_edges(i);
            }
        }
    }

    /// RobustPrune 剪枝算法
    fn robust_prune(&mut self, i: usize, candidates: &[Candidate], alpha: f32, r: usize) {
        let mut neighbors = Vec::with_capacity(r);
        let mut covered = HashSet::new();

        for cand in candidates {
            if neighbors.len() >= r { break; }

            // 检查是否被已有邻居覆盖
            let mut dominated = false;
            for &nbr in &neighbors {
                let dist_to_nbr = self.distance(cand.id, nbr.id);
                if dist_to_nbr * alpha < cand.distance {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                neighbors.push(cand);
                covered.insert(cand.id);
            }
        }

        self.graph[i] = neighbors;
    }
}
```

**工作量**: 5 天
**验收标准**:
- [ ] 正确实现 RobustPrune
- [ ] 两轮迭代优化
- [ ] Recall@10 ≥ 95% (SIFT-1M)

#### 2.2 磁盘存储格式

**文件**: `src/storage/diskann_format.rs`

```
文件布局:
├── header.bin      # 元数据 (dim, n, r, etc.)
├── vectors.bin     # 原始向量 (可选)
├── pq_codes.bin    # PQ 编码向量
├── graph.bin       # 图结构
└── centroids.bin   # PQ 码书
```

**工作量**: 3 天
**验收标准**:
- [ ] 紧凑二进制格式
- [ ] 支持内存映射
- [ ] 支持增量加载

#### 2.3 PQ 编码支持

**工作量**: 2 天

#### 2.4 搜索优化

```rust
impl VamanaIndex {
    pub fn search(&self, query: &[f32], k: usize, l: usize) -> Vec<(i64, f32)> {
        // 1. 贪婪搜索获取候选
        let candidates = self.greedy_search(self.medoid, query, l);

        // 2. 如果使用 PQ，进行距离表计算
        let pq_table = self.compute_distance_table(query);

        // 3. 重排序
        let reranked = self.rerank(query, candidates, k);

        reranked
    }
}
```

**工作量**: 2 天

### 交付物

- [ ] `src/faiss/diskann_vamana.rs` Vamana 实现
- [ ] `src/storage/diskann_format.rs` 磁盘格式
- [ ] SIFT-1M 基准测试结果

---

## M3: HNSW 重构 (Week 6-7)

### 目标
- 多层图结构
- 动态删除支持
- 性能提升 2-3x

### 任务清单

#### 3.1 多层 HNSW 结构

**文件**: `src/faiss/hnsw_multi_layer.rs`

```rust
pub struct HnswMultiLayer {
    // 层级结构
    max_level: usize,
    level_mult: f64,          // 层级分配参数 (1/ln(M))
    entry_point: usize,

    // 每层图
    layers: Vec<HashMap<usize, Vec<Neighbor>>>,

    // 向量数据
    data: Vec<f32>,
    dim: usize,

    // 参数
    m: usize,                 // 每层最大连接数
    ef_construction: usize,
    ef_search: usize,
}

impl HnswMultiLayer {
    /// 随机层级分配
    fn random_level(&self) -> usize {
        let mut level = 0;
        let mut r: f64 = random();
        while r < 1.0 / self.level_mult && level < self.max_level {
            level += 1;
            r = random();
        }
        level
    }

    /// 插入向量
    pub fn insert(&mut self, vector: &[f32]) -> usize {
        let id = self.data.len() / self.dim;
        let level = self.random_level();

        // 自顶向下搜索
        let mut entry = self.entry_point;
        for l in (level..=self.max_level).rev() {
            let nearest = self.search_layer(entry, vector, 1, l);
            if !nearest.is_empty() {
                entry = nearest[0].id;
            }
        }

        // 自底向上连接
        for l in (0..=level).rev() {
            let candidates = self.search_layer(entry, vector, self.ef_construction, l);
            let neighbors = self.select_neighbors_heuristic(&candidates, self.m);
            self.connect(id, &neighbors, l);
        }

        if level > self.get_level(self.entry_point) {
            self.entry_point = id;
        }

        id
    }

    /// Heuristic 邻居选择
    fn select_neighbors_heuristic(&self, candidates: &[Candidate], m: usize) -> Vec<usize> {
        // 实现 heuristic-select-neighbors 算法
        // 参考论文: Efficient and robust approximate nearest neighbor search
    }
}
```

**工作量**: 4 天
**验收标准**:
- [ ] 正确的层级分配 (指数分布)
- [ ] 逐层搜索
- [ ] Heuristic 邻居选择
- [ ] Recall@10 ≥ 98%

#### 3.2 动态删除

```rust
impl HnswMultiLayer {
    /// 软删除 (标记)
    pub fn mark_deleted(&mut self, id: usize) {
        self.deleted.insert(id);
    }

    /// 硬删除 (重建局部图)
    pub fn delete(&mut self, id: usize) {
        // 1. 移除所有指向该节点的边
        // 2. 重新连接邻居
        // 3. 更新入口点 (如需要)
    }
}
```

**工作量**: 2 天

#### 3.3 性能优化

- [ ] 预取优化
- [ ] 紧凑内存布局
- [ ] 并行构建

**工作量**: 2 天

### 交付物

- [ ] `src/faiss/hnsw_multi_layer.rs`
- [ ] 动态删除 API
- [ ] 性能对比报告

---

## M4: 量化完善 (Week 8-9)

### 目标
- IVF-SQ8 索引
- RaBitQ 量化
- HNSW-SQ 变体

### 任务清单

#### 4.1 IVF-SQ8 索引

**文件**: `src/faiss/ivf_sq8.rs`

```rust
pub struct IvfSq8Index {
    // IVF 结构
    nlist: usize,
    centroids: Vec<f32>,

    // SQ8 量化
    sq: ScalarQuantizer,

    // 倒排列表 (量化编码)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
}

impl IvfSq8Index {
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(i64, f32)> {
        // 1. 找到最近的 nprobe 个聚类
        let clusters = self.find_nearest_clusters(query, nprobe);

        // 2. 遍历倒排列表
        let mut candidates = Vec::new();
        for &cluster in &clusters {
            for &(id, codes) in &self.inverted_lists[&cluster] {
                // 3. 解码并计算距离
                let vector = self.sq.decode(&codes);
                let dist = l2_distance(query, &vector);
                candidates.push((id, dist));
            }
        }

        // 4. 排序取 top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }
}
```

**工作量**: 3 天

#### 4.2 RaBitQ 量化

**文件**: `src/quantization/rabitq.rs`

```rust
/// RaBitQ: 1-bit Random Quantization
pub struct RaBitQ {
    dim: usize,

    // 随机旋转矩阵
    rotation: Vec<f32>,

    // 二值码书
    binary_centroids: Vec<u64>,  // 每个质心 64 位
    residual_centroids: Vec<f32>, // 残差修正
}

impl RaBitQ {
    /// 训练
    pub fn train(&mut self, data: &[f32]) {
        // 1. 应用随机旋转
        let rotated = self.apply_rotation(data);

        // 2. 二值化
        for vec in rotated.chunks(self.dim) {
            let binary = self.binarize(vec);
            self.binary_centroids.push(binary);
        }

        // 3. 计算残差
        // ...
    }

    /// 编码 (32x 压缩)
    pub fn encode(&self, vector: &[f32]) -> u64 {
        let rotated = self.rotate(vector);
        self.binarize(&rotated)
    }

    /// 距离计算 (使用 popcount)
    pub fn distance(&self, a: u64, b: u64) -> f32 {
        let xor = a ^ b;
        let hamming = xor.count_ones() as f32;
        hamming / 64.0
    }
}
```

**工作量**: 4 天
**验收标准**:
- [ ] 32x 压缩比
- [ ] Recall@10 ≥ 90%
- [ ] QPS ≥ IVF-PQ 1.5x

#### 4.3 HNSW-SQ

**工作量**: 2 天

### 交付物

- [ ] `src/faiss/ivf_sq8.rs`
- [ ] `src/quantization/rabitq.rs`
- [ ] `src/faiss/hnsw_sq.rs`

---

## M5: FFI/JNI (Week 10-12)

### 目标
- 完整 C API
- JNI 绑定
- Python 绑定 (可选)

### 任务清单

#### 5.1 C FFI 完善

**文件**: `src/ffi.rs`

```c
// knowhere.h
typedef struct CIndex CIndex;
typedef struct CBitset CBitset;

// 生命周期
CIndex* knowhere_index_create(IndexType type, size_t dim, MetricType metric);
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

// 序列化
int knowhere_index_save(CIndex* index, const char* path);
CIndex* knowhere_index_load(const char* path);

// Bitset
CBitset* knowhere_bitset_create(size_t n);
void knowhere_bitset_set(CBitset* bitset, size_t i, bool value);
bool knowhere_bitset_get(CBitset* bitset, size_t i);
void knowhere_bitset_free(CBitset* bitset);
```

**工作量**: 4 天

#### 5.2 JNI 绑定

**文件**: `src/jni/mod.rs`

```rust
use jni::JNIEnv;
use jni::objects::{JClass, JLongArray, JFloatArray, JObject};

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_create(
    mut env: JNIEnv,
    _class: JClass,
    index_type: jint,
    dim: jint,
    metric_type: jint,
) -> jlong {
    // 创建索引，返回句柄
}

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_search(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    queries: JFloatArray,
    k: jint,
) -> JObject {
    // 执行搜索，返回 Java 对象
}
```

**Java 包装类**:

```java
// src/java/io/milvus/knowhere/KnowhereIndex.java
package io.milvus.knowhere;

public class KnowhereIndex implements AutoCloseable {
    private long handle;

    static {
        System.loadLibrary("knowhere_rs");
    }

    public native static KnowhereIndex create(IndexType type, int dim, MetricType metric);
    public native void train(float[] data);
    public native void add(float[] data, long[] ids);
    public native SearchResult search(float[] queries, int k);
    public native void save(String path);
    public native void load(String path);
    public native void close();
}
```

**工作量**: 5 天

#### 5.3 Python 绑定 (可选)

使用 PyO3:

```python
# Python API
import knowhere_rs

index = knowhere_rs.HnswIndex(dim=128, m=16, ef_construction=200)
index.train(train_data)
index.add(vectors)
results = index.search(queries, k=10)
```

**工作量**: 3 天

### 交付物

- [ ] `include/knowhere.h` C 头文件
- [ ] `src/ffi.rs` 完整实现
- [ ] `src/jni/` JNI 绑定
- [ ] `src/java/` Java 包装类
- [ ] 示例代码

---

## M6: 质量保证 (Week 13-14)

### 目标
- 测试覆盖率 ≥ 80%
- 完整文档
- 性能基准

### 任务清单

#### 6.1 测试完善

```rust
// tests/integration_test.rs

#[test]
fn test_sift_1m_hnsw() {
    // 加载 SIFT-1M 数据集
    let (base, query, gt) = load_sift_1m();

    // 构建索引
    let mut index = HnswMultiLayer::new(128, 16, 200);
    index.train(&base);
    index.add(&base);

    // 搜索
    let mut recall = 0;
    for (i, q) in query.chunks(128).enumerate() {
        let results = index.search(q, 10, 100);
        recall += compute_recall(&results, &gt[i]);
    }

    assert!(recall / query.len() >= 0.95);
}
```

**工作量**: 3 天

#### 6.2 基准测试

```rust
// benches/overall_bench.rs

#[bench]
fn bench_hnsw_build_1m(b: &mut Bencher) { ... }
#[bench]
fn bench_hnsw_search_1m(b: &mut Bencher) { ... }
#[bench]
fn bench_ivf_search_1m(b: &mut Bencher) { ... }
```

**工作量**: 2 天

#### 6.3 文档

- [ ] API 文档 (rustdoc)
- [ ] 架构文档
- [ ] 使用指南
- [ ] 性能调优指南

**工作量**: 3 天

### 交付物

- [ ] 测试覆盖率 ≥ 80%
- [ ] 基准测试报告
- [ ] 完整文档

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|-----|------|------|---------|
| SIMD 兼容性问题 | 高 | 中 | 充分测试不同 CPU |
| Vamana 算法复杂 | 高 | 中 | 参考参考实现 |
| JNI 内存管理 | 中 | 中 | 仔细处理生命周期 |
| 性能目标未达成 | 高 | 低 | 增加优化迭代 |
| 时间延期 | 中 | 中 | 预留缓冲时间 |

---

## 资源需求

### 人力

- **主力开发**: 1 人 (全职)
- **代码审查**: 1 人 (兼职)
- **测试**: 1 人 (兼职)

### 硬件

- **开发机**: x86_64 (支持 AVX-512) + ARM (M1/M2)
- **测试服务器**: 32GB+ 内存
- **GPU 服务器** (可选): NVIDIA GPU

---

## 成功标准

### 6 个月目标

| 指标 | 目标 |
|-----|------|
| 索引类型 | 10+ 种 |
| SIMD 覆盖 | 100% L2/IP |
| 测试覆盖 | ≥ 80% |
| Recall@10 (SIFT-1M) | ≥ 95% |
| QPS (vs C++) | ≥ 70% |
| 文档完整性 | ≥ 90% |

### 1.0.0 发布标准

- [ ] 所有 P0/P1 功能完成
- [ ] 测试覆盖 ≥ 80%
- [ ] 性能达标
- [ ] 文档完整
- [ ] 无 P0/P1 Bug

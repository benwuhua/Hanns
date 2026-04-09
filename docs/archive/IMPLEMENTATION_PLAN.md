# Hanns 实现计划

基于与 C++ Knowhere 的差距分析，本文档详细规划实现路径。

> **更新 (2026-02-24)**: 远程已实现 SIMD、SQ 量化、并行 K-means 等功能。计划已更新。

---

## 已完成功能 ✅

| 功能 | 状态 | 提交 |
|-----|------|------|
| SIMD 距离 (AVX2/SSE/NEON) | ✅ 完成 | `362c8d4` |
| 并行 K-means | ✅ 完成 | `362c8d4` |
| SQ8/SQ4 标量量化 | ✅ 完成 | `c37c9e2` |
| Range 搜索 API | ✅ 完成 | `c37c9e2` |
| DiskANN SIMD 优化 | ✅ 完成 | `4b8fba9` |
| IVF-PQ 标准化 | ✅ 完成 | `ea6ebc7` |

---

## 阶段 0: 基础设施准备 (已完成)

### 0.1 构建系统优化
- [ ] 添加 SIMD feature flags (`sse`, `avx2`, `avx512`, `neon`)
- [ ] 配置 `build.rs` 进行运行时 CPU 特性检测
- [ ] 添加 benchmark 基线测试

### 0.2 测试框架
- [ ] 添加 property-based testing (proptest)
- [ ] 创建性能回归测试
- [ ] 添加与 C++ Knowhere 的输出对比测试

---

## 阶段 1: SIMD 优化 (✅ 已完成)

### 1.1 x86 SIMD 实现

**文件**: `src/simd.rs`

```
状态: ✅ 已完成
提交: 362c8d4, 8cf53b7
```

**已实现**:
- [x] SSE 4.2 L2 距离实现
- [x] SSE 4.2 内积实现
- [x] AVX2 L2 距离实现 (256-bit)
- [x] AVX2 内积实现
- [x] AVX-512 支持
- [x] 运行时特性检测与分发

**代码结构**:
```rust
// src/simd/x86.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    // 8 f32 并行处理
    assert_eq!(a.len() % 8, 0);

    let mut sum = _mm256_setzero_ps();
    for i in (0..a.len()).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // 水平求和
    horizontal_sum_avx2(sum).sqrt()
}
```

### 1.2 ARM NEON 实现

**文件**: `src/simd.rs`

```
状态: ✅ 已完成
```

**已实现**:
- [x] NEON L2 距离
- [x] NEON 内积
- [x] NEON 余弦距离

### 1.3 SIMD 分发机制

**文件**: `src/simd/dispatch.rs`

```rust
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_supported() {
            return unsafe { l2_distance_avx2(a, b) };
        }
        if is_sse42_supported() {
            return unsafe { l2_distance_sse(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_distance_neon(a, b) };
    }
    l2_distance_scalar(a, b)
}
```

---

## 阶段 2: DiskANN 完整实现 (P0)

### 2.1 Vamana 图构建算法

**文件**: `src/faiss/diskann_vamana.rs`

```
优先级: P0
工作量: 5-7 天
依赖: SIMD 优化
```

**算法核心**:
1. 随机初始化图
2. 迭代优化:
   - 对每个点执行贪婪搜索获取候选邻居
   - 剪枝邻居列表 (RobustPrune)
   - 维护双向连接

**任务**:
- [ ] 实现 `RobustPrune` 剪枝算法
- [ ] 实现带 alpha 参数的贪婪搜索
- [ ] 支持增量构建
- [ ] 添加搜索路径缓存

**代码框架**:
```rust
pub struct VamanaBuilder {
    r: usize,          // 最大出度
    l: usize,          // 搜索列表大小
    alpha: f32,        // 贪婪参数
    graph: Vec<Vec<Neighbor>>,
}

impl VamanaBuilder {
    pub fn build(&mut self, data: &[f32]) {
        // 1. 随机初始化 R-regular 图
        self.random_init();

        // 2. 计算中位点作为入口
        self.medoid = self.find_medoid();

        // 3. 迭代优化
        for iter in 0..self.max_iter {
            for i in 0..self.n {
                let candidates = self.search_with_visit(self.medoid, &data[i..], self.l);
                self.robust_prune(i, &candidates, self.alpha, self.r);
            }
        }
    }
}
```

### 2.2 磁盘存储优化

**文件**: `src/storage/diskann_disk.rs`

**任务**:
- [ ] 实现 PQ 编码向量存储
- [ ] 实现图结构紧凑存储
- [ ] 支持内存映射读取
- [ ] 实现预取优化

---

## 阶段 3: 标准化 IVF-PQ (P0)

### 3.1 标准 PQ 编码

**文件**: `src/quantization/pq.rs`

```
优先级: P0
工作量: 4-5 天
依赖: SIMD, K-means
```

**当前问题**: IVF-PQ 存储原始残差而非 PQ 编码

**任务**:
- [ ] 实现子向量分割
- [ ] 每个子空间独立 K-means
- [ ] 实现 PQ 编码 (uint8)
- [ ] 实现 PQ 解码 (近似向量)
- [ ] 实现非对称距离计算 (ADC)

**代码框架**:
```rust
pub struct ProductQuantizer {
    m: usize,              // 子空间数量
    nbits: usize,          // 每个子空间比特数
    ksub: usize,           // 每个子空间中心数 = 2^nbits
    dim: usize,            // 原始维度
    dsub: usize,           // 子空间维度 = dim / m
    centroids: Vec<f32>,   // [m * ksub * dsub]
}

impl ProductQuantizer {
    /// 训练: 对每个子空间独立 k-means
    pub fn train(&mut self, data: &[f32]) {
        for m in 0..self.m {
            let sub_vectors = self.extract_subspace(data, m);
            let mut kmeans = KMeans::new(self.ksub, self.dsub);
            kmeans.train(&sub_vectors);
            // 存储到 centroids[m * ksub * dsub ..]
        }
    }

    /// 编码: 向量 -> uint8 codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|m| {
                let sub = &vector[m * self.dsub..(m+1) * self.dsub];
                self.find_nearest_centroid(m, sub)
            })
            .collect()
    }

    /// 非对称距离计算 (query vs PQ codes)
    pub fn adc_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // 预计算查询到所有中心的距离表
        let table = self.compute_distance_table(query);
        // 查表求和
        codes.iter().enumerate()
            .map(|(m, &code)| table[m * self.ksub + code as usize])
            .sum()
    }
}
```

### 3.2 IVF-PQ 索引重构

**文件**: `src/faiss/ivfpq_standard.rs`

**任务**:
- [ ] 使用标准 PQ 编码替代残差存储
- [ ] 实现距离表预计算
- [ ] 支持批量 ADC 查询
- [ ] 添加 nprobe 自适应

---

## 阶段 4: IVF-SQ8 量化 (✅ 已完成)

### 4.1 标量量化实现

**文件**: `src/quantization/sq.rs`

```
状态: ✅ 已完成
提交: c37c9e2
```

**已实现**:
- [x] 实现 8-bit 均匀量化
- [x] 实现 4-bit 量化 (SQ4)
- [x] 计算全局 min/max
- [x] 编码/解码函数
- [x] 量化误差计算

```rust
pub struct ScalarQuantizer {
    mins: Vec<f32>,   // 每维最小值
    maxs: Vec<f32>,   // 每维最大值
    dim: usize,
}

impl ScalarQuantizer {
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        for d in 0..self.dim {
            self.mins[d] = f32::MAX;
            self.maxs[d] = f32::MIN;
            for i in 0..n {
                let v = data[i * self.dim + d];
                self.mins[d] = self.mins[d].min(v);
                self.maxs[d] = self.maxs[d].max(v);
            }
        }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().enumerate()
            .map(|(d, &v)| {
                let scale = (self.maxs[d] - self.mins[d]) / 255.0;
                ((v - self.mins[d]) / scale) as u8
            })
            .collect()
    }
}
```

---

## 阶段 5: Range 搜索 (✅ 部分完成)

### 5.1 Range 搜索接口

**文件**: `src/api/search.rs`

```
状态: ✅ API 已完成，索引实现进行中
提交: c37c9e2
```

**已实现**:
- [x] 定义 RangeSearchRequest
- [x] RangePredicate 结构

**待完成**:
- [ ] 实现 Flat 索引 range 搜索
- [ ] 实现 HNSW range 搜索
- [ ] 实现 IVF range 搜索

```rust
pub struct RangeSearchRequest {
    pub radius: f32,
    pub min_score: Option<f32>,
    pub filter: Option<Arc<BitsetView>>,
}

pub struct RangeSearchResult {
    pub ids: Vec<i64>,
    pub distances: Vec<f32>,
    pub counts: Vec<usize>,  // 每个查询的结果数
}

// Index trait 扩展
pub trait Index {
    fn range_search(&self, query: &[f32], req: &RangeSearchRequest)
        -> Result<RangeSearchResult>;
}
```

---

## 阶段 6: FFI/JNI 层完善 (P1)

### 6.1 C FFI 完善

**文件**: `src/ffi.rs`

```
优先级: P1
工作量: 3-4 天
依赖: 索引实现
```

**当前问题**: 大部分函数是空实现

**任务**:
- [ ] 实现 `knowhere_index_create`
- [ ] 实现 `knowhere_index_train`
- [ ] 实现 `knowhere_index_add`
- [ ] 实现 `knowhere_index_search`
- [ ] 实现 `knowhere_index_save/load`
- [ ] 完善 bitset 操作

### 6.2 JNI 绑定

**文件**: `src/jni/`

```
优先级: P1
工作量: 5-7 天
依赖: C FFI
```

**任务**:
- [ ] 添加 jni crate 依赖
- [ ] 实现 Java 包装类
- [ ] 实现 KnowhereIndex JNI 接口
- [ ] 内存管理 (跨语言对象生命周期)
- [ ] 单元测试

```rust
// src/jni/mod.rs
use jni::JNIEnv;
use jni::objects::{JClass, JObject, JLongArray, JFloatArray};

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_create(
    mut env: JNIEnv,
    _class: JClass,
    index_type: jint,
    dim: jint,
    metric_type: jint,
) -> jlong {
    // 创建索引并返回指针
}

#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereIndex_search(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    queries: JFloatArray,
    k: jint,
) -> JObject {
    // 执行搜索并返回 Java 对象
}
```

---

## 阶段 7: HNSW 增强 (P1)

### 7.1 多层图结构

**文件**: `src/faiss/hnsw.rs`

```
优先级: P1
工作量: 3-4 天
依赖: SIMD
```

**当前问题**: 单层简化实现

**任务**:
- [ ] 实现多层索引结构
- [ ] 实现 level 分配 (指数分布)
- [ ] 实现逐层贪婪搜索
- [ ] 动态邻居选择 (heuristic select)

### 7.2 动态删除支持

**任务**:
- [ ] 软删除标记
- [ ] 延迟图修复
- [ ] 空间回收

---

## 阶段 8: 二值索引 (P2)

### 8.1 真正的二值向量

**文件**: `src/faiss/binary_index.rs`

```
优先级: P2
工作量: 2-3 天
```

**任务**:
- [ ] 位向量存储 (每维 1 bit)
- [ ] 真正的 Hamming 距离
- [ ] 真正的 Jaccard 距离
- [ ] 二值索引 (IndexBinaryFlat, IndexBinaryHNSW)

---

## 阶段 9: 稀疏向量 (P2)

### 9.1 稀疏向量支持

**文件**: `src/sparse/`

```
优先级: P2
工作量: 5-7 天
```

**任务**:
- [ ] 稀疏向量数据结构
- [ ] 稀疏内积计算
- [ ] 稀疏索引 (ScaNN-inspired)

---

## 阶段 10: ANNOY 索引 (P2)

### 10.1 ANNOY 实现

**文件**: `src/faiss/annoy.rs`

```
优先级: P2
工作量: 3-4 天
```

**任务**:
- [ ] 随机投影树构建
- [ ] 多树搜索
- [ ] 优先队列搜索

---

## 性能目标

### 距离计算 (vs C++ Knowhere)

| 操作 | 当前 (Scalar) | 目标 (SIMD) | C++ Knowhere |
|-----|--------------|-------------|--------------|
| L2 (128-dim) | 120ns | 25ns | 20ns |
| L2 (960-dim) | 900ns | 180ns | 150ns |
| IP (128-dim) | 100ns | 20ns | 18ns |
| Batch L2 (1K x 1K, 128-dim) | 120ms | 25ms | 20ms |

### 索引性能

| 索引 | 当前 QPS | 目标 QPS | C++ Knowhere |
|-----|---------|----------|--------------|
| Flat (1M, 128-dim) | 500 | 2000 | 2500 |
| HNSW (1M, 128-dim) | 5000 | 15000 | 18000 |
| IVF-PQ (1M, 128-dim) | 2000 | 8000 | 10000 |

---

## 测试计划

### 单元测试
- 每个 SIMD 函数有独立测试
- 量化编码/解码正确性测试
- 索引构建/搜索正确性测试

### 集成测试
- SIFT-1M 基准测试
- GloVe-1.2M 基准测试
- 与 C++ Knowhere recall 对比

### 性能测试
- CI 中添加性能回归检测
- 定期运行完整基准测试

---

## 文件变更清单

### 新增文件
```
src/simd/
├── mod.rs
├── dispatch.rs
├── x86.rs          # SSE/AVX
├── neon.rs         # ARM NEON
└── scalar.rs       # 标量后端

src/quantization/
├── mod.rs
├── kmeans.rs       # 已存在，优化
├── pq.rs           # 标准 PQ
├── sq.rs           # 标量量化
└── rabitq.rs       # RaBitQ (P3)

src/faiss/
├── diskann_vamana.rs  # 真正的 Vamana
├── ivfpq_standard.rs  # 标准 IVF-PQ
├── binary_index.rs    # 二值索引
└── annoy.rs           # ANNOY

src/jni/
├── mod.rs
├── index.rs
└── bitset.rs

src/api/
├── range_search.rs    # Range 搜索
└── sparse.rs          # 稀疏向量
```

### 修改文件
```
src/simd.rs           -> 重构为模块
src/faiss/diskann.rs  -> 使用 Vamana
src/faiss/ivfpq.rs    -> 使用标准 PQ
src/faiss/hnsw.rs     -> 多层结构
src/ffi.rs            -> 完整实现
```

---

## 时间线 (更新 2026-02-24)

| 阶段 | 内容 | 状态 |
|-----|------|------|
| 0 | 基础设施 | ✅ 完成 |
| 1 | SIMD | ✅ 完成 |
| 2 | DiskANN | 🔄 进行中 |
| 3 | IVF-PQ | ✅ 完成 |
| 4 | IVF-SQ8 | ✅ 完成 |
| 5 | Range 搜索 | 🔄 API 完成，索引实现中 |
| 6 | FFI/JNI | ⏳ 待开始 |
| 7 | HNSW 增强 | ⏳ 待开始 |
| 8-10 | P2 功能 | ⏳ 待开始 |

**当前进度**: 约 50% P0-P1 功能已完成

---

## 风险与依赖

1. **SIMD 跨平台**: 需要测试 x86 + ARM
2. **内存安全**: FFI/JNI 需要仔细处理生命周期
3. **C++ 兼容性**: 序列化格式可能无法完全兼容
4. **性能验证**: 需要建立完整的 benchmark 套件

---

## 开始命令

```bash
# 创建分支
git checkout -b feature/simd-optimization

# 添加 feature flags
# Cargo.toml 添加:
# [features]
# default = []
# simd = []
# sse = ["simd"]
# avx2 = ["simd"]
# neon = ["simd"]

# 运行测试
cargo test

# 运行 benchmark
cargo bench
```

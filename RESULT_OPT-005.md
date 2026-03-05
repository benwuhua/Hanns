# OPT-005: Mini-Batch K-Means 优化实现

## 任务概述

实现 mini-batch k-means 算法，支持大数据集增量训练，减少内存占用和训练时间。

## 实现细节

### 1. 新增模块结构

```
src/clustering/
├── mod.rs                    # 模块导出
└── mini_batch_kmeans.rs      # Mini-Batch K-Means 核心实现
```

### 2. 核心算法实现

**文件**: `src/clustering/mini_batch_kmeans.rs`

#### MiniBatchKMeansConfig

```rust
pub struct MiniBatchKMeansConfig {
    pub batch_size: usize,        // 默认：10,000
    pub max_iterations: usize,    // 默认：100
    pub tolerance: f32,           // 默认：1e-4
    pub seed: Option<u64>,        // 随机种子
    pub random_seed: Option<u64>, // 别名 (兼容)
}
```

#### MiniBatchKMeans 主要方法

- `new(k, dim)` - 创建默认配置的聚类器
- `with_config(k, dim, config)` - 创建自定义配置的聚类器
- `train(vectors)` - 训练模型，返回处理的样本数
- `find_nearest_centroid(vector)` - 查找最近的 centroid
- `find_nearest_batch(vectors)` - 批量查找 (支持并行)
- `centroids()` - 获取训练后的 centroids

#### 算法特点

1. **K-Means++ 初始化**: 使用概率选择初始 centroid，提高收敛质量
2. **增量更新**: 使用在线学习公式更新 centroid:
   ```
   new_centroid = old_centroid + (vector - old_centroid) / count
   ```
3. **Mini-Batch 采样**: 每次迭代只处理一个 batch 的数据，降低内存占用
4. **并行支持**: 通过 `#[cfg(feature = "parallel")]` 支持 rayon 并行加速

### 3. IVF-Flat 集成

**文件**: `src/faiss/ivf_flat.rs`

#### 新增方法

```rust
/// 使用 Mini-Batch K-Means 训练
pub fn train_mini_batch(
    &mut self, 
    vectors: &[f32], 
    config: Option<MiniBatchKMeansConfig>
) -> Result<usize>
```

#### 配置驱动的训练

在 `train_ivf()` 方法中自动检测配置:

```rust
let use_mini_batch = self.config.params.use_mini_batch.unwrap_or(false);

if use_mini_batch {
    // 使用 Mini-Batch K-Means
    let mb_config = MiniBatchKMeansConfig {
        batch_size: self.config.params.mini_batch_size.unwrap_or(10_000),
        max_iterations: self.config.params.max_iterations.unwrap_or(100),
        tolerance: self.config.params.kmeans_tolerance.unwrap_or(1e-4),
        seed: None,
        random_seed: None,
    };
    // ...
} else {
    // 使用标准 K-Means
    // ...
}
```

#### IndexParams 扩展

在 `src/api/index.rs` 中已存在的配置项:

```rust
pub struct IndexParams {
    // ...
    pub use_mini_batch: Option<bool>,        // 启用 mini-batch k-means
    pub mini_batch_size: Option<usize>,      // batch 大小
    pub max_iterations: Option<usize>,       // 最大迭代次数
    pub kmeans_tolerance: Option<f32>,       // 收敛容忍度
}
```

便捷构造函数:

```rust
IndexParams::ivf_mini_batch(nlist, nprobe, batch_size, max_iter, tol)
```

### 4. 单元测试

#### Mini-Batch K-Means 测试 (7 个)

1. `test_mini_batch_kmeans_basic` - 基础功能测试
2. `test_mini_batch_kmeans_convergence` - 收敛性测试
3. `test_mini_batch_kmeans_config` - 配置测试
4. `test_mini_batch_kmeans_large_dataset` - 大数据集测试 (100K 样本)
5. `test_find_nearest_batch` - 批量查找测试
6. `test_empty_data` - 空数据边界测试
7. `test_insufficient_data` - 数据不足边界测试

#### IVF-Flat 集成测试 (2 个)

1. `test_ivf_flat_mini_batch_train` - 小数据集集成测试
2. `test_ivf_flat_mini_batch_large_dataset` - 大数据集集成测试 (1000 样本，10 聚类)

### 5. 使用示例

```rust
use knowhere_rs::clustering::{MiniBatchKMeans, MiniBatchKMeansConfig};
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, IndexParams};
use knowhere_rs::faiss::IvfFlatIndex;

// 方法 1: 直接使用 MiniBatchKMeans
let config = MiniBatchKMeansConfig::default()
    .with_batch_size(5000)
    .with_max_iterations(50)
    .with_tolerance(1e-3);

let mut mbkm = MiniBatchKMeans::with_config(100, 128, config);
mbkm.train(&training_data);
let centroids = mbkm.centroids();

// 方法 2: 通过 IVF-Flat 索引配置
let config = IndexConfig {
    index_type: IndexType::IvfFlat,
    metric_type: MetricType::L2,
    dim: 128,
    params: IndexParams::ivf_mini_batch(100, 10, 10000, 100, 1e-4),
};

let mut index = IvfFlatIndex::new(&config).unwrap();
index.train(&training_data).unwrap();  // 自动使用 mini-batch k-means
```

## 性能优势

### 内存优化

- **标准 K-Means**: 需要存储所有样本的分配结果 (O(n) 内存)
- **Mini-Batch K-Means**: 只需存储一个 batch 的分配结果 (O(batch_size) 内存)

### 训练速度

- **标准 K-Means**: 每次迭代处理全部 n 个样本
- **Mini-Batch K-Means**: 每次迭代只处理 batch_size 个样本

对于大数据集 (n >> batch_size), mini-batch 方法显著更快。

### 适用场景

| 场景 | 推荐方法 |
|------|----------|
| 小数据集 (< 10K 样本) | 标准 K-Means |
| 中等数据集 (10K - 100K) | Mini-Batch K-Means |
| 大数据集 (> 100K) | Mini-Batch K-Means |
| 内存受限环境 | Mini-Batch K-Means |
| 需要精确收敛 | 标准 K-Means |
| 快速原型/探索 | Mini-Batch K-Means |

## 编译与测试

### 编译检查

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo check
# 结果：Finished `dev` profile [unoptimized + debuginfo] target(s)
```

### 单元测试

```bash
# 测试 mini-batch k-means 模块
cargo test --lib clustering

# 测试 IVF-Flat 集成
cargo test --lib ivf_flat

# 测试结果：
# - clustering: 7 passed
# - ivf_flat: 19 passed (包括 2 个新的 mini-batch 测试)
```

## 参考实现

本实现参考了 Microsoft Research 的论文:

> "Web Scale Clustering using Mini-Batch K-Means" (2010)
> - 核心思想：使用小批量增量更新 centroid
> - 优势：适合大规模数据集，支持在线学习

## 后续优化建议

1. **自适应 batch size**: 根据数据集大小动态调整 batch_size
2. **学习率衰减**: 随着迭代进行降低学习率，提高收敛质量
3. **多轮次训练**: 支持多个 epoch 的训练
4. **分布式训练**: 支持多节点并行训练
5. **GPU 加速**: 使用 GPU 进行距离计算和 centroid 更新

## 完成状态

✅ 1. 实现 mini-batch k-means 算法 (`src/clustering/mini_batch_kmeans.rs`)  
✅ 2. 支持大数据集增量训练  
✅ 3. 配置参数：batch_size (10000), max_iterations (100), tolerance (1e-4)  
✅ 4. 集成到 IVF-Flat 索引的 train() 方法  
✅ 5. 添加单元测试 (7 个 clustering 测试 + 2 个 ivf_flat 集成测试)  
✅ 6. cargo check 编译通过  
✅ 7. cargo test 测试通过  
✅ 8. 文档记录完成  

---

**实现日期**: 2026-03-01  
**实现者**: builder agent  
**任务编号**: OPT-005

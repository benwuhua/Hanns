//! Mini-Batch K-Means Clustering
//!
//! 基于 mini-batch 的 k-means 实现，适用于大数据集的增量训练
//! 特点：
//! - 内存占用低：每次只处理一个 batch
//! - 训练速度快：无需全量数据迭代
//! - 支持在线学习：可增量更新 centroid
//!
//! 算法参考：
//! "Web Scale Clustering using Mini-Batch K-Means" (Microsoft, 2010)

use crate::simd::{l2_distance, l2_distance_sq};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Mini-Batch K-Means 配置
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansConfig {
    /// 每个 batch 的样本数
    pub batch_size: usize,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 收敛容忍度 (centroid 移动距离)
    pub tolerance: f32,
    /// 随机数种子 (None 表示随机)
    pub seed: Option<u64>,
    /// 随机数种子 (别名，用于兼容)
    pub random_seed: Option<u64>,
}

impl Default for MiniBatchKMeansConfig {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            max_iterations: 100,
            tolerance: 1e-4,
            seed: None,
            random_seed: None,
        }
    }
}

impl MiniBatchKMeansConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Mini-Batch K-Means 聚类器
pub struct MiniBatchKMeans {
    k: usize,
    dim: usize,
    config: MiniBatchKMeansConfig,
    centroids: Vec<f32>,
    /// 每个 centroid 的累计计数 (用于增量更新)
    counts: Vec<usize>,
    rng: StdRng,
}

impl MiniBatchKMeans {
    /// 创建新的 Mini-Batch K-Means 聚类器
    pub fn new(k: usize, dim: usize) -> Self {
        Self::with_config(k, dim, MiniBatchKMeansConfig::default())
    }

    /// 使用自定义配置创建聚类器
    pub fn with_config(k: usize, dim: usize, config: MiniBatchKMeansConfig) -> Self {
        // 支持 seed 和 random_seed 两种字段
        let seed = config.seed.or(config.random_seed);
        let rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_entropy()
        };

        Self {
            k,
            dim,
            config,
            centroids: vec![0.0f32; k * dim],
            counts: vec![0usize; k],
            rng,
        }
    }

    /// 训练模型
    ///
    /// # Arguments
    /// * `vectors` - 训练数据，按行优先存储 (n * dim)
    ///
    /// # Returns
    /// 实际处理的样本数
    pub fn train(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        if n == 0 || n < self.k {
            return 0;
        }

        // 初始化 centroid
        self.init_centroids(vectors, n);

        // Mini-batch 迭代
        let mut max_shift = f32::MAX;
        let mut iteration = 0;

        while iteration < self.config.max_iterations && max_shift >= self.config.tolerance {
            max_shift = self.process_batch(vectors, n);
            iteration += 1;
        }

        n
    }

    /// 使用 k-means++ 初始化 centroid
    fn init_centroids(&mut self, vectors: &[f32], n: usize) {
        // 随机选择第一个 centroid
        let first_idx = self.rng.gen_range(0..n);
        for j in 0..self.dim {
            self.centroids[j] = vectors[first_idx * self.dim + j];
        }
        self.counts[0] = 1;

        // k-means++ 初始化剩余 centroid
        for c in 1..self.k {
            // 计算每个点到最近已选 centroid 的距离
            let mut distances = vec![0.0f32; n];
            let mut total_dist = 0.0f32;

            for i in 0..n {
                let mut min_dist = f32::MAX;
                for selected_c in 0..c {
                    let dist = l2_distance_sq(
                        &vectors[i * self.dim..(i + 1) * self.dim],
                        &self.centroids[selected_c * self.dim..(selected_c + 1) * self.dim],
                    );
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist;
                total_dist += min_dist;
            }

            // 按距离比例选择下一个 centroid
            if total_dist > 0.0 {
                let mut choice = self.rng.gen::<f32>() * total_dist;
                let mut chosen_idx = 0;
                for (i, &dist) in distances.iter().enumerate().take(n) {
                    choice -= dist;
                    if choice <= 0.0 {
                        chosen_idx = i;
                        break;
                    }
                }

                // 复制选中的向量到 centroid
                for j in 0..self.dim {
                    self.centroids[c * self.dim + j] = vectors[chosen_idx * self.dim + j];
                }
                self.counts[c] = 1;
            } else {
                // 如果所有距离都是 0，随机选择一个
                let idx = self.rng.gen_range(0..n);
                for j in 0..self.dim {
                    self.centroids[c * self.dim + j] = vectors[idx * self.dim + j];
                }
                self.counts[c] = 1;
            }
        }
    }

    /// 处理一个 mini-batch
    ///
    /// # Returns
    /// centroid 的最大移动距离
    fn process_batch(&mut self, vectors: &[f32], n: usize) -> f32 {
        let batch_size = self.config.batch_size.min(n);

        // 随机采样一个 batch
        let batch_indices: Vec<usize> = if n <= batch_size {
            (0..n).collect()
        } else {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut self.rng);
            indices.truncate(batch_size);
            indices
        };

        // 为 batch 中的每个点分配最近的 centroid
        let assignments: Vec<usize> = batch_indices
            .iter()
            .map(|&i| {
                let vec = &vectors[i * self.dim..(i + 1) * self.dim];
                self.find_nearest_centroid(vec)
            })
            .collect();

        // 增量更新 centroid
        let mut max_shift = 0.0f32;
        let learning_rate = 1.0; // 学习率

        for (batch_idx, &centroid_id) in assignments.iter().enumerate() {
            let vector_idx = batch_indices[batch_idx];
            let vector = &vectors[vector_idx * self.dim..];

            // 更新计数
            self.counts[centroid_id] += 1;
            let count = self.counts[centroid_id] as f32;

            // 增量更新 centroid: new_centroid = old_centroid + (vector - old_centroid) / count
            for (j, &value) in vector.iter().enumerate().take(self.dim) {
                let old_val = self.centroids[centroid_id * self.dim + j];
                let new_val = old_val + (value - old_val) / count * learning_rate;
                let shift = (new_val - old_val).abs();
                if shift > max_shift {
                    max_shift = shift;
                }
                self.centroids[centroid_id * self.dim + j] = new_val;
            }
        }

        max_shift
    }

    /// 查找最近的 centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;

        for c in 0..self.k {
            let dist = l2_distance_sq(vector, &self.centroids[c * self.dim..(c + 1) * self.dim]);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }

        best
    }

    /// 批量查找最近 centroid (并行版本)
    #[cfg(feature = "parallel")]
    pub fn find_nearest_batch(&self, vectors: &[f32]) -> Vec<usize> {
        let n = vectors.len() / self.dim;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &vectors[i * self.dim..];
                self.find_nearest_centroid(vec)
            })
            .collect()
    }

    /// 批量查找最近 centroid (串行版本)
    pub fn find_nearest_batch_serial(&self, vectors: &[f32]) -> Vec<usize> {
        let n = vectors.len() / self.dim;
        (0..n)
            .map(|i| {
                let vec = &vectors[i * self.dim..];
                self.find_nearest_centroid(vec)
            })
            .collect()
    }

    /// 获取 centroids
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// 获取聚类数
    pub fn k(&self) -> usize {
        self.k
    }

    /// 获取维度
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// 获取配置
    pub fn config(&self) -> &MiniBatchKMeansConfig {
        &self.config
    }

    /// 计算两个向量的 L2 距离 (用于测试)
    pub fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_distance(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_batch_kmeans_basic() {
        let mut mbkm = MiniBatchKMeans::new(4, 128);
        let mut data = vec![0.0f32; 512];

        // 创建 4 个明显的聚类
        for i in 0..4 {
            for j in 0..128 {
                data[i * 128 + j] = (i as f32) * 100.0 + (j as f32 * 0.1);
            }
        }

        let n = mbkm.train(&data);
        assert_eq!(n, 4);
        assert_eq!(mbkm.centroids().len(), 4 * 128);
    }

    #[test]
    fn test_mini_batch_kmeans_convergence() {
        let mut mbkm = MiniBatchKMeans::new(2, 2);
        // Two clusters, 6 vectors total
        let data = vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, // Cluster 1: 3 vectors
            10.0, 10.0, 10.1, 10.1, 10.2, 10.0, // Cluster 2: 3 vectors
        ];

        let n = mbkm.train(&data);
        assert_eq!(n, 6);

        // Check centroids are far apart
        let dist = mbkm.l2_distance(&mbkm.centroids()[0..2], &mbkm.centroids()[2..4]);
        assert!(dist > 5.0, "Centroids should be separated");
    }

    #[test]
    fn test_mini_batch_kmeans_config() {
        let config = MiniBatchKMeansConfig::default()
            .with_batch_size(5000)
            .with_max_iterations(50)
            .with_tolerance(1e-3)
            .with_seed(42);

        let mbkm = MiniBatchKMeans::with_config(10, 64, config);

        assert_eq!(mbkm.config().batch_size, 5000);
        assert_eq!(mbkm.config().max_iterations, 50);
        assert!((mbkm.config().tolerance - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_mini_batch_kmeans_large_dataset() {
        // 测试大数据集
        let dim = 128;
        let n_clusters = 10;
        let n_samples = 100_000;

        let mut mbkm = MiniBatchKMeans::with_config(
            n_clusters,
            dim,
            MiniBatchKMeansConfig::default().with_batch_size(1000),
        );

        // 生成合成数据
        let mut data = vec![0.0f32; n_samples * dim];
        for i in 0..n_samples {
            let cluster_id = i % n_clusters;
            for j in 0..dim {
                data[i * dim + j] = (cluster_id as f32) * 10.0 + (j as f32 * 0.01);
            }
        }

        let n = mbkm.train(&data);
        assert_eq!(n, n_samples);
        assert_eq!(mbkm.centroids().len(), n_clusters * dim);
    }

    #[test]
    fn test_find_nearest_batch() {
        let mut mbkm = MiniBatchKMeans::new(3, 4);

        // 训练简单数据
        let data = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        mbkm.train(&data);

        // 测试批量查找
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let assignments = mbkm.find_nearest_batch_serial(&query);
        assert_eq!(assignments.len(), 1);
    }

    #[test]
    fn test_empty_data() {
        let mut mbkm = MiniBatchKMeans::new(5, 128);
        let data: Vec<f32> = vec![];

        let n = mbkm.train(&data);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_insufficient_data() {
        let mut mbkm = MiniBatchKMeans::new(10, 4); // k=10
        let data = vec![1.0, 2.0, 3.0, 4.0]; // only 1 sample

        let n = mbkm.train(&data);
        assert_eq!(n, 0); // Should return 0 when n < k
    }
}

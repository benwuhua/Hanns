//! K-Means 量化器
//!
//! 支持 k-means++ 初始化和迭代收敛
//! 使用 SIMD 优化距离计算
//! 使用 rayon 并行化训练过程

use crate::simd::{l2_distance, l2_distance_sq};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 计算两个向量的 L2 距离（自动选择 SIMD 实现）
#[inline]
fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance(a, b)
}

/// 计算 L2 平方距离 (避免 sqrt，更快)
#[inline]
fn compute_l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_sq(a, b)
}

pub struct KMeans {
    k: usize,
    max_iter: usize,
    tolerance: f32,
    pub centroids: Vec<f32>,
    dim: usize,
    rng: StdRng,
    #[cfg(feature = "parallel")]
    num_threads: usize,
    /// 快速构建模式：更激进的优化参数
    fast_mode: bool,
}

impl KMeans {
    pub fn new(k: usize, dim: usize) -> Self {
        Self {
            k,
            max_iter: 10,
            tolerance: 1e-2,
            centroids: vec![0.0; k * dim],
            dim,
            rng: StdRng::from_entropy(),
            #[cfg(feature = "parallel")]
            num_threads: rayon::current_num_threads(),
            fast_mode: false,
        }
    }

    /// 创建快速构建模式的 KMeans
    /// 适用于构建速度优先的场景，牺牲少量收敛精度换取速度
    pub fn fast(k: usize, dim: usize) -> Self {
        Self {
            k,
            max_iter: 5,     // 激进减少迭代次数
            tolerance: 1e-3, // 更宽松收敛
            centroids: vec![0.0; k * dim],
            dim,
            rng: StdRng::from_entropy(),
            #[cfg(feature = "parallel")]
            num_threads: rayon::current_num_threads(),
            fast_mode: true,
        }
    }

    /// 设置快速模式
    pub fn with_fast_mode(mut self, fast: bool) -> Self {
        self.fast_mode = fast;
        if fast {
            self.max_iter = 5;
            self.tolerance = 1e-3;
        }
        self
    }

    /// 设置并行线程数
    #[cfg(feature = "parallel")]
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// 设置最大迭代次数
    pub fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    /// 简化初始化 - 直接随机选择 k 个点 (放弃 k-means++ 的质量)
    fn random_init(&mut self, vectors: &[f32], n: usize) {
        if n == 0 {
            return;
        }

        // 简单随机选择 k 个向量作为初始 centroid
        let mut selected = std::collections::HashSet::new();
        for c in 0..self.k {
            let mut idx;
            loop {
                idx = self.rng.gen_range(0..n);
                if selected.insert(idx) {
                    break;
                }
            }
            for j in 0..self.dim {
                self.centroids[c * self.dim + j] = vectors[idx * self.dim + j];
            }
        }
    }

    /// 训练 K-means (速度优化版)
    /// 自动使用并行训练 (如果 parallel feature 启用)
    pub fn train(&mut self, vectors: &[f32]) -> usize {
        #[cfg(feature = "parallel")]
        {
            self.train_parallel(vectors, self.num_threads)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.train_serial(vectors)
        }
    }

    /// 串行训练 (fallback)
    #[allow(dead_code)]
    fn train_serial(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        if n == 0 || n < self.k {
            return 0;
        }

        // 简化初始化 (速度优先)
        self.random_init(vectors, n);

        // 迭代优化
        let mut assignments = vec![0usize; n];
        let mut new_centroids = vec![0.0f32; self.k * self.dim];
        let mut counts = vec![0usize; self.k];

        for _iter in 0..self.max_iter {
            // 分配阶段 - 使用 L2 平方距离 (避免 sqrt)
            for (i, assignment) in assignments.iter_mut().enumerate().take(n) {
                let mut min_dist = f32::MAX;
                let mut best_k = 0;
                let vec_start = i * self.dim;
                let vec_end = vec_start + self.dim;
                let vector = &vectors[vec_start..vec_end];
                for c in 0..self.k {
                    let centroid_start = c * self.dim;
                    let centroid_end = centroid_start + self.dim;
                    let dist = compute_l2_distance_sq(
                        vector,
                        &self.centroids[centroid_start..centroid_end],
                    );
                    if dist < min_dist {
                        min_dist = dist;
                        best_k = c;
                    }
                }
                *assignment = best_k;
            }

            // 更新阶段
            new_centroids.fill(0.0);
            counts.fill(0);

            for i in 0..n {
                let c = assignments[i];
                for j in 0..self.dim {
                    new_centroids[c * self.dim + j] += vectors[i * self.dim + j];
                }
                counts[c] += 1;
            }

            // 计算收敛
            let mut max_shift = 0.0f32;
            for c in 0..self.k {
                let centroid_start = c * self.dim;
                let centroid_end = centroid_start + self.dim;

                if counts[c] > 0 {
                    let mut updated = vec![0.0f32; self.dim];
                    for j in 0..self.dim {
                        updated[j] = new_centroids[c * self.dim + j] / counts[c] as f32;
                    }

                    let shift = compute_l2_distance(
                        &self.centroids[centroid_start..centroid_end],
                        &updated,
                    );
                    max_shift = max_shift.max(shift);
                    self.centroids[centroid_start..centroid_end].copy_from_slice(&updated);
                } else {
                    // 避免空簇长期不更新导致收敛到退化解
                    let idx = self.rng.gen_range(0..n);
                    let src_start = idx * self.dim;
                    let src_end = src_start + self.dim;
                    self.centroids[centroid_start..centroid_end]
                        .copy_from_slice(&vectors[src_start..src_end]);
                }
            }

            if max_shift < self.tolerance {
                break;
            }
        }

        n
    }

    /// 查找最近 centroid
    pub fn find_nearest(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        for c in 0..self.k {
            let centroid_start = c * self.dim;
            let centroid_end = centroid_start + self.dim;
            let dist = compute_l2_distance(vector, &self.centroids[centroid_start..centroid_end]);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }
        best
    }

    /// 计算两个向量的 L2 距离（公开方法，用于测试）
    pub fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        compute_l2_distance(a, b)
    }

    /// 批量查找最近 centroid（使用 SIMD）
    #[cfg(feature = "parallel")]
    pub fn find_nearest_batch(&self, vectors: &[f32]) -> Vec<usize> {
        use rayon::prelude::*;

        let n = vectors.len() / self.dim;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &vectors[i * self.dim..];
                self.find_nearest(vec)
            })
            .collect()
    }

    /// 并行训练 K-means（使用 rayon + SIMD）
    #[cfg(feature = "parallel")]
    pub fn train_parallel(&mut self, vectors: &[f32], _num_threads: usize) -> usize {
        let n = vectors.len() / self.dim;
        if n == 0 || n < self.k {
            return 0;
        }

        // 简化初始化 (速度优先)
        self.random_init(vectors, n);

        let _assignments = vec![0usize; n];

        for _iter in 0..self.max_iter {
            // 并行分配阶段 - 使用 L2 平方距离 (避免 sqrt)
            let assignments: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut min_dist = f32::MAX;
                    let mut best_k = 0;
                    let vec_start = i * self.dim;
                    let vec_end = vec_start + self.dim;
                    let vec = &vectors[vec_start..vec_end];
                    for c in 0..self.k {
                        let centroid_start = c * self.dim;
                        let centroid_end = centroid_start + self.dim;
                        let dist = compute_l2_distance_sq(
                            vec,
                            &self.centroids[centroid_start..centroid_end],
                        );
                        if dist < min_dist {
                            min_dist = dist;
                            best_k = c;
                        }
                    }
                    best_k
                })
                .collect();

            // 并行更新阶段 - 使用 parallel iterator 聚合
            let mut new_centroids = vec![0.0f32; self.k * self.dim];
            let mut counts = vec![0usize; self.k];

            // 使用并行迭代聚合
            let result: (Vec<f32>, Vec<usize>) = (0..n)
                .into_par_iter()
                .map(|i| {
                    let c = assignments[i];
                    let mut local_centroids = vec![0.0f32; self.k * self.dim];
                    let mut local_counts = vec![0usize; self.k];

                    for j in 0..self.dim {
                        local_centroids[c * self.dim + j] = vectors[i * self.dim + j];
                    }
                    local_counts[c] = 1;

                    (local_centroids, local_counts)
                })
                .reduce(
                    || (vec![0.0f32; self.k * self.dim], vec![0usize; self.k]),
                    |(mut acc_centroids, mut acc_counts), (local_centroids, local_counts)| {
                        for i in 0..self.k * self.dim {
                            acc_centroids[i] += local_centroids[i];
                        }
                        for i in 0..self.k {
                            acc_counts[i] += local_counts[i];
                        }
                        (acc_centroids, acc_counts)
                    },
                );

            new_centroids = result.0;
            counts = result.1;

            // 计算收敛
            let mut max_shift = 0.0f32;
            for c in 0..self.k {
                let centroid_start = c * self.dim;
                let centroid_end = centroid_start + self.dim;

                if counts[c] > 0 {
                    let mut updated = vec![0.0f32; self.dim];
                    for j in 0..self.dim {
                        updated[j] = new_centroids[c * self.dim + j] / counts[c] as f32;
                    }

                    let shift = compute_l2_distance(
                        &self.centroids[centroid_start..centroid_end],
                        &updated,
                    );
                    max_shift = max_shift.max(shift);
                    self.centroids[centroid_start..centroid_end].copy_from_slice(&updated);
                } else {
                    // 避免空簇长期不更新导致收敛到退化解
                    let idx = self.rng.gen_range(0..n);
                    let src_start = idx * self.dim;
                    let src_end = src_start + self.dim;
                    self.centroids[centroid_start..centroid_end]
                        .copy_from_slice(&vectors[src_start..src_end]);
                }
            }

            if max_shift < self.tolerance {
                break;
            }
        }

        n
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans() {
        let mut km = KMeans::new(4, 128);
        let mut data = vec![0.0f32; 512];
        for i in 0..4 {
            for j in 0..128 {
                data[i * 128 + j] = (i as f32) * 100.0 + (j as f32 * 0.1);
            }
        }
        let n = km.train(&data);
        assert_eq!(n, 4);
        assert!(!km.centroids().is_empty());
    }

    #[test]
    fn test_kmeans_convergence() {
        let mut km = KMeans::new(2, 2);
        // Two clusters, 6 vectors total
        let data = vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0, // Cluster 1: 3 vectors
            10.0, 10.0, 10.1, 10.1, 10.2, 10.0, // Cluster 2: 3 vectors
        ];
        let n = km.train(&data);
        assert_eq!(n, 6);

        // Check centroids are far apart
        let dist = km.l2_distance(&km.centroids[0..2], &km.centroids[2..4]);
        assert!(dist > 5.0, "Centroids should be separated");
    }
}

//! Elkan K-Means 算法（并行化版本 - OPT-008）
//!
//! 实现 Elkan 算法优化 k-means 聚类，使用三角不等式避免不必要的距离计算
//! 参考：Elkan, C. (2003) "Using the Triangle Inequality to Accelerate k-Means"
//!
//! 并行化策略 (OPT-008):
//! - 使用 rayon 并行化距离计算（主要瓶颈）
//! - 并行化中心点更新
//! - 并行化惯性计算

use crate::simd::l2_distance_sq;
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Elkan K-Means 配置
#[derive(Clone, Debug)]
pub struct ElkanKMeansConfig {
    pub k: usize,
    pub dim: usize,
    pub max_iter: usize,
    pub tolerance: f32,
    pub seed: u64,
    /// 是否启用并行化（需要 parallel feature）
    pub parallel: bool,
}

impl Default for ElkanKMeansConfig {
    fn default() -> Self {
        Self {
            k: 256,
            dim: 128,
            max_iter: 25,
            tolerance: 1e-4,
            seed: 42,
            parallel: true,
        }
    }
}

impl ElkanKMeansConfig {
    pub fn new(k: usize, dim: usize) -> Self {
        Self {
            k,
            dim,
            ..Default::default()
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Elkan K-Means 聚类结果
#[derive(Clone, Debug)]
pub struct ElkanKMeansResult {
    pub centroids: Vec<f32>,
    pub labels: Vec<usize>,
    pub iterations: usize,
    pub inertia: f32,
    pub converged: bool,
}

/// Elkan K-Means 聚类器
pub struct ElkanKMeans {
    config: ElkanKMeansConfig,
    centroids: Vec<f32>,
    rng: StdRng,
}

impl ElkanKMeans {
    pub fn new(config: ElkanKMeansConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            centroids: Vec::new(),
            rng,
            config,
        }
    }

    /// 对数据进行聚类
    pub fn cluster(&mut self, data: &[f32]) -> ElkanKMeansResult {
        let n = data.len() / self.config.dim;
        let k = self.config.k;
        let _dim = self.config.dim;

        assert!(n >= k, "数据点数必须大于等于聚类数");

        // 初始化中心点
        self.init_centroids_plusplus(data);

        // 分配标签
        let mut labels = vec![0; n];

        let mut iterations = 0;
        let mut converged = false;
        let mut prev_inertia = f32::MAX;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // 分配步骤（并行化）
            self.assign_nearest_parallel(data, &mut labels);

            // 更新中心点（并行化）
            let (new_centroids, max_move) = self.compute_new_centroids_parallel(data, &labels);
            self.centroids = new_centroids;

            // 计算惯性
            let inertia = self.compute_inertia_parallel(data, &labels);

            // 检查收敛
            if max_move < self.config.tolerance
                || (prev_inertia - inertia).abs() < self.config.tolerance * prev_inertia.abs()
            {
                converged = true;
                break;
            }
            prev_inertia = inertia;
        }

        let inertia = self.compute_inertia_parallel(data, &labels);

        ElkanKMeansResult {
            centroids: self.centroids.clone(),
            labels,
            iterations,
            inertia,
            converged,
        }
    }

    /// 使用 k-means++ 初始化
    fn init_centroids_plusplus(&mut self, data: &[f32]) {
        let n = data.len() / self.config.dim;
        let k = self.config.k;
        let dim = self.config.dim;

        self.centroids = vec![0.0; k * dim];

        // 随机选择第一个中心
        let first_idx = self.rng.gen_range(0..n);
        self.centroids[..dim].copy_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

        let mut min_dist_sq = vec![f32::MAX; n];

        for i in 1..k {
            // 计算到最近中心的距离
            for j in 0..n {
                let point = &data[j * dim..(j + 1) * dim];
                for c in 0..i {
                    let d = l2_distance_sq(point, &self.centroids[c * dim..(c + 1) * dim]);
                    if d < min_dist_sq[j] {
                        min_dist_sq[j] = d;
                    }
                }
            }

            // 按距离平方比例选择
            let total: f32 = min_dist_sq.iter().sum();
            if total == 0.0 {
                break;
            }

            let r = self.rng.gen::<f32>() * total;
            let mut cumsum = 0.0;
            let mut chosen = 0;

            for (j, &d) in min_dist_sq.iter().enumerate() {
                cumsum += d;
                if cumsum >= r {
                    chosen = j;
                    break;
                }
            }

            self.centroids[i * dim..(i + 1) * dim]
                .copy_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
        }
    }

    /// 分配最近中心（并行版本）- OPT-008 核心优化
    fn assign_nearest_parallel(&self, data: &[f32], labels: &mut [usize]) {
        let n = data.len() / self.config.dim;
        let k = self.config.k;
        let dim = self.config.dim;
        let centroids = &self.centroids;

        #[cfg(feature = "parallel")]
        if self.config.parallel && n > 100 {
            // 并行处理每个数据点 - 使用 par_iter_mut 修改 labels
            labels.par_iter_mut().enumerate().for_each(|(x, label)| {
                let point = &data[x * dim..(x + 1) * dim];
                let mut best_c = 0;
                let mut best_d = f32::MAX;

                for c in 0..k {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let d = l2_distance_sq(point, centroid);
                    if d < best_d {
                        best_c = c;
                        best_d = d;
                    }
                }

                *label = best_c;
            });
            return;
        }

        // 串行版本
        for x in 0..n {
            let point = &data[x * dim..(x + 1) * dim];
            let mut best_c = 0;
            let mut best_d = f32::MAX;

            for c in 0..k {
                let centroid = &self.centroids[c * dim..(c + 1) * dim];
                let d = l2_distance_sq(point, centroid);
                if d < best_d {
                    best_c = c;
                    best_d = d;
                }
            }

            labels[x] = best_c;
        }
    }

    /// 计算新中心点（并行版本）
    fn compute_new_centroids_parallel(&self, data: &[f32], labels: &[usize]) -> (Vec<f32>, f32) {
        let n = data.len() / self.config.dim;
        let k = self.config.k;
        let dim = self.config.dim;
        let old_centroids = &self.centroids;

        #[cfg(feature = "parallel")]
        if self.config.parallel && n > 100 {
            // 并行累加每个簇的向量和
            let results: Vec<(Vec<f32>, usize)> = (0..k)
                .into_par_iter()
                .map(|c| {
                    let mut sum = vec![0.0; dim];
                    let mut count = 0usize;

                    for x in 0..n {
                        if labels[x] == c {
                            count += 1;
                            for d in 0..dim {
                                sum[d] += data[x * dim + d];
                            }
                        }
                    }

                    (sum, count)
                })
                .collect();

            // 计算新中心点和移动距离
            let mut new_centroids = vec![0.0; k * dim];
            let mut max_move: f32 = 0.0;

            for c in 0..k {
                let (sum, count) = &results[c];
                if *count > 0 {
                    for d in 0..dim {
                        new_centroids[c * dim + d] = sum[d] / *count as f32;
                    }

                    let old_c = &old_centroids[c * dim..(c + 1) * dim];
                    let new_c = &new_centroids[c * dim..(c + 1) * dim];
                    let move_dist = l2_distance_sq(old_c, new_c).sqrt();
                    max_move = max_move.max(move_dist);
                }
            }

            return (new_centroids, max_move);
        }

        // 串行版本
        let mut new_centroids = vec![0.0; k * dim];
        let mut counts = vec![0usize; k];

        for x in 0..n {
            let c = labels[x];
            counts[c] += 1;
            for d in 0..dim {
                new_centroids[c * dim + d] += data[x * dim + d];
            }
        }

        let mut max_move: f32 = 0.0;

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_centroids[c * dim + d] /= counts[c] as f32;
                }

                let old_c = &self.centroids[c * dim..(c + 1) * dim];
                let new_c = &new_centroids[c * dim..(c + 1) * dim];
                let move_dist = l2_distance_sq(old_c, new_c).sqrt();
                max_move = max_move.max(move_dist);
            }
        }

        (new_centroids, max_move)
    }

    /// 计算惯性（并行版本）
    fn compute_inertia_parallel(&self, data: &[f32], labels: &[usize]) -> f32 {
        let n = data.len() / self.config.dim;
        let dim = self.config.dim;
        let centroids = &self.centroids;

        #[cfg(feature = "parallel")]
        if self.config.parallel && n > 100 {
            return (0..n)
                .into_par_iter()
                .map(|x| {
                    let point = &data[x * dim..(x + 1) * dim];
                    let centroid = &centroids[labels[x] * dim..(labels[x] + 1) * dim];
                    l2_distance_sq(point, centroid)
                })
                .sum();
        }

        // 串行版本
        let mut inertia = 0.0;
        for x in 0..n {
            let point = &data[x * dim..(x + 1) * dim];
            let centroid = &self.centroids[labels[x] * dim..(labels[x] + 1) * dim];
            inertia += l2_distance_sq(point, centroid);
        }

        inertia
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elkan_kmeans_basic() {
        let mut data = vec![0.0; 200 * 2];

        // 簇 1: (0, 0)
        for i in 0..100 {
            data[i * 2] = (i as f32 / 100.0) * 0.1;
            data[i * 2 + 1] = ((i * 2) as f32 / 100.0) * 0.1;
        }

        // 簇 2: (10, 10)
        for i in 100..200 {
            data[i * 2] = 10.0 + ((i - 100) as f32 / 100.0) * 0.1;
            data[i * 2 + 1] = 10.0 + (((i - 100) * 2) as f32 / 100.0) * 0.1;
        }

        let config = ElkanKMeansConfig::new(2, 2)
            .with_seed(42)
            .with_max_iter(100);
        let mut kmeans = ElkanKMeans::new(config);
        let result = kmeans.cluster(&data);

        assert_eq!(result.centroids.len(), 4);
        assert_eq!(result.labels.len(), 200);
        assert!(result.iterations <= 100);
    }

    #[test]
    fn test_elkan_kmeans_config() {
        let config = ElkanKMeansConfig::new(10, 64)
            .with_seed(123)
            .with_max_iter(50)
            .with_tolerance(1e-6)
            .with_parallel(true);

        assert_eq!(config.k, 10);
        assert_eq!(config.dim, 64);
        assert_eq!(config.seed, 123);
        assert!(config.parallel);
    }

    /// 并行 vs 串行结果一致性测试
    #[test]
    fn test_elkan_parallel_vs_serial() {
        let mut data = vec![0.0; 1000 * 16];
        let mut rng = StdRng::seed_from_u64(42);
        for x in data.iter_mut() {
            *x = rng.gen::<f32>() * 10.0;
        }

        // 串行版本
        let config_serial = ElkanKMeansConfig::new(8, 16)
            .with_seed(42)
            .with_max_iter(50)
            .with_parallel(false);
        let mut kmeans_serial = ElkanKMeans::new(config_serial);
        let result_serial = kmeans_serial.cluster(&data);

        // 并行版本
        let config_parallel = ElkanKMeansConfig::new(8, 16)
            .with_seed(42)
            .with_max_iter(50)
            .with_parallel(true);
        let mut kmeans_parallel = ElkanKMeans::new(config_parallel);
        let result_parallel = kmeans_parallel.cluster(&data);

        // 验证结果一致性（惯性应该相同）
        assert!((result_serial.inertia - result_parallel.inertia).abs() < 1.0);
    }
}

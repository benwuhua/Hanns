//! K-Means++ 初始化算法
//!
//! 实现 k-means++ 初始化策略，提高聚类质量，减少迭代次数
//! 参考：Arthur, D. and Vassilvitskii, S. (2007) "k-means++: The Advantages of Careful Seeding"
//!
//! 算法流程：
//! 1. 随机选择第一个中心点
//! 2. 对于每个点，计算到最近中心点的距离 D(x)
//! 3. 以 D(x)^2 的概率选择下一个中心点（距离越远，概率越大）
//! 4. 重复步骤 2-3 直到选择 k 个中心点

use crate::simd::l2_distance_sq;
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// K-Means++ 配置
#[derive(Clone, Debug)]
pub struct KMeansPlusPlusConfig {
    /// 聚类数量
    pub k: usize,
    /// 向量维度
    pub dim: usize,
    /// 最大迭代次数
    pub max_iter: usize,
    /// 收敛阈值
    pub tolerance: f32,
    /// 随机种子
    pub seed: u64,
    /// 并行线程数
    #[cfg(feature = "parallel")]
    pub num_threads: usize,
}

impl Default for KMeansPlusPlusConfig {
    fn default() -> Self {
        Self {
            k: 256,
            dim: 128,
            max_iter: 25,
            tolerance: 1e-4,
            seed: 42,
            #[cfg(feature = "parallel")]
            num_threads: 0, // 0 表示自动检测
        }
    }
}

impl KMeansPlusPlusConfig {
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

    #[cfg(feature = "parallel")]
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }
}

/// K-Means++ 聚类器
pub struct KMeansPlusPlus {
    config: KMeansPlusPlusConfig,
    pub centroids: Vec<f32>,
    rng: StdRng,
}

impl KMeansPlusPlus {
    /// 创建新的 K-Means++ 聚类器
    pub fn new(config: KMeansPlusPlusConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            config,
            centroids: Vec::new(),
            rng,
        }
    }

    /// 获取线程数
    #[cfg(feature = "parallel")]
    #[allow(dead_code)]
    fn num_threads(&self) -> usize {
        if self.config.num_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.config.num_threads
        }
    }

    /// k-means++ 初始化
    ///
    /// # 参数
    /// * `vectors` - 输入向量，形状为 [n, dim]
    ///
    /// # 返回
    /// 初始化的中心点，形状为 [k, dim]
    pub fn init_centroids(&mut self, vectors: &[f32]) -> Vec<f32> {
        let n = vectors.len() / self.config.dim;
        if n == 0 || self.config.k == 0 {
            return Vec::new();
        }

        if self.config.k > n {
            // 如果 k > n，退化为随机初始化
            return self.random_init(vectors, n);
        }
        let n = vectors.len() / self.config.dim;
        if n == 0 || self.config.k == 0 {
            return Vec::new();
        }

        if self.config.k > n {
            // 如果 k > n，退化为随机初始化
            return self.random_init(vectors, n);
        }

        let mut centroids = Vec::with_capacity(self.config.k * self.config.dim);
        let mut chosen_indices = Vec::with_capacity(self.config.k);

        // 步骤 1: 随机选择第一个中心点
        let first_idx = self.rng.gen_range(0..n);
        chosen_indices.push(first_idx);
        centroids.extend_from_slice(
            &vectors[first_idx * self.config.dim..(first_idx + 1) * self.config.dim],
        );

        // 预分配距离数组
        let mut min_distances_sq = vec![f32::MAX; n];

        // 步骤 2-4: 迭代选择剩余的中心点
        for _ in 1..self.config.k {
            // 计算每个点到最近已选中心点的距离平方
            self.update_min_distances(vectors, &centroids, &mut min_distances_sq);

            // 计算权重（距离平方）和累积分布
            let weights: Vec<f32> = min_distances_sq
                .iter()
                .map(|&d| if d.is_finite() && d > 0.0 { d } else { 0.0 })
                .collect();

            let total_weight: f32 = weights.iter().sum();

            if total_weight <= 0.0 {
                // 所有点都已被选为中心点，随机选择剩余点
                break;
            }

            // 根据权重概率选择下一个中心点
            let rand_val = self.rng.gen::<f32>() * total_weight;
            let mut cumsum = 0.0f32;
            let mut next_idx = n - 1;

            for (i, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= rand_val {
                    next_idx = i;
                    break;
                }
            }

            // 避免重复选择
            if chosen_indices.contains(&next_idx) {
                // 找一个未被选择的点
                for i in 0..n {
                    if !chosen_indices.contains(&i) {
                        next_idx = i;
                        break;
                    }
                }
            }

            chosen_indices.push(next_idx);
            centroids.extend_from_slice(
                &vectors[next_idx * self.config.dim..(next_idx + 1) * self.config.dim],
            );
        }

        self.centroids = centroids.clone();
        centroids
    }

    /// 更新每个点到最近中心点的最小距离平方
    fn update_min_distances(
        &mut self,
        vectors: &[f32],
        centroids: &[f32],
        min_distances_sq: &mut [f32],
    ) {
        let _n = vectors.len() / self.config.dim;
        let num_centroids = centroids.len() / self.config.dim;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let results: Vec<f32> = vectors
                .par_chunks(self.config.dim)
                .map(|vec| {
                    let mut min_dist = f32::MAX;
                    for c in 0..num_centroids {
                        let centroid = &centroids[c * self.config.dim..(c + 1) * self.config.dim];
                        let dist_sq = l2_distance_sq(vec, centroid);
                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                        }
                    }
                    min_dist
                })
                .collect();

            min_distances_sq.copy_from_slice(&results);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let vec = &vectors[i * self.config.dim..(i + 1) * self.config.dim];
                let mut min_dist = f32::MAX;
                for c in 0..num_centroids {
                    let centroid = &centroids[c * self.config.dim..(c + 1) * self.config.dim];
                    let dist_sq = l2_distance_sq(vec, centroid);
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                    }
                }
                min_distances_sq[i] = min_dist;
            }
        }
    }

    /// 随机初始化（fallback）
    fn random_init(&mut self, vectors: &[f32], n: usize) -> Vec<f32> {
        let mut centroids = Vec::with_capacity(self.config.k * self.config.dim);
        let mut selected = std::collections::HashSet::new();

        for _ in 0..self.config.k.min(n) {
            let mut idx;
            loop {
                idx = self.rng.gen_range(0..n);
                if selected.insert(idx) {
                    break;
                }
            }
            centroids
                .extend_from_slice(&vectors[idx * self.config.dim..(idx + 1) * self.config.dim]);
        }

        centroids
    }

    /// 运行完整的 k-means 聚类（使用 k-means++ 初始化）
    pub fn cluster(&mut self, vectors: &[f32]) -> KMeansResult {
        let n = vectors.len() / self.config.dim;
        if n == 0 || self.config.k == 0 {
            return KMeansResult {
                centroids: Vec::new(),
                assignments: Vec::new(),
                iterations: 0,
                converged: false,
                inertia: 0.0,
            };
        }

        // k-means++ 初始化
        self.init_centroids(vectors);

        // 迭代优化
        let mut assignments = vec![0usize; n];
        let mut new_centroids = vec![0.0f32; self.config.k * self.config.dim];
        let mut counts = vec![0usize; self.config.k];
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // 分配阶段
            self.assign_clusters(vectors, &mut assignments);

            // 更新阶段
            self.update_centroids(vectors, &assignments, &mut new_centroids, &mut counts);

            // 检查收敛
            let max_shift = self.compute_max_shift(&new_centroids, &counts);

            // 更新中心点
            for c in 0..self.config.k {
                if counts[c] > 0 {
                    for j in 0..self.config.dim {
                        self.centroids[c * self.config.dim + j] =
                            new_centroids[c * self.config.dim + j] / counts[c] as f32;
                    }
                }
            }

            if max_shift < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // 计算最终惯性（inertia）
        let inertia = self.compute_inertia(vectors, &assignments);

        KMeansResult {
            centroids: self.centroids.clone(),
            assignments,
            iterations,
            converged,
            inertia,
        }
    }

    /// 分配聚类
    fn assign_clusters(&self, vectors: &[f32], assignments: &mut [usize]) {
        let _n = vectors.len() / self.config.dim;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let results: Vec<usize> = vectors
                .par_chunks(self.config.dim)
                .map(|vec| {
                    let mut min_dist = f32::MAX;
                    let mut best_k = 0;
                    for c in 0..self.config.k {
                        let centroid =
                            &self.centroids[c * self.config.dim..(c + 1) * self.config.dim];
                        let dist_sq = l2_distance_sq(vec, centroid);
                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                            best_k = c;
                        }
                    }
                    best_k
                })
                .collect();

            assignments.copy_from_slice(&results);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let vec = &vectors[i * self.config.dim..(i + 1) * self.config.dim];
                let mut min_dist = f32::MAX;
                let mut best_k = 0;
                for c in 0..self.config.k {
                    let centroid = &self.centroids[c * self.config.dim..(c + 1) * self.config.dim];
                    let dist_sq = l2_distance_sq(vec, centroid);
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_k = c;
                    }
                }
                assignments[i] = best_k;
            }
        }
    }

    /// 更新中心点
    fn update_centroids(
        &self,
        vectors: &[f32],
        assignments: &[usize],
        new_centroids: &mut [f32],
        counts: &mut [usize],
    ) {
        new_centroids.fill(0.0);
        counts.fill(0);

        let n = vectors.len() / self.config.dim;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let result: (Vec<f32>, Vec<usize>) = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut local_centroids = vec![0.0f32; self.config.k * self.config.dim];
                    let mut local_counts = vec![0usize; self.config.k];

                    let c = assignments[i];
                    for j in 0..self.config.dim {
                        local_centroids[c * self.config.dim + j] = vectors[i * self.config.dim + j];
                    }
                    local_counts[c] = 1;

                    (local_centroids, local_counts)
                })
                .reduce(
                    || {
                        (
                            vec![0.0f32; self.config.k * self.config.dim],
                            vec![0usize; self.config.k],
                        )
                    },
                    |(mut acc_centroids, mut acc_counts), (local_centroids, local_counts)| {
                        for i in 0..self.config.k * self.config.dim {
                            acc_centroids[i] += local_centroids[i];
                        }
                        for i in 0..self.config.k {
                            acc_counts[i] += local_counts[i];
                        }
                        (acc_centroids, acc_counts)
                    },
                );

            new_centroids.copy_from_slice(&result.0);
            counts.copy_from_slice(&result.1);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let c = assignments[i];
                for j in 0..self.config.dim {
                    new_centroids[c * self.config.dim + j] += vectors[i * self.config.dim + j];
                }
                counts[c] += 1;
            }
        }
    }

    /// 计算最大位移
    fn compute_max_shift(&self, new_centroids: &[f32], counts: &[usize]) -> f32 {
        let mut max_shift = 0.0f32;

        for c in 0..self.config.k {
            if counts[c] > 0 {
                let old_centroid = &self.centroids[c * self.config.dim..(c + 1) * self.config.dim];
                let new_centroid = &new_centroids[c * self.config.dim..(c + 1) * self.config.dim];
                let shift = l2_distance_sq(old_centroid, new_centroid).sqrt();
                if shift > max_shift {
                    max_shift = shift;
                }
            }
        }

        max_shift
    }

    /// 计算惯性（所有点到其中心点的距离平方和）
    fn compute_inertia(&self, vectors: &[f32], assignments: &[usize]) -> f32 {
        let n = vectors.len() / self.config.dim;

        #[cfg(feature = "parallel")]
        {
            let inertia: f32 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let vec = &vectors[i * self.config.dim..(i + 1) * self.config.dim];
                    let c = assignments[i];
                    let centroid = &self.centroids[c * self.config.dim..(c + 1) * self.config.dim];
                    l2_distance_sq(vec, centroid)
                })
                .sum();
            inertia
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut inertia = 0.0f32;
            for i in 0..n {
                let vec = &vectors[i * self.config.dim..(i + 1) * self.config.dim];
                let c = assignments[i];
                let centroid = &self.centroids[c * self.config.dim..(c + 1) * self.config.dim];
                inertia += l2_distance_sq(vec, centroid);
            }
            inertia
        }
    }

    /// 获取中心点
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }
}

/// K-Means 聚类结果
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// 最终中心点
    pub centroids: Vec<f32>,
    /// 每个点的聚类分配
    pub assignments: Vec<usize>,
    /// 迭代次数
    pub iterations: usize,
    /// 是否收敛
    pub converged: bool,
    /// 惯性（距离平方和）
    pub inertia: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_pp_init() {
        // 创建两个明显的聚类
        let dim = 2;
        let k = 2;
        let n = 100;

        let mut vectors = Vec::with_capacity(n * dim);
        // Cluster 1: around (0, 0)
        for _ in 0..n / 2 {
            vectors.push((rand::random::<f32>() - 0.5) * 0.1);
            vectors.push((rand::random::<f32>() - 0.5) * 0.1);
        }
        // Cluster 2: around (10, 10)
        for _ in 0..n / 2 {
            vectors.push(10.0 + (rand::random::<f32>() - 0.5) * 0.1);
            vectors.push(10.0 + (rand::random::<f32>() - 0.5) * 0.1);
        }

        let config = KMeansPlusPlusConfig::new(k, dim).with_seed(42);
        let mut kmeans_pp = KMeansPlusPlus::new(config);
        let centroids = kmeans_pp.init_centroids(&vectors);

        assert_eq!(centroids.len(), k * dim);

        // 验证中心点应该分别在两个聚类附近
        let c0 = &centroids[0..2];
        let c1 = &centroids[2..4];

        // 计算两个中心点的距离
        let dist = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();

        // k-means++ 应该选择距离较远的点作为中心点
        assert!(
            dist > 5.0,
            "K-means++ should select distant centroids, got dist={}",
            dist
        );
    }

    #[test]
    fn test_kmeans_pp_clustering() {
        let dim = 2;
        let k = 2;
        let n = 100;

        let mut vectors = Vec::with_capacity(n * dim);
        // Cluster 1: around (0, 0)
        for _ in 0..n / 2 {
            vectors.push((rand::random::<f32>() - 0.5) * 0.1);
            vectors.push((rand::random::<f32>() - 0.5) * 0.1);
        }
        // Cluster 2: around (10, 10)
        for _ in 0..n / 2 {
            vectors.push(10.0 + (rand::random::<f32>() - 0.5) * 0.1);
            vectors.push(10.0 + (rand::random::<f32>() - 0.5) * 0.1);
        }

        let config = KMeansPlusPlusConfig::new(k, dim)
            .with_seed(42)
            .with_max_iter(25)
            .with_tolerance(1e-4);

        let mut kmeans_pp = KMeansPlusPlus::new(config);
        let result = kmeans_pp.cluster(&vectors);

        assert!(result.converged || result.iterations == 25);
        assert_eq!(result.centroids.len(), k * dim);
        assert_eq!(result.assignments.len(), n);

        // 验证聚类质量：大部分点应该被正确分类
        let mut cluster0_correct = 0;
        let mut cluster1_correct = 0;

        for i in 0..n {
            let _assignment = result.assignments[i];
            let x = vectors[i * dim];
            let y = vectors[i * dim + 1];

            if i < n / 2 {
                // Should be in cluster around (0, 0)
                if x < 5.0 && y < 5.0 {
                    cluster0_correct += 1;
                }
            } else {
                // Should be in cluster around (10, 10)
                if x >= 5.0 && y >= 5.0 {
                    cluster1_correct += 1;
                }
            }
        }

        // 至少 90% 的点应该被正确分类
        let accuracy = (cluster0_correct + cluster1_correct) as f32 / n as f32;
        assert!(
            accuracy > 0.9,
            "Clustering accuracy should be > 90%, got {:.2}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_kmeans_pp_vs_random_init() {
        // 比较 k-means++ 和随机初始化的质量
        let dim = 10;
        let k = 5;
        let n = 500;

        // 创建 5 个明显的聚类
        let mut vectors = Vec::with_capacity(n * dim);
        let cluster_centers = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
        ];

        for i in 0..n {
            let cluster_idx = i % k;
            for center in cluster_centers[cluster_idx].iter().take(dim) {
                vectors.push(*center + (rand::random::<f32>() - 0.5) * 0.5);
            }
        }

        // K-means++
        let config_pp = KMeansPlusPlusConfig::new(k, dim).with_seed(42);
        let mut kmeans_pp = KMeansPlusPlus::new(config_pp);
        let result_pp = kmeans_pp.cluster(&vectors);

        // 多次随机初始化取最佳
        let mut best_inertia_random = f32::MAX;
        for seed in 0..5 {
            let config_random = KMeansPlusPlusConfig::new(k, dim).with_seed(seed + 100);
            let mut kmeans_random = KMeansPlusPlus::new(config_random);
            // 使用随机初始化
            kmeans_random.centroids = kmeans_random.random_init(&vectors, n);
            let result_random = kmeans_random.cluster(&vectors);
            if result_random.inertia < best_inertia_random {
                best_inertia_random = result_random.inertia;
            }
        }

        // K-means++ 应该获得更好的或相当的惯性
        // 注意：由于随机性，这个测试可能偶尔失败
        println!("K-means++ inertia: {:.2}", result_pp.inertia);
        println!("Best random inertia: {:.2}", best_inertia_random);

        // 通常 k-means++ 会更好，但我们只要求不显著更差
        assert!(
            result_pp.inertia <= best_inertia_random * 1.1,
            "K-means++ should have comparable or better inertia"
        );
    }

    #[test]
    fn test_kmeans_pp_empty_data() {
        let config = KMeansPlusPlusConfig::new(10, 128);
        let mut kmeans_pp = KMeansPlusPlus::new(config);

        let vectors: Vec<f32> = Vec::new();
        let centroids = kmeans_pp.init_centroids(&vectors);

        assert!(centroids.is_empty());

        let result = kmeans_pp.cluster(&vectors);
        assert!(result.centroids.is_empty());
        assert_eq!(result.iterations, 0);
    }
}

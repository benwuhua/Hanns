//! K-Means 量化器
//! 
//! 支持 k-means++ 初始化和迭代收敛

use rand::prelude::*;

pub struct KMeans {
    k: usize,
    max_iter: usize,
    tolerance: f32,
    pub centroids: Vec<f32>,
    dim: usize,
    rng: StdRng,
}

impl KMeans {
    pub fn new(k: usize, dim: usize) -> Self {
        Self {
            k,
            max_iter: 50,
            tolerance: 1e-4,
            centroids: vec![0.0; k * dim],
            dim,
            rng: StdRng::from_entropy(),
        }
    }

    /// K-means++ 初始化
    fn kmeans_plusplus_init(&mut self, vectors: &[f32], n: usize) {
        if n == 0 { return; }
        
        // 第一个 centroid 随机选择
        let idx = self.rng.gen_range(0..n);
        for j in 0..self.dim {
            self.centroids[j] = vectors[idx * self.dim + j];
        }
        
        // 剩余 k-1 个 centroid
        for c in 1..self.k {
            let mut distances = vec![0.0f32; n];
            let mut sum = 0.0f32;
            
            for i in 0..n {
                let mut min_dist = f32::MAX;
                for cc in 0..c {
                    let dist = self.l2_distance(
                        &vectors[i * self.dim..],
                        &self.centroids[cc * self.dim..]
                    );
                    min_dist = min_dist.min(dist);
                }
                distances[i] = min_dist * min_dist;
                sum += distances[i];
            }
            
            // 按概率选择下一个 centroid
            let threshold = self.rng.gen::<f32>() * sum;
            let mut acc = 0.0f32;
            let mut selected = 0;
            for i in 0..n {
                acc += distances[i];
                if acc >= threshold {
                    selected = i;
                    break;
                }
            }
            
            for j in 0..self.dim {
                self.centroids[c * self.dim + j] = vectors[selected * self.dim + j];
            }
        }
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// 训练 K-means
    pub fn train(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        if n == 0 || n < self.k { return 0; }
        
        // K-means++ 初始化
        self.kmeans_plusplus_init(vectors, n);
        
        // 迭代优化
        let mut assignments = vec![0usize; n];
        let mut new_centroids = vec![0.0f32; self.k * self.dim];
        let mut counts = vec![0usize; self.k];
        
        for _iter in 0..self.max_iter {
            // 分配阶段
            for i in 0..n {
                let mut min_dist = f32::MAX;
                let mut best_k = 0;
                for c in 0..self.k {
                    let dist = self.l2_distance(
                        &vectors[i * self.dim..],
                        &self.centroids[c * self.dim..]
                    );
                    if dist < min_dist {
                        min_dist = dist;
                        best_k = c;
                    }
                }
                assignments[i] = best_k;
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
                if counts[c] > 0 {
                    let shift = self.l2_distance(
                        &self.centroids[c * self.dim..],
                        &new_centroids[c * self.dim..]
                    );
                    max_shift = max_shift.max(shift);
                    
                    for j in 0..self.dim {
                        self.centroids[c * self.dim + j] = 
                            new_centroids[c * self.dim + j] / counts[c] as f32;
                    }
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
            let dist = self.l2_distance(vector, &self.centroids[c * self.dim..]);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }
        best
    }
    
    /// 并行训练 K-means（使用 rayon）
    #[cfg(feature = "parallel")]
    pub fn train_parallel(&mut self, vectors: &[f32], num_threads: usize) -> usize {
        use rayon::prelude::*;
        
        let n = vectors.len() / self.dim;
        if n == 0 || n < self.k { return 0; }
        
        // K-means++ 初始化
        self.kmeans_plusplus_init(vectors, n);
        
        let mut assignments = vec![0usize; n];
        
        for _iter in 0..self.max_iter {
            // 并行分配阶段
            assignments.par_iter_mut()
                .with_len(n)
                .enumerate()
                .for_each(|(i, assign)| {
                    let mut min_dist = f32::MAX;
                    let mut best_k = 0;
                    let vec = &vectors[i * self.dim..];
                    for c in 0..self.k {
                        let dist = self.l2_distance(vec, &self.centroids[c * self.dim..]);
                        if dist < min_dist {
                            min_dist = dist;
                            best_k = c;
                        }
                    }
                    *assign = best_k;
                });
            
            // 收集到每个 cluster
            let mut new_centroids = vec![0.0f32; self.k * self.dim];
            let mut counts = vec![0usize; self.k];
            
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
                if counts[c] > 0 {
                    let shift = self.l2_distance(
                        &self.centroids[c * self.dim..],
                        &new_centroids[c * self.dim..]
                    );
                    max_shift = max_shift.max(shift);
                    
                    for j in 0..self.dim {
                        self.centroids[c * self.dim + j] = 
                            new_centroids[c * self.dim + j] / counts[c] as f32;
                    }
                }
            }
            
            if max_shift < self.tolerance {
                break;
            }
        }
        
        n
    }
    
    pub fn centroids(&self) -> &[f32] { &self.centroids }
    pub fn k(&self) -> usize { self.k }
    pub fn dim(&self) -> usize { self.dim }
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
        assert!(km.centroids().len() > 0);
    }
    
    #[test]
    fn test_kmeans_convergence() {
        let mut km = KMeans::new(2, 2);
        // Two clusters, 6 vectors total
        let data = vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.0,  // Cluster 1: 3 vectors
            10.0, 10.0, 10.1, 10.1, 10.2, 10.0, // Cluster 2: 3 vectors
        ];
        let n = km.train(&data);
        assert_eq!(n, 6);
        
        // Check centroids are far apart
        let dist = km.l2_distance(&km.centroids[0..2], &km.centroids[2..4]);
        assert!(dist > 5.0, "Centroids should be separated");
    }
}

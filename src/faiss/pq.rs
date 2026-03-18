//! PQ (Product Quantization) 编码器
//!
//! 标准 Product Quantization 实现，支持:
//! - K-means 码书训练
//! - SIMD 距离计算
//! - 查表距离 (AD, ASD)

use rayon::prelude::*;
use crate::simd;

/// PQ 编码器
pub struct PqEncoder {
    pub m: usize,            // 子向量数
    pub k: usize,            // 每个子空间的聚类数 (必须是 2^n)
    pub nbits: usize,        // 编码位数
    pub dim: usize,          // 原始维度
    pub sub_dim: usize,      // 子向量维度
    pub codebooks: Vec<f32>, // 码书 [m * k * sub_dim]
}

impl PqEncoder {
    /// 创建新的 PQ 编码器
    pub fn new(dim: usize, m: usize, k: usize) -> Self {
        let sub_dim = dim / m;
        let nbits = (k as f64).log2() as usize;

        Self {
            m,
            k,
            nbits,
            dim,
            sub_dim,
            codebooks: vec![0.0; m * k * sub_dim],
        }
    }

    /// 训练码书 (使用 K-means，子空间并行)
    pub fn train(&mut self, data_in: &[f32], max_iter: usize) {
        let n = data_in.len() / self.dim;
        if n == 0 {
            return;
        }

        const MAX_TRAIN_SAMPLES: usize = 200_000;

        let data = if n > MAX_TRAIN_SAMPLES {
            // 随机子采样 MAX_TRAIN_SAMPLES 行
            use rand::seq::SliceRandom;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(MAX_TRAIN_SAMPLES);
            let mut sampled = Vec::with_capacity(MAX_TRAIN_SAMPLES * self.dim);
            for &i in &indices {
                sampled.extend_from_slice(&data_in[i * self.dim..(i + 1) * self.dim]);
            }
            std::borrow::Cow::Owned(sampled)
        } else {
            std::borrow::Cow::Borrowed(data_in)
        };
        let n = data.len() / self.dim;  // 重新计算 n（可能已缩小）

        let m = self.m;
        let dim = self.dim;
        let sub_dim = self.sub_dim;
        let k = self.k;

        // Extract sub-vectors per subspace and train all m subspaces in parallel.
        // Each subspace is independent — this is embarrassingly parallel.
        let codebook_size = k * sub_dim;
        let trained: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|m_idx| {
                let mut sub_vectors = Vec::with_capacity(n * sub_dim);
                for i in 0..n {
                    let base = i * dim + m_idx * sub_dim;
                    sub_vectors.extend_from_slice(&data[base..base + sub_dim]);
                }
                Self::train_sub_codebook_pure(sub_dim, k, &sub_vectors, max_iter)
            })
            .collect();

        // Write results back to codebooks sequentially (no contention).
        for (m_idx, codebook) in trained.into_iter().enumerate() {
            let offset = m_idx * codebook_size;
            self.codebooks[offset..offset + codebook_size].copy_from_slice(&codebook);
        }
    }

    /// Pure-function k-means for one subspace — no &self mutation, safe for par_iter.
    fn train_sub_codebook_pure(sub_dim: usize, k: usize, vectors: &[f32], max_iter: usize) -> Vec<f32> {
        let n = vectors.len() / sub_dim;
        if n < k {
            return vec![0.0f32; k * sub_dim];
        }

        // K-means++ 初始化
        let mut centroids = vec![0.0f32; k * sub_dim];
        Self::kmeans_init_pure(sub_dim, k, vectors, &mut centroids);

        // 迭代优化：assignment 步并行，update 步串行累加
        let mut assignments = vec![0usize; n];
        let mut new_centroids = vec![0.0f32; k * sub_dim];
        let mut counts = vec![0usize; k];

        for _ in 0..max_iter {
            // Assignment: parallel over vectors
            assignments
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, assignment)| {
                    let sub_vec = &vectors[i * sub_dim..(i + 1) * sub_dim];
                    let mut min_dist = f32::MAX;
                    let mut best = 0;
                    for c in 0..k {
                        let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                        let dist: f32 = sub_vec
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        if dist < min_dist {
                            min_dist = dist;
                            best = c;
                        }
                    }
                    *assignment = best;
                });

            // Update: serial accumulation (avoids atomic f32 contention)
            new_centroids.fill(0.0);
            counts.fill(0);
            for i in 0..n {
                let c = assignments[i];
                for j in 0..sub_dim {
                    new_centroids[c * sub_dim + j] += vectors[i * sub_dim + j];
                }
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..sub_dim {
                        centroids[c * sub_dim + j] =
                            new_centroids[c * sub_dim + j] / counts[c] as f32;
                    }
                }
            }
        }

        centroids
    }

    /// K-means++ 初始化 (pure, no &self needed)
    fn kmeans_init_pure(sub_dim: usize, k: usize, vectors: &[f32], centroids: &mut [f32]) {
        let n = vectors.len() / sub_dim;
        if n == 0 {
            return;
        }

        use rand::prelude::*;
        let mut rng = StdRng::from_entropy();

        let idx = rng.gen_range(0..n);
        centroids[..sub_dim].copy_from_slice(&vectors[idx * sub_dim..(idx + 1) * sub_dim]);

        for c in 1..k {
            let mut distances = vec![0.0f32; n];
            let mut sum = 0.0f32;
            for i in 0..n {
                let sub_vec = &vectors[i * sub_dim..(i + 1) * sub_dim];
                let mut min_dist = f32::MAX;
                for cc in 0..c {
                    let centroid = &centroids[cc * sub_dim..(cc + 1) * sub_dim];
                    let d: f32 = sub_vec
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    min_dist = min_dist.min(d);
                }
                distances[i] = min_dist;
                sum += min_dist;
            }
            let threshold = rng.gen::<f32>() * sum;
            let mut acc = 0.0f32;
            let mut selected = 0;
            for (i, &d) in distances.iter().enumerate() {
                acc += d;
                if acc >= threshold {
                    selected = i;
                    break;
                }
            }
            centroids[c * sub_dim..(c + 1) * sub_dim]
                .copy_from_slice(&vectors[selected * sub_dim..(selected + 1) * sub_dim]);
        }
    }

    /// 编码：将向量转换为紧凑码
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = vec![0u8; self.m];

        for m_idx in 0..self.m {
            let sub_vec = &vector[m_idx * self.sub_dim..(m_idx + 1) * self.sub_dim];
            let codebook =
                &self.codebooks[m_idx * self.k * self.sub_dim..(m_idx + 1) * self.k * self.sub_dim];

            // 找最近邻
            let mut min_dist = f32::MAX;
            let mut best = 0;

            for c in 0..self.k {
                let cent = &codebook[c * self.sub_dim..(c + 1) * self.sub_dim];
                let d = simd::l2_distance(sub_vec, cent);
                if d < min_dist {
                    min_dist = d;
                    best = c;
                }
            }

            codes[m_idx] = best as u8;
        }

        codes
    }

    /// 解码：从码恢复向量
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = vec![0.0; self.dim];

        for m_idx in 0..self.m {
            let c = codes[m_idx] as usize;
            let offset = m_idx * self.k * self.sub_dim + c * self.sub_dim;
            let cent = &self.codebooks[offset..offset + self.sub_dim];
            for j in 0..self.sub_dim {
                result[m_idx * self.sub_dim + j] = cent[j];
            }
        }

        result
    }

    /// 构建查询距离表 (Asymmetric Distance Computation)
    ///
    /// 返回扁平距离表，布局为 [m0c0, m0c1, ..., m1c0, ...]，步长为 `self.k`
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = vec![0.0f32; self.m * self.k];

        for m_idx in 0..self.m {
            let query_sub = &query[m_idx * self.sub_dim..(m_idx + 1) * self.sub_dim];
            let codebook =
                &self.codebooks[m_idx * self.k * self.sub_dim..(m_idx + 1) * self.k * self.sub_dim];

            for c in 0..self.k {
                let cent = &codebook[c * self.sub_dim..(c + 1) * self.sub_dim];
                table[m_idx * self.k + c] = simd::l2_distance(query_sub, cent);
            }
        }

        table
    }

    /// 使用距离表计算与编码向量的距离 (ADC)
    #[inline(always)]
    pub fn compute_distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        let m = self.m.min(codes.len()).min(table.len() / self.k.max(1));
        distance_with_table_simd_flat(table, codes, m, self.k)
    }

}

#[inline(always)]
fn distance_with_table_simd_flat(table: &[f32], codes: &[u8], m: usize, k: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(table.as_ptr() as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }

    let mut sum = 0.0f32;
    let chunks4 = (m / 4) * 4;
    let mut i = 0usize;
    while i < chunks4 {
        sum += table[i * k + codes[i] as usize];
        sum += table[(i + 1) * k + codes[i + 1] as usize];
        sum += table[(i + 2) * k + codes[i + 2] as usize];
        sum += table[(i + 3) * k + codes[i + 3] as usize];
        i += 4;
    }
    while i < m {
        sum += table[i * k + codes[i] as usize];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_new() {
        let pq = PqEncoder::new(128, 8, 256);
        assert_eq!(pq.m, 8);
        assert_eq!(pq.k, 256);
        assert_eq!(pq.sub_dim, 16);
    }

    #[test]
    fn test_pq_encode_decode() {
        let pq = PqEncoder::new(8, 2, 4);

        // 简单数据
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let code = pq.encode(&vector);
        assert_eq!(code.len(), 2);

        let decoded = pq.decode(&code);
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn test_pq_train() {
        let mut pq = PqEncoder::new(8, 2, 4);

        // 生成训练数据：两个明显不同的簇
        let mut data = vec![0.0f32; 32];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 8 + j] = (j as f32) * 0.1;
                data[i * 8 + j + 4] = 10.0 + (j as f32) * 0.1;
            }
        }

        pq.train(&data, 10);

        // 验证码书已填充
        assert!(pq.codebooks.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_distance_table() {
        let pq = PqEncoder::new(8, 2, 4);

        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let table = pq.build_distance_table(&query);

        assert_eq!(table.len(), 8);
    }
}

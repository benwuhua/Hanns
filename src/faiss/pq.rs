//! PQ (Product Quantization) 编码器
//! 
//! 标准 Product Quantization 实现，支持:
//! - K-means 码书训练
//! - SIMD 距离计算
//! - 查表距离 (AD, ASD)

use crate::simd;

/// PQ 编码器
pub struct PqEncoder {
    pub m: usize,           // 子向量数
    pub k: usize,           // 每个子空间的聚类数 (必须是 2^n)
    pub nbits: usize,       // 编码位数
    pub dim: usize,         // 原始维度
    pub sub_dim: usize,     // 子向量维度
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
    
    /// 训练码书 (使用 K-means)
    pub fn train(&mut self, data: &[f32], max_iter: usize) {
        let n = data.len() / self.dim;
        if n == 0 { return; }
        
        // 对每个子空间进行 K-means
        for m_idx in 0..self.m {
            // 提取子向量
            let mut sub_vectors = Vec::with_capacity(n * self.sub_dim);
            for i in 0..n {
                for j in 0..self.sub_dim {
                    sub_vectors.push(data[i * self.dim + m_idx * self.sub_dim + j]);
                }
            }
            
            // 运行 K-means
            self.train_sub_codebook(m_idx, &sub_vectors, max_iter);
        }
    }
    
    /// 训练单个子空间的码书
    fn train_sub_codebook(&mut self, m_idx: usize, vectors: &[f32], max_iter: usize) {
        let n = vectors.len() / self.sub_dim;
        if n < self.k { return; }
        
        // K-means++ 初始化
        let mut centroids = vec![0.0f32; self.k * self.sub_dim];
        self.kmeans_init(&vectors, &mut centroids);
        
        // 迭代优化
        let mut assignments = vec![0usize; n];
        let mut new_centroids = vec![0.0f32; self.k * self.sub_dim];
        let mut counts = vec![0usize; self.k];
        
        for _ in 0..max_iter {
            // 分配
            for i in 0..n {
                let sub_vec = &vectors[i * self.sub_dim..];
                let mut min_dist = f32::MAX;
                let mut best = 0;
                
                for c in 0..self.k {
                    let dist = self.l2_sqr(sub_vec, &centroids[c * self.sub_dim..]);
                    if dist < min_dist {
                        min_dist = dist;
                        best = c;
                    }
                }
                assignments[i] = best;
            }
            
            // 更新
            new_centroids.fill(0.0);
            counts.fill(0);
            
            for i in 0..n {
                let c = assignments[i];
                for j in 0..self.sub_dim {
                    new_centroids[c * self.sub_dim + j] += vectors[i * self.sub_dim + j];
                }
                counts[c] += 1;
            }
            
            for c in 0..self.k {
                if counts[c] > 0 {
                    for j in 0..self.sub_dim {
                        centroids[c * self.sub_dim + j] = 
                            new_centroids[c * self.sub_dim + j] / counts[c] as f32;
                    }
                }
            }
        }
        
        // 保存码书
        let offset = m_idx * self.k * self.sub_dim;
        for i in 0..(self.k * self.sub_dim) {
            self.codebooks[offset + i] = centroids[i];
        }
    }
    
    /// K-means++ 初始化
    fn kmeans_init(&self, vectors: &[f32], centroids: &mut [f32]) {
        let n = vectors.len() / self.sub_dim;
        if n == 0 { return; }
        
        use rand::prelude::*;
        let mut rng = StdRng::from_entropy();
        
        // 第一个 centroid 随机选择
        let idx = rng.gen_range(0..n);
        for j in 0..self.sub_dim {
            centroids[j] = vectors[idx * self.sub_dim + j];
        }
        
        // 剩余 k-1 个
        for c in 1..self.k {
            let mut distances = vec![0.0f32; n];
            let mut sum = 0.0f32;
            
            for i in 0..n {
                let sub_vec = &vectors[i * self.sub_dim..];
                let mut min_dist = f32::MAX;
                for cc in 0..c {
                    let d = self.l2_sqr(sub_vec, &centroids[cc * self.sub_dim..]);
                    min_dist = min_dist.min(d);
                }
                distances[i] = min_dist;
                sum += min_dist;
            }
            
            let threshold = rng.gen::<f32>() * sum;
            let mut acc = 0.0f32;
            let mut selected = 0;
            for i in 0..n {
                acc += distances[i];
                if acc >= threshold {
                    selected = i;
                    break;
                }
            }
            
            for j in 0..self.sub_dim {
                centroids[c * self.sub_dim + j] = vectors[selected * self.sub_dim + j];
            }
        }
    }
    
    /// 编码：将向量转换为紧凑码
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = vec![0u8; self.m];
        
        for m_idx in 0..self.m {
            let sub_vec = &vector[m_idx * self.sub_dim..];
            let codebook = &self.codebooks[m_idx * self.k * self.sub_dim..];
            
            // 找最近邻
            let mut min_dist = f32::MAX;
            let mut best = 0;
            
            for c in 0..self.k {
                let cent = &codebook[c * self.sub_dim..];
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
    /// 返回 [m][k] 的距离表
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = vec![vec![0.0; self.k]; self.m];
        
        for m_idx in 0..self.m {
            let query_sub = &query[m_idx * self.sub_dim..];
            let codebook = &self.codebooks[m_idx * self.k * self.sub_dim..];
            
            for c in 0..self.k {
                let cent = &codebook[c * self.sub_dim..];
                table[m_idx][c] = simd::l2_distance(query_sub, cent);
            }
        }
        
        table
    }
    
    /// 使用距离表计算与编码向量的距离 (ADC)
    #[inline]
    pub fn compute_distance_with_table(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for m_idx in 0..self.m {
            let c = codes[m_idx] as usize;
            sum += table[m_idx][c];
        }
        sum
    }
    
    /// 计算 L2 距离 (平方)
    #[inline]
    fn l2_sqr(&self, a: &[f32], b: &[f32]) -> f32 {
        let d = simd::l2_distance(a, b);
        d * d
    }
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
        
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].len(), 4);
    }
}

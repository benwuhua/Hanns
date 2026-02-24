//! PQ (Product Quantization) 编码器

/// PQ 编码器
pub struct PqEncoder {
    pub m: usize,           // 子向量数
    pub k: usize,           // 每个子空间的聚类数
    pub nbits: usize,       // 编码位数
    pub dim: usize,         // 原始维度
    pub codebooks: Vec<f32>, // 码书 [m * k * (dim/m)]
}

impl PqEncoder {
    pub fn new(dim: usize, m: usize, k: usize) -> Self {
        let sub_dim = dim / m;
        Self {
            m,
            k,
            nbits: (k as f64).log2() as usize,
            dim,
            codebooks: vec![0.0; m * k * sub_dim],
        }
    }
    
    /// 训练码书
    pub fn train(&mut self, data: &[f32]) {
        let sub_dim = self.dim / self.m;
        
        // 对每个子空间进行 K-Means
        for m in 0..self.m {
            // 提取子向量
            let mut sub_vectors = vec![0.0; data.len() / self.dim * sub_dim];
            let n = data.len() / self.dim;
            
            for i in 0..n {
                for j in 0..sub_dim {
                    sub_vectors[i * sub_dim + j] = data[i * self.dim + m * sub_dim + j];
                }
            }
            
            // 简单聚类：取前 k 个样本作为质心
            let n_sub = sub_vectors.len() / sub_dim;
            for c in 0..self.k.min(n_sub) {
                for j in 0..sub_dim {
                    self.codebooks[m * self.k * sub_dim + c * sub_dim + j] = 
                        sub_vectors[c * sub_dim + j];
                }
            }
        }
    }
    
    /// 编码：将向量转换为紧凑码
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = self.dim / self.m;
        let mut codes = vec![0u8; self.m];
        
        for m in 0..self.m {
            let sub_vec = &vector[m * sub_dim..];
            let codebook = &self.codebooks[m * self.k * sub_dim..];
            
            // 找最近邻
            let mut min_dist = f32::MAX;
            let mut best = 0;
            
            for c in 0..self.k {
                let cent = &codebook[c * sub_dim..];
                let d = self.l2_sqr(sub_vec, cent, sub_dim);
                if d < min_dist {
                    min_dist = d;
                    best = c;
                }
            }
            
            codes[m] = best as u8;
        }
        
        codes
    }
    
    /// 解码：从码恢复向量
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let sub_dim = self.dim / self.m;
        let mut result = vec![0.0; self.dim];
        
        for m in 0..self.m {
            let c = codes[m] as usize;
            let cent = &self.codebooks[m * self.k * sub_dim + c * sub_dim..];
            for j in 0..sub_dim {
                result[m * sub_dim + j] = cent[j];
            }
        }
        
        result
    }
    
    /// 计算 L2 距离（查表）
    pub fn compute_distances(&self, query: &[f32], codes: &[u8]) -> f32 {
        let sub_dim = self.dim / self.m;
        let mut sum = 0.0f32;
        
        for m in 0..self.m {
            let c = codes[m] as usize;
            let cent = &self.codebooks[m * self.k * sub_dim + c * sub_dim..];
            let d = self.l2_sqr(&query[m * sub_dim..], cent, sub_dim);
            sum += d;
        }
        
        sum
    }
    
    #[inline]
    fn l2_sqr(&self, a: &[f32], b: &[f32], dim: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..dim {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
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
    fn test_pq_distance() {
        let pq = PqEncoder::new(8, 2, 4);
        
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let code = vec![0, 0];
        
        let dist = pq.compute_distances(&query, &code);
        assert!(dist >= 0.0);
    }
}

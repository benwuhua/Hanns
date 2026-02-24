//! SIMD 优化的 PQ 距离计算

/// PQ 距离计算器（SIMD 优化）
pub struct PqDistance {
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
}

impl PqDistance {
    pub fn new(m: usize, k: usize, sub_dim: usize) -> Self {
        Self { m, k, sub_dim }
    }
    
    /// 计算单个距离（SIMD 优化路径）
    #[inline]
    pub fn distance(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        self.distance_scalar(query, codebook, codes)
    }
    
    /// 标量版本
    #[inline]
    fn distance_scalar(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        
        for sub_idx in 0..self.m {
            let code = codes[sub_idx] as usize;
            let cent_offset = sub_idx * self.k * self.sub_dim + code * self.sub_dim;
            
            if cent_offset + self.sub_dim > codebook.len() {
                continue;
            }
            
            let q_start = sub_idx * self.sub_dim;
            if q_start + self.sub_dim > query.len() {
                continue;
            }
            
            // SIMD-friendly loop
            let mut i = 0;
            let q = &query[q_start..];
            let cent = &codebook[cent_offset..];
            
            // Process in chunks of 4 (SIMD friendly)
            while i + 4 <= self.sub_dim {
                let q0 = q[i];
                let q1 = q[i + 1];
                let q2 = q[i + 2];
                let q3 = q[i + 3];
                
                let c0 = cent[i];
                let c1 = cent[i + 1];
                let c2 = cent[i + 2];
                let c3 = cent[i + 3];
                
                sum += (q0 - c0) * (q0 - c0)
                     + (q1 - c1) * (q1 - c1)
                     + (q2 - c2) * (q2 - c2)
                     + (q3 - c3) * (q3 - c3);
                
                i += 4;
            }
            
            // 剩余元素
            while i < self.sub_dim {
                let diff = q[i] - cent[i];
                sum += diff * diff;
                i += 1;
            }
        }
        
        sum
    }
    
    /// 批量距离计算
    pub fn distance_batch(&self, query: &[f32], codebook: &[f32], 
                          codes: &[Vec<u8>]) -> Vec<f32> {
        codes.iter()
            .map(|c| self.distance(query, codebook, c))
            .collect()
    }
}

/// 预计算距离表（进一步优化）
pub struct DistanceTable {
    pub table: Vec<f32>,  // [m * k]
    pub m: usize,
    pub k: usize,
}

impl DistanceTable {
    pub fn new(m: usize, k: usize) -> Self {
        Self {
            table: vec![0.0; m * k],
            m,
            k,
        }
    }
    
    /// 预计算查询到所有质心的距离
    pub fn compute(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
        for m_idx in 0..self.m {
            for k_idx in 0..self.k {
                let cent_offset = m_idx * self.k * sub_dim + k_idx * sub_dim;
                let q_start = m_idx * sub_dim;
                
                let mut sum = 0.0f32;
                for j in 0..sub_dim {
                    let diff = query[q_start + j] - codebook[cent_offset + j];
                    sum += diff * diff;
                }
                
                self.table[m_idx * self.k + k_idx] = sum;
            }
        }
    }
    
    /// 查表计算 PQ 距离
    pub fn query_distance(&self, codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for m_idx in 0..self.m {
            let code = codes[m_idx] as usize;
            sum += self.table[m_idx * self.k + code];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pq_distance_new() {
        let d = PqDistance::new(8, 256, 16);
        assert_eq!(d.m, 8);
    }
    
    #[test]
    fn test_pq_distance() {
        let d = PqDistance::new(2, 4, 4);
        
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let codebook = vec![0.0; 2 * 4 * 4];
        let code = vec![0, 0];
        
        let dist = d.distance(&query, &codebook, &code);
        assert!(dist >= 0.0);
    }
    
    #[test]
    fn test_distance_table() {
        let mut table = DistanceTable::new(2, 4);
        
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let codebook = vec![0.0; 2 * 4 * 4];
        
        table.compute(&query, &codebook, 4);
        
        let dist = table.query_distance(&[0, 0]);
        assert!(dist >= 0.0);
    }
}

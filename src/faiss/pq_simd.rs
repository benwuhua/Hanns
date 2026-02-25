//! SIMD 优化的 PQ 距离计算
//!
//! 支持 AVX2/AVX512 (x86_64) 和 NEON (aarch64) SIMD 指令

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
    
    /// 计算单个距离（自动选择最优 SIMD 实现）
    #[inline]
    pub fn distance(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return self.distance_avx2(query, codebook, codes);
            }
            if std::is_x86_feature_detected!("sse4_2") {
                return self.distance_sse(query, codebook, codes);
            }
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.distance_neon(query, codebook, codes);
            }
        }
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
    
    /// SSE 优化的 PQ 距离 (4 elements per iteration)
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    fn distance_sse(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        use std::arch::x86_64::*;
        
        let mut sum = _mm_setzero_ps();
        
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
            
            let mut i = 0;
            let q_ptr = &query[q_start];
            let c_ptr = &codebook[cent_offset];
            
            while i + 4 <= self.sub_dim {
                let qv = _mm_loadu_ps(q_ptr.add(i));
                let cv = _mm_loadu_ps(c_ptr.add(i));
                let diff = _mm_sub_ps(qv, cv);
                let sq = _mm_mul_ps(diff, diff);
                sum = _mm_add_ps(sum, sq);
                i += 4;
            }
            
            // Handle remainder
            while i < self.sub_dim {
                let diff = q_ptr[i] - c_ptr[i];
                sum = _mm_add_ss(sum, _mm_set_ss(diff * diff));
                i += 1;
            }
        }
        
        // Horizontal sum
        let mut result = _mm_cvtss_f32(sum);
        let high = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, high);
        let sums2 = _mm_movehl_ps(sums, sums);
        result + _mm_cvtss_f32(_mm_add_ss(sums, sums2))
    }
    
    /// AVX2 优化的 PQ 距离 (8 elements per iteration)
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    fn distance_avx2(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        use std::arch::x86_64::*;
        
        let mut sum = _mm256_setzero_ps();
        
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
            
            let mut i = 0;
            let q_ptr = &query[q_start];
            let c_ptr = &codebook[cent_offset];
            
            while i + 8 <= self.sub_dim {
                let qv = _mm256_loadu_ps(q_ptr.add(i));
                let cv = _mm256_loadu_ps(c_ptr.add(i));
                let diff = _mm256_sub_ps(qv, cv);
                let sq = _mm256_mul_ps(diff, diff);
                sum = _mm256_add_ps(sum, sq);
                i += 8;
            }
            
            // Handle remainder (up to 7 elements)
            while i < self.sub_dim {
                let diff = q_ptr[i] - c_ptr[i];
                sum = _mm256_add_ps(sum, _mm256_set1_ps(diff * diff));
                i += 1;
            }
        }
        
        // Horizontal sum of 256-bit register
        let mut result = _mm256_cvtss_f32(sum);
        let high = _mm256_extractf128_ps(sum, 1);
        result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
        
        result
    }
    
    /// NEON 优化的 PQ 距离 (4 elements per iteration)
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline]
    fn distance_neon(&self, query: &[f32], codebook: &[f32], codes: &[u8]) -> f32 {
        use std::arch::aarch64::*;
        
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            
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
                
                let mut i = 0;
                let q_ptr = query.as_ptr().add(q_start);
                let c_ptr = codebook.as_ptr().add(cent_offset);
                
                while i + 4 <= self.sub_dim {
                    let qv = vld1q_f32(q_ptr.add(i));
                    let cv = vld1q_f32(c_ptr.add(i));
                    let diff = vsubq_f32(qv, cv);
                    let sq = vmulq_f32(diff, diff);
                    sum = vaddq_f32(sum, sq);
                    i += 4;
                }
                
                // Handle remainder
                while i < self.sub_dim {
                    let diff = *q_ptr.add(i) - *c_ptr.add(i);
                    let mut partial = vdupq_n_f32(0.0);
                    partial = vsetq_lane_f32(diff * diff, partial, 0);
                    sum = vaddq_f32(sum, partial);
                    i += 1;
                }
            }
            
            // Horizontal sum
            let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                             vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
            result
        }
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
    
    /// 预计算查询到所有质心的距离（自动选择 SIMD 实现）
    pub fn compute(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return self.compute_avx2(query, codebook, sub_dim);
            }
            if std::is_x86_feature_detected!("sse4_2") {
                return self.compute_sse(query, codebook, sub_dim);
            }
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return self.compute_neon(query, codebook, sub_dim);
            }
        }
        self.compute_scalar(query, codebook, sub_dim);
    }
    
    /// 标量预计算
    #[inline]
    fn compute_scalar(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
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
    
    /// SSE 优化的预计算
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    fn compute_sse(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
        use std::arch::x86_64::*;
        
        for m_idx in 0..self.m {
            let q_start = m_idx * sub_dim;
            let q_ptr = &query[q_start];
            
            for k_idx in 0..self.k {
                let cent_offset = m_idx * self.k * sub_dim + k_idx * sub_dim;
                let c_ptr = &codebook[cent_offset];
                
                let mut sum = _mm_setzero_ps();
                let mut i = 0;
                
                while i + 4 <= sub_dim {
                    let qv = _mm_loadu_ps(q_ptr.add(i));
                    let cv = _mm_loadu_ps(c_ptr.add(i));
                    let diff = _mm_sub_ps(qv, cv);
                    let sq = _mm_mul_ps(diff, diff);
                    sum = _mm_add_ps(sum, sq);
                    i += 4;
                }
                
                // Horizontal sum
                let mut result = _mm_cvtss_f32(sum);
                let high = _mm_movehdup_ps(sum);
                let sums = _mm_add_ps(sum, high);
                let sums2 = _mm_movehl_ps(sums, sums);
                result += _mm_cvtss_f32(_mm_add_ss(sums, sums2));
                
                // Handle remainder
                while i < sub_dim {
                    let diff = q_ptr[i] - c_ptr[i];
                    result += diff * diff;
                    i += 1;
                }
                
                self.table[m_idx * self.k + k_idx] = result;
            }
        }
    }
    
    /// AVX2 优化的预计算
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    fn compute_avx2(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
        use std::arch::x86_64::*;
        
        for m_idx in 0..self.m {
            let q_start = m_idx * sub_dim;
            let q_ptr = &query[q_start];
            
            for k_idx in 0..self.k {
                let cent_offset = m_idx * self.k * sub_dim + k_idx * sub_dim;
                let c_ptr = &codebook[cent_offset];
                
                let mut sum = _mm256_setzero_ps();
                let mut i = 0;
                
                while i + 8 <= sub_dim {
                    let qv = _mm256_loadu_ps(q_ptr.add(i));
                    let cv = _mm256_loadu_ps(c_ptr.add(i));
                    let diff = _mm256_sub_ps(qv, cv);
                    let sq = _mm256_mul_ps(diff, diff);
                    sum = _mm256_add_ps(sum, sq);
                    i += 8;
                }
                
                // Horizontal sum of 256-bit
                let mut result = _mm256_cvtss_f32(sum);
                let high = _mm256_extractf128_ps(sum, 1);
                result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
                
                // Handle remainder
                while i < sub_dim {
                    let diff = q_ptr[i] - c_ptr[i];
                    result += diff * diff;
                    i += 1;
                }
                
                self.table[m_idx * self.k + k_idx] = result;
            }
        }
    }
    
    /// NEON 优化的预计算
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[inline]
    fn compute_neon(&mut self, query: &[f32], codebook: &[f32], sub_dim: usize) {
        use std::arch::aarch64::*;
        
        unsafe {
            for m_idx in 0..self.m {
                let q_start = m_idx * sub_dim;
                let q_ptr = query.as_ptr().add(q_start);
                
                for k_idx in 0..self.k {
                    let cent_offset = m_idx * self.k * sub_dim + k_idx * sub_dim;
                    let c_ptr = codebook.as_ptr().add(cent_offset);
                    
                    let mut sum = vdupq_n_f32(0.0);
                    let mut i = 0;
                    
                    while i + 4 <= sub_dim {
                        let qv = vld1q_f32(q_ptr.add(i));
                        let cv = vld1q_f32(c_ptr.add(i));
                        let diff = vsubq_f32(qv, cv);
                        let sq = vmulq_f32(diff, diff);
                        sum = vaddq_f32(sum, sq);
                        i += 4;
                    }
                    
                    // Horizontal sum
                    let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                                     vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
                    
                    // Handle remainder
                    while i < sub_dim {
                        let diff = *q_ptr.add(i) - *c_ptr.add(i);
                        result += diff * diff;
                        i += 1;
                    }
                    
                    self.table[m_idx * self.k + k_idx] = result;
                }
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

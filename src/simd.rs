//! SIMD 距离计算实现
//! 
//! 检测运行时 CPU 并选择最优实现（NEON for ARM, SSE/AVX for x86）

use crate::metrics::Distance;

/// CPU 特性检测
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    NEON,
    SSE,
    AVX2,
    AVX512,
}

/// 获取当前 CPU 支持的最高 SIMD 级别
#[cfg(target_arch = "x86_64")]
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(feature = "simd")]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return SimdLevel::AVX512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return SimdLevel::AVX2;
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return SimdLevel::SSE;
        }
    }
    SimdLevel::Scalar
}

#[cfg(target_arch = "aarch64")]
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(feature = "simd")]
    {
        if std::is_aarch64_feature_detected!("neon") {
            return SimdLevel::NEON;
        }
    }
    SimdLevel::Scalar
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn detect_simd_level() -> SimdLevel {
    SimdLevel::Scalar
}

/// L2 距离计算
pub struct L2DistanceSimd {
    level: SimdLevel,
}

impl L2DistanceSimd {
    pub fn new() -> Self {
        let level = detect_simd_level();
        Self { level }
    }
    
    pub fn level(&self) -> SimdLevel {
        self.level
    }
}

impl Default for L2DistanceSimd {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for L2DistanceSimd {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_distance(a, b)
    }
    
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        l2_batch(a, b, dim)
    }
}

/// L2 距离（自动选择最优实现）
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return l2_avx2(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return l2_sse(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::is_aarch64_feature_detected!("neon") {
            return l2_neon(a, b);
        }
    }
    l2_scalar(a, b)
}

/// L2 距离（标量）
#[inline]
pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// L2 距离（SSE）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm_setzero_ps();
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let va = _mm_loadu_ps(&a[i * 4]);
        let vb = _mm_loadu_ps(&b[i * 4]);
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }
    
    let mut result = _mm_cvtss_f32(sum);
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result.sqrt()
}

/// L2 距离（AVX2）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    for i in 0..chunks {
        let va = _mm256_loadu_ps(&a[i * 8]);
        let vb = _mm256_loadu_ps(&b[i * 8]);
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }
    
    // Horizontal add
    let mut result = _mm256_cvtss_f32(sum);
    // Sum the rest of the 256-bit register
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result.sqrt()
}

/// L2 距离（NEON）
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn l2_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let va = vld1q_f32(&a[i * 4]);
        let vb = vld1q_f32(&b[i * 4]);
        let diff = vsubq_f32(va, vb);
        let sq = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, sq);
    }
    
    let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                     vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
    
    for i in (chunks * 4)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result.sqrt()
}

/// Batch L2 距离（并行 + SIMD 优化）
#[cfg(feature = "parallel")]
pub fn l2_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = vec![0.0f32; na * nb];
    
    result.par_iter_mut().enumerate().for_each(|(idx, r)| {
        let i = idx / nb;
        let j = idx % nb;
        *r = l2_distance(
            &a[i * dim..(i + 1) * dim],
            &b[j * dim..(j + 1) * dim],
        );
    });
    result
}

/// Batch L2 距离（串行版本）
#[cfg(not(feature = "parallel"))]
pub fn l2_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);
    
    for i in 0..na {
        for j in 0..nb {
            result.push(l2_distance(
                &a[i * dim..(i + 1) * dim],
                &b[j * dim..(j + 1) * dim],
            ));
        }
    }
    result
}

/// 高效批量 L2: 一个查询向量 vs 多个库向量
pub fn l2_batch_query_vs_database(query: &[f32], database: &[f32], dim: usize) -> Vec<f32> {
    let nq = query.len() / dim;
    let nb = database.len() / dim;
    let mut result = Vec::with_capacity(nq * nb);
    
    for i in 0..nq {
        let q = &query[i * dim..(i + 1) * dim];
        for j in 0..nb {
            let v = &database[j * dim..(j + 1) * dim];
            result.push(l2_distance(q, v));
        }
    }
    result
}

/// 内积距离计算
pub struct InnerProductSimd {
    level: SimdLevel,
}

impl InnerProductSimd {
    pub fn new() -> Self {
        let level = detect_simd_level();
        Self { level }
    }
}

impl Default for InnerProductSimd {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for InnerProductSimd {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        inner_product(a, b)
    }
    
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        ip_batch(a, b, dim)
    }
}

/// 内积
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Batch 内积（并行 + SIMD 优化）
#[cfg(feature = "parallel")]
pub fn ip_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    use rayon::prelude::*;
    
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = vec![0.0f32; na * nb];
    
    result.par_iter_mut().enumerate().for_each(|(idx, r)| {
        let i = idx / nb;
        let j = idx % nb;
        *r = inner_product(
            &a[i * dim..(i + 1) * dim],
            &b[j * dim..(j + 1) * dim],
        );
    });
    result
}

/// Batch 内积（串行版本）
#[cfg(not(feature = "parallel"))]
pub fn ip_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);
    
    for i in 0..na {
        for j in 0..nb {
            result.push(inner_product(
                &a[i * dim..(i + 1) * dim],
                &b[j * dim..(j + 1) * dim],
            ));
        }
    }
    result
}

/// 高效批量内积: 一个查询向量 vs 多个库向量
pub fn ip_batch_query_vs_database(query: &[f32], database: &[f32], dim: usize) -> Vec<f32> {
    let nq = query.len() / dim;
    let nb = database.len() / dim;
    let mut result = Vec::with_capacity(nq * nb);
    
    for i in 0..nq {
        let q = &query[i * dim..(i + 1) * dim];
        for j in 0..nb {
            let v = &database[j * dim..(j + 1) * dim];
            result.push(inner_product(q, v));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_scalar() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = l2_scalar(&a, &b);
        assert!((dist - 5.196).abs() < 0.01);
    }
    
    #[test]
    fn test_l2_equivalence() {
        let a = vec![1.0; 128];
        let b = vec![0.0; 128];
        let scalar = l2_scalar(&a, &b);
        let simd = l2_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
    }
    
    #[test]
    fn test_inner_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ip = inner_product(&a, &b);
        assert!((ip - 32.0).abs() < 0.01);
    }
}

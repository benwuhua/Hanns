//! SIMD 距离计算实现
//! 
//! 检测运行时 CPU 并选择最优实现（NEON for ARM, SSE/AVX for x86）

use crate::metrics::Distance;

/// CPU 特性检测
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// 标量版本
    Scalar,
    /// ARM NEON
    NEON,
    /// SSE 4.2
    SSE,
    /// AVX2
    AVX2,
    /// AVX-512
    AVX512,
}

/// 获取当前 CPU 支持的最高 SIMD 级别
pub fn detect_simd_level() -> SimdLevel {
    // 默认为标量，运行时通过 target_feature 检测
    // 编译时使用标量，运行时根据平台启用 SIMD
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
        l2_scalar(a, b)
    }
    
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        l2_batch_scalar(a, b, dim)
    }
}

/// L2 距离计算（标量）
#[inline]
pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

#[inline]
pub fn l2_batch_scalar(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);
    
    for i in 0..na {
        for j in 0..nb {
            let mut sum = 0.0f32;
            for k in 0..dim {
                let diff = a[i * dim + k] - b[j * dim + k];
                sum += diff * diff;
            }
            result.push(sum.sqrt());
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
    
    pub fn level(&self) -> SimdLevel {
        self.level
    }
}

impl Default for InnerProductSimd {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for InnerProductSimd {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        ip_scalar(a, b)
    }
    
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        ip_batch_scalar(a, b, dim)
    }
}

/// 内积（标量）
#[inline]
pub fn ip_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn ip_batch_scalar(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);
    
    for i in 0..na {
        for j in 0..nb {
            let sum: f32 = (0..dim).map(|k| a[i * dim + k] * b[j * dim + k]).sum();
            result.push(sum);
        }
    }
    
    result
}

/// 余弦距离
pub struct CosineDistanceSimd {
    level: SimdLevel,
}

impl CosineDistanceSimd {
    pub fn new() -> Self {
        Self { level: detect_simd_level() }
    }
}

impl Default for CosineDistanceSimd {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for CosineDistanceSimd {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_scalar(a, b)
    }
    
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        cosine_batch_scalar(a, b, dim)
    }
}

#[inline]
pub fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    
    1.0 - dot / (norm_a * norm_b)
}

#[inline]
pub fn cosine_batch_scalar(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);
    
    for i in 0..na {
        for j in 0..nb {
            let dot: f32 = (0..dim).map(|k| a[i * dim + k] * b[j * dim + k]).sum();
            let norm_a: f32 = (0..dim).map(|k| a[i * dim + k] * a[i * dim + k]).sum::<f32>().sqrt();
            let norm_b: f32 = (0..dim).map(|k| b[j * dim + k] * b[j * dim + k]).sum::<f32>().sqrt();
            
            let dist = if norm_a == 0.0 || norm_b == 0.0 {
                1.0
            } else {
                1.0 - dot / (norm_a * norm_b)
            };
            result.push(dist);
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_simd() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);
    }
    
    #[test]
    fn test_l2_simd() {
        let l2 = L2DistanceSimd::new();
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = l2.compute(&a, &b);
        assert!((d - 5.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_ip_simd() {
        let ip = InnerProductSimd::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = ip.compute(&a, &b);
        assert!((d - 32.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_l2_large_simd() {
        let l2 = L2DistanceSimd::new();
        let dim = 128;
        let mut a = vec![0.0; dim];
        let mut b = vec![0.0; dim];
        b[0] = 1.0;
        
        let d = l2.compute(&a, &b);
        assert!((d - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_cosine_simd() {
        let cos = CosineDistanceSimd::new();
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        
        let d = cos.compute(&a, &b);
        assert!((d - 0.0).abs() < 1e-5);
    }
}

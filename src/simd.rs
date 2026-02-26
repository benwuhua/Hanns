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
        // NEON is available on all aarch64 CPUs, use runtime detection via std::arch
        if std::arch::is_aarch64_feature_detected!("neon") {
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
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return l2_avx512(a, b);
        }
        if std::is_x86_feature_detected!("avx2") {
            return l2_avx2(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return l2_sse(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return l2_neon(a, b);
        }
    }
    l2_scalar(a, b)
}
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

/// L2 距离（AVX-512）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero512();
    let chunks = a.len() / 16;
    let remainder = a.len() % 16;
    
    for i in 0..chunks {
        let va = _mm512_loadu_ps(&a[i * 16]);
        let vb = _mm512_loadu_ps(&b[i * 16]);
        let diff = _mm512_sub_ps(va, vb);
        let sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }
    
    // Use AVX-512 reduction
    let mut result = _mm512_reduce_add_ps(sum);
    
    // Handle remainder
    for i in (chunks * 16)..a.len() {
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
    unsafe {
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

/// 内积（自动选择最优实现）
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return ip_avx512(a, b);
        }
        if std::is_x86_feature_detected!("avx2") {
            return ip_avx2(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return ip_sse(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return ip_neon(a, b);
        }
    }
    ip_scalar(a, b)
}

/// 内积（标量）
#[inline]
pub fn ip_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 内积（SSE）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn ip_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm_setzero_ps();
    let chunks = a.len() / 4;
    
    for i in 0..chunks {
        let va = _mm_loadu_ps(&a[i * 4]);
        let vb = _mm_loadu_ps(&b[i * 4]);
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }
    
    // Horizontal add
    let mut result = _mm_cvtss_f32(sum);
    let high = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, high);
    let sums2 = _mm_movehl_ps(sums, sums);
    result += _mm_cvtss_f32(_mm_add_ss(sums, sums2));
    
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result += a[i] * b[i];
    }
    result
}

/// 内积（AVX2）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn ip_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let va = _mm256_loadu_ps(&a[i * 8]);
        let vb = _mm256_loadu_ps(&b[i * 8]);
        let prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }
    
    // Horizontal add of 256-bit
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }
    result
}

/// 内积（NEON）
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn ip_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        
        for i in 0..chunks {
            let va = vld1q_f32(&a[i * 4]);
            let vb = vld1q_f32(&b[i * 4]);
            let prod = vmulq_f32(va, vb);
            sum = vaddq_f32(sum, prod);
        }
        
        let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                         vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        
        // Handle remainder
        for i in (chunks * 4)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

/// 内积（AVX-512）
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn ip_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero512();
    let chunks = a.len() / 16;
    
    for i in 0..chunks {
        let va = _mm512_loadu_ps(&a[i * 16]);
        let vb = _mm512_loadu_ps(&b[i * 16]);
        let prod = _mm512_mul_ps(va, vb);
        sum = _mm512_add_ps(sum, prod);
    }
    
    // Horizontal add of 512-bit
    let mut result = _mm512_reduce_add_ps(sum);
    
    // Handle remainder
    for i in (chunks * 16)..a.len() {
        result += a[i] * b[i];
    }
    result
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

// ============================================================
// L1 (Manhattan) 距离计算 - SIMD 优化
// ============================================================

/// L1 距离（曼哈顿距离）- 自动选择最优实现
#[inline]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return l1_avx512(a, b);
        }
        if std::is_x86_feature_detected!("avx2") {
            return l1_avx2(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return l1_sse(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return l1_neon(a, b);
        }
    }
    l1_scalar(a, b)
}

/// L1 距离（标量参考实现）
#[inline]
pub fn l1_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// L1 距离（SSE）- 4 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l1_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm_setzero_ps();
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let va = _mm_loadu_ps(&a[i * 4]);
        let vb = _mm_loadu_ps(&b[i * 4]);
        let diff = _mm_sub_ps(va, vb);
        // _mm_abs_ps is not available in SSE, use max-min trick
        let abs_diff = _mm_max_ps(diff, _mm_neg_ps(diff));
        // _mm_neg_ps: negate using xor with sign bit
        let neg_diff = _mm_xor_ps(diff, _mm_set1_ps(-0.0));
        let abs_diff = _mm_max_ps(diff, neg_diff);
        sum = _mm_add_ps(sum, abs_diff);
    }
    
    // Horizontal add
    let mut result = _mm_cvtss_f32(sum);
    let high = _mm_movehl_ps(sum, sum);
    result += _mm_cvtss_f32(high);
    let mid = _mm_movehdup_ps(sum);
    result += _mm_cvtss_f32(mid);
    
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result += (a[i] - b[i]).abs();
    }
    result
}

/// L1 距离（AVX2）- 8 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l1_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    for i in 0..chunks {
        let va = _mm256_loadu_ps(&a[i * 8]);
        let vb = _mm256_loadu_ps(&b[i * 8]);
        let diff = _mm256_sub_ps(va, vb);
        // Absolute value: max with negation
        let neg = _mm256_xor_ps(diff, _mm256_set1_ps(-0.0));
        let abs_diff = _mm256_max_ps(diff, neg);
        sum = _mm256_add_ps(sum, abs_diff);
    }
    
    // Sum 256-bit register
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result += (a[i] - b[i]).abs();
    }
    result
}

/// L1 距离（AVX-512）- 16 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l1_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero512();
    let chunks = a.len() / 16;
    let remainder = a.len() % 16;
    
    for i in 0..chunks {
        let va = _mm512_loadu_ps(&a[i * 16]);
        let vb = _mm512_loadu_ps(&b[i * 16]);
        let diff = _mm512_sub_ps(va, vb);
        // AVX-512 has _mm512_abs_ps
        let abs_diff = _mm512_abs_ps(diff);
        sum = _mm512_add_ps(sum, abs_diff);
    }
    
    // AVX-512 reduction
    let mut result = _mm512_reduce_add_ps(sum);
    
    // Handle remainder
    for i in (chunks * 16)..a.len() {
        result += (a[i] - b[i]).abs();
    }
    result
}

/// L1 距离（NEON）- 4 元素并行
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn l1_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;
        
        for i in 0..chunks {
            let va = vld1q_f32(&a[i * 4]);
            let vb = vld1q_f32(&b[i * 4]);
            let diff = vsubq_f32(va, vb);
            let abs_diff = vabsq_f32(diff);
            sum = vaddq_f32(sum, abs_diff);
        }
        
        // Horizontal add
        let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                         vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        
        for i in (chunks * 4)..a.len() {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}

// ============================================================
// Linf (Chebyshev) 距离计算 - SIMD 优化
// ============================================================

/// Linf 距离（切比雪夫距离）- 自动选择最优实现
#[inline]
pub fn linf_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return linf_avx512(a, b);
        }
        if std::is_x86_feature_detected!("avx2") {
            return linf_avx2(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return linf_sse(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return linf_neon(a, b);
        }
    }
    linf_scalar(a, b)
}

/// Linf 距离（标量参考实现）
#[inline]
pub fn linf_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |max, v| max.max(v))
}

/// Linf 距离（SSE）- 4 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn linf_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut max_val = _mm_setzero_ps();
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    for i in 0..chunks {
        let va = _mm_loadu_ps(&a[i * 4]);
        let vb = _mm_loadu_ps(&b[i * 4]);
        let diff = _mm_sub_ps(va, vb);
        let abs_diff = _mm_max_ps(diff, _mm_xor_ps(diff, _mm_set1_ps(-0.0)));
        max_val = _mm_max_ps(max_val, abs_diff);
    }
    
    // Horizontal max
    let mut result = _mm_cvtss_f32(max_val);
    let high = _mm_movehl_ps(max_val, max_val);
    result = result.max(_mm_cvtss_f32(high));
    let mid = _mm_movehdup_ps(max_val);
    result = result.max(_mm_cvtss_f32(mid));
    
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result = result.max((a[i] - b[i]).abs());
    }
    result
}

/// Linf 距离（AVX2）- 8 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn linf_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut max_val = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    for i in 0..chunks {
        let va = _mm256_loadu_ps(&a[i * 8]);
        let vb = _mm256_loadu_ps(&b[i * 8]);
        let diff = _mm256_sub_ps(va, vb);
        let neg = _mm256_xor_ps(diff, _mm256_set1_ps(-0.0));
        let abs_diff = _mm256_max_ps(diff, neg);
        max_val = _mm256_max_ps(max_val, abs_diff);
    }
    
    // Max across 256-bit register
    let mut result = _mm256_cvtss_f32(max_val);
    let high = _mm256_extractf128_ps(max_val, 1);
    result = result.max(_mm_cvtss_f32(_mm_max_ps(high, _mm256_castps256to128(max_val))));
    
    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result = result.max((a[i] - b[i]).abs());
    }
    result
}

/// Linf 距离（AVX-512）- 16 元素并行
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn linf_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut max_val = _mm512_setzero512();
    let chunks = a.len() / 16;
    let remainder = a.len() % 16;
    
    for i in 0..chunks {
        let va = _mm512_loadu_ps(&a[i * 16]);
        let vb = _mm512_loadu_ps(&b[i * 16]);
        let diff = _mm512_sub_ps(va, vb);
        let abs_diff = _mm512_abs_ps(diff);
        max_val = _mm512_max_ps(max_val, abs_diff);
    }
    
    // AVX-512 reduction for max
    let mut result = _mm512_reduce_max_ps(max_val);
    
    // Handle remainder
    for i in (chunks * 16)..a.len() {
        result = result.max((a[i] - b[i]).abs());
    }
    result
}

/// Linf 距离（NEON）- 4 元素并行
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn linf_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut max_val = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;
        
        for i in 0..chunks {
            let va = vld1q_f32(&a[i * 4]);
            let vb = vld1q_f32(&b[i * 4]);
            let diff = vsubq_f32(va, vb);
            let abs_diff = vabsq_f32(diff);
            max_val = vmaxq_f32(max_val, abs_diff);
        }
        
        // Horizontal max: pairwise max reduction
        let mut result = vgetq_lane_f32(max_val, 0);
        result = result.max(vgetq_lane_f32(max_val, 1));
        result = result.max(vgetq_lane_f32(max_val, 2));
        result = result.max(vgetq_lane_f32(max_val, 3));
        
        for i in (chunks * 4)..a.len() {
            result = result.max((a[i] - b[i]).abs());
        }
        result
    }
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
    
    #[test]
    fn test_l1_scalar() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = l1_scalar(&a, &b);
        assert!((dist - 9.0).abs() < 0.01); // |1-4| + |2-5| + |3-6| = 3+3+3 = 9
    }
    
    #[test]
    fn test_l1_equivalence() {
        let a = vec![1.0; 128];
        let b = vec![0.0; 128];
        let scalar = l1_scalar(&a, &b);
        let simd = l1_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
    }
    
    #[test]
    fn test_l1_128() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (128..256).map(|i| i as f32).collect();
        let scalar = l1_scalar(&a, &b);
        let simd = l1_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
    }
    
    #[test]
    fn test_linf_scalar() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dist = linf_scalar(&a, &b);
        assert!((dist - 3.0).abs() < 0.01); // max(|1-4|, |2-5|, |3-6|) = 3
    }
    
    #[test]
    fn test_linf_equivalence() {
        let a = vec![1.0; 128];
        let b = vec![0.0; 128];
        let scalar = linf_scalar(&a, &b);
        let simd = linf_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
    }
    
    #[test]
    fn test_linf_128() {
        let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let b: Vec<f32> = (128..256).map(|i| i as f32).collect();
        let scalar = linf_scalar(&a, &b);
        let simd = linf_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
    }
    
    #[test]
    fn test_linf_mixed() {
        // Test with mixed positive/negative values
        let a: Vec<f32> = vec![-10.0, 5.0, 3.0, 100.0];
        let b: Vec<f32> = vec![20.0, -15.0, 3.0, 50.0];
        let scalar = linf_scalar(&a, &b);
        let simd = linf_distance(&a, &b);
        assert!((scalar - simd).abs() < 1e-5);
        // max(|-30|, |20|, |0|, |50|) = 50
        assert!((scalar - 50.0).abs() < 0.01);
    }
}

/// Binary distance functions - Hamming and Jaccard
/// Optimized with SIMD POPCNT instructions where available

/// Hamming distance for binary vectors (u8 slices)
/// Returns the number of differing bits
pub fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("popcnt") {
            return unsafe { hamming_popcnt(a, b) };
        }
    }
    
    // Scalar fallback
    hamming_scalar(a, b)
}

/// Scalar Hamming distance
#[inline]
pub fn hamming_scalar(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x ^ *y).count_ones() as usize)
        .sum()
}

/// POPCNT-optimized Hamming distance (x86_64 only)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn hamming_popcnt(a: &[u8], b: &[u8]) -> usize {
    use std::arch::x86_64::*;
    
    let mut total = 0usize;
    let chunks = a.len() / 32;
    let remainder = a.len() % 32;
    
    // Process 32 bytes at a time using AVX2 if available
    if std::is_x86_feature_detected!("avx2") {
        for i in 0..chunks {
            let offset = i * 32;
            let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
            let vx = _mm256_xor_si256(va, vb);
            
            // Extract each byte and count bits
            let bytes = std::slice::from_raw_parts(
                &vx as *const _ as *const u8,
                32
            );
            for &byte in bytes {
                total += _popcnt64(byte as i64) as usize;
            }
        }
    } else {
        // SSE or scalar chunks
        for i in 0..chunks {
            let offset = i * 32;
            for j in 0..32 {
                total += ((a[offset + j] ^ b[offset + j]).count_ones()) as usize;
            }
        }
    }
    
    // Remainder
    let start = chunks * 32;
    for i in start..a.len() {
        total += ((a[i] ^ b[i]).count_ones()) as usize;
    }
    
    total
}

/// Jaccard similarity for binary vectors
/// Returns intersection / union
pub fn jaccard_similarity(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let (intersection, union) = jaccard_counts(a, b);
    
    if union == 0 {
        1.0 // Both empty = identical
    } else {
        intersection as f32 / union as f32
    }
}

/// Jaccard distance (1 - similarity)
pub fn jaccard_distance(a: &[u8], b: &[u8]) -> f32 {
    1.0 - jaccard_similarity(a, b)
}

/// Compute intersection and union counts for Jaccard
#[inline]
pub fn jaccard_counts(a: &[u8], b: &[u8]) -> (usize, usize) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("popcnt") {
            return unsafe { jaccard_counts_popcnt(a, b) };
        }
    }
    
    jaccard_counts_scalar(a, b)
}

/// Scalar Jaccard counts
#[inline]
pub fn jaccard_counts_scalar(a: &[u8], b: &[u8]) -> (usize, usize) {
    let mut intersection = 0usize;
    let mut union_count = 0usize;
    
    for (x, y) in a.iter().zip(b.iter()) {
        let ix = *x;
        let iy = *y;
        intersection += (ix & iy).count_ones() as usize;
        union_count += (ix | iy).count_ones() as usize;
    }
    
    (intersection, union_count)
}

/// POPCNT-optimized Jaccard counts (x86_64 only)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn jaccard_counts_popcnt(a: &[u8], b: &[u8]) -> (usize, usize) {
    use std::arch::x86_64::*;
    
    let mut intersection = 0usize;
    let mut union_count = 0usize;
    
    let chunks = a.len() / 8;
    let remainder = a.len() % 8;
    
    // Process 8 bytes at a time
    for i in 0..chunks {
        let offset = i * 8;
        let va: u64 = *(a.as_ptr().add(offset) as *const u64);
        let vb: u64 = *(b.as_ptr().add(offset) as *const u64);
        
        let v_and = va & vb;
        let v_or = va | vb;
        
        intersection += _popcnt64(v_and as i64) as usize;
        union_count += _popcnt64(v_or as i64) as usize;
    }
    
    // Remainder
    let start = chunks * 8;
    for i in start..a.len() {
        let ix = a[i];
        let iy = b[i];
        intersection += (ix & iy).count_ones() as usize;
        union_count += (ix | iy).count_ones() as usize;
    }
    
    (intersection, union_count)
}

#[cfg(test)]
mod binary_tests {
    use super::*;
    
    #[test]
    fn test_hamming_basic() {
        let a = vec![0b00001111u8, 0b11110000];
        let b = vec![0b00001111u8, 0b11110000];
        assert_eq!(hamming_distance(&a, &b), 0);
        
        let c = vec![0b11110000u8, 0b00001111];
        assert_eq!(hamming_distance(&a, &c), 16);
    }
    
    #[test]
    fn test_jaccard_basic() {
        // 0b00001111 & 0b00000111 = 0b00000111 (3 bits)
        // 0b00001111 | 0b00000111 = 0b00001111 (4 bits)
        let a = vec![0b00001111u8];
        let b = vec![0b00000111u8];
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.75).abs() < 0.01); // 3/4
        
        let dist = jaccard_distance(&a, &b);
        assert!((dist - 0.25).abs() < 0.01); // 1 - 3/4
    }
    
    #[test]
    fn test_hamming_large() {
        // Test with larger vectors to exercise SIMD path
        let n = 256;
        let a: Vec<u8> = (0..n).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| (i ^ 0xFF) as u8).collect();
        
        let simd_dist = hamming_distance(&a, &b);
        let scalar_dist = hamming_scalar(&a, &b);
        
        assert_eq!(simd_dist, scalar_dist);
        assert_eq!(simd_dist, n * 8); // All bits differ
    }
    
    #[test]
    fn test_jaccard_large() {
        let n = 256;
        let a: Vec<u8> = vec![0xAA; n]; // 10101010
        let b: Vec<u8> = vec![0x55; n]; // 01010101
        
        let (intersection, union) = jaccard_counts(&a, &b);
        let (int_scalar, uni_scalar) = jaccard_counts_scalar(&a, &b);
        
        assert_eq!(intersection, int_scalar);
        assert_eq!(union, uni_scalar);
        assert_eq!(intersection, 0); // No overlapping bits
        assert_eq!(union, n * 8); // All bits set in union
    }
}

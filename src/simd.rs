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
    assert_eq!(a.len(), b.len(), "l2_distance requires equal lengths");
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

/// L2 平方距离（避免 sqrt，更快，用于最近邻搜索）
#[inline]
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_distance_sq requires equal lengths");
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return l2_avx512_sq(a, b);
        }
        if std::is_x86_feature_detected!("avx2") {
            return l2_avx2_sq(a, b);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return l2_sse_sq(a, b);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return l2_neon_sq(a, b);
        }
    }
    l2_scalar_sq(a, b)
}

/// L2 平方距离（原始指针版本）
///
/// # Safety
/// - `a` 和 `b` 必须分别指向至少 `dim` 个有效 `f32`
/// - 两段内存必须允许只读访问
#[inline]
pub unsafe fn l2_distance_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return l2_avx512_sq_ptr(a, b, dim);
        }
        if std::is_x86_feature_detected!("avx2") {
            return l2_avx2_sq_ptr(a, b, dim);
        }
        if std::is_x86_feature_detected!("sse4_2") {
            return l2_sse_sq_ptr(a, b, dim);
        }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return l2_neon_sq_ptr(a, b, dim);
        }
    }
    l2_scalar_sq_ptr(a, b, dim)
}

#[inline]
pub fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[inline]
pub fn l2_scalar_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

/// 标量 L2 平方距离（原始指针版本）
///
/// 手工 8 路展开，避免热点路径中的切片构造和 bounds check。
///
/// # Safety
/// - `a` 和 `b` 必须分别指向至少 `dim` 个有效 `f32`
#[inline(always)]
pub unsafe fn l2_scalar_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;
    let mut i = 0usize;

    while i + 8 <= dim {
        let d0 = *a.add(i) - *b.add(i);
        let d1 = *a.add(i + 1) - *b.add(i + 1);
        let d2 = *a.add(i + 2) - *b.add(i + 2);
        let d3 = *a.add(i + 3) - *b.add(i + 3);
        let d4 = *a.add(i + 4) - *b.add(i + 4);
        let d5 = *a.add(i + 5) - *b.add(i + 5);
        let d6 = *a.add(i + 6) - *b.add(i + 6);
        let d7 = *a.add(i + 7) - *b.add(i + 7);
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
        i += 8;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    while i < dim {
        let d = *a.add(i) - *b.add(i);
        sum += d * d;
        i += 1;
    }
    sum
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

/// L2 平方距离（SSE）- 避免 sqrt
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_sse_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm_setzero_ps();
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let va = _mm_loadu_ps(&a[i * 4]);
        let vb = _mm_loadu_ps(&b[i * 4]);
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    // Horizontal add
    let mut result = _mm_cvtss_f32(sum);
    let high = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, high);
    let sums2 = _mm_movehl_ps(sums, sums);
    result += _mm_cvtss_f32(_mm_add_ss(sums, sums2));

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn l2_sse_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm_setzero_ps();
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.add(offset));
        let vb = _mm_loadu_ps(b.add(offset));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    let mut result = _mm_cvtss_f32(sum);
    let high = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, high);
    let sums2 = _mm_movehl_ps(sums, sums);
    result += _mm_cvtss_f32(_mm_add_ss(sums, sums2));

    for i in (chunks * 4)..dim {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
    }
    result
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

/// L2 平方距离（AVX2）- 避免 sqrt
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_avx2_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(&a[i * 8]);
        let vb = _mm256_loadu_ps(&b[i * 8]);
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    // Better horizontal add for AVX2
    let result = horizontal_sum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn l2_avx2_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    let mut result = horizontal_sum_avx2(sum);
    for i in (chunks * 8)..dim {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
    }
    result
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
unsafe fn l2_neon_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);
    }

    let mut result = vgetq_lane_f32(sum, 0)
        + vgetq_lane_f32(sum, 1)
        + vgetq_lane_f32(sum, 2)
        + vgetq_lane_f32(sum, 3);
    for i in (chunks * 4)..dim {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
    }
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn horizontal_sum_avx2(sum: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(lo, hi);

        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(final_sum)
    }
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

/// L2 平方距离（AVX-512）- 避免 sqrt
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
fn l2_avx512_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero512();
    let chunks = a.len() / 16;

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
    result
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn l2_avx512_sq_ptr(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);
    for i in (chunks * 16)..dim {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
    }
    result
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

        let mut result = vgetq_lane_f32(sum, 0)
            + vgetq_lane_f32(sum, 1)
            + vgetq_lane_f32(sum, 2)
            + vgetq_lane_f32(sum, 3);

        for i in (chunks * 4)..a.len() {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        result.sqrt()
    }
}

/// L2 平方距离（NEON）- 避免 sqrt
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
fn l2_neon_sq(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let va = vld1q_f32(&a[i * 4]);
            let vb = vld1q_f32(&b[i * 4]);
            let diff = vsubq_f32(va, vb);
            let sq = vmulq_f32(diff, diff);
            sum = vaddq_f32(sum, sq);
        }

        let mut result = vgetq_lane_f32(sum, 0)
            + vgetq_lane_f32(sum, 1)
            + vgetq_lane_f32(sum, 2)
            + vgetq_lane_f32(sum, 3);

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
        result
    }
}

/// Batch L2 距离（并行 + SIMD 批量优化）
#[cfg(feature = "parallel")]
pub fn l2_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = vec![0.0f32; na * nb];

    // 使用批量计算优化：每次处理 4 个数据库向量
    // 注意：l2_batch_4 返回的是平方距离，需要开方
    result.par_chunks_mut(nb).enumerate().for_each(|(i, row)| {
        let query = &a[i * dim..(i + 1) * dim];

        // 每 4 个向量一组进行批量计算
        // OPT-006: 添加显式边界检查，避免并行环境下的竞态条件
        let mut j = 0;
        while j + 4 <= nb {
            let end_idx = (j + 4) * dim;
            // 安全检查：确保不会越界
            if end_idx > b.len() {
                break;
            }
            let dists_sq = l2_batch_4(
                query,
                &b[j * dim..(j + 1) * dim],
                &b[(j + 1) * dim..(j + 2) * dim],
                &b[(j + 2) * dim..(j + 3) * dim],
                &b[(j + 3) * dim..(j + 4) * dim],
            );
            // 对平方距离开方
            row[j] = dists_sq[0].sqrt();
            row[j + 1] = dists_sq[1].sqrt();
            row[j + 2] = dists_sq[2].sqrt();
            row[j + 3] = dists_sq[3].sqrt();
            j += 4;
        }

        // 处理剩余的向量
        for k in j..nb {
            let end_vec = (k + 1) * dim;
            if end_vec > b.len() {
                break;
            }
            row[k] = l2_distance(query, &b[k * dim..end_vec]);
        }
    });
    result
}

/// Batch L2 距离（串行版本 + SIMD 批量优化）
#[cfg(not(feature = "parallel"))]
pub fn l2_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);

    for i in 0..na {
        let query = &a[i * dim..(i + 1) * dim];

        // 每 4 个向量一组进行批量计算
        // 注意：l2_batch_4 返回的是平方距离，需要开方
        let mut j = 0;
        while j + 4 <= nb {
            let dists_sq = l2_batch_4(
                query,
                &b[j * dim..(j + 1) * dim],
                &b[(j + 1) * dim..(j + 2) * dim],
                &b[(j + 2) * dim..(j + 3) * dim],
                &b[(j + 3) * dim..(j + 4) * dim],
            );
            // 对平方距离开方
            result.push(dists_sq[0].sqrt());
            result.push(dists_sq[1].sqrt());
            result.push(dists_sq[2].sqrt());
            result.push(dists_sq[3].sqrt());
            j += 4;
        }

        // 处理剩余的向量
        for k in j..nb {
            result.push(l2_distance(query, &b[k * dim..(k + 1) * dim]));
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

// ============================================================
// 批量距离计算优化 (Batch-4 SIMD)
// 对标 C++ knowhere 的 fvec_L2sqr_batch_4_avx 和 fvec_inner_product_batch_4_avx
// ============================================================

/// 标量版本：一次计算 1 个查询向量与 4 个数据库向量的 L2 平方距离
#[inline]
pub fn l2_batch_4_scalar(
    query: &[f32],
    db0: &[f32],
    db1: &[f32],
    db2: &[f32],
    db3: &[f32],
) -> [f32; 4] {
    let mut dists = [0.0f32; 4];
    for i in 0..query.len() {
        let d0 = query[i] - db0[i];
        let d1 = query[i] - db1[i];
        let d2 = query[i] - db2[i];
        let d3 = query[i] - db3[i];
        dists[0] += d0 * d0;
        dists[1] += d1 * d1;
        dists[2] += d2 * d2;
        dists[3] += d3 * d3;
    }
    dists
}

/// AVX2 版本：一次计算 1 个查询向量与 4 个数据库向量的 L2 平方距离
/// 使用 FMA (Fused Multiply-Add) 指令加速
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn l2_batch_4_avx2(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    // 主循环：每次处理 8 个元素，使用 FMA 指令
    for i in 0..chunks {
        let offset = i * 8;
        // 加载查询向量 (复用)
        let q = _mm256_loadu_ps(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = _mm256_loadu_ps(db0.add(offset));
        let v1 = _mm256_loadu_ps(db1.add(offset));
        let v2 = _mm256_loadu_ps(db2.add(offset));
        let v3 = _mm256_loadu_ps(db3.add(offset));

        // 计算差值
        let diff0 = _mm256_sub_ps(q, v0);
        let diff1 = _mm256_sub_ps(q, v1);
        let diff2 = _mm256_sub_ps(q, v2);
        let diff3 = _mm256_sub_ps(q, v3);

        // 使用 FMA 计算 diff * diff 并累加：sum = diff * diff + sum
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
    }

    // 水平求和得到 4 个距离值
    let result = [
        horizontal_sum_avx2(sum0),
        horizontal_sum_avx2(sum1),
        horizontal_sum_avx2(sum2),
        horizontal_sum_avx2(sum3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 8;
        let mut scalar_dists = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dists[0] += (q_val - *db0.add(i)).powi(2);
            scalar_dists[1] += (q_val - *db1.add(i)).powi(2);
            scalar_dists[2] += (q_val - *db2.add(i)).powi(2);
            scalar_dists[3] += (q_val - *db3.add(i)).powi(2);
        }
        return [
            result[0] + scalar_dists[0],
            result[1] + scalar_dists[1],
            result[2] + scalar_dists[2],
            result[3] + scalar_dists[3],
        ];
    }

    result
}

/// ARM NEON 版本：一次计算 1 个查询向量与 4 个数据库向量的 L2 平方距离
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn l2_batch_4_neon(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks = dim / 4;
    let remainder = dim % 4;

    // 主循环：每次处理 4 个元素
    for i in 0..chunks {
        let offset = i * 4;
        // 加载查询向量 (复用)
        let q = vld1q_f32(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = vld1q_f32(db0.add(offset));
        let v1 = vld1q_f32(db1.add(offset));
        let v2 = vld1q_f32(db2.add(offset));
        let v3 = vld1q_f32(db3.add(offset));

        // 计算差值
        let diff0 = vsubq_f32(q, v0);
        let diff1 = vsubq_f32(q, v1);
        let diff2 = vsubq_f32(q, v2);
        let diff3 = vsubq_f32(q, v3);

        // 计算平方并累加
        sum0 = vmlaq_f32(sum0, diff0, diff0);
        sum1 = vmlaq_f32(sum1, diff1, diff1);
        sum2 = vmlaq_f32(sum2, diff2, diff2);
        sum3 = vmlaq_f32(sum3, diff3, diff3);
    }

    // 水平求和
    let result = [
        vgetq_lane_f32(sum0, 0)
            + vgetq_lane_f32(sum0, 1)
            + vgetq_lane_f32(sum0, 2)
            + vgetq_lane_f32(sum0, 3),
        vgetq_lane_f32(sum1, 0)
            + vgetq_lane_f32(sum1, 1)
            + vgetq_lane_f32(sum1, 2)
            + vgetq_lane_f32(sum1, 3),
        vgetq_lane_f32(sum2, 0)
            + vgetq_lane_f32(sum2, 1)
            + vgetq_lane_f32(sum2, 2)
            + vgetq_lane_f32(sum2, 3),
        vgetq_lane_f32(sum3, 0)
            + vgetq_lane_f32(sum3, 1)
            + vgetq_lane_f32(sum3, 2)
            + vgetq_lane_f32(sum3, 3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 4;
        let mut scalar_dists = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dists[0] += (q_val - *db0.add(i)).powi(2);
            scalar_dists[1] += (q_val - *db1.add(i)).powi(2);
            scalar_dists[2] += (q_val - *db2.add(i)).powi(2);
            scalar_dists[3] += (q_val - *db3.add(i)).powi(2);
        }
        return [
            result[0] + scalar_dists[0],
            result[1] + scalar_dists[1],
            result[2] + scalar_dists[2],
            result[3] + scalar_dists[3],
        ];
    }

    result
}

/// AVX512 版本：一次计算 1 个查询向量与 4 个数据库向量的 L2 平方距离
/// 使用 AVX512 FMA 指令加速，每次处理 16 个元素
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn l2_batch_4_avx512(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    let chunks = dim / 16;
    let remainder = dim % 16;

    // 主循环：每次处理 16 个元素，使用 FMA 指令
    for i in 0..chunks {
        let offset = i * 16;
        // 加载查询向量 (复用)
        let q = _mm512_loadu_ps(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = _mm512_loadu_ps(db0.add(offset));
        let v1 = _mm512_loadu_ps(db1.add(offset));
        let v2 = _mm512_loadu_ps(db2.add(offset));
        let v3 = _mm512_loadu_ps(db3.add(offset));

        // 计算差值
        let diff0 = _mm512_sub_ps(q, v0);
        let diff1 = _mm512_sub_ps(q, v1);
        let diff2 = _mm512_sub_ps(q, v2);
        let diff3 = _mm512_sub_ps(q, v3);

        // 使用 FMA 计算 diff * diff 并累加：sum = diff * diff + sum
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
    }

    // 水平求和得到 4 个距离值
    let result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 16;
        let mut scalar_dists = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dists[0] += (q_val - *db0.add(i)).powi(2);
            scalar_dists[1] += (q_val - *db1.add(i)).powi(2);
            scalar_dists[2] += (q_val - *db2.add(i)).powi(2);
            scalar_dists[3] += (q_val - *db3.add(i)).powi(2);
        }
        return [
            result[0] + scalar_dists[0],
            result[1] + scalar_dists[1],
            result[2] + scalar_dists[2],
            result[3] + scalar_dists[3],
        ];
    }

    result
}

/// 批量 L2 平方距离：自动选择最优实现
pub fn l2_batch_4(query: &[f32], db0: &[f32], db1: &[f32], db2: &[f32], db3: &[f32]) -> [f32; 4] {
    assert_eq!(
        query.len(),
        db0.len(),
        "query.len() ({}) != db0.len() ({})",
        query.len(),
        db0.len()
    );
    assert_eq!(
        query.len(),
        db1.len(),
        "query.len() ({}) != db1.len() ({})",
        query.len(),
        db1.len()
    );
    assert_eq!(
        query.len(),
        db2.len(),
        "query.len() ({}) != db2.len() ({})",
        query.len(),
        db2.len()
    );
    assert_eq!(
        query.len(),
        db3.len(),
        "query.len() ({}) != db3.len() ({})",
        query.len(),
        db3.len()
    );
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let dim = query.len();
        // AVX512 优先：每次处理 16 个元素
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            unsafe {
                return l2_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        // AVX2 + FMA: 每次处理 8 个元素
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return l2_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        let dim = query.len();
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return l2_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
    }

    l2_batch_4_scalar(query, db0, db1, db2, db3)
}

/// 批量 L2 平方距离（原始指针版本）
///
/// `db_base` 指向连续存储的 4 个数据库向量起点，向量间步长为 `stride`。
///
/// # Safety
/// - `query` 必须指向至少 `dim` 个有效 `f32`
/// - `db_base` 必须指向至少 `stride * 3 + dim` 个有效 `f32`
#[inline]
pub unsafe fn l2_batch_4_ptr(
    query: *const f32,
    db_base: *const f32,
    dim: usize,
    stride: usize,
) -> [f32; 4] {
    let db0 = db_base;
    let db1 = db_base.add(stride);
    let db2 = db_base.add(stride * 2);
    let db3 = db_base.add(stride * 3);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return l2_batch_4_avx512(query, db0, db1, db2, db3, dim);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return l2_batch_4_avx2(query, db0, db1, db2, db3, dim);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return l2_batch_4_neon(query, db0, db1, db2, db3, dim);
        }
    }

    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut d3 = 0.0f32;
    let mut i = 0usize;

    while i + 4 <= dim {
        let q0 = *query.add(i);
        let q1 = *query.add(i + 1);
        let q2 = *query.add(i + 2);
        let q3 = *query.add(i + 3);

        let x00 = q0 - *db0.add(i);
        let x01 = q1 - *db0.add(i + 1);
        let x02 = q2 - *db0.add(i + 2);
        let x03 = q3 - *db0.add(i + 3);
        d0 += x00 * x00 + x01 * x01 + x02 * x02 + x03 * x03;

        let x10 = q0 - *db1.add(i);
        let x11 = q1 - *db1.add(i + 1);
        let x12 = q2 - *db1.add(i + 2);
        let x13 = q3 - *db1.add(i + 3);
        d1 += x10 * x10 + x11 * x11 + x12 * x12 + x13 * x13;

        let x20 = q0 - *db2.add(i);
        let x21 = q1 - *db2.add(i + 1);
        let x22 = q2 - *db2.add(i + 2);
        let x23 = q3 - *db2.add(i + 3);
        d2 += x20 * x20 + x21 * x21 + x22 * x22 + x23 * x23;

        let x30 = q0 - *db3.add(i);
        let x31 = q1 - *db3.add(i + 1);
        let x32 = q2 - *db3.add(i + 2);
        let x33 = q3 - *db3.add(i + 3);
        d3 += x30 * x30 + x31 * x31 + x32 * x32 + x33 * x33;

        i += 4;
    }

    while i < dim {
        let q = *query.add(i);
        let x0 = q - *db0.add(i);
        let x1 = q - *db1.add(i);
        let x2 = q - *db2.add(i);
        let x3 = q - *db3.add(i);
        d0 += x0 * x0;
        d1 += x1 * x1;
        d2 += x2 * x2;
        d3 += x3 * x3;
        i += 1;
    }

    [d0, d1, d2, d3]
}

/// 标量版本：一次计算 1 个查询向量与 4 个数据库向量的内积
#[inline]
pub fn ip_batch_4_scalar(
    query: &[f32],
    db0: &[f32],
    db1: &[f32],
    db2: &[f32],
    db3: &[f32],
) -> [f32; 4] {
    let mut dots = [0.0f32; 4];
    for i in 0..query.len() {
        let q = query[i];
        dots[0] += q * db0[i];
        dots[1] += q * db1[i];
        dots[2] += q * db2[i];
        dots[3] += q * db3[i];
    }
    dots
}

/// AVX2 版本：一次计算 1 个查询向量与 4 个数据库向量的内积
/// 使用 FMA (Fused Multiply-Add) 指令加速
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ip_batch_4_avx2(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;
    let remainder = dim % 8;

    // 主循环：每次处理 8 个元素，使用 FMA 指令
    for i in 0..chunks {
        let offset = i * 8;
        // 加载查询向量 (复用)
        let q = _mm256_loadu_ps(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = _mm256_loadu_ps(db0.add(offset));
        let v1 = _mm256_loadu_ps(db1.add(offset));
        let v2 = _mm256_loadu_ps(db2.add(offset));
        let v3 = _mm256_loadu_ps(db3.add(offset));

        // 使用 FMA 计算 q * v 并累加：sum = q * v + sum
        sum0 = _mm256_fmadd_ps(q, v0, sum0);
        sum1 = _mm256_fmadd_ps(q, v1, sum1);
        sum2 = _mm256_fmadd_ps(q, v2, sum2);
        sum3 = _mm256_fmadd_ps(q, v3, sum3);
    }

    // 水平求和得到 4 个内积值
    let result = [
        horizontal_sum_avx2(sum0),
        horizontal_sum_avx2(sum1),
        horizontal_sum_avx2(sum2),
        horizontal_sum_avx2(sum3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 8;
        let mut scalar_dots = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dots[0] += q_val * *db0.add(i);
            scalar_dots[1] += q_val * *db1.add(i);
            scalar_dots[2] += q_val * *db2.add(i);
            scalar_dots[3] += q_val * *db3.add(i);
        }
        return [
            result[0] + scalar_dots[0],
            result[1] + scalar_dots[1],
            result[2] + scalar_dots[2],
            result[3] + scalar_dots[3],
        ];
    }

    result
}

/// ARM NEON 版本：一次计算 1 个查询向量与 4 个数据库向量的内积
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn ip_batch_4_neon(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks = dim / 4;
    let remainder = dim % 4;

    // 主循环：每次处理 4 个元素
    for i in 0..chunks {
        let offset = i * 4;
        // 加载查询向量 (复用)
        let q = vld1q_f32(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = vld1q_f32(db0.add(offset));
        let v1 = vld1q_f32(db1.add(offset));
        let v2 = vld1q_f32(db2.add(offset));
        let v3 = vld1q_f32(db3.add(offset));

        // 使用 FMLA (Fused Multiply-Add) 计算并累加
        sum0 = vmlaq_f32(sum0, q, v0);
        sum1 = vmlaq_f32(sum1, q, v1);
        sum2 = vmlaq_f32(sum2, q, v2);
        sum3 = vmlaq_f32(sum3, q, v3);
    }

    // 水平求和
    let result = [
        vgetq_lane_f32(sum0, 0)
            + vgetq_lane_f32(sum0, 1)
            + vgetq_lane_f32(sum0, 2)
            + vgetq_lane_f32(sum0, 3),
        vgetq_lane_f32(sum1, 0)
            + vgetq_lane_f32(sum1, 1)
            + vgetq_lane_f32(sum1, 2)
            + vgetq_lane_f32(sum1, 3),
        vgetq_lane_f32(sum2, 0)
            + vgetq_lane_f32(sum2, 1)
            + vgetq_lane_f32(sum2, 2)
            + vgetq_lane_f32(sum2, 3),
        vgetq_lane_f32(sum3, 0)
            + vgetq_lane_f32(sum3, 1)
            + vgetq_lane_f32(sum3, 2)
            + vgetq_lane_f32(sum3, 3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 4;
        let mut scalar_dots = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dots[0] += q_val * *db0.add(i);
            scalar_dots[1] += q_val * *db1.add(i);
            scalar_dots[2] += q_val * *db2.add(i);
            scalar_dots[3] += q_val * *db3.add(i);
        }
        return [
            result[0] + scalar_dots[0],
            result[1] + scalar_dots[1],
            result[2] + scalar_dots[2],
            result[3] + scalar_dots[3],
        ];
    }

    result
}

/// AVX512 版本：一次计算 1 个查询向量与 4 个数据库向量的内积
/// 使用 AVX512 FMA 指令加速，每次处理 16 个元素
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn ip_batch_4_avx512(
    query: *const f32,
    db0: *const f32,
    db1: *const f32,
    db2: *const f32,
    db3: *const f32,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();

    let chunks = dim / 16;
    let remainder = dim % 16;

    // 主循环：每次处理 16 个元素，使用 FMA 指令
    for i in 0..chunks {
        let offset = i * 16;
        // 加载查询向量 (复用)
        let q = _mm512_loadu_ps(query.add(offset));

        // 加载 4 个数据库向量
        let v0 = _mm512_loadu_ps(db0.add(offset));
        let v1 = _mm512_loadu_ps(db1.add(offset));
        let v2 = _mm512_loadu_ps(db2.add(offset));
        let v3 = _mm512_loadu_ps(db3.add(offset));

        // 使用 FMA 计算 q * v 并累加：sum = q * v + sum
        sum0 = _mm512_fmadd_ps(q, v0, sum0);
        sum1 = _mm512_fmadd_ps(q, v1, sum1);
        sum2 = _mm512_fmadd_ps(q, v2, sum2);
        sum3 = _mm512_fmadd_ps(q, v3, sum3);
    }

    // 水平求和得到 4 个内积值
    let result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    // 处理余数
    if remainder > 0 {
        let offset = chunks * 16;
        let mut scalar_dots = [0.0f32; 4];
        for i in offset..dim {
            let q_val = *query.add(i);
            scalar_dots[0] += q_val * *db0.add(i);
            scalar_dots[1] += q_val * *db1.add(i);
            scalar_dots[2] += q_val * *db2.add(i);
            scalar_dots[3] += q_val * *db3.add(i);
        }
        return [
            result[0] + scalar_dots[0],
            result[1] + scalar_dots[1],
            result[2] + scalar_dots[2],
            result[3] + scalar_dots[3],
        ];
    }

    result
}

/// 批量内积：自动选择最优实现
pub fn ip_batch_4(query: &[f32], db0: &[f32], db1: &[f32], db2: &[f32], db3: &[f32]) -> [f32; 4] {
    assert_eq!(query.len(), db0.len());
    assert_eq!(query.len(), db1.len());
    assert_eq!(query.len(), db2.len());
    assert_eq!(query.len(), db3.len());
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let dim = query.len();
        // AVX512 优先：每次处理 16 个元素
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            unsafe {
                return ip_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        // AVX2 + FMA: 每次处理 8 个元素
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return ip_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        let dim = query.len();
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                return ip_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
    }

    ip_batch_4_scalar(query, db0, db1, db2, db3)
}

/// 内积距离计算
#[allow(dead_code)]
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
    assert_eq!(a.len(), b.len(), "inner_product requires equal lengths");
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

        let mut result = vgetq_lane_f32(sum, 0)
            + vgetq_lane_f32(sum, 1)
            + vgetq_lane_f32(sum, 2)
            + vgetq_lane_f32(sum, 3);

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

/// Batch 内积（并行 + SIMD 批量优化）
#[cfg(feature = "parallel")]
pub fn ip_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    use rayon::prelude::*;

    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = vec![0.0f32; na * nb];

    // 使用批量计算优化：每次处理 4 个数据库向量
    result.par_chunks_mut(nb).enumerate().for_each(|(i, row)| {
        let query = &a[i * dim..(i + 1) * dim];

        // 每 4 个向量一组进行批量计算
        // OPT-006: 添加显式边界检查，避免并行环境下的竞态条件
        let mut j = 0;
        while j + 4 <= nb {
            let end_idx = (j + 4) * dim;
            // 安全检查：确保不会越界
            if end_idx > b.len() {
                break;
            }
            let dots = ip_batch_4(
                query,
                &b[j * dim..(j + 1) * dim],
                &b[(j + 1) * dim..(j + 2) * dim],
                &b[(j + 2) * dim..(j + 3) * dim],
                &b[(j + 3) * dim..(j + 4) * dim],
            );
            row[j..j + 4].copy_from_slice(&dots);
            j += 4;
        }

        // 处理剩余的向量
        for k in j..nb {
            let end_vec = (k + 1) * dim;
            if end_vec > b.len() {
                break;
            }
            row[k] = inner_product(query, &b[k * dim..end_vec]);
        }
    });
    result
}

/// Batch 内积（串行版本 + SIMD 批量优化）
#[cfg(not(feature = "parallel"))]
pub fn ip_batch(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let na = a.len() / dim;
    let nb = b.len() / dim;
    let mut result = Vec::with_capacity(na * nb);

    for i in 0..na {
        let query = &a[i * dim..(i + 1) * dim];

        // 每 4 个向量一组进行批量计算
        let mut j = 0;
        while j + 4 <= nb {
            let dots = ip_batch_4(
                query,
                &b[j * dim..(j + 1) * dim],
                &b[(j + 1) * dim..(j + 2) * dim],
                &b[(j + 2) * dim..(j + 3) * dim],
                &b[(j + 3) * dim..(j + 4) * dim],
            );
            result.extend_from_slice(&dots);
            j += 4;
        }

        // 处理剩余的向量
        for k in j..nb {
            result.push(inner_product(query, &b[k * dim..(k + 1) * dim]));
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
        let mut result = vgetq_lane_f32(sum, 0)
            + vgetq_lane_f32(sum, 1)
            + vgetq_lane_f32(sum, 2)
            + vgetq_lane_f32(sum, 3);

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
    result = result.max(_mm_cvtss_f32(_mm_max_ps(
        high,
        _mm256_castps256to128(max_val),
    )));

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

    #[test]
    fn test_l2_batch_4_scalar() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let db0 = vec![0.0, 0.0, 0.0, 0.0];
        let db1 = vec![1.0, 1.0, 1.0, 1.0];
        let db2 = vec![2.0, 2.0, 2.0, 2.0];
        let db3 = vec![3.0, 3.0, 3.0, 3.0];

        let dists = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

        // L2^2 distances:
        // db0: (1-0)^2 + (2-0)^2 + (3-0)^2 + (4-0)^2 = 1+4+9+16 = 30
        // db1: (1-1)^2 + (2-1)^2 + (3-1)^2 + (4-1)^2 = 0+1+4+9 = 14
        // db2: (1-2)^2 + (2-2)^2 + (3-2)^2 + (4-2)^2 = 1+0+1+4 = 6
        // db3: (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 = 4+1+0+1 = 6
        assert!((dists[0] - 30.0).abs() < 1e-5);
        assert!((dists[1] - 14.0).abs() < 1e-5);
        assert!((dists[2] - 6.0).abs() < 1e-5);
        assert!((dists[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_batch_4_autoselect() {
        let query: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let db0: Vec<f32> = (0..128).map(|i| (i + 1) as f32).collect();
        let db1: Vec<f32> = (0..128).map(|i| (i + 2) as f32).collect();
        let db2: Vec<f32> = (0..128).map(|i| (i + 3) as f32).collect();
        let db3: Vec<f32> = (0..128).map(|i| (i + 4) as f32).collect();

        let dists_simd = l2_batch_4(&query, &db0, &db1, &db2, &db3);
        let dists_scalar = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

        for i in 0..4 {
            assert!(
                (dists_simd[i] - dists_scalar[i]).abs() < 1e-4,
                "Distance {} mismatch: simd={}, scalar={}",
                i,
                dists_simd[i],
                dists_scalar[i]
            );
        }
    }

    #[test]
    fn test_l2_distance_sq_ptr_matches_slice() {
        let a: Vec<f32> = (0..127).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..127).map(|i| 100.0 - i as f32 * 0.5).collect();

        let expected = l2_distance_sq(&a, &b);
        let actual = unsafe { l2_distance_sq_ptr(a.as_ptr(), b.as_ptr(), a.len()) };

        assert!((expected - actual).abs() < 1e-4);
    }

    #[test]
    fn test_l2_batch_4_ptr_matches_slice() {
        let query: Vec<f32> = (0..130).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..130).map(|i| i as f32 * 0.2).collect();
        let db1: Vec<f32> = (0..130).map(|i| i as f32 * 0.3 + 1.0).collect();
        let db2: Vec<f32> = (0..130).map(|i| i as f32 * 0.4 + 2.0).collect();
        let db3: Vec<f32> = (0..130).map(|i| i as f32 * 0.5 + 3.0).collect();

        let expected = l2_batch_4(&query, &db0, &db1, &db2, &db3);

        let mut packed = Vec::with_capacity(query.len() * 4);
        packed.extend_from_slice(&db0);
        packed.extend_from_slice(&db1);
        packed.extend_from_slice(&db2);
        packed.extend_from_slice(&db3);

        let actual =
            unsafe { l2_batch_4_ptr(query.as_ptr(), packed.as_ptr(), query.len(), query.len()) };

        for i in 0..4 {
            assert!(
                (expected[i] - actual[i]).abs() < 5e-3,
                "lane {} mismatch: expected={}, actual={}",
                i,
                expected[i],
                actual[i]
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_l2_batch_4_avx512() {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // Skip if AVX512 not available
        }

        let query: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let db0: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();
        let db1: Vec<f32> = (0..256).map(|i| (i + 2) as f32).collect();
        let db2: Vec<f32> = (0..256).map(|i| (i + 3) as f32).collect();
        let db3: Vec<f32> = (0..256).map(|i| (i + 4) as f32).collect();

        unsafe {
            let dists_avx512 = l2_batch_4_avx512(
                query.as_ptr(),
                db0.as_ptr(),
                db1.as_ptr(),
                db2.as_ptr(),
                db3.as_ptr(),
                256,
            );
            let dists_scalar = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

            for i in 0..4 {
                assert!(
                    (dists_avx512[i] - dists_scalar[i]).abs() < 1e-4,
                    "AVX512 Distance {} mismatch: avx512={}, scalar={}",
                    i,
                    dists_avx512[i],
                    dists_scalar[i]
                );
            }
        }
    }

    #[test]
    fn test_ip_batch_4_scalar() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let db0 = vec![1.0, 0.0, 0.0, 0.0];
        let db1 = vec![0.0, 1.0, 0.0, 0.0];
        let db2 = vec![0.0, 0.0, 1.0, 0.0];
        let db3 = vec![0.0, 0.0, 0.0, 1.0];

        let dots = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

        // Inner products:
        // db0: 1*1 + 2*0 + 3*0 + 4*0 = 1
        // db1: 1*0 + 2*1 + 3*0 + 4*0 = 2
        // db2: 1*0 + 2*0 + 3*1 + 4*0 = 3
        // db3: 1*0 + 2*0 + 3*0 + 4*1 = 4
        assert!((dots[0] - 1.0).abs() < 1e-5);
        assert!((dots[1] - 2.0).abs() < 1e-5);
        assert!((dots[2] - 3.0).abs() < 1e-5);
        assert!((dots[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_ip_batch_4_autoselect() {
        let query: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let db0: Vec<f32> = (0..128).map(|i| (i * 2) as f32).collect();
        let db1: Vec<f32> = (0..128).map(|i| (i * 3) as f32).collect();
        let db2: Vec<f32> = (0..128).map(|i| (i * 4) as f32).collect();
        let db3: Vec<f32> = (0..128).map(|i| (i * 5) as f32).collect();

        let dots_simd = ip_batch_4(&query, &db0, &db1, &db2, &db3);
        let dots_scalar = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

        for i in 0..4 {
            assert!(
                (dots_simd[i] - dots_scalar[i]).abs() < 1e-3,
                "Inner product {} mismatch: simd={}, scalar={}",
                i,
                dots_simd[i],
                dots_scalar[i]
            );
        }
    }

    #[test]
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    fn test_ip_batch_4_avx512() {
        if !std::is_x86_feature_detected!("avx512f") {
            return; // Skip if AVX512 not available
        }

        let query: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let db0: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();
        let db1: Vec<f32> = (0..256).map(|i| (i * 3) as f32).collect();
        let db2: Vec<f32> = (0..256).map(|i| (i * 4) as f32).collect();
        let db3: Vec<f32> = (0..256).map(|i| (i * 5) as f32).collect();

        unsafe {
            let dots_avx512 = ip_batch_4_avx512(
                query.as_ptr(),
                db0.as_ptr(),
                db1.as_ptr(),
                db2.as_ptr(),
                db3.as_ptr(),
                256,
            );
            let dots_scalar = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);

            for i in 0..4 {
                assert!(
                    (dots_avx512[i] - dots_scalar[i]).abs() < 1e-3,
                    "AVX512 Inner product {} mismatch: avx512={}, scalar={}",
                    i,
                    dots_avx512[i],
                    dots_scalar[i]
                );
            }
        }
    }

    #[test]
    fn test_l2_batch_optimized() {
        // Test that the optimized l2_batch produces same results as naive version
        let dim = 128;
        let na = 3;
        let nb = 8;

        let a: Vec<f32> = (0..na * dim).map(|i| (i % 256) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..nb * dim).map(|i| (i % 128) as f32 * 0.05).collect();

        let result_optimized = l2_batch(&a, &b, dim);

        // Naive reference implementation
        let mut result_naive = Vec::with_capacity(na * nb);
        for i in 0..na {
            for j in 0..nb {
                result_naive.push(l2_distance(
                    &a[i * dim..(i + 1) * dim],
                    &b[j * dim..(j + 1) * dim],
                ));
            }
        }

        assert_eq!(result_optimized.len(), result_naive.len());
        for i in 0..result_optimized.len() {
            assert!(
                (result_optimized[i] - result_naive[i]).abs() < 1e-4,
                "Batch L2 mismatch at index {}: optimized={}, naive={}",
                i,
                result_optimized[i],
                result_naive[i]
            );
        }
    }

    #[test]
    fn test_ip_batch_optimized() {
        // Test that the optimized ip_batch produces same results as naive version
        let dim = 128;
        let na = 3;
        let nb = 8;

        let a: Vec<f32> = (0..na * dim).map(|i| (i % 256) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..nb * dim).map(|i| (i % 128) as f32 * 0.05).collect();

        let result_optimized = ip_batch(&a, &b, dim);

        // Naive reference implementation
        let mut result_naive = Vec::with_capacity(na * nb);
        for i in 0..na {
            for j in 0..nb {
                result_naive.push(inner_product(
                    &a[i * dim..(i + 1) * dim],
                    &b[j * dim..(j + 1) * dim],
                ));
            }
        }

        assert_eq!(result_optimized.len(), result_naive.len());
        for i in 0..result_optimized.len() {
            assert!(
                (result_optimized[i] - result_naive[i]).abs() < 1e-3,
                "Batch IP mismatch at index {}: optimized={}, naive={}",
                i,
                result_optimized[i],
                result_naive[i]
            );
        }
    }

    #[test]
    fn test_l2_distance_sq_requires_equal_lengths() {
        let a = vec![1.0f32; 128];
        let b = vec![1.0f32; 256];

        let result = std::panic::catch_unwind(|| l2_distance_sq(&a, &b));
        assert!(result.is_err(), "mismatched lengths must panic");
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
            let bytes = std::slice::from_raw_parts(&vx as *const _ as *const u8, 32);
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

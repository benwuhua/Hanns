/// Compute dot product between two f32 slices.
/// Falls back to scalar on unsupported architectures or short vectors.
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product_f32 requires equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") && a.len() >= 16 {
            // SAFETY: guarded by runtime feature detection and equal lengths.
            return unsafe { dot_product_avx512(a, b) };
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && a.len() >= 8
        {
            // SAFETY: guarded by runtime feature detection and equal lengths.
            return unsafe { dot_product_avx2_fma(a, b) };
        }
        if std::arch::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: guarded by runtime feature detection and equal lengths.
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && a.len() >= 4 {
            // SAFETY: guarded by runtime feature detection and equal lengths.
            return unsafe { dot_product_neon(a, b) };
        }
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let off = i * 8;
        // SAFETY: off..off+8 is in-bounds; loadu supports unaligned access.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(off)) };
        // SAFETY: off..off+8 is in-bounds; loadu supports unaligned access.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(off)) };
        let prod = _mm256_mul_ps(va, vb);
        acc = _mm256_add_ps(acc, prod);
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut sum4 = _mm_add_ps(lo, hi);
    let shuf1 = _mm_movehdup_ps(sum4);
    sum4 = _mm_add_ps(sum4, shuf1);
    let shuf2 = _mm_movehl_ps(shuf1, sum4);
    let total = _mm_add_ss(sum4, shuf2);
    let mut result = _mm_cvtss_f32(total);

    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let off = i * 8;
        // SAFETY: off..off+8 is in-bounds; loadu supports unaligned access.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(off)) };
        // SAFETY: off..off+8 is in-bounds; loadu supports unaligned access.
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(off)) };
        // fmadd: acc += va * vb
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut sum4 = _mm_add_ps(lo, hi);
    let shuf1 = _mm_movehdup_ps(sum4);
    sum4 = _mm_add_ps(sum4, shuf1);
    let shuf2 = _mm_movehl_ps(shuf1, sum4);
    let total = _mm_add_ss(sum4, shuf2);
    let mut result = _mm_cvtss_f32(total);

    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 16;
    let mut acc = _mm512_setzero_ps();

    for i in 0..chunks {
        let off = i * 16;
        // SAFETY: off..off+16 is in-bounds; loadu supports unaligned access.
        let va = unsafe { _mm512_loadu_ps(a.as_ptr().add(off)) };
        // SAFETY: off..off+16 is in-bounds; loadu supports unaligned access.
        let vb = unsafe { _mm512_loadu_ps(b.as_ptr().add(off)) };
        let prod = _mm512_mul_ps(va, vb);
        acc = _mm512_add_ps(acc, prod);
    }

    // SAFETY: local stack array is valid for unaligned store.
    let mut lanes = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut result: f32 = lanes.iter().sum();

    for i in (chunks * 16)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        // SAFETY: off..off+4 is in-bounds; vld1q accepts unaligned pointers.
        let va = unsafe { vld1q_f32(a.as_ptr().add(off)) };
        // SAFETY: off..off+4 is in-bounds; vld1q accepts unaligned pointers.
        let vb = unsafe { vld1q_f32(b.as_ptr().add(off)) };
        acc = vmlaq_f32(acc, va, vb);
    }

    // SAFETY: local stack array is valid for store.
    let mut lanes = [0.0f32; 4];
    unsafe { vst1q_f32(lanes.as_mut_ptr(), acc) };
    let mut result: f32 = lanes.iter().sum();

    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_matches_scalar_dim_768() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.013).sin()).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.021).cos()).collect();
        let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let simd = dot_product_f32(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-3,
            "scalar={} simd={}",
            scalar,
            simd
        );
    }
}

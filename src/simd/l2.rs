/// Compute squared L2 distance between two f32 slices.
/// Falls back to scalar on non-x86_64 or when avx2 is unavailable.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // SAFETY: guarded by runtime feature detection and equal lengths.
            return unsafe { l2_sq_avx2(a, b) };
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
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
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        acc = _mm256_add_ps(acc, sq);
    }

    // Horizontal sum of 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut sum4 = _mm_add_ps(lo, hi);
    let shuf1 = _mm_movehdup_ps(sum4);
    sum4 = _mm_add_ps(sum4, shuf1);
    let shuf2 = _mm_movehl_ps(shuf1, sum4);
    let total = _mm_add_ss(sum4, shuf2);
    let mut result = _mm_cvtss_f32(total);

    for i in (chunks * 8)..n {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_sq_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| i as f32 * 0.02).collect();
        let scalar: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum();
        let simd = l2_sq(&a, &b);
        assert!(
            (scalar - simd).abs() < 1e-3,
            "scalar={} simd={}",
            scalar,
            simd
        );
    }

    #[test]
    fn test_l2_sq_zero() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = l2_sq(&a, &a);
        assert!(result.abs() < 1e-6, "expected 0, got {}", result);
    }
}

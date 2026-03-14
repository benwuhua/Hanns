//! Half-precision float support (fp16/bf16)
//!
//! 对齐 C++ knowhere 的实现，支持 fp16 (IEEE 754) 和 bf16 (Brain Float)
//!
//! 参考: /Users/ryan/Code/vectorDB/knowhere

/// FP16 (IEEE 754 half-precision, 16-bit)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Fp16(pub u16);

/// BF16 (Brain Float, 16-bit with 8-bit exponent)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Bf16(pub u16);

// Keep the feature flag stable on current toolchains even though std::f16 is
// not available on the stable compiler used by the authority lane.
#[cfg(feature = "std-f16")]
pub type F16 = Fp16;

impl Fp16 {
    /// 从 f32 转换
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        // Use bit manipulation
        let bits = f.to_bits();

        let sign = (bits >> 31) & 1;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exponent == 0 {
            // Zero or subnormal
            if mantissa == 0 {
                // Zero
                return Self((sign << 15) as u16);
            }
            // Subnormal - convert to zero in fp16 for simplicity
            return Self((sign << 15) as u16);
        }

        if exponent == 255 {
            // Infinity or NaN
            let mantissa_nonzero = mantissa != 0;
            if mantissa_nonzero {
                // NaN - preserve mantissa
                let qnan: u16 = 0x7E00 | ((mantissa >> 13) as u16);
                let bits: u32 = (sign << 15) | (qnan as u32);
                return Self(bits as u16);
            } else {
                // Infinity
                let bits: u32 = (sign << 15) | 0x7C00;
                return Self(bits as u16);
            }
        }

        // Normal number
        let new_exp = exponent - 127 + 15;

        if new_exp <= 0 {
            // Underflow - convert to zero
            Self((sign << 15) as u16)
        } else if new_exp >= 31 {
            // Overflow - convert to infinity
            Self(((sign << 15) | 0x7C00) as u16)
        } else {
            // Normal fp16
            let fp16_exp = new_exp as u16;
            let fp16_mantissa = (mantissa >> 13) as u16;
            Self((sign as u16) << 15 | fp16_exp << 10 | fp16_mantissa)
        }
    }
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;

        let sign = ((bits >> 15) & 1) as u32;
        let exponent = ((bits >> 10) & 0x1F) as u32;
        let mantissa = (bits & 0x3FF) as u32;

        if exponent == 0 {
            if mantissa == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Subnormal
            return f32::from_bits((sign << 31) | (mantissa << 13));
        }

        if exponent == 31 {
            // Infinity or NaN
            if mantissa == 0 {
                return f32::from_bits((sign << 31) | 0x7F800000);
            } else {
                // NaN
                return f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 13));
            }
        }

        // Normal number
        let f32_exp = exponent + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }

    /// 转换为原始 u16 位
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// 从原始 u16 位创建
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// 零值
    pub const ZERO: Self = Self(0);
    /// 1.0
    pub const ONE: Self = Self(0x3C00);
}

impl Bf16 {
    /// 从 f32 转换（截断尾数）
    #[inline]
    pub fn from_f32(f: f32) -> Self {
        let bits = f.to_bits();
        // 直接取高 16 位
        Self((bits >> 16) as u16)
    }

    /// 转换为 f32
    #[inline]
    pub fn to_f32(self) -> f32 {
        // 扩展到 32 位，补零尾数
        f32::from_bits((self.0 as u32) << 16)
    }

    /// 转换为原始 u16 位
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// 从原始 u16 位创建
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// 零值
    pub const ZERO: Self = Self(0);
    /// 1.0
    pub const ONE: Self = Self(0x3F80);
}

impl From<f32> for Fp16 {
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

impl From<Fp16> for f32 {
    fn from(h: Fp16) -> Self {
        h.to_f32()
    }
}

impl From<f32> for Bf16 {
    fn from(f: f32) -> Self {
        Self::from_f32(f)
    }
}

impl From<Bf16> for f32 {
    fn from(b: Bf16) -> Self {
        b.to_f32()
    }
}

/// 批量转换 f32 数组到 fp16
pub fn f32_to_fp16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&f| Fp16::from_f32(f).to_bits()).collect()
}

/// 批量转换 fp16 到 f32 数组
pub fn fp16_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&b| Fp16::from_bits(b).to_f32()).collect()
}

/// 批量转换 f32 数组到 bf16
pub fn f32_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&f| Bf16::from_f32(f).to_bits()).collect()
}

/// 批量转换 bf16 到 f32 数组
pub fn bf16_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&b| Bf16::from_bits(b).to_f32()).collect()
}

/// FP16 L2 距离 (标量实现)
#[inline]
pub fn fp16_l2_scalar(a: &[u16], b: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let a_f = Fp16::from_bits(a[i]).to_f32();
        let b_f = Fp16::from_bits(b[i]).to_f32();
        let diff = a_f - b_f;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// BF16 L2 距离 (标量实现)
#[inline]
pub fn bf16_l2_scalar(a: &[u16], b: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let a_f = Bf16::from_bits(a[i]).to_f32();
        let b_f = Bf16::from_bits(b[i]).to_f32();
        let diff = a_f - b_f;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// FP16 L2 距离 - 自动选择 SIMD 或标量实现
#[inline]
pub fn fp16_l2(a: &[u16], b: &[u16]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { fp16_l2_avx2(a, b) };
        }
    }
    fp16_l2_scalar(a, b)
}

/// BF16 L2 距离 - 自动选择 SIMD 或标量实现
#[inline]
pub fn bf16_l2(a: &[u16], b: &[u16]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { bf16_l2_avx2(a, b) };
        }
    }
    bf16_l2_scalar(a, b)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn load_fp16x8_as_m256(ptr: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(ptr as *const __m128i);
    _mm256_cvtph_ps(raw)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn load_bf16x8_as_m256(ptr: *const u16) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let raw = _mm_loadu_si128(ptr as *const __m128i);
    let expanded = _mm256_cvtepu16_epi32(raw);
    let bits = _mm256_slli_epi32(expanded, 16);
    _mm256_castsi256_ps(bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
unsafe fn load_fp16x16_as_m512(ptr: *const u16) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let lo = load_fp16x8_as_m256(ptr);
    let hi = load_fp16x8_as_m256(ptr.add(8));
    let mut tmp = [0.0f32; 16];
    _mm256_storeu_ps(tmp.as_mut_ptr(), lo);
    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), hi);
    _mm512_loadu_ps(tmp.as_ptr())
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
unsafe fn load_bf16x16_as_m512(ptr: *const u16) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let raw = _mm256_loadu_si256(ptr as *const __m256i);
    let expanded = _mm512_cvtepu16_epi32(raw);
    let bits = _mm512_slli_epi32(expanded, 16);
    _mm512_castsi512_ps(bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn horizontal_sum_avx2_fp16(sum: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

/// FP16 L2 距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn fp16_l2_avx2(a: &[u16], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = load_fp16x8_as_m256(a.as_ptr().add(offset));
        let vb = load_fp16x8_as_m256(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    let mut result = horizontal_sum_avx2_fp16(sum);
    for i in (chunks * 8)..a.len() {
        let diff = Fp16::from_bits(a[i]).to_f32() - Fp16::from_bits(b[i]).to_f32();
        result += diff * diff;
    }
    result.sqrt()
}

/// BF16 L2 距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn bf16_l2_avx2(a: &[u16], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = load_bf16x8_as_m256(a.as_ptr().add(offset));
        let vb = load_bf16x8_as_m256(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    let mut result = horizontal_sum_avx2_fp16(sum);
    for i in (chunks * 8)..a.len() {
        let diff = Bf16::from_bits(a[i]).to_f32() - Bf16::from_bits(b[i]).to_f32();
        result += diff * diff;
    }
    result.sqrt()
}

/// FP16 内积 (标量)
#[inline]
pub fn fp16_ip_scalar(a: &[u16], b: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let a_f = Fp16::from_bits(a[i]).to_f32();
        let b_f = Fp16::from_bits(b[i]).to_f32();
        sum += a_f * b_f;
    }
    sum
}

/// BF16 内积 (标量)
#[inline]
pub fn bf16_ip_scalar(a: &[u16], b: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let a_f = Bf16::from_bits(a[i]).to_f32();
        let b_f = Bf16::from_bits(b[i]).to_f32();
        sum += a_f * b_f;
    }
    sum
}

/// FP16 内积 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn fp16_ip_avx2(a: &[u16], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = load_fp16x8_as_m256(a.as_ptr().add(offset));
        let vb = load_fp16x8_as_m256(b.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    let mut result = horizontal_sum_avx2_fp16(sum);
    for i in (chunks * 8)..a.len() {
        let a_f = Fp16::from_bits(a[i]).to_f32();
        let b_f = Fp16::from_bits(b[i]).to_f32();
        result += a_f * b_f;
    }
    result
}

/// BF16 内积 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn bf16_ip_avx2(a: &[u16], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = load_bf16x8_as_m256(a.as_ptr().add(offset));
        let vb = load_bf16x8_as_m256(b.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    let mut result = horizontal_sum_avx2_fp16(sum);
    for i in (chunks * 8)..a.len() {
        let a_f = Bf16::from_bits(a[i]).to_f32();
        let b_f = Bf16::from_bits(b[i]).to_f32();
        result += a_f * b_f;
    }
    result
}

/// FP16 内积
#[inline]
pub fn fp16_ip(a: &[u16], b: &[u16]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { fp16_ip_avx2(a, b) };
        }
    }
    fp16_ip_scalar(a, b)
}

/// BF16 内积
#[inline]
pub fn bf16_ip(a: &[u16], b: &[u16]) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { bf16_ip_avx2(a, b) };
        }
    }
    bf16_ip_scalar(a, b)
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离
/// 对标 C++ knowhere 的 fvec_L2sqr_batch_4
#[inline]
pub fn fp16_l2_batch_4_scalar(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let d0 = fp16_l2_scalar(query, db0);
    let d1 = fp16_l2_scalar(query, db1);
    let d2 = fp16_l2_scalar(query, db2);
    let d3 = fp16_l2_scalar(query, db3);
    [d0, d1, d2, d3]
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn fp16_l2_batch_4_avx2(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let q_f32 = load_fp16x8_as_m256(query.add(offset));
        let db0_f32 = load_fp16x8_as_m256(db0.add(offset));
        let db1_f32 = load_fp16x8_as_m256(db1.add(offset));
        let db2_f32 = load_fp16x8_as_m256(db2.add(offset));
        let db3_f32 = load_fp16x8_as_m256(db3.add(offset));

        let diff0 = _mm256_sub_ps(q_f32, db0_f32);
        let diff1 = _mm256_sub_ps(q_f32, db1_f32);
        let diff2 = _mm256_sub_ps(q_f32, db2_f32);
        let diff3 = _mm256_sub_ps(q_f32, db3_f32);

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(diff0, diff0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff1, diff1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff2, diff2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(diff3, diff3));
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    result[0] = horizontal_sum_avx2_fp16(sum0);
    result[1] = horizontal_sum_avx2_fp16(sum1);
    result[2] = horizontal_sum_avx2_fp16(sum2);
    result[3] = horizontal_sum_avx2_fp16(sum3);

    for i in (chunks * 8)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    // 开方
    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();

    result
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (AVX512 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn fp16_l2_batch_4_avx512(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let q = load_fp16x16_as_m512(query.add(offset));
        let v0 = load_fp16x16_as_m512(db0.add(offset));
        let v1 = load_fp16x16_as_m512(db1.add(offset));
        let v2 = load_fp16x16_as_m512(db2.add(offset));
        let v3 = load_fp16x16_as_m512(db3.add(offset));

        let diff0 = _mm512_sub_ps(q, v0);
        let diff1 = _mm512_sub_ps(q, v1);
        let diff2 = _mm512_sub_ps(q, v2);
        let diff3 = _mm512_sub_ps(q, v3);

        sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(diff0, diff0));
        sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(diff1, diff1));
        sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(diff2, diff2));
        sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(diff3, diff3));
    }

    let mut result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    for i in (chunks * 16)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();
    result
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (NEON SIMD)
///
/// # Safety
/// - `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim`
///   readable `u16` values.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn fp16_l2_batch_4_neon(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let q_buf = [
            Fp16::from_bits(*query.add(offset)).to_f32(),
            Fp16::from_bits(*query.add(offset + 1)).to_f32(),
            Fp16::from_bits(*query.add(offset + 2)).to_f32(),
            Fp16::from_bits(*query.add(offset + 3)).to_f32(),
        ];
        let db0_buf = [
            Fp16::from_bits(*db0.add(offset)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 3)).to_f32(),
        ];
        let db1_buf = [
            Fp16::from_bits(*db1.add(offset)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 3)).to_f32(),
        ];
        let db2_buf = [
            Fp16::from_bits(*db2.add(offset)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 3)).to_f32(),
        ];
        let db3_buf = [
            Fp16::from_bits(*db3.add(offset)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 3)).to_f32(),
        ];

        let q = vld1q_f32(q_buf.as_ptr());
        let v0 = vld1q_f32(db0_buf.as_ptr());
        let v1 = vld1q_f32(db1_buf.as_ptr());
        let v2 = vld1q_f32(db2_buf.as_ptr());
        let v3 = vld1q_f32(db3_buf.as_ptr());

        let diff0 = vsubq_f32(q, v0);
        let diff1 = vsubq_f32(q, v1);
        let diff2 = vsubq_f32(q, v2);
        let diff3 = vsubq_f32(q, v3);

        sum0 = vaddq_f32(sum0, vmulq_f32(diff0, diff0));
        sum1 = vaddq_f32(sum1, vmulq_f32(diff1, diff1));
        sum2 = vaddq_f32(sum2, vmulq_f32(diff2, diff2));
        sum3 = vaddq_f32(sum3, vmulq_f32(diff3, diff3));
    }

    let mut result = [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ];

    for i in (chunks * 4)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();
    result
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离（自动选择最优实现）
#[inline]
pub fn fp16_l2_batch_4(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let dim = query.len();
    assert_eq!(db0.len(), dim);
    assert_eq!(db1.len(), dim);
    assert_eq!(db2.len(), dim);
    assert_eq!(db3.len(), dim);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe {
                fp16_l2_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                fp16_l2_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                fp16_l2_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    fp16_l2_batch_4_scalar(query, db0, db1, db2, db3)
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离
#[inline]
pub fn bf16_l2_batch_4_scalar(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let d0 = bf16_l2_scalar(query, db0);
    let d1 = bf16_l2_scalar(query, db1);
    let d2 = bf16_l2_scalar(query, db2);
    let d3 = bf16_l2_scalar(query, db3);
    [d0, d1, d2, d3]
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 平方距离（标量实现）
#[inline]
pub fn bf16_l2_sq_batch_4_scalar(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let dim = query.len();
    assert_eq!(db0.len(), dim);
    assert_eq!(db1.len(), dim);
    assert_eq!(db2.len(), dim);
    assert_eq!(db3.len(), dim);

    let mut out = [0.0f32; 4];
    for i in 0..dim {
        let q = Bf16::from_bits(query[i]).to_f32();
        let v0 = Bf16::from_bits(db0[i]).to_f32();
        let v1 = Bf16::from_bits(db1[i]).to_f32();
        let v2 = Bf16::from_bits(db2[i]).to_f32();
        let v3 = Bf16::from_bits(db3[i]).to_f32();
        let d0 = q - v0;
        let d1 = q - v1;
        let d2 = q - v2;
        let d3 = q - v3;
        out[0] += d0 * d0;
        out[1] += d1 * d1;
        out[2] += d2 * d2;
        out[3] += d3 * d3;
    }
    out
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_l2_batch_4_avx2(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let q = load_bf16x8_as_m256(query.add(offset));
        let v0 = load_bf16x8_as_m256(db0.add(offset));
        let v1 = load_bf16x8_as_m256(db1.add(offset));
        let v2 = load_bf16x8_as_m256(db2.add(offset));
        let v3 = load_bf16x8_as_m256(db3.add(offset));

        let diff0 = _mm256_sub_ps(q, v0);
        let diff1 = _mm256_sub_ps(q, v1);
        let diff2 = _mm256_sub_ps(q, v2);
        let diff3 = _mm256_sub_ps(q, v3);

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(diff0, diff0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff1, diff1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff2, diff2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(diff3, diff3));
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    result[0] = horizontal_sum_avx2_fp16(sum0);
    result[1] = horizontal_sum_avx2_fp16(sum1);
    result[2] = horizontal_sum_avx2_fp16(sum2);
    result[3] = horizontal_sum_avx2_fp16(sum3);

    // 处理 remainder
    for i in (chunks * 8)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    // 开方
    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();

    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 平方距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_l2_sq_batch_4_avx2(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let q = load_bf16x8_as_m256(query.add(offset));
        let v0 = load_bf16x8_as_m256(db0.add(offset));
        let v1 = load_bf16x8_as_m256(db1.add(offset));
        let v2 = load_bf16x8_as_m256(db2.add(offset));
        let v3 = load_bf16x8_as_m256(db3.add(offset));

        let diff0 = _mm256_sub_ps(q, v0);
        let diff1 = _mm256_sub_ps(q, v1);
        let diff2 = _mm256_sub_ps(q, v2);
        let diff3 = _mm256_sub_ps(q, v3);

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(diff0, diff0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(diff1, diff1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(diff2, diff2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(diff3, diff3));
    }

    let mut result = [0.0f32; 4];
    result[0] = horizontal_sum_avx2_fp16(sum0);
    result[1] = horizontal_sum_avx2_fp16(sum1);
    result[2] = horizontal_sum_avx2_fp16(sum2);
    result[3] = horizontal_sum_avx2_fp16(sum3);

    for i in (chunks * 8)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (AVX512 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_l2_batch_4_avx512(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let q = load_bf16x16_as_m512(query.add(offset));
        let v0 = load_bf16x16_as_m512(db0.add(offset));
        let v1 = load_bf16x16_as_m512(db1.add(offset));
        let v2 = load_bf16x16_as_m512(db2.add(offset));
        let v3 = load_bf16x16_as_m512(db3.add(offset));

        let diff0 = _mm512_sub_ps(q, v0);
        let diff1 = _mm512_sub_ps(q, v1);
        let diff2 = _mm512_sub_ps(q, v2);
        let diff3 = _mm512_sub_ps(q, v3);

        sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(diff0, diff0));
        sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(diff1, diff1));
        sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(diff2, diff2));
        sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(diff3, diff3));
    }

    let mut result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    for i in (chunks * 16)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();
    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 平方距离 (AVX512 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_l2_sq_batch_4_avx512(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let q = load_bf16x16_as_m512(query.add(offset));
        let v0 = load_bf16x16_as_m512(db0.add(offset));
        let v1 = load_bf16x16_as_m512(db1.add(offset));
        let v2 = load_bf16x16_as_m512(db2.add(offset));
        let v3 = load_bf16x16_as_m512(db3.add(offset));

        let diff0 = _mm512_sub_ps(q, v0);
        let diff1 = _mm512_sub_ps(q, v1);
        let diff2 = _mm512_sub_ps(q, v2);
        let diff3 = _mm512_sub_ps(q, v3);

        sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(diff0, diff0));
        sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(diff1, diff1));
        sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(diff2, diff2));
        sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(diff3, diff3));
    }

    let mut result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    for i in (chunks * 16)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }
    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离 (NEON SIMD)
///
/// # Safety
/// - `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim`
///   readable `u16` values.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn bf16_l2_batch_4_neon(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let q_buf = [
            Bf16::from_bits(*query.add(offset)).to_f32(),
            Bf16::from_bits(*query.add(offset + 1)).to_f32(),
            Bf16::from_bits(*query.add(offset + 2)).to_f32(),
            Bf16::from_bits(*query.add(offset + 3)).to_f32(),
        ];
        let db0_buf = [
            Bf16::from_bits(*db0.add(offset)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 3)).to_f32(),
        ];
        let db1_buf = [
            Bf16::from_bits(*db1.add(offset)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 3)).to_f32(),
        ];
        let db2_buf = [
            Bf16::from_bits(*db2.add(offset)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 3)).to_f32(),
        ];
        let db3_buf = [
            Bf16::from_bits(*db3.add(offset)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 3)).to_f32(),
        ];

        let q = vld1q_f32(q_buf.as_ptr());
        let v0 = vld1q_f32(db0_buf.as_ptr());
        let v1 = vld1q_f32(db1_buf.as_ptr());
        let v2 = vld1q_f32(db2_buf.as_ptr());
        let v3 = vld1q_f32(db3_buf.as_ptr());

        let diff0 = vsubq_f32(q, v0);
        let diff1 = vsubq_f32(q, v1);
        let diff2 = vsubq_f32(q, v2);
        let diff3 = vsubq_f32(q, v3);

        sum0 = vaddq_f32(sum0, vmulq_f32(diff0, diff0));
        sum1 = vaddq_f32(sum1, vmulq_f32(diff1, diff1));
        sum2 = vaddq_f32(sum2, vmulq_f32(diff2, diff2));
        sum3 = vaddq_f32(sum3, vmulq_f32(diff3, diff3));
    }

    let mut result = [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ];

    for i in (chunks * 4)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }

    result[0] = result[0].sqrt();
    result[1] = result[1].sqrt();
    result[2] = result[2].sqrt();
    result[3] = result[3].sqrt();
    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 平方距离 (NEON SIMD)
///
/// # Safety
/// - `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim`
///   readable `u16` values.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn bf16_l2_sq_batch_4_neon(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let q_buf = [
            Bf16::from_bits(*query.add(offset)).to_f32(),
            Bf16::from_bits(*query.add(offset + 1)).to_f32(),
            Bf16::from_bits(*query.add(offset + 2)).to_f32(),
            Bf16::from_bits(*query.add(offset + 3)).to_f32(),
        ];
        let db0_buf = [
            Bf16::from_bits(*db0.add(offset)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 3)).to_f32(),
        ];
        let db1_buf = [
            Bf16::from_bits(*db1.add(offset)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 3)).to_f32(),
        ];
        let db2_buf = [
            Bf16::from_bits(*db2.add(offset)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 3)).to_f32(),
        ];
        let db3_buf = [
            Bf16::from_bits(*db3.add(offset)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 3)).to_f32(),
        ];

        let q = vld1q_f32(q_buf.as_ptr());
        let v0 = vld1q_f32(db0_buf.as_ptr());
        let v1 = vld1q_f32(db1_buf.as_ptr());
        let v2 = vld1q_f32(db2_buf.as_ptr());
        let v3 = vld1q_f32(db3_buf.as_ptr());

        let diff0 = vsubq_f32(q, v0);
        let diff1 = vsubq_f32(q, v1);
        let diff2 = vsubq_f32(q, v2);
        let diff3 = vsubq_f32(q, v3);

        sum0 = vaddq_f32(sum0, vmulq_f32(diff0, diff0));
        sum1 = vaddq_f32(sum1, vmulq_f32(diff1, diff1));
        sum2 = vaddq_f32(sum2, vmulq_f32(diff2, diff2));
        sum3 = vaddq_f32(sum3, vmulq_f32(diff3, diff3));
    }

    let mut result = [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ];

    for i in (chunks * 4)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += (q - d0) * (q - d0);
        result[1] += (q - d1) * (q - d1);
        result[2] += (q - d2) * (q - d2);
        result[3] += (q - d3) * (q - d3);
    }
    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 平方距离（自动选择最优实现）
#[inline]
pub fn bf16_l2_sq_batch_4(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let dim = query.len();
    assert_eq!(db0.len(), dim);
    assert_eq!(db1.len(), dim);
    assert_eq!(db2.len(), dim);
    assert_eq!(db3.len(), dim);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe {
                bf16_l2_sq_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                bf16_l2_sq_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                bf16_l2_sq_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    bf16_l2_sq_batch_4_scalar(query, db0, db1, db2, db3)
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的 L2 距离（自动选择最优实现）
#[inline]
pub fn bf16_l2_batch_4(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let mut out = bf16_l2_sq_batch_4(query, db0, db1, db2, db3);
    out[0] = out[0].sqrt();
    out[1] = out[1].sqrt();
    out[2] = out[2].sqrt();
    out[3] = out[3].sqrt();
    out
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的内积
#[inline]
pub fn fp16_ip_batch_4_scalar(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let d0 = fp16_ip_scalar(query, db0);
    let d1 = fp16_ip_scalar(query, db1);
    let d2 = fp16_ip_scalar(query, db2);
    let d3 = fp16_ip_scalar(query, db3);
    [d0, d1, d2, d3]
}

/// FP16 batch_4 内积 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn fp16_ip_batch_4_avx2(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let q_f32 = load_fp16x8_as_m256(query.add(offset));
        let db0_f32 = load_fp16x8_as_m256(db0.add(offset));
        let db1_f32 = load_fp16x8_as_m256(db1.add(offset));
        let db2_f32 = load_fp16x8_as_m256(db2.add(offset));
        let db3_f32 = load_fp16x8_as_m256(db3.add(offset));

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q_f32, db0_f32));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q_f32, db1_f32));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(q_f32, db2_f32));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(q_f32, db3_f32));
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    result[0] = horizontal_sum_avx2_fp16(sum0);
    result[1] = horizontal_sum_avx2_fp16(sum1);
    result[2] = horizontal_sum_avx2_fp16(sum2);
    result[3] = horizontal_sum_avx2_fp16(sum3);

    for i in (chunks * 8)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }

    result
}

/// FP16 batch_4 内积 (AVX512 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn fp16_ip_batch_4_avx512(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let q = load_fp16x16_as_m512(query.add(offset));
        let v0 = load_fp16x16_as_m512(db0.add(offset));
        let v1 = load_fp16x16_as_m512(db1.add(offset));
        let v2 = load_fp16x16_as_m512(db2.add(offset));
        let v3 = load_fp16x16_as_m512(db3.add(offset));

        sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(q, v0));
        sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(q, v1));
        sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(q, v2));
        sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(q, v3));
    }

    let mut result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    for i in (chunks * 16)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }
    result
}

/// FP16 batch_4 内积 (NEON SIMD)
///
/// # Safety
/// - `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim`
///   readable `u16` values.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn fp16_ip_batch_4_neon(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let q_buf = [
            Fp16::from_bits(*query.add(offset)).to_f32(),
            Fp16::from_bits(*query.add(offset + 1)).to_f32(),
            Fp16::from_bits(*query.add(offset + 2)).to_f32(),
            Fp16::from_bits(*query.add(offset + 3)).to_f32(),
        ];
        let db0_buf = [
            Fp16::from_bits(*db0.add(offset)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db0.add(offset + 3)).to_f32(),
        ];
        let db1_buf = [
            Fp16::from_bits(*db1.add(offset)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db1.add(offset + 3)).to_f32(),
        ];
        let db2_buf = [
            Fp16::from_bits(*db2.add(offset)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db2.add(offset + 3)).to_f32(),
        ];
        let db3_buf = [
            Fp16::from_bits(*db3.add(offset)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 1)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 2)).to_f32(),
            Fp16::from_bits(*db3.add(offset + 3)).to_f32(),
        ];

        let q = vld1q_f32(q_buf.as_ptr());
        let v0 = vld1q_f32(db0_buf.as_ptr());
        let v1 = vld1q_f32(db1_buf.as_ptr());
        let v2 = vld1q_f32(db2_buf.as_ptr());
        let v3 = vld1q_f32(db3_buf.as_ptr());

        sum0 = vaddq_f32(sum0, vmulq_f32(q, v0));
        sum1 = vaddq_f32(sum1, vmulq_f32(q, v1));
        sum2 = vaddq_f32(sum2, vmulq_f32(q, v2));
        sum3 = vaddq_f32(sum3, vmulq_f32(q, v3));
    }

    let mut result = [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ];

    for i in (chunks * 4)..dim {
        let q = Fp16::from_bits(*query.add(i)).to_f32();
        let d0 = Fp16::from_bits(*db0.add(i)).to_f32();
        let d1 = Fp16::from_bits(*db1.add(i)).to_f32();
        let d2 = Fp16::from_bits(*db2.add(i)).to_f32();
        let d3 = Fp16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }
    result
}

/// FP16 batch_4: 计算一个查询向量与 4 个数据库向量的内积（自动选择最优实现）
#[inline]
pub fn fp16_ip_batch_4(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let dim = query.len();
    assert_eq!(db0.len(), dim);
    assert_eq!(db1.len(), dim);
    assert_eq!(db2.len(), dim);
    assert_eq!(db3.len(), dim);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe {
                fp16_ip_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                fp16_ip_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                fp16_ip_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    fp16_ip_batch_4_scalar(query, db0, db1, db2, db3)
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的内积
#[inline]
pub fn bf16_ip_batch_4_scalar(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let d0 = bf16_ip_scalar(query, db0);
    let d1 = bf16_ip_scalar(query, db1);
    let d2 = bf16_ip_scalar(query, db2);
    let d3 = bf16_ip_scalar(query, db3);
    [d0, d1, d2, d3]
}

/// BF16 batch_4 内积 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_ip_batch_4_avx2(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    let chunks = dim / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let q = load_bf16x8_as_m256(query.add(offset));
        let v0 = load_bf16x8_as_m256(db0.add(offset));
        let v1 = load_bf16x8_as_m256(db1.add(offset));
        let v2 = load_bf16x8_as_m256(db2.add(offset));
        let v3 = load_bf16x8_as_m256(db3.add(offset));

        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q, v0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q, v1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(q, v2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(q, v3));
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    result[0] = horizontal_sum_avx2_fp16(sum0);
    result[1] = horizontal_sum_avx2_fp16(sum1);
    result[2] = horizontal_sum_avx2_fp16(sum2);
    result[3] = horizontal_sum_avx2_fp16(sum3);

    // 处理 remainder
    for i in (chunks * 8)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }

    result
}

/// BF16 batch_4 内积 (AVX512 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::incompatible_msrv)]
/// # Safety
/// `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim` contiguous `u16`
/// elements.
pub unsafe fn bf16_ip_batch_4_avx512(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::*;

    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut sum2 = _mm512_setzero_ps();
    let mut sum3 = _mm512_setzero_ps();
    let chunks = dim / 16;

    for i in 0..chunks {
        let offset = i * 16;
        let q = load_bf16x16_as_m512(query.add(offset));
        let v0 = load_bf16x16_as_m512(db0.add(offset));
        let v1 = load_bf16x16_as_m512(db1.add(offset));
        let v2 = load_bf16x16_as_m512(db2.add(offset));
        let v3 = load_bf16x16_as_m512(db3.add(offset));

        sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(q, v0));
        sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(q, v1));
        sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(q, v2));
        sum3 = _mm512_add_ps(sum3, _mm512_mul_ps(q, v3));
    }

    let mut result = [
        _mm512_reduce_add_ps(sum0),
        _mm512_reduce_add_ps(sum1),
        _mm512_reduce_add_ps(sum2),
        _mm512_reduce_add_ps(sum3),
    ];

    for i in (chunks * 16)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }
    result
}

/// BF16 batch_4 内积 (NEON SIMD)
///
/// # Safety
/// - `query`, `db0`, `db1`, `db2`, and `db3` must each point to at least `dim`
///   readable `u16` values.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[inline]
pub unsafe fn bf16_ip_batch_4_neon(
    query: *const u16,
    db0: *const u16,
    db1: *const u16,
    db2: *const u16,
    db3: *const u16,
    dim: usize,
) -> [f32; 4] {
    use std::arch::aarch64::*;

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let chunks = dim / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let q_buf = [
            Bf16::from_bits(*query.add(offset)).to_f32(),
            Bf16::from_bits(*query.add(offset + 1)).to_f32(),
            Bf16::from_bits(*query.add(offset + 2)).to_f32(),
            Bf16::from_bits(*query.add(offset + 3)).to_f32(),
        ];
        let db0_buf = [
            Bf16::from_bits(*db0.add(offset)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db0.add(offset + 3)).to_f32(),
        ];
        let db1_buf = [
            Bf16::from_bits(*db1.add(offset)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db1.add(offset + 3)).to_f32(),
        ];
        let db2_buf = [
            Bf16::from_bits(*db2.add(offset)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db2.add(offset + 3)).to_f32(),
        ];
        let db3_buf = [
            Bf16::from_bits(*db3.add(offset)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 1)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 2)).to_f32(),
            Bf16::from_bits(*db3.add(offset + 3)).to_f32(),
        ];

        let q = vld1q_f32(q_buf.as_ptr());
        let v0 = vld1q_f32(db0_buf.as_ptr());
        let v1 = vld1q_f32(db1_buf.as_ptr());
        let v2 = vld1q_f32(db2_buf.as_ptr());
        let v3 = vld1q_f32(db3_buf.as_ptr());

        sum0 = vaddq_f32(sum0, vmulq_f32(q, v0));
        sum1 = vaddq_f32(sum1, vmulq_f32(q, v1));
        sum2 = vaddq_f32(sum2, vmulq_f32(q, v2));
        sum3 = vaddq_f32(sum3, vmulq_f32(q, v3));
    }

    let mut result = [
        vaddvq_f32(sum0),
        vaddvq_f32(sum1),
        vaddvq_f32(sum2),
        vaddvq_f32(sum3),
    ];

    for i in (chunks * 4)..dim {
        let q = Bf16::from_bits(*query.add(i)).to_f32();
        let d0 = Bf16::from_bits(*db0.add(i)).to_f32();
        let d1 = Bf16::from_bits(*db1.add(i)).to_f32();
        let d2 = Bf16::from_bits(*db2.add(i)).to_f32();
        let d3 = Bf16::from_bits(*db3.add(i)).to_f32();

        result[0] += q * d0;
        result[1] += q * d1;
        result[2] += q * d2;
        result[3] += q * d3;
    }
    result
}

/// BF16 batch_4: 计算一个查询向量与 4 个数据库向量的内积（自动选择最优实现）
#[inline]
pub fn bf16_ip_batch_4(
    query: &[u16],
    db0: &[u16],
    db1: &[u16],
    db2: &[u16],
    db3: &[u16],
) -> [f32; 4] {
    let dim = query.len();
    assert_eq!(db0.len(), dim);
    assert_eq!(db1.len(), dim);
    assert_eq!(db2.len(), dim);
    assert_eq!(db3.len(), dim);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe {
                bf16_ip_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe {
                bf16_ip_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                bf16_ip_batch_4_neon(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                )
            };
        }
    }

    bf16_ip_batch_4_scalar(query, db0, db1, db2, db3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_basic() {
        let f: f32 = 1.0;
        let h = Fp16::from_f32(f);
        assert_eq!(h, Fp16::ONE);

        let back: f32 = h.to_f32();
        assert!((back - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fp16_zero() {
        let f: f32 = 0.0;
        let h = Fp16::from_f32(f);
        assert_eq!(h, Fp16::ZERO);

        let back: f32 = h.to_f32();
        assert_eq!(back, 0.0);
    }

    #[test]
    fn test_fp16_negative() {
        let f: f32 = -std::f32::consts::PI;
        let h = Fp16::from_f32(f);
        let back: f32 = h.to_f32();
        // fp16 has limited precision
        assert!((back - f).abs() < 0.1);
    }

    #[test]
    fn test_bf16_basic() {
        let f: f32 = 1.0;
        let b = Bf16::from_f32(f);
        assert_eq!(b, Bf16::ONE);

        let back: f32 = b.to_f32();
        assert_eq!(back, 1.0);
    }

    #[test]
    fn test_bf16_zero() {
        let f: f32 = 0.0;
        let b = Bf16::from_f32(f);
        assert_eq!(b, Bf16::ZERO);
    }

    #[test]
    fn test_bf16_precision_loss() {
        // bf16 有更少的尾数位，会有精度损失
        let f: f32 = 1.234_567_9;
        let b = Bf16::from_f32(f);
        let back: f32 = b.to_f32();

        // bf16 精度约为 2^-7 ≈ 0.0078
        assert!((back - f).abs() < 0.01);
    }

    #[test]
    fn test_batch_convert() {
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, -1.0, 100.0];

        // fp16 round-trip
        let fp16_data = f32_to_fp16(&src);
        let dst_fp16 = fp16_to_f32(&fp16_data);
        for (a, b) in src.iter().zip(dst_fp16.iter()) {
            assert!((a - b).abs() < 0.1);
        }

        // bf16 round-trip
        let bf16_data = f32_to_bf16(&src);
        let dst_bf16 = bf16_to_f32(&bf16_data);
        for (a, b) in src.iter().zip(dst_bf16.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_fp16_l2_distance() {
        let a: Vec<u16> = f32_to_fp16(&[0.0, 0.0]);
        let b: Vec<u16> = f32_to_fp16(&[3.0, 4.0]);

        let dist = fp16_l2(&a, &b);
        assert!((dist - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_bf16_l2_distance() {
        let a: Vec<u16> = f32_to_bf16(&[0.0, 0.0]);
        let b: Vec<u16> = f32_to_bf16(&[3.0, 4.0]);

        let dist = bf16_l2(&a, &b);
        assert!((dist - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_fp16_inf_nan() {
        // 测试无穷大
        let inf = f32::INFINITY;
        let h_inf = Fp16::from_f32(inf);
        let back_inf: f32 = h_inf.to_f32();
        assert!(back_inf.is_infinite());

        // 测试 NaN
        let nan = f32::NAN;
        let h_nan = Fp16::from_f32(nan);
        let back_nan: f32 = h_nan.to_f32();
        assert!(back_nan.is_nan());
    }

    #[test]
    fn test_bf16_inf_nan() {
        let inf = f32::INFINITY;
        let b_inf = Bf16::from_f32(inf);
        let back_inf: f32 = b_inf.to_f32();
        assert!(back_inf.is_infinite());

        let nan = f32::NAN;
        let b_nan = Bf16::from_f32(nan);
        let back_nan: f32 = b_nan.to_f32();
        assert!(back_nan.is_nan());
    }

    #[test]
    fn test_fp16_ip() {
        let a: Vec<u16> = f32_to_fp16(&[1.0, 2.0, 3.0]);
        let b: Vec<u16> = f32_to_fp16(&[4.0, 5.0, 6.0]);

        let ip = fp16_ip(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((ip - 32.0).abs() < 0.5);
    }

    #[test]
    fn test_bf16_ip() {
        let a: Vec<u16> = f32_to_bf16(&[1.0, 2.0, 3.0]);
        let b: Vec<u16> = f32_to_bf16(&[4.0, 5.0, 6.0]);

        let ip = bf16_ip(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((ip - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_fp16_ip_simd_large() {
        // Large vector to exercise AVX2 SIMD path (processes 4 elements at a time)
        let n = 128;
        let a_f32: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

        let a: Vec<u16> = f32_to_fp16(&a_f32);
        let b: Vec<u16> = f32_to_fp16(&b_f32);

        let ip = fp16_ip(&a, &b);

        // Calculate expected result using scalar method
        let expected: f32 = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).sum();

        // FP16 has limited precision, allow larger tolerance
        assert!((ip - expected).abs() < expected.abs() * 0.05 + 1.0);
    }

    #[test]
    fn test_bf16_ip_simd_large() {
        // Large vector to exercise AVX2 SIMD path (processes 8 elements at a time)
        let n = 256;
        let a_f32: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.5).collect();

        let a: Vec<u16> = f32_to_bf16(&a_f32);
        let b: Vec<u16> = f32_to_bf16(&b_f32);

        let ip = bf16_ip(&a, &b);

        // Calculate expected result
        let expected: f32 = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).sum();

        // BF16 has better precision than FP16
        assert!((ip - expected).abs() < expected.abs() * 0.01 + 0.5);
    }

    #[test]
    fn test_fp16_ip_large_vectors() {
        // 测试较大向量的内积
        let size = 1024;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

        let a_fp16 = f32_to_fp16(&a);
        let b_fp16 = f32_to_fp16(&b);

        let ip = fp16_ip(&a_fp16, &b_fp16);

        // 验证结果在合理范围内
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((ip - expected).abs() / expected.abs() < 0.01);
    }

    #[test]
    fn test_bf16_ip_large_vectors() {
        // 测试较大向量的内积
        let size = 1024;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.01).collect();

        let a_bf16 = f32_to_bf16(&a);
        let b_bf16 = f32_to_bf16(&b);

        let ip = bf16_ip(&a_bf16, &b_bf16);

        // 验证结果在合理范围内
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((ip - expected).abs() / expected.abs() < 0.01);
    }

    #[test]
    fn test_fp16_l2_large_vectors() {
        // 测试较大向量的 L2 距离
        let size = 512;
        let a: Vec<f32> = vec![0.0; size];
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

        let a_fp16 = f32_to_fp16(&a);
        let b_fp16 = f32_to_fp16(&b);

        let dist = fp16_l2(&a_fp16, &b_fp16);

        // 验证结果
        let expected: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((dist - expected).abs() / expected.abs() < 0.01);
    }

    #[test]
    fn test_bf16_l2_large_vectors() {
        // 测试较大向量的 L2 距离
        let size = 512;
        let a: Vec<f32> = vec![0.0; size];
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();

        let a_bf16 = f32_to_bf16(&a);
        let b_bf16 = f32_to_bf16(&b);

        let dist = bf16_l2(&a_bf16, &b_bf16);

        // 验证结果
        let expected: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((dist - expected).abs() / expected.abs() < 0.01);
    }

    #[test]
    fn test_fp16_l2_batch_4() {
        // 测试 FP16 batch_4 L2 距离
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let db1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 2.0).collect();
        let db2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 3.0).collect();
        let db3: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 4.0).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let results = fp16_l2_batch_4(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        // 验证结果
        assert_eq!(results.len(), 4);

        // 计算期望值
        let expected0 = fp16_l2(&query_fp16, &db0_fp16);
        let expected1 = fp16_l2(&query_fp16, &db1_fp16);
        let expected2 = fp16_l2(&query_fp16, &db2_fp16);
        let expected3 = fp16_l2(&query_fp16, &db3_fp16);

        assert!((results[0] - expected0).abs() < 0.01);
        assert!((results[1] - expected1).abs() < 0.01);
        assert!((results[2] - expected2).abs() < 0.01);
        assert!((results[3] - expected3).abs() < 0.01);
    }

    #[test]
    fn test_bf16_l2_batch_4() {
        // 测试 BF16 batch_4 L2 距离
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let db1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 2.0).collect();
        let db2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 3.0).collect();
        let db3: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 4.0).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let results = bf16_l2_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        // 验证结果
        assert_eq!(results.len(), 4);

        // 计算期望值
        let expected0 = bf16_l2(&query_bf16, &db0_bf16);
        let expected1 = bf16_l2(&query_bf16, &db1_bf16);
        let expected2 = bf16_l2(&query_bf16, &db2_bf16);
        let expected3 = bf16_l2(&query_bf16, &db3_bf16);

        assert!((results[0] - expected0).abs() < 0.01);
        assert!((results[1] - expected1).abs() < 0.01);
        assert!((results[2] - expected2).abs() < 0.01);
        assert!((results[3] - expected3).abs() < 0.01);
    }

    #[test]
    fn test_bf16_l2_sq_batch_4_matches_l2_squared() {
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let db1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 2.0).collect();
        let db2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 3.0).collect();
        let db3: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 4.0).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let sq = bf16_l2_sq_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);
        let d = bf16_l2_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        for i in 0..4 {
            assert!(
                (sq[i] - d[i] * d[i]).abs() < 0.05,
                "bf16 l2 sq mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_fp16_ip_batch_4() {
        // 测试 FP16 batch_4 内积
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let db1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 2.0).collect();
        let db2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 3.0).collect();
        let db3: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 4.0).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let results = fp16_ip_batch_4(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        // 验证结果
        assert_eq!(results.len(), 4);

        // 计算期望值
        let expected0 = fp16_ip(&query_fp16, &db0_fp16);
        let expected1 = fp16_ip(&query_fp16, &db1_fp16);
        let expected2 = fp16_ip(&query_fp16, &db2_fp16);
        let expected3 = fp16_ip(&query_fp16, &db3_fp16);

        assert!((results[0] - expected0).abs() < 0.01);
        assert!((results[1] - expected1).abs() < 0.01);
        assert!((results[2] - expected2).abs() < 0.01);
        assert!((results[3] - expected3).abs() < 0.01);
    }

    #[test]
    fn test_bf16_ip_batch_4() {
        // 测试 BF16 batch_4 内积
        let dim = 128;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let db1: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 2.0).collect();
        let db2: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 3.0).collect();
        let db3: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 4.0).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let results = bf16_ip_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        // 验证结果
        assert_eq!(results.len(), 4);

        // 计算期望值
        let expected0 = bf16_ip(&query_bf16, &db0_bf16);
        let expected1 = bf16_ip(&query_bf16, &db1_bf16);
        let expected2 = bf16_ip(&query_bf16, &db2_bf16);
        let expected3 = bf16_ip(&query_bf16, &db3_bf16);

        assert!((results[0] - expected0).abs() < 0.01);
        assert!((results[1] - expected1).abs() < 0.01);
        assert!((results[2] - expected2).abs() < 0.01);
        assert!((results[3] - expected3).abs() < 0.01);
    }

    #[test]
    fn test_fp16_l2_batch_4_remainder() {
        let dim = 11;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 5.0) * 0.25).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.5).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 1.25).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 1.5).collect();
        let db3: Vec<f32> = query.iter().map(|v| -v).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let results = fp16_l2_batch_4(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);
        let expected =
            fp16_l2_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 0.01,
                "fp16 l2 remainder mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_bf16_l2_batch_4_remainder() {
        let dim = 13;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.75).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 1.0).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 0.5).collect();
        let db3: Vec<f32> = query.iter().map(|v| v + (v.sin() * 0.2)).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let results = bf16_l2_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);
        let expected =
            bf16_l2_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 0.02,
                "bf16 l2 remainder mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_fp16_ip_batch_4_remainder() {
        let dim = 9;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 3.0) * 0.4).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.5).collect();
        let db1: Vec<f32> = query.iter().map(|v| v * -1.0).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 2.0).collect();
        let db3: Vec<f32> = query.iter().map(|v| v - 0.25).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let results = fp16_ip_batch_4(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);
        let expected =
            fp16_ip_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 0.01,
                "fp16 ip remainder mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_bf16_ip_batch_4_remainder() {
        let dim = 15;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 2.0) * -0.2).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.5).collect();
        let db1: Vec<f32> = query.iter().map(|v| v * 1.25).collect();
        let db2: Vec<f32> = query.iter().map(|v| v - 1.5).collect();
        let db3: Vec<f32> = query.iter().map(|v| v.cos()).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let results = bf16_ip_batch_4(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);
        let expected =
            bf16_ip_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 0.02,
                "bf16 ip remainder mismatch at {}",
                i
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_fp16_batch_4_neon_direct() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let dim = 17;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.2 - 1.0).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.5).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 0.75).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 1.1).collect();
        let db3: Vec<f32> = query.iter().map(|v| -v * 0.8).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let l2_neon = unsafe {
            fp16_l2_batch_4_neon(
                query_fp16.as_ptr(),
                db0_fp16.as_ptr(),
                db1_fp16.as_ptr(),
                db2_fp16.as_ptr(),
                db3_fp16.as_ptr(),
                dim,
            )
        };
        let ip_neon = unsafe {
            fp16_ip_batch_4_neon(
                query_fp16.as_ptr(),
                db0_fp16.as_ptr(),
                db1_fp16.as_ptr(),
                db2_fp16.as_ptr(),
                db3_fp16.as_ptr(),
                dim,
            )
        };

        let l2_expected =
            fp16_l2_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);
        let ip_expected =
            fp16_ip_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        for i in 0..4 {
            assert!(
                (l2_neon[i] - l2_expected[i]).abs() < 0.01,
                "fp16 neon l2 mismatch at {}",
                i
            );
            assert!(
                (ip_neon[i] - ip_expected[i]).abs() < 0.01,
                "fp16 neon ip mismatch at {}",
                i
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_bf16_batch_4_neon_direct() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }

        let dim = 19;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * -0.15 + 0.5).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.4).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 0.9).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 0.7).collect();
        let db3: Vec<f32> = query.iter().map(|v| v.sin()).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let l2_neon = unsafe {
            bf16_l2_batch_4_neon(
                query_bf16.as_ptr(),
                db0_bf16.as_ptr(),
                db1_bf16.as_ptr(),
                db2_bf16.as_ptr(),
                db3_bf16.as_ptr(),
                dim,
            )
        };
        let ip_neon = unsafe {
            bf16_ip_batch_4_neon(
                query_bf16.as_ptr(),
                db0_bf16.as_ptr(),
                db1_bf16.as_ptr(),
                db2_bf16.as_ptr(),
                db3_bf16.as_ptr(),
                dim,
            )
        };

        let l2_expected =
            bf16_l2_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);
        let ip_expected =
            bf16_ip_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        for i in 0..4 {
            assert!(
                (l2_neon[i] - l2_expected[i]).abs() < 0.02,
                "bf16 neon l2 mismatch at {}",
                i
            );
            assert!(
                (ip_neon[i] - ip_expected[i]).abs() < 0.02,
                "bf16 neon ip mismatch at {}",
                i
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_fp16_batch_4_avx512_direct() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }

        let dim = 33;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.125 - 1.25).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.5).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 0.3).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 1.2).collect();
        let db3: Vec<f32> = query.iter().map(|v| -v * 0.9).collect();

        let query_fp16 = f32_to_fp16(&query);
        let db0_fp16 = f32_to_fp16(&db0);
        let db1_fp16 = f32_to_fp16(&db1);
        let db2_fp16 = f32_to_fp16(&db2);
        let db3_fp16 = f32_to_fp16(&db3);

        let l2_avx512 = unsafe {
            fp16_l2_batch_4_avx512(
                query_fp16.as_ptr(),
                db0_fp16.as_ptr(),
                db1_fp16.as_ptr(),
                db2_fp16.as_ptr(),
                db3_fp16.as_ptr(),
                dim,
            )
        };
        let ip_avx512 = unsafe {
            fp16_ip_batch_4_avx512(
                query_fp16.as_ptr(),
                db0_fp16.as_ptr(),
                db1_fp16.as_ptr(),
                db2_fp16.as_ptr(),
                db3_fp16.as_ptr(),
                dim,
            )
        };

        let l2_expected =
            fp16_l2_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);
        let ip_expected =
            fp16_ip_batch_4_scalar(&query_fp16, &db0_fp16, &db1_fp16, &db2_fp16, &db3_fp16);

        for i in 0..4 {
            assert!(
                (l2_avx512[i] - l2_expected[i]).abs() < 0.01,
                "fp16 avx512 l2 mismatch at {}",
                i
            );
            assert!(
                (ip_avx512[i] - ip_expected[i]).abs() < 0.01,
                "fp16 avx512 ip mismatch at {}",
                i
            );
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_bf16_batch_4_avx512_direct() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }

        let dim = 35;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * -0.2 + 0.75).collect();
        let db0: Vec<f32> = query.iter().map(|v| v + 0.4).collect();
        let db1: Vec<f32> = query.iter().map(|v| v - 0.6).collect();
        let db2: Vec<f32> = query.iter().map(|v| v * 0.8).collect();
        let db3: Vec<f32> = query.iter().map(|v| v.cos()).collect();

        let query_bf16 = f32_to_bf16(&query);
        let db0_bf16 = f32_to_bf16(&db0);
        let db1_bf16 = f32_to_bf16(&db1);
        let db2_bf16 = f32_to_bf16(&db2);
        let db3_bf16 = f32_to_bf16(&db3);

        let l2_avx512 = unsafe {
            bf16_l2_batch_4_avx512(
                query_bf16.as_ptr(),
                db0_bf16.as_ptr(),
                db1_bf16.as_ptr(),
                db2_bf16.as_ptr(),
                db3_bf16.as_ptr(),
                dim,
            )
        };
        let ip_avx512 = unsafe {
            bf16_ip_batch_4_avx512(
                query_bf16.as_ptr(),
                db0_bf16.as_ptr(),
                db1_bf16.as_ptr(),
                db2_bf16.as_ptr(),
                db3_bf16.as_ptr(),
                dim,
            )
        };

        let l2_expected =
            bf16_l2_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);
        let ip_expected =
            bf16_ip_batch_4_scalar(&query_bf16, &db0_bf16, &db1_bf16, &db2_bf16, &db3_bf16);

        for i in 0..4 {
            assert!(
                (l2_avx512[i] - l2_expected[i]).abs() < 0.02,
                "bf16 avx512 l2 mismatch at {}",
                i
            );
            assert!(
                (ip_avx512[i] - ip_expected[i]).abs() < 0.02,
                "bf16 avx512 ip mismatch at {}",
                i
            );
        }
    }
}

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

// Use the standard library's f16 if available (Rust 1.69+)
#[cfg(feature = "std-f16")]
pub use std::f32::F16;

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
            Self(((sign as u16) << 15 | fp16_exp << 10 | fp16_mantissa) as u16)
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

/// FP16 L2 距离 (AVX2 SIMD)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
unsafe fn fp16_l2_avx2(a: &[u16], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    // 每次处理 4 个 fp16
    for i in 0..chunks {
        // 加载 4 个 fp16
        let va_raw = _mm_loadu_si64(a.as_ptr().add(i * 4) as *const i64);
        let vb_raw = _mm_loadu_si64(b.as_ptr().add(i * 4) as *const i64);
        
        // 扩展到 f32
        let va_f32 = _mm256_cvtph_ps(_mm_set_epi64x(
            ((va_raw >> 48) & 0xFFFF) as i16,
            ((va_raw >> 32) & 0xFFFF) as i16,
            ((va_raw >> 16) & 0xFFFF) as i16,
            (va_raw & 0xFFFF) as i16,
        ));
        
        let vb_f32 = _mm256_cvtph_ps(_mm_set_epi64x(
            ((vb_raw >> 48) & 0xFFFF) as i16,
            ((vb_raw >> 32) & 0xFFFF) as i16,
            ((vb_raw >> 16) & 0xFFFF) as i16,
            (vb_raw & 0xFFFF) as i16,
        ));
        
        let diff = _mm256_sub_ps(va_f32, vb_f32);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    
    // Horizontal add
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // 处理 remainder
    for i in (chunks * 4)..a.len() {
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
    let remainder = a.len() % 8;
    
    // 每次处理 8 个 bf16
    for i in 0..chunks {
        // 加载 8 个 bf16
        let va = _mm_loadu_si128(a.as_ptr().add(i * 8) as *const __m256i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i * 8) as *const __m256i);
        
        // 扩展 bf16 到 f32
        let va_lo = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(va, _mm_setzero_si128()));
        let va_hi = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(va, _mm_setzero_si128()));
        let vb_lo = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(vb, _mm_setzero_si128()));
        let vb_hi = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(vb, _mm_setzero_si128()));
        
        let va_lo_f = _mm256_castsi256_ps(va_lo);
        let va_hi_f = _mm256_castsi256_ps(va_hi);
        let vb_lo_f = _mm256_castsi256_ps(vb_lo);
        let vb_hi_f = _mm256_castsi256_ps(vb_hi);
        
        let diff_lo = _mm256_sub_ps(va_lo_f, vb_lo_f);
        let diff_hi = _mm256_sub_ps(va_hi_f, vb_hi_f);
        
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff_lo, diff_lo));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff_hi, diff_hi));
    }
    
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // 处理 remainder
    for i in (chunks * 8)..a.len() {
        let a_f = Bf16::from_bits(a[i]).to_f32();
        let b_f = Bf16::from_bits(b[i]).to_f32();
        let diff = a_f - b_f;
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
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    
    // 每次处理 4 个 fp16
    for i in 0..chunks {
        // 加载 4 个 fp16
        let va_raw = _mm_loadu_si64(a.as_ptr().add(i * 4) as *const i64);
        let vb_raw = _mm_loadu_si64(b.as_ptr().add(i * 4) as *const i64);
        
        // 扩展到 f32
        let va_f32 = _mm256_cvtph_ps(_mm_set_epi64x(
            ((va_raw >> 48) & 0xFFFF) as i16,
            ((va_raw >> 32) & 0xFFFF) as i16,
            ((va_raw >> 16) & 0xFFFF) as i16,
            (va_raw & 0xFFFF) as i16,
        ));
        
        let vb_f32 = _mm256_cvtph_ps(_mm_set_epi64x(
            ((vb_raw >> 48) & 0xFFFF) as i16,
            ((vb_raw >> 32) & 0xFFFF) as i16,
            ((vb_raw >> 16) & 0xFFFF) as i16,
            (vb_raw & 0xFFFF) as i16,
        ));
        
        // Multiply and add
        sum = _mm256_fmadd_ps(va_f32, vb_f32, sum);
    }
    
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // 处理 remainder
    for i in (chunks * 4)..a.len() {
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
    let remainder = a.len() % 8;
    
    // 每次处理 8 个 bf16
    for i in 0..chunks {
        // 加载 8 个 bf16
        let va = _mm_loadu_si128(a.as_ptr().add(i * 8) as *const __m256i);
        let vb = _mm_loadu_si128(b.as_ptr().add(i * 8) as *const __m256i);
        
        // 扩展 bf16 到 f32
        let va_lo = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(va, _mm_setzero_si128()));
        let va_hi = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(va, _mm_setzero_si128()));
        let vb_lo = _mm256_cvtepu16_epi32(_mm_unpacklo_epi16(vb, _mm_setzero_si128()));
        let vb_hi = _mm256_cvtepu16_epi32(_mm_unpackhi_epi16(vb, _mm_setzero_si128()));
        
        let va_lo_f = _mm256_castsi256_ps(va_lo);
        let va_hi_f = _mm256_castsi256_ps(va_hi);
        let vb_lo_f = _mm256_castsi256_ps(vb_lo);
        let vb_hi_f = _mm256_castsi256_ps(vb_hi);
        
        // Multiply and add
        sum = _mm256_fmadd_ps(va_lo_f, vb_lo_f, sum);
        sum = _mm256_fmadd_ps(va_hi_f, vb_hi_f, sum);
    }
    
    let mut result = _mm256_cvtss_f32(sum);
    let high = _mm256_extractf128_ps(sum, 1);
    result += _mm_cvtss_f32(_mm_add_ps(high, _mm256_castps256to128(sum)));
    
    // 处理 remainder
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
        let f: f32 = -3.14;
        let h = Fp16::from_f32(f);
        let back: f32 = h.to_f32();
        // fp16 has limited precision
        assert!((back + 3.14).abs() < 0.1);
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
        let f: f32 = 1.23456789;
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
}

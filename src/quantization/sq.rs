//! SQ (Scalar Quantization) 量化器
//!
//! 将浮点数映射到低精度表示 (通常是 8-bit)
//! 参考: https://faiss.ai/cpp/html/ScalarQuantizer_8bit_avx512_8h.html

/// SQ 量化类型
#[derive(Debug, Clone, Copy)]
pub enum QuantizerType {
    /// 8-bit 均匀量化
    Uniform,
    /// 8-bit 非均匀 (Learnable)
    Learned,
    /// 4-bit 量化
    Quant4,
}

/// SQ 量化器配置
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    pub dim: usize,
    pub bit: usize, // 量化位数 (4 或 8)
    pub quantizer_type: QuantizerType,

    // 量化参数
    pub min_val: f32, // 最小值
    pub max_val: f32, // 最大值
    pub scale: f32,   // 缩放因子
    pub offset: f32,  // 偏移
}

impl ScalarQuantizer {
    /// 创建新的量化器
    pub fn new(dim: usize, bit: usize) -> Self {
        Self {
            dim,
            bit: bit.min(8),
            quantizer_type: QuantizerType::Uniform,
            min_val: f32::MAX,
            max_val: f32::MIN,
            scale: 0.0,
            offset: 0.0,
        }
    }

    /// 训练量化器 (确定 min/max/scale)
    pub fn train(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        // 计算 min/max
        self.min_val = data.iter().cloned().fold(f32::MAX, f32::min);
        self.max_val = data.iter().cloned().fold(f32::MIN, f32::max);

        // 计算 scale 和 offset
        let range = self.max_val - self.min_val;
        let levels = (1 << self.bit) as f32;

        self.scale = if range > 0.0 {
            (levels - 1.0) / range
        } else {
            1.0
        };
        self.offset = self.min_val;
    }

    /// 量化：将 f32 转换为 u8/u4
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let levels = (1 << self.bit) as f32;
        vector
            .iter()
            .map(|&v| {
                let scaled = (v - self.offset) * self.scale;

                scaled.clamp(0.0, levels - 1.0).round() as u8
            })
            .collect()
    }

    /// 解码：将 u8/u4 转换回 f32
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let _levels = (1 << self.bit) as f32;
        codes
            .iter()
            .map(|&c| {
                let v = c as f32 / self.scale + self.offset;
                v.clamp(self.min_val, self.max_val)
            })
            .collect()
    }

    /// 解码到预分配 buffer（零分配）
    pub fn decode_into(&self, codes: &[u8], out: &mut [f32]) {
        debug_assert_eq!(codes.len(), out.len());
        for (o, &c) in out.iter_mut().zip(codes.iter()) {
            *o = (c as f32 / self.scale + self.offset).clamp(self.min_val, self.max_val);
        }
    }

    /// Fused decode + dot product（一次扫描，零中间 buffer）
    ///
    /// 等价于: decode_into(codes, buf); dot_product_f32(query, buf)
    /// 但只读一次 codes，不写中间 float buffer。
    pub fn decode_dot_f32(&self, codes: &[u8], query: &[f32]) -> f32 {
        debug_assert_eq!(codes.len(), query.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") && codes.len() >= 16 {
            // SAFETY: AVX-512F path is guarded by runtime feature detection.
            return unsafe {
                decode_dot_avx512(
                    codes,
                    query,
                    self.scale,
                    self.offset,
                    self.min_val,
                    self.max_val,
                )
            };
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && codes.len() >= 8
            {
                // SAFETY: AVX2+FMA path is guarded by runtime feature detection.
                return unsafe {
                    decode_dot_avx2_fma(
                        codes,
                        query,
                        self.scale,
                        self.offset,
                        self.min_val,
                        self.max_val,
                    )
                };
            }
            if std::arch::is_x86_feature_detected!("avx2") && codes.len() >= 8 {
                // SAFETY: AVX2 path is guarded by runtime feature detection.
                return unsafe {
                    decode_dot_avx2(
                        codes,
                        query,
                        self.scale,
                        self.offset,
                        self.min_val,
                        self.max_val,
                    )
                };
            }
        }

        codes
            .iter()
            .zip(query.iter())
            .map(|(&c, &q)| {
                let v = (c as f32 / self.scale + self.offset).clamp(self.min_val, self.max_val);
                v * q
            })
            .sum()
    }

    /// 计算量化误差
    pub fn compute_error(&self, original: &[f32], reconstructed: &[f32]) -> f32 {
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// 批量编码
    pub fn encode_batch(&self, vectors: &[f32]) -> Vec<Vec<u8>> {
        let n = vectors.len() / self.dim;
        (0..n)
            .map(|i| {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                self.encode(v)
            })
            .collect()
    }

    /// 批量解码
    pub fn decode_batch(&self, codes: &[Vec<u8>]) -> Vec<f32> {
        let mut result = Vec::with_capacity(codes.len() * self.dim);
        for code in codes {
            result.extend_from_slice(&self.decode(code));
        }
        result
    }

    /// 在 uint8 量化域计算近似 L2 距离（无需 decode）
    ///
    /// q_residual: 查询向量的残差（float）
    /// db_code: 数据库向量的 uint8 量化码
    /// 返回: 近似 L2² 距离（未开根号）
    pub fn sq_l2_asymmetric(&self, q_residual: &[f32], db_code: &[u8]) -> f32 {
        debug_assert_eq!(q_residual.len(), db_code.len());
        let inv_scale = 1.0 / self.scale;

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && q_residual.len() >= 8 {
            // SAFETY: AVX2 path is guarded by runtime feature detection.
            return unsafe { sq_l2_asymmetric_avx2(q_residual, db_code, inv_scale, self.offset) };
        }

        q_residual
            .iter()
            .zip(db_code.iter())
            .map(|(&qr, &db)| {
                let db_residual = db as f32 * inv_scale + self.offset;
                let diff = qr - db_residual;
                diff * diff
            })
            .sum()
    }

    /// Precompute query values in quantized integer domain.
    ///
    /// q_i16[i] = round((q[i] - offset) * scale)
    pub fn precompute_query(&self, q_residual: &[f32]) -> Vec<i16> {
        let mut out = vec![0i16; q_residual.len()];
        self.precompute_query_into(q_residual, &mut out);
        out
    }

    /// Precompute query values into a caller-provided buffer to avoid allocations.
    pub fn precompute_query_into(&self, q_residual: &[f32], out: &mut [i16]) {
        debug_assert_eq!(q_residual.len(), out.len());
        for (&v, o) in q_residual.iter().zip(out.iter_mut()) {
            let q = ((v - self.offset) * self.scale)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32);
            *o = q as i16;
        }
    }

    /// Compute SQ8 asymmetric L2 distance using a precomputed integer-domain query.
    ///
    /// Returns distance in original float domain (L2^2).
    pub fn sq_l2_precomputed(&self, q_i16: &[i16], db_code: &[u8]) -> f32 {
        debug_assert_eq!(q_i16.len(), db_code.len());
        let inv_scale = 1.0 / self.scale;
        let acc: i64 = {
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx512bw") && q_i16.len() >= 32 {
                    // SAFETY: AVX-512 path is guarded by runtime feature detection.
                    unsafe { sq_l2_precomputed_avx512(q_i16, db_code) }
                } else
                if std::arch::is_x86_feature_detected!("avx2") && q_i16.len() >= 16 {
                    // SAFETY: AVX2 path is guarded by runtime feature detection.
                    unsafe { sq_l2_precomputed_avx2(q_i16, db_code) }
                } else {
                    q_i16
                        .iter()
                        .zip(db_code.iter())
                        .map(|(&qv, &db)| {
                            let diff = qv as i32 - db as i32;
                            (diff * diff) as i64
                        })
                        .sum()
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                q_i16
                    .iter()
                    .zip(db_code.iter())
                    .map(|(&qv, &db)| {
                        let diff = qv as i32 - db as i32;
                        (diff * diff) as i64
                    })
                    .sum()
            }
        };
        acc as f32 * inv_scale * inv_scale
    }
}

/// SQ8 量化器 (8-bit, 简化别名)
pub type Sq8Quantizer = ScalarQuantizer;

/// SQ4 量化器 (4-bit)
pub struct Sq4Quantizer {
    sq8: ScalarQuantizer,
}

impl Sq4Quantizer {
    pub fn new(dim: usize) -> Self {
        Self {
            sq8: ScalarQuantizer::new(dim, 4),
        }
    }

    pub fn train(&mut self, data: &[f32]) {
        self.sq8.train(data);
    }

    /// 编码到 4-bit (每 byte 存2个值)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let codes8 = self.sq8.encode(vector);
        let mut result = Vec::with_capacity(codes8.len() / 2);

        for chunk in codes8.chunks(2) {
            let low = chunk.first().unwrap_or(&0);
            let high = chunk.get(1).unwrap_or(&0);
            let byte = ((high & 0x0F) << 4) | (low & 0x0F);
            result.push(byte);
        }

        result
    }

    /// 从 4-bit 解码
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut codes8 = Vec::with_capacity(codes.len() * 2);

        for &byte in codes {
            codes8.push(byte & 0x0F);
            codes8.push((byte >> 4) & 0x0F);
        }

        self.sq8.decode(&codes8)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{_mm_add_ps, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps};
    let hi = std::arch::x86_64::_mm256_extractf128_ps(v, 1);
    let lo = std::arch::x86_64::_mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x55);
    let sums2 = _mm_add_ps(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq_l2_asymmetric_avx2(
    q_residual: &[f32],
    db_code: &[u8],
    inv_scale: f32,
    offset: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let len = q_residual.len().min(db_code.len());
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let inv_scale_vec = _mm256_set1_ps(inv_scale);
    let offset_vec = _mm256_set1_ps(offset);

    while i + 8 <= len {
        let qv = _mm256_loadu_ps(q_residual.as_ptr().add(i));

        // Load 8 u8 and widen to 8 i32 lanes.
        let bytes64 = _mm_loadl_epi64(db_code.as_ptr().add(i) as *const __m128i);
        let codes_i32 = _mm256_cvtepu8_epi32(bytes64);
        let dbf = _mm256_cvtepi32_ps(codes_i32);

        let db_res = _mm256_add_ps(_mm256_mul_ps(dbf, inv_scale_vec), offset_vec);
        let diff = _mm256_sub_ps(qv, db_res);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        i += 8;
    }

    // SAFETY: same target feature context and valid register value.
    let mut sum = unsafe { hsum_avx2(acc) };
    for (&qr, &db) in q_residual[i..len].iter().zip(db_code[i..len].iter()) {
        let db_residual = db as f32 * inv_scale + offset;
        let diff = qr - db_residual;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_dot_avx2(
    codes: &[u8],
    query: &[f32],
    scale: f32,
    offset: f32,
    min_val: f32,
    max_val: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let len = codes.len().min(query.len());
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();
    let v_inv_scale = _mm256_set1_ps(1.0 / scale);
    let v_offset = _mm256_set1_ps(offset);
    let v_min = _mm256_set1_ps(min_val);
    let v_max = _mm256_set1_ps(max_val);

    for i in 0..chunks {
        let off = i * 8;
        let c8 = _mm_loadl_epi64(codes.as_ptr().add(off) as *const __m128i);
        let c32 = _mm256_cvtepu8_epi32(c8);
        let cf = _mm256_cvtepi32_ps(c32);

        // decode: c / scale + offset = c * (1 / scale) + offset
        let decoded = _mm256_add_ps(_mm256_mul_ps(cf, v_inv_scale), v_offset);
        let clamped = _mm256_min_ps(_mm256_max_ps(decoded, v_min), v_max);

        let q = _mm256_loadu_ps(query.as_ptr().add(off));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(clamped, q));
    }

    // SAFETY: same target feature context and valid register value.
    let mut result = unsafe { hsum_avx2(acc) };
    for i in (chunks * 8)..len {
        let v = (codes[i] as f32 / scale + offset).clamp(min_val, max_val);
        result += v * query[i];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_dot_avx2_fma(
    codes: &[u8],
    query: &[f32],
    scale: f32,
    offset: f32,
    min_val: f32,
    max_val: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let len = codes.len().min(query.len());
    let chunks4 = len / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let v_inv_scale = _mm256_set1_ps(1.0 / scale);
    let v_offset = _mm256_set1_ps(offset);
    let v_min = _mm256_set1_ps(min_val);
    let v_max = _mm256_set1_ps(max_val);

    for i in 0..chunks4 {
        let off = i * 32;

        let c8_0 = _mm_loadl_epi64(codes.as_ptr().add(off) as *const __m128i);
        let c32_0 = _mm256_cvtepu8_epi32(c8_0);
        let cf_0 = _mm256_cvtepi32_ps(c32_0);
        let decoded_0 = _mm256_fmadd_ps(cf_0, v_inv_scale, v_offset);
        let clamped_0 = _mm256_min_ps(_mm256_max_ps(decoded_0, v_min), v_max);
        let q0 = _mm256_loadu_ps(query.as_ptr().add(off));
        acc0 = _mm256_fmadd_ps(clamped_0, q0, acc0);

        let c8_1 = _mm_loadl_epi64(codes.as_ptr().add(off + 8) as *const __m128i);
        let c32_1 = _mm256_cvtepu8_epi32(c8_1);
        let cf_1 = _mm256_cvtepi32_ps(c32_1);
        let decoded_1 = _mm256_fmadd_ps(cf_1, v_inv_scale, v_offset);
        let clamped_1 = _mm256_min_ps(_mm256_max_ps(decoded_1, v_min), v_max);
        let q1 = _mm256_loadu_ps(query.as_ptr().add(off + 8));
        acc1 = _mm256_fmadd_ps(clamped_1, q1, acc1);

        let c8_2 = _mm_loadl_epi64(codes.as_ptr().add(off + 16) as *const __m128i);
        let c32_2 = _mm256_cvtepu8_epi32(c8_2);
        let cf_2 = _mm256_cvtepi32_ps(c32_2);
        let decoded_2 = _mm256_fmadd_ps(cf_2, v_inv_scale, v_offset);
        let clamped_2 = _mm256_min_ps(_mm256_max_ps(decoded_2, v_min), v_max);
        let q2 = _mm256_loadu_ps(query.as_ptr().add(off + 16));
        acc2 = _mm256_fmadd_ps(clamped_2, q2, acc2);

        let c8_3 = _mm_loadl_epi64(codes.as_ptr().add(off + 24) as *const __m128i);
        let c32_3 = _mm256_cvtepu8_epi32(c8_3);
        let cf_3 = _mm256_cvtepi32_ps(c32_3);
        let decoded_3 = _mm256_fmadd_ps(cf_3, v_inv_scale, v_offset);
        let clamped_3 = _mm256_min_ps(_mm256_max_ps(decoded_3, v_min), v_max);
        let q3 = _mm256_loadu_ps(query.as_ptr().add(off + 24));
        acc3 = _mm256_fmadd_ps(clamped_3, q3, acc3);
    }

    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let remaining_start = chunks4 * 32;
    let chunks_remain = (len - remaining_start) / 8;
    for i in 0..chunks_remain {
        let off = remaining_start + i * 8;
        let c8 = _mm_loadl_epi64(codes.as_ptr().add(off) as *const __m128i);
        let c32 = _mm256_cvtepu8_epi32(c8);
        let cf = _mm256_cvtepi32_ps(c32);
        let decoded = _mm256_fmadd_ps(cf, v_inv_scale, v_offset);
        let clamped = _mm256_min_ps(_mm256_max_ps(decoded, v_min), v_max);
        let q = _mm256_loadu_ps(query.as_ptr().add(off));
        acc = _mm256_fmadd_ps(clamped, q, acc);
    }

    // SAFETY: same target feature context and valid register value.
    let mut result = unsafe { hsum_avx2(acc) };
    let scalar_start = remaining_start + chunks_remain * 8;
    for i in scalar_start..len {
        let v = (codes[i] as f32 / scale + offset).clamp(min_val, max_val);
        result += v * query[i];
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn decode_dot_avx512(
    codes: &[u8],
    query: &[f32],
    scale: f32,
    offset: f32,
    min_val: f32,
    max_val: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let len = codes.len().min(query.len());
    let chunks = len / 16;
    let mut acc = _mm512_setzero_ps();
    let v_inv_scale = _mm512_set1_ps(1.0 / scale);
    let v_offset = _mm512_set1_ps(offset);
    let v_min = _mm512_set1_ps(min_val);
    let v_max = _mm512_set1_ps(max_val);

    for i in 0..chunks {
        let off = i * 16;
        let c16 = _mm_loadu_si128(codes.as_ptr().add(off) as *const __m128i);
        let c32 = _mm512_cvtepu8_epi32(c16);
        let cf = _mm512_cvtepi32_ps(c32);

        let decoded = _mm512_fmadd_ps(cf, v_inv_scale, v_offset);
        let clamped = _mm512_min_ps(_mm512_max_ps(decoded, v_min), v_max);

        let q = _mm512_loadu_ps(query.as_ptr().add(off));
        acc = _mm512_fmadd_ps(clamped, q, acc);
    }

    let mut result = _mm512_reduce_add_ps(acc);
    for i in (chunks * 16)..len {
        let v = (codes[i] as f32 / scale + offset).clamp(min_val, max_val);
        result += v * query[i];
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq_l2_precomputed_avx2(q_i16: &[i16], db_code: &[u8]) -> i64 {
    use std::arch::x86_64::*;

    let len = q_i16.len().min(db_code.len());
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    while i + 64 <= len {
        // chunk 0
        let qv0 = unsafe { _mm256_loadu_si256(q_i16.as_ptr().add(i) as *const __m256i) };
        let db0 = unsafe { _mm_loadu_si128(db_code.as_ptr().add(i) as *const __m128i) };
        let dbv0 = _mm256_cvtepu8_epi16(db0);
        let d0 = _mm256_sub_epi16(qv0, dbv0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));

        // chunk 1
        let qv1 = unsafe { _mm256_loadu_si256(q_i16.as_ptr().add(i + 16) as *const __m256i) };
        let db1 = unsafe { _mm_loadu_si128(db_code.as_ptr().add(i + 16) as *const __m128i) };
        let dbv1 = _mm256_cvtepu8_epi16(db1);
        let d1 = _mm256_sub_epi16(qv1, dbv1);
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));

        // chunk 2
        let qv2 = unsafe { _mm256_loadu_si256(q_i16.as_ptr().add(i + 32) as *const __m256i) };
        let db2 = unsafe { _mm_loadu_si128(db_code.as_ptr().add(i + 32) as *const __m128i) };
        let dbv2 = _mm256_cvtepu8_epi16(db2);
        let d2 = _mm256_sub_epi16(qv2, dbv2);
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d2, d2));

        // chunk 3
        let qv3 = unsafe { _mm256_loadu_si256(q_i16.as_ptr().add(i + 48) as *const __m256i) };
        let db3 = unsafe { _mm_loadu_si128(db_code.as_ptr().add(i + 48) as *const __m128i) };
        let dbv3 = _mm256_cvtepu8_epi16(db3);
        let d3 = _mm256_sub_epi16(qv3, dbv3);
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d3, d3));

        i += 64;
    }

    let mut acc = _mm256_add_epi32(acc0, _mm256_add_epi32(acc1, _mm256_add_epi32(acc2, acc3)));

    while i + 16 <= len {
        let qv = unsafe { _mm256_loadu_si256(q_i16.as_ptr().add(i) as *const __m256i) };
        let db_bytes = unsafe { _mm_loadu_si128(db_code.as_ptr().add(i) as *const __m128i) };
        let dbv = _mm256_cvtepu8_epi16(db_bytes);
        let diff = _mm256_sub_epi16(qv, dbv);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff, diff));
        i += 16;
    }

    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let mut lanes = [0i32; 4];
    // SAFETY: lanes has exact space for one 128-bit store.
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, sum128) };
    let mut total = lanes.iter().map(|&v| v as i64).sum::<i64>();

    for (&qv, &db) in q_i16[i..len].iter().zip(db_code[i..len].iter()) {
        let diff = qv as i32 - db as i32;
        total += (diff * diff) as i64;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn sq_l2_precomputed_avx512(q_i16: &[i16], db_code: &[u8]) -> i64 {
    use std::arch::x86_64::*;

    let len = q_i16.len().min(db_code.len());
    let mut i = 0usize;
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    while i + 128 <= len {
        // chunk 0: 32 elements
        let q0 = _mm512_loadu_si512(q_i16.as_ptr().add(i) as *const __m512i);
        let cbytes0 = _mm256_loadu_si256(db_code.as_ptr().add(i) as *const __m256i);
        let c0 = _mm512_cvtepu8_epi16(cbytes0);
        let d0 = _mm512_sub_epi16(q0, c0);
        acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d0, d0));

        // chunk 1
        let q1 = _mm512_loadu_si512(q_i16.as_ptr().add(i + 32) as *const __m512i);
        let cbytes1 = _mm256_loadu_si256(db_code.as_ptr().add(i + 32) as *const __m256i);
        let c1 = _mm512_cvtepu8_epi16(cbytes1);
        let d1 = _mm512_sub_epi16(q1, c1);
        acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d1, d1));

        // chunk 2
        let q2 = _mm512_loadu_si512(q_i16.as_ptr().add(i + 64) as *const __m512i);
        let cbytes2 = _mm256_loadu_si256(db_code.as_ptr().add(i + 64) as *const __m256i);
        let c2 = _mm512_cvtepu8_epi16(cbytes2);
        let d2 = _mm512_sub_epi16(q2, c2);
        acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d2, d2));

        // chunk 3
        let q3 = _mm512_loadu_si512(q_i16.as_ptr().add(i + 96) as *const __m512i);
        let cbytes3 = _mm256_loadu_si256(db_code.as_ptr().add(i + 96) as *const __m256i);
        let c3 = _mm512_cvtepu8_epi16(cbytes3);
        let d3 = _mm512_sub_epi16(q3, c3);
        acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d3, d3));

        i += 128;
    }

    let mut acc = _mm512_add_epi32(acc0, _mm512_add_epi32(acc1, _mm512_add_epi32(acc2, acc3)));

    while i + 32 <= len {
        let q = _mm512_loadu_si512(q_i16.as_ptr().add(i) as *const __m512i);
        let cbytes = _mm256_loadu_si256(db_code.as_ptr().add(i) as *const __m256i);
        let c = _mm512_cvtepu8_epi16(cbytes);
        let d = _mm512_sub_epi16(q, c);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(d, d));
        i += 32;
    }

    let mut lanes = [0i32; 16];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, acc);
    let mut total = lanes.iter().map(|&v| v as i64).sum::<i64>();

    for (&qv, &db) in q_i16[i..len].iter().zip(db_code[i..len].iter()) {
        let diff = qv as i32 - db as i32;
        total += (diff * diff) as i64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_new() {
        let sq = ScalarQuantizer::new(128, 8);
        assert_eq!(sq.bit, 8);
    }

    #[test]
    fn test_sq8_train() {
        let mut sq = ScalarQuantizer::new(4, 8);
        let data = vec![0.0, 1.0, 2.0, 3.0];

        sq.train(&data);

        assert_eq!(sq.min_val, 0.0);
        assert_eq!(sq.max_val, 3.0);
    }

    #[test]
    fn test_sq8_encode_decode() {
        let mut sq = ScalarQuantizer::new(4, 8);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        sq.train(&data);

        let codes = sq.encode(&data);
        assert_eq!(codes.len(), 4);

        let decoded = sq.decode(&codes);
        assert_eq!(decoded.len(), 4);

        // 检查误差
        let error = sq.compute_error(&data, &decoded);
        assert!(error < 0.1, "Error {} too large", error);
    }

    #[test]
    fn test_sq4() {
        let mut sq4 = Sq4Quantizer::new(4);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        sq4.train(&data);

        let codes = sq4.encode(&data);
        assert_eq!(codes.len(), 2);

        let decoded = sq4.decode(&codes);
        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn test_sq_l2_asymmetric_avx2_matches_scalar() {
        let dim = 33usize; // non-multiple of 8 to exercise tail path
        let mut sq = ScalarQuantizer::new(dim, 8);
        let train: Vec<f32> = (0..dim).map(|i| i as f32 * 0.25 - 3.0).collect();
        sq.train(&train);

        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
        let db: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.21).cos() * 4.0).collect();
        let db_code = sq.encode(&db);
        let inv_scale = 1.0 / sq.scale;

        let scalar: f32 = q
            .iter()
            .zip(db_code.iter())
            .map(|(&qr, &dbq)| {
                let db_residual = dbq as f32 * inv_scale + sq.offset;
                let diff = qr - db_residual;
                diff * diff
            })
            .sum();
        let dispatch = sq.sq_l2_asymmetric(&q, &db_code);
        assert!(
            (dispatch - scalar).abs() < 1e-4,
            "dispatch={} scalar={}",
            dispatch,
            scalar
        );

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by AVX2 runtime check.
            let avx = unsafe { sq_l2_asymmetric_avx2(&q, &db_code, inv_scale, sq.offset) };
            assert!((avx - scalar).abs() < 1e-4, "avx={} scalar={}", avx, scalar);
        }
    }

    #[test]
    fn test_sq_l2_precomputed_matches_asymmetric() {
        let dim = 128usize;
        let mut sq = ScalarQuantizer::new(dim, 8);
        let train: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).sin() * 4.0).collect();
        sq.train(&train);

        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.07).cos() * 3.0).collect();
        let db: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin() * 2.5).collect();
        let db_code = sq.encode(&db);

        let q_i16 = sq.precompute_query(&q);
        let int_dist = sq.sq_l2_precomputed(&q_i16, &db_code);
        let float_dist = sq.sq_l2_asymmetric(&q, &db_code);
        assert!(
            (int_dist - float_dist).abs() < 1.0,
            "int_dist={} float_dist={}",
            int_dist,
            float_dist
        );
    }

    #[test]
    fn test_decode_dot_matches_scalar() {
        let dim = 768usize;
        let mut sq = ScalarQuantizer::new(dim, 8);
        let train: Vec<f32> = (0..(dim * 2))
            .map(|i| (i as f32 * 0.013).sin() * 10.0)
            .collect();
        sq.train(&train);

        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.021).cos() * 5.0).collect();
        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.017).sin() * 3.0).collect();
        let code = sq.encode(&v);

        let decoded = sq.decode(&code);
        let scalar: f32 = decoded.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum();
        let fused = sq.decode_dot_f32(&code, &q);

        assert!(
            (scalar - fused).abs() < 1e-3,
            "scalar={} fused={}",
            scalar,
            fused
        );
    }
}

#[inline]
fn top_bit_position(lane: usize) -> usize {
    ((lane & 7) << 3) | (lane >> 3)
}

#[inline]
fn read_top_bit(top_bits: u64, lane: usize) -> u8 {
    ((top_bits >> top_bit_position(lane)) & 1) as u8
}

#[inline]
fn read_u64_le(bytes: &[u8]) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(raw)
}

#[inline]
fn decode_value_fxu8(code: &[u8], idx: usize) -> u8 {
    code[idx]
}

#[inline]
fn decode_value_fxu4(code: &[u8], idx: usize) -> u8 {
    let block = idx / 32;
    let lane = idx % 32;
    let packed = code[block * 16 + (lane & 15)];
    if lane < 16 {
        packed & 0x0F
    } else {
        (packed >> 4) & 0x0F
    }
}

#[inline]
fn decode_value_fxu2(code: &[u8], idx: usize) -> u8 {
    let block = idx / 64;
    let lane = idx % 64;
    let packed = code[block * 16 + (lane & 15)];
    (packed >> (2 * (lane / 16))) & 0x03
}

#[inline]
fn decode_value_fxu3(code: &[u8], idx: usize) -> u8 {
    let block = idx / 64;
    let lane = idx % 64;
    let block_offset = block * 24;
    let packed = code[block_offset + (lane & 15)];
    let base = (packed >> (2 * (lane / 16))) & 0x03;
    let top_bits = read_u64_le(&code[block_offset + 16..block_offset + 24]);
    base | (read_top_bit(top_bits, lane) << 2)
}

#[inline]
fn decode_value_fxu6(code: &[u8], idx: usize) -> u8 {
    let block = idx / 64;
    let lane = idx % 64;
    let block_offset = block * 48;
    match lane {
        0..=15 => code[block_offset + lane] & 0x3F,
        16..=31 => code[block_offset + 16 + (lane - 16)] & 0x3F,
        32..=47 => {
            let lane = lane - 32;
            let hi2 = (code[block_offset + lane] >> 6) & 0x03;
            let lo4 = code[block_offset + 32 + lane] & 0x0F;
            lo4 | (hi2 << 4)
        }
        _ => {
            let lane = lane - 48;
            let hi2 = (code[block_offset + 16 + lane] >> 6) & 0x03;
            let lo4 = (code[block_offset + 32 + lane] >> 4) & 0x0F;
            lo4 | (hi2 << 4)
        }
    }
}

#[inline]
fn decode_value_fxu7(code: &[u8], idx: usize) -> u8 {
    let block = idx / 64;
    let lane = idx % 64;
    let block_offset = block * 56;
    let base = decode_value_fxu6(&code[block_offset..block_offset + 48], lane);
    let top_bits = read_u64_le(&code[block_offset + 48..block_offset + 56]);
    base | (read_top_bit(top_bits, lane) << 6)
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn decode_chunk_with<F>(dim_base: usize, mut decode_value: F) -> [u8; 16]
where
    F: FnMut(usize) -> u8,
{
    let mut decoded = [0u8; 16];
    for (lane, value) in decoded.iter_mut().enumerate() {
        *value = decode_value(dim_base + lane);
    }
    decoded
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,fma")]
unsafe fn ip_avx512_with_decoder<F>(query: &[f32], d: usize, mut decode_chunk: F) -> f32
where
    F: FnMut(usize) -> [u8; 16],
{
    use std::arch::x86_64::*;

    let num16 = d & !15;
    let mut sum = _mm512_setzero_ps();
    for i in (0..num16).step_by(16) {
        let decoded = decode_chunk(i);
        let q = _mm512_loadu_ps(query.as_ptr().add(i));
        let codes = _mm_loadu_si128(decoded.as_ptr() as *const __m128i);
        let codes_f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes));
        sum = _mm512_fmadd_ps(q, codes_f, sum);
    }
    _mm512_reduce_add_ps(sum)
}

pub(crate) fn decode_compact_levels(code: &[u8], d: usize, ex_bits: usize) -> Vec<u8> {
    let mut decoded = vec![0u8; d];
    match ex_bits {
        0 => {}
        8 => {
            decoded.copy_from_slice(&code[..d]);
        }
        4 => {
            assert_eq!(d % 32, 0);
            for (idx, dst) in decoded.iter_mut().enumerate() {
                *dst = decode_value_fxu4(code, idx);
            }
        }
        2 => {
            assert_eq!(d % 64, 0);
            for (idx, dst) in decoded.iter_mut().enumerate() {
                *dst = decode_value_fxu2(code, idx);
            }
        }
        3 => {
            assert_eq!(d % 64, 0);
            for (idx, dst) in decoded.iter_mut().enumerate() {
                *dst = decode_value_fxu3(code, idx);
            }
        }
        6 => {
            assert_eq!(d % 64, 0);
            for (idx, dst) in decoded.iter_mut().enumerate() {
                *dst = decode_value_fxu6(code, idx);
            }
        }
        7 => {
            assert_eq!(d % 64, 0);
            for (idx, dst) in decoded.iter_mut().enumerate() {
                *dst = decode_value_fxu7(code, idx);
            }
        }
        bits => unreachable!("unsupported ex_bits {bits}"),
    }
    decoded
}

fn ip_scalar_with_decoder<F>(query: &[f32], d: usize, mut decode_value: F) -> f32
where
    F: FnMut(usize) -> u8,
{
    let mut sum = 0.0f32;
    for (idx, &value) in query[..d].iter().enumerate() {
        sum += value * decode_value(idx) as f32;
    }
    sum
}

pub(crate) fn ip_fxu8(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert!(code.len() >= d);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        let num16 = d & !15;
        let mut sum = unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu8(code, idx))
            })
        };
        for idx in num16..d {
            sum += query[idx] * decode_value_fxu8(code, idx) as f32;
        }
        return sum;
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu8(code, idx))
}

pub(crate) fn ip_fxu4(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert_eq!(d % 32, 0);
    assert!(code.len() >= d / 2);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu4(code, idx))
            })
        };
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu4(code, idx))
}

pub(crate) fn ip_fxu2(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert_eq!(d % 64, 0);
    assert!(code.len() >= d / 4);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu2(code, idx))
            })
        };
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu2(code, idx))
}

pub(crate) fn ip_fxu3(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert_eq!(d % 64, 0);
    assert!(code.len() >= (d / 64) * 24);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu3(code, idx))
            })
        };
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu3(code, idx))
}

pub(crate) fn ip_fxu6(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert_eq!(d % 64, 0);
    assert!(code.len() >= (d / 64) * 48);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu6(code, idx))
            })
        };
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu6(code, idx))
}

pub(crate) fn ip_fxu7(query: &[f32], code: &[u8], d: usize) -> f32 {
    assert!(query.len() >= d);
    assert_eq!(d % 64, 0);
    assert!(code.len() >= (d / 64) * 56);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe {
            ip_avx512_with_decoder(query, d, |dim_base| {
                decode_chunk_with(dim_base, |idx| decode_value_fxu7(code, idx))
            })
        };
    }
    ip_scalar_with_decoder(query, d, |idx| decode_value_fxu7(code, idx))
}

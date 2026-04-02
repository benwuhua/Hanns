use std::collections::BinaryHeap;

use super::config::UsqConfig;
use super::layout::UsqLayout;

/// Precomputed LUT for fast 1-bit candidate scoring.
#[derive(Clone, Debug)]
pub struct UsqFastScanState {
    /// Quantized LUT: `n_groups * 16` i8 entries.
    /// `lut[g*16 + nibble]` = dot(q_rot[g*4..g*4+4], sign(nibble)) quantized to i8.
    pub lut: Vec<i8>,
    /// Scale factor: `raw_i32_score * lut_scale` ≈ float dot product.
    pub lut_scale: f32,
}

impl UsqFastScanState {
    /// Build a fastscan LUT from an already-rotated query.
    ///
    /// For each group of 4 consecutive padded dimensions, enumerates all 16 sign
    /// combinations (nibbles 0..16) and records the dot product with `q_rot`.
    /// All float values are then globally quantized to i8.
    pub fn new(q_rot: &[f32], config: &UsqConfig) -> Self {
        let padded = config.padded_dim();
        debug_assert_eq!(q_rot.len(), padded, "q_rot length must equal padded_dim");

        let n_groups = padded / 4; // padded_dim is always a multiple of 4 (multiple of 64)
        let mut lut_f32 = vec![0.0f32; n_groups * 16];

        for g in 0..n_groups {
            let base = g * 4;
            // Load the 4 query components for this group (safe: padded ensures in-bounds).
            let q = [q_rot[base], q_rot[base + 1], q_rot[base + 2], q_rot[base + 3]];
            for nibble in 0..16usize {
                let mut val = 0.0f32;
                for bit in 0..4usize {
                    let sign = if (nibble >> bit) & 1 != 0 { 1.0f32 } else { -1.0f32 };
                    val += q[bit] * sign;
                }
                lut_f32[g * 16 + nibble] = val;
            }
        }

        // Global quantize to i8.
        let max_abs = lut_f32
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let lut_scale = (max_abs / 127.0).max(1e-6);
        let lut: Vec<i8> = lut_f32
            .iter()
            .map(|&v| (v / lut_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        UsqFastScanState { lut, lut_scale }
    }
}

/// A candidate from the fastscan pass.
#[derive(Clone, Copy, Debug)]
pub struct FsCandidate {
    /// Vector index within the layout.
    pub idx: usize,
    /// Raw 1-bit approximate score (i32 sum of quantized LUT lookups).
    pub raw_score: i32,
}

// ─── Heap ordering ─────────────────────────────────────────────────────────────
// We want the TOP-n by raw_score, so we maintain a min-heap of size n (evict the
// smallest element when full).  BinaryHeap is a max-heap, so we wrap in Reverse.

use std::cmp::Reverse;

/// Scan all fastscan blocks in `layout` and return up to `top_n` candidates
/// with the highest raw scores (sorted descending).
pub fn fastscan_topk(layout: &UsqLayout, state: &UsqFastScanState, top_n: usize) -> Vec<FsCandidate> {
    if top_n == 0 || layout.is_empty() {
        return Vec::new();
    }

    // Min-heap: Reverse((raw_score, idx)) — evict lowest score when full.
    let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::with_capacity(top_n + 1);

    #[cfg(target_arch = "x86_64")]
    let use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let n_groups = layout.padded_dim() / 4;

    for block_idx in 0..layout.n_blocks() {
        let block = layout.fastscan_block(block_idx);

        #[cfg(target_arch = "x86_64")]
        let raw_scores = if use_avx512 {
            // SAFETY: feature detection done above, block is valid slice.
            unsafe { fastscan_block_avx512(state, block, n_groups) }
        } else {
            fastscan_block_scalar(state, block, n_groups)
        };

        #[cfg(not(target_arch = "x86_64"))]
        let raw_scores = fastscan_block_scalar(state, block, n_groups);

        for (slot, &raw_score) in raw_scores.iter().enumerate() {
            let vid = block_idx * 32 + slot;
            if vid >= layout.len() {
                continue;
            }

            let entry = Reverse((raw_score, vid));
            if heap.len() < top_n {
                heap.push(entry);
            } else if let Some(&Reverse((worst_score, worst_id))) = heap.peek() {
                if raw_score > worst_score || (raw_score == worst_score && vid < worst_id) {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }
    }

    // Drain heap and sort descending by score.
    let mut results: Vec<FsCandidate> = heap
        .into_iter()
        .map(|Reverse((raw_score, idx))| FsCandidate { idx, raw_score })
        .collect();
    results.sort_by(|a, b| b.raw_score.cmp(&a.raw_score).then_with(|| a.idx.cmp(&b.idx)));
    results
}

// ─── Scalar block scan ─────────────────────────────────────────────────────────

fn fastscan_block_scalar(state: &UsqFastScanState, block: &[u8], n_groups: usize) -> [i32; 32] {
    let mut scores = [0i32; 32];
    for group_idx in 0..n_groups {
        let lut = &state.lut[group_idx * 16..(group_idx + 1) * 16];
        let group_base = group_idx * 16;
        for slot in 0..32usize {
            let byte = block[group_base + slot / 2];
            let nibble = if slot % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            scores[slot] += lut[nibble as usize] as i32;
        }
    }
    scores
}

// ─── AVX-512 block scan ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx2,ssse3")]
unsafe fn fastscan_block_avx512(
    state: &UsqFastScanState,
    block: &[u8],
    n_groups: usize,
) -> [i32; 32] {
    use std::arch::x86_64::*;

    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let nibble_mask_128 = _mm_set1_epi8(0x0F_u8 as i8);

    for group_idx in 0..n_groups {
        let group_offset = group_idx * 16;

        let lut_ptr = state.lut.as_ptr().add(group_idx * 16) as *const __m128i;
        let lut_128 = _mm_loadu_si128(lut_ptr);

        // Low half of the 16-byte group (8 bytes → 16 slots).
        let data_lo_ptr = block.as_ptr().add(group_offset) as *const __m128i;
        let data_lo_64 = _mm_loadl_epi64(data_lo_ptr);
        let lo_nibbles = _mm_and_si128(data_lo_64, nibble_mask_128);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_lo_64, 4), nibble_mask_128);
        let interleaved_lo = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
        let partial_lo = _mm_shuffle_epi8(lut_128, interleaved_lo);
        let partial_lo_i16 = _mm256_cvtepi8_epi16(partial_lo);
        acc_lo = _mm256_add_epi16(acc_lo, partial_lo_i16);

        // High half of the 16-byte group (bytes 8..16 → slots 16..32).
        let data_hi_ptr = block.as_ptr().add(group_offset + 8) as *const __m128i;
        let data_hi_64 = _mm_loadl_epi64(data_hi_ptr);
        let hi_lo_nibbles = _mm_and_si128(data_hi_64, nibble_mask_128);
        let hi_hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_hi_64, 4), nibble_mask_128);
        let interleaved_hi = _mm_unpacklo_epi8(hi_lo_nibbles, hi_hi_nibbles);
        let partial_hi = _mm_shuffle_epi8(lut_128, interleaved_hi);
        let partial_hi_i16 = _mm256_cvtepi8_epi16(partial_hi);
        acc_hi = _mm256_add_epi16(acc_hi, partial_hi_i16);
    }

    let mut scores = [0i32; 32];

    let lo_128_lo = _mm256_castsi256_si128(acc_lo);
    let lo_128_hi = _mm256_extracti128_si256(acc_lo, 1);
    let lo_i32_0 = _mm256_cvtepi16_epi32(lo_128_lo);
    let lo_i32_1 = _mm256_cvtepi16_epi32(lo_128_hi);
    _mm256_storeu_si256(scores.as_mut_ptr() as *mut __m256i, lo_i32_0);
    _mm256_storeu_si256(scores.as_mut_ptr().add(8) as *mut __m256i, lo_i32_1);

    let hi_128_lo = _mm256_castsi256_si128(acc_hi);
    let hi_128_hi = _mm256_extracti128_si256(acc_hi, 1);
    let hi_i32_0 = _mm256_cvtepi16_epi32(hi_128_lo);
    let hi_i32_1 = _mm256_cvtepi16_epi32(hi_128_hi);
    _mm256_storeu_si256(scores.as_mut_ptr().add(16) as *mut __m256i, hi_i32_0);
    _mm256_storeu_si256(scores.as_mut_ptr().add(24) as *mut __m256i, hi_i32_1);

    scores
}

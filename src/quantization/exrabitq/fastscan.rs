use super::{ExRaBitQLayout, FAST_SIZE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const HIGH_ACC_CONST_BOUND: f32 = 0.58;

#[derive(Clone, Debug)]
pub struct ExRaBitQFastScanState {
    pub use_high_accuracy: bool,
    pub residual: Vec<f32>,
    pub unit_query: Vec<f32>,
    pub y2: f32,
    pub y: f32,
    pub lut: Vec<f32>,
    pub half_sum_residual: f32,
    pub sumq: f32,
    pub vl: f32,
    pub width: f32,
    pub delta: f32,
    pub one_over_sqrt_d: f32,
    lut_lower: Vec<u8>,
    lut_upper: Vec<u8>,
    lut_shift: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredCandidate {
    pub idx: usize,
    pub distance: f32,
    pub rabitq_ip: f32,
}

impl Eq for ScoredCandidate {}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ExRaBitQFastScanState {
    pub fn new(q_rot_residual: &[f32], y2: f32) -> Self {
        let half_sum_residual = 0.5 * q_rot_residual.iter().sum::<f32>();
        let y = y2.sqrt();
        let (vl, vr) = q_rot_residual.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min_v, max_v), value| (min_v.min(value), max_v.max(value)),
        );
        let width = if !vl.is_finite() || !vr.is_finite() || (vr - vl).abs() <= 1e-12 {
            1.0
        } else {
            (vr - vl) / ((1u32 << 14) as f32 - 1.0)
        };
        let n_groups = q_rot_residual.len().div_ceil(4);
        let mut lut = vec![0.0f32; n_groups * 16];
        let mut lut_lower = vec![0u8; n_groups * 16];
        let mut lut_upper = vec![0u8; n_groups * 16];
        for group_idx in 0..n_groups {
            let mut group = [0u16; 4];
            for bit_pos in 0..4usize {
                let dim_idx = group_idx * 4 + bit_pos;
                if dim_idx < q_rot_residual.len() {
                    group[bit_pos] = quantize_query_value(q_rot_residual[dim_idx], vl, width);
                }
            }
            for nibble in 0..16usize {
                let mut value = 0u32;
                for bit_pos in 0..4usize {
                    if ((nibble >> bit_pos) & 1) != 0 {
                        value += group[bit_pos] as u32;
                    }
                }
                lut[group_idx * 16 + nibble] = value as f32;
                lut_lower[group_idx * 16 + nibble] = value as u8;
                lut_upper[group_idx * 16 + nibble] = (value >> 8) as u8;
            }
        }

        Self {
            use_high_accuracy: false,
            residual: q_rot_residual.to_vec(),
            unit_query: Vec::new(),
            y2,
            y,
            lut,
            half_sum_residual,
            sumq: 0.0,
            vl,
            width,
            delta: 0.0,
            one_over_sqrt_d: 1.0 / (q_rot_residual.len() as f32).sqrt(),
            lut_lower,
            lut_upper,
            lut_shift: 0,
        }
    }

    pub fn new_high_accuracy(q_rot_residual: &[f32], y2: f32) -> Self {
        let y = y2.sqrt();
        let padded_dim = q_rot_residual.len();
        let one_over_sqrt_d = 1.0 / (padded_dim as f32).sqrt();
        let (unit_query, sumq) = if y > 1e-5 {
            let unit: Vec<f32> = q_rot_residual.iter().map(|value| *value / y).collect();
            let sum = unit.iter().sum::<f32>();
            (unit, sum)
        } else {
            let value = one_over_sqrt_d;
            (vec![value; padded_dim], padded_dim as f32 * value)
        };

        let vmax = unit_query
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let delta = if vmax <= 1e-12 {
            1.0
        } else {
            vmax / (((1u32 << 13) - 1) as f32)
        };

        let n_groups = padded_dim.div_ceil(4);
        let mut lut = vec![0.0f32; n_groups * 16];
        let mut lut_lower = vec![0u8; n_groups * 16];
        let mut lut_upper = vec![0u8; n_groups * 16];
        let mut lut_shift = 0i32;
        for group_idx in 0..n_groups {
            let mut group = [0i16; 4];
            for bit_pos in 0..4usize {
                let dim_idx = group_idx * 4 + bit_pos;
                if dim_idx < padded_dim {
                    group[bit_pos] = quantize_signed_query_value(unit_query[dim_idx], delta);
                }
            }
            for nibble in 0..16usize {
                let mut value = 0i32;
                for bit_pos in 0..4usize {
                    if ((nibble >> bit_pos) & 1) != 0 {
                        value += group[bit_pos] as i32;
                    }
                }
                lut[group_idx * 16 + nibble] = value as f32;
            }

            let group_slice = &lut[group_idx * 16..(group_idx + 1) * 16];
            let group_min = group_slice
                .iter()
                .copied()
                .map(|value| value as i32)
                .min()
                .unwrap_or(0);
            lut_shift += group_min;
            for nibble in 0..16usize {
                let shifted = (group_slice[nibble] as i32 - group_min) as u32;
                lut_lower[group_idx * 16 + nibble] = shifted as u8;
                lut_upper[group_idx * 16 + nibble] = (shifted >> 8) as u8;
            }
        }

        Self {
            use_high_accuracy: true,
            residual: q_rot_residual.to_vec(),
            unit_query,
            y2,
            y,
            lut,
            half_sum_residual: 0.5 * q_rot_residual.iter().sum::<f32>(),
            sumq,
            vl: 0.0,
            width: 0.0,
            delta,
            one_over_sqrt_d,
            lut_lower,
            lut_upper,
            lut_shift,
        }
    }
}

pub fn reference_short_distance(
    state: &ExRaBitQFastScanState,
    short_code: &[u8],
    short_ip_factor: f32,
    sum_xb: f32,
    short_err: f32,
    x2: f32,
) -> f32 {
    if state.use_high_accuracy {
        let mut selected_sum = 0i32;
        for dim_idx in 0..state.unit_query.len() {
            let byte = short_code[dim_idx / 8];
            let bit = (byte >> (dim_idx % 8)) & 1;
            if bit != 0 {
                selected_sum +=
                    quantize_signed_query_value(state.unit_query[dim_idx], state.delta) as i32;
            }
        }
        let ip_xb_qprime = selected_sum as f32 * state.delta;
        return x2 + state.y2
            - 5.0
                * state.y
                * x2.sqrt()
                * state.one_over_sqrt_d
                * (ip_xb_qprime - 0.5 * state.sumq + HIGH_ACC_CONST_BOUND);
    }

    let mut selected_sum = 0u32;
    for dim_idx in 0..state.residual.len() {
        let byte = short_code[dim_idx / 8];
        let bit = (byte >> (dim_idx % 8)) & 1;
        if bit != 0 {
            selected_sum +=
                quantize_query_value(state.residual[dim_idx], state.vl, state.width) as u32;
        }
    }
    let rabitq_ip = selected_sum as f32 * state.width + sum_xb * state.vl - state.half_sum_residual;
    x2 + state.y2 - (short_ip_factor * rabitq_ip + short_err)
}

pub fn scalar_scan_layout(
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Vec<ScoredCandidate> {
    if top_k == 0 || layout.is_empty() {
        return Vec::new();
    }

    let mut heap = BinaryHeap::with_capacity(top_k + 1);
    let n_groups = layout.padded_dim().div_ceil(4);

    for block_idx in 0..layout.n_blocks() {
        let block = layout.fastscan_block(block_idx);
        let mut selected_sums = [0.0f32; FAST_SIZE];

        for group_idx in 0..n_groups {
            let group_base = group_idx * 16;
            let lut = &state.lut[group_idx * 16..(group_idx + 1) * 16];
            for slot in 0..FAST_SIZE {
                let byte = block[group_base + slot / 2];
                let nibble = if slot % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                selected_sums[slot] += lut[nibble as usize];
            }
        }

        for (slot, &selected_sum) in selected_sums.iter().enumerate() {
            let idx = block_idx * FAST_SIZE + slot;
            if idx >= layout.len() {
                continue;
            }

            let (rabitq_ip, distance) = if state.use_high_accuracy {
                let ip_xb_qprime = selected_sum * state.delta;
                let distance = layout.x2_at(idx) + state.y2
                    - 5.0
                        * state.y
                        * layout.x_norm_at(idx)
                        * state.one_over_sqrt_d
                        * (ip_xb_qprime - 0.5 * state.sumq + HIGH_ACC_CONST_BOUND);
                (ip_xb_qprime, distance)
            } else {
                let rabitq_ip = selected_sum * state.width + layout.short_sum_xb_at(idx) * state.vl
                    - state.half_sum_residual;
                let distance = layout.x2_at(idx) + state.y2
                    - (layout.short_ip_at(idx) * rabitq_ip + layout.short_err_at(idx));
                (rabitq_ip, distance)
            };

            let candidate = ScoredCandidate {
                idx,
                distance,
                rabitq_ip,
            };
            if heap.len() < top_k {
                heap.push(candidate);
                continue;
            }
            if let Some(worst) = heap.peek() {
                if candidate.distance < worst.distance
                    || (candidate.distance == worst.distance && candidate.idx < worst.idx)
                {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        out.push(item);
    }
    out.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.idx.cmp(&b.idx))
    });
    out
}

pub fn scan_layout(
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Vec<ScoredCandidate> {
    if let Some(simd) = simd_scan_layout(layout, state, top_k) {
        return simd;
    }
    scalar_scan_layout(layout, state, top_k)
}

#[cfg(target_arch = "x86_64")]
pub fn simd_scan_layout(
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Option<Vec<ScoredCandidate>> {
    if !std::arch::is_x86_feature_detected!("avx512bw")
        || !std::arch::is_x86_feature_detected!("avx512f")
    {
        return None;
    }
    Some(unsafe { scan_layout_avx512(layout, state, top_k) })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_scan_layout(
    _layout: &ExRaBitQLayout,
    _state: &ExRaBitQFastScanState,
    _top_k: usize,
) -> Option<Vec<ScoredCandidate>> {
    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f,avx2,ssse3")]
unsafe fn scan_layout_avx512(
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Vec<ScoredCandidate> {
    if top_k == 0 || layout.is_empty() {
        return Vec::new();
    }

    let mut heap = BinaryHeap::with_capacity(top_k + 1);
    for block_idx in 0..layout.n_blocks() {
        let block = layout.fastscan_block(block_idx);
        let selected_sums = unsafe { fastscan_block_avx512(state, block) };

        for (slot, &raw_selected_sum) in selected_sums.iter().enumerate() {
            let idx = block_idx * FAST_SIZE + slot;
            if idx >= layout.len() {
                continue;
            }

            let (rabitq_ip, distance) = if state.use_high_accuracy {
                let selected_sum = raw_selected_sum as i32 + state.lut_shift;
                let ip_xb_qprime = selected_sum as f32 * state.delta;
                let distance = layout.x2_at(idx) + state.y2
                    - 5.0
                        * state.y
                        * layout.x_norm_at(idx)
                        * state.one_over_sqrt_d
                        * (ip_xb_qprime - 0.5 * state.sumq + HIGH_ACC_CONST_BOUND);
                (ip_xb_qprime, distance)
            } else {
                let rabitq_ip = raw_selected_sum as f32 * state.width
                    + layout.short_sum_xb_at(idx) * state.vl
                    - state.half_sum_residual;
                let distance = layout.x2_at(idx) + state.y2
                    - (layout.short_ip_at(idx) * rabitq_ip + layout.short_err_at(idx));
                (rabitq_ip, distance)
            };

            let candidate = ScoredCandidate {
                idx,
                distance,
                rabitq_ip,
            };
            if heap.len() < top_k {
                heap.push(candidate);
                continue;
            }
            if let Some(worst) = heap.peek() {
                if candidate.distance < worst.distance
                    || (candidate.distance == worst.distance && candidate.idx < worst.idx)
                {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        out.push(item);
    }
    out.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.idx.cmp(&b.idx))
    });
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f,avx2,ssse3")]
unsafe fn fastscan_block_avx512(state: &ExRaBitQFastScanState, block: &[u8]) -> [u32; FAST_SIZE] {
    use std::arch::x86_64::*;

    debug_assert_eq!(block.len(), state.lut_lower.len());
    debug_assert_eq!(state.lut_lower.len(), state.lut_upper.len());

    let n_groups = state.lut_lower.len() / 16;
    let mut acc_lo_lower = _mm256_setzero_si256();
    let mut acc_lo_upper = _mm256_setzero_si256();
    let mut acc_hi_lower = _mm256_setzero_si256();
    let mut acc_hi_upper = _mm256_setzero_si256();
    let nibble_mask_128 = _mm_set1_epi8(0x0F_u8 as i8);

    for group_idx in 0..n_groups {
        let group_offset = group_idx * 16;
        let lut_lower_ptr =
            unsafe { state.lut_lower.as_ptr().add(group_idx * 16) } as *const __m128i;
        let lut_upper_ptr =
            unsafe { state.lut_upper.as_ptr().add(group_idx * 16) } as *const __m128i;
        let lut_lower_128 = unsafe { _mm_loadu_si128(lut_lower_ptr) };
        let lut_upper_128 = unsafe { _mm_loadu_si128(lut_upper_ptr) };

        let data_lo_ptr = unsafe { block.as_ptr().add(group_offset) } as *const __m128i;
        let data_lo_64 = unsafe { _mm_loadl_epi64(data_lo_ptr) };
        let lo_nibbles = _mm_and_si128(data_lo_64, nibble_mask_128);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_lo_64, 4), nibble_mask_128);
        let interleaved_lo = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
        let partial_lo_lower = _mm_shuffle_epi8(lut_lower_128, interleaved_lo);
        let partial_lo_upper = _mm_shuffle_epi8(lut_upper_128, interleaved_lo);
        acc_lo_lower = _mm256_add_epi16(acc_lo_lower, _mm256_cvtepu8_epi16(partial_lo_lower));
        acc_lo_upper = _mm256_add_epi16(acc_lo_upper, _mm256_cvtepu8_epi16(partial_lo_upper));

        let data_hi_ptr = unsafe { block.as_ptr().add(group_offset + 8) } as *const __m128i;
        let data_hi_64 = unsafe { _mm_loadl_epi64(data_hi_ptr) };
        let hi_lo_nibbles = _mm_and_si128(data_hi_64, nibble_mask_128);
        let hi_hi_nibbles = _mm_and_si128(_mm_srli_epi16(data_hi_64, 4), nibble_mask_128);
        let interleaved_hi = _mm_unpacklo_epi8(hi_lo_nibbles, hi_hi_nibbles);
        let partial_hi_lower = _mm_shuffle_epi8(lut_lower_128, interleaved_hi);
        let partial_hi_upper = _mm_shuffle_epi8(lut_upper_128, interleaved_hi);
        acc_hi_lower = _mm256_add_epi16(acc_hi_lower, _mm256_cvtepu8_epi16(partial_hi_lower));
        acc_hi_upper = _mm256_add_epi16(acc_hi_upper, _mm256_cvtepu8_epi16(partial_hi_upper));
    }

    unsafe { combine_accumulators(acc_lo_lower, acc_lo_upper, acc_hi_lower, acc_hi_upper) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512f,avx2,ssse3")]
unsafe fn combine_accumulators(
    acc_lo_lower: std::arch::x86_64::__m256i,
    acc_lo_upper: std::arch::x86_64::__m256i,
    acc_hi_lower: std::arch::x86_64::__m256i,
    acc_hi_upper: std::arch::x86_64::__m256i,
) -> [u32; FAST_SIZE] {
    use std::arch::x86_64::*;

    let mut scores = [0u32; FAST_SIZE];

    unsafe fn store_lane_pair(out: *mut u32, lower: __m256i, upper: __m256i) {
        use std::arch::x86_64::*;

        let lower_lo = _mm256_castsi256_si128(lower);
        let lower_hi = _mm256_extracti128_si256(lower, 1);
        let upper_lo = _mm256_castsi256_si128(upper);
        let upper_hi = _mm256_extracti128_si256(upper, 1);

        let part0 = _mm256_add_epi32(
            _mm256_cvtepu16_epi32(lower_lo),
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(upper_lo), 8),
        );
        let part1 = _mm256_add_epi32(
            _mm256_cvtepu16_epi32(lower_hi),
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(upper_hi), 8),
        );

        _mm256_storeu_si256(out as *mut __m256i, part0);
        _mm256_storeu_si256(out.add(8) as *mut __m256i, part1);
    }

    unsafe { store_lane_pair(scores.as_mut_ptr(), acc_lo_lower, acc_lo_upper) };
    unsafe { store_lane_pair(scores.as_mut_ptr().add(16), acc_hi_lower, acc_hi_upper) };
    scores
}

fn quantize_query_value(value: f32, vl: f32, width: f32) -> u16 {
    if width <= 1e-12 {
        return 0;
    }
    ((value - vl) / width + 0.5).clamp(0.0, ((1u32 << 14) - 1) as f32) as u16
}

fn quantize_signed_query_value(value: f32, delta: f32) -> i16 {
    if delta <= 1e-12 {
        return 0;
    }
    let scaled = value / delta;
    let rounded = scaled + 0.5 - if scaled < 0.0 { 1.0 } else { 0.0 };
    let limit = ((1u32 << 13) - 1) as f32;
    rounded.clamp(-limit, limit) as i16
}

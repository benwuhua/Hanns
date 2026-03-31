use super::{ExRaBitQLayout, ExRaBitQQuantizer, FAST_SIZE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const HIGH_ACC_CONST_BOUND: f32 = 0.58;
const HIGH_ACC_POS: [usize; 16] = [0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0];

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
    #[allow(dead_code)]
    lut_lower: Vec<u8>,
    #[allow(dead_code)]
    lut_upper: Vec<u8>,
    #[allow(dead_code)]
    lut_shift: i32,
    #[allow(dead_code)]
    packed_high_acc_lut: Vec<u8>,
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
        let mut state = Self {
            use_high_accuracy: false,
            residual: Vec::new(),
            unit_query: Vec::new(),
            y2: 0.0,
            y: 0.0,
            lut: Vec::new(),
            half_sum_residual: 0.0,
            sumq: 0.0,
            vl: 0.0,
            width: 0.0,
            delta: 0.0,
            one_over_sqrt_d: 1.0,
            lut_lower: Vec::new(),
            lut_upper: Vec::new(),
            lut_shift: 0,
            packed_high_acc_lut: Vec::new(),
        };
        state.reset(q_rot_residual, y2);
        state
    }

    pub fn new_high_accuracy(q_rot_residual: &[f32], y2: f32) -> Self {
        let mut state = Self {
            use_high_accuracy: true,
            residual: Vec::new(),
            unit_query: Vec::new(),
            y2: 0.0,
            y: 0.0,
            lut: Vec::new(),
            half_sum_residual: 0.0,
            sumq: 0.0,
            vl: 0.0,
            width: 0.0,
            delta: 0.0,
            one_over_sqrt_d: 1.0,
            lut_lower: Vec::new(),
            lut_upper: Vec::new(),
            lut_shift: 0,
            packed_high_acc_lut: Vec::new(),
        };
        state.reset(q_rot_residual, y2);
        state
    }

    pub fn reset(&mut self, q_rot_residual: &[f32], y2: f32) {
        self.residual.clear();
        self.residual.extend_from_slice(q_rot_residual);
        self.y2 = y2;
        self.y = y2.sqrt();
        self.half_sum_residual = 0.5 * q_rot_residual.iter().sum::<f32>();
        self.one_over_sqrt_d = 1.0 / (q_rot_residual.len() as f32).sqrt();

        if self.use_high_accuracy {
            self.reset_high_accuracy(q_rot_residual);
        } else {
            self.reset_standard(q_rot_residual);
        }
    }

    fn reset_standard(&mut self, q_rot_residual: &[f32]) {
        let (vl, vr) = q_rot_residual.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min_v, max_v), value| (min_v.min(value), max_v.max(value)),
        );
        self.vl = vl;
        self.width = if !vl.is_finite() || !vr.is_finite() || (vr - vl).abs() <= 1e-12 {
            1.0
        } else {
            (vr - vl) / ((1u32 << 14) as f32 - 1.0)
        };
        self.sumq = 0.0;
        self.delta = 0.0;
        self.lut_shift = 0;
        self.unit_query.clear();
        self.packed_high_acc_lut.clear();

        let n_groups = q_rot_residual.len().div_ceil(4);
        self.lut.resize(n_groups * 16, 0.0);
        self.lut_lower.resize(n_groups * 16, 0);
        self.lut_upper.resize(n_groups * 16, 0);

        for group_idx in 0..n_groups {
            let mut group = [0u16; 4];
            for bit_pos in 0..4usize {
                let dim_idx = group_idx * 4 + bit_pos;
                if dim_idx < q_rot_residual.len() {
                    group[bit_pos] =
                        quantize_query_value(q_rot_residual[dim_idx], self.vl, self.width);
                }
            }
            for nibble in 0..16usize {
                let mut value = 0u32;
                for bit_pos in 0..4usize {
                    if ((nibble >> bit_pos) & 1) != 0 {
                        value += group[bit_pos] as u32;
                    }
                }
                self.lut[group_idx * 16 + nibble] = value as f32;
                self.lut_lower[group_idx * 16 + nibble] = value as u8;
                self.lut_upper[group_idx * 16 + nibble] = (value >> 8) as u8;
            }
        }
    }

    fn reset_high_accuracy(&mut self, q_rot_residual: &[f32]) {
        let padded_dim = q_rot_residual.len();
        let one_over_sqrt_d = self.one_over_sqrt_d;
        self.unit_query.resize(padded_dim, 0.0);
        if self.y > 1e-5 {
            for (dst, value) in self.unit_query.iter_mut().zip(q_rot_residual.iter()) {
                *dst = *value / self.y;
            }
            self.sumq = self.unit_query.iter().sum::<f32>();
        } else {
            let value = one_over_sqrt_d;
            self.unit_query.fill(value);
            self.sumq = padded_dim as f32 * value;
        }

        let vmax = self
            .unit_query
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        self.delta = if vmax <= 1e-12 {
            1.0
        } else {
            vmax / (((1u32 << 13) - 1) as f32)
        };
        self.vl = 0.0;
        self.width = 0.0;

        let n_groups = padded_dim.div_ceil(4);
        self.lut.resize(n_groups * 16, 0.0);
        self.lut_lower.resize(n_groups * 16, 0);
        self.lut_upper.resize(n_groups * 16, 0);
        self.packed_high_acc_lut.resize(n_groups * 32, 0);
        self.lut_shift = 0;

        for group_idx in 0..n_groups {
            let mut quant_query = [0i32; 4];
            for bit_pos in 0..4usize {
                let dim_idx = group_idx * 4 + bit_pos;
                if dim_idx < padded_dim {
                    quant_query[bit_pos] =
                        quantize_signed_query_value(self.unit_query[dim_idx], self.delta) as i32;
                }
            }
            let mut group_lut = [0i32; 16];
            let mut group_min = 0i32;
            for nibble in 1..16usize {
                let lowbit = nibble & nibble.wrapping_neg();
                group_lut[nibble] =
                    group_lut[nibble - lowbit] + quant_query[HIGH_ACC_POS[nibble]];
                if group_lut[nibble] < group_min {
                    group_min = group_lut[nibble];
                }
            }
            self.lut_shift += group_min;

            let fill_lo = (group_idx / 4) * 128 + (group_idx % 4) * 16;
            let fill_hi = fill_lo + 64;
            for nibble in 0..16usize {
                let value = group_lut[nibble];
                self.lut[group_idx * 16 + nibble] = value as f32;
                let shifted = (value - group_min) as u32;
                let lo = shifted as u8;
                let hi = (shifted >> 8) as u8;
                self.lut_lower[group_idx * 16 + nibble] = lo;
                self.lut_upper[group_idx * 16 + nibble] = hi;
                self.packed_high_acc_lut[fill_lo + nibble] = lo;
                self.packed_high_acc_lut[fill_hi + nibble] = hi;
            }
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
        for group_idx in 0..state.unit_query.len().div_ceil(4) {
            let mut quant_query = [0i32; 4];
            for bit_pos in 0..4usize {
                let dim_idx = group_idx * 4 + bit_pos;
                if dim_idx >= state.unit_query.len() {
                    break;
                }
                quant_query[bit_pos] =
                    quantize_signed_query_value(state.unit_query[dim_idx], state.delta) as i32;
                let byte = short_code[dim_idx / 8];
                let bit = (byte >> (dim_idx % 8)) & 1;
                if bit != 0 {
                    selected_sum += quant_query[bit_pos];
                }
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

pub fn scan_layout_bitmask(
    layout: &ExRaBitQLayout,
    quantizer: &ExRaBitQQuantizer,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Vec<(i64, f32)> {
    if top_k == 0 || layout.is_empty() {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    if state.use_high_accuracy
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
    {
        return unsafe { scan_layout_bitmask_avx512(layout, quantizer, state, top_k) };
    }

    let fac_rescale = (1u32 << quantizer.config().ex_bits()) as f32;
    let candidates = scalar_scan_layout(layout, state, layout.len());
    let mut exact = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let idx = candidate.idx;
        let distance = if state.use_high_accuracy {
            let long_ip = quantizer.long_code_inner_product(&state.unit_query, layout.long_code_at(idx));
            layout.x2_at(idx) + state.y2
                - layout.factor_at(idx).xipnorm
                    * state.y
                    * (fac_rescale * candidate.rabitq_ip
                        + long_ip
                        - (fac_rescale - 0.5) * state.sumq)
        } else {
            let long_ip = quantizer.long_code_inner_product(&state.residual, layout.long_code_at(idx));
            layout.x2_at(idx) + state.y2
                - layout.factor_at(idx).xipnorm
                    * (fac_rescale * candidate.rabitq_ip
                        + long_ip
                        - (fac_rescale - 1.0) * state.half_sum_residual)
        };
        exact.push((layout.id_at(idx), distance));
    }
    exact.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    exact.truncate(top_k);
    exact
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
#[target_feature(enable = "avx512bw,avx512f,avx2,ssse3,fma")]
unsafe fn scan_layout_bitmask_avx512(
    layout: &ExRaBitQLayout,
    quantizer: &ExRaBitQQuantizer,
    state: &ExRaBitQFastScanState,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let fac_rescale = (1u32 << quantizer.config().ex_bits()) as f32;
    let mut heap = BinaryHeap::with_capacity(top_k + 1);
    let mut distk = f32::INFINITY;

    for block_idx in 0..layout.n_blocks() {
        let idx_base = block_idx * FAST_SIZE;
        let num_points = layout.len().saturating_sub(idx_base).min(FAST_SIZE);
        if num_points == 0 {
            break;
        }

        let block = layout.fastscan_block(block_idx);
        let selected_sums = fastscan_block_avx512(state, block);
        for (slot, &raw_selected_sum) in selected_sums.iter().take(num_points).enumerate() {
            let idx = idx_base + slot;
            let selected_sum = raw_selected_sum as i32 + state.lut_shift;
            let ip_xb_qprime = selected_sum as f32 * state.delta;
            let lower_bound = layout.x2_at(idx) + state.y2
                - 5.0
                    * state.y
                    * layout.x_norm_at(idx)
                    * state.one_over_sqrt_d
                    * (ip_xb_qprime - 0.5 * state.sumq + HIGH_ACC_CONST_BOUND);
            if lower_bound >= distk {
                continue;
            }

            let long_ip =
                quantizer.long_code_inner_product(&state.unit_query, layout.long_code_at(idx));
            let distance = layout.x2_at(idx) + state.y2
                - layout.factor_at(idx).xipnorm
                    * state.y
                    * (fac_rescale * ip_xb_qprime
                        + long_ip
                        - (fac_rescale - 0.5) * state.sumq);
            let candidate = ScoredCandidate {
                idx,
                distance,
                rabitq_ip: ip_xb_qprime,
            };

            if heap.len() < top_k {
                heap.push(candidate);
            } else if let Some(worst) = heap.peek() {
                if candidate.distance < worst.distance
                    || (candidate.distance == worst.distance && candidate.idx < worst.idx)
                {
                    heap.pop();
                    heap.push(candidate);
                }
            }

            if heap.len() == top_k {
                if let Some(worst) = heap.peek() {
                    distk = worst.distance;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        out.push((layout.id_at(item.idx), item.distance));
    }
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
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
#[allow(dead_code)]
#[target_feature(enable = "avx512bw,avx512f,avx2,ssse3,fma")]
unsafe fn accumulate_one_block_high_acc_avx512(
    block: &[u8],
    packed_lut: &[u8],
    onorm: &[f32; FAST_SIZE],
    sumq: f32,
    qnorm: f32,
    delta: f32,
    shift: i32,
    one_over_sqrt_d: f32,
    distk: f32,
    padded_dim: usize,
) -> (u32, [f32; FAST_SIZE]) {
    use std::arch::x86_64::*;

    let mut ip_xb_qprime = [0.0f32; FAST_SIZE];
    let low_mask = _mm512_set1_epi8(0x0F_u8 as i8);
    let mut accu = [[_mm512_setzero_si512(); 4]; 2];
    let n_groups = padded_dim >> 2;

    let mut codes_ptr = block.as_ptr();
    let mut lut_ptr = packed_lut.as_ptr();
    for _ in (0..n_groups).step_by(4) {
        let c = _mm512_loadu_si512(codes_ptr as *const __m512i);
        let lo = _mm512_and_si512(c, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        for layer in 0..2 {
            let lut = _mm512_loadu_si512(lut_ptr as *const __m512i);
            let res_lo = _mm512_shuffle_epi8(lut, lo);
            let res_hi = _mm512_shuffle_epi8(lut, hi);

            accu[layer][0] = _mm512_add_epi16(accu[layer][0], res_lo);
            accu[layer][1] = _mm512_add_epi16(accu[layer][1], _mm512_srli_epi16(res_lo, 8));
            accu[layer][2] = _mm512_add_epi16(accu[layer][2], res_hi);
            accu[layer][3] = _mm512_add_epi16(accu[layer][3], _mm512_srli_epi16(res_hi, 8));

            lut_ptr = lut_ptr.add(64);
        }
        codes_ptr = codes_ptr.add(64);
    }

    let mut dis0 = [_mm512_setzero_si512(); 2];
    let mut dis1 = [_mm512_setzero_si512(); 2];
    for layer in 0..2 {
        let mut tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[layer][0]),
            _mm512_extracti64x4_epi64(accu[layer][0], 1),
        );
        let tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[layer][1]),
            _mm512_extracti64x4_epi64(accu[layer][1], 1),
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));
        dis0[layer] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xF0)),
        );

        let mut tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[layer][2]),
            _mm512_extracti64x4_epi64(accu[layer][2], 1),
        );
        let tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[layer][3]),
            _mm512_extracti64x4_epi64(accu[layer][3], 1),
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));
        dis1[layer] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xF0)),
        );
    }

    let res = [
        _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8)),
        _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8)),
    ];

    let simd_shift = _mm512_set1_epi32(shift);
    let simd_delta = _mm512_set1_ps(delta);
    let simd_sumq_const_bound = _mm512_set1_ps(0.5 * sumq - HIGH_ACC_CONST_BOUND);
    let simd_qnorm_over_sqrtd = _mm512_set1_ps(-5.0 * qnorm * one_over_sqrt_d);
    let simd_qnorm_sqr = _mm512_set1_ps(qnorm * qnorm);
    let simd_distk = _mm512_set1_ps(distk);

    let mut mask = 0u32;
    for lane_group in 0..2usize {
        let shifted = _mm512_add_epi32(res[lane_group], simd_shift);
        let mut tmp = _mm512_cvtepi32_ps(shifted);
        tmp = _mm512_mul_ps(tmp, simd_delta);
        _mm512_storeu_ps(ip_xb_qprime.as_mut_ptr().add(lane_group * 16), tmp);

        tmp = _mm512_sub_ps(tmp, simd_sumq_const_bound);
        let simd_onorm = _mm512_loadu_ps(onorm.as_ptr().add(lane_group * 16));
        tmp = _mm512_mul_ps(tmp, simd_qnorm_over_sqrtd);
        tmp = _mm512_mul_ps(tmp, simd_onorm);
        tmp = _mm512_add_ps(tmp, simd_qnorm_sqr);
        tmp = _mm512_add_ps(tmp, _mm512_mul_ps(simd_onorm, simd_onorm));
        mask |= (_mm512_cmp_ps_mask(tmp, simd_distk, _CMP_LT_OS) as u32) << (lane_group * 16);
    }

    (mask, ip_xb_qprime)
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

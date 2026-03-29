use super::{ExRaBitQLayout, FAST_SIZE};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct ExRaBitQFastScanState {
    pub residual: Vec<f32>,
    pub y2: f32,
    pub lut: Vec<f32>,
    pub half_sum_residual: f32,
    pub vl: f32,
    pub width: f32,
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
            }
        }

        Self {
            residual: q_rot_residual.to_vec(),
            y2,
            lut,
            half_sum_residual,
            vl,
            width,
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
            let rabitq_ip = selected_sum * state.width + layout.short_sum_xb_at(idx) * state.vl
                - state.half_sum_residual;
            let distance = layout.x2_at(idx) + state.y2
                - (layout.short_ip_at(idx) * rabitq_ip + layout.short_err_at(idx));
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

fn quantize_query_value(value: f32, vl: f32, width: f32) -> u16 {
    if width <= 1e-12 {
        return 0;
    }
    ((value - vl) / width + 0.5).clamp(0.0, ((1u32 << 14) - 1) as f32) as u16
}

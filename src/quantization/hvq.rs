//! **EXPERIMENTAL** — HvqQuantizer has no native knowhere equivalent.
//! Not suitable for cross-implementation parity testing.
//! Benchmark results should not be compared against native IVF-PQ/SQ metrics.

use crate::quantization::turboquant::packed::{pack_codes, unpack_codes};
use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
pub struct HvqConfig {
    pub dim: usize,
    pub nbits: u8,
}

#[derive(Clone, Debug)]
pub struct HvqQueryState {
    pub q_rot: Vec<f32>,
    pub centroid_score: f32,
}

#[doc = "⚠️ Experimental: no native parity"]
pub struct HvqQuantizer {
    pub config: HvqConfig,
    pub rotation_matrix: Vec<f32>, // row-major dim x dim
    pub scale: f32,
    pub offset: f32,
    pub centroid: Vec<f32>,
    pub rotated_centroid: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct CriticalValue {
    threshold: f32,
    dim: usize,
    next_level: u16,
}

impl PartialEq for CriticalValue {
    fn eq(&self, other: &Self) -> bool {
        self.threshold.to_bits() == other.threshold.to_bits()
            && self.dim == other.dim
            && self.next_level == other.next_level
    }
}

impl Eq for CriticalValue {}

impl PartialOrd for CriticalValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CriticalValue {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .threshold
            .total_cmp(&self.threshold)
            .then_with(|| other.dim.cmp(&self.dim))
            .then_with(|| other.next_level.cmp(&self.next_level))
    }
}

impl HvqQuantizer {
    pub fn new(config: HvqConfig, seed: u64) -> Self {
        assert!(config.dim > 0, "dim must be > 0");
        assert!(
            (1..=8).contains(&config.nbits),
            "nbits must be in range 1..=8"
        );

        let dim = config.dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let random_values: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let random_matrix = DMatrix::from_row_slice(dim, dim, &random_values);
        let q = QR::new(random_matrix).q();

        let mut rotation_matrix = vec![0.0f32; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                rotation_matrix[r * dim + c] = q[(r, c)];
            }
        }

        Self {
            config,
            rotation_matrix,
            scale: 1.0,
            offset: 0.0,
            centroid: vec![0.0; dim],
            rotated_centroid: vec![0.0; dim],
        }
    }

    pub fn train(&mut self, n: usize, data: &[f32]) {
        assert!(n > 0, "n must be > 0");
        assert_eq!(
            data.len(),
            n * self.config.dim,
            "expected {} floats, got {}",
            n * self.config.dim,
            data.len()
        );

        self.centroid.fill(0.0);
        for row in data.chunks_exact(self.config.dim) {
            for (c, &value) in self.centroid.iter_mut().zip(row.iter()) {
                *c += value;
            }
        }

        let inv_n = 1.0 / n as f32;
        for value in &mut self.centroid {
            *value *= inv_n;
        }
        self.rotated_centroid = self.rotate(&self.centroid);
    }

    fn rotate_scalar(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for (r, out_r) in out.iter_mut().enumerate().take(dim) {
            let row = &self.rotation_matrix[r * dim..(r + 1) * dim];
            *out_r = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>();
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rotate_avx2(&self, v: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let simd_len = dim / 8 * 8;
        let mut out = vec![0.0f32; dim];

        for (r, out_r) in out.iter_mut().enumerate().take(dim) {
            let row = &self.rotation_matrix[r * dim..(r + 1) * dim];
            let mut acc = _mm256_setzero_ps();
            let mut c = 0usize;

            while c < simd_len {
                let row_chunk = unsafe { _mm256_loadu_ps(row.as_ptr().add(c)) };
                let vec_chunk = unsafe { _mm256_loadu_ps(v.as_ptr().add(c)) };
                acc = _mm256_fmadd_ps(row_chunk, vec_chunk, acc);
                c += 8;
            }

            let mut lanes = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
            let mut sum = lanes.into_iter().sum::<f32>();

            while c < dim {
                sum += row[c] * v[c];
                c += 1;
            }

            *out_r = sum;
        }

        out
    }

    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: the required CPU features are checked at runtime.
                return unsafe { self.rotate_avx2(v) };
            }
        }

        self.rotate_scalar(v)
    }

    /// Explicit query rotation helper for search path.
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        self.rotate(query)
    }

    fn inverse_rotate_scalar(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for (c, out_c) in out.iter_mut().enumerate().take(dim) {
            let mut sum = 0.0f32;
            for (r, vr) in v.iter().enumerate().take(dim) {
                sum += self.rotation_matrix[r * dim + c] * vr;
            }
            *out_c = sum;
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn inverse_rotate_avx2(&self, v: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let simd_len = dim / 8 * 8;
        let mut out = vec![0.0f32; dim];
        let out_ptr = out.as_mut_ptr();

        for (r, &vr) in v.iter().enumerate().take(dim) {
            let row = &self.rotation_matrix[r * dim..(r + 1) * dim];
            let scale = _mm256_set1_ps(vr);
            let mut c = 0usize;

            while c < simd_len {
                let row_chunk = unsafe { _mm256_loadu_ps(row.as_ptr().add(c)) };
                let out_chunk = unsafe { _mm256_loadu_ps(out_ptr.add(c)) };
                let updated = _mm256_fmadd_ps(row_chunk, scale, out_chunk);
                unsafe { _mm256_storeu_ps(out_ptr.add(c), updated) };
                c += 8;
            }

            while c < dim {
                out[c] += row[c] * vr;
                c += 1;
            }
        }

        out
    }

    pub fn inverse_rotate(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: the required CPU features are checked at runtime.
                return unsafe { self.inverse_rotate_avx2(v) };
            }
        }

        self.inverse_rotate_scalar(v)
    }

    pub fn compute_scale_offset(rotated: &[f32], nbits: u8) -> (f32, f32) {
        let max_v = rotated
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let levels = (1u32 << nbits) as f32;
        let mut scale = (2.0 * max_v) / levels.max(1.0);
        if scale.abs() < 1e-6 {
            scale = 1e-6;
        }
        (scale, -max_v)
    }

    pub fn quantize_scalar(x: f32, scale: f32, offset: f32, nbits: u8) -> u8 {
        let max_code = ((1u32 << nbits) - 1) as f32;
        (((x - offset) / scale).round().clamp(0.0, max_code)) as u8
    }

    pub fn dequantize_scalar(code: u8, scale: f32, offset: f32) -> f32 {
        code as f32 * scale + offset
    }

    pub fn code_size_bytes(&self) -> usize {
        12 + (self.config.dim * self.config.nbits as usize).div_ceil(8)
    }

    fn parse_code<'a>(&self, code: &'a [u8]) -> (f32, f32, f32, &'a [u8]) {
        debug_assert!(
            code.len() >= 12,
            "HVQ code too short: expected at least 12 bytes, got {}",
            code.len()
        );
        let norm_o = f32::from_le_bytes(code[0..4].try_into().unwrap());
        let vmax = f32::from_le_bytes(code[4..8].try_into().unwrap());
        let base_quant_dist = f32::from_le_bytes(code[8..12].try_into().unwrap());
        let packed = &code[12..];
        debug_assert_eq!(
            packed.len(),
            (self.config.dim * self.config.nbits as usize).div_ceil(8),
            "HVQ packed payload size mismatch"
        );
        (norm_o, vmax, base_quant_dist, packed)
    }

    fn decode_unit(&self, packed_code: &[u8], vmax: f32) -> Vec<f32> {
        let indices = unpack_codes(packed_code, self.config.dim, self.config.nbits);
        let levels = (1u32 << self.config.nbits) as f32;
        let scale = (2.0 * vmax) / levels.max(1.0);
        indices
            .into_iter()
            .map(|c| Self::dequantize_scalar(c as u8, scale, -vmax))
            .collect()
    }

    fn reconstruct_rotated(&self, code: &[u8]) -> Vec<f32> {
        let (norm_o, vmax, base_quant_dist, packed_code) = self.parse_code(code);
        let scale = if base_quant_dist > 1e-12 {
            norm_o / base_quant_dist
        } else {
            0.0
        };
        let mut reconstructed = self.decode_unit(packed_code, vmax);
        for (value, &centroid) in reconstructed.iter_mut().zip(self.rotated_centroid.iter()) {
            *value = *value * scale + centroid;
        }
        reconstructed
    }

    fn unit_vmax(o_hat: &[f32]) -> f32 {
        o_hat
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6)
    }

    fn objective(ip: f32, qed_length: f32) -> f32 {
        if qed_length > 1e-12 {
            ip * ip / qed_length
        } else {
            0.0
        }
    }

    fn greedy_quantize(&self, o_hat: &[f32], nrefine: usize) -> (Vec<u16>, f32, f32) {
        debug_assert_eq!(o_hat.len(), self.config.dim);
        let nbits = self.config.nbits;
        let vmax = Self::unit_vmax(o_hat);
        let (scale, offset) = Self::compute_scale_offset(o_hat, nbits);
        let max_code = ((1u32 << nbits) - 1) as u16;

        let mut codes: Vec<u16> = o_hat
            .iter()
            .map(|&x| Self::quantize_scalar(x, scale, offset, nbits) as u16)
            .collect();

        let dequant = |code: u16| Self::dequantize_scalar(code as u8, scale, offset);
        let mut ip: f32 = codes
            .iter()
            .zip(o_hat.iter())
            .map(|(&code, &target)| dequant(code) * target)
            .sum();
        let mut qed_length: f32 = codes
            .iter()
            .map(|&code| {
                let value = dequant(code);
                value * value
            })
            .sum();
        let mut objective = Self::objective(ip, qed_length);

        let max_refine = nrefine.min(6);
        for _round in 0..max_refine {
            let mut improved = false;
            for i in 0..self.config.dim {
                let current = codes[i];
                let old_val = dequant(current);
                let target_val = if ip.abs() > 1e-12 {
                    o_hat[i] * qed_length / ip
                } else {
                    o_hat[i]
                };
                let target_code = ((target_val - offset) / scale)
                    .round()
                    .clamp(0.0, max_code as f32) as u16;

                let mut candidates = [u16::MAX; 5];
                let mut candidate_len = 0usize;
                let mut push_candidate = |candidate: i32| {
                    if candidate < 0 || candidate > max_code as i32 {
                        return;
                    }
                    let candidate = candidate as u16;
                    if candidate == current {
                        return;
                    }
                    if candidates[..candidate_len].contains(&candidate) {
                        return;
                    }
                    candidates[candidate_len] = candidate;
                    candidate_len += 1;
                };
                push_candidate(target_code as i32 - 1);
                push_candidate(target_code as i32);
                push_candidate(target_code as i32 + 1);
                push_candidate(current as i32 - 1);
                push_candidate(current as i32 + 1);

                let mut best_code = current;
                let mut best_ip = ip;
                let mut best_qed = qed_length;
                let mut best_objective = objective;
                for &candidate in &candidates[..candidate_len] {
                    let new_val = dequant(candidate);
                    let new_ip = ip - o_hat[i] * old_val + o_hat[i] * new_val;
                    let new_qed = qed_length - old_val * old_val + new_val * new_val;
                    if new_qed <= 1e-12 {
                        continue;
                    }
                    let new_objective = Self::objective(new_ip, new_qed);
                    if new_objective > best_objective + 1e-10 {
                        best_code = candidate;
                        best_ip = new_ip;
                        best_qed = new_qed;
                        best_objective = new_objective;
                    }
                }

                if best_code != current {
                    codes[i] = best_code;
                    ip = best_ip;
                    qed_length = best_qed;
                    objective = best_objective;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        let _ = vmax;
        (codes, ip, qed_length)
    }

    fn fast_quantize(&self, o_hat: &[f32]) -> (Vec<u16>, f32, f32) {
        debug_assert_eq!(o_hat.len(), self.config.dim);

        let dim = self.config.dim;
        let levels = 1u16 << self.config.nbits;
        let half_levels = levels / 2;
        let half_levels_f = half_levels as f32;
        let max_code = half_levels; // max magnitude for negative dims
        let vmax = Self::unit_vmax(o_hat);

        let mut abs_o = Vec::with_capacity(dim);
        let mut non_negative = Vec::with_capacity(dim);
        let mut max_level = Vec::with_capacity(dim);
        for &value in o_hat {
            abs_o.push(value.abs());
            let is_non_neg = value >= 0.0;
            non_negative.push(is_non_neg);
            max_level.push(if is_non_neg {
                half_levels.saturating_sub(1)
            } else {
                max_code
            });
        }

        // Find max abs value for t_start pruning (ExRaBitQ style)
        let max_abs = abs_o.iter().cloned().fold(0.0f32, f32::max);
        if max_abs < 1e-12 {
            let codes = vec![half_levels; dim];
            return (codes, 0.0, 0.0);
        }

        // t_start: conservative start — only skip obviously-too-small t values
        // ExRaBitQ uses max_code/3, we use 0 to guarantee dominance over greedy
        let t_start = 0.0f32;

        // Pre-generate ALL critical values with exact levels, sort once, scan linearly
        // Critical value for dim i, level k: t = (k - 0.5) / abs_o[i]
        let mut events: Vec<(f32, usize, u16)> = Vec::new(); // (threshold, dim, level)
        for i in 0..dim {
            let abs_val = abs_o[i];
            if abs_val < 1e-12 {
                continue;
            }
            let ml = max_level[i];
            for k in 1..=ml {
                let t = (k as f32 - 0.5) / abs_val;
                events.push((t, i, k));
            }
        }

        // Sort by threshold ascending, then by dim and level (match heap ordering)
        events.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut levels_abs = vec![0u16; dim];
        let mut sum_sq = 0.0f32;
        let mut sum_xc = 0.0f32;
        let mut best_objective = 0.0f32;
        let mut best_t = 0.0f32;
        let mut best_sum_sq = 0.0f32;
        let mut best_sum_xc = 0.0f32;

        // Linear scan through sorted events, batch same-threshold events
        let n_events = events.len();
        let mut ei = 0;
        while ei < n_events {
            let cur_t = events[ei].0;
            // Process all events at this exact threshold (bit-exact match)
            while ei < n_events && events[ei].0.to_bits() == cur_t.to_bits() {
                let (_, idx, new_level) = events[ei];
                let old_level = levels_abs[idx];
                if new_level > old_level {
                    sum_sq += (new_level as f32) * (new_level as f32)
                        - (old_level as f32) * (old_level as f32);
                    sum_xc += (new_level - old_level) as f32 * abs_o[idx];
                    levels_abs[idx] = new_level;
                }
                ei += 1;
            }
            // Evaluate objective once per unique threshold
            let objective = if sum_sq > 0.0 {
                (sum_xc * sum_xc) / sum_sq
            } else {
                0.0
            };
            if objective > best_objective + 1e-10 {
                best_objective = objective;
                best_t = cur_t;
                best_sum_sq = sum_sq;
                best_sum_xc = sum_xc;
            }
        }

        // Reconstruct codes from best_t; use exact accumulated sums for ip/qed
        let scale = vmax / half_levels_f.max(1.0);
        let ip = scale * best_sum_xc;
        let qed_length = scale * scale * best_sum_sq;
        let mut codes = Vec::with_capacity(dim);
        for i in 0..dim {
            let magnitude = if abs_o[i] > 1e-12 {
                ((best_t * abs_o[i] + 0.5).floor() as u16).min(max_level[i])
            } else {
                0
            };
            let code = if non_negative[i] {
                half_levels + magnitude
            } else {
                half_levels - magnitude
            };
            codes.push(code);
        }

        (codes, ip, qed_length)
    }

    pub fn encode(&self, v: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(v.len(), self.config.dim);
        let v_centered: Vec<f32> = v
            .iter()
            .zip(self.centroid.iter())
            .map(|(a, b)| a - b)
            .collect();
        let o = self.rotate(&v_centered);
        let nbits = self.config.nbits;
        let norm_o = o.iter().map(|x| x * x).sum::<f32>().sqrt();
        let o_hat: Vec<f32> = if norm_o > 1e-12 {
            o.iter().map(|x| x / norm_o).collect()
        } else {
            o.clone()
        };

        let vmax = Self::unit_vmax(&o_hat);
        let (codes, ip, qed_length) = self.fast_quantize(&o_hat);
        let _ = (nrefine, ip);

        let base_quant_dist = qed_length.sqrt();

        let mut packed_bits = Vec::with_capacity((self.config.dim * nbits as usize).div_ceil(8));
        pack_codes(&codes, nbits, &mut packed_bits);

        let mut packed = Vec::with_capacity(self.code_size_bytes());
        packed.extend_from_slice(&norm_o.to_le_bytes());
        packed.extend_from_slice(&vmax.to_le_bytes());
        packed.extend_from_slice(&base_quant_dist.to_le_bytes());
        packed.extend_from_slice(&packed_bits);
        packed
    }

    pub fn encode_batch(&self, n: usize, data: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(data.len(), n * self.config.dim);
        let per = self.code_size_bytes();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let encoded: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let v = &data[i * self.config.dim..(i + 1) * self.config.dim];
                    self.encode(v, nrefine)
                })
                .collect();
            let mut out = Vec::with_capacity(n * per);
            for e in encoded {
                out.extend_from_slice(&e);
            }
            return out;
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::with_capacity(n * per);
            for i in 0..n {
                let v = &data[i * self.config.dim..(i + 1) * self.config.dim];
                out.extend_from_slice(&self.encode(v, nrefine));
            }
            out
        }
    }

    pub fn base_quant_dist(&self, _v: &[f32], code: &[u8]) -> f32 {
        let (_, _, base_quant_dist, _) = self.parse_code(code);
        base_quant_dist
    }

    pub fn adc_distance(&self, query: &[f32], code: &[u8], base_dist: f32) -> f32 {
        let q_rot = self.rotate(query);
        self.adc_distance_prerotated(&q_rot, code, base_dist)
    }

    pub fn precompute_query_state(&self, q_rot: &[f32]) -> HvqQueryState {
        assert_eq!(q_rot.len(), self.config.dim);
        let centroid_score = q_rot
            .iter()
            .zip(self.rotated_centroid.iter())
            .map(|(a, b)| a * b)
            .sum();
        HvqQueryState {
            q_rot: q_rot.to_vec(),
            centroid_score,
        }
    }

    fn score_code_alloc(&self, state: &HvqQueryState, code: &[u8]) -> f32 {
        debug_assert_eq!(state.q_rot.len(), self.config.dim);
        let (norm_o, vmax, base_quant_dist, packed_code) = self.parse_code(code);
        let o_hat = self.decode_unit(packed_code, vmax);
        let ip: f32 = state
            .q_rot
            .iter()
            .zip(o_hat.iter())
            .map(|(a, b)| a * b)
            .sum();
        let centered_score = if base_quant_dist > 1e-12 {
            norm_o * ip / base_quant_dist
        } else {
            0.0
        };
        state.centroid_score + centered_score
    }

    /// Zero-allocation hot path: stream quantized residual bits and accumulate the
    /// dot product directly instead of materializing temporary decoded vectors.
    pub fn score_code(&self, state: &HvqQueryState, code: &[u8]) -> f32 {
        debug_assert_eq!(state.q_rot.len(), self.config.dim);
        let (norm_o, vmax, base_quant_dist, packed_code) = self.parse_code(code);
        if base_quant_dist <= 1e-12 {
            return state.centroid_score;
        }

        let nbits = self.config.nbits as usize;
        let levels = (1u32 << self.config.nbits) as f32;
        let scale = (2.0 * vmax) / levels.max(1.0);
        let offset = -vmax;
        let mask = (1u32 << nbits) - 1;

        let mut bit_buffer = 0u32;
        let mut bits_in_buffer = 0usize;
        let mut byte_idx = 0usize;
        let mut ip = 0.0f32;

        for &q in &state.q_rot {
            while bits_in_buffer < nbits {
                let next = packed_code.get(byte_idx).copied().unwrap_or(0) as u32;
                bit_buffer |= next << bits_in_buffer;
                bits_in_buffer += 8;
                byte_idx += 1;
            }

            let raw = (bit_buffer & mask) as u8;
            bit_buffer >>= nbits;
            bits_in_buffer -= nbits;

            let decoded = Self::dequantize_scalar(raw, scale, offset);
            ip += q * decoded;
        }

        state.centroid_score + norm_o * ip / base_quant_dist
    }

    pub fn adc_distance_prerotated(&self, q_rot: &[f32], code: &[u8], _base_dist: f32) -> f32 {
        let state = self.precompute_query_state(q_rot);
        self.score_code(&state, code)
    }
}

/// dot product of u8 query codes × i8 database codes.
/// Falls back to scalar on non-x86_64 or when avx512vnni is unavailable.
pub fn dot_u8_i8_avx512(a: &[u8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512f")
            && a.len() >= 64
        {
            // SAFETY: feature detection is done at runtime, and slices are valid for loads.
            return unsafe { dot_u8_i8_avx512_impl(a, b) };
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn dot_u8_i8_avx512_impl(a: &[u8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        // SAFETY: off..off+64 are in-bounds by chunk construction.
        let va = unsafe { _mm512_loadu_si512(a[off..].as_ptr() as *const __m512i) };
        // SAFETY: same bounds as above; raw bytes are interpreted as signed i8 lanes.
        let vb = unsafe { _mm512_loadu_si512(b[off..].as_ptr() as *const __m512i) };
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }

    // Reduce 16 i32 lanes manually for compatibility.
    // SAFETY: __m512i is exactly 64 bytes, same as [i32; 16].
    let lanes: [i32; 16] = unsafe { std::mem::transmute(acc) };
    let mut total = lanes.iter().sum::<i32>();

    for i in (chunks * 64)..n {
        total += a[i] as i32 * b[i] as i32;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    #[test]
    fn test_hvq_roundtrip() {
        let dim = 32usize;
        let n = 100usize;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let mut hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 8 }, 42);
        hvq.train(n, &data);

        let mut rel_err_sum = 0.0f32;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let code = hvq.encode(v, 6);
            let recon_rot = hvq.reconstruct_rotated(&code);
            let recon = hvq.inverse_rotate(&recon_rot);
            let err = l2_sq(v, &recon).sqrt();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            rel_err_sum += err / norm.max(1e-6);
        }
        let avg_rel_err = rel_err_sum / n as f32;
        println!("hvq_roundtrip avg_relative_error={:.4}", avg_rel_err);
        assert!(
            avg_rel_err < 0.15,
            "avg relative error too high: {}",
            avg_rel_err
        );
    }

    #[test]
    fn test_hvq_recall() {
        let dim = 128usize;
        let n = 1000usize;
        let nq = 100usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(42);
        let train: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let normalize_rows = |data: &[f32], dim: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(data.len());
            for row in data.chunks_exact(dim) {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                out.extend(row.iter().map(|x| x / norm));
            }
            out
        };
        let train = normalize_rows(&train, dim);
        let queries = normalize_rows(&queries, dim);

        let mut hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 42);
        hvq.train(n, &train);

        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let v = &train[i * dim..(i + 1) * dim];
            let code = hvq.encode(v, 6);
            codes.push(code);
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];

            let mut gt: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &train[i * dim..(i + 1) * dim];
                    (i, q.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>())
                })
                .collect();
            gt.sort_by(|a, b| b.1.total_cmp(&a.1));
            let gt_topk: Vec<usize> = gt.iter().take(top_k).map(|(i, _)| *i).collect();

            let mut approx: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, hvq.adc_distance(q, &codes[i], 0.0)))
                .collect();
            approx.sort_by(|a, b| b.1.total_cmp(&a.1));
            let approx_topk: Vec<usize> = approx.iter().take(top_k).map(|(i, _)| *i).collect();

            for idx in gt_topk {
                total += 1;
                if approx_topk.contains(&idx) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f32 / total as f32;
        println!("hvq_recall@10={:.3}", recall);
        assert!(recall.is_finite());
    }

    #[test]
    fn test_score_code_zero_alloc_matches_reference() {
        let dim = 96usize;
        let n = 128usize;
        let nq = 8usize;
        let nrefine = 4usize;

        let mut rng = StdRng::seed_from_u64(123);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let mut hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 7);
        hvq.train(n, &data);
        let codes = hvq.encode_batch(n, &data, nrefine);
        let code_size = hvq.code_size_bytes();

        for query in queries.chunks_exact(dim) {
            let q_rot = hvq.rotate_query(query);
            let state = hvq.precompute_query_state(&q_rot);

            for code in codes.chunks_exact(code_size) {
                let reference = hvq.score_code_alloc(&state, code);
                let streamed = hvq.score_code(&state, code);
                let diff = (reference - streamed).abs();
                let tol = 1e-5 * reference.abs().max(1.0);
                assert!(
                    diff <= tol,
                    "score mismatch: reference={} streamed={} diff={} tol={}",
                    reference,
                    streamed,
                    diff,
                    tol
                );
            }
        }
    }

    #[test]
    fn test_dot_u8_i8() {
        let a: Vec<u8> = (0..128).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..128).map(|i| (i as i32 - 64) as i8).collect();
        let scalar: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        let simd = dot_u8_i8_avx512(&a, &b);
        assert_eq!(scalar, simd, "SIMD result must match scalar");
        println!("dot_u8_i8 scalar={} simd={}", scalar, simd);
    }

    #[test]
    fn test_hvq_rotate_matches_scalar_reference() {
        let dim = 24usize;
        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 7);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 - 12.0) * 0.125).collect();

        let scalar = hvq.rotate_scalar(&query);
        let rotated = hvq.rotate(&query);

        for (lhs, rhs) in scalar.iter().zip(rotated.iter()) {
            assert!(
                (*lhs - *rhs).abs() <= 1e-5,
                "rotate mismatch: scalar={} rotated={}",
                lhs,
                rhs
            );
        }
    }

    #[test]
    fn test_hvq_inverse_rotate_matches_scalar_reference() {
        let dim = 24usize;
        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 11);
        let query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.25).sin()).collect();

        let scalar = hvq.inverse_rotate_scalar(&query);
        let rotated = hvq.inverse_rotate(&query);

        for (lhs, rhs) in scalar.iter().zip(rotated.iter()) {
            assert!(
                (*lhs - *rhs).abs() <= 1e-5,
                "inverse_rotate mismatch: scalar={} rotated={}",
                lhs,
                rhs
            );
        }
    }

    fn normalize_rows(data: &[f32], dim: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(data.len());
        for row in data.chunks_exact(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            out.extend(row.iter().map(|x| x / norm));
        }
        out
    }

    fn encode_with_strategy(
        hvq: &HvqQuantizer,
        v: &[f32],
        use_fast: bool,
        nrefine: usize,
    ) -> Vec<u8> {
        let v_centered: Vec<f32> = v
            .iter()
            .zip(hvq.centroid.iter())
            .map(|(a, b)| a - b)
            .collect();
        let o = hvq.rotate(&v_centered);
        let norm_o = o.iter().map(|x| x * x).sum::<f32>().sqrt();
        let o_hat: Vec<f32> = if norm_o > 1e-12 {
            o.iter().map(|x| x / norm_o).collect()
        } else {
            o.clone()
        };
        let vmax = HvqQuantizer::unit_vmax(&o_hat);
        let (codes, _ip, qed_length) = if use_fast {
            hvq.fast_quantize(&o_hat)
        } else {
            hvq.greedy_quantize(&o_hat, nrefine)
        };
        let base_quant_dist = qed_length.sqrt();

        let mut packed_bits =
            Vec::with_capacity((hvq.config.dim * hvq.config.nbits as usize).div_ceil(8));
        pack_codes(&codes, hvq.config.nbits, &mut packed_bits);

        let mut packed = Vec::with_capacity(hvq.code_size_bytes());
        packed.extend_from_slice(&norm_o.to_le_bytes());
        packed.extend_from_slice(&vmax.to_le_bytes());
        packed.extend_from_slice(&base_quant_dist.to_le_bytes());
        packed.extend_from_slice(&packed_bits);
        packed
    }

    fn brute_force_topk_ip(
        data: &[f32],
        queries: &[f32],
        dim: usize,
        top_k: usize,
    ) -> Vec<Vec<usize>> {
        let n = data.len() / dim;
        let mut out = Vec::with_capacity(queries.len() / dim);
        for query in queries.chunks_exact(dim) {
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    (
                        i,
                        query
                            .iter()
                            .zip(v.iter())
                            .map(|(&a, &b)| a * b)
                            .sum::<f32>(),
                    )
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            out.push(scored.into_iter().take(top_k).map(|(i, _)| i).collect());
        }
        out
    }

    fn quantized_topk(
        hvq: &HvqQuantizer,
        codes: &[u8],
        queries: &[f32],
        dim: usize,
        top_k: usize,
    ) -> Vec<Vec<usize>> {
        let code_size = hvq.code_size_bytes();
        let n = codes.len() / code_size;
        let mut out = Vec::with_capacity(queries.len() / dim);
        for query in queries.chunks_exact(dim) {
            let q_rot = hvq.rotate_query(query);
            let state = hvq.precompute_query_state(&q_rot);
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let code = &codes[i * code_size..(i + 1) * code_size];
                    (i, hvq.score_code(&state, code))
                })
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            out.push(scored.into_iter().take(top_k).map(|(i, _)| i).collect());
        }
        out
    }

    fn compute_recall(results: &[Vec<usize>], gt: &[Vec<usize>], top_k: usize) -> f32 {
        let mut hits = 0usize;
        let total = results.len() * top_k;
        for (result, truth) in results.iter().zip(gt.iter()) {
            for &idx in truth.iter().take(top_k) {
                if result.iter().take(top_k).any(|&candidate| candidate == idx) {
                    hits += 1;
                }
            }
        }
        hits as f32 / total.max(1) as f32
    }

    #[test]
    fn test_fast_quantize_dominates_greedy() {
        let dim = 64usize;
        let samples = 100usize;
        let mut rng = StdRng::seed_from_u64(2026);

        for &nbits in &[1u8, 2, 4] {
            let hvq = HvqQuantizer::new(HvqConfig { dim, nbits }, 99);
            for _ in 0..samples {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for value in &mut v {
                    *value /= norm;
                }

                let (_fast_codes, fast_ip, fast_qed) = hvq.fast_quantize(&v);
                let (_greedy_codes, greedy_ip, greedy_qed) = hvq.greedy_quantize(&v, 6);
                let fast_objective = HvqQuantizer::objective(fast_ip, fast_qed);
                let greedy_objective = HvqQuantizer::objective(greedy_ip, greedy_qed);
                let tol = 1e-5 * fast_objective.abs().max(greedy_objective.abs()).max(1.0);

                assert!(
                    fast_objective + tol >= greedy_objective,
                    "fast objective regressed for nbits={}: fast={} greedy={} tol={}",
                    nbits,
                    fast_objective,
                    greedy_objective,
                    tol
                );
            }
        }
    }

    #[test]
    fn test_fast_quantize_recall_improvement() {
        let dim = 32usize;
        let clusters = 8usize;
        let per_cluster = 40usize;
        let nq = 32usize;
        let top_k = 10usize;
        let n = clusters * per_cluster;

        let mut rng = StdRng::seed_from_u64(4242);
        let centers: Vec<f32> = normalize_rows(
            &(0..clusters * dim)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            dim,
        );

        let mut base = Vec::with_capacity(n * dim);
        for cluster in 0..clusters {
            let center = &centers[cluster * dim..(cluster + 1) * dim];
            for _ in 0..per_cluster {
                for &value in center {
                    base.push(value + rng.gen_range(-0.08f32..0.08));
                }
            }
        }
        let base = normalize_rows(&base, dim);

        let mut queries = Vec::with_capacity(nq * dim);
        for _ in 0..nq {
            let cluster = rng.gen_range(0..clusters);
            let center = &centers[cluster * dim..(cluster + 1) * dim];
            for &value in center {
                queries.push(value + rng.gen_range(-0.08f32..0.08));
            }
        }
        let queries = normalize_rows(&queries, dim);

        let mut hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 1 }, 42);
        hvq.train(n, &base);

        let mut fast_codes = Vec::with_capacity(n * hvq.code_size_bytes());
        let mut greedy_codes = Vec::with_capacity(n * hvq.code_size_bytes());
        for row in base.chunks_exact(dim) {
            fast_codes.extend_from_slice(&encode_with_strategy(&hvq, row, true, 6));
            greedy_codes.extend_from_slice(&encode_with_strategy(&hvq, row, false, 6));
        }

        let gt = brute_force_topk_ip(&base, &queries, dim, top_k);
        let fast_results = quantized_topk(&hvq, &fast_codes, &queries, dim, top_k);
        let greedy_results = quantized_topk(&hvq, &greedy_codes, &queries, dim, top_k);
        let fast_recall = compute_recall(&fast_results, &gt, top_k);
        let greedy_recall = compute_recall(&greedy_results, &gt, top_k);

        println!(
            "fast_quantize recall@{} fast={:.4} greedy={:.4}",
            top_k, fast_recall, greedy_recall
        );
        assert!(
            fast_recall + 1e-6 >= greedy_recall,
            "fast quantize recall regressed: fast={} greedy={}",
            fast_recall,
            greedy_recall
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_hvq_avx2_helpers_match_scalar_reference() {
        let dim = 31usize;
        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 6 }, 13);
        let query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.17).cos()).collect();

        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            let rotate_scalar = hvq.rotate_scalar(&query);
            let inverse_scalar = hvq.inverse_rotate_scalar(&query);
            let rotate_simd = unsafe { hvq.rotate_avx2(&query) };
            let inverse_simd = unsafe { hvq.inverse_rotate_avx2(&query) };

            for (lhs, rhs) in rotate_scalar.iter().zip(rotate_simd.iter()) {
                assert!(
                    (*lhs - *rhs).abs() <= 1e-5,
                    "rotate_avx2 mismatch: scalar={} simd={}",
                    lhs,
                    rhs
                );
            }
            for (lhs, rhs) in inverse_scalar.iter().zip(inverse_simd.iter()) {
                assert!(
                    (*lhs - *rhs).abs() <= 1e-5,
                    "inverse_rotate_avx2 mismatch: scalar={} simd={}",
                    lhs,
                    rhs
                );
            }
        }
    }
}

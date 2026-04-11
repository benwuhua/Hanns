use std::cell::RefCell;

use super::config::UsqConfig;
use super::rotator::UsqRotator;
use crate::quantization::turboquant::packed::pack_codes;

/// Per-thread scratch buffers reused across encode() calls to eliminate hot-path allocations.
struct EncodeWorkspace {
    residual: Vec<f32>,
    rotated: Vec<f32>,
    o_hat: Vec<f32>,
    codes: Vec<u16>,
    sign_codes: Vec<u16>,
}

impl EncodeWorkspace {
    fn ensure(&mut self, padded_dim: usize) {
        if self.residual.len() < padded_dim {
            self.residual.resize(padded_dim, 0.0);
            self.rotated.resize(padded_dim, 0.0);
            self.o_hat.resize(padded_dim, 0.0);
            self.codes.resize(padded_dim, 0);
            self.sign_codes.resize(padded_dim, 0);
        }
    }
}

thread_local! {
    static ENCODE_WS: RefCell<EncodeWorkspace> = RefCell::new(EncodeWorkspace {
        residual: Vec::new(),
        rotated: Vec::new(),
        o_hat: Vec::new(),
        codes: Vec::new(),
        sign_codes: Vec::new(),
    });
}

/// Output of a single vector encoding pass.
#[derive(Clone, Debug)]
pub struct UsqEncoded {
    /// B-bit packed codes (padded_dim dimensions at nbits bits each).
    pub packed_bits: Vec<u8>,
    /// 1-bit sign codes — fastscan approximation.
    /// When nbits == 1, this is a clone of packed_bits.
    pub sign_bits: Vec<u8>,
    /// ‖rotated_residual‖
    pub norm: f32,
    /// norm²
    pub norm_sq: f32,
    /// max(|unit[i]|) — dequant scale anchor.
    pub vmax: f32,
    /// sqrt(Σ quantization_error²) — denominator for scoring.
    pub quant_quality: f32,
}

/// Precomputed query state for efficient SIMD scoring.
pub struct UsqQueryState {
    pub q_rot: Vec<f32>,
    pub q_sum: f32,
    pub q_quantized: Vec<i8>,
    pub q_quantized_sum: i32,
    pub q_scale: f32,
    pub centroid_score: f32,
}

/// Unified scalar quantizer: center → pad → rotate → normalize → quantize.
///
/// Replaces the equivalent logic duplicated across HVQ and ExRaBitQ.
#[derive(Clone)]
pub struct UsqQuantizer {
    config: UsqConfig,
    rotator: UsqRotator,
    centroid: Vec<f32>,
    /// Centroid rotated into the padded rotation space (length = padded_dim).
    rotated_centroid: Vec<f32>,
}

impl UsqQuantizer {
    pub fn new(config: UsqConfig) -> Self {
        let rotator = UsqRotator::new(&config);
        let padded = config.padded_dim();
        let centroid = vec![0.0f32; config.dim];
        let rotated_centroid = vec![0.0f32; padded];
        Self {
            config,
            rotator,
            centroid,
            rotated_centroid,
        }
    }

    pub fn config(&self) -> &UsqConfig {
        &self.config
    }

    pub fn rotator(&self) -> &UsqRotator {
        &self.rotator
    }

    /// Returns the current centroid (length = dim).
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Set (or update) the centroid.  Recomputes the rotated centroid.
    pub fn set_centroid(&mut self, centroid: &[f32]) {
        assert_eq!(
            centroid.len(),
            self.config.dim,
            "centroid length must equal dim"
        );
        self.centroid.clear();
        self.centroid.extend_from_slice(centroid);

        // Pad centroid to padded_dim and rotate.
        let padded_dim = self.config.padded_dim();
        let mut padded_centroid = vec![0.0f32; padded_dim];
        padded_centroid[..self.config.dim].copy_from_slice(centroid);
        self.rotated_centroid = self.rotator.rotate(&padded_centroid);
    }

    /// Encode a raw vector: center → pad → rotate → encode_rotated.
    pub fn encode(&self, vector: &[f32]) -> UsqEncoded {
        assert_eq!(
            vector.len(),
            self.config.dim,
            "vector length must equal dim"
        );

        let padded_dim = self.config.padded_dim();
        let norm_sq = vector.iter().map(|x| x * x).sum::<f32>();

        ENCODE_WS.with(|cell| {
            let ws = &mut *cell.borrow_mut();
            ws.ensure(padded_dim);

            // 1. Center and pad to padded_dim (padding dims stay 0).
            for i in 0..self.config.dim {
                ws.residual[i] = vector[i] - self.centroid[i];
            }
            for i in self.config.dim..padded_dim {
                ws.residual[i] = 0.0;
            }

            // 2. Rotate into ws.rotated (no allocation).
            self.rotator
                .rotate_into(&ws.residual[..padded_dim], &mut ws.rotated[..padded_dim]);

            // 3. Encode the already-rotated residual using workspace buffers.
            let mut encoded = self.encode_rotated_ws(ws, padded_dim);

            // 4. Override norm_sq: encode_rotated stores ||residual||², L2 needs ||x||².
            encoded.norm_sq = norm_sq;
            encoded
        })
    }

    /// Encode an already-rotated residual of length padded_dim.
    /// Used by IVF to avoid re-rotation.
    pub fn encode_rotated(&self, rotated: &[f32]) -> UsqEncoded {
        let padded_dim = self.config.padded_dim();
        assert_eq!(
            rotated.len(),
            padded_dim,
            "rotated length must equal padded_dim"
        );
        ENCODE_WS.with(|cell| {
            let ws = &mut *cell.borrow_mut();
            ws.ensure(padded_dim);
            ws.rotated[..padded_dim].copy_from_slice(rotated);
            self.encode_rotated_ws(ws, padded_dim)
        })
    }

    /// Core encode logic operating on pre-allocated workspace buffers.
    /// `ws.rotated[..padded_dim]` must be filled by the caller.
    fn encode_rotated_ws(&self, ws: &mut EncodeWorkspace, padded_dim: usize) -> UsqEncoded {
        let rotated = &ws.rotated[..padded_dim];

        // Compute norm and normalize into ws.o_hat.
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for (o, r) in ws.o_hat[..padded_dim].iter_mut().zip(rotated.iter()) {
                *o = r * inv;
            }
        } else {
            ws.o_hat[..padded_dim].copy_from_slice(rotated);
        }
        let o_hat = &ws.o_hat[..padded_dim];

        // Quantize into ws.codes.
        let (_ip, qed_length) = if self.config.nbits >= 4 {
            self.greedy_quantize_into(o_hat, 6, &mut ws.codes[..padded_dim])
        } else {
            self.fast_quantize_into(o_hat, &mut ws.codes[..padded_dim])
        };

        // Per-vector metadata.
        let vmax = Self::unit_vmax(o_hat);
        let quant_quality = qed_length.sqrt().max(1e-12);

        // Pack B-bit codes (output Vec must be allocated — it's returned to caller).
        let code_bytes = (padded_dim * self.config.nbits as usize).div_ceil(8);
        let mut packed_bits = Vec::with_capacity(code_bytes);
        pack_codes(&ws.codes[..padded_dim], self.config.nbits, &mut packed_bits);

        // 1-bit sign codes (output Vec must be allocated — it's returned to caller).
        let sign_bits = if self.config.nbits == 1 {
            packed_bits.clone()
        } else {
            for (s, &o) in ws.sign_codes[..padded_dim].iter_mut().zip(o_hat.iter()) {
                *s = if o >= 0.0 { 1u16 } else { 0u16 };
            }
            let mut sb = Vec::with_capacity(padded_dim / 8);
            pack_codes(&ws.sign_codes[..padded_dim], 1, &mut sb);
            sb
        };

        UsqEncoded {
            packed_bits,
            sign_bits,
            norm,
            norm_sq,
            vmax,
            quant_quality,
        }
    }

    /// Precompute query state for efficient SIMD scoring.
    ///
    /// Rotates the query, computes centroid score, and quantizes the query
    /// for integer dot-product acceleration (VNNI path).
    pub fn precompute_query_state(&self, query: &[f32]) -> UsqQueryState {
        let padded_dim = self.config.padded_dim();
        let mut padded_q = vec![0.0f32; padded_dim];
        padded_q[..self.config.dim].copy_from_slice(query);
        let q_rot = self.rotator.rotate(&padded_q);

        let q_sum: f32 = q_rot.iter().sum();
        let q_max = q_rot
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let q_scale = q_max / 127.0;
        let q_quantized: Vec<i8> = q_rot
            .iter()
            .map(|&v| (v / q_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        let q_quantized_sum: i32 = q_quantized.iter().map(|&v| v as i32).sum();

        let centroid_score: f32 = q_rot
            .iter()
            .zip(self.rotated_centroid.iter())
            .map(|(a, b)| a * b)
            .sum();

        UsqQueryState {
            q_rot,
            q_sum,
            q_quantized,
            q_quantized_sum,
            q_scale,
            centroid_score,
        }
    }

    /// Score for rerank using precomputed query state:
    ///   score = centroid_score + norm * dequant_ip(q, packed_bits) / quant_quality
    pub fn score_with_meta(
        &self,
        state: &UsqQueryState,
        norm: f32,
        vmax: f32,
        quant_quality: f32,
        packed_bits: &[u8],
    ) -> f32 {
        if quant_quality <= 1e-12 {
            return state.centroid_score;
        }

        let ip = self.compute_ip(state, vmax, packed_bits);
        state.centroid_score + norm * ip / quant_quality
    }

    /// Dispatch inner product computation to the best available implementation.
    fn compute_ip(&self, state: &UsqQueryState, vmax: f32, packed_bits: &[u8]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                if self.config.nbits == 1 {
                    return unsafe { self.compute_ip_1bit_avx512(state, vmax, packed_bits) };
                }
                if matches!(self.config.nbits, 4 | 8)
                    && std::arch::is_x86_feature_detected!("avx512vnni")
                {
                    return unsafe { self.compute_ip_vnni(state, vmax, packed_bits) };
                }
            }
        }
        self.compute_ip_scalar(&state.q_rot, vmax, packed_bits)
    }

    /// Scalar fallback: bit-streaming dequantize + dot product.
    fn compute_ip_scalar(&self, q_rot: &[f32], vmax: f32, packed_bits: &[u8]) -> f32 {
        let nbits = self.config.nbits as usize;
        let levels = (1u32 << self.config.nbits) as f32;
        let scale = (2.0 * vmax) / levels.max(1.0);
        let offset = -vmax;
        let mask = (1u32 << nbits) - 1;

        let mut bit_buffer = 0u32;
        let mut bits_in_buffer = 0usize;
        let mut byte_idx = 0usize;
        let mut ip = 0.0f32;

        for &q in q_rot {
            while bits_in_buffer < nbits {
                let next = packed_bits.get(byte_idx).copied().unwrap_or(0) as u32;
                bit_buffer |= next << bits_in_buffer;
                bits_in_buffer += 8;
                byte_idx += 1;
            }

            let raw = (bit_buffer & mask) as u8;
            bit_buffer >>= nbits;
            bits_in_buffer -= nbits;

            let decoded = raw as f32 * scale + offset;
            ip += q * decoded;
        }

        ip
    }

    /// 1-bit AVX512 inner product.
    ///
    /// For 1-bit codes: code 0 → decoded = -vmax, code 1 → decoded = 0.
    /// ip = sum(q[bit=0] * (-vmax)) = -vmax * (q_sum - float_sum)
    ///    = vmax * (float_sum - q_sum)
    ///
    /// padded_dim is always a multiple of 64, so no tail loop needed.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compute_ip_1bit_avx512(
        &self,
        state: &UsqQueryState,
        vmax: f32,
        packed_bits: &[u8],
    ) -> f32 {
        use std::arch::x86_64::*;

        let padded_dim = self.config.padded_dim();
        let mut acc = _mm512_setzero_ps();

        for chunk_idx in 0..(padded_dim / 16) {
            let q_offset = chunk_idx * 16;
            let bit_offset = chunk_idx * 2;
            let mask_bits =
                u16::from_le_bytes([packed_bits[bit_offset], packed_bits[bit_offset + 1]]);
            let mask = mask_bits as __mmask16;
            let q_vec = _mm512_loadu_ps(state.q_rot.as_ptr().add(q_offset));
            let selected = _mm512_maskz_mov_ps(mask, q_vec);
            acc = _mm512_add_ps(acc, selected);
        }

        let float_sum = _mm512_reduce_add_ps(acc);
        vmax * (float_sum - state.q_sum)
    }

    /// VNNI-accelerated inner product for 4-bit and 8-bit codes.
    ///
    /// Uses integer dot product (dpbusd) between quantized query (i8) and
    /// unpacked codes (u8), then converts to float:
    ///   float_ip = q_scale * (code_scale * int_ip + offset * q_quantized_sum)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn compute_ip_vnni(&self, state: &UsqQueryState, vmax: f32, packed_bits: &[u8]) -> f32 {
        let levels = (1u32 << self.config.nbits) as f32;
        let code_scale = (2.0 * vmax) / levels.max(1.0);
        let offset = -vmax;

        let int_ip = match self.config.nbits {
            8 => dot_u8_i8_avx512(packed_bits, &state.q_quantized),
            4 => {
                let padded_dim = self.config.padded_dim();
                let mut expanded = vec![0u8; padded_dim];
                unpack_packed_u4_into(packed_bits, &mut expanded);
                dot_u8_i8_avx512(&expanded, &state.q_quantized)
            }
            _ => return self.compute_ip_scalar(&state.q_rot, vmax, packed_bits),
        };

        state.q_scale * (code_scale * int_ip as f32 + offset * state.q_quantized_sum as f32)
    }

    /// Convenience score: precompute query state and call `score_with_meta`.
    pub fn score(&self, encoded: &UsqEncoded, query: &[f32]) -> f32 {
        assert_eq!(query.len(), self.config.dim);

        let state = self.precompute_query_state(query);

        self.score_with_meta(
            &state,
            encoded.norm,
            encoded.vmax,
            encoded.quant_quality,
            &encoded.packed_bits,
        )
    }

    // -----------------------------------------------------------------------
    // Internal quantization helpers — ported from hvq.rs.
    // -----------------------------------------------------------------------

    /// max(|unit[i]|), clamped to ≥ 1e-6.
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

    /// Greedy refinement quantizer for `nbits >= 4`.
    /// Writes quantized codes into `codes` (caller-owned, no allocation).
    /// Returns `(ip, qed_length)`.
    fn greedy_quantize_into(&self, o_hat: &[f32], nrefine: usize, codes: &mut [u16]) -> (f32, f32) {
        debug_assert!(
            o_hat.len() % 64 == 0,
            "o_hat must be padded to multiple of 64"
        );
        debug_assert_eq!(codes.len(), o_hat.len());
        let dim = o_hat.len();
        let nbits = self.config.nbits;
        let vmax = Self::unit_vmax(o_hat);
        let (scale, offset) = Self::compute_scale_offset(o_hat, nbits);
        let max_code = ((1u32 << nbits) - 1) as u16;

        let dequant = |code: u16| code as f32 * scale + offset;

        for (c, &x) in codes[..dim].iter_mut().zip(o_hat.iter()) {
            *c = Self::quantize_scalar(x, scale, offset, nbits) as u16;
        }

        let mut ip: f32 = codes
            .iter()
            .zip(o_hat.iter())
            .map(|(&c, &t)| dequant(c) * t)
            .sum();
        let mut qed_length: f32 = codes.iter().map(|&c| dequant(c).powi(2)).sum();
        let mut objective = Self::objective(ip, qed_length);

        let max_refine = nrefine.min(6);
        for _round in 0..max_refine {
            let mut improved = false;
            for i in 0..dim {
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
        (ip, qed_length)
    }

    /// Threshold-sweep quantizer for `nbits < 4` (ExRaBitQ style).
    /// Writes quantized codes into `codes` (caller-owned, no allocation).
    /// Returns `(ip, qed_length)`.
    fn fast_quantize_into(&self, o_hat: &[f32], codes: &mut [u16]) -> (f32, f32) {
        debug_assert!(
            o_hat.len() % 64 == 0,
            "o_hat must be padded to multiple of 64"
        );
        let dim = o_hat.len();
        let levels = 1u16 << self.config.nbits;
        let half_levels = levels / 2;
        let half_levels_f = half_levels as f32;
        let max_code = half_levels; // max magnitude for negative dims
        let vmax = Self::unit_vmax(o_hat);

        let mut abs_o = Vec::with_capacity(dim);
        let mut non_negative = Vec::with_capacity(dim);
        let mut max_level = Vec::with_capacity(dim);
        for &value in o_hat {
            let is_non_neg = value >= 0.0;
            abs_o.push(value.abs());
            non_negative.push(is_non_neg);
            max_level.push(if is_non_neg {
                half_levels.saturating_sub(1)
            } else {
                max_code
            });
        }

        let max_abs = abs_o.iter().cloned().fold(0.0f32, f32::max);
        if max_abs < 1e-12 {
            codes[..dim].fill(half_levels);
            return (0.0, 0.0);
        }

        // Build and sort critical-value events.
        let mut events: Vec<(f32, usize, u16)> = Vec::new();
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

        let n_events = events.len();
        let mut ei = 0;
        while ei < n_events {
            let cur_t = events[ei].0;
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
            let obj = if sum_sq > 0.0 {
                (sum_xc * sum_xc) / sum_sq
            } else {
                0.0
            };
            if obj > best_objective + 1e-10 {
                best_objective = obj;
                best_t = cur_t;
                best_sum_sq = sum_sq;
                best_sum_xc = sum_xc;
            }
        }

        let scale = vmax / half_levels_f.max(1.0);
        let ip = scale * best_sum_xc;
        let qed_length = scale * scale * best_sum_sq;
        for i in 0..dim {
            let magnitude = if abs_o[i] > 1e-12 {
                ((best_t * abs_o[i] + 0.5).floor() as u16).min(max_level[i])
            } else {
                0
            };
            codes[i] = if non_negative[i] {
                half_levels + magnitude
            } else {
                half_levels - magnitude
            };
        }

        (ip, qed_length)
    }

    /// Compute scale/offset for dequantization.
    /// Returns `(scale, offset)` where `offset = -vmax`, `scale = 2*vmax/levels`.
    fn compute_scale_offset(o_hat: &[f32], nbits: u8) -> (f32, f32) {
        let max_v = o_hat
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

    fn quantize_scalar(x: f32, scale: f32, offset: f32, nbits: u8) -> u8 {
        let max_code = ((1u32 << nbits) - 1) as f32;
        (((x - offset) / scale).round().clamp(0.0, max_code)) as u8
    }
}

// ---------------------------------------------------------------------------
// SIMD helper functions (module-level, not in impl block)
// ---------------------------------------------------------------------------

/// AVX512 VNNI dot product: sum(a[i] * b[i]) where a is u8, b is i8.
///
/// Processes 64 elements per iteration (4x `_mm512_dpbusd_epi32`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
#[allow(dead_code)]
unsafe fn dot_u8_i8_avx512(a: &[u8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 64;
    let mut acc = _mm512_setzero_si512();
    for i in 0..chunks {
        let off = i * 64;
        let va = _mm512_loadu_si512(a[off..].as_ptr() as *const __m512i);
        let vb = _mm512_loadu_si512(b[off..].as_ptr() as *const __m512i);
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    let lanes: [i32; 16] = std::mem::transmute(acc);
    let mut total = lanes.iter().sum::<i32>();
    for i in (chunks * 64)..n {
        total += a[i] as i32 * b[i] as i32;
    }
    total
}

/// Scalar fallback for 4-bit unpacking.
fn unpack_packed_u4_scalar(packed: &[u8], out: &mut [u8]) {
    for (idx, &byte) in packed.iter().enumerate() {
        out[idx * 2] = byte & 0x0F;
        out[idx * 2 + 1] = (byte >> 4) & 0x0F;
    }
}

/// Unpack packed 4-bit codes (2 codes per byte, low nibble first) into u8 array.
///
/// Uses SSE2 for the main loop, scalar fallback for the tail.
#[cfg(target_arch = "x86_64")]
fn unpack_packed_u4_into(packed: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let nibble_mask = unsafe { _mm_set1_epi8(0x0F) };
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    while in_off + 16 <= packed.len() {
        unsafe {
            let chunk = _mm_loadu_si128(packed.as_ptr().add(in_off) as *const __m128i);
            let low = _mm_and_si128(chunk, nibble_mask);
            let shifted = _mm_srli_epi16(chunk, 4);
            let high = _mm_and_si128(shifted, nibble_mask);
            let lo = _mm_unpacklo_epi8(low, high);
            let hi = _mm_unpackhi_epi8(low, high);
            _mm_storeu_si128(out.as_mut_ptr().add(out_off) as *mut __m128i, lo);
            _mm_storeu_si128(out.as_mut_ptr().add(out_off + 16) as *mut __m128i, hi);
        }
        in_off += 16;
        out_off += 32;
    }
    unpack_packed_u4_scalar(&packed[in_off..], &mut out[out_off..]);
}

/// Scalar fallback for unpacking 4-bit codes on non-x86_64.
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
fn unpack_packed_u4_into(packed: &[u8], out: &mut [u8]) {
    unpack_packed_u4_scalar(packed, out);
}

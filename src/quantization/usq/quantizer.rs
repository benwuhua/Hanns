use super::config::UsqConfig;
use super::rotator::UsqRotator;
use crate::quantization::turboquant::packed::pack_codes;

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

/// Unified scalar quantizer: center → pad → rotate → normalize → quantize.
///
/// Replaces the equivalent logic duplicated across HVQ and ExRaBitQ.
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

        // 1. Center and pad to padded_dim.
        let padded_dim = self.config.padded_dim();
        let mut residual = vec![0.0f32; padded_dim];
        for i in 0..self.config.dim {
            residual[i] = vector[i] - self.centroid[i];
        }
        // Dimensions beyond config.dim remain 0 (already zeroed).

        // 2. Rotate.
        let rotated = self.rotator.rotate(&residual);

        // 3. Encode the already-rotated residual.
        let mut encoded = self.encode_rotated(&rotated);

        // 4. Override norm_sq with the original vector's ||x||².
        //    encode_rotated stores ||residual||² but L2 distance needs ||x||².
        encoded.norm_sq = vector.iter().map(|x| x * x).sum();

        encoded
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

        // 4. Compute norm and normalize to unit sphere.
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        let o_hat: Vec<f32> = if norm > 1e-12 {
            rotated.iter().map(|x| x / norm).collect()
        } else {
            rotated.to_vec()
        };

        // 5. Quantize.
        let (codes, _ip, qed_length) = if self.config.nbits >= 4 {
            self.greedy_quantize(&o_hat, 6)
        } else {
            self.fast_quantize(&o_hat)
        };

        // 6. Compute per-vector metadata.
        let vmax = Self::unit_vmax(&o_hat);
        let quant_quality = qed_length.sqrt().max(1e-12);

        // 7. Pack B-bit codes.
        let mut packed_bits =
            Vec::with_capacity((padded_dim * self.config.nbits as usize).div_ceil(8));
        pack_codes(&codes, self.config.nbits, &mut packed_bits);

        // 8. 1-bit sign codes.
        let sign_bits = if self.config.nbits == 1 {
            // Same data.
            packed_bits.clone()
        } else {
            let sign_codes: Vec<u16> = o_hat
                .iter()
                .map(|&v| if v >= 0.0 { 1u16 } else { 0u16 })
                .collect();
            let mut sb = Vec::with_capacity(padded_dim / 8);
            pack_codes(&sign_codes, 1, &mut sb);
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

    /// Score for rerank:
    ///   score = centroid_score + norm * dequant_ip(q_rot, packed_bits) / quant_quality
    ///
    /// `q_rot` must already be in the rotated+padded space (length = padded_dim).
    pub fn score_with_meta(
        &self,
        q_rot: &[f32],
        centroid_score: f32,
        norm: f32,
        vmax: f32,
        quant_quality: f32,
        packed_bits: &[u8],
    ) -> f32 {
        let padded_dim = self.config.padded_dim();
        debug_assert_eq!(q_rot.len(), padded_dim);

        if quant_quality <= 1e-12 {
            return centroid_score;
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

        centroid_score + norm * ip / quant_quality
    }

    /// Convenience score: rotates the query and calls `score_with_meta`.
    pub fn score(&self, encoded: &UsqEncoded, query: &[f32]) -> f32 {
        assert_eq!(query.len(), self.config.dim);

        // Pad query.
        let padded_dim = self.config.padded_dim();
        let mut padded_q = vec![0.0f32; padded_dim];
        padded_q[..self.config.dim].copy_from_slice(query);

        // Rotate.
        let q_rot = self.rotator.rotate(&padded_q);

        // centroid_score = q_rot · rotated_centroid
        let centroid_score: f32 = q_rot
            .iter()
            .zip(self.rotated_centroid.iter())
            .map(|(a, b)| a * b)
            .sum();

        self.score_with_meta(
            &q_rot,
            centroid_score,
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
    /// Returns `(codes, ip, qed_length)`.
    fn greedy_quantize(&self, o_hat: &[f32], nrefine: usize) -> (Vec<u16>, f32, f32) {
        debug_assert!(o_hat.len() % 64 == 0, "o_hat must be padded to multiple of 64");
        let dim = o_hat.len();
        let nbits = self.config.nbits;
        let vmax = Self::unit_vmax(o_hat);
        let (scale, offset) = Self::compute_scale_offset(o_hat, nbits);
        let max_code = ((1u32 << nbits) - 1) as u16;

        let dequant = |code: u16| code as f32 * scale + offset;

        let mut codes: Vec<u16> = o_hat
            .iter()
            .map(|&x| Self::quantize_scalar(x, scale, offset, nbits) as u16)
            .collect();

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
        (codes, ip, qed_length)
    }

    /// Threshold-sweep quantizer for `nbits < 4` (ExRaBitQ style).
    /// Returns `(codes, ip, qed_length)`.
    fn fast_quantize(&self, o_hat: &[f32]) -> (Vec<u16>, f32, f32) {
        debug_assert!(o_hat.len() % 64 == 0, "o_hat must be padded to multiple of 64");
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
            let codes = vec![half_levels; dim];
            return (codes, 0.0, 0.0);
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

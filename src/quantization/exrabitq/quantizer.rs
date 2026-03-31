use super::{
    config::ExRaBitQConfig,
    rotator::ExRaBitQRotator,
    space::{decode_compact_levels, ip_fxu2, ip_fxu3, ip_fxu4, ip_fxu6, ip_fxu7, ip_fxu8},
};
use crate::quantization::turboquant::packed::{pack_codes, unpack_codes};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExFactor {
    pub xipnorm: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExShortFactors {
    pub ip: f32,
    pub sum_xb: f32,
    pub err: f32,
}

#[derive(Clone, Debug)]
pub struct QuantizationResult {
    pub codes: Vec<u8>,
    pub numerator: f32,
    pub denominator: f32,
    pub objective: f32,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct EncodedVector {
    pub short_code: Vec<u8>,
    pub long_code: Vec<u8>,
    pub factor: ExFactor,
    pub short_factors: ExShortFactors,
    pub x_norm: f32,
    pub x2: f32,
}

#[derive(Clone, Copy, Debug)]
struct ThresholdEvent {
    threshold: f64,
    dim: usize,
    level: u8,
}

#[derive(Clone)]
pub struct ExRaBitQQuantizer {
    config: ExRaBitQConfig,
    rotator: ExRaBitQRotator,
}

impl ExRaBitQQuantizer {
    pub fn new(config: ExRaBitQConfig) -> Result<Self, String> {
        let rotator = ExRaBitQRotator::new(&config);
        Ok(Self { config, rotator })
    }

    pub fn config(&self) -> &ExRaBitQConfig {
        &self.config
    }

    pub fn rotation_matrix(&self) -> &[f32] {
        self.rotator.matrix()
    }

    pub fn pad_vector(&self, vector: &[f32]) -> Vec<f32> {
        let mut padded = vec![0.0f32; self.config.padded_dim()];
        self.pad_vector_into(vector, &mut padded);
        padded
    }

    pub fn pad_vector_into(&self, vector: &[f32], padded: &mut [f32]) {
        assert_eq!(vector.len(), self.config.dim);
        assert_eq!(padded.len(), self.config.padded_dim());
        padded.fill(0.0);
        padded[..self.config.dim].copy_from_slice(vector);
    }

    pub fn rotate_padded(&self, vector: &[f32]) -> Vec<f32> {
        self.rotator.rotate_padded(vector)
    }

    pub fn rotate_padded_into(&self, vector: &[f32], out: &mut [f32]) {
        self.rotator.rotate_padded_into(vector, out);
    }

    pub fn inverse_rotate_padded(&self, vector: &[f32]) -> Vec<f32> {
        self.rotator.inverse_rotate_padded(vector)
    }

    pub fn fast_quantize_for_test(&self, vector: &[f32]) -> QuantizationResult {
        let padded = self.normalize_abs_padded(vector);
        self.fast_quantize_abs(&padded)
    }

    pub fn greedy_quantize_for_test(&self, vector: &[f32], rounds: usize) -> QuantizationResult {
        let padded = self.normalize_abs_padded(vector);
        self.greedy_quantize_abs(&padded, rounds)
    }

    pub fn compact_long_code_for_test(&self, raw_levels: &[u8]) -> Vec<u8> {
        self.store_compacted_code(raw_levels)
    }

    pub fn decode_long_code_levels_for_test(&self, long_code: &[u8]) -> Vec<u8> {
        decode_compact_levels(long_code, self.config.padded_dim(), self.config.ex_bits())
    }

    pub fn rotate_query_residual(&self, query: &[f32], centroid: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(query.len(), self.config.dim);
        assert_eq!(centroid.len(), self.config.dim);

        let mut residual = vec![0.0f32; self.config.padded_dim()];
        self.pad_vector_into(query, &mut residual);
        for (dst, &c) in residual[..self.config.dim].iter_mut().zip(centroid.iter()) {
            *dst -= c;
        }
        let rotated = self.rotate_padded(&residual);
        let y2 = rotated.iter().map(|x| x * x).sum::<f32>();
        (rotated, y2)
    }

    pub fn encode_with_centroid(&self, vector: &[f32], centroid: &[f32]) -> EncodedVector {
        assert_eq!(vector.len(), self.config.dim);
        assert_eq!(centroid.len(), self.config.dim);

        let mut residual = vec![0.0f32; self.config.padded_dim()];
        self.pad_vector_into(vector, &mut residual);
        for (dst, &c) in residual[..self.config.dim].iter_mut().zip(centroid.iter()) {
            *dst -= c;
        }
        let rotated = self.rotate_padded(&residual);
        self.encode_from_rotated(&rotated)
    }

    pub fn encode_with_rotated_centroid_into(
        &self,
        vector: &[f32],
        rotated_centroid: &[f32],
        padded: &mut [f32],
        rotated: &mut [f32],
    ) -> EncodedVector {
        assert_eq!(vector.len(), self.config.dim);
        assert_eq!(rotated_centroid.len(), self.config.padded_dim());
        assert_eq!(padded.len(), self.config.padded_dim());
        assert_eq!(rotated.len(), self.config.padded_dim());

        self.pad_vector_into(vector, padded);
        self.rotate_padded_into(padded, rotated);
        for (value, centroid_value) in rotated.iter_mut().zip(rotated_centroid.iter()) {
            *value -= *centroid_value;
        }
        self.encode_from_rotated(rotated)
    }

    pub fn encode_from_rotated(&self, rotated: &[f32]) -> EncodedVector {
        assert_eq!(rotated.len(), self.config.padded_dim());
        self.encode_rotated(rotated)
    }

    fn encode_rotated(&self, rotated: &[f32]) -> EncodedVector {
        let x2 = rotated.iter().map(|x| x * x).sum::<f32>();
        let x_norm = x2.sqrt();

        if x_norm <= 1e-12 {
            return EncodedVector {
                short_code: vec![0u8; self.config.short_code_bytes()],
                long_code: vec![0u8; self.config.long_code_bytes()],
                factor: ExFactor { xipnorm: 1.0 },
                short_factors: ExShortFactors::default(),
                x_norm: 0.0,
                x2: 0.0,
            };
        }

        let mut unit = rotated.to_vec();
        for value in &mut unit {
            *value /= x_norm;
        }

        let short_levels: Vec<u16> = unit
            .iter()
            .map(|&value| if value >= 0.0 { 1u16 } else { 0u16 })
            .collect();
        let mut short_code = Vec::with_capacity(self.config.short_code_bytes());
        pack_codes(&short_levels, 1, &mut short_code);

        let abs_unit: Vec<f32> = unit.iter().map(|value| value.abs()).collect();
        let quantized = self.fast_quantize_abs(&abs_unit);
        let max_level = self.max_level();
        let signed_long_levels: Vec<u16> = quantized
            .codes
            .iter()
            .zip(unit.iter())
            .map(|(&code, &value)| {
                if value >= 0.0 {
                    code as u16
                } else {
                    (max_level - code) as u16
                }
            })
            .collect();
        let signed_long_levels_u8: Vec<u8> =
            signed_long_levels.iter().map(|&value| value as u8).collect();
        let long_code = self.store_compacted_code(&signed_long_levels_u8);

        let factor = ExFactor {
            xipnorm: if quantized.numerator > 1e-12 {
                2.0 * x_norm / quantized.numerator
            } else {
                1.0
            },
        };

        let sum_abs = unit.iter().map(|value| value.abs()).sum::<f32>();
        let fac_norm = 1.0 / (self.config.padded_dim() as f32).sqrt();
        let fac_err_norm = if self.config.padded_dim() > 1 {
            2.0 / ((self.config.padded_dim() - 1) as f32).sqrt()
        } else {
            0.0
        };
        let mut o_obar = sum_abs * fac_norm;
        if !o_obar.is_finite() {
            o_obar = 0.8;
        }
        o_obar = o_obar.clamp(1e-4, 1.0);
        let err_term = ((1.0 - o_obar * o_obar).max(0.0) / (o_obar * o_obar)).sqrt();
        let short_factors = ExShortFactors {
            ip: if sum_abs > 1e-12 {
                4.0 * x_norm / sum_abs
            } else {
                1.0
            },
            sum_xb: short_levels.iter().map(|&value| value as f32).sum::<f32>(),
            err: err_term * fac_err_norm * 2.0 * x_norm,
        };

        EncodedVector {
            short_code,
            long_code,
            factor,
            short_factors,
            x_norm,
            x2,
        }
    }

    pub fn decode_unit_from_codes(&self, short_code: &[u8], long_code: &[u8]) -> Vec<f32> {
        let signs = unpack_codes(short_code, self.config.padded_dim(), 1);
        let stored_levels =
            decode_compact_levels(long_code, self.config.padded_dim(), self.config.ex_bits());
        let max_level = self.max_level() as u16;

        let mut magnitudes = vec![0u16; self.config.padded_dim()];
        let mut denominator = self.config.padded_dim() as f32 * 0.25;
        for i in 0..self.config.padded_dim() {
            let sign_positive = signs[i] != 0;
            let magnitude = if sign_positive {
                stored_levels[i] as u16
            } else {
                max_level.saturating_sub(stored_levels[i] as u16)
            };
            magnitudes[i] = magnitude;
            let mag_f = magnitude as f32;
            denominator += mag_f * mag_f + mag_f;
        }

        let norm = denominator.sqrt().max(1e-12);
        let mut decoded = vec![0.0f32; self.config.padded_dim()];
        for i in 0..self.config.padded_dim() {
            let sign = if signs[i] != 0 { 1.0 } else { -1.0 };
            decoded[i] = sign * (magnitudes[i] as f32 + 0.5) / norm;
        }
        decoded
    }

    pub fn approximate_distance_from_decoded(
        &self,
        q_rot_residual: &[f32],
        y2: f32,
        encoded: &EncodedVector,
    ) -> f32 {
        let unit = self.decode_unit_from_codes(&encoded.short_code, &encoded.long_code);
        let dot = q_rot_residual
            .iter()
            .zip(unit.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        encoded.x2 + y2 - 2.0 * encoded.x_norm * dot
    }

    pub fn rerank_distance(
        &self,
        residual: &[f32],
        half_sum_residual: f32,
        y2: f32,
        rabitq_ip: f32,
        encoded: &EncodedVector,
    ) -> f32 {
        self.rerank_distance_from_parts(
            residual,
            half_sum_residual,
            y2,
            rabitq_ip,
            encoded.factor.xipnorm,
            encoded.x2,
            &encoded.long_code,
        )
    }

    pub(crate) fn rerank_distance_from_parts(
        &self,
        residual: &[f32],
        half_sum_residual: f32,
        y2: f32,
        rabitq_ip: f32,
        xipnorm: f32,
        x2: f32,
        long_code: &[u8],
    ) -> f32 {
        let fac_rescale = (1u32 << self.config.ex_bits()) as f32;
        let long_ip = self.long_code_inner_product(residual, long_code);
        x2 + y2 - xipnorm * (fac_rescale * rabitq_ip + long_ip - (fac_rescale - 1.0) * half_sum_residual)
    }

    pub fn rerank_distance_high_accuracy(
        &self,
        unit_query: &[f32],
        sumq: f32,
        y: f32,
        y2: f32,
        ip_xb_qprime: f32,
        encoded: &EncodedVector,
    ) -> f32 {
        self.rerank_distance_high_accuracy_from_parts(
            unit_query,
            sumq,
            y,
            y2,
            ip_xb_qprime,
            encoded.factor.xipnorm,
            encoded.x2,
            &encoded.long_code,
        )
    }

    pub(crate) fn rerank_distance_high_accuracy_from_parts(
        &self,
        unit_query: &[f32],
        sumq: f32,
        y: f32,
        y2: f32,
        ip_xb_qprime: f32,
        xipnorm: f32,
        x2: f32,
        long_code: &[u8],
    ) -> f32 {
        let fac_rescale = (1u32 << self.config.ex_bits()) as f32;
        let long_ip = self.long_code_inner_product(unit_query, long_code);
        x2 + y2 - xipnorm * y * (fac_rescale * ip_xb_qprime + long_ip - (fac_rescale - 0.5) * sumq)
    }

    pub fn long_code_inner_product(&self, values: &[f32], long_code: &[u8]) -> f32 {
        debug_assert_eq!(values.len(), self.config.padded_dim());
        match self.config.ex_bits() {
            2 => ip_fxu2(values, long_code, self.config.padded_dim()),
            3 => ip_fxu3(values, long_code, self.config.padded_dim()),
            4 => ip_fxu4(values, long_code, self.config.padded_dim()),
            6 => ip_fxu6(values, long_code, self.config.padded_dim()),
            7 => ip_fxu7(values, long_code, self.config.padded_dim()),
            8 => ip_fxu8(values, long_code, self.config.padded_dim()),
            bits => unreachable!("unsupported ex_bits {bits}"),
        }
    }

    fn normalize_abs_padded(&self, vector: &[f32]) -> Vec<f32> {
        let dim = self.config.padded_dim();
        let mut padded = vec![0.0f32; dim];
        let len = vector.len().min(dim);
        padded[..len].copy_from_slice(&vector[..len]);
        let norm = padded.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for value in &mut padded {
            *value = (*value / norm).abs();
        }
        padded
    }

    fn max_level(&self) -> u8 {
        ((1usize << self.config.ex_bits()) - 1) as u8
    }

    fn evaluate_codes(&self, codes: &[u8], values: &[f32]) -> QuantizationResult {
        debug_assert_eq!(codes.len(), values.len());
        let mut numerator = 0.0f32;
        let mut denominator = values.len() as f32 * 0.25;

        for (&code, &value) in codes.iter().zip(values.iter()) {
            let code_f = code as f32;
            numerator += (code_f + 0.5) * value;
            denominator += code_f * code_f + code_f;
        }

        let objective = if denominator > 0.0 {
            (numerator * numerator) / denominator
        } else {
            0.0
        };

        QuantizationResult {
            codes: codes.to_vec(),
            numerator,
            denominator,
            objective,
            scale: 0.0,
        }
    }

    fn store_compacted_code(&self, raw_levels: &[u8]) -> Vec<u8> {
        debug_assert_eq!(raw_levels.len(), self.config.padded_dim());
        let mut compact = vec![0u8; self.config.long_code_bytes()];
        let ex_bits = self.config.ex_bits();

        match ex_bits {
            8 => {
                compact.copy_from_slice(raw_levels);
            }
            4 => {
                for (src, dst) in raw_levels
                    .chunks_exact(32)
                    .zip(compact.chunks_exact_mut(16))
                {
                    for lane in 0..16 {
                        dst[lane] = (src[lane] & 0x0F) | ((src[16 + lane] & 0x0F) << 4);
                    }
                }
            }
            2 => {
                for (src, dst) in raw_levels
                    .chunks_exact(64)
                    .zip(compact.chunks_exact_mut(16))
                {
                    for lane in 0..16 {
                        dst[lane] = (src[lane] & 0x03)
                            | ((src[16 + lane] & 0x03) << 2)
                            | ((src[32 + lane] & 0x03) << 4)
                            | ((src[48 + lane] & 0x03) << 6);
                    }
                }
            }
            3 => {
                for (src, dst) in raw_levels
                    .chunks_exact(64)
                    .zip(compact.chunks_exact_mut(24))
                {
                    for lane in 0..16 {
                        dst[lane] = (src[lane] & 0x03)
                            | ((src[16 + lane] & 0x03) << 2)
                            | ((src[32 + lane] & 0x03) << 4)
                            | ((src[48 + lane] & 0x03) << 6);
                    }

                    let mut top_bits = 0u64;
                    for lane in 0..64 {
                        let top = ((src[lane] >> 2) & 0x01) as u64;
                        let pos = ((lane & 7) << 3) | (lane >> 3);
                        top_bits |= top << pos;
                    }
                    dst[16..24].copy_from_slice(&top_bits.to_le_bytes());
                }
            }
            6 => {
                for (src, dst) in raw_levels
                    .chunks_exact(64)
                    .zip(compact.chunks_exact_mut(48))
                {
                    for lane in 0..16 {
                        dst[lane] =
                            (src[lane] & 0x3F) | (((src[32 + lane] >> 4) & 0x03) << 6);
                        dst[16 + lane] = (src[16 + lane] & 0x3F)
                            | (((src[48 + lane] >> 4) & 0x03) << 6);
                        dst[32 + lane] =
                            (src[32 + lane] & 0x0F) | ((src[48 + lane] & 0x0F) << 4);
                    }
                }
            }
            7 => {
                for (src, dst) in raw_levels
                    .chunks_exact(64)
                    .zip(compact.chunks_exact_mut(56))
                {
                    for lane in 0..16 {
                        dst[lane] =
                            (src[lane] & 0x3F) | (((src[32 + lane] >> 4) & 0x03) << 6);
                        dst[16 + lane] = (src[16 + lane] & 0x3F)
                            | (((src[48 + lane] >> 4) & 0x03) << 6);
                        dst[32 + lane] =
                            (src[32 + lane] & 0x0F) | ((src[48 + lane] & 0x0F) << 4);
                    }

                    let mut top_bits = 0u64;
                    for lane in 0..64 {
                        let top = ((src[lane] >> 6) & 0x01) as u64;
                        let pos = ((lane & 7) << 3) | (lane >> 3);
                        top_bits |= top << pos;
                    }
                    dst[48..56].copy_from_slice(&top_bits.to_le_bytes());
                }
            }
            bits => unreachable!("unsupported ex_bits {bits}"),
        }

        compact
    }

    fn fast_quantize_abs(&self, values: &[f32]) -> QuantizationResult {
        debug_assert_eq!(values.len(), self.config.padded_dim());
        let max_level = self.max_level();
        let max_o = values.iter().copied().fold(0.0f32, f32::max).max(1e-12) as f64;
        let t_start = (((1usize << self.config.ex_bits()) - 1) / 3) as f64 / max_o;
        let t_end = (((1usize << self.config.ex_bits()) - 1) + 10) as f64 / max_o;
        let mut current = vec![0u8; values.len()];
        let mut sqr_denominator = values.len() as f64 * 0.25;
        let mut numerator = 0.0f64;

        for (code, &value) in current.iter_mut().zip(values.iter()) {
            let level = (t_start * value as f64 + 1e-5f64).floor();
            *code = level.clamp(0.0, max_level as f64) as u8;
            let level = f64::from(*code);
            sqr_denominator += level * level + level;
            numerator += (level + 0.5) * value as f64;
        }

        let mut events = Vec::new();
        for (dim, &value) in values.iter().enumerate() {
            if value <= 0.0 {
                continue;
            }
            for level in current[dim].saturating_add(1)..=max_level {
                let threshold = f64::from(level) / value as f64;
                if threshold >= t_end {
                    break;
                }
                events.push(ThresholdEvent {
                    threshold,
                    dim,
                    level,
                });
            }
        }
        events.sort_by(|a, b| {
            a.threshold
                .total_cmp(&b.threshold)
                .then_with(|| a.dim.cmp(&b.dim))
                .then_with(|| a.level.cmp(&b.level))
        });

        let mut max_ip = 0.0f64;
        let mut best_t = 0.0f64;
        for event in events {
            let dim = event.dim;
            if current[dim] >= event.level {
                continue;
            }

            let previous = f64::from(current[dim]);
            let next = f64::from(event.level);
            current[dim] = event.level;
            sqr_denominator += (next * next + next) - (previous * previous + previous);
            numerator += (next - previous) * values[dim] as f64;

            let cur_ip = numerator / sqr_denominator.sqrt();
            if cur_ip > max_ip {
                max_ip = cur_ip;
                best_t = event.threshold;
            }
        }

        let mut final_codes = vec![0u8; values.len()];
        sqr_denominator = values.len() as f64 * 0.25;
        numerator = 0.0;
        for (idx, &value) in values.iter().enumerate() {
            let level = (best_t * value as f64 + 1e-5f64).floor();
            let level = level.clamp(0.0, max_level as f64) as u8;
            final_codes[idx] = level;
            let level = f64::from(level);
            sqr_denominator += level * level + level;
            numerator += (level + 0.5) * value as f64;
        }

        let mut _ip_norm = (1.0 / numerator) as f32;
        if !_ip_norm.is_finite() {
            _ip_norm = 1.0;
        }

        QuantizationResult {
            codes: final_codes,
            numerator: numerator as f32,
            denominator: sqr_denominator as f32,
            objective: if sqr_denominator > 0.0 {
                ((numerator * numerator) / sqr_denominator) as f32
            } else {
                0.0
            },
            scale: best_t as f32,
        }
    }

    fn greedy_quantize_abs(&self, values: &[f32], rounds: usize) -> QuantizationResult {
        debug_assert_eq!(values.len(), self.config.padded_dim());
        let max_level = self.max_level();
        let mut current = vec![0u8; values.len()];
        let mut best = self.evaluate_codes(&current, values);

        for _ in 0..rounds.max(1) {
            let mut improved = false;
            let mut best_dim = None;
            let mut best_eval = best.clone();

            for dim in 0..current.len() {
                if current[dim] >= max_level {
                    continue;
                }
                current[dim] += 1;
                let candidate = self.evaluate_codes(&current, values);
                current[dim] -= 1;

                if candidate.objective > best_eval.objective {
                    best_eval = candidate;
                    best_dim = Some(dim);
                }
            }

            if let Some(dim) = best_dim {
                current[dim] += 1;
                best = self.evaluate_codes(&current, values);
                improved = true;
            }

            if !improved {
                break;
            }
        }

        best
    }
}

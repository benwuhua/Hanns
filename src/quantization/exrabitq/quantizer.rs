use super::{config::ExRaBitQConfig, rotator::ExRaBitQRotator};
use crate::quantization::turboquant::packed::{pack_codes, unpack_codes};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
    threshold: f32,
    dim: usize,
}

impl PartialEq for ThresholdEvent {
    fn eq(&self, other: &Self) -> bool {
        self.threshold.to_bits() == other.threshold.to_bits() && self.dim == other.dim
    }
}

impl Eq for ThresholdEvent {}

impl PartialOrd for ThresholdEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ThresholdEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .threshold
            .total_cmp(&self.threshold)
            .then_with(|| other.dim.cmp(&self.dim))
    }
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
        assert_eq!(vector.len(), self.config.dim);
        let mut padded = vec![0.0f32; self.config.padded_dim()];
        padded[..self.config.dim].copy_from_slice(vector);
        padded
    }

    pub fn rotate_padded(&self, vector: &[f32]) -> Vec<f32> {
        self.rotator.rotate_padded(vector)
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

    pub fn rotate_query_residual(&self, query: &[f32], centroid: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(query.len(), self.config.dim);
        assert_eq!(centroid.len(), self.config.dim);

        let mut residual = self.pad_vector(query);
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

        let mut residual = self.pad_vector(vector);
        for (dst, &c) in residual[..self.config.dim].iter_mut().zip(centroid.iter()) {
            *dst -= c;
        }
        let rotated = self.rotate_padded(&residual);
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

        let mut unit = rotated.clone();
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
        let mut long_code = Vec::with_capacity(self.config.long_code_bytes());
        pack_codes(
            &signed_long_levels,
            self.config.ex_bits() as u8,
            &mut long_code,
        );

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
        let stored_levels = unpack_codes(
            long_code,
            self.config.padded_dim(),
            self.config.ex_bits() as u8,
        );
        let max_level = self.max_level() as u16;

        let mut magnitudes = vec![0u16; self.config.padded_dim()];
        let mut denominator = self.config.padded_dim() as f32 * 0.25;
        for i in 0..self.config.padded_dim() {
            let sign_positive = signs[i] != 0;
            let magnitude = if sign_positive {
                stored_levels[i]
            } else {
                max_level.saturating_sub(stored_levels[i])
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
        let fac_rescale = (1u32 << self.config.ex_bits()) as f32;
        let long_ip = self.long_code_inner_product(residual, &encoded.long_code);
        encoded.x2 + y2
            - encoded.factor.xipnorm
                * (fac_rescale * rabitq_ip + long_ip - (fac_rescale - 1.0) * half_sum_residual)
    }

    pub fn long_code_inner_product(&self, values: &[f32], long_code: &[u8]) -> f32 {
        debug_assert_eq!(values.len(), self.config.padded_dim());
        let bits = self.config.ex_bits();
        let mut result = 0.0f32;
        let mut bit_offset = 0usize;
        for &value in values {
            let mut code = 0u32;
            let mut written = 0usize;
            let mut remaining = bits;
            while remaining > 0 {
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let take = remaining.min(8 - bit_idx);
                let chunk_mask = (1u32 << take) - 1;
                let chunk = ((long_code.get(byte_idx).copied().unwrap_or(0) as u32) >> bit_idx)
                    & chunk_mask;
                code |= chunk << written;
                written += take;
                bit_offset += take;
                remaining -= take;
            }
            result += value * code as f32;
        }
        result
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

    fn fast_quantize_abs(&self, values: &[f32]) -> QuantizationResult {
        debug_assert_eq!(values.len(), self.config.padded_dim());
        let max_level = self.max_level();
        let max_o = values.iter().copied().fold(0.0f32, f32::max).max(1e-12);
        let t_start = (max_level as f32 / 3.0) / max_o;
        let mut current = vec![0u8; values.len()];

        for (code, &value) in current.iter_mut().zip(values.iter()) {
            let level = (t_start * value + 1e-5).floor();
            *code = level.clamp(0.0, max_level as f32) as u8;
        }

        let mut current_eval = self.evaluate_codes(&current, values);
        let mut best = current_eval.clone();
        best.scale = t_start;

        let mut heap = BinaryHeap::new();
        for (dim, &value) in values.iter().enumerate() {
            if value <= 1e-12 {
                continue;
            }
            if current[dim] < max_level {
                heap.push(ThresholdEvent {
                    threshold: (current[dim] as f32 + 1.0) / value,
                    dim,
                });
            }
        }

        while let Some(event) = heap.pop() {
            let dim = event.dim;
            if current[dim] >= max_level {
                continue;
            }

            current[dim] += 1;
            let code_f = current[dim] as f32;
            current_eval.numerator += values[dim];
            current_eval.denominator += 2.0 * code_f;
            current_eval.objective = if current_eval.denominator > 0.0 {
                (current_eval.numerator * current_eval.numerator) / current_eval.denominator
            } else {
                0.0
            };

            if current_eval.objective > best.objective {
                best = current_eval.clone();
                best.codes = current.clone();
                best.scale = event.threshold;
            }

            if current[dim] < max_level {
                heap.push(ThresholdEvent {
                    threshold: (current[dim] as f32 + 1.0) / values[dim],
                    dim,
                });
            }
        }

        best
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

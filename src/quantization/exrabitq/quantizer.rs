use super::{config::ExRaBitQConfig, rotator::ExRaBitQRotator};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, Default)]
pub struct ExFactor {
    pub xipnorm: f32,
}

#[derive(Clone, Debug)]
pub struct QuantizationResult {
    pub codes: Vec<u8>,
    pub numerator: f32,
    pub denominator: f32,
    pub objective: f32,
    pub scale: f32,
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

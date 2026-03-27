use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::codebook::gaussian_lloyd_max_centroids;
use super::config::{TurboQuantConfig, TurboRotationBackend};
use super::packed::{pack_codes, unpack_codes};
use super::rotation::HadamardRotation;

enum Rotation {
    Dense { matrix: Vec<f32>, dim: usize },
    Hadamard(HadamardRotation),
}

impl Rotation {
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense { matrix, dim } => {
                let mut out = vec![0.0f32; *dim];
                for (row_idx, out_value) in out.iter_mut().enumerate() {
                    let row = &matrix[row_idx * *dim..(row_idx + 1) * *dim];
                    *out_value = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                }
                out
            }
            Self::Hadamard(rotation) => rotation.rotate(v),
        }
    }

    fn inverse_rotate(&self, y: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense { matrix, dim } => {
                let mut out = vec![0.0f32; *dim];
                for col in 0..*dim {
                    let mut sum = 0.0f32;
                    for row in 0..*dim {
                        sum += matrix[row * *dim + col] * y[row];
                    }
                    out[col] = sum;
                }
                out
            }
            Self::Hadamard(rotation) => rotation.inverse_rotate(y),
        }
    }

    fn output_len(&self) -> usize {
        match self {
            Self::Dense { dim, .. } => *dim,
            Self::Hadamard(rotation) => rotation.d_pad,
        }
    }
}

pub struct TurboQuantMse {
    pub config: TurboQuantConfig,
    rotation: Rotation,
    centroids: Vec<f32>,
}

impl TurboQuantMse {
    pub fn new(config: TurboQuantConfig) -> Self {
        assert!(config.dim > 0, "dim must be > 0");
        assert!(
            (1..=8).contains(&config.bits_per_dim),
            "bits_per_dim must be in 1..=8"
        );

        let rotation = match config.rotation_backend {
            TurboRotationBackend::DenseOrthogonal => Rotation::Dense {
                matrix: build_dense_orthogonal_rotation(config.dim, config.rotation_seed),
                dim: config.dim,
            },
            TurboRotationBackend::Hadamard => {
                Rotation::Hadamard(HadamardRotation::new(config.dim, config.rotation_seed))
            }
        };
        let scale = (config.padded_dim() as f32).sqrt().recip();
        let centroids = gaussian_lloyd_max_centroids(config.bits_per_dim, config.padded_dim())
            .into_iter()
            .map(|c| (c as f32) * scale)
            .collect();

        Self {
            config,
            rotation,
            centroids,
        }
    }

    pub fn code_size_bytes(&self) -> usize {
        self.config.code_bytes()
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.config.dim);
        let owned = self.preprocess_input(v);
        let rotated = self.rotation.rotate(&owned); // d_pad for Hadamard, dim for Dense
        let codes: Vec<u16> = rotated
            .iter()
            .map(|&value| self.closest_centroid_index(value) as u16)
            .collect();
        let mut packed = Vec::with_capacity(self.code_size_bytes());
        pack_codes(&codes, self.config.bits_per_dim, &mut packed);
        packed
    }

    pub fn encode_batch(&self, n: usize, data: &[f32]) -> Vec<u8> {
        assert_eq!(data.len(), n * self.config.dim);
        let code_bytes = self.code_size_bytes();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let encoded: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| self.encode(&data[i * self.config.dim..(i + 1) * self.config.dim]))
                .collect();

            let mut out = Vec::with_capacity(n * code_bytes);
            for code in encoded {
                out.extend_from_slice(&code);
            }
            out
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::with_capacity(n * code_bytes);
            for i in 0..n {
                out.extend_from_slice(&self.encode(&data[i * self.config.dim..(i + 1) * self.config.dim]));
            }
            out
        }
    }

    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        let rotated = self.decode_rotated(code); // length = quantize_dim (d_pad or dim)
        self.rotation.inverse_rotate(&rotated)
    }

    /// Rotate query and return all quantize_dim coordinates (d_pad for Hadamard).
    /// Must be passed to score_ip / score_l2 as-is (do NOT truncate to dim).
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.config.dim);
        let owned = self.preprocess_input(query);
        self.rotation.rotate(&owned) // returns d_pad coords for Hadamard, dim for Dense
    }

    pub fn score_ip(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        self.score_ip_noalloc(q_rotated, code)
    }

    pub fn score_l2(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        self.score_l2_noalloc(q_rotated, code)
    }

    /// Zero-alloc IP score: decode packed indices on the fly.
    pub fn score_ip_noalloc(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        let qdim = self.rotation.output_len();
        assert_eq!(q_rotated.len(), qdim);
        let bits = self.config.bits_per_dim as usize;
        let mask = (1u32 << bits) - 1;
        let mut score = 0.0f32;
        for (i, &q) in q_rotated.iter().enumerate() {
            let idx = decode_packed_index(code, i, bits, mask);
            score += q * self.centroids[idx];
        }
        score
    }

    /// Zero-alloc L2 score: decode packed indices on the fly.
    pub fn score_l2_noalloc(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        let qdim = self.rotation.output_len();
        assert_eq!(q_rotated.len(), qdim);
        let bits = self.config.bits_per_dim as usize;
        let mask = (1u32 << bits) - 1;
        let mut score = 0.0f32;
        for (i, &q) in q_rotated.iter().enumerate() {
            let idx = decode_packed_index(code, i, bits, mask);
            let diff = q - self.centroids[idx];
            score += diff * diff;
        }
        score
    }

    /// Precompute ADC table for a rotated query. Most useful for bits <= 6.
    pub fn precompute_adc_table(&self, q_rotated: &[f32]) -> Vec<f32> {
        let qdim = self.rotation.output_len();
        assert_eq!(q_rotated.len(), qdim);
        let n_centroids = 1usize << self.config.bits_per_dim;
        let mut table = vec![0.0f32; qdim * n_centroids];
        for i in 0..qdim {
            let base = i * n_centroids;
            for c in 0..n_centroids {
                table[base + c] = q_rotated[i] * self.centroids[c];
            }
        }
        table
    }

    /// Score using a precomputed ADC table. Zero alloc, no multiply in the hot loop.
    pub fn score_ip_adc(&self, adc_table: &[f32], code: &[u8]) -> f32 {
        let qdim = self.rotation.output_len();
        let bits = self.config.bits_per_dim as usize;
        let n_centroids = 1usize << bits;
        assert_eq!(adc_table.len(), qdim * n_centroids);

        let mask = (1u32 << bits) - 1;
        let mut score = 0.0f32;
        for i in 0..qdim {
            let idx = decode_packed_index(code, i, bits, mask);
            score += adc_table[i * n_centroids + idx];
        }
        score
    }

    pub fn reconstruction_mse(&self, data: &[f32]) -> f32 {
        assert_eq!(data.len() % self.config.dim, 0);
        let n = data.len() / self.config.dim;
        if n == 0 {
            return 0.0;
        }

        let mut total = 0.0f32;
        for chunk in data.chunks_exact(self.config.dim) {
            let decoded = self.decode(&self.encode(chunk));
            total += chunk
                .iter()
                .zip(decoded.iter())
                .map(|(&a, &b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum::<f32>();
        }
        total / data.len() as f32
    }

    fn decode_rotated(&self, code: &[u8]) -> Vec<f32> {
        let qdim = self.rotation.output_len();
        let indices = unpack_codes(code, qdim, self.config.bits_per_dim);
        indices
            .into_iter()
            .map(|idx| self.centroids[idx as usize])
            .collect()
    }

    fn preprocess_input(&self, v: &[f32]) -> Vec<f32> {
        if !self.config.normalize_for_cosine {
            return v.to_vec();
        }
        normalize_vector(v)
    }

    fn closest_centroid_index(&self, value: f32) -> usize {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (idx, &centroid) in self.centroids.iter().enumerate() {
            let dist = (value - centroid).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        best_idx
    }
}

fn build_dense_orthogonal_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let random_values: Vec<f32> = (0..dim * dim)
        .map(|_| sample_standard_normal(&mut rng))
        .collect();
    let random_matrix = DMatrix::from_row_slice(dim, dim, &random_values);
    let q = QR::new(random_matrix).q();

    let mut rotation = vec![0.0f32; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            rotation[row * dim + col] = q[(row, col)];
        }
    }
    rotation
}

fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

fn decode_packed_index(code: &[u8], coord: usize, bits: usize, mask: u32) -> usize {
    let bit_offset = coord * bits;
    let byte_idx = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;
    let low = code.get(byte_idx).copied().unwrap_or(0) as u32;
    let high = code.get(byte_idx + 1).copied().unwrap_or(0) as u32;
    let word = low | (high << 8);
    ((word >> bit_in_byte) & mask) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::turboquant::config::TurboQuantConfig;

    #[test]
    fn test_encode_decode_roundtrip_shape() {
        let dim = 64usize;
        let bits = 4u8;
        let config = TurboQuantConfig::new(dim, bits);
        let code_bytes = config.code_bytes();
        let tq = TurboQuantMse::new(config);

        let v: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) - 0.5).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = v.iter().map(|x| x / norm).collect();

        let code = tq.encode(&v);
        assert_eq!(code.len(), code_bytes, "code size mismatch");

        let decoded = tq.decode(&code);
        assert_eq!(decoded.len(), dim, "decoded len mismatch");
    }

    #[test]
    fn test_hadamard_backend_keeps_code_size_and_decode_shape() {
        let dim = 96usize;
        let bits = 4u8;
        let config = TurboQuantConfig::new(dim, bits).with_hadamard();
        let code_bytes = config.code_bytes();
        let tq = TurboQuantMse::new(config);

        let vector: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) - 0.5).collect();
        let code = tq.encode(&vector);
        let decoded = tq.decode(&code);

        assert_eq!(code.len(), code_bytes);
        assert_eq!(decoded.len(), dim);
    }

    #[test]
    fn test_seed_determinism() {
        let dim = 32usize;
        let config1 = TurboQuantConfig::new(dim, 4);
        let config2 = TurboQuantConfig::new(dim, 4);
        let tq1 = TurboQuantMse::new(config1);
        let tq2 = TurboQuantMse::new(config2);

        let v: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let c1 = tq1.encode(&v);
        let c2 = tq2.encode(&v);
        assert_eq!(c1, c2, "same seed must produce identical codes");
    }

    #[test]
    fn test_mse_decreases_with_more_bits() {
        let dim = 128usize;
        let mut rng_val = 42u64;
        let v: Vec<f32> = (0..dim).map(|_| {
            rng_val = rng_val.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng_val >> 33) as f32 / u32::MAX as f32) - 0.5
        }).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = v.iter().map(|x| x / norm).collect();

        let mut prev_mse = f32::INFINITY;
        for bits in [2u8, 4, 6, 8] {
            let config = TurboQuantConfig::new(dim, bits);
            let tq = TurboQuantMse::new(config);
            let code = tq.encode(&v);
            let decoded = tq.decode(&code);
            let mse: f32 = v.iter().zip(&decoded).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / dim as f32;
            assert!(mse < prev_mse, "bits={bits}: mse={mse} should be < prev={prev_mse}");
            prev_mse = mse;
        }
    }

    #[test]
    fn test_zero_alloc_scores_match_existing_paths() {
        let dim = 64usize;
        let config = TurboQuantConfig::new(dim, 4).with_hadamard();
        let tq = TurboQuantMse::new(config);

        let vector: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) - 0.5).collect();
        let query: Vec<f32> = (0..dim).map(|i| 0.3 - i as f32 / dim as f32).collect();
        let code = tq.encode(&vector);
        let q_rot = tq.rotate_query(&query);
        let adc = tq.precompute_adc_table(&q_rot);

        let ip = tq.score_ip(&q_rot, &code);
        let ip_noalloc = tq.score_ip_noalloc(&q_rot, &code);
        let ip_adc = tq.score_ip_adc(&adc, &code);
        let l2 = tq.score_l2(&q_rot, &code);
        let l2_noalloc = tq.score_l2_noalloc(&q_rot, &code);

        assert!((ip - ip_noalloc).abs() < 1e-6);
        assert!((ip - ip_adc).abs() < 1e-6);
        assert!((l2 - l2_noalloc).abs() < 1e-6);
    }
}

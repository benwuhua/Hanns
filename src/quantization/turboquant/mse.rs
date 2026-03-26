use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::codebook::gaussian_lloyd_max_centroids;
use super::config::TurboQuantConfig;
use super::packed::{pack_codes, unpack_codes};

pub struct TurboQuantMse {
    pub config: TurboQuantConfig,
    rotation_matrix: Vec<f32>,
    centroids: Vec<f32>,
}

impl TurboQuantMse {
    pub fn new(config: TurboQuantConfig) -> Self {
        assert!(config.dim > 0, "dim must be > 0");
        assert!(
            (1..=8).contains(&config.bits_per_dim),
            "bits_per_dim must be in 1..=8"
        );

        let rotation_matrix = build_dense_orthogonal_rotation(config.dim, config.rotation_seed);
        let scale = (config.dim as f32).sqrt().recip();
        let centroids = gaussian_lloyd_max_centroids(config.bits_per_dim, config.dim)
            .into_iter()
            .map(|c| (c as f32) * scale)
            .collect();

        Self {
            config,
            rotation_matrix,
            centroids,
        }
    }

    pub fn code_size_bytes(&self) -> usize {
        self.config.code_bytes()
    }

    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.config.dim);
        let owned = self.preprocess_input(v);
        let rotated = self.rotate_slice(&owned);
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
        let rotated = self.decode_rotated(code);
        self.inverse_rotate_slice(&rotated)
    }

    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.config.dim);
        let owned = self.preprocess_input(query);
        self.rotate_slice(&owned)
    }

    pub fn score_ip(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        assert_eq!(q_rotated.len(), self.config.dim);
        let indices = unpack_codes(code, self.config.dim, self.config.bits_per_dim);
        q_rotated
            .iter()
            .zip(indices.iter())
            .map(|(&q, &idx)| q * self.centroids[idx as usize])
            .sum()
    }

    pub fn score_l2(&self, q_rotated: &[f32], code: &[u8]) -> f32 {
        assert_eq!(q_rotated.len(), self.config.dim);
        let indices = unpack_codes(code, self.config.dim, self.config.bits_per_dim);
        q_rotated
            .iter()
            .zip(indices.iter())
            .map(|(&q, &idx)| {
                let diff = q - self.centroids[idx as usize];
                diff * diff
            })
            .sum()
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
        let indices = unpack_codes(code, self.config.dim, self.config.bits_per_dim);
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

    fn rotate_slice(&self, v: &[f32]) -> Vec<f32> {
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for (row_idx, out_value) in out.iter_mut().enumerate() {
            let row = &self.rotation_matrix[row_idx * dim..(row_idx + 1) * dim];
            *out_value = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        }
        out
    }

    fn inverse_rotate_slice(&self, v: &[f32]) -> Vec<f32> {
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for col in 0..dim {
            let mut sum = 0.0f32;
            for row in 0..dim {
                sum += self.rotation_matrix[row * dim + col] * v[row];
            }
            out[col] = sum;
        }
        out
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
}

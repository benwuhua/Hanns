use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::quantization::turboquant::config::TurboQuantConfig;
use crate::quantization::turboquant::mse::TurboQuantMse;

const QJL_SCALE: f32 = 1.253_314_1;

pub struct TurboQuantProd {
    mse: TurboQuantMse,
    qjl_matrix: Vec<f32>,
    pub dim: usize,
    pub bits_per_dim: u8,
}

impl TurboQuantProd {
    pub fn new(config: TurboQuantConfig) -> Self {
        assert!(
            config.bits_per_dim >= 2,
            "TurboQuantProd requires bits_per_dim >= 2"
        );

        let mut mse_config = config.clone();
        mse_config.bits_per_dim -= 1;
        let mse = TurboQuantMse::new(mse_config);

        let mut rng = StdRng::seed_from_u64(config.rotation_seed.wrapping_add(1));
        let qjl_matrix = (0..config.dim * config.dim)
            .map(|_| sample_standard_normal(&mut rng))
            .collect();

        Self {
            mse,
            qjl_matrix,
            dim: config.dim,
            bits_per_dim: config.bits_per_dim,
        }
    }

    pub fn mse_bytes(&self) -> usize {
        self.mse.code_size_bytes()
    }

    pub fn qjl_bytes(&self) -> usize {
        self.dim.div_ceil(8)
    }

    pub fn code_bytes(&self) -> usize {
        self.mse_bytes() + self.qjl_bytes()
    }

    pub fn encode(&self, x: &[f32]) -> (Vec<u8>, f32) {
        assert_eq!(x.len(), self.dim);

        let mse_code = self.mse.encode(x);
        let x_mse = self.mse.decode(&mse_code);
        let residual: Vec<f32> = x.iter().zip(x_mse.iter()).map(|(&a, &b)| a - b).collect();
        let gamma = residual.iter().map(|&v| v * v).sum::<f32>().sqrt();

        let mut packed = Vec::with_capacity(self.code_bytes());
        packed.extend_from_slice(&mse_code);

        let mut qjl_bits = vec![0u8; self.qjl_bytes()];
        if gamma > 1e-12 {
            for row_idx in 0..self.dim {
                let row = &self.qjl_matrix[row_idx * self.dim..(row_idx + 1) * self.dim];
                let projection: f32 = row.iter().zip(residual.iter()).map(|(&a, &b)| a * b).sum();
                if projection >= 0.0 {
                    qjl_bits[row_idx / 8] |= 1 << (row_idx % 8);
                }
            }
        }
        packed.extend_from_slice(&qjl_bits);

        (packed, gamma)
    }

    pub fn rotate_query(&self, y: &[f32]) -> Vec<f32> {
        self.mse.rotate_query(y)
    }

    pub fn qjl_project_query(&self, y: &[f32]) -> Vec<f32> {
        assert_eq!(y.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        for (row_idx, out_value) in out.iter_mut().enumerate() {
            let row = &self.qjl_matrix[row_idx * self.dim..(row_idx + 1) * self.dim];
            *out_value = row.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        }
        out
    }

    pub fn score_ip(&self, y_rotated: &[f32], s_y: &[f32], packed: &[u8], gamma: f32) -> f32 {
        assert_eq!(s_y.len(), self.dim);
        let (mse_code, qjl_bits) = self.split_code(packed);
        let mse_score = self.mse.score_ip(y_rotated, mse_code);
        if gamma <= 1e-12 {
            return mse_score;
        }

        let qjl_dot: f32 = s_y
            .iter()
            .enumerate()
            .map(|(idx, &value)| {
                let bit = (qjl_bits[idx / 8] >> (idx % 8)) & 1;
                let sign = if bit == 1 { 1.0 } else { -1.0 };
                value * sign
            })
            .sum();

        mse_score + QJL_SCALE * gamma * qjl_dot / self.dim as f32
    }

    pub fn decode_mse(&self, packed: &[u8]) -> Vec<f32> {
        let (mse_code, _) = self.split_code(packed);
        self.mse.decode(mse_code)
    }

    pub fn reconstruction_mse(&self, data: &[f32]) -> f32 {
        assert_eq!(data.len() % self.dim, 0);
        if data.is_empty() {
            return 0.0;
        }

        let mut total = 0.0f32;
        for vector in data.chunks_exact(self.dim) {
            let (packed, _) = self.encode(vector);
            let decoded = self.decode_mse(&packed);
            total += vector
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

    fn split_code<'a>(&self, packed: &'a [u8]) -> (&'a [u8], &'a [u8]) {
        let mse_bytes = self.mse_bytes();
        let qjl_bytes = self.qjl_bytes();
        assert_eq!(
            packed.len(),
            mse_bytes + qjl_bytes,
            "packed code length mismatch"
        );
        packed.split_at(mse_bytes)
    }
}

fn sample_standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
    let u2 = rng.gen::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::turboquant::config::TurboQuantConfig;

    #[test]
    fn test_turboquant_prod_code_layout_and_score_is_finite() {
        let dim = 32usize;
        let quantizer = TurboQuantProd::new(TurboQuantConfig::new(dim, 4));
        let vector: Vec<f32> = (0..dim).map(|i| (i as f32 / dim as f32) - 0.5).collect();
        let query: Vec<f32> = (0..dim).map(|i| 0.25 - i as f32 / dim as f32).collect();

        let (packed, gamma) = quantizer.encode(&vector);
        let y_rot = quantizer.rotate_query(&query);
        let s_y = quantizer.qjl_project_query(&query);
        let score = quantizer.score_ip(&y_rot, &s_y, &packed, gamma);

        assert_eq!(packed.len(), quantizer.code_bytes());
        assert!(score.is_finite());
    }

    #[test]
    fn test_turboquant_prod_seed_determinism() {
        let config = TurboQuantConfig::new(16, 5).with_rotation_seed(7);
        let q1 = TurboQuantProd::new(config.clone());
        let q2 = TurboQuantProd::new(config);
        let vector: Vec<f32> = (0..16).map(|i| i as f32 * 0.125 - 1.0).collect();

        let (packed1, gamma1) = q1.encode(&vector);
        let (packed2, gamma2) = q2.encode(&vector);

        assert_eq!(packed1, packed2);
        assert_eq!(gamma1.to_bits(), gamma2.to_bits());
    }
}

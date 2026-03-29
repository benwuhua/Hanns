use super::config::ExRaBitQConfig;
use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone)]
pub struct ExRaBitQRotator {
    dim: usize,
    matrix: Vec<f32>,           // row-major padded_dim x padded_dim
    transpose: Vec<f32>,        // row-major padded_dim x padded_dim
}

impl ExRaBitQRotator {
    pub fn new(config: &ExRaBitQConfig) -> Self {
        let dim = config.padded_dim();
        let mut rng = StdRng::seed_from_u64(config.rotation_seed);
        let values: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let random = DMatrix::from_row_slice(dim, dim, &values);
        let q = QR::new(random).q();

        let mut matrix = vec![0.0f32; dim * dim];
        let mut transpose = vec![0.0f32; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                let value = q[(r, c)];
                matrix[r * dim + c] = value;
                transpose[c * dim + r] = value;
            }
        }

        Self {
            dim,
            matrix,
            transpose,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    pub fn rotate_padded(&self, padded: &[f32]) -> Vec<f32> {
        debug_assert_eq!(padded.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        for (r, out_r) in out.iter_mut().enumerate() {
            let row = &self.matrix[r * self.dim..(r + 1) * self.dim];
            *out_r = row.iter().zip(padded.iter()).map(|(a, b)| a * b).sum();
        }
        out
    }

    pub fn inverse_rotate_padded(&self, padded: &[f32]) -> Vec<f32> {
        debug_assert_eq!(padded.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        for (r, out_r) in out.iter_mut().enumerate() {
            let row = &self.transpose[r * self.dim..(r + 1) * self.dim];
            *out_r = row.iter().zip(padded.iter()).map(|(a, b)| a * b).sum();
        }
        out
    }
}

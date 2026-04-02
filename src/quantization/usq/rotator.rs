use super::config::UsqConfig;
use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone)]
pub struct UsqRotator {
    dim: usize,
    matrix: Vec<f32>,    // row-major padded_dim x padded_dim
    transpose: Vec<f32>, // row-major padded_dim x padded_dim
}

impl UsqRotator {
    pub fn new(config: &UsqConfig) -> Self {
        let dim = config.padded_dim();
        let mut rng = StdRng::seed_from_u64(config.seed);
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

    pub fn rotate(&self, padded: &[f32]) -> Vec<f32> {
        debug_assert_eq!(padded.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        self.rotate_into(padded, &mut out);
        out
    }

    pub fn rotate_into(&self, padded: &[f32], out: &mut [f32]) {
        debug_assert_eq!(padded.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        self.matvec_into(&self.matrix, padded, out);
    }

    pub fn inverse_rotate(&self, rotated: &[f32]) -> Vec<f32> {
        debug_assert_eq!(rotated.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        self.inverse_rotate_into(rotated, &mut out);
        out
    }

    pub fn inverse_rotate_into(&self, rotated: &[f32], out: &mut [f32]) {
        debug_assert_eq!(rotated.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        self.matvec_into(&self.transpose, rotated, out);
    }

    #[inline(always)]
    fn matvec_into(&self, matrix: &[f32], vec: &[f32], out: &mut [f32]) {
        let dim = self.dim;
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                unsafe { self.matvec_avx512(matrix, vec, out) };
                return;
            }
        }

        // Scalar fallback
        for r in 0..dim {
            let row = &matrix[r * dim..(r + 1) * dim];
            out[r] = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn matvec_avx512(&self, matrix: &[f32], vec: &[f32], out: &mut [f32]) {
        use std::arch::x86_64::*;
        let dim = self.dim;
        let vec_ptr = vec.as_ptr();
        let num16 = dim & !15; // round down to multiple of 16

        for r in 0..dim {
            let row_ptr = matrix.as_ptr().add(r * dim);
            let mut sum = _mm512_setzero_ps();
            for c in (0..num16).step_by(16) {
                let a = _mm512_loadu_ps(row_ptr.add(c));
                let b = _mm512_loadu_ps(vec_ptr.add(c));
                sum = _mm512_fmadd_ps(a, b, sum);
            }
            let mut total = _mm512_reduce_add_ps(sum);
            // Handle remainder
            for c in num16..dim {
                total += *row_ptr.add(c) * *vec_ptr.add(c);
            }
            *out.get_unchecked_mut(r) = total;
        }
    }
}

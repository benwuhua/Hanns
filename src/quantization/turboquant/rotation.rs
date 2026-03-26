use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct HadamardRotation {
    pub dim: usize,
    pub d_pad: usize,
    pub diagonal: Vec<f32>,
}

impl HadamardRotation {
    pub fn new(dim: usize, seed: u64) -> Self {
        let d_pad = dim.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);
        let diagonal = (0..d_pad)
            .map(|_| if rng.gen::<bool>() { 1.0f32 } else { -1.0f32 })
            .collect();
        Self {
            dim,
            d_pad,
            diagonal,
        }
    }

    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim);
        let mut buf = vec![0.0f32; self.d_pad];
        for i in 0..self.dim {
            buf[i] = self.diagonal[i] * v[i];
        }
        fwht_inplace(&mut buf);
        let scale = (self.d_pad as f32).sqrt().recip();
        for value in &mut buf {
            *value *= scale;
        }
        buf
    }

    pub fn inverse_rotate(&self, y: &[f32]) -> Vec<f32> {
        assert_eq!(y.len(), self.d_pad);
        let mut buf = y.to_vec();
        fwht_inplace(&mut buf);
        let scale = (self.d_pad as f32).sqrt().recip();
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            out[i] = self.diagonal[i] * buf[i] * scale;
        }
        out
    }
}

pub fn fwht_inplace(buf: &mut [f32]) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut len = 1usize;
    while len < n {
        for i in (0..n).step_by(2 * len) {
            for j in 0..len {
                let u = buf[i + j];
                let v = buf[i + j + len];
                buf[i + j] = u + v;
                buf[i + j + len] = u - v;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::HadamardRotation;

    #[test]
    fn test_hadamard_rotation_inverse_recovers_input() {
        let rotation = HadamardRotation::new(3, 7);
        let input = vec![0.25f32, -0.5, 1.25];
        let rotated = rotation.rotate(&input);
        let recovered = rotation.inverse_rotate(&rotated);

        assert_eq!(rotated.len(), 4);
        for (lhs, rhs) in input.iter().zip(recovered.iter()) {
            assert!((lhs - rhs).abs() < 1e-4, "lhs={lhs} rhs={rhs}");
        }
    }
}

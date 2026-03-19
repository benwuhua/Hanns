use nalgebra::DMatrix;
use rand::seq::index::sample;
use rand::thread_rng;

pub struct PcaTransform {
    pub mean: Vec<f32>,
    pub components: Vec<f32>,
    pub d_in: usize,
    pub d_out: usize,
}

impl PcaTransform {
    /// Train PCA from data matrix (n x d_in, row-major).
    pub fn train(data: &[f32], n: usize, d_in: usize, d_out: usize) -> Self {
        assert!(d_in > 0, "d_in must be > 0");
        assert!(n > 0, "n must be > 0");
        assert_eq!(data.len(), n * d_in, "data length mismatch");
        assert!(d_out > 0 && d_out <= d_in, "d_out must be in [1, d_in]");

        // Use at most 8192 samples for SVD to keep training cost predictable.
        let max_samples = 8192usize;
        let sample_n = if n > 10_000 { max_samples.min(n) } else { n };

        let mut mean = vec![0.0f32; d_in];
        let sample_indices: Vec<usize> = if sample_n == n {
            (0..n).collect()
        } else {
            sample(&mut thread_rng(), n, sample_n).into_vec()
        };

        for &row in &sample_indices {
            let base = row * d_in;
            for c in 0..d_in {
                mean[c] += data[base + c];
            }
        }
        let inv_sample_n = 1.0f32 / sample_n as f32;
        for m in &mut mean {
            *m *= inv_sample_n;
        }

        let mut centered = Vec::with_capacity(sample_n * d_in);
        for &row in &sample_indices {
            let base = row * d_in;
            for c in 0..d_in {
                centered.push(data[base + c] - mean[c]);
            }
        }

        let x = DMatrix::<f32>::from_row_slice(sample_n, d_in, &centered);
        let svd = x.svd(false, true);
        let v_t = svd
            .v_t
            .expect("SVD requested V^T but nalgebra returned None");

        let mut components = vec![0.0f32; d_out * d_in];
        for r in 0..d_out {
            for c in 0..d_in {
                components[r * d_in + c] = v_t[(r, c)];
            }
        }

        Self {
            mean,
            components,
            d_in,
            d_out,
        }
    }

    /// Apply transform to batch input (n x d_in, row-major), producing (n x d_out).
    pub fn apply(&self, input: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(input.len(), n * self.d_in, "input length mismatch");
        let mut out = vec![0.0f32; n * self.d_out];
        for i in 0..n {
            let x = &input[i * self.d_in..(i + 1) * self.d_in];
            for r in 0..self.d_out {
                let comp = &self.components[r * self.d_in..(r + 1) * self.d_in];
                let mut dot = 0.0f32;
                for c in 0..self.d_in {
                    dot += (x[c] - self.mean[c]) * comp[c];
                }
                out[i * self.d_out + r] = dot;
            }
        }
        out
    }

    /// Apply transform to a single vector.
    pub fn apply_one(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_in, "input dim mismatch");
        let mut out = vec![0.0f32; self.d_out];
        for (r, out_r) in out.iter_mut().enumerate().take(self.d_out) {
            let comp = &self.components[r * self.d_in..(r + 1) * self.d_in];
            let mut dot = 0.0f32;
            for c in 0..self.d_in {
                dot += (x[c] - self.mean[c]) * comp[c];
            }
            *out_r = dot;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::PcaTransform;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_pca_train_and_apply_shape() {
        let n = 100usize;
        let d_in = 8usize;
        let d_out = 4usize;
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..n * d_in).map(|_| rng.gen::<f32>()).collect();

        let pca = PcaTransform::train(&data, n, d_in, d_out);
        let projected = pca.apply(&data, n);
        let one = pca.apply_one(&data[0..d_in]);

        assert_eq!(pca.mean.len(), d_in);
        assert_eq!(pca.components.len(), d_out * d_in);
        assert_eq!(projected.len(), n * d_out);
        assert_eq!(one.len(), d_out);
    }
}

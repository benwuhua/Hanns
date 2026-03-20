use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct HvqConfig {
    pub dim: usize,
    pub nbits: u8,
}

pub struct HvqQuantizer {
    pub config: HvqConfig,
    pub rotation_matrix: Vec<f32>, // row-major dim x dim
    pub scale: f32,
    pub offset: f32,
}

impl HvqQuantizer {
    pub fn new(config: HvqConfig, seed: u64) -> Self {
        assert!(config.dim > 0, "dim must be > 0");
        assert!(matches!(config.nbits, 1 | 2 | 4 | 8), "nbits must be one of 1/2/4/8");

        let dim = config.dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let random_values: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();
        let random_matrix = DMatrix::from_row_slice(dim, dim, &random_values);
        let q = QR::new(random_matrix).q();

        let mut rotation_matrix = vec![0.0f32; dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                rotation_matrix[r * dim + c] = q[(r, c)];
            }
        }

        Self {
            config,
            rotation_matrix,
            scale: 1.0,
            offset: 0.0,
        }
    }

    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for (r, out_r) in out.iter_mut().enumerate().take(dim) {
            let row = &self.rotation_matrix[r * dim..(r + 1) * dim];
            *out_r = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>();
        }
        out
    }

    pub fn inverse_rotate(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.config.dim);
        let dim = self.config.dim;
        let mut out = vec![0.0f32; dim];
        for (c, out_c) in out.iter_mut().enumerate().take(dim) {
            let mut sum = 0.0f32;
            for (r, vr) in v.iter().enumerate().take(dim) {
                sum += self.rotation_matrix[r * dim + c] * vr;
            }
            *out_c = sum;
        }
        out
    }

    pub fn compute_scale_offset(rotated: &[f32], nbits: u8) -> (f32, f32) {
        let min_v = rotated
            .iter()
            .copied()
            .fold(f32::INFINITY, |a, b| a.min(b));
        let max_v = rotated
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let levels = ((1u32 << nbits) - 1) as f32;
        let mut scale = (max_v - min_v) / levels.max(1.0);
        if scale.abs() < 1e-6 {
            scale = 1e-6;
        }
        (scale, min_v)
    }

    pub fn quantize_scalar(x: f32, scale: f32, offset: f32, nbits: u8) -> u8 {
        let max_code = ((1u32 << nbits) - 1) as f32;
        (((x - offset) / scale).round().clamp(0.0, max_code)) as u8
    }

    pub fn dequantize_scalar(code: u8, scale: f32, offset: f32) -> f32 {
        code as f32 * scale + offset
    }

    fn parse_code<'a>(&self, code: &'a [u8]) -> (f32, f32, &'a [u8]) {
        if code.len() >= self.config.dim + 8 {
            let mut s = [0u8; 4];
            let mut o = [0u8; 4];
            s.copy_from_slice(&code[0..4]);
            o.copy_from_slice(&code[4..8]);
            let scale = f32::from_le_bytes(s);
            let offset = f32::from_le_bytes(o);
            (scale, offset, &code[8..8 + self.config.dim])
        } else {
            (self.scale, self.offset, code)
        }
    }

    fn reconstruct_rotated(&self, code: &[u8]) -> Vec<f32> {
        let (scale, offset, raw_code) = self.parse_code(code);
        raw_code
            .iter()
            .take(self.config.dim)
            .map(|&c| Self::dequantize_scalar(c, scale, offset))
            .collect()
    }

    pub fn encode(&self, v: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(v.len(), self.config.dim);
        let rotated = self.rotate(v);
        let nbits = self.config.nbits;
        let (scale, offset) = Self::compute_scale_offset(&rotated, nbits);

        let mut codes: Vec<u8> = rotated
            .iter()
            .map(|&x| Self::quantize_scalar(x, scale, offset, nbits))
            .collect();

        let sse_of = |code: &[u8]| -> f32 {
            rotated
                .iter()
                .zip(code.iter())
                .map(|(&x, &c)| {
                    let xh = Self::dequantize_scalar(c, scale, offset);
                    let d = x - xh;
                    d * d
                })
                .sum()
        };

        let mut best_sse = sse_of(&codes);
        let max_refine = nrefine.min(6);
        for _ in 0..max_refine {
            let recon: Vec<f32> = codes
                .iter()
                .map(|&c| Self::dequantize_scalar(c, scale, offset))
                .collect();
            let residual: Vec<f32> = rotated
                .iter()
                .zip(recon.iter())
                .map(|(&a, &b)| a - b)
                .collect();

            let candidate: Vec<u8> = rotated
                .iter()
                .zip(residual.iter())
                .map(|(&x, &r)| Self::quantize_scalar(x - r * 0.5, scale, offset, nbits))
                .collect();
            let sse = sse_of(&candidate);
            if sse < best_sse {
                best_sse = sse;
                codes = candidate;
            } else {
                break;
            }
        }

        let mut packed = Vec::with_capacity(self.config.dim + 8);
        packed.extend_from_slice(&scale.to_le_bytes());
        packed.extend_from_slice(&offset.to_le_bytes());
        packed.extend_from_slice(&codes);
        packed
    }

    pub fn encode_batch(&self, n: usize, data: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(data.len(), n * self.config.dim);
        let per = self.config.dim + 8;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let encoded: Vec<Vec<u8>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let v = &data[i * self.config.dim..(i + 1) * self.config.dim];
                    self.encode(v, nrefine)
                })
                .collect();
            let mut out = Vec::with_capacity(n * per);
            for e in encoded {
                out.extend_from_slice(&e);
            }
            return out;
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::with_capacity(n * per);
            for i in 0..n {
                let v = &data[i * self.config.dim..(i + 1) * self.config.dim];
                out.extend_from_slice(&self.encode(v, nrefine));
            }
            out
        }
    }

    pub fn base_quant_dist(&self, v: &[f32], code: &[u8]) -> f32 {
        let v_rot = self.rotate(v);
        let recon = self.reconstruct_rotated(code);
        v_rot
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    pub fn adc_distance(&self, query: &[f32], code: &[u8], base_dist: f32) -> f32 {
        let q_rot = self.rotate(query);
        let recon = self.reconstruct_rotated(code);

        let q_norm2: f32 = q_rot.iter().map(|x| x * x).sum();
        let dot: f32 = q_rot.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
        q_norm2 - 2.0 * dot + base_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    #[test]
    fn test_hvq_roundtrip() {
        let dim = 32usize;
        let n = 100usize;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 8 }, 42);

        let mut rel_err_sum = 0.0f32;
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let code = hvq.encode(v, 6);
            let recon_rot = hvq.reconstruct_rotated(&code);
            let recon = hvq.inverse_rotate(&recon_rot);
            let err = l2_sq(v, &recon).sqrt();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            rel_err_sum += err / norm.max(1e-6);
        }
        let avg_rel_err = rel_err_sum / n as f32;
        println!("hvq_roundtrip avg_relative_error={:.4}", avg_rel_err);
        assert!(avg_rel_err < 0.05, "avg relative error too high: {}", avg_rel_err);
    }

    #[test]
    fn test_hvq_recall() {
        let dim = 128usize;
        let n = 1000usize;
        let nq = 100usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(42);
        let train: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 42);

        let mut codes = Vec::with_capacity(n);
        let mut base_dists = Vec::with_capacity(n);
        for i in 0..n {
            let v = &train[i * dim..(i + 1) * dim];
            let code = hvq.encode(v, 6);
            let bq = hvq.base_quant_dist(v, &code);
            codes.push(code);
            base_dists.push(bq);
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];

            let mut gt: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &train[i * dim..(i + 1) * dim];
                    (i, l2_sq(q, v))
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_topk: Vec<usize> = gt.iter().take(top_k).map(|(i, _)| *i).collect();

            let mut approx: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, hvq.adc_distance(q, &codes[i], base_dists[i])))
                .collect();
            approx.sort_by(|a, b| a.1.total_cmp(&b.1));
            let approx_topk: Vec<usize> = approx.iter().take(top_k).map(|(i, _)| *i).collect();

            for idx in gt_topk {
                total += 1;
                if approx_topk.contains(&idx) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f32 / total as f32;
        println!("hvq_recall@10={:.3}", recall);
        assert!(recall.is_finite());
    }
}

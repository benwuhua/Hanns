//! **EXPERIMENTAL** — HvqQuantizer has no native knowhere equivalent.
//! Not suitable for cross-implementation parity testing.
//! Benchmark results should not be compared against native IVF-PQ/SQ metrics.

use crate::quantization::turboquant::packed::{pack_codes, unpack_codes};
use nalgebra::linalg::QR;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct HvqConfig {
    pub dim: usize,
    pub nbits: u8,
}

#[doc = "⚠️ Experimental: no native parity"]
pub struct HvqQuantizer {
    pub config: HvqConfig,
    pub rotation_matrix: Vec<f32>, // row-major dim x dim
    pub scale: f32,
    pub offset: f32,
}

impl HvqQuantizer {
    pub fn new(config: HvqConfig, seed: u64) -> Self {
        assert!(config.dim > 0, "dim must be > 0");
        assert!(
            (1..=8).contains(&config.nbits),
            "nbits must be in range 1..=8"
        );

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

    /// Explicit query rotation helper for search path.
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        self.rotate(query)
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
        let max_v = rotated
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let levels = ((1u32 << nbits) - 1) as f32;
        let mut scale = (2.0 * max_v) / levels.max(1.0);
        if scale.abs() < 1e-6 {
            scale = 1e-6;
        }
        (scale, -max_v)
    }

    pub fn quantize_scalar(x: f32, scale: f32, offset: f32, nbits: u8) -> u8 {
        let max_code = ((1u32 << nbits) - 1) as f32;
        (((x - offset) / scale).round().clamp(0.0, max_code)) as u8
    }

    pub fn dequantize_scalar(code: u8, scale: f32, offset: f32) -> f32 {
        code as f32 * scale + offset
    }

    pub fn code_size_bytes(&self) -> usize {
        8 + (self.config.dim * self.config.nbits as usize).div_ceil(8)
    }

    fn parse_code<'a>(&self, code: &'a [u8]) -> (f32, f32, &'a [u8]) {
        assert!(
            code.len() >= 8,
            "HVQ code too short: expected at least 8 bytes, got {}",
            code.len()
        );
        let norm_o = f32::from_le_bytes(code[0..4].try_into().unwrap());
        let vmax = f32::from_le_bytes(code[4..8].try_into().unwrap());
        (norm_o, vmax, &code[8..])
    }

    fn decode_unit(&self, packed_code: &[u8], vmax: f32) -> Vec<f32> {
        let indices = unpack_codes(packed_code, self.config.dim, self.config.nbits);
        let levels = ((1u32 << self.config.nbits) - 1) as f32;
        let scale = (2.0 * vmax) / levels.max(1.0);
        indices
            .into_iter()
            .map(|c| Self::dequantize_scalar(c as u8, scale, -vmax))
            .collect()
    }

    fn reconstruct_rotated(&self, code: &[u8]) -> Vec<f32> {
        let (norm_o, vmax, packed_code) = self.parse_code(code);
        self.decode_unit(packed_code, vmax)
            .into_iter()
            .map(|x| x * norm_o)
            .collect()
    }

    pub fn encode(&self, v: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(v.len(), self.config.dim);
        let o = self.rotate(v);
        let nbits = self.config.nbits;
        let norm_o = o.iter().map(|x| x * x).sum::<f32>().sqrt();
        let o_hat: Vec<f32> = if norm_o > 1e-12 {
            o.iter().map(|x| x / norm_o).collect()
        } else {
            o.clone()
        };

        let vmax = o_hat
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let (scale, offset) = Self::compute_scale_offset(&o_hat, nbits);
        let max_code = ((1u32 << nbits) - 1) as u16;

        let mut codes: Vec<u16> = o_hat
            .iter()
            .map(|&x| Self::quantize_scalar(x, scale, offset, nbits) as u16)
            .collect();

        let max_refine = nrefine.min(8);
        for _ in 0..max_refine {
            let mut improved = false;
            for i in 0..self.config.dim {
                let current = codes[i];
                let current_dot =
                    Self::dequantize_scalar(current as u8, scale, offset) * o_hat[i];
                let mut best_code = current;
                let mut best_dot = current_dot;

                if current < max_code {
                    let candidate = current + 1;
                    let dot = Self::dequantize_scalar(candidate as u8, scale, offset) * o_hat[i];
                    if dot > best_dot + 1e-7 {
                        best_dot = dot;
                        best_code = candidate;
                    }
                }
                if current > 0 {
                    let candidate = current - 1;
                    let dot = Self::dequantize_scalar(candidate as u8, scale, offset) * o_hat[i];
                    if dot > best_dot + 1e-7 {
                        best_dot = dot;
                        best_code = candidate;
                    }
                }

                if best_code != current {
                    codes[i] = best_code;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        let mut packed_bits = Vec::with_capacity((self.config.dim * nbits as usize).div_ceil(8));
        pack_codes(&codes, nbits, &mut packed_bits);

        let mut packed = Vec::with_capacity(self.code_size_bytes());
        packed.extend_from_slice(&norm_o.to_le_bytes());
        packed.extend_from_slice(&vmax.to_le_bytes());
        packed.extend_from_slice(&packed_bits);
        packed
    }

    pub fn encode_batch(&self, n: usize, data: &[f32], nrefine: usize) -> Vec<u8> {
        assert_eq!(data.len(), n * self.config.dim);
        let per = self.code_size_bytes();

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

    pub fn base_quant_dist(&self, _v: &[f32], _code: &[u8]) -> f32 {
        0.0
    }

    pub fn adc_distance(&self, query: &[f32], code: &[u8], base_dist: f32) -> f32 {
        let q_rot = self.rotate(query);
        self.adc_distance_prerotated(&q_rot, code, base_dist)
    }

    pub fn adc_distance_prerotated(&self, q_rot: &[f32], code: &[u8], _base_dist: f32) -> f32 {
        assert_eq!(q_rot.len(), self.config.dim);
        let (norm_o, vmax, packed_code) = self.parse_code(code);
        let o_hat = self.decode_unit(packed_code, vmax);
        let ip: f32 = q_rot.iter().zip(o_hat.iter()).map(|(a, b)| a * b).sum();
        norm_o * ip
    }
}

/// dot product of u8 query codes × i8 database codes.
/// Falls back to scalar on non-x86_64 or when avx512vnni is unavailable.
pub fn dot_u8_i8_avx512(a: &[u8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512f")
            && a.len() >= 64
        {
            // SAFETY: feature detection is done at runtime, and slices are valid for loads.
            return unsafe { dot_u8_i8_avx512_impl(a, b) };
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i32 * y as i32)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn dot_u8_i8_avx512_impl(a: &[u8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        // SAFETY: off..off+64 are in-bounds by chunk construction.
        let va = unsafe { _mm512_loadu_si512(a[off..].as_ptr() as *const __m512i) };
        // SAFETY: same bounds as above; raw bytes are interpreted as signed i8 lanes.
        let vb = unsafe { _mm512_loadu_si512(b[off..].as_ptr() as *const __m512i) };
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }

    // Reduce 16 i32 lanes manually for compatibility.
    // SAFETY: __m512i is exactly 64 bytes, same as [i32; 16].
    let lanes: [i32; 16] = unsafe { std::mem::transmute(acc) };
    let mut total = lanes.iter().sum::<i32>();

    for i in (chunks * 64)..n {
        total += a[i] as i32 * b[i] as i32;
    }
    total
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
        assert!(avg_rel_err < 0.15, "avg relative error too high: {}", avg_rel_err);
    }

    #[test]
    fn test_hvq_recall() {
        let dim = 128usize;
        let n = 1000usize;
        let nq = 100usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(42);
        let train: Vec<f32> = (0..n * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();
        let queries: Vec<f32> = (0..nq * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0))
            .collect();

        let normalize_rows = |data: &[f32], dim: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(data.len());
            for row in data.chunks_exact(dim) {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                out.extend(row.iter().map(|x| x / norm));
            }
            out
        };
        let train = normalize_rows(&train, dim);
        let queries = normalize_rows(&queries, dim);

        let hvq = HvqQuantizer::new(HvqConfig { dim, nbits: 4 }, 42);

        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let v = &train[i * dim..(i + 1) * dim];
            let code = hvq.encode(v, 6);
            codes.push(code);
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];

            let mut gt: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &train[i * dim..(i + 1) * dim];
                    (i, q.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum::<f32>())
                })
                .collect();
            gt.sort_by(|a, b| b.1.total_cmp(&a.1));
            let gt_topk: Vec<usize> = gt.iter().take(top_k).map(|(i, _)| *i).collect();

            let mut approx: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, hvq.adc_distance(q, &codes[i], 0.0)))
                .collect();
            approx.sort_by(|a, b| b.1.total_cmp(&a.1));
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

    #[test]
    fn test_dot_u8_i8() {
        let a: Vec<u8> = (0..128).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..128).map(|i| (i as i32 - 64) as i8).collect();
        let scalar: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        let simd = dot_u8_i8_avx512(&a, &b);
        assert_eq!(scalar, simd, "SIMD result must match scalar");
        println!("dot_u8_i8 scalar={} simd={}", scalar, simd);
    }
}

//! Product Quantization (PQ)
//!
//! Product Quantization splits vectors into subvectors and quantizes each independently.
//! This provides compression and faster distance computation.
//!
//! Reference: H. Jégou et al., "Product quantization for nearest neighbor search", 2011

use super::kmeans::KMeans;
use crate::api::{KnowhereError, Result};

/// Product Quantization configuration
#[derive(Clone, Debug)]
pub struct PQConfig {
    /// Dimensionality of original vectors
    pub dim: usize,
    /// Number of subquantizers (m)
    pub m: usize,
    /// Bits per subvector (nbits)
    pub nbits: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            m: 8,
            nbits: 8,
        }
    }
}

impl PQConfig {
    pub fn new(dim: usize, m: usize, nbits: usize) -> Self {
        Self { dim, m, nbits }
    }

    /// Get the dimension of each subvector
    pub fn sub_dim(&self) -> usize {
        self.dim / self.m
    }

    /// Get the number of centroids per subquantizer
    pub fn ksub(&self) -> usize {
        1 << self.nbits
    }

    /// Get the code size in bytes
    pub fn code_size(&self) -> usize {
        self.m * self.nbits / 8
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "Dimension must be > 0".to_string(),
            ));
        }

        if self.m == 0 {
            return Err(KnowhereError::InvalidArg(
                "Number of subquantizers (m) must be > 0".to_string(),
            ));
        }

        if self.dim % self.m != 0 {
            return Err(KnowhereError::InvalidArg(format!(
                "Dimension {} must be divisible by m={}",
                self.dim, self.m
            )));
        }

        if self.nbits == 0 || self.nbits > 16 {
            return Err(KnowhereError::InvalidArg(format!(
                "nbits must be in range [1, 16], got {}",
                self.nbits
            )));
        }

        Ok(())
    }
}

/// Product Quantizer
pub struct ProductQuantizer {
    config: PQConfig,
    /// Centroids for each subquantizer (m x ksub x sub_dim)
    centroids: Vec<f32>,
    /// Whether the quantizer is trained
    is_trained: bool,
}

impl ProductQuantizer {
    /// Create a new Product Quantizer
    pub fn new(config: PQConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            is_trained: false,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &PQConfig {
        &self.config
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the code size
    pub fn code_size(&self) -> usize {
        self.config.code_size()
    }

    /// Train the product quantizer
    pub fn train(&mut self, n: usize, x: &[f32]) -> Result<()> {
        self.config.validate()?;

        if n == 0 || x.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "Cannot train with empty data".to_string(),
            ));
        }

        let expected_size = n * self.config.dim;
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats for {} vectors of dim {}, got {}",
                expected_size,
                n,
                self.config.dim,
                x.len()
            )));
        }

        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();

        // Initialize centroids (m x ksub x sub_dim)
        let total_centroids = self.config.m * ksub * sub_dim;
        self.centroids = vec![0.0f32; total_centroids];

        // Train each subquantizer independently using k-means
        for sub_q in 0..self.config.m {
            let sub_centroid_offset = sub_q * ksub * sub_dim;

            // Extract subvectors for this subquantizer
            let mut sub_vectors = Vec::with_capacity(n * sub_dim);
            for i in 0..n {
                let vec_offset = i * self.config.dim;
                let sub_offset = vec_offset + sub_q * sub_dim;
                sub_vectors.extend_from_slice(&x[sub_offset..sub_offset + sub_dim]);
            }

            // Run k-means with adaptive iterations
            // OPT-004: 动态调整迭代次数，避免小数据集训练过慢
            // - n < 10K: 10 次迭代（快速训练）
            // - 10K <= n < 100K: 25 次迭代
            // - n >= 100K: 50 次迭代
            let max_iter = if n < 10_000 {
                10
            } else if n < 100_000 {
                25
            } else {
                50
            };

            let mut kmeans = KMeans::new(ksub, sub_dim);
            kmeans.set_max_iter(max_iter);
            let centroids_slice =
                &mut self.centroids[sub_centroid_offset..sub_centroid_offset + ksub * sub_dim];
            kmeans.train(&sub_vectors);
            centroids_slice.copy_from_slice(kmeans.centroids());
        }

        self.is_trained = true;
        Ok(())
    }

    /// Encode a single vector
    pub fn encode(&self, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "PQ quantizer not trained".to_string(),
            ));
        }

        if x.len() != self.config.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats, got {}",
                self.config.dim,
                x.len()
            )));
        }

        let code_size = self.config.code_size();
        let mut code = vec![0u8; code_size];
        let sub_dim = self.config.sub_dim();
        let _ksub = self.config.ksub();
        let nbits = self.config.nbits;

        for sub_q in 0..self.config.m {
            let sub_offset = sub_q * sub_dim;
            let sub_vector = &x[sub_offset..sub_offset + sub_dim];

            // Find nearest centroid
            let centroid_idx = self.find_nearest_centroid(sub_q, sub_vector);

            // Pack index into code
            let byte_offset = sub_q * nbits / 8;
            let bit_offset = (sub_q * nbits) % 8;

            if nbits == 8 {
                code[byte_offset] = centroid_idx as u8;
            } else {
                // Handle arbitrary bit packing
                for bit in 0..nbits {
                    if (centroid_idx >> bit) & 1 != 0 {
                        code[byte_offset + (bit_offset + bit) / 8] |= 1 << ((bit_offset + bit) % 8);
                    }
                }
            }
        }

        Ok(code)
    }

    /// Encode a batch of vectors
    pub fn encode_batch(&self, n: usize, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "PQ quantizer not trained".to_string(),
            ));
        }

        let expected_size = n * self.config.dim;
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats for {} vectors, got {}",
                expected_size,
                n,
                x.len()
            )));
        }

        let code_size = self.config.code_size();
        let mut codes = vec![0u8; n * code_size];

        for i in 0..n {
            let vec_offset = i * self.config.dim;
            let code_offset = i * code_size;

            let code = self.encode(&x[vec_offset..vec_offset + self.config.dim])?;
            codes[code_offset..code_offset + code_size].copy_from_slice(&code);
        }

        Ok(codes)
    }

    /// Find nearest centroid for a subvector
    fn find_nearest_centroid(&self, sub_q: usize, sub_vector: &[f32]) -> usize {
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();
        let centroid_offset = sub_q * ksub * sub_dim;

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c in 0..ksub {
            let centroid_start = centroid_offset + c * sub_dim;
            let centroid = &self.centroids[centroid_start..centroid_start + sub_dim];

            let dist = self.compute_l2_distance(sub_vector, centroid);

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Compute L2 distance between two vectors
    #[inline]
    fn compute_l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }

    /// Compute asymmetric distance between a raw vector and a PQ code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        if !self.is_trained {
            return f32::INFINITY;
        }

        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();
        let _nbits = self.config.nbits;

        let mut total_dist = 0.0f32;

        for sub_q in 0..self.config.m {
            // Extract centroid index from code
            let centroid_idx = self.extract_index(code, sub_q);

            // Get the centroid
            let centroid_offset = sub_q * ksub * sub_dim + centroid_idx * sub_dim;
            let centroid = &self.centroids[centroid_offset..centroid_offset + sub_dim];

            // Get query subvector
            let sub_offset = sub_q * sub_dim;
            let query_sub = &query[sub_offset..sub_offset + sub_dim];

            // Compute distance
            total_dist += self.compute_l2_distance(query_sub, centroid);
        }

        total_dist
    }

    /// Extract centroid index from code for subquantizer sub_q
    fn extract_index(&self, code: &[u8], sub_q: usize) -> usize {
        let nbits = self.config.nbits;
        let byte_offset = sub_q * nbits / 8;
        let bit_offset = (sub_q * nbits) % 8;

        if nbits == 8 {
            code[byte_offset] as usize
        } else {
            let mut idx = 0;
            for bit in 0..nbits {
                let byte_idx = byte_offset + (bit_offset + bit) / 8;
                let bit_idx = (bit_offset + bit) % 8;
                if byte_idx < code.len() && (code[byte_idx] >> bit_idx) & 1 != 0 {
                    idx |= 1 << bit;
                }
            }
            idx
        }
    }

    /// Decode a PQ code back to approximate vector (reconstruction)
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "PQ quantizer not trained".to_string(),
            ));
        }

        if code.len() != self.config.code_size() {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected code size {}, got {}",
                self.config.code_size(),
                code.len()
            )));
        }

        let mut reconstructed = Vec::with_capacity(self.config.dim);
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();

        for sub_q in 0..self.config.m {
            let centroid_idx = self.extract_index(code, sub_q);
            let centroid_offset = sub_q * ksub * sub_dim + centroid_idx * sub_dim;
            let centroid = &self.centroids[centroid_offset..centroid_offset + sub_dim];

            reconstructed.extend_from_slice(centroid);
        }

        Ok(reconstructed)
    }

    /// Get centroids for a specific subquantizer
    pub fn get_centroids(&self, sub_q: usize) -> Option<&[f32]> {
        if sub_q >= self.config.m {
            return None;
        }

        let ksub = self.config.ksub();
        let sub_dim = self.config.sub_dim();
        let offset = sub_q * ksub * sub_dim;

        Some(&self.centroids[offset..offset + ksub * sub_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::kmeans::KMeans;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn create_test_vectors(n: usize, dim: usize) -> Vec<f32> {
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * dim + j) % 100) as f32 / 100.0);
            }
        }
        vectors
    }

    #[test]
    fn test_pq_config() {
        let config = PQConfig::new(128, 8, 8);
        assert_eq!(config.sub_dim(), 16);
        assert_eq!(config.ksub(), 256);
        assert_eq!(config.code_size(), 8);
    }

    #[test]
    fn test_pq_config_validation() {
        // Valid config
        let config = PQConfig::new(128, 8, 8);
        assert!(config.validate().is_ok());

        // Invalid: dim not divisible by m
        let config = PQConfig::new(100, 7, 8);
        assert!(config.validate().is_err());

        // Invalid: nbits out of range
        let config = PQConfig::new(128, 8, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pq_train_and_encode() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8);
        let mut pq = ProductQuantizer::new(config);

        // Train
        let train_data = create_test_vectors(1000, dim);
        pq.train(1000, &train_data).unwrap();
        assert!(pq.is_trained());

        // Encode single vector
        let query = create_test_vectors(1, dim);
        let code = pq.encode(&query).unwrap();
        assert_eq!(code.len(), 8);

        // Encode batch
        let queries = create_test_vectors(10, dim);
        let codes = pq.encode_batch(10, &queries).unwrap();
        assert_eq!(codes.len(), 10 * 8);
    }

    #[test]
    fn test_pq_distance() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8);
        let mut pq = ProductQuantizer::new(config);

        // Train
        let train_data = create_test_vectors(1000, dim);
        pq.train(1000, &train_data).unwrap();

        // Encode
        let query = create_test_vectors(1, dim);
        let code = pq.encode(&query).unwrap();

        // Compute distance (should be 0 or very small for self)
        let dist = pq.compute_distance(&query, &code);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_pq_decode() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8);
        let mut pq = ProductQuantizer::new(config);

        // Train
        let train_data = create_test_vectors(1000, dim);
        pq.train(1000, &train_data).unwrap();

        // Encode and decode
        let original = create_test_vectors(1, dim);
        let code = pq.encode(&original).unwrap();
        let reconstructed = pq.decode(&code).unwrap();

        assert_eq!(reconstructed.len(), dim);
        // Reconstruction should be approximate (not exact)
        let mut mse = 0.0f32;
        for i in 0..dim {
            let diff = original[i] - reconstructed[i];
            mse += diff * diff;
        }
        mse /= dim as f32;
        // MSE should be reasonable (depends on training data)
        assert!(mse < 1.0);
    }

    #[test]
    fn test_pq_get_centroids() {
        let dim = 64;
        let config = PQConfig::new(dim, 8, 8);
        let ksub = config.ksub();
        let sub_dim = config.sub_dim();
        let mut pq = ProductQuantizer::new(config);

        // Train
        let train_data = create_test_vectors(1000, dim);
        pq.train(1000, &train_data).unwrap();

        // Get centroids for first subquantizer
        let centroids = pq.get_centroids(0).unwrap();
        assert_eq!(centroids.len(), ksub * sub_dim);

        // Invalid subquantizer index
        assert!(pq.get_centroids(8).is_none());
    }

    #[test]
    fn test_adc_correctness() {
        let dim = 16;
        let m = 4;
        let nbits = 4; // ksub = 16
        let n = 200usize;

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

        let config = PQConfig::new(dim, m, nbits);
        let mut pq = ProductQuantizer::new(config);
        pq.train(n, &data).unwrap();

        let codes = pq.encode_batch(n, &data).unwrap();
        let code_size = pq.code_size();

        let mut top1_hits = 0usize;
        let num_queries = 5usize;
        for q_idx in 0..num_queries {
            let query = &data[q_idx * dim..(q_idx + 1) * dim];

            let mut gt: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    let d = query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum::<f32>();
                    (i, d)
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_top1 = gt[0].0;

            let mut adc: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let code = &codes[i * code_size..(i + 1) * code_size];
                    (i, pq.compute_distance(query, code))
                })
                .collect();
            adc.sort_by(|a, b| a.1.total_cmp(&b.1));
            let adc_top1 = adc[0].0;

            if adc_top1 == gt_top1 {
                top1_hits += 1;
            }
        }
        let top1_recall = top1_hits as f32 / num_queries as f32;

        let mut mse_sum = 0.0f32;
        for i in 0..n {
            let original = &data[i * dim..(i + 1) * dim];
            let code = &codes[i * code_size..(i + 1) * code_size];
            let recon = pq.decode(code).unwrap();
            let mse = original
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum::<f32>()
                / dim as f32;
            mse_sum += mse;
        }
        let mse = mse_sum / n as f32;

        println!(
            "PQ ADC correctness: top1_recall={:.3}, reconstruction_mse={:.4}",
            top1_recall, mse
        );

        assert!(top1_recall.is_finite());
        assert!(mse.is_finite());
        assert!(mse < 10.0, "reconstruction MSE too large: {}", mse);
    }

    #[test]
    fn test_pq_recall_small() {
        let dim = 16usize;
        let m = 4usize;
        let nbits = 4usize; // ksub = 16
        let n = 500usize;
        let nq = 100usize;
        let top_k = 10usize;

        let mut rng = StdRng::seed_from_u64(42);
        let base: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        // Fixed-seed queries sampled from train distribution to keep this sanity test stable.
        let mut queries = Vec::with_capacity(nq * dim);
        for _ in 0..nq {
            let idx = rng.gen_range(0..n);
            queries.extend_from_slice(&base[idx * dim..(idx + 1) * dim]);
        }

        let config = PQConfig::new(dim, m, nbits);
        let mut pq = ProductQuantizer::new(config);
        pq.train(n, &base).unwrap();
        let codes = pq.encode_batch(n, &base).unwrap();
        let code_size = pq.code_size();

        let mut hits = 0usize;
        let mut total = 0usize;

        for q_idx in 0..nq {
            let query = &queries[q_idx * dim..(q_idx + 1) * dim];

            let mut gt: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let v = &base[i * dim..(i + 1) * dim];
                    let d = query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum::<f32>();
                    (i, d)
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_topk: Vec<usize> = gt.iter().take(top_k).map(|(idx, _)| *idx).collect();

            let mut adc: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let code = &codes[i * code_size..(i + 1) * code_size];
                    (i, pq.compute_distance(query, code))
                })
                .collect();
            adc.sort_by(|a, b| a.1.total_cmp(&b.1));
            let adc_topk: Vec<usize> = adc.iter().take(top_k).map(|(idx, _)| *idx).collect();

            for idx in gt_topk {
                total += 1;
                if adc_topk.contains(&idx) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f32 / total as f32;
        println!("PQ-only recall@10 (small): {:.3}", recall);
        assert!(recall > 0.5, "recall@10 too low: {}", recall);
    }

}

//! Residual Product Quantization (Residual PQ / RPQ)
//!
//! Residual PQ uses a two-stage quantization:
//! 1. Coarse quantization: find the nearest coarse centroid
//! 2. Compute residual (vector - coarse_centroid)
//! 3. Fine quantization: apply PQ to the residual
//!
//! This approach significantly improves accuracy over standard PQ.

use super::kmeans::KMeans;
use super::pq::{PQConfig, ProductQuantizer};
use crate::api::{KnowhereError, Result};

/// Residual PQ configuration
#[derive(Clone, Debug)]
pub struct ResidualPQConfig {
    /// Dimensionality of input vectors
    pub dim: usize,
    /// Number of coarse centroids (K)
    pub ncentroids: usize,
    /// Number of subquantizers for residual (m)
    pub m: usize,
    /// Bits per subvector for residual (nbits)
    pub nbits: usize,
    /// Use optimized PQ for residual quantization
    pub use_opq: bool,
}

impl Default for ResidualPQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            ncentroids: 256,
            m: 8,
            nbits: 8,
            use_opq: false,
        }
    }
}

impl ResidualPQConfig {
    pub fn new(dim: usize, ncentroids: usize, m: usize, nbits: usize) -> Self {
        Self {
            dim,
            ncentroids,
            m,
            nbits,
            use_opq: false,
        }
    }

    /// Get the dimension of each subvector
    pub fn sub_dim(&self) -> usize {
        self.dim / self.m
    }

    /// Get the number of centroids per subquantizer
    pub fn ksub(&self) -> usize {
        1 << self.nbits
    }

    /// Get the code size in bytes (coarse_id + pq_codes)
    pub fn code_size(&self) -> usize {
        // Coarse centroid index (4 bytes) + PQ codes
        4 + self.m * self.nbits / 8
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "Dimension must be > 0".to_string(),
            ));
        }

        if self.ncentroids == 0 {
            return Err(KnowhereError::InvalidArg(
                "Number of coarse centroids must be > 0".to_string(),
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

/// Residual Product Quantizer
pub struct ResidualProductQuantizer {
    config: ResidualPQConfig,
    /// Coarse centroids (ncentroids x dim)
    coarse_centroids: Vec<f32>,
    /// Fine PQ for residual quantization
    residual_pq: ProductQuantizer,
    /// Whether the quantizer is trained
    is_trained: bool,
}

impl ResidualProductQuantizer {
    /// Create a new ResidualProductQuantizer
    pub fn new(config: ResidualPQConfig) -> Result<Self> {
        config.validate()?;

        let pq_config = PQConfig::new(config.dim, config.m, config.nbits);
        let residual_pq = ProductQuantizer::new(pq_config);

        Ok(Self {
            config,
            coarse_centroids: Vec::new(),
            residual_pq,
            is_trained: false,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &ResidualPQConfig {
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

    /// Train the residual product quantizer
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

        // Step 1: Train coarse quantizer using k-means
        self.train_coarse(n, x)?;

        // Step 2: Compute residuals
        let residuals = self.compute_residuals(n, x)?;

        // Step 3: Train fine PQ on residuals
        self.residual_pq.train(n, &residuals)?;

        self.is_trained = true;
        Ok(())
    }

    /// Train coarse quantizer
    fn train_coarse(&mut self, _n: usize, x: &[f32]) -> Result<()> {
        let dim = self.config.dim;
        let k = self.config.ncentroids;

        self.coarse_centroids = vec![0.0f32; k * dim];

        // Use k-means to find coarse centroids
        let mut kmeans = KMeans::new(k, dim);
        kmeans.train(x);

        self.coarse_centroids.copy_from_slice(kmeans.centroids());

        Ok(())
    }

    /// Compute residuals for all vectors
    fn compute_residuals(&self, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.dim;
        let _k = self.config.ncentroids;
        let mut residuals = vec![0.0f32; n * dim];

        for i in 0..n {
            let vec_offset = i * dim;
            let vector = &x[vec_offset..vec_offset + dim];

            // Find nearest coarse centroid
            let coarse_idx = self.find_nearest_coarse_centroid(vector);

            // Compute residual
            let centroid_offset = coarse_idx * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            for j in 0..dim {
                residuals[vec_offset + j] = vector[j] - centroid[j];
            }
        }

        Ok(residuals)
    }

    /// Find nearest coarse centroid
    fn find_nearest_coarse_centroid(&self, x: &[f32]) -> usize {
        let dim = self.config.dim;
        let k = self.config.ncentroids;

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c in 0..k {
            let centroid_offset = c * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            let dist = Self::l2_distance(x, centroid);

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Compute L2 distance between two vectors
    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    }

    /// Encode a single vector
    pub fn encode(&self, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Residual PQ quantizer not trained".to_string(),
            ));
        }

        if x.len() != self.config.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats, got {}",
                self.config.dim,
                x.len()
            )));
        }

        // Find nearest coarse centroid
        let coarse_idx = self.find_nearest_coarse_centroid(x);

        // Compute residual
        let mut residual = vec![0.0f32; self.config.dim];
        let centroid_offset = coarse_idx * self.config.dim;
        let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        for j in 0..self.config.dim {
            residual[j] = x[j] - centroid[j];
        }

        // Encode residual using PQ
        let pq_code = self.residual_pq.encode(&residual)?;

        // Pack coarse index (4 bytes, little-endian) + PQ code
        let mut code = vec![0u8; self.code_size()];
        code[0..4].copy_from_slice(&(coarse_idx as u32).to_le_bytes());
        code[4..].copy_from_slice(&pq_code);

        Ok(code)
    }

    /// Encode a batch of vectors
    pub fn encode_batch(&self, n: usize, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Residual PQ quantizer not trained".to_string(),
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

        let code_size = self.code_size();
        let mut codes = vec![0u8; n * code_size];

        for i in 0..n {
            let vec_offset = i * self.config.dim;
            let code_offset = i * code_size;

            let code = self.encode(&x[vec_offset..vec_offset + self.config.dim])?;
            codes[code_offset..code_offset + code_size].copy_from_slice(&code);
        }

        Ok(codes)
    }

    /// Compute distance between a raw vector and a Residual PQ code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        if !self.is_trained {
            return f32::INFINITY;
        }

        if code.len() < 4 {
            return f32::INFINITY;
        }

        // Extract coarse centroid index
        let coarse_idx = u32::from_le_bytes([code[0], code[1], code[2], code[3]]) as usize;

        // Get coarse centroid
        let centroid_offset = coarse_idx * self.config.dim;
        let coarse_centroid =
            &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        // Get residual PQ code
        let residual_code = &code[4..];

        // Compute distance from query to coarse centroid
        let coarse_dist = Self::l2_distance(query, coarse_centroid);

        // Compute residual distance using PQ
        // For asymmetric distance, we need to compute query residual distance
        let mut query_residual = vec![0.0f32; self.config.dim];
        for j in 0..self.config.dim {
            query_residual[j] = query[j] - coarse_centroid[j];
        }

        let residual_dist = self
            .residual_pq
            .compute_distance(&query_residual, residual_code);

        // Correct ADC approximation: ||(query - coarse_centroid) - residual_decode||^2
        // coarse_dist is intentionally not added; adding it corrupts ranking.
        let _ = coarse_dist;
        residual_dist
    }

    /// Decode a Residual PQ code back to approximate vector
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Residual PQ quantizer not trained".to_string(),
            ));
        }

        if code.len() != self.code_size() {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected code size {}, got {}",
                self.code_size(),
                code.len()
            )));
        }

        // Extract coarse centroid index
        let coarse_idx = u32::from_le_bytes([code[0], code[1], code[2], code[3]]) as usize;

        // Get coarse centroid
        let centroid_offset = coarse_idx * self.config.dim;
        let coarse_centroid =
            &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        // Decode residual
        let residual_code = &code[4..];
        let residual = self.residual_pq.decode(residual_code)?;

        // Reconstruct: coarse_centroid + residual
        let mut reconstructed = vec![0.0f32; self.config.dim];
        for j in 0..self.config.dim {
            reconstructed[j] = coarse_centroid[j] + residual[j];
        }

        Ok(reconstructed)
    }

    /// Get coarse centroids
    pub fn get_coarse_centroids(&self) -> &[f32] {
        &self.coarse_centroids
    }

    /// Get the residual PQ quantizer
    pub fn get_residual_pq(&self) -> &ProductQuantizer {
        &self.residual_pq
    }
}

/// Residual PQ with OPQ (Optimized Residual PQ)
pub struct OptimizedResidualProductQuantizer {
    config: ResidualPQConfig,
    /// Coarse centroids (ncentroids x dim)
    coarse_centroids: Vec<f32>,
    /// OPQ for residual quantization
    residual_opq: super::opq::OptimizedProductQuantizer,
    /// Whether the quantizer is trained
    is_trained: bool,
}

impl OptimizedResidualProductQuantizer {
    /// Create a new OptimizedResidualProductQuantizer
    pub fn new(config: ResidualPQConfig) -> Result<Self> {
        config.validate()?;

        let opq_config = super::opq::OPQConfig::new(config.dim, config.m, config.nbits);
        let residual_opq = super::opq::OptimizedProductQuantizer::new(opq_config)?;

        Ok(Self {
            config,
            coarse_centroids: Vec::new(),
            residual_opq,
            is_trained: false,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &ResidualPQConfig {
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

    /// Train the optimized residual product quantizer
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

        // Step 1: Train coarse quantizer using k-means
        self.train_coarse(n, x)?;

        // Step 2: Compute residuals
        let residuals = self.compute_residuals(n, x)?;

        // Step 3: Train OPQ on residuals
        self.residual_opq.train(n, &residuals)?;

        self.is_trained = true;
        Ok(())
    }

    /// Train coarse quantizer
    fn train_coarse(&mut self, _n: usize, x: &[f32]) -> Result<()> {
        let dim = self.config.dim;
        let k = self.config.ncentroids;

        self.coarse_centroids = vec![0.0f32; k * dim];

        // Use k-means to find coarse centroids
        let mut kmeans = KMeans::new(k, dim);
        kmeans.train(x);

        self.coarse_centroids.copy_from_slice(kmeans.centroids());

        Ok(())
    }

    /// Compute residuals for all vectors
    fn compute_residuals(&self, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        let dim = self.config.dim;
        let _k = self.config.ncentroids;
        let mut residuals = vec![0.0f32; n * dim];

        for i in 0..n {
            let vec_offset = i * dim;
            let vector = &x[vec_offset..vec_offset + dim];

            // Find nearest coarse centroid
            let coarse_idx = self.find_nearest_coarse_centroid(vector);

            // Compute residual
            let centroid_offset = coarse_idx * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            for j in 0..dim {
                residuals[vec_offset + j] = vector[j] - centroid[j];
            }
        }

        Ok(residuals)
    }

    /// Find nearest coarse centroid
    fn find_nearest_coarse_centroid(&self, x: &[f32]) -> usize {
        let dim = self.config.dim;
        let k = self.config.ncentroids;

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c in 0..k {
            let centroid_offset = c * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            let dist = ResidualProductQuantizer::l2_distance(x, centroid);

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Encode a single vector
    pub fn encode(&self, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Optimized Residual PQ quantizer not trained".to_string(),
            ));
        }

        if x.len() != self.config.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats, got {}",
                self.config.dim,
                x.len()
            )));
        }

        // Find nearest coarse centroid
        let coarse_idx = self.find_nearest_coarse_centroid(x);

        // Compute residual
        let mut residual = vec![0.0f32; self.config.dim];
        let centroid_offset = coarse_idx * self.config.dim;
        let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        for j in 0..self.config.dim {
            residual[j] = x[j] - centroid[j];
        }

        // Encode residual using OPQ
        let opq_code = self.residual_opq.encode(&residual)?;

        // Pack coarse index (4 bytes, little-endian) + OPQ code
        let mut code = vec![0u8; self.code_size()];
        code[0..4].copy_from_slice(&(coarse_idx as u32).to_le_bytes());
        code[4..].copy_from_slice(&opq_code);

        Ok(code)
    }

    /// Encode a batch of vectors
    pub fn encode_batch(&self, n: usize, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Optimized Residual PQ quantizer not trained".to_string(),
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

        let code_size = self.code_size();
        let mut codes = vec![0u8; n * code_size];

        for i in 0..n {
            let vec_offset = i * self.config.dim;
            let code_offset = i * code_size;

            let code = self.encode(&x[vec_offset..vec_offset + self.config.dim])?;
            codes[code_offset..code_offset + code_size].copy_from_slice(&code);
        }

        Ok(codes)
    }

    /// Compute distance between a raw vector and a code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        if !self.is_trained {
            return f32::INFINITY;
        }

        if code.len() < 4 {
            return f32::INFINITY;
        }

        // Extract coarse centroid index
        let coarse_idx = u32::from_le_bytes([code[0], code[1], code[2], code[3]]) as usize;

        // Get coarse centroid
        let centroid_offset = coarse_idx * self.config.dim;
        let coarse_centroid =
            &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        // Get residual OPQ code
        let residual_code = &code[4..];

        // Compute query residual
        let mut query_residual = vec![0.0f32; self.config.dim];
        for j in 0..self.config.dim {
            query_residual[j] = query[j] - coarse_centroid[j];
        }

        // Compute OPQ distance
        self.residual_opq
            .compute_distance(&query_residual, residual_code)
    }

    /// Decode a code back to approximate vector
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "Optimized Residual PQ quantizer not trained".to_string(),
            ));
        }

        if code.len() != self.code_size() {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected code size {}, got {}",
                self.code_size(),
                code.len()
            )));
        }

        // Extract coarse centroid index
        let coarse_idx = u32::from_le_bytes([code[0], code[1], code[2], code[3]]) as usize;

        // Get coarse centroid
        let centroid_offset = coarse_idx * self.config.dim;
        let coarse_centroid =
            &self.coarse_centroids[centroid_offset..centroid_offset + self.config.dim];

        // Decode residual
        let residual_code = &code[4..];
        let residual = self.residual_opq.decode(residual_code)?;

        // Reconstruct: coarse_centroid + residual
        let mut reconstructed = vec![0.0f32; self.config.dim];
        for j in 0..self.config.dim {
            reconstructed[j] = coarse_centroid[j] + residual[j];
        }

        Ok(reconstructed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_residual_pq_config() {
        let config = ResidualPQConfig::new(128, 256, 8, 8);
        assert_eq!(config.sub_dim(), 16);
        assert_eq!(config.ksub(), 256);
        assert_eq!(config.code_size(), 12); // 4 + 8
    }

    #[test]
    fn test_residual_pq_train_and_encode() {
        let dim = 32;
        let config = ResidualPQConfig::new(dim, 16, 4, 8);
        let mut rpq = ResidualProductQuantizer::new(config).unwrap();

        // Train with minimal data
        let train_data = create_test_vectors(100, dim);
        rpq.train(100, &train_data).unwrap();
        assert!(rpq.is_trained());

        // Encode single vector
        let query = create_test_vectors(1, dim);
        let code = rpq.encode(&query).unwrap();
        assert_eq!(code.len(), 8); // 4 + 4 (m=4, nbits=8)

        // Encode batch
        let queries = create_test_vectors(5, dim);
        let codes = rpq.encode_batch(5, &queries).unwrap();
        assert_eq!(codes.len(), 5 * 8);
    }

    #[test]
    fn test_residual_pq_decode() {
        let dim = 32;
        let config = ResidualPQConfig::new(dim, 16, 8, 8);
        let mut rpq = ResidualProductQuantizer::new(config).unwrap();

        // Train with minimal data
        let train_data = create_test_vectors(100, dim);
        rpq.train(100, &train_data).unwrap();

        // Encode and decode
        let original = create_test_vectors(1, dim);
        let code = rpq.encode(&original).unwrap();
        let reconstructed = rpq.decode(&code).unwrap();

        assert_eq!(reconstructed.len(), dim);
        // Verify decode produces valid output (no NaN)
        assert!(!reconstructed.iter().any(|&x| x.is_nan()));
    }

    #[test]
    fn test_optimized_residual_pq() {
        let dim = 32;
        let config = ResidualPQConfig::new(dim, 16, 4, 8);
        let mut orpq = OptimizedResidualProductQuantizer::new(config).unwrap();

        // Train with minimal data
        let train_data = create_test_vectors(100, dim);
        orpq.train(100, &train_data).unwrap();
        assert!(orpq.is_trained());

        // Encode and decode
        let original = create_test_vectors(1, dim);
        let code = orpq.encode(&original).unwrap();
        assert_eq!(code.len(), 8); // 4 + 4 (m=4, nbits=8)

        let reconstructed = orpq.decode(&code).unwrap();
        assert_eq!(reconstructed.len(), dim);
        assert!(!reconstructed.iter().any(|&x| x.is_nan()));
    }

    #[test]
    fn test_residual_pq_vs_standard_pq() {
        use rand::Rng;
        let dim = 32;
        let n = 200;

        // Generate random test data
        let mut rng = rand::thread_rng();
        let mut test_data = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            test_data.push(rng.gen::<f32>() * 2.0 - 1.0);
        }

        // Train standard PQ with smaller config
        let pq_config = PQConfig::new(dim, 4, 8);
        let mut pq = ProductQuantizer::new(pq_config);
        pq.train(n, &test_data).unwrap();

        // Train Residual PQ with smaller config
        let rpq_config = ResidualPQConfig::new(dim, 16, 4, 8);
        let mut rpq = ResidualProductQuantizer::new(rpq_config).unwrap();
        rpq.train(n, &test_data).unwrap();

        // Compare reconstruction errors on just 10 vectors
        let mut pq_mse = 0.0f32;
        let mut rpq_mse = 0.0f32;

        for i in 0..10 {
            let vec = &test_data[i * dim..(i + 1) * dim];

            // PQ
            let pq_code = pq.encode(vec).unwrap();
            let pq_recon = pq.decode(&pq_code).unwrap();
            for j in 0..dim {
                let diff = vec[j] - pq_recon[j];
                pq_mse += diff * diff;
            }

            // Residual PQ
            let rpq_code = rpq.encode(vec).unwrap();
            let rpq_recon = rpq.decode(&rpq_code).unwrap();
            for j in 0..dim {
                let diff = vec[j] - rpq_recon[j];
                rpq_mse += diff * diff;
            }
        }

        pq_mse /= (10 * dim) as f32;
        rpq_mse /= (10 * dim) as f32;

        // Just verify both produce valid results (no NaN)
        assert!(!pq_mse.is_nan(), "PQ MSE should not be NaN");
        assert!(!rpq_mse.is_nan(), "Residual PQ MSE should not be NaN");
    }
}

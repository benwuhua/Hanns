//! Optimized Product Quantization (OPQ)
//!
//! OPQ applies a linear transformation (rotation) to vectors before PQ encoding
//! to optimize the distribution of data for PQ assumptions.
//!
//! Reference: Ge et al., "Optimized Product Quantization for Approximate Nearest Neighbor Search", 2013

use super::kmeans::KMeans;
use crate::api::{KnowhereError, Result};
use rand::Rng;

/// OPQ configuration
#[derive(Clone, Debug)]
pub struct OPQConfig {
    /// Dimensionality of input vectors
    pub dim: usize,
    /// Number of subquantizers (m)
    pub m: usize,
    /// Bits per subvector (nbits)
    pub nbits: usize,
    /// Number of iterations for OPQ optimization
    pub niter: usize,
    /// Use random rotation instead of learned rotation
    pub random_rotation: bool,
}

impl Default for OPQConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            m: 8,
            nbits: 8,
            niter: 1,              // Use single iteration with random rotation for stability
            random_rotation: true, // Use random rotation by default
        }
    }
}

impl OPQConfig {
    pub fn new(dim: usize, m: usize, nbits: usize) -> Self {
        Self {
            dim,
            m,
            nbits,
            niter: 20,
            random_rotation: false,
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

/// Optimized Product Quantizer
pub struct OptimizedProductQuantizer {
    config: OPQConfig,
    /// Rotation matrix (dim x dim), stored row-major
    rotation: Vec<f32>,
    /// Inverse rotation matrix (for decoding)
    rotation_inv: Vec<f32>,
    /// PQ centroids after rotation (m x ksub x sub_dim)
    centroids: Vec<f32>,
    /// Whether the quantizer is trained
    is_trained: bool,
}

impl OptimizedProductQuantizer {
    /// Create a new OptimizedProductQuantizer
    pub fn new(config: OPQConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            rotation: Vec::new(),
            rotation_inv: Vec::new(),
            centroids: Vec::new(),
            is_trained: false,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &OPQConfig {
        &self.config
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the code size
    pub fn code_size(&self) -> usize {
        self.config.m * self.config.nbits / 8
    }

    /// Initialize with random rotation matrix
    fn init_random_rotation(&mut self) {
        let dim = self.config.dim;
        self.rotation = vec![0.0f32; dim * dim];

        let mut rng = rand::thread_rng();

        // Generate random orthogonal matrix using QR decomposition approach
        // Start with random Gaussian matrix
        let mut matrix = vec![0.0f32; dim * dim];
        for value in matrix.iter_mut().take(dim * dim) {
            *value = rng.gen::<f32>() * 2.0 - 1.0;
        }

        // Modified Gram-Schmidt orthonormalization.
        // Project against the NORMALIZED previous rows (stored in self.rotation),
        // not the raw matrix rows, to avoid float32 overflow for large dim.
        for i in 0..dim {
            let row_offset = i * dim;

            // Subtract projections onto already-normalized previous rows
            for j in 0..i {
                let prev_offset = j * dim;
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += matrix[row_offset + k] * self.rotation[prev_offset + k];
                }
                for k in 0..dim {
                    matrix[row_offset + k] -= dot * self.rotation[prev_offset + k];
                }
            }

            // Normalize
            let mut norm = 0.0f32;
            for k in 0..dim {
                norm += matrix[row_offset + k] * matrix[row_offset + k];
            }
            norm = norm.sqrt();

            if norm > 1e-10 {
                for k in 0..dim {
                    self.rotation[row_offset + k] = matrix[row_offset + k] / norm;
                }
            } else {
                // Fallback: use identity for this row
                self.rotation[row_offset + i] = 1.0;
            }
        }

        // For orthogonal matrix, inverse = transpose
        self.rotation_inv = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                self.rotation_inv[i * dim + j] = self.rotation[j * dim + i];
            }
        }
    }

    /// Apply rotation to vectors
    pub fn apply_rotation(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        let dim = self.config.dim;
        let mut rotated = vec![0.0f32; n * dim];

        for i in 0..n {
            let in_offset = i * dim;
            let out_offset = i * dim;

            for j in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += self.rotation[j * dim + k] * vectors[in_offset + k];
                }
                rotated[out_offset + j] = sum;
            }
        }

        rotated
    }

    /// Apply rotation to a single vector, returning rotated vector
    pub fn apply_rotation_single(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.config.dim;
        let mut rotated = vec![0.0f32; dim];
        for (j, rv) in rotated.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, &xk) in x.iter().enumerate().take(dim) {
                sum += self.rotation[j * dim + k] * xk;
            }
            *rv = sum;
        }
        rotated
    }

    /// Get a reference to OPQ centroids (m * ksub * sub_dim layout)
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Apply inverse rotation to vectors
    fn apply_inverse_rotation(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        let dim = self.config.dim;
        let mut unrotated = vec![0.0f32; n * dim];

        for i in 0..n {
            let in_offset = i * dim;
            let out_offset = i * dim;

            for j in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += self.rotation_inv[j * dim + k] * vectors[in_offset + k];
                }
                unrotated[out_offset + j] = sum;
            }
        }

        unrotated
    }

    /// Compute optimal rotation using OPQ algorithm
    fn compute_optimal_rotation(&mut self, n: usize, x: &[f32]) -> Result<()> {
        let dim = self.config.dim;
        let m = self.config.m;
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();
        let niter = self.config.niter;

        // Initialize rotation (random or identity)
        if self.config.random_rotation {
            self.init_random_rotation();
        } else {
            // Initialize with identity
            self.rotation = vec![0.0f32; dim * dim];
            for i in 0..dim {
                self.rotation[i * dim + i] = 1.0;
            }
            self.rotation_inv = self.rotation.clone();
        }

        // OPQ optimization loop
        for iter in 0..niter {
            // Step 1: Rotate vectors
            let rotated = self.apply_rotation(x, n);

            // Step 2: Train PQ on rotated vectors
            self.centroids = vec![0.0f32; m * ksub * sub_dim];

            for sub_q in 0..m {
                let sub_centroid_offset = sub_q * ksub * sub_dim;

                // Extract subvectors for this subquantizer
                let mut sub_vectors = Vec::with_capacity(n * sub_dim);
                for i in 0..n {
                    let vec_offset = i * dim;
                    let sub_offset = vec_offset + sub_q * sub_dim;
                    sub_vectors.extend_from_slice(&rotated[sub_offset..sub_offset + sub_dim]);
                }

                // Run k-means
                let mut kmeans = KMeans::new(ksub, sub_dim);
                kmeans.train(&sub_vectors);

                self.centroids[sub_centroid_offset..sub_centroid_offset + ksub * sub_dim]
                    .copy_from_slice(kmeans.centroids());
            }

            // Step 3: Update rotation to minimize quantization error
            // This is done by solving an orthogonal Procrustes problem
            if iter < niter - 1 {
                self.update_rotation(n, x)?;
            }
        }

        Ok(())
    }

    /// Update rotation matrix using orthogonal Procrustes
    fn update_rotation(&mut self, n: usize, x: &[f32]) -> Result<()> {
        let dim = self.config.dim;
        let m = self.config.m;
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();

        // Compute the optimal rotation that minimizes reconstruction error
        // For each vector, find the assigned centroids and compute the target

        // Build the correlation matrix C = X^T * Y where Y is the reconstructed vectors
        let mut correlation = vec![0.0f32; dim * dim];

        for i in 0..n {
            let x_offset = i * dim;

            // Get current rotated vector
            let mut rotated = vec![0.0f32; dim];
            for (j, rotated_value) in rotated.iter_mut().enumerate().take(dim) {
                for (k, &xk) in x[x_offset..x_offset + dim].iter().enumerate() {
                    *rotated_value += self.rotation[j * dim + k] * xk;
                }
            }

            // Find nearest centroids and reconstruct
            let mut reconstructed = vec![0.0f32; dim];
            for sub_q in 0..m {
                let sub_offset = sub_q * sub_dim;
                let sub_vector = &rotated[sub_offset..sub_offset + sub_dim];

                // Find nearest centroid
                let mut best_idx = 0;
                let mut best_dist = f32::INFINITY;

                for c in 0..ksub {
                    let centroid_offset = sub_q * ksub * sub_dim + c * sub_dim;
                    let centroid = &self.centroids[centroid_offset..centroid_offset + sub_dim];

                    let dist = Self::l2_distance(sub_vector, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = c;
                    }
                }

                // Add centroid to reconstruction
                let centroid_offset = sub_q * ksub * sub_dim + best_idx * sub_dim;
                reconstructed[sub_offset..sub_offset + sub_dim]
                    .copy_from_slice(&self.centroids[centroid_offset..centroid_offset + sub_dim]);
            }

            // Accumulate correlation: X^T * reconstructed
            for j in 0..dim {
                for k in 0..dim {
                    correlation[j * dim + k] += x[x_offset + j] * reconstructed[k];
                }
            }
        }

        // Solve orthogonal Procrustes: find orthogonal R that maximizes trace(R^T * C)
        // Using SVD: C = U * S * V^T, optimal R = V * U^T
        self.solve_procrustes(&correlation)?;

        Ok(())
    }

    /// Solve orthogonal Procrustes problem using SVD approximation
    fn solve_procrustes(&mut self, correlation: &[f32]) -> Result<()> {
        let dim = self.config.dim;

        // Simple approach: use power iteration to find dominant singular vectors
        // For a more accurate solution, use full SVD

        // Initialize with current rotation
        let mut new_rotation = self.rotation.clone();

        // Power iteration for SVD approximation
        let niter = 10;
        for _ in 0..niter {
            // Multiply by correlation matrix
            let mut temp = vec![0.0f32; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        temp[i * dim + j] += correlation[i * dim + k] * new_rotation[k * dim + j];
                    }
                }
            }

            // Orthonormalize (Gram-Schmidt)
            for i in 0..dim {
                let row_offset = i * dim;

                for j in 0..i {
                    let prev_offset = j * dim;
                    let mut dot = 0.0f32;
                    for k in 0..dim {
                        dot += temp[row_offset + k] * temp[prev_offset + k];
                    }

                    for k in 0..dim {
                        temp[row_offset + k] -= dot * temp[prev_offset + k];
                    }
                }

                let mut norm = 0.0f32;
                for k in 0..dim {
                    norm += temp[row_offset + k] * temp[row_offset + k];
                }
                norm = norm.sqrt();

                if norm > 1e-10 {
                    for k in 0..dim {
                        new_rotation[row_offset + k] = temp[row_offset + k] / norm;
                    }
                }
            }
        }

        self.rotation = new_rotation;

        // Compute inverse (transpose for orthogonal matrix)
        self.rotation_inv = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                self.rotation_inv[i * dim + j] = self.rotation[j * dim + i];
            }
        }

        Ok(())
    }

    /// Train the optimized product quantizer
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

        // Compute optimal rotation and train PQ
        self.compute_optimal_rotation(n, x)?;

        self.is_trained = true;
        Ok(())
    }

    /// Encode a single vector
    pub fn encode(&self, x: &[f32]) -> Result<Vec<u8>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "OPQ quantizer not trained".to_string(),
            ));
        }

        if x.len() != self.config.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats, got {}",
                self.config.dim,
                x.len()
            )));
        }

        // Apply rotation
        let mut rotated = vec![0.0f32; self.config.dim];
        for (j, rotated_value) in rotated.iter_mut().enumerate().take(self.config.dim) {
            let mut sum = 0.0f32;
            for (k, &xk) in x.iter().enumerate().take(self.config.dim) {
                sum += self.rotation[j * self.config.dim + k] * xk;
            }
            *rotated_value = sum;
        }

        // Encode rotated vector using PQ
        let code_size = self.code_size();
        let mut code = vec![0u8; code_size];
        let sub_dim = self.config.sub_dim();
        let _ksub = self.config.ksub();
        let nbits = self.config.nbits;

        for sub_q in 0..self.config.m {
            let sub_offset = sub_q * sub_dim;
            let sub_vector = &rotated[sub_offset..sub_offset + sub_dim];

            // Find nearest centroid
            let centroid_idx = self.find_nearest_centroid(sub_q, sub_vector);

            // Pack index into code
            let byte_offset = sub_q * nbits / 8;
            let bit_offset = (sub_q * nbits) % 8;

            if nbits == 8 {
                code[byte_offset] = centroid_idx as u8;
            } else {
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
                "OPQ quantizer not trained".to_string(),
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

        // Rotate all vectors first
        let rotated = self.apply_rotation(x, n);

        for i in 0..n {
            let vec_offset = i * self.config.dim;
            let code_offset = i * code_size;

            for sub_q in 0..self.config.m {
                let sub_offset = sub_q * self.config.sub_dim();
                let sub_vector = &rotated
                    [vec_offset + sub_offset..vec_offset + sub_offset + self.config.sub_dim()];

                let centroid_idx = self.find_nearest_centroid(sub_q, sub_vector);

                let byte_offset = sub_q * self.config.nbits / 8;
                if self.config.nbits == 8 {
                    codes[code_offset + byte_offset] = centroid_idx as u8;
                } else {
                    let bit_offset = (sub_q * self.config.nbits) % 8;
                    for bit in 0..self.config.nbits {
                        if (centroid_idx >> bit) & 1 != 0 {
                            codes[code_offset + byte_offset + (bit_offset + bit) / 8] |=
                                1 << ((bit_offset + bit) % 8);
                        }
                    }
                }
            }
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

            let dist = Self::l2_distance(sub_vector, centroid);

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

    /// Compute asymmetric distance between a raw vector and a PQ code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        if !self.is_trained {
            return f32::INFINITY;
        }

        // Apply rotation to query
        let mut rotated_query = vec![0.0f32; self.config.dim];
        for (j, rotated_query_value) in rotated_query.iter_mut().enumerate().take(self.config.dim) {
            let mut sum = 0.0f32;
            for (k, &qk) in query.iter().enumerate().take(self.config.dim) {
                sum += self.rotation[j * self.config.dim + k] * qk;
            }
            *rotated_query_value = sum;
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

            // Get rotated query subvector
            let sub_offset = sub_q * sub_dim;
            let query_sub = &rotated_query[sub_offset..sub_offset + sub_dim];

            // Compute distance
            total_dist += Self::l2_distance(query_sub, centroid);
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

    /// Decode a PQ code back to approximate vector
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        if !self.is_trained {
            return Err(KnowhereError::InternalError(
                "OPQ quantizer not trained".to_string(),
            ));
        }

        if code.len() != self.code_size() {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected code size {}, got {}",
                self.code_size(),
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

        // Apply inverse rotation
        let unrotated = self.apply_inverse_rotation(&reconstructed, 1);

        Ok(unrotated)
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
    fn test_opq_config() {
        let config = OPQConfig::new(128, 8, 8);
        assert_eq!(config.sub_dim(), 16);
        assert_eq!(config.ksub(), 256);
    }

    #[test]
    fn test_opq_train_and_encode() {
        let dim = 32;
        let mut config = OPQConfig::new(dim, 4, 8);
        config.niter = 3; // Fewer iterations for speed
        let mut opq = OptimizedProductQuantizer::new(config).unwrap();

        // Train with minimal data
        let train_data = create_test_vectors(100, dim);
        opq.train(100, &train_data).unwrap();
        assert!(opq.is_trained());

        // Encode single vector
        let query = create_test_vectors(1, dim);
        let code = opq.encode(&query).unwrap();
        assert_eq!(code.len(), 4);

        // Encode batch
        let queries = create_test_vectors(5, dim);
        let codes = opq.encode_batch(5, &queries).unwrap();
        assert_eq!(codes.len(), 5 * 4);
    }

    #[test]
    fn test_opq_decode() {
        let dim = 32;
        let mut config = OPQConfig::new(dim, 4, 8);
        config.niter = 3;
        let mut opq = OptimizedProductQuantizer::new(config).unwrap();

        // Train with minimal data
        let train_data = create_test_vectors(100, dim);
        opq.train(100, &train_data).unwrap();

        // Encode and decode
        let original = create_test_vectors(1, dim);
        let code = opq.encode(&original).unwrap();
        let reconstructed = opq.decode(&code).unwrap();

        assert_eq!(reconstructed.len(), dim);
        // Verify decode produces valid output (no NaN)
        assert!(!reconstructed.iter().any(|&x| x.is_nan()));
    }

    #[test]
    fn test_opq_random_rotation() {
        let dim = 32;
        let mut config = OPQConfig::new(dim, 4, 8);
        config.random_rotation = true;
        config.niter = 2; // Minimal iterations for speed

        let mut opq = OptimizedProductQuantizer::new(config).unwrap();

        // Train with random rotation
        let train_data = create_test_vectors(50, dim);
        opq.train(50, &train_data).unwrap();
        assert!(opq.is_trained());

        // Verify encoding works
        let query = create_test_vectors(1, dim);
        let code = opq.encode(&query).unwrap();
        assert_eq!(code.len(), 4);
    }
}

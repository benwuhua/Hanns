//! Product Quantization (PQ)
//!
//! Product Quantization splits vectors into subvectors and quantizes each independently.
//! This provides compression and faster distance computation.
//!
//! Reference: H. Jégou et al., "Product quantization for nearest neighbor search", 2011

use super::kmeans::KMeans;
use crate::api::{KnowhereError, Result};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_add_ps, _mm256_cvtepu8_epi32,
    _mm256_i32gather_ps, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm_loadl_epi64,
};

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
        (self.m * self.nbits).div_ceil(8)
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

    pub fn m(&self) -> usize {
        self.config.m
    }

    pub fn ksub(&self) -> usize {
        self.config.ksub()
    }

    pub fn sub_dim(&self) -> usize {
        self.config.sub_dim()
    }

    pub fn dim(&self) -> usize {
        self.config.dim
    }

    pub fn nbits(&self) -> usize {
        self.config.nbits
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get the code size
    pub fn code_size(&self) -> usize {
        self.config.code_size()
    }

    /// Return the full centroid table in `[m][ksub][sub_dim]` flattened order.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Restore a trained centroid table without re-running k-means.
    pub fn set_centroids(&mut self, centroids: Vec<f32>) -> Result<()> {
        self.config.validate()?;
        let expected = self.config.m * self.config.ksub() * self.config.sub_dim();
        if centroids.len() != expected {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} PQ centroids, got {}",
                expected,
                centroids.len()
            )));
        }
        self.centroids = centroids;
        self.is_trained = true;
        Ok(())
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
        let dim = self.config.dim;
        let m = self.config.m;

        // Initialize centroids (m x ksub x sub_dim)
        let total_centroids = m * ksub * sub_dim;
        self.centroids = vec![0.0f32; total_centroids];

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

        let trained: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub_q| {
                let mut sub_vectors = Vec::with_capacity(n * sub_dim);
                for i in 0..n {
                    let offset = i * dim + sub_q * sub_dim;
                    sub_vectors.extend_from_slice(&x[offset..offset + sub_dim]);
                }

                let mut kmeans = KMeans::new(ksub, sub_dim);
                kmeans.set_max_iter(max_iter);
                kmeans.train(&sub_vectors);
                kmeans.centroids().to_vec()
            })
            .collect();

        for (sub_q, codebook) in trained.into_iter().enumerate() {
            let offset = sub_q * ksub * sub_dim;
            self.centroids[offset..offset + ksub * sub_dim].copy_from_slice(&codebook);
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
        let dim = self.config.dim;
        let codes: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec_offset = i * dim;
                self.encode(&x[vec_offset..vec_offset + dim])
                    .expect("validated PQ encode_batch input")
            })
            .collect();

        let mut flat = Vec::with_capacity(n * code_size);
        for code in codes {
            flat.extend_from_slice(&code);
        }

        Ok(flat)
    }

    /// Build L2 distance table for ADC (Asymmetric Distance Computation)
    /// Returns flat table [m * ksub] where table[sub_q * ksub + c] = L2(query_sub, centroid_c)
    pub fn build_distance_table_l2(&self, query: &[f32]) -> Vec<f32> {
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();
        let mut table = vec![0.0f32; self.config.m * ksub];
        for sub_q in 0..self.config.m {
            let query_sub = &query[sub_q * sub_dim..(sub_q + 1) * sub_dim];
            for c in 0..ksub {
                let offset = sub_q * ksub * sub_dim + c * sub_dim;
                let centroid = &self.centroids[offset..offset + sub_dim];
                let mut dist = 0.0f32;
                for j in 0..sub_dim {
                    let d = query_sub[j] - centroid[j];
                    dist += d * d;
                }
                table[sub_q * ksub + c] = dist;
            }
        }
        table
    }

    /// Build IP distance table for ADC
    /// Returns flat table [m * ksub] where table[sub_q * ksub + c] = dot(query_sub, centroid_c)
    pub fn build_distance_table_ip(&self, query: &[f32]) -> Vec<f32> {
        let sub_dim = self.config.sub_dim();
        let ksub = self.config.ksub();
        let mut table = vec![0.0f32; self.config.m * ksub];
        for sub_q in 0..self.config.m {
            let query_sub = &query[sub_q * sub_dim..(sub_q + 1) * sub_dim];
            for c in 0..ksub {
                let offset = sub_q * ksub * sub_dim + c * sub_dim;
                let centroid = &self.centroids[offset..offset + sub_dim];
                let mut ip = 0.0f32;
                for j in 0..sub_dim {
                    ip += query_sub[j] * centroid[j];
                }
                table[sub_q * ksub + c] = ip;
            }
        }
        table
    }

    /// Score a PQ code using a precomputed distance table. O(m) per code.
    /// For L2: lower is better. For IP: higher is better.
    pub fn compute_distance_with_table(&self, table: &[f32], code: &[u8]) -> f32 {
        let ksub = self.config.ksub();
        let m = self.config.m;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                table.as_ptr() as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }

        let mut sum = 0.0f32;
        let chunks8 = (m / 8) * 8;

        if self.config.nbits == 8 {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    unsafe {
                        let simd_sum = self.compute_distance_with_table_avx2_gather(
                            table.as_ptr(),
                            code.as_ptr(),
                            ksub,
                            chunks8,
                        );
                        sum += simd_sum;
                    }

                    let mut sub_q = chunks8;
                    while sub_q < m {
                        sum += table[sub_q * ksub + code[sub_q] as usize];
                        sub_q += 1;
                    }
                    return sum;
                }
            }

            let mut sub_q = 0usize;
            while sub_q < chunks8 {
                sum += table[sub_q * ksub + code[sub_q] as usize];
                sum += table[(sub_q + 1) * ksub + code[sub_q + 1] as usize];
                sum += table[(sub_q + 2) * ksub + code[sub_q + 2] as usize];
                sum += table[(sub_q + 3) * ksub + code[sub_q + 3] as usize];
                sum += table[(sub_q + 4) * ksub + code[sub_q + 4] as usize];
                sum += table[(sub_q + 5) * ksub + code[sub_q + 5] as usize];
                sum += table[(sub_q + 6) * ksub + code[sub_q + 6] as usize];
                sum += table[(sub_q + 7) * ksub + code[sub_q + 7] as usize];
                sub_q += 8;
            }
            while sub_q < m {
                sum += table[sub_q * ksub + code[sub_q] as usize];
                sub_q += 1;
            }
            return sum;
        }

        let mut sub_q = 0usize;
        while sub_q < chunks8 {
            sum += table[sub_q * ksub + self.extract_index(code, sub_q)];
            sum += table[(sub_q + 1) * ksub + self.extract_index(code, sub_q + 1)];
            sum += table[(sub_q + 2) * ksub + self.extract_index(code, sub_q + 2)];
            sum += table[(sub_q + 3) * ksub + self.extract_index(code, sub_q + 3)];
            sum += table[(sub_q + 4) * ksub + self.extract_index(code, sub_q + 4)];
            sum += table[(sub_q + 5) * ksub + self.extract_index(code, sub_q + 5)];
            sum += table[(sub_q + 6) * ksub + self.extract_index(code, sub_q + 6)];
            sum += table[(sub_q + 7) * ksub + self.extract_index(code, sub_q + 7)];
            sub_q += 8;
        }
        while sub_q < m {
            sum += table[sub_q * ksub + self.extract_index(code, sub_q)];
            sub_q += 1;
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_distance_with_table_avx2_gather(
        &self,
        table_ptr: *const f32,
        code_ptr: *const u8,
        ksub: usize,
        chunks8: usize,
    ) -> f32 {
        let ksub_i32 = ksub as i32;
        let mut offsets = _mm256_setr_epi32(
            0,
            ksub_i32,
            2 * ksub_i32,
            3 * ksub_i32,
            4 * ksub_i32,
            5 * ksub_i32,
            6 * ksub_i32,
            7 * ksub_i32,
        );
        let step = _mm256_set1_epi32(8 * ksub_i32);
        let mut acc = _mm256_setzero_ps();

        let mut sub_q = 0usize;
        while sub_q < chunks8 {
            let idx_bytes = _mm_loadl_epi64(code_ptr.add(sub_q) as *const __m128i);
            let idx = _mm256_cvtepu8_epi32(idx_bytes);
            let gather_idx: __m256i = _mm256_add_epi32(idx, offsets);
            let vals: __m256 = _mm256_i32gather_ps(table_ptr, gather_idx, 4);
            acc = _mm256_add_ps(acc, vals);
            offsets = _mm256_add_epi32(offsets, step);
            sub_q += 8;
        }

        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        lanes.into_iter().sum()
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


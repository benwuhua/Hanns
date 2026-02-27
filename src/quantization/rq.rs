//! Residual Quantizer (RQ) Implementation
//! 
//! Residual quantization with variable number of bits per sub-quantizer.
//! The residual centroids are stored in a big cumulative centroid table.

use crate::api::{KnowhereError, Result};
use std::collections::HashMap;

/// Residual Quantizer configuration
#[derive(Clone, Debug)]
pub struct RQConfig {
    /// Dimensionality of input vectors
    pub d: usize,
    /// Number of subquantizers
    pub m: usize,
    /// Number of bits per subvector index
    pub nbits: usize,
    /// Beam size for encoding
    pub max_beam_size: usize,
}

impl Default for RQConfig {
    fn default() -> Self {
        Self {
            d: 0,
            m: 8,
            nbits: 8,
            max_beam_size: 5,
        }
    }
}

/// Residual Quantizer
pub struct ResidualQuantizer {
    config: RQConfig,
    /// Codebooks: flattened array of size (m * (1 << nbits) * sub_dim)
    pub codebooks: Vec<f32>,
    /// Whether the quantizer is trained
    pub is_trained: bool,
    /// Dimension of each subvector
    sub_dim: usize,
    /// Codebook offsets for each subquantizer
    codebook_offsets: Vec<usize>,
    /// Total codebook size
    total_codebook_size: usize,
}

impl ResidualQuantizer {
    /// Create a new ResidualQuantizer
    pub fn new(config: RQConfig) -> Result<Self> {
        if config.d == 0 {
            return Err(KnowhereError::InvalidArg("dimension must be > 0".to_string()));
        }
        if config.m == 0 {
            return Err(KnowhereError::InvalidArg("m must be > 0".to_string()));
        }
        if config.nbits == 0 || config.nbits > 16 {
            return Err(KnowhereError::InvalidArg("nbits must be in (0, 16]".to_string()));
        }

        let sub_dim = config.d / config.m;
        if sub_dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be >= m".to_string(),
            ));
        }

        let codebooks_per_sub = 1 << config.nbits;
        let total_codebook_size = config.m * codebooks_per_sub;

        // Compute codebook offsets
        let mut codebook_offsets = Vec::with_capacity(config.m);
        for i in 0..config.m {
            codebook_offsets.push(i * codebooks_per_sub);
        }

        Ok(Self {
            config,
            codebooks: Vec::new(),
            is_trained: false,
            sub_dim,
            codebook_offsets,
            total_codebook_size,
        })
    }

    /// Get the code size in bytes
    pub fn code_size(&self) -> usize {
        // Each subvector uses nbits bits, packed into bytes
        let total_bits = self.config.m * self.config.nbits;
        (total_bits + 7) / 8
    }

    /// Train the residual quantizer
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.config.d;
        if n == 0 {
            return Err(KnowhereError::InvalidArg("no training vectors".to_string()));
        }

        let codebooks_per_sub = 1 << self.config.nbits;
        
        // Initialize codebooks using k-means on each subspace
        self.codebooks = vec![0.0f32; self.total_codebook_size * self.sub_dim];

        // Train each subquantizer independently
        for sub_idx in 0..self.config.m {
            let sub_codebook_offset = sub_idx * codebooks_per_sub * self.sub_dim;
            
            // Extract subvectors for this subspace
            let mut sub_vectors = Vec::with_capacity(n * self.sub_dim);
            for i in 0..n {
                let vec_start = i * self.config.d + sub_idx * self.sub_dim;
                sub_vectors.extend_from_slice(
                    &vectors[vec_start..vec_start + self.sub_dim],
                );
            }

            // Perform k-means clustering for this subspace
            let codebook_slice = codebooks_per_sub * self.sub_dim;
            ResidualQuantizer::kmeans_train_static(
                &sub_vectors,
                n,
                codebooks_per_sub,
                self.sub_dim,
                &mut self.codebooks[sub_codebook_offset..sub_codebook_offset + codebook_slice],
            )?;
        }

        self.is_trained = true;
        Ok(())
    }

    /// K-means training for a single subspace (static version to avoid borrow issues)
    fn kmeans_train_static(
        vectors: &[f32],
        n: usize,
        k: usize,
        dim: usize,
        centroids: &mut [f32],
    ) -> Result<()> {
        if n < k {
            return Err(KnowhereError::InvalidArg(
                "not enough vectors for k-means".to_string(),
            ));
        }
        
        // Initialize centroids with random vectors
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 0..k {
            let idx = rng.gen_range(0..n);
            centroids[i * dim..(i + 1) * dim]
                .copy_from_slice(&vectors[idx * dim..(idx + 1) * dim]);
        }

        // K-means iterations
        let max_iters = 20;
        let mut assignments = vec![0usize; n];

        for _iter in 0..max_iters {
            // Assignment step
            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let mut best_idx = 0;
                let mut best_dist = f32::MAX;

                for j in 0..k {
                    let centroid = &centroids[j * dim..(j + 1) * dim];
                    let dist = ResidualQuantizer::l2_distance_static(vec, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = j;
                    }
                }
                assignments[i] = best_idx;
            }

            // Update step
            let mut new_centroids = vec![0.0f32; k * dim];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for d in 0..dim {
                    new_centroids[cluster * dim + d] += vectors[i * dim + d];
                }
            }

            // Average
            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        new_centroids[j * dim + d] /= counts[j] as f32;
                    }
                }
            }

            centroids.copy_from_slice(&new_centroids);
        }

        Ok(())
    }

    /// Compute L2 distance between two vectors (static version)
    #[inline]
    fn l2_distance_static(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    }

    /// Compute L2 distance between two vectors
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        Self::l2_distance_static(a, b)
    }

    /// Encode a single vector using beam search
    pub fn encode(&self, vector: &[f32], codes: &mut [u8]) -> Result<()> {
        if !self.is_trained {
            return Err(KnowhereError::InvalidArg(
                "quantizer must be trained first".to_string(),
            ));
        }

        let codebooks_per_sub = 1 << self.config.nbits;
        let mut residual = vector.to_vec();
        
        // Store codes as unpacked indices first
        let mut indices = vec![0u32; self.config.m];

        // Beam search: maintain a set of candidate code sequences
        // For simplicity, use greedy encoding (beam_size = 1)
        for sub_idx in 0..self.config.m {
            let sub_vec = &residual[sub_idx * self.sub_dim..(sub_idx + 1) * self.sub_dim];
            
            // Find the best codebook entry
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for j in 0..codebooks_per_sub {
                let codebook_idx = self.codebook_offsets[sub_idx] + j;
                let centroid = &self.codebooks
                    [codebook_idx * self.sub_dim..(codebook_idx + 1) * self.sub_dim];
                
                let dist = self.l2_distance(sub_vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }

            indices[sub_idx] = best_idx as u32;

            // Update residual
            let codebook_idx = self.codebook_offsets[sub_idx] + best_idx;
            let centroid = &self.codebooks
                [codebook_idx * self.sub_dim..(codebook_idx + 1) * self.sub_dim];
            
            for d in 0..self.sub_dim {
                residual[sub_idx * self.sub_dim + d] -= centroid[d];
            }
        }

        // Pack indices into bytes
        self.pack_codes(&indices, codes);

        Ok(())
    }

    /// Encode multiple vectors
    pub fn encode_batch(&self, vectors: &[f32], codes: &mut [u8]) -> Result<()> {
        let n = vectors.len() / self.config.d;
        let code_size = self.code_size();
        
        if codes.len() < n * code_size {
            return Err(KnowhereError::InvalidArg(
                "codes buffer too small".to_string(),
            ));
        }

        for i in 0..n {
            let vec = &vectors[i * self.config.d..(i + 1) * self.config.d];
            let code = &mut codes[i * code_size..(i + 1) * code_size];
            self.encode(vec, code)?;
        }

        Ok(())
    }

    /// Pack code indices into bit-compact format
    fn pack_codes(&self, indices: &[u32], codes: &mut [u8]) {
        let code_size = self.code_size();
        codes.fill(0);
        
        let mut bit_offset = 0;
        for &idx in indices {
            let nbits = self.config.nbits;
            for bit in 0..nbits {
                if (idx >> bit) & 1 == 1 {
                    let byte_idx = (bit_offset + bit) / 8;
                    let bit_pos = (bit_offset + bit) % 8;
                    codes[byte_idx] |= 1 << bit_pos;
                }
            }
            bit_offset += nbits;
        }
    }

    /// Decode a single code
    pub fn decode(&self, code: &[u8], output: &mut [f32]) -> Result<()> {
        if !self.is_trained {
            return Err(KnowhereError::InvalidArg(
                "quantizer must be trained first".to_string(),
            ));
        }

        if output.len() < self.config.d {
            return Err(KnowhereError::InvalidArg(
                "output buffer too small".to_string(),
            ));
        }

        // Unpack indices from bit-compact format
        let indices = self.unpack_codes(code);

        // Decode by summing codebook entries
        output.fill(0.0);
        for sub_idx in 0..self.config.m {
            let codebook_idx = self.codebook_offsets[sub_idx] + indices[sub_idx] as usize;
            let centroid = &self.codebooks
                [codebook_idx * self.sub_dim..(codebook_idx + 1) * self.sub_dim];
            
            for d in 0..self.sub_dim {
                output[sub_idx * self.sub_dim + d] += centroid[d];
            }
        }

        Ok(())
    }

    /// Unpack code indices from bit-compact format
    fn unpack_codes(&self, code: &[u8]) -> Vec<u32> {
        let mut indices = vec![0u32; self.config.m];
        let mut bit_offset = 0;
        
        for i in 0..self.config.m {
            let nbits = self.config.nbits;
            let mut idx = 0u32;
            for bit in 0..nbits {
                let byte_idx = (bit_offset + bit) / 8;
                let bit_pos = (bit_offset + bit) % 8;
                if byte_idx < code.len() && (code[byte_idx] >> bit_pos) & 1 == 1 {
                    idx |= 1 << bit;
                }
            }
            indices[i] = idx;
            bit_offset += nbits;
        }

        indices
    }

    /// Compute distance between a query vector and a code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        // Decode the code and compute L2 distance
        let mut reconstructed = vec![0.0f32; self.config.d];
        if let Ok(_) = self.decode(code, &mut reconstructed) {
            self.l2_distance(query, &reconstructed)
        } else {
            f32::MAX
        }
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.config.d
    }

    /// Get the number of subquantizers
    pub fn num_subquantizers(&self) -> usize {
        self.config.m
    }

    /// Get bits per subvector
    pub fn bits_per_subvector(&self) -> usize {
        self.config.nbits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rq_creation() {
        let config = RQConfig {
            d: 64,
            m: 8,
            nbits: 8,
            max_beam_size: 5,
        };
        let rq = ResidualQuantizer::new(config).unwrap();
        assert_eq!(rq.dim(), 64);
        assert_eq!(rq.num_subquantizers(), 8);
        assert!(!rq.is_trained);
    }

    #[test]
    fn test_rq_train_and_encode() {
        let config = RQConfig {
            d: 16,
            m: 4,
            nbits: 8,
            max_beam_size: 5,
        };
        let mut rq = ResidualQuantizer::new(config).unwrap();

        // Generate training data
        let n_train = 100;
        let mut train_data = vec![0.0f32; n_train * 16];
        for i in 0..n_train {
            for j in 0..16 {
                train_data[i * 16 + j] = (i + j) as f32 * 0.01;
            }
        }

        rq.train(&train_data).unwrap();
        assert!(rq.is_trained);

        // Encode a test vector
        let test_vec: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut code = vec![0u8; rq.code_size()];
        rq.encode(&test_vec, &mut code).unwrap();

        // Decode and verify
        let mut decoded = vec![0.0f32; 16];
        rq.decode(&code, &mut decoded).unwrap();

        // The decoded vector should be close to the original (within quantization error)
        let dist: f32 = test_vec.iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(dist < 10.0); // Reasonable quantization error
    }
}

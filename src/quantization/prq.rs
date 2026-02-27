//! Product Residual Quantizer (PRQ) Implementation
//! 
//! Product Additive Quantizer variant that splits the vector space into
//! multiple orthogonal subspaces and quantizes each with an independent
//! Residual Quantizer.

use crate::api::{KnowhereError, Result};
use super::rq::{ResidualQuantizer, RQConfig};

/// Product Residual Quantizer configuration
#[derive(Clone, Debug)]
pub struct PRQConfig {
    /// Dimensionality of input vectors
    pub d: usize,
    /// Number of residual quantizers (splits)
    pub nsplits: usize,
    /// Number of subquantizers per RQ
    pub msub: usize,
    /// Number of bits per subvector index
    pub nbits: usize,
    /// Beam size for encoding
    pub max_beam_size: usize,
}

impl Default for PRQConfig {
    fn default() -> Self {
        Self {
            d: 0,
            nsplits: 2,
            msub: 4,
            nbits: 8,
            max_beam_size: 5,
        }
    }
}

/// Product Residual Quantizer
pub struct ProductResidualQuantizer {
    config: PRQConfig,
    /// Sub-quantizers (one per split)
    pub quantizers: Vec<ResidualQuantizer>,
    /// Whether the quantizer is trained
    pub is_trained: bool,
    /// Dimension of each split
    split_dim: usize,
}

impl ProductResidualQuantizer {
    /// Create a new ProductResidualQuantizer
    pub fn new(config: PRQConfig) -> Result<Self> {
        if config.d == 0 {
            return Err(KnowhereError::InvalidArg("dimension must be > 0".to_string()));
        }
        if config.nsplits == 0 {
            return Err(KnowhereError::InvalidArg("nsplits must be > 0".to_string()));
        }
        if config.msub == 0 {
            return Err(KnowhereError::InvalidArg("msub must be > 0".to_string()));
        }

        let split_dim = config.d / config.nsplits;
        if split_dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be >= nsplits".to_string(),
            ));
        }

        // Create sub-quantizers
        let mut quantizers = Vec::with_capacity(config.nsplits);
        for _ in 0..config.nsplits {
            let rq_config = RQConfig {
                d: split_dim,
                m: config.msub,
                nbits: config.nbits,
                max_beam_size: config.max_beam_size,
            };
            quantizers.push(ResidualQuantizer::new(rq_config)?);
        }

        Ok(Self {
            config,
            quantizers,
            is_trained: false,
            split_dim,
        })
    }

    /// Get the code size in bytes
    pub fn code_size(&self) -> usize {
        // Each RQ produces code_size bytes, total is nsplits * code_size
        self.config.nsplits * self.quantizers[0].code_size()
    }

    /// Train the product residual quantizer
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.config.d;
        if n == 0 {
            return Err(KnowhereError::InvalidArg("no training vectors".to_string()));
        }

        // Train each sub-quantizer on its subspace
        for (split_idx, quantizer) in self.quantizers.iter_mut().enumerate() {
            // Extract subvectors for this split
            let mut sub_vectors = Vec::with_capacity(n * self.split_dim);
            for i in 0..n {
                let vec_start = i * self.config.d + split_idx * self.split_dim;
                sub_vectors.extend_from_slice(
                    &vectors[vec_start..vec_start + self.split_dim],
                );
            }

            // Train this sub-quantizer
            quantizer.train(&sub_vectors)?;
        }

        self.is_trained = true;
        Ok(())
    }

    /// Encode a single vector
    pub fn encode(&self, vector: &[f32], codes: &mut [u8]) -> Result<()> {
        if !self.is_trained {
            return Err(KnowhereError::InvalidArg(
                "quantizer must be trained first".to_string(),
            ));
        }

        if vector.len() != self.config.d {
            return Err(KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        // Encode each subspace independently
        for (split_idx, quantizer) in self.quantizers.iter().enumerate() {
            let sub_vec = &vector[split_idx * self.split_dim..(split_idx + 1) * self.split_dim];
            let code_offset = split_idx * quantizer.code_size();
            let code = &mut codes[code_offset..code_offset + quantizer.code_size()];
            quantizer.encode(sub_vec, code)?;
        }

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

        // Decode each subspace independently
        for (split_idx, quantizer) in self.quantizers.iter().enumerate() {
            let code_offset = split_idx * quantizer.code_size();
            let sub_code = &code[code_offset..code_offset + quantizer.code_size()];
            let sub_output = &mut output[split_idx * self.split_dim..(split_idx + 1) * self.split_dim];
            quantizer.decode(sub_code, sub_output)?;
        }

        Ok(())
    }

    /// Compute distance between a query vector and a code
    pub fn compute_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        // Decode the code and compute L2 distance
        let mut reconstructed = vec![0.0f32; self.config.d];
        if let Ok(_) = self.decode(code, &mut reconstructed) {
            query.iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        } else {
            f32::MAX
        }
    }

    /// Compute distance using lookup tables (faster for search)
    pub fn compute_distance_lut(&self, query: &[f32], code: &[u8]) -> f32 {
        // For PRQ, we can compute distances by summing sub-distances
        let mut total_dist = 0.0f32;

        for (split_idx, quantizer) in self.quantizers.iter().enumerate() {
            let sub_query = &query[split_idx * self.split_dim..(split_idx + 1) * self.split_dim];
            let code_offset = split_idx * quantizer.code_size();
            let sub_code = &code[code_offset..code_offset + quantizer.code_size()];
            
            total_dist += quantizer.compute_distance(sub_query, sub_code);
        }

        total_dist
    }

    /// Get the dimension
    pub fn dim(&self) -> usize {
        self.config.d
    }

    /// Get the number of splits
    pub fn num_splits(&self) -> usize {
        self.config.nsplits
    }

    /// Get subquantizers per split
    pub fn subquantizers_per_split(&self) -> usize {
        self.config.msub
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
    fn test_prq_creation() {
        let config = PRQConfig {
            d: 64,
            nsplits: 2,
            msub: 4,
            nbits: 8,
            max_beam_size: 5,
        };
        let prq = ProductResidualQuantizer::new(config).unwrap();
        assert_eq!(prq.dim(), 64);
        assert_eq!(prq.num_splits(), 2);
        assert!(!prq.is_trained);
    }

    #[test]
    fn test_prq_train_and_encode() {
        let config = PRQConfig {
            d: 32,
            nsplits: 2,
            msub: 4,
            nbits: 6,  // Reduced from 8 to 6 (64 codebooks) to work with fewer training vectors
            max_beam_size: 5,
        };
        let mut prq = ProductResidualQuantizer::new(config).unwrap();

        // Generate training data - need at least as many vectors as codebooks per subquantizer
        let n_train = 512;  // Increased to ensure enough vectors for k-means
        let mut train_data = vec![0.0f32; n_train * 32];
        for i in 0..n_train {
            for j in 0..32 {
                train_data[i * 32 + j] = (i + j) as f32 * 0.01;
            }
        }

        prq.train(&train_data).unwrap();
        assert!(prq.is_trained);

        // Encode a test vector
        let test_vec: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut code = vec![0u8; prq.code_size()];
        prq.encode(&test_vec, &mut code).unwrap();

        // Decode and verify
        let mut decoded = vec![0.0f32; 32];
        prq.decode(&code, &mut decoded).unwrap();

        // The decoded vector should be close to the original
        let dist: f32 = test_vec.iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(dist < 20.0); // Reasonable quantization error
    }
}

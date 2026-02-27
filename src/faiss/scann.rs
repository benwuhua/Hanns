//! SCANN: Scalable Nearest Neighbors
//!
//! Reference: Google Research ScaNN
//! Paper: Accelerating Large-Scale Inference with Anisotropic Vector Quantization
//!
//! SCANN uses anisotropic vector quantization to achieve high recall
//! with improved throughput compared to traditional PQ.

use std::collections::HashMap;
use std::sync::RwLock;
use thiserror::Error;
use serde::{Serialize, Deserialize};

#[derive(Error, Debug)]
pub enum ScannError {
    #[error("Index not trained")]
    NotTrained,
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// SCANN configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScaNNConfig {
    /// Number of partitions (subspaces)
    pub num_partitions: usize,
    /// Number of centroids per partition (must be power of 2 for efficiency)
    pub num_centroids: usize,
    /// Number of candidates to rerank
    pub reorder_k: usize,
    /// Anisotropic weight parameter (higher = more anisotropic)
    pub anisotropic_alpha: f32,
}

impl Default for ScaNNConfig {
    fn default() -> Self {
        Self {
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            anisotropic_alpha: 0.2,
        }
    }
}

impl ScaNNConfig {
    pub fn new(num_partitions: usize, num_centroids: usize, reorder_k: usize) -> Self {
        Self {
            num_partitions,
            num_centroids,
            reorder_k,
            anisotropic_alpha: 0.2,
        }
    }

    /// Validate configuration
    pub fn validate(&self, dim: usize) -> Result<(), ScannError> {
        if self.num_partitions == 0 {
            return Err(ScannError::InvalidConfig(
                "num_partitions must be > 0".to_string(),
            ));
        }
        if dim % self.num_partitions != 0 {
            return Err(ScannError::InvalidConfig(format!(
                "dim {} must be divisible by num_partitions {}",
                dim, self.num_partitions
            )));
        }
        if !self.num_centroids.is_power_of_two() {
            return Err(ScannError::InvalidConfig(
                "num_centroids must be a power of 2".to_string(),
            ));
        }
        Ok(())
    }
}

/// Anisotropic quantizer for SCANN
pub struct AnisotropicQuantizer {
    dim: usize,
    config: ScaNNConfig,
    /// Subspace dimension
    sub_dim: usize,
    /// Codebook: [num_partitions, num_centroids, sub_dim]
    codebook: Vec<f32>,
    /// Anisotropic weights: [num_partitions, sub_dim]
    weights: Vec<f32>,
    /// Centroid norms: [num_partitions, num_centroids]
    centroid_norms: Vec<f32>,
    trained: bool,
}

impl AnisotropicQuantizer {
    pub fn new(dim: usize, config: ScaNNConfig) -> Self {
        let sub_dim = dim / config.num_partitions;
        Self {
            dim,
            config,
            sub_dim,
            codebook: Vec::new(),
            weights: Vec::new(),
            centroid_norms: Vec::new(),
            trained: false,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Train the quantizer using anisotropic k-means
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) {
        let n = data.len() / self.dim;
        if n == 0 {
            return;
        }

        // Compute anisotropic weights
        if let Some(queries) = query_sample {
            self.compute_weights(queries);
        } else {
            // Default isotropic weights
            self.weights = vec![1.0; self.config.num_partitions * self.sub_dim];
        }

        // Allocate codebook
        let codebook_size = self.config.num_partitions * self.config.num_centroids * self.sub_dim;
        self.codebook.resize(codebook_size, 0.0);
        self.centroid_norms
            .resize(self.config.num_partitions * self.config.num_centroids, 0.0);

        // Train each partition
        for p in 0..self.config.num_partitions {
            self.train_partition(p, data, n);
        }

        self.trained = true;
    }

    /// Compute anisotropic weights based on query distribution
    fn compute_weights(&mut self, queries: &[f32]) {
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            self.weights = vec![1.0; self.config.num_partitions * self.sub_dim];
            return;
        }

        let alpha = self.config.anisotropic_alpha;

        // Compute mean query direction
        let mut mean_dir = vec![0.0f32; self.dim];
        for q in queries.chunks(self.dim) {
            for (i, &v) in q.iter().enumerate() {
                mean_dir[i] += v;
            }
        }
        for v in mean_dir.iter_mut() {
            *v /= n_queries as f32;
        }

        // Normalize
        let norm: f32 = mean_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in mean_dir.iter_mut() {
                *v /= norm;
            }
        }

        // Compute weights per subspace
        self.weights.clear();
        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            for d in 0..self.sub_dim {
                let cos_theta = mean_dir[start + d].abs();
                let w = 1.0 / (1.0 + alpha * cos_theta);
                self.weights.push(w);
            }
        }
    }

    /// Train a single partition using weighted k-means
    fn train_partition(&mut self, partition: usize, data: &[f32], n: usize) {
        let k = self.config.num_centroids;
        let sub_dim = self.sub_dim;

        // Extract subspace vectors
        let mut subspace_vectors = vec![0.0f32; n * sub_dim];
        for i in 0..n {
            let src_start = i * self.dim + partition * sub_dim;
            let dst_start = i * sub_dim;
            subspace_vectors[dst_start..dst_start + sub_dim]
                .copy_from_slice(&data[src_start..src_start + sub_dim]);
        }

        // Initialize centroids using k-means++
        let mut centroids = self.kmeans_plusplus(&subspace_vectors, k, sub_dim);

        // Get weights for this partition
        let weight_offset = partition * sub_dim;
        let weights = &self.weights[weight_offset..weight_offset + sub_dim];

        // Iterative k-means
        for _ in 0..50 {
            // Assignment step
            let mut assignments = vec![0usize; n];
            let mut new_centroids = vec![0.0f32; k * sub_dim];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let vec = &subspace_vectors[i * sub_dim..(i + 1) * sub_dim];
                let mut min_dist = f32::MAX;
                let mut best = 0;

                for c in 0..k {
                    let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                    let dist = self.weighted_l2_squared(vec, centroid, weights);

                    if dist < min_dist {
                        min_dist = dist;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // Update step
            for i in 0..n {
                let c = assignments[i];
                let vec = &subspace_vectors[i * sub_dim..(i + 1) * sub_dim];
                for j in 0..sub_dim {
                    new_centroids[c * sub_dim + j] += vec[j];
                }
                counts[c] += 1;
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..sub_dim {
                        centroids[c * sub_dim + j] = new_centroids[c * sub_dim + j] / counts[c] as f32;
                    }
                }
            }
        }

        // Store codebook
        let offset = partition * k * sub_dim;
        self.codebook[offset..offset + k * sub_dim].copy_from_slice(&centroids);

        // Compute centroid norms
        for c in 0..k {
            let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
            let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.centroid_norms[partition * k + c] = norm;
        }
    }

    /// K-means++ initialization
    fn kmeans_plusplus(&self, vectors: &[f32], k: usize, dim: usize) -> Vec<f32> {
        let n = vectors.len() / dim;
        let mut centroids = Vec::with_capacity(k * dim);

        // First centroid: random
        let first_idx = 0;
        centroids.extend_from_slice(&vectors[first_idx * dim..(first_idx + 1) * dim]);

        // Remaining centroids
        let mut distances = vec![0.0f32; n];

        for _ in 1..k {
            // Compute distances to nearest centroid
            let mut max_dist = 0.0f32;
            let mut max_idx = 0;

            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let mut min_dist = f32::MAX;

                for (c_idx, centroid) in centroids.chunks(dim).enumerate() {
                    let dist = self.l2_squared(vec, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist;

                if min_dist > max_dist {
                    max_dist = min_dist;
                    max_idx = i;
                }
            }

            // Add new centroid
            centroids.extend_from_slice(&vectors[max_idx * dim..(max_idx + 1) * dim]);
        }

        centroids
    }

    /// Weighted L2 squared distance
    #[inline]
    fn weighted_l2_squared(&self, a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .zip(weights.iter())
            .map(|((&x, &y), &w)| w * (x - y) * (x - y))
            .sum()
    }

    /// L2 squared distance
    #[inline]
    fn l2_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }

    /// Encode a vector
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert!(self.trained);
        let mut codes = Vec::with_capacity(self.config.num_partitions);

        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vec = &vector[start..end];

            let code = self.find_nearest_centroid(p, sub_vec);
            // num_centroids is power of 2, so we can store in log2(num_centroids) bits
            codes.push(code as u8);
        }

        codes
    }

    /// Find nearest centroid in a partition
    #[inline]
    fn find_nearest_centroid(&self, partition: usize, sub_vec: &[f32]) -> usize {
        let k = self.config.num_centroids;
        let offset = partition * k * self.sub_dim;
        let weights_offset = partition * self.sub_dim;
        let weights = &self.weights[weights_offset..weights_offset + self.sub_dim];

        let mut min_dist = f32::MAX;
        let mut best = 0;

        for c in 0..k {
            let centroid = &self.codebook[offset + c * self.sub_dim..offset + (c + 1) * self.sub_dim];
            let dist = self.weighted_l2_squared(sub_vec, centroid, weights);

            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }

        best
    }

    /// Asymmetric Distance Calculation (ADC)
    /// Compute distance from query to encoded vector
    pub fn adc_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        debug_assert!(self.trained);
        debug_assert_eq!(codes.len(), self.config.num_partitions);

        let mut distance = 0.0f32;

        for p in 0..self.config.num_partitions {
            let start = p * self.sub_dim;
            let sub_query = &query[start..start + self.sub_dim];

            let code = codes[p] as usize;
            let k = self.config.num_centroids;
            let offset = p * k * self.sub_dim + code * self.sub_dim;
            let centroid = &self.codebook[offset..offset + self.sub_dim];

            distance += self.l2_squared(sub_query, centroid);
        }

        distance.sqrt()
    }

    /// Get codebook size in bytes
    pub fn codebook_size(&self) -> usize {
        self.codebook.len() * std::mem::size_of::<f32>()
    }
}

/// SCANN index
pub struct ScaNNIndex {
    dim: usize,
    config: ScaNNConfig,
    quantizer: AnisotropicQuantizer,

    /// Inverted lists: partition -> [(id, codes)]
    inverted_lists: RwLock<HashMap<usize, Vec<(i64, Vec<u8>)>>>,

    /// Original vectors for reranking
    vectors: RwLock<Vec<f32>>,
    ids: RwLock<Vec<i64>>,

    trained: bool,
}

impl ScaNNIndex {
    /// Create a new SCANN index
    pub fn new(dim: usize, config: ScaNNConfig) -> Result<ScaNNIndex, ScannError> {
        config.validate(dim)?;
        Ok(Self {
            dim,
            config: config.clone(),
            quantizer: AnisotropicQuantizer::new(dim, config),
            inverted_lists: RwLock::new(HashMap::new()),
            vectors: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            trained: false,
        })
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of vectors
    pub fn count(&self) -> usize {
        self.ids.read().unwrap().len()
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Train the index
    pub fn train(&mut self, data: &[f32], query_sample: Option<&[f32]>) {
        if data.len() / self.dim == 0 {
            return;
        }

        self.quantizer.train(data, query_sample);
        self.trained = true;
    }

    /// Add vectors to the index
    pub fn add(&self, vectors: &[f32], ids: Option<&[i64]>) -> usize {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return 0;
        }

        let mut inverted_lists = self.inverted_lists.write().unwrap();
        let mut vecs = self.vectors.write().unwrap();
        let mut id_list = self.ids.write().unwrap();

        for i in 0..n {
            let vector = &vectors[i * self.dim..(i + 1) * self.dim];
            let codes = self.quantizer.encode(vector);

            // Assign to first partition (simplified - could use better partitioning)
            let partition = codes[0] as usize % self.config.num_partitions;

            let id = ids.map(|ids| ids[i]).unwrap_or(id_list.len() as i64);

            inverted_lists
                .entry(partition)
                .or_insert_with(Vec::new)
                .push((id, codes));

            vecs.extend_from_slice(vector);
            id_list.push(id);
        }

        n
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if !self.trained {
            return Vec::new();
        }

        // Phase 1: Coarse search using ADC
        let candidates = self.coarse_search(query, self.config.reorder_k);

        // Phase 2: Rerank using original vectors
        let reranked = self.rerank(query, candidates, k);

        reranked
    }

    /// Coarse search using ADC
    fn coarse_search(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let inverted_lists = self.inverted_lists.read().unwrap();
        let mut candidates = Vec::new();

        for (_, list) in inverted_lists.iter() {
            for &(id, ref codes) in list {
                let dist = self.quantizer.adc_distance(query, codes);
                candidates.push((id, dist));
            }
        }

        // Sort by distance and take top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);

        candidates
    }

    /// Rerank candidates using original vectors
    fn rerank(&self, query: &[f32], candidates: Vec<(i64, f32)>, k: usize) -> Vec<(i64, f32)> {
        let vecs = self.vectors.read().unwrap();
        let id_list = self.ids.read().unwrap();

        let mut reranked: Vec<(i64, f32)> = candidates
            .iter()
            .filter_map(|&(id, _)| {
                let pos = id_list.iter().position(|&x| x == id)?;
                let vector = &vecs[pos * self.dim..(pos + 1) * self.dim];
                let dist = self.l2_distance(query, vector);
                Some((id, dist))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked.truncate(k);

        reranked
    }

    /// L2 distance
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Get vectors by their IDs
    pub fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>, ScannError> {
        let vectors = self.vectors.read().unwrap();
        let stored_ids = self.ids.read().unwrap();

        if vectors.is_empty() {
            return Err(ScannError::NotTrained);
        }

        // Build ID to index mapping
        let id_to_idx: HashMap<i64, usize> = stored_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Collect vectors in the order of requested IDs
        let mut result = Vec::with_capacity(ids.len() * self.dim);
        let mut found_count = 0;

        for &id in ids {
            if let Some(&idx) = id_to_idx.get(&id) {
                let start = idx * self.dim;
                result.extend_from_slice(&vectors[start..start + self.dim]);
                found_count += 1;
            }
        }

        if found_count == 0 {
            return Err(ScannError::InvalidConfig(
                "none of the requested IDs found".to_string(),
            ));
        }

        Ok(result)
    }

    /// Save index to file
    pub fn save(&self, path: &str) -> Result<(), ScannError> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Save config
        let config_bytes = serde_json::to_vec(&self.config).map_err(|e| {
            ScannError::InvalidConfig(e.to_string())
        })?;
        file.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        file.write_all(&config_bytes)?;

        // Save quantizer codebook
        let codebook_len = self.quantizer.codebook.len() as u32;
        file.write_all(&codebook_len.to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&self.quantizer.codebook))?;

        // Save weights
        let weights_len = self.quantizer.weights.len() as u32;
        file.write_all(&weights_len.to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&self.quantizer.weights))?;

        // Save centroid norms
        let norms_len = self.quantizer.centroid_norms.len() as u32;
        file.write_all(&norms_len.to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&self.quantizer.centroid_norms))?;

        // Save vectors and ids
        let vecs = self.vectors.read().unwrap();
        let id_list = self.ids.read().unwrap();

        let vec_len = vecs.len() as u32;
        file.write_all(&vec_len.to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&vecs))?;

        let id_len = id_list.len() as u32;
        file.write_all(&id_len.to_le_bytes())?;
        file.write_all(bytemuck::cast_slice(&id_list))?;

        // Save inverted lists
        let inverted_lists = self.inverted_lists.read().unwrap();
        let list_count = inverted_lists.len() as u32;
        file.write_all(&list_count.to_le_bytes())?;

        for (partition, list) in inverted_lists.iter() {
            let partition_u32 = *partition as u32;
            file.write_all(&partition_u32.to_le_bytes())?;

            let list_len = list.len() as u32;
            file.write_all(&list_len.to_le_bytes())?;

            for &(id, ref codes) in list {
                file.write_all(&id.to_le_bytes())?;
                let code_len = codes.len() as u32;
                file.write_all(&code_len.to_le_bytes())?;
                file.write_all(codes)?;
            }
        }

        // Save trained flag
        let trained: u8 = if self.trained { 1 } else { 0 };
        file.write_all(&[trained])?;

        Ok(())
    }

    /// Load index from file
    pub fn load(&mut self, path: &str) -> Result<(), ScannError> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;

        // Load config
        let mut config_len_bytes = [0u8; 4];
        file.read_exact(&mut config_len_bytes)?;
        let config_len = u32::from_le_bytes(config_len_bytes) as usize;
        let mut config_bytes = vec![0u8; config_len];
        file.read_exact(&mut config_bytes)?;
        let loaded_config: ScaNNConfig = serde_json::from_slice(&config_bytes).map_err(|e| {
            ScannError::InvalidConfig(e.to_string())
        })?;

        // Verify config matches
        if loaded_config.num_partitions != self.config.num_partitions
            || loaded_config.num_centroids != self.config.num_centroids
        {
            return Err(ScannError::InvalidConfig(
                "Config mismatch during load".to_string(),
            ));
        }

        // Load codebook
        let mut codebook_len_bytes = [0u8; 4];
        file.read_exact(&mut codebook_len_bytes)?;
        let codebook_len = u32::from_le_bytes(codebook_len_bytes) as usize;
        self.quantizer.codebook.resize(codebook_len, 0.0);
        file.read_exact(bytemuck::cast_slice_mut(&mut self.quantizer.codebook))?;

        // Load weights
        let mut weights_len_bytes = [0u8; 4];
        file.read_exact(&mut weights_len_bytes)?;
        let weights_len = u32::from_le_bytes(weights_len_bytes) as usize;
        self.quantizer.weights.resize(weights_len, 0.0);
        file.read_exact(bytemuck::cast_slice_mut(&mut self.quantizer.weights))?;

        // Load centroid norms
        let mut norms_len_bytes = [0u8; 4];
        file.read_exact(&mut norms_len_bytes)?;
        let norms_len = u32::from_le_bytes(norms_len_bytes) as usize;
        self.quantizer.centroid_norms.resize(norms_len, 0.0);
        file.read_exact(bytemuck::cast_slice_mut(&mut self.quantizer.centroid_norms))?;

        // Load vectors
        let mut vec_len_bytes = [0u8; 4];
        file.read_exact(&mut vec_len_bytes)?;
        let vec_len = u32::from_le_bytes(vec_len_bytes) as usize;
        let mut vecs = vec![0.0f32; vec_len];
        file.read_exact(bytemuck::cast_slice_mut(&mut vecs))?;

        // Load ids
        let mut id_len_bytes = [0u8; 4];
        file.read_exact(&mut id_len_bytes)?;
        let id_len = u32::from_le_bytes(id_len_bytes) as usize;
        let mut id_list = vec![0i64; id_len];
        file.read_exact(bytemuck::cast_slice_mut(&mut id_list))?;

        // Load inverted lists
        let mut list_count_bytes = [0u8; 4];
        file.read_exact(&mut list_count_bytes)?;
        let list_count = u32::from_le_bytes(list_count_bytes) as usize;

        let mut inverted_lists = HashMap::new();
        for _ in 0..list_count {
            let mut partition_bytes = [0u8; 4];
            file.read_exact(&mut partition_bytes)?;
            let partition = u32::from_le_bytes(partition_bytes) as usize;

            let mut list_len_bytes = [0u8; 4];
            file.read_exact(&mut list_len_bytes)?;
            let list_len = u32::from_le_bytes(list_len_bytes) as usize;

            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                let id = i64::from_le_bytes(id_bytes);

                let mut code_len_bytes = [0u8; 4];
                file.read_exact(&mut code_len_bytes)?;
                let code_len = u32::from_le_bytes(code_len_bytes) as usize;
                let mut codes = vec![0u8; code_len];
                file.read_exact(&mut codes)?;

                list.push((id, codes));
            }

            inverted_lists.insert(partition, list);
        }

        // Load trained flag
        let mut trained_byte = [0u8; 1];
        file.read_exact(&mut trained_byte)?;
        let trained = trained_byte[0] == 1;

        // Apply loaded data
        *self.vectors.write().unwrap() = vecs;
        *self.ids.write().unwrap() = id_list;
        *self.inverted_lists.write().unwrap() = inverted_lists;
        self.quantizer.trained = trained;
        self.trained = trained;

        Ok(())
    }

    /// Check if this index contains raw data
    /// 
    /// ScaNN stores raw vectors for re-ranking when reorder_k > 0
    pub fn has_raw_data(&self) -> bool {
        self.config.reorder_k > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scann_config_validation() {
        let config = ScaNNConfig::default();
        assert!(config.validate(128).is_ok());
        assert!(config.validate(127).is_err());

        let mut config = ScaNNConfig::default();
        config.num_partitions = 0;
        assert!(config.validate(128).is_err());
    }

    #[test]
    fn test_scann_basic() {
        let dim = 128;
        let config = ScaNNConfig::new(16, 256, 100);

        let mut index = ScaNNIndex::new(dim, config).unwrap();

        // Generate test data
        let n = 1000;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.01).sin();
        }

        // Train
        index.train(&data, None);
        assert!(index.is_trained());

        // Add
        let added = index.add(&data, None);
        assert_eq!(added, n);
        assert_eq!(index.count(), n);

        // Search
        let query = &data[0..dim];
        let results = index.search(query, 10);
        assert_eq!(results.len(), 10);

        // First result should be closest (self)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_scann_with_query_sample() {
        let dim = 64;
        let config = ScaNNConfig::new(8, 64, 50);

        let mut index = ScaNNIndex::new(dim, config).unwrap();

        let n = 500;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.02).cos();
        }

        // Generate query sample
        let mut query_sample = vec![0.0f32; 100 * dim];
        for i in 0..query_sample.len() {
            query_sample[i] = (i as f32 * 0.03).sin();
        }

        // Train with query sample
        index.train(&data, Some(&query_sample));
        assert!(index.is_trained());

        // Add and search
        index.add(&data, None);
        let query = &data[0..dim];
        let results = index.search(query, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_scann_save_load() {
        let dim = 64;
        let config = ScaNNConfig::new(8, 64, 30);

        let mut index = ScaNNIndex::new(dim, config).unwrap();

        let n = 100;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.05).sin();
        }

        index.train(&data, None);
        index.add(&data, None);

        // Save
        let path = "/tmp/test_scann.index";
        index.save(path).unwrap();

        // Load
        let mut loaded = ScaNNIndex::new(dim, ScaNNConfig::new(8, 64, 30)).unwrap();
        loaded.load(path).unwrap();

        // Verify
        assert_eq!(loaded.count(), n);
        assert!(loaded.is_trained());

        // Search with loaded index
        let query = &data[0..dim];
        let results = loaded.search(query, 5);
        assert_eq!(results.len(), 5);

        // Clean up
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_scann_with_ids() {
        let dim = 32;
        let config = ScaNNConfig::new(4, 32, 20);

        let mut index = ScaNNIndex::new(dim, config).unwrap();

        let n = 50;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.05).sin();
        }

        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        // Train first
        index.train(&data, None);
        
        // Then add
        index.add(&data, Some(&ids));
        assert_eq!(index.count(), n);

        // Search and verify IDs
        let query = &data[0..dim];
        let results = index.search(query, 5);
        for (id, _) in &results {
            assert!(*id >= 1000 && *id < 1000 + n as i64);
        }
    }

    #[test]
    fn test_scann_empty_search() {
        let dim = 64;
        let config = ScaNNConfig::default();
        let index = ScaNNIndex::new(dim, config).unwrap();

        let query = vec![0.1; dim];
        let results = index.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_anisotropic_quantizer() {
        let dim = 64;
        let config = ScaNNConfig::new(8, 64, 30);
        let mut quantizer = AnisotropicQuantizer::new(dim, config);

        let n = 200;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..data.len() {
            data[i] = (i as f32 * 0.02).sin();
        }

        quantizer.train(&data, None);
        assert!(quantizer.is_trained());

        // Test encode
        let vector = &data[0..dim];
        let codes = quantizer.encode(vector);
        assert_eq!(codes.len(), 8); // num_partitions

        // Test ADC distance
        let dist = quantizer.adc_distance(vector, &codes);
        assert!(dist.is_finite());
    }
}

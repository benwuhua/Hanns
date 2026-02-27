//! HNSW-PRQ Index Implementation
//! 
//! HNSW index with Product Residual Quantization for storage compression.
//! Combines HNSW graph structure with PRQ for efficient vector storage and search.

use std::collections::{HashMap, HashSet};
use rand::Rng;

use crate::api::{MetricType, Result as ApiResult};
use crate::dataset::Dataset;
use crate::index::{Index, IndexError, SearchResult};
use crate::bitset::BitsetView;
use crate::quantization::prq::{ProductResidualQuantizer, PRQConfig};

/// Maximum number of layers in the HNSW graph
const MAX_LAYERS: usize = 16;

/// HNSW-PRQ configuration
#[derive(Clone, Debug)]
pub struct HnswPrqConfig {
    /// Dimensionality of vectors
    pub dim: usize,
    /// Maximum number of connections per layer (M)
    pub m: usize,
    /// Maximum number of connections at layer 0 (M_max0, typically 2*M)
    pub m_max0: usize,
    /// Size of dynamic candidate list during construction (efConstruction)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (efSearch)
    pub ef_search: usize,
    /// Number of splits for PRQ
    pub prq_nsplits: usize,
    /// Number of subquantizers per split
    pub prq_msub: usize,
    /// Bits per subvector index
    pub prq_nbits: usize,
    /// Metric type (L2, IP, COSINE)
    pub metric_type: MetricType,
}

impl Default for HnswPrqConfig {
    fn default() -> Self {
        Self {
            dim: 0,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            prq_nsplits: 2,
            prq_msub: 4,
            prq_nbits: 8,
            metric_type: MetricType::L2,
        }
    }
}

impl HnswPrqConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max0 = m * 2;
        self
    }

    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn with_prq_params(mut self, nsplits: usize, msub: usize, nbits: usize) -> Self {
        self.prq_nsplits = nsplits;
        self.prq_msub = msub;
        self.prq_nbits = nbits;
        self
    }

    pub fn with_metric_type(mut self, metric: MetricType) -> Self {
        self.metric_type = metric;
        self
    }
}

/// Neighbor connection at a specific layer
#[derive(Clone, Debug)]
pub struct LayerNeighbors {
    /// Neighbor IDs and their distances
    pub neighbors: Vec<(i64, f32)>,
}

impl LayerNeighbors {
    fn new() -> Self {
        Self {
            neighbors: Vec::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            neighbors: Vec::with_capacity(capacity),
        }
    }
}

/// Node information
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Maximum layer this node exists in
    pub max_layer: usize,
    /// Neighbor connections per layer
    pub layer_neighbors: Vec<LayerNeighbors>,
    /// PRQ-encoded vector
    pub code: Vec<u8>,
}

impl NodeInfo {
    fn new(max_layer: usize, m: usize, code_size: usize) -> Self {
        let mut layer_neighbors = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            layer_neighbors.push(LayerNeighbors::with_capacity(m * 2));
        }
        Self {
            max_layer,
            layer_neighbors,
            code: vec![0u8; code_size],
        }
    }
}

/// HNSW-PRQ Index
pub struct HnswPrqIndex {
    config: HnswPrqConfig,
    /// Entry point for search
    entry_point: Option<i64>,
    /// Maximum layer in the graph
    max_level: usize,
    /// Node information
    node_info: Vec<NodeInfo>,
    /// Vector IDs
    ids: Vec<i64>,
    /// ID to index mapping
    id_to_idx: HashMap<i64, usize>,
    /// Next ID to assign
    next_id: i64,
    /// Whether the index is trained
    trained: bool,
    /// Level multiplier for random level generation
    level_multiplier: f32,
    /// PRQ quantizer
    prq: ProductResidualQuantizer,
}

impl HnswPrqIndex {
    /// Create a new HNSW-PRQ index
    pub fn new(config: HnswPrqConfig) -> ApiResult<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".into(),
            ));
        }

        // Validate PRQ parameters
        if config.dim % config.prq_nsplits != 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be divisible by prq_nsplits".into(),
            ));
        }

        // Create PRQ quantizer
        let prq_config = PRQConfig {
            d: config.dim,
            nsplits: config.prq_nsplits,
            msub: config.prq_msub,
            nbits: config.prq_nbits,
            max_beam_size: 5,
        };
        let prq = ProductResidualQuantizer::new(prq_config)?;
        let code_size = prq.code_size();

        // Calculate level multiplier: m_l = 1 / ln(M)
        let level_multiplier = 1.0 / (config.m as f32).ln().max(1.0);

        Ok(Self {
            config,
            entry_point: None,
            max_level: 0,
            node_info: Vec::new(),
            ids: Vec::new(),
            id_to_idx: HashMap::new(),
            next_id: 0,
            trained: false,
            level_multiplier,
            prq,
        })
    }

    /// Generate a random level for a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let level = (-r.ln() * self.level_multiplier) as usize;
        level.min(MAX_LAYERS - 1)
    }

    /// Get maximum connections for a layer
    #[inline]
    fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m
        }
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric_type {
            MetricType::L2 => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
            }
            MetricType::Ip => {
                -a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f32>()
            }
            MetricType::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                if norm_a > 0.0 && norm_b > 0.0 {
                    1.0 - dot / (norm_a * norm_b)
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute distance between a query and a stored code
    fn compute_distance_to_code(&self, query: &[f32], code: &[u8]) -> f32 {
        self.prq.compute_distance(query, code)
    }

    /// Train the PRQ quantizer
    pub fn train(&mut self, vectors: &[f32]) -> Result<(), IndexError> {
        let n = vectors.len() / self.config.dim;
        if n == 0 {
            return Err(IndexError::Unsupported("no training vectors".into()));
        }

        // Train PRQ - note: prq.train returns ApiResult, need to convert
        self.prq.train(vectors).map_err(|_| IndexError::Unsupported("PRQ train failed".into()))?;
        self.trained = true;
        Ok(())
    }

    /// Add vectors to the index (returns Result with IndexError)
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize, IndexError> {
        if !self.trained {
            return Err(IndexError::NotTrained);
        }

        let n = vectors.len() / self.config.dim;
        if n == 0 {
            return Ok(0);
        }

        let base_count = self.ids.len();
        self.node_info.reserve(n);
        self.ids.reserve(n);

        let code_size = self.prq.code_size();

        for i in 0..n {
            let start = i * self.config.dim;
            let vec = &vectors[start..start + self.config.dim];

            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            // Generate random level
            let node_level = self.random_level();

            // Encode vector with PRQ
            let mut code = vec![0u8; code_size];
            self.prq.encode(vec, &mut code).map_err(|_| IndexError::Unsupported("PRQ encode failed".into()))?;

            // Create node info
            let node_info = NodeInfo::new(node_level, self.config.m, code_size);

            // Store metadata
            let idx = self.ids.len();
            self.ids.push(id);
            self.id_to_idx.insert(id, idx);

            // We need to store the original vector temporarily for graph construction
            // In a real implementation, we'd use a different approach
            let vec_clone = vec.to_vec();

            self.node_info.push(node_info);
            self.node_info[idx].code = code;

            // If first node, set as entry point
            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_level;
                continue;
            }

            // Insert into graph
            self.insert_node(idx, &vec_clone, node_level);

            // Update max level
            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(id);
            }
        }

        Ok(n)
    }

    /// Insert a node into the HNSW graph
    fn insert_node(&mut self, idx: usize, vector: &[f32], node_level: usize) {
        let mut entry_point = self.entry_point.unwrap();
        let mut entry_point_idx = self.id_to_idx[&entry_point];

        // Search for entry point at each level from max_level down to node_level + 1
        for layer in (node_level + 1..=self.max_level).rev() {
            let ef = 1; // Greedy search at upper levels
            let candidates = self.search_at_layer(
                vector,
                entry_point_idx,
                layer,
                ef,
            );

            if let Some(&(best_idx, _)) = candidates.first() {
                entry_point_idx = best_idx;
                entry_point = self.ids[best_idx];
            }
        }

        // Connect at layers 0 to node_level
        for layer in 0..=node_level {
            let ef = if layer == 0 {
                self.config.ef_construction
            } else {
                1
            };

            let candidates = self.search_at_layer(
                vector,
                entry_point_idx,
                layer,
                ef,
            );

            // Select neighbors using heuristics
            let neighbors = self.select_neighbors(candidates, layer);

            // Add bidirectional connections
            let max_m = self.max_connections_for_layer(layer);

            // Connect new node to neighbors
            self.node_info[idx].layer_neighbors[layer].neighbors = neighbors
                .iter()
                .take(max_m)
                .map(|&(n_idx, dist)| (self.ids[n_idx], dist))
                .collect();

            // Connect neighbors to new node (bidirectional)
            for &(n_idx, dist) in &neighbors {
                let neighbors_list = &mut self.node_info[n_idx].layer_neighbors[layer].neighbors;
                if neighbors_list.len() < max_m {
                    neighbors_list.push((self.ids[idx], dist));
                } else {
                    // Find the furthest neighbor and replace if closer
                    let mut furthest_idx = 0;
                    let mut furthest_dist = f32::MIN;
                    for (i, &(_, d)) in neighbors_list.iter().enumerate() {
                        if d > furthest_dist {
                            furthest_dist = d;
                            furthest_idx = i;
                        }
                    }
                    if dist < furthest_dist {
                        neighbors_list[furthest_idx] = (self.ids[idx], dist);
                    }
                }
            }
        }
    }

    /// Search at a specific layer
    fn search_at_layer(
        &self,
        query: &[f32],
        entry_point_idx: usize,
        layer: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates: Vec<(f32, usize)> = Vec::new(); // (distance, idx)
        let mut results: Vec<(usize, f32)> = Vec::new(); // (idx, distance)

        // Initialize with entry point
        let entry_dist = self.compute_distance_to_code(query, &self.node_info[entry_point_idx].code);
        candidates.push((entry_dist, entry_point_idx));
        visited.insert(entry_point_idx);

        // Use a sorted approach (simple implementation)
        while let Some((dist, idx)) = candidates.pop() {
            // If this is worse than the worst in results and we have enough results
            if results.len() >= ef && dist > results.last().map(|(_, d)| *d).unwrap_or(f32::MAX) {
                break;
            }

            results.push((idx, dist));
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(ef);

            // Explore neighbors
            let neighbors = &self.node_info[idx].layer_neighbors[layer].neighbors;
            for &(neighbor_id, _) in neighbors {
                if let Some(&neighbor_idx) = self.id_to_idx.get(&neighbor_id) {
                    if !visited.contains(&neighbor_idx) {
                        visited.insert(neighbor_idx);
                        let neighbor_dist =
                            self.compute_distance_to_code(query, &self.node_info[neighbor_idx].code);
                        candidates.push((neighbor_dist, neighbor_idx));
                        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    }
                }
            }
        }

        results
    }

    /// Select neighbors using heuristics
    fn select_neighbors(&self, candidates: Vec<(usize, f32)>, layer: usize) -> Vec<(usize, f32)> {
        // Simple selection: take the closest ones
        let max_m = self.max_connections_for_layer(layer);
        let mut sorted = candidates;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.into_iter().take(max_m).collect()
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], top_k: usize, bitset: Option<&BitsetView>) -> Result<SearchResult, IndexError> {
        if !self.trained || self.ids.is_empty() {
            return Err(IndexError::NotTrained);
        }

        if query.len() != self.config.dim {
            return Err(IndexError::DimMismatch);
        }

        let entry_point = match self.entry_point {
            Some(id) => self.id_to_idx[&id],
            None => return Ok(SearchResult::new(Vec::new(), Vec::new(), 0.0)),
        };

        // Search from top layer down to layer 0
        let mut current_idx = entry_point;

        for layer in (1..=self.max_level).rev() {
            let candidates = self.search_at_layer(
                query,
                current_idx,
                layer,
                1, // Greedy search at upper levels
            );
            if let Some(&(idx, _)) = candidates.first() {
                current_idx = idx;
            }
        }

        // Final search at layer 0 with ef_search
        let ef = self.config.ef_search.max(top_k);
        let candidates = self.search_at_layer(
            query,
            current_idx,
            0,
            ef,
        );

        // Apply bitset filter and collect results
        let mut results: Vec<(i64, f32)> = Vec::new();
        for (idx, dist) in candidates {
            // Check bitset
            if let Some(bs) = bitset {
                if idx < bs.len() && bs.get(idx) {
                    continue; // Filtered out
                }
            }
            results.push((self.ids[idx], dist));
        }

        // Sort by distance and take top_k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);

        let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = results.iter().map(|(_, d)| *d).collect();

        Ok(SearchResult::new(ids, distances, 0.0))
    }

    /// Get the number of vectors
    pub fn count(&self) -> usize {
        self.ids.len()
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }
}

impl Index for HnswPrqIndex {
    fn index_type(&self) -> &str {
        "HNSW_PRQ"
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn count(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> Result<(), IndexError> {
        let vectors = dataset.vectors();
        self.train(vectors)
    }

    fn add(&mut self, dataset: &Dataset) -> Result<usize, IndexError> {
        let vectors = dataset.vectors();
        self.add(vectors, None)
    }

    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult, IndexError> {
        let vectors = query.vectors();
        if vectors.is_empty() {
            return Err(IndexError::Empty);
        }
        // Search with first query vector
        self.search(&vectors[0..self.config.dim], top_k, None)
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult, IndexError> {
        let vectors = query.vectors();
        if vectors.is_empty() {
            return Err(IndexError::Empty);
        }
        self.search(&vectors[0..self.config.dim], top_k, Some(bitset))
    }

    fn save(&self, _path: &str) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("serialization not implemented".into()))
    }

    fn load(&mut self, _path: &str) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("deserialization not implemented".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_prq_creation() {
        let config = HnswPrqConfig::new(64)
            .with_m(16)
            .with_ef_construction(200)
            .with_ef_search(64)
            .with_prq_params(2, 4, 8);

        let index = HnswPrqIndex::new(config).unwrap();
        assert_eq!(index.dim(), 64);
        assert!(!index.is_trained());
        assert_eq!(index.count(), 0);
    }

    #[test]
    fn test_hnsw_prq_train_and_search() {
        let config = HnswPrqConfig::new(16)
            .with_m(8)
            .with_ef_construction(100)
            .with_ef_search(32)
            .with_prq_params(2, 4, 8);

        let mut index = HnswPrqIndex::new(config).unwrap();

        // Generate training data
        let n_train = 100;
        let mut train_data = vec![0.0f32; n_train * 16];
        for i in 0..n_train {
            for j in 0..16 {
                train_data[i * 16 + j] = (i + j) as f32 * 0.01;
            }
        }

        index.train(&train_data).unwrap();
        assert!(index.is_trained());

        // Add vectors
        index.add(&train_data, None).unwrap();
        assert_eq!(index.count(), 100);

        // Search
        let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let results = index.search(&query, 5, None).unwrap();
        assert!(results.ids.len() <= 5);
        assert_eq!(results.ids.len(), results.distances.len());
    }
}

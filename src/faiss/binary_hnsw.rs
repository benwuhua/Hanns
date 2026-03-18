//! BinaryHNSW - HNSW Index for Binary Vectors
//!
//! Implements HNSW (Hierarchical Navigable Small World) index for binary vectors.
//! Uses Hamming distance for similarity search.
//!
//! Reference: Faiss IndexBinaryHNSW

use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::api::{IndexConfig, Result, SearchResult as ApiSearchResult};

/// Maximum number of layers in the HNSW graph
const MAX_LAYERS: usize = 16;
const BINARY_HNSW_MAGIC: &[u8; 6] = b"BNHNSW";
const BINARY_HNSW_VERSION: u32 = 1;

/// A single node's neighbor connections at a specific layer
#[derive(Clone, Debug)]
pub struct LayerNeighbors {
    /// Neighbor IDs at this layer
    pub neighbors: Vec<i64>,
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

/// Node information including its layer assignment and connections
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// The highest layer this node exists in (0 = only base layer)
    pub max_layer: usize,
    /// Neighbor connections per layer (index 0 = layer 0, etc.)
    pub layer_neighbors: Vec<LayerNeighbors>,
}

impl NodeInfo {
    fn new(max_layer: usize, m: usize) -> Self {
        let mut layer_neighbors = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            layer_neighbors.push(LayerNeighbors::with_capacity(m * 2));
        }
        Self {
            max_layer,
            layer_neighbors,
        }
    }
}

/// Binary HNSW index for binary vectors (u8 arrays)
#[allow(dead_code)]
pub struct BinaryHnswIndex {
    config: IndexConfig,
    entry_point: Option<i64>,
    max_level: usize,
    /// Binary vectors stored as u8 arrays (dim bits / 8 bytes per vector)
    vectors: Vec<u8>,
    ids: Vec<i64>,
    id_to_idx: std::collections::HashMap<i64, usize>,
    node_info: Vec<NodeInfo>,
    next_id: i64,
    trained: bool,
    /// Dimension in bits
    dim_bits: usize,
    /// Dimension in bytes (dim_bits / 8, rounded up)
    dim_bytes: usize,
    ef_construction: usize,
    ef_search: usize,
    m: usize,
    m_max0: usize,
    level_multiplier: f32,
}

impl BinaryHnswIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        // Get M parameter (default 16, typical range 5-48)
        let m = config.params.m.unwrap_or(16).clamp(2, 64);

        // M_max0 is typically 2*M for layer 0 (denser connections at base)
        let m_max0 = m * 2;

        // Get ef_search parameter (default 64)
        let ef_search = config.params.ef_search.unwrap_or(64).max(1);

        // Get ef_construction parameter (default 200)
        let ef_construction = config.params.ef_construction.unwrap_or(200).max(1);

        // Calculate level multiplier: m_l = 1 / ln(M)
        let level_multiplier = 1.0 / (m as f32).ln().max(1.0);

        // dim is in bits, convert to bytes
        let dim_bits = config.dim;
        let dim_bytes = dim_bits.div_ceil(8);

        Ok(Self {
            config: config.clone(),
            entry_point: None,
            vectors: Vec::new(),
            ids: Vec::new(),
            id_to_idx: std::collections::HashMap::new(),
            node_info: Vec::new(),
            next_id: 0,
            trained: false,
            dim_bits,
            dim_bytes,
            ef_construction,
            ef_search,
            m,
            m_max0,
            max_level: 0,
            level_multiplier,
        })
    }

    /// Generate a random level for a new node using exponential distribution
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen(); // Uniform [0, 1)

        // Inverse CDF of exponential distribution
        let level = (-r.ln() * self.level_multiplier) as usize;

        level.min(MAX_LAYERS - 1)
    }

    /// Get the maximum allowed connections for a given layer
    #[inline]
    #[allow(dead_code)]
    fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    /// Compute Hamming distance between two binary vectors
    #[inline]
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    pub fn train(&mut self, vectors: &[u8]) -> Result<()> {
        let n = vectors.len() / self.dim_bytes;
        if n * self.dim_bytes != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        self.trained = true;
        Ok(())
    }

    /// Add binary vectors to the index
    pub fn add(&mut self, vectors: &[u8], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim_bytes;
        if n == 0 {
            return Ok(0);
        }

        let base_count = self.ids.len();
        self.vectors.reserve(n * self.dim_bytes);
        self.ids.reserve(n);
        self.node_info.reserve(n);

        for i in 0..n {
            let start = i * self.dim_bytes;
            let new_vec = &vectors[start..start + self.dim_bytes];

            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            // Assign random level to this node
            let node_level = self.random_level();

            // Create node info with appropriate layer structure
            let node_info = NodeInfo::new(node_level, self.m);

            // Store vector and metadata
            let idx = self.ids.len();
            self.ids.push(id);
            self.id_to_idx.insert(id, idx);
            self.vectors.extend_from_slice(new_vec);
            self.node_info.push(node_info);

            // If this is the first node, set it as entry point
            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_level;
                continue;
            }

            // Insert node into the graph layer by layer
            self.insert_node(idx, new_vec, node_level);

            // Update global max level and entry point if needed
            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(id);
            }
        }

        Ok(n)
    }

    /// Insert a node into the multi-layer graph
    fn insert_node(&mut self, new_idx: usize, new_vec: &[u8], node_level: usize) {
        let new_id = self.ids[new_idx];

        // Start from the top layer and work down
        let mut curr_ep = self.entry_point.unwrap();

        // For each layer from max_level down to 0
        for level in (0..=self.max_level.min(node_level)).rev() {
            // Search for nearest neighbor at this layer
            let nearest_results = self.search_layer(new_vec, curr_ep, level, 1);

            if !nearest_results.is_empty() {
                curr_ep = nearest_results[0].0;
            }

            // If this node exists at this level, connect it
            if level <= node_level {
                // Find efConstruction candidates at this layer
                let candidates = self.search_layer(new_vec, curr_ep, level, self.ef_construction);

                // Select best M neighbors using heuristic
                let selected = self.select_neighbors_heuristic(new_vec, &candidates, self.m);

                // Add bidirectional connections
                self.add_bidirectional_connections(new_idx, new_id, level, &selected);
            }
        }
    }

    /// Search for nearest neighbors at a specific layer
    fn search_layer(
        &self,
        query: &[u8],
        entry_id: i64,
        level: usize,
        ef: usize,
    ) -> Vec<(i64, u32)> {
        use std::collections::BinaryHeap;

        // Wrapper for u32 to implement Ord for BinaryHeap
        #[derive(Clone, Copy, PartialEq)]
        struct OrderedDist(u32);

        impl Eq for OrderedDist {}

        impl PartialOrd for OrderedDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for OrderedDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse order for min-heap behavior
                other.0.cmp(&self.0)
            }
        }

        // Visited set
        let mut visited: HashSet<i64> = HashSet::with_capacity(ef * 2);
        visited.insert(entry_id);

        // Candidate heap (max-heap, will keep ef smallest)
        let mut candidates: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::new();

        // Result list (sorted by distance)
        let mut results: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::new();

        // Initialize with entry point
        let entry_idx = self.id_to_idx[&entry_id];
        let entry_vec = self.get_vector(entry_idx);
        let entry_dist = self.hamming_distance(query, entry_vec);
        candidates.push((OrderedDist(entry_dist), entry_id));
        results.push((OrderedDist(entry_dist), entry_id));

        // Greedy search
        while let Some((OrderedDist(curr_dist), curr_id)) = candidates.pop() {
            // Check if we can stop (all candidates are worse than best in results)
            if let Some((OrderedDist(best_dist), _)) = results.peek() {
                if curr_dist > *best_dist {
                    break;
                }
            }

            // Get neighbors at this layer
            let curr_idx = self.id_to_idx[&curr_id];
            let neighbors = self.get_neighbors_at_layer(curr_idx, level);

            for &neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let neighbor_idx = self.id_to_idx[&neighbor_id];
                let neighbor_vec = self.get_vector(neighbor_idx);
                let dist = self.hamming_distance(query, neighbor_vec);

                // Check if we should add to results
                let should_add = results.len() < ef
                    || dist < results.peek().map(|(d, _)| d.0).unwrap_or(u32::MAX);

                if should_add {
                    candidates.push((OrderedDist(dist), neighbor_id));
                    results.push((OrderedDist(dist), neighbor_id));

                    // Keep only ef best
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to sorted vec (ascending by distance)
        let mut result_vec: Vec<(i64, u32)> =
            results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_by_key(|(_, d)| *d);
        result_vec
    }

    /// Get neighbors of a node at a specific layer
    fn get_neighbors_at_layer(&self, node_idx: usize, layer: usize) -> &[i64] {
        if layer >= self.node_info[node_idx].layer_neighbors.len() {
            &[]
        } else {
            &self.node_info[node_idx].layer_neighbors[layer].neighbors
        }
    }

    /// Get vector by index
    fn get_vector(&self, idx: usize) -> &[u8] {
        let start = idx * self.dim_bytes;
        &self.vectors[start..start + self.dim_bytes]
    }

    /// Select neighbors using heuristic (prefer closer and diverse neighbors)
    fn select_neighbors_heuristic(
        &self,
        _query: &[u8],
        candidates: &[(i64, u32)],
        m: usize,
    ) -> Vec<i64> {
        // Simple heuristic: select m closest neighbors
        let mut sorted: Vec<(i64, u32)> = candidates.to_vec();
        sorted.sort_by_key(|(_, d)| *d);

        sorted.iter().take(m).map(|(id, _)| *id).collect()
    }

    /// Add bidirectional connections between nodes
    fn add_bidirectional_connections(
        &mut self,
        new_idx: usize,
        new_id: i64,
        level: usize,
        neighbors: &[i64],
    ) {
        // Ensure node has enough layer capacity
        while self.node_info[new_idx].layer_neighbors.len() <= level {
            self.node_info[new_idx]
                .layer_neighbors
                .push(LayerNeighbors::new());
        }

        // Add forward connections (new node -> neighbors)
        self.node_info[new_idx].layer_neighbors[level]
            .neighbors
            .extend(neighbors.iter());

        // Add backward connections (neighbors -> new node)
        for &neighbor_id in neighbors {
            let neighbor_idx = self.id_to_idx[&neighbor_id];

            // Ensure neighbor has enough layer capacity
            while self.node_info[neighbor_idx].layer_neighbors.len() <= level {
                self.node_info[neighbor_idx]
                    .layer_neighbors
                    .push(LayerNeighbors::new());
            }

            self.node_info[neighbor_idx].layer_neighbors[level]
                .neighbors
                .push(new_id);
        }
    }

    /// Search for k nearest neighbors (single query)
    pub fn search(&self, query: &[u8], k: usize) -> ApiSearchResult {
        // Search from entry point at layer 0
        let entry_point = self.entry_point.unwrap_or(0);
        let results = self.search_layer(query, entry_point, 0, self.ef_search);

        let mut labels = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);

        // Take top k results
        for (id, dist) in results.iter().take(k) {
            labels.push(*id);
            distances.push(*dist as f32);
        }

        // Pad with -1 if not enough results
        while labels.len() < k {
            labels.push(-1);
            distances.push(f32::INFINITY);
        }

        ApiSearchResult {
            ids: labels,
            distances,
            elapsed_ms: 0.0,
            num_visited: results.len(),
        }
    }

    pub fn reset(&mut self) {
        self.vectors.clear();
        self.ids.clear();
        self.id_to_idx.clear();
        self.node_info.clear();
        self.entry_point = None;
        self.max_level = 0;
        self.next_id = 0;
        self.trained = false;
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn save(&self, path: &Path) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;

        file.write_all(BINARY_HNSW_MAGIC)?;
        file.write_all(&BINARY_HNSW_VERSION.to_le_bytes())?;

        file.write_all(&(self.dim_bits as u32).to_le_bytes())?;
        file.write_all(&(self.dim_bytes as u32).to_le_bytes())?;
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.m_max0 as u32).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(self.ef_search as u32).to_le_bytes())?;
        file.write_all(&self.level_multiplier.to_le_bytes())?;
        file.write_all(&[u8::from(self.trained)])?;
        file.write_all(&(self.max_level as u32).to_le_bytes())?;
        file.write_all(&self.entry_point.unwrap_or(-1).to_le_bytes())?;
        file.write_all(&self.next_id.to_le_bytes())?;

        file.write_all(&(self.vectors.len() as u64).to_le_bytes())?;
        file.write_all(&self.vectors)?;

        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        file.write_all(&(self.node_info.len() as u64).to_le_bytes())?;
        for node in &self.node_info {
            file.write_all(&(node.max_layer as u32).to_le_bytes())?;
            file.write_all(&(node.layer_neighbors.len() as u32).to_le_bytes())?;
            for layer in &node.layer_neighbors {
                file.write_all(&(layer.neighbors.len() as u32).to_le_bytes())?;
                for &neighbor in &layer.neighbors {
                    file.write_all(&neighbor.to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    pub fn load(
        path: &Path,
        dim: usize,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != BINARY_HNSW_MAGIC {
            return Err("invalid BinaryHNSW magic".into());
        }

        let mut u32_buf = [0u8; 4];
        file.read_exact(&mut u32_buf)?;
        let version = u32::from_le_bytes(u32_buf);
        if version != BINARY_HNSW_VERSION {
            return Err(format!("unsupported BinaryHNSW version: {}", version).into());
        }

        file.read_exact(&mut u32_buf)?;
        let dim_bits = u32::from_le_bytes(u32_buf) as usize;
        if dim_bits != dim {
            return Err(format!("dim mismatch: load dim {} vs stored {}", dim, dim_bits).into());
        }

        file.read_exact(&mut u32_buf)?;
        let dim_bytes = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let m = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let m_max0 = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_construction = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_search = u32::from_le_bytes(u32_buf) as usize;

        let mut f32_buf = [0u8; 4];
        file.read_exact(&mut f32_buf)?;
        let level_multiplier = f32::from_le_bytes(f32_buf);

        let mut trained_buf = [0u8; 1];
        file.read_exact(&mut trained_buf)?;
        let trained = trained_buf[0] != 0;

        file.read_exact(&mut u32_buf)?;
        let max_level = u32::from_le_bytes(u32_buf) as usize;

        let mut i64_buf = [0u8; 8];
        file.read_exact(&mut i64_buf)?;
        let entry_raw = i64::from_le_bytes(i64_buf);
        let entry_point = if entry_raw < 0 { None } else { Some(entry_raw) };

        file.read_exact(&mut i64_buf)?;
        let next_id = i64::from_le_bytes(i64_buf);

        let mut u64_buf = [0u8; 8];
        file.read_exact(&mut u64_buf)?;
        let vectors_len = u64::from_le_bytes(u64_buf) as usize;
        let mut vectors = vec![0u8; vectors_len];
        file.read_exact(&mut vectors)?;

        file.read_exact(&mut u64_buf)?;
        let ids_len = u64::from_le_bytes(u64_buf) as usize;
        let mut ids = vec![0i64; ids_len];
        for id in &mut ids {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            *id = i64::from_le_bytes(buf);
        }

        file.read_exact(&mut u64_buf)?;
        let node_count = u64::from_le_bytes(u64_buf) as usize;
        let mut node_info = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            file.read_exact(&mut u32_buf)?;
            let node_max_layer = u32::from_le_bytes(u32_buf) as usize;
            file.read_exact(&mut u32_buf)?;
            let n_layers = u32::from_le_bytes(u32_buf) as usize;
            let mut layer_neighbors = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                file.read_exact(&mut u32_buf)?;
                let n_neighbors = u32::from_le_bytes(u32_buf) as usize;
                let mut neighbors = Vec::with_capacity(n_neighbors);
                for _ in 0..n_neighbors {
                    let mut buf = [0u8; 8];
                    file.read_exact(&mut buf)?;
                    neighbors.push(i64::from_le_bytes(buf));
                }
                layer_neighbors.push(LayerNeighbors { neighbors });
            }
            node_info.push(NodeInfo {
                max_layer: node_max_layer,
                layer_neighbors,
            });
        }

        let id_to_idx = ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect::<std::collections::HashMap<_, _>>();

        let mut params = crate::api::IndexParams::default();
        params.m = Some(m);
        params.ef_construction = Some(ef_construction);
        params.ef_search = Some(ef_search);
        let config = IndexConfig {
            index_type: crate::api::IndexType::BinaryHnsw,
            dim: dim_bits,
            metric_type: crate::api::MetricType::Hamming,
            data_type: crate::api::DataType::Binary,
            params,
        };

        Ok(Self {
            config,
            entry_point,
            max_level,
            vectors,
            ids,
            id_to_idx,
            node_info,
            next_id,
            trained,
            dim_bits,
            dim_bytes,
            ef_construction,
            ef_search,
            m,
            m_max0,
            level_multiplier,
        })
    }
}

// Implement BinaryIndex trait for BinaryHnswIndex
impl crate::index::BinaryIndex for BinaryHnswIndex {
    fn index_type(&self) -> &str {
        "BinaryHNSW"
    }

    fn dim_bits(&self) -> usize {
        self.dim_bits
    }

    fn count(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(
        &mut self,
        dataset: &crate::dataset::BinaryDataset,
    ) -> std::result::Result<(), crate::index::IndexError> {
        let vectors = dataset.vectors();
        self.train(vectors)
            .map_err(|e| crate::index::IndexError::Unsupported(e.to_string()))
    }

    fn add(
        &mut self,
        dataset: &crate::dataset::BinaryDataset,
    ) -> std::result::Result<usize, crate::index::IndexError> {
        let vectors = dataset.vectors();
        let ids = dataset.ids();
        self.add(vectors, ids)
            .map_err(|e| crate::index::IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &crate::dataset::BinaryDataset,
        top_k: usize,
    ) -> std::result::Result<crate::index::SearchResult, crate::index::IndexError> {
        let vectors = query.vectors();
        let num_queries = vectors.len() / self.dim_bytes;

        // For simplicity, we only support single query search in BinaryIndex trait
        if num_queries == 0 {
            return Ok(crate::index::SearchResult {
                ids: Vec::new(),
                distances: Vec::new(),
                elapsed_ms: 0.0,
            });
        }

        let query_vec = &vectors[0..self.dim_bytes];
        let result = self.search(query_vec, top_k);

        Ok(crate::index::SearchResult {
            ids: result.ids,
            distances: result.distances,
            elapsed_ms: result.elapsed_ms,
        })
    }

    fn search_with_bitset(
        &self,
        query: &crate::dataset::BinaryDataset,
        top_k: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> std::result::Result<crate::index::SearchResult, crate::index::IndexError> {
        // Search with bitset filtering
        let vectors = query.vectors();
        let num_queries = vectors.len() / self.dim_bytes;

        if num_queries == 0 {
            return Ok(crate::index::SearchResult {
                ids: Vec::new(),
                distances: Vec::new(),
                elapsed_ms: 0.0,
            });
        }

        let query_vec = &vectors[0..self.dim_bytes];
        let result = self.search(query_vec, top_k);

        // Filter out results where bitset is set
        let mut filtered_ids = Vec::new();
        let mut filtered_distances = Vec::new();

        for (id, dist) in result.ids.iter().zip(result.distances.iter()) {
            let idx = *id as usize;
            // Only include if bit is NOT set (i.e., vector is not filtered out)
            if idx >= bitset.len() || !bitset.get(idx) {
                filtered_ids.push(*id);
                filtered_distances.push(*dist);
            }
        }

        Ok(crate::index::SearchResult {
            ids: filtered_ids,
            distances: filtered_distances,
            elapsed_ms: result.elapsed_ms,
        })
    }

    fn save(&self, path: &str) -> std::result::Result<(), crate::index::IndexError> {
        self.save(Path::new(path))
            .map_err(|e| crate::index::IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), crate::index::IndexError> {
        let loaded = Self::load(Path::new(path), self.dim_bits())
            .map_err(|e| crate::index::IndexError::Unsupported(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn has_raw_data(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexConfig, IndexParams, IndexType, MetricType};

    fn create_test_config(dim_bits: usize) -> IndexConfig {
        IndexConfig {
            index_type: IndexType::BinaryHnsw,
            dim: dim_bits,
            metric_type: MetricType::Hamming,
            data_type: crate::api::DataType::Binary,
            params: IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(64),
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_binary_hnsw_new() {
        let config = create_test_config(128);
        let index = BinaryHnswIndex::new(&config).unwrap();

        assert_eq!(index.dim_bits, 128);
        assert_eq!(index.dim_bytes, 16);
        assert_eq!(index.m, 16);
        assert!(!index.trained);
    }

    #[test]
    fn test_binary_hnsw_train_add_search() {
        let config = create_test_config(64);
        let mut index = BinaryHnswIndex::new(&config).unwrap();

        // Train
        let train_data = vec![0u8; 100 * 8]; // 100 vectors of 64 bits (8 bytes)
        index.train(&train_data).unwrap();
        assert!(index.trained);

        // Add vectors
        let mut vectors = Vec::new();
        let ids: Vec<i64> = (0..100).collect();
        for i in 0..100 {
            let mut vec = vec![0u8; 8];
            vec[0] = i as u8; // Simple pattern
            vectors.extend(vec);
        }

        let added = index.add(&vectors, Some(&ids)).unwrap();
        assert_eq!(added, 100);
        assert_eq!(index.len(), 100);

        // Search
        let query = vec![0u8; 8];
        let result = index.search(&query, 5);

        assert_eq!(result.ids.len(), 5);
        assert_eq!(result.distances.len(), 5);
        assert!(result.ids[0] >= 0); // Found at least one result
    }

    #[test]
    fn test_hamming_distance() {
        let config = create_test_config(64);
        let index = BinaryHnswIndex::new(&config).unwrap();

        let a = vec![0b00000000u8, 0b00000000];
        let b = vec![0b11111111u8, 0b00000000];
        let c = vec![0b00001111u8, 0b00000000];

        assert_eq!(index.hamming_distance(&a, &b), 8);
        assert_eq!(index.hamming_distance(&a, &c), 4);
        assert_eq!(index.hamming_distance(&b, &c), 4);
    }

    #[test]
    fn test_binary_hnsw_reset() {
        let config = create_test_config(64);
        let mut index = BinaryHnswIndex::new(&config).unwrap();

        // Add some data
        index.train(&[0u8; 80]).unwrap();
        index.add(&[0u8; 80], None).unwrap();
        assert_eq!(index.len(), 10);

        // Reset
        index.reset();
        assert_eq!(index.len(), 0);
        assert!(!index.trained);
        assert!(index.is_empty());
    }

    #[test]
    fn test_binary_hnsw_save_load() {
        let config = create_test_config(64);
        let mut index = BinaryHnswIndex::new(&config).unwrap();

        let train_data = vec![0u8; 120 * 8];
        index.train(&train_data).unwrap();

        let mut vectors = Vec::new();
        let ids: Vec<i64> = (1000..1120).collect();
        for i in 0..120 {
            let mut v = vec![0u8; 8];
            v[0] = i as u8;
            v[1] = (i * 3) as u8;
            vectors.extend(v);
        }
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vec![7u8; 8];
        let before = index.search(&query, 10);

        let path = Path::new("/tmp/binary_hnsw_test.bin");
        let _ = std::fs::remove_file(path);
        index.save(path).unwrap();

        let loaded = BinaryHnswIndex::load(path, 64).unwrap();
        let after = loaded.search(&query, 10);

        assert_eq!(before.ids, after.ids);
        assert_eq!(before.distances, after.distances);
    }
}

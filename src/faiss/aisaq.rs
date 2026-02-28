//! AISAQ Index (Adaptive Iterative Scalar Adaptive Quantization)
//!
//! A DiskANN-based index with adaptive scalar quantization.
//! Reference: C++ knowhere `INDEX_AISAQ`
//!
//! Features:
//! - Vamana graph algorithm with beam search
//! - PQ compression support
//! - Multiple entry points (medoids)
//! - Cache-aware search
//! - L2 and IP (inner product) distance

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::sync::{Arc, RwLock};

use crate::api::{IndexConfig, MetricType, Result, KnowhereError, SearchResult};
use crate::faiss::pq::PqEncoder;

/// AISAQ configuration parameters
/// Reference: C++ knowhere `AisaqConfig`
#[derive(Clone, Debug)]
pub struct AisaqConfig {
    /// Graph degree (max neighbors per node), typically 48-150
    pub max_degree: usize,
    /// Search list size (L parameter for Vamana), typically 75-200
    pub search_list_size: usize,
    /// Beamwidth for search (IO parallelism), default 8
    pub beamwidth: usize,
    /// PQ vector beam width, default 1
    pub vectors_beamwidth: usize,
    /// PQ code budget in GB (for compression)
    pub pq_code_budget_gb: f32,
    /// Build DRAM budget in GB
    pub build_dram_budget_gb: f32,
    /// Disk PQ dimensions (0 = uncompressed)
    pub disk_pq_dims: usize,
    /// Cache DRAM budget in bytes
    pub pq_cache_size: usize,
    /// Enable compressed vectors reordering search optimization
    pub rearrange: bool,
    /// Number of entry points (medoids)
    pub num_entry_points: usize,
    /// Inline PQ vectors (stored within nodes)
    pub inline_pq: usize,
    /// Warm-up before search
    pub warm_up: bool,
    /// Filter threshold for PQ+Refine (0.0-1.0, -1 = auto)
    pub filter_threshold: f32,
}

impl Default for AisaqConfig {
    fn default() -> Self {
        Self {
            max_degree: 48,
            search_list_size: 128,
            beamwidth: 8,
            vectors_beamwidth: 1,
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            pq_cache_size: 0,
            rearrange: false,
            num_entry_points: 1,
            inline_pq: 0,
            warm_up: false,
            filter_threshold: -1.0,
        }
    }
}

impl AisaqConfig {
    pub fn from_index_config(config: &IndexConfig) -> Self {
        let params = &config.params;
        Self {
            max_degree: params.max_degree.unwrap_or(48),
            search_list_size: params.search_list_size.unwrap_or(128),
            beamwidth: params.beamwidth.unwrap_or(8),
            vectors_beamwidth: 1,
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            pq_cache_size: 0,
            rearrange: false,
            num_entry_points: 1,
            inline_pq: 0,
            warm_up: false,
            filter_threshold: -1.0,
        }
    }
}

/// Statistics about the AISAQ index
#[derive(Debug, Clone, Default)]
pub struct AisaqStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub is_trained: bool,
    pub num_entry_points: usize,
    pub memory_usage_bytes: usize,
}

/// Node in the Vamana graph
#[derive(Clone)]
struct GraphNode {
    /// Vector data
    data: Vec<f32>,
    /// PQ compressed code (optional)
    pq_code: Option<Vec<u8>>,
    /// Neighbors in the graph
    neighbors: Vec<usize>,
    /// Layer number (for multi-layer graphs)
    layer: usize,
}

impl GraphNode {
    fn new(data: Vec<f32>, layer: usize) -> Self {
        Self {
            data,
            pq_code: None,
            neighbors: Vec::new(),
            layer,
        }
    }
}

/// AISAQ Index (Adaptive Iterative Scalar Adaptive Quantization)
pub struct AisaqIndex {
    config: AisaqConfig,
    metric_type: MetricType,
    dim: usize,
    nodes: Vec<GraphNode>,
    entry_points: Vec<usize>,
    pq_encoder: Option<PqEncoder>,
    is_trained: bool,
    cache: Vec<usize>,
}

impl AisaqIndex {
    /// Create a new AISAQ index
    pub fn new(config: AisaqConfig, metric_type: MetricType, dim: usize) -> Self {
        Self {
            config,
            metric_type,
            dim,
            nodes: Vec::new(),
            entry_points: Vec::new(),
            pq_encoder: None,
            is_trained: false,
            cache: Vec::new(),
        }
    }

    /// Train the index (build PQ encoder and initialize graph)
    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        let num_vectors = data.len() / self.dim;
        if num_vectors == 0 {
            return Err(KnowhereError::InvalidArg("No data to train".to_string()));
        }

        // Build PQ encoder if disk_pq_dims > 0
        if self.config.disk_pq_dims > 0 && num_vectors >= 256 {
            let m = self.config.disk_pq_dims.min(self.dim / 2);
            let k = 256; // 8 bits per sub-vector
            let mut pq_encoder = PqEncoder::new(self.dim, m, k);
            pq_encoder.train(data, 50);
            self.pq_encoder = Some(pq_encoder);
        }

        // Select entry points (medoids) - simplified: use first k vectors
        self.entry_points.clear();
        let num_entry_points = self.config.num_entry_points.min(num_vectors);
        for i in 0..num_entry_points {
            self.entry_points.push(i);
        }

        self.is_trained = true;
        Ok(())
    }

    /// Add vectors to the index
    pub fn add(&mut self, data: &[f32]) -> Result<()> {
        if !self.is_trained && !data.is_empty() {
            self.train(data)?;
        }

        let num_vectors = data.len() / self.dim;
        let base_id = self.nodes.len();

        for i in 0..num_vectors {
            let offset = i * self.dim;
            let vec = data[offset..offset + self.dim].to_vec();
            
            let mut node = GraphNode::new(vec, 0);
            
            // Encode with PQ if available
            if let Some(ref pq_encoder) = self.pq_encoder {
                let pq_code = pq_encoder.encode(&node.data);
                node.pq_code = Some(pq_code);
            }

            // Connect to nearest existing nodes (simplified Vamana)
            if !self.nodes.is_empty() {
                let neighbors = self.find_nearest_nodes(&node.data, self.config.max_degree);
                node.neighbors = neighbors;
            }

            self.nodes.push(node);
        }

        // Update graph connections (bidirectional)
        self.update_graph_connections(base_id, num_vectors);

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        if !self.is_trained || self.nodes.is_empty() {
            return Err(KnowhereError::InternalError("AISAQ index not trained".to_string()));
        }

        if query.len() != self.dim {
            return Err(KnowhereError::InvalidArg(
                format!("Query dimension {} != index dimension {}", query.len(), self.dim)
            ));
        }

        let k = k.min(self.nodes.len());
        let results = self.beam_search(query, k, self.config.beamwidth)?;

        // Refine with exact distances if PQ was used
        let mut refined_results = results;
        if self.pq_encoder.is_some() {
            self.refine_results(query, &mut refined_results);
        }

        Ok(SearchResult::new(
            refined_results.iter().map(|(id, _)| *id as i64).collect(),
            refined_results.iter().map(|(_, dist)| *dist).collect(),
            0.0,
        ))
    }

    /// Beam search algorithm
    fn beam_search(&self, query: &[f32], k: usize, beamwidth: usize) -> Result<Vec<(usize, f32)>> {
        // Use entry points as starting points
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        // Initialize with entry points
        for &entry_point in &self.entry_points {
            if entry_point < self.nodes.len() {
                let dist = self.compute_distance(query, &self.nodes[entry_point].data);
                candidates.push(Candidate { id: entry_point, dist });
                visited.insert(entry_point);
            }
        }

        // Beam search
        while let Some(candidate) = candidates.pop() {
            // Add to results
            results.push(Candidate {
                id: candidate.id,
                dist: candidate.dist,
            });

            // Keep only k best results
            while results.len() > k {
                results.pop();
            }

            // Expand neighbors
            if candidate.id < self.nodes.len() {
                for &neighbor in &self.nodes[candidate.id].neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        let dist = self.compute_distance(query, &self.nodes[neighbor].data);
                        candidates.push(Candidate { id: neighbor, dist });
                    }
                }
            }

            // Limit beam width
            while candidates.len() > beamwidth * k {
                candidates.pop();
            }
        }

        // Extract results
        let mut result_vec: Vec<(usize, f32)> = results
            .into_iter()
            .map(|c| (c.id, c.dist))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec.truncate(k);

        Ok(result_vec)
    }

    /// Find nearest nodes for a new vector
    fn find_nearest_nodes(&self, data: &[f32], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = self.nodes
            .iter()
            .enumerate()
            .map(|(id, node)| (id, self.compute_distance(data, &node.data)))
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances.truncate(k.min(self.nodes.len()));
        distances.iter().map(|(id, _)| *id).collect()
    }

    /// Update graph connections (bidirectional links)
    fn update_graph_connections(&mut self, base_id: usize, num_vectors: usize) {
        // Make connections bidirectional
        for i in 0..num_vectors {
            let node_id = base_id + i;
            if node_id >= self.nodes.len() {
                continue;
            }
            
            let neighbors: Vec<usize> = self.nodes[node_id].neighbors.clone();
            for &neighbor in &neighbors {
                if neighbor < self.nodes.len() {
                    if !self.nodes[neighbor].neighbors.contains(&node_id) {
                        self.nodes[neighbor].neighbors.push(node_id);
                    }
                }
            }
        }
    }

    /// Refine results with exact distances
    fn refine_results(&self, query: &[f32], results: &mut Vec<(usize, f32)>) {
        for (id, dist) in results.iter_mut() {
            if *id < self.nodes.len() {
                *dist = self.compute_distance(query, &self.nodes[*id].data);
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    }

    /// Compute distance based on metric type
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric_type {
            MetricType::L2 => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            MetricType::Ip => {
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            MetricType::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
            MetricType::Hamming => {
                // Not typically used for AISAQ (float vectors)
                0.0
            }
        }
    }

    /// Get vector by ID
    pub fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(ids.len());
        for &id in ids {
            if id < 0 || id as usize >= self.nodes.len() {
                return Err(KnowhereError::InvalidArg(
                    format!("Invalid vector ID: {}", id)
                ));
            }
            vectors.push(self.nodes[id as usize].data.clone());
        }
        Ok(vectors)
    }

    /// Get number of vectors
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get statistics
    pub fn stats(&self) -> AisaqStats {
        let num_nodes = self.nodes.len();
        let num_edges: usize = self.nodes.iter().map(|n| n.neighbors.len()).sum();
        let degrees: Vec<usize> = self.nodes.iter().map(|n| n.neighbors.len()).collect();
        
        AisaqStats {
            num_nodes,
            num_edges,
            avg_degree: if num_nodes > 0 { num_edges as f32 / num_nodes as f32 } else { 0.0 },
            max_degree: degrees.iter().copied().max().unwrap_or(0),
            min_degree: degrees.iter().copied().min().unwrap_or(0),
            is_trained: self.is_trained,
            num_entry_points: self.entry_points.len(),
            memory_usage_bytes: num_nodes * (self.dim * std::mem::size_of::<f32>() + 24),
        }
    }
}

/// Candidate for beam search (max-heap, so we negate distance)
#[derive(PartialEq)]
struct Candidate {
    id: usize,
    dist: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (we want smallest distance first)
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aisaq_new() {
        let config = AisaqConfig::default();
        let index = AisaqIndex::new(config, MetricType::L2, 128);
        
        assert_eq!(index.count(), 0);
        assert_eq!(index.dim(), 128);
        assert!(!index.stats().is_trained);
    }

    #[test]
    fn test_aisaq_train_and_search() {
        let config = AisaqConfig {
            max_degree: 32,
            search_list_size: 64,
            ..Default::default()
        };
        let mut index = AisaqIndex::new(config, MetricType::L2, 4);
        
        // Generate test data
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        
        // Train and add
        index.add(&data).expect("Failed to add data");
        
        // Search
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let results = index.search(&query, 5).expect("Search failed");
        
        assert_eq!(results.ids.len(), 5);
        assert_eq!(results.distances.len(), 5);
    }

    #[test]
    fn test_aisaq_beam_search() {
        let config = AisaqConfig {
            beamwidth: 4,
            ..Default::default()
        };
        let mut index = AisaqIndex::new(config, MetricType::L2, 8);
        
        // Add some data
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        index.add(&data).expect("Failed to add data");
        
        // Search with different beamwidths
        let query = vec![1.0; 8];
        
        let results_bw4 = index.search(&query, 3).expect("Search failed");
        assert_eq!(results_bw4.ids.len(), 3);
    }

    #[test]
    fn test_aisaq_get_vectors() {
        let config = AisaqConfig::default();
        let mut index = AisaqIndex::new(config, MetricType::L2, 3);
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        index.add(&data).expect("Failed to add data");
        
        let vectors = index.get_vector_by_ids(&[0, 1]).expect("Get vectors failed");
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(vectors[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_aisaq_metrics() {
        // Test L2
        let mut index_l2 = AisaqIndex::new(AisaqConfig::default(), MetricType::L2, 2);
        let data = vec![0.0, 0.0, 3.0, 4.0];
        index_l2.add(&data).unwrap();
        let result_l2 = index_l2.search(&[0.0, 0.0], 1).unwrap();
        assert!(!result_l2.distances.is_empty());

        // Test IP
        let mut index_ip = AisaqIndex::new(AisaqConfig::default(), MetricType::Ip, 2);
        index_ip.add(&data).unwrap();
        let result_ip = index_ip.search(&[1.0, 1.0], 1).unwrap();
        assert!(!result_ip.distances.is_empty());
    }
}

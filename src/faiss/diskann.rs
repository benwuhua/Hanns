//! DiskANN-inspired index (Vamana algorithm) - Enhanced
//! 
//! A graph-based index optimized for SSD storage.
//! Features:
//! - Vamana graph algorithm with beam search
//! - L2 and IP (inner product) distance
//! - Beam search with early termination
//! - Range search support
//! - Iterator support for streaming search
//! - PQ compression support (simplified)
//! - Configurable parameters (max_degree, search_list_size, beamwidth, etc.)
//! - Statistics API for monitoring

use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::cmp::Ordering;

use crate::api::{IndexConfig, MetricType, Result, SearchRequest, SearchResult};
use crate::simd;

/// DiskANN configuration parameters
/// Mirrors the C++ DiskANNConfig structure
#[derive(Clone, Debug)]
pub struct DiskAnnConfig {
    /// Graph degree (max neighbors per node), typically 48-150
    pub max_degree: usize,
    /// Search list size (L parameter for Vamana), typically 75-200
    pub search_list_size: usize,
    /// PQ code budget in GB (for compression)
    pub pq_code_budget_gb: f32,
    /// Build DRAM budget in GB
    pub build_dram_budget_gb: f32,
    /// Disk PQ dimensions (0 = uncompressed)
    pub disk_pq_dims: usize,
    /// Beamwidth for search (IO parallelism), default 8
    pub beamwidth: usize,
    /// Cache DRAM budget in GB
    pub cache_dram_budget_gb: f32,
    /// Warm-up before search
    pub warm_up: bool,
    /// Filter threshold for PQ+Refine (0.0-1.0, -1 = auto)
    pub filter_threshold: f32,
    /// Accelerate build (skip full 2-round build)
    pub accelerate_build: bool,
    /// Min K for range search
    pub min_k: usize,
    /// Max K for range search
    pub max_k: usize,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            max_degree: 48,
            search_list_size: 128,
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            beamwidth: 8,
            cache_dram_budget_gb: 0.0,
            warm_up: false,
            filter_threshold: -1.0,
            accelerate_build: false,
            min_k: 100,
            max_k: usize::MAX,
        }
    }
}

impl DiskAnnConfig {
    pub fn from_index_config(config: &IndexConfig) -> Self {
        let params = &config.params;
        Self {
            max_degree: params.max_degree.unwrap_or(48),
            search_list_size: params.search_list_size.unwrap_or(128),
            beamwidth: params.beamwidth.unwrap_or(8),
            pq_code_budget_gb: 0.0,  // Not exposed in IndexParams yet
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            cache_dram_budget_gb: 0.0,
            warm_up: false,
            filter_threshold: -1.0,
            accelerate_build: false,
            min_k: 100,
            max_k: usize::MAX,
        }
    }
}

/// Statistics about the DiskANN index
#[derive(Debug, Clone, Default)]
pub struct DiskAnnStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub is_trained: bool,
    pub memory_usage_bytes: usize,
}

/// PQ-compressed vector (simplified)
#[derive(Clone)]
struct PQCode {
    codes: Vec<u8>,
    dims: usize,
}

impl PQCode {
    fn new(dims: usize) -> Self {
        Self {
            codes: Vec::new(),
            dims,
        }
    }
    
    fn encode(&mut self, vectors: &[f32], dim: usize) {
        if self.dims == 0 {
            return; // No compression
        }
        
        // Simplified PQ: just quantize to u8 per sub-dimension
        let num_subdims = self.dims.min(dim);
        let subdim_size = dim / num_subdims;
        
        for vec in vectors.chunks(dim) {
            for i in 0..num_subdims {
                let start = i * subdim_size;
                let end = start + subdim_size;
                let subvec = &vec[start..end.min(dim)];
                
                // Simple mean quantization
                let mean = subvec.iter().sum::<f32>() / subvec.len() as f32;
                let code = ((mean + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
                self.codes.push(code);
            }
        }
    }
    
    fn distance(&self, query: &[f32], idx: usize, dim: usize) -> f32 {
        if self.dims == 0 || self.codes.is_empty() {
            return f32::MAX; // Fallback to full distance
        }
        
        let num_subdims = self.dims.min(dim);
        let subdim_size = dim / num_subdims;
        let mut dist = 0.0f32;
        
        for i in 0..num_subdims {
            let start = i * subdim_size;
            let end = start + subdim_size;
            let subvec = &query[start..end.min(dim)];
            
            let mean = subvec.iter().sum::<f32>() / subvec.len() as f32;
            let code_val = self.codes[idx * num_subdims + i] as f32 / 127.5 - 1.0;
            
            dist += (mean - code_val).powi(2);
        }
        
        dist
    }
}

/// DiskANN-style graph index (Vamana) - Enhanced
pub struct DiskAnnIndex {
    config: IndexConfig,
    dann_config: DiskAnnConfig,
    dim: usize,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    graph: Vec<Vec<(i64, f32)>>,
    next_id: i64,
    trained: bool,
    alpha: f32,
    entry_point: Option<usize>,
    /// PQ compression (optional)
    pq_codes: Option<PQCode>,
    /// Cached nodes for faster search
    cached_nodes: HashSet<usize>,
}

impl DiskAnnIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let dann_config = DiskAnnConfig::from_index_config(config);

        Ok(Self {
            config: config.clone(),
            dann_config,
            dim: config.dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            next_id: 0,
            trained: false,
            alpha: 1.2,
            entry_point: None,
            pq_codes: None,
            cached_nodes: HashSet::new(),
        })
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        
        self.vectors.reserve(n * self.dim);
        self.graph.reserve(n);
        
        // Store vectors
        for i in 0..n {
            let start = i * self.dim;
            self.vectors.extend_from_slice(&vectors[start..start + self.dim]);
            self.ids.push(i as i64);
        }
        
        self.next_id = n as i64;
        
        // Build graph with Vamana algorithm
        self.build_vamana_graph();
        
        // Build PQ codes if configured
        if self.dann_config.disk_pq_dims > 0 {
            self.build_pq_codes(vectors);
        }
        
        // Warm-up if configured
        if self.dann_config.warm_up {
            self.warm_up();
        }
        
        self.trained = true;
        tracing::info!("Built DiskANN graph with {} nodes, max_degree={}, search_list={}", 
                      n, self.dann_config.max_degree, self.dann_config.search_list_size);
        Ok(())
    }

    /// Build PQ codes for compression
    fn build_pq_codes(&mut self, vectors: &[f32]) {
        let mut pq = PQCode::new(self.dann_config.disk_pq_dims);
        pq.encode(vectors, self.dim);
        self.pq_codes = Some(pq);
    }

    /// Warm-up: cache frequently accessed nodes
    fn warm_up(&mut self) {
        // Simple warm-up: cache entry point and its neighbors
        if let Some(entry) = self.entry_point {
            self.cached_nodes.insert(entry);
            if let Some(nbrs) = self.graph.get(entry) {
                for &(id, _) in nbrs {
                    if (id as usize) < self.ids.len() {
                        self.cached_nodes.insert(id as usize);
                    }
                }
            }
        }
    }

    /// Build Vamana graph using the proper algorithm
    fn build_vamana_graph(&mut self) {
        let n = self.ids.len();
        if n == 0 { return; }
        
        let L = self.dann_config.search_list_size;
        let R = self.dann_config.max_degree;
        
        // Sort vectors by first dimension for better entry point selection
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.vectors[a * self.dim]
                .partial_cmp(&self.vectors[b * self.dim])
                .unwrap_or(Ordering::Equal)
        });
        
        // Build graph incrementally (Vamana style)
        let mut current_graph: Vec<Vec<(i64, f32)>> = Vec::with_capacity(n);
        
        // First node is entry point
        self.entry_point = Some(0);
        current_graph.push(Vec::new());
        
        // Insert remaining nodes one by one
        for i in 1..n {
            let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
            
            // Search for neighbors using current graph
            let neighbors = self.vamana_search(query, L, R, &current_graph);
            
            // Add bidirectional edges
            let mut node_neighbors: Vec<(i64, f32)> = neighbors
                .iter()
                .map(|&(idx, dist)| (self.ids[idx], dist))
                .collect();
            
            // Prune to max_degree using Vamana pruning
            node_neighbors = self.prune_neighbors(i, &node_neighbors, R);
            
            current_graph.push(node_neighbors);
            
            // Add reverse edges
            for &(idx, dist) in &neighbors {
                if idx < current_graph.len() {
                    current_graph[idx].push((self.ids[i], dist));
                    // Prune reverse edges too
                    current_graph[idx] = self.prune_neighbors(idx, &current_graph[idx], R);
                }
            }
        }
        
        self.graph = current_graph;
        
        // Second pass for better connectivity (unless accelerate_build)
        if !self.dann_config.accelerate_build {
            self.refine_graph();
        }
    }

    /// Vamana search for finding neighbors during build
    fn vamana_search(&self, query: &[f32], L: usize, R: usize, graph: &[Vec<(i64, f32)>]) -> Vec<(usize, f32)> {
        let mut visited = vec![false; self.ids.len()];
        let mut candidates: BinaryHeap<ReverseOrderedFloat> = BinaryHeap::new();
        let mut results: Vec<(f32, usize)> = Vec::new();
        
        // Start from entry point
        if let Some(entry) = self.entry_point {
            let dist = self.compute_dist(query, entry);
            candidates.push(ReverseOrderedFloat(dist, entry));
            visited[entry] = true;
        }
        
        // Beam search
        let beam_size = self.dann_config.beamwidth;
        
        while !candidates.is_empty() && results.len() < L {
            // Get best candidate
            let ReverseOrderedFloat(dist, idx) = candidates.pop().unwrap();
            results.push((dist, idx));
            
            // Explore neighbors (beam search)
            let mut nbr_dists: Vec<(f32, usize)> = Vec::new();
            
            if let Some(nbrs) = graph.get(idx) {
                for &(nbr_id, _) in nbrs {
                    let n_idx = nbr_id as usize;
                    if n_idx < visited.len() && !visited[n_idx] {
                        let d = self.compute_dist(query, n_idx);
                        nbr_dists.push((d, n_idx));
                    }
                }
            }
            
            // Sort and add best beamwidth neighbors
            nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (d, n_idx) in nbr_dists.into_iter().take(beam_size) {
                visited[n_idx] = true;
                candidates.push(ReverseOrderedFloat(d, n_idx));
            }
        }
        
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.into_iter()
            .map(|(d, idx)| (idx, d))
            .collect()
    }

    /// Prune neighbors using Vamana pruning strategy
    fn prune_neighbors(&self, node_idx: usize, neighbors: &[(i64, f32)], R: usize) -> Vec<(i64, f32)> {
        if neighbors.len() <= R {
            return neighbors.to_vec();
        }
        
        // Sort by distance
        let mut sorted: Vec<(i64, f32)> = neighbors.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Vamana pruning: keep diverse neighbors
        let mut selected: Vec<(i64, f32)> = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();
        
        for &(id, dist) in &sorted {
            if selected.len() >= R {
                break;
            }
            
            let idx = id as usize;
            if idx >= self.vectors.len() / self.dim {
                continue;
            }
            
            // Check angular diversity
            let vec = &self.vectors[idx * self.dim..(idx + 1) * self.dim];
            let mut is_diverse = true;
            
            for &(sel_id, _) in &selected {
                let sel_idx = sel_id as usize;
                if sel_idx < self.vectors.len() / self.dim {
                    let sel_vec = &self.vectors[sel_idx * self.dim..(sel_idx + 1) * self.dim];
                    let angle = self.cosine_similarity(vec, sel_vec);
                    if angle > self.alpha {
                        is_diverse = false;
                        break;
                    }
                }
            }
            
            if is_diverse {
                selected.push((id, dist));
            }
        }
        
        // If still not enough, just take closest
        if selected.len() < R.min(neighbors.len()) {
            selected = sorted.into_iter().take(R).collect();
        }
        
        selected
    }

    /// Refine graph with second pass
    fn refine_graph(&mut self) {
        let n = self.ids.len();
        let R = self.dann_config.max_degree;
        
        // For each node, search again and update edges
        for i in 0..n {
            let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
            let neighbors = self.vamana_search(query, self.dann_config.search_list_size, R, &self.graph);
            
            let new_neighbors: Vec<(i64, f32)> = neighbors
                .iter()
                .filter(|&&(idx, _)| idx != i)
                .map(|&(idx, dist)| (self.ids[idx], dist))
                .collect();
            
            if !new_neighbors.is_empty() {
                self.graph[i] = self.prune_neighbors(i, &new_neighbors, R);
            }
        }
    }

    #[inline]
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a < 1e-6 || norm_b < 1e-6 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    #[inline]
    fn l2_sqr(&self, a: &[f32], b_idx: usize) -> f32 {
        let start = b_idx * self.dim;
        let b = &self.vectors[start..start + self.dim];
        
        // Use SIMD for distance, then square
        let dist = simd::l2_distance(a, b);
        dist * dist
    }

    #[inline]
    fn ip_distance(&self, a: &[f32], b_idx: usize) -> f32 {
        let start = b_idx * self.dim;
        let b = &self.vectors[start..start + self.dim];
        
        // For IP, higher is better, so we return negative (for consistent sorting)
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            sum += a[i] * b[i];
        }
        -sum // Negative so that max becomes min in sorting
    }

    /// Compute distance based on metric type
    #[inline]
    fn compute_dist(&self, query: &[f32], idx: usize) -> f32 {
        match self.config.metric_type {
            MetricType::L2 => self.l2_sqr(query, idx),
            MetricType::Ip => self.ip_distance(query, idx),
            _ => self.l2_sqr(query, idx),
        }
    }

    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            let start = i * self.dim;
            self.vectors.extend_from_slice(&vectors[start..start + self.dim]);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            self.ids.push(id);
            self.graph.push(Vec::new());
        }
        
        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n_queries = query.len() / self.dim;
        if n_queries * self.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let L = req.nprobe.max(self.dann_config.search_list_size / 2);
        let k = req.top_k;
        
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            // Use improved beam search
            let results = self.beam_search(query_vec, L);
            
            for i in 0..k {
                if i < results.len() {
                    all_ids.push(results[i].0);
                    // For L2: sqrt the squared distance; For IP: negate to get positive similarity
                    let dist = match self.config.metric_type {
                        MetricType::Ip => -results[i].1, // Convert back to positive similarity
                        _ => results[i].1.sqrt(),
                    };
                    all_dists.push(dist);
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    /// Beam search with early termination
    fn beam_search(&self, query: &[f32], L: usize) -> Vec<(i64, f32)> {
        let n = self.ids.len();
        if n == 0 { return vec![]; }
        
        let beamwidth = self.dann_config.beamwidth;
        let mut visited = vec![false; n];
        let mut candidates: BinaryHeap<ReverseOrderedFloat> = BinaryHeap::new();
        let mut results: Vec<(f32, usize)> = Vec::new();
        
        // Start from entry point or node 0
        let start = self.entry_point.unwrap_or(0);
        let dist = self.compute_dist(query, start);
        candidates.push(ReverseOrderedFloat(dist, start));
        visited[start] = true;
        
        // Early termination tracking
        let mut no_progress_count = 0;
        let mut best_dist = dist;
        
        // Beam search loop
        while !candidates.is_empty() && results.len() < L {
            let ReverseOrderedFloat(dist, idx) = candidates.pop().unwrap();
            results.push((dist, idx));
            
            // Check for early termination
            if dist < best_dist * 0.99 {
                best_dist = dist;
                no_progress_count = 0;
            } else {
                no_progress_count += 1;
            }
            
            // Early termination: stop if no progress
            if no_progress_count > L / 4 {
                break;
            }
            
            // Explore neighbors with beamwidth limit
            if let Some(nbrs) = self.graph.get(idx) {
                let mut nbr_dists: Vec<(f32, usize)> = Vec::new();
                
                for &(nbr_id, _) in nbrs {
                    let n_idx = nbr_id as usize;
                    if n_idx < n && !visited[n_idx] {
                        // Use PQ distance if available and node is not cached
                        let d = if self.pq_codes.is_some() && !self.cached_nodes.contains(&n_idx) {
                            self.pq_codes.as_ref().unwrap().distance(query, n_idx, self.dim)
                        } else {
                            self.compute_dist(query, n_idx)
                        };
                        nbr_dists.push((d, n_idx));
                    }
                }
                
                // Sort and add best beamwidth neighbors
                nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (d, n_idx) in nbr_dists.into_iter().take(beamwidth) {
                    visited[n_idx] = true;
                    candidates.push(ReverseOrderedFloat(d, n_idx));
                }
            }
        }
        
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(L);
        
        results.into_iter()
            .map(|(d, idx)| (self.ids[idx], d))
            .collect()
    }

    /// Range search: find all vectors within radius
    pub fn range_search(&self, query: &[f32], radius: f32, max_results: usize) -> Result<Vec<(i64, f32)>> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n = self.ids.len();
        let mut results: Vec<(i64, f32)> = Vec::new();
        let mut visited = vec![false; n];
        let mut candidates: VecDeque<(usize, f32)> = VecDeque::new();
        
        // Start from entry point
        let start = self.entry_point.unwrap_or(0);
        let start_dist = self.compute_dist(query, start);
        
        if start_dist <= radius * radius {
            results.push((self.ids[start], start_dist.sqrt()));
        }
        
        candidates.push_back((start, start_dist));
        visited[start] = true;
        
        // BFS-style range search
        let beamwidth = self.dann_config.beamwidth;
        
        while !candidates.is_empty() && results.len() < max_results {
            let (idx, dist) = candidates.pop_front().unwrap();
            
            // Explore neighbors
            if let Some(nbrs) = self.graph.get(idx) {
                let mut nbr_dists: Vec<(f32, usize)> = Vec::new();
                
                for &(nbr_id, _) in nbrs {
                    let n_idx = nbr_id as usize;
                    if n_idx < n && !visited[n_idx] {
                        let d = self.compute_dist(query, n_idx);
                        if d <= radius * radius {
                            results.push((self.ids[n_idx], d.sqrt()));
                            nbr_dists.push((d, n_idx));
                        }
                        visited[n_idx] = true;
                    }
                }
                
                // Continue searching from closest neighbors
                nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (d, n_idx) in nbr_dists.into_iter().take(beamwidth) {
                    candidates.push_back((n_idx, d));
                }
            }
        }
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    /// Create an iterator for streaming search results
    pub fn search_iterator<'a>(&'a self, query: &'a [f32], L: usize) -> DiskAnnIterator<'a> {
        DiskAnnIterator::new(self, query, L)
    }

    /// Get statistics about the index
    pub fn get_stats(&self) -> DiskAnnStats {
        let n = self.ids.len();
        let mut num_edges = 0;
        let mut max_deg = 0;
        let mut min_deg = usize::MAX;
        
        for nbrs in &self.graph {
            let deg = nbrs.len();
            num_edges += deg;
            max_deg = max_deg.max(deg);
            min_deg = min_deg.min(deg);
        }
        
        if n == 0 {
            min_deg = 0;
        }
        
        let avg_deg = if n > 0 { num_edges as f32 / n as f32 } else { 0.0 };
        
        // Estimate memory usage
        let memory = self.vectors.len() * 4  // vectors
            + self.ids.len() * 8              // ids
            + num_edges * 12                  // graph edges (8 byte id + 4 byte dist)
            + self.pq_codes.as_ref().map(|p| p.codes.len()).unwrap_or(0); // PQ codes
        
        DiskAnnStats {
            num_nodes: n,
            num_edges,
            avg_degree: avg_deg,
            max_degree: max_deg,
            min_degree: min_deg,
            is_trained: self.trained,
            memory_usage_bytes: memory,
        }
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        file.write_all(b"DANN")?;
        file.write_all(&2u32.to_le_bytes())?; // Version 2
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        
        // Write config
        file.write_all(&(self.dann_config.max_degree as u32).to_le_bytes())?;
        file.write_all(&(self.dann_config.search_list_size as u32).to_le_bytes())?;
        file.write_all(&(self.dann_config.beamwidth as u32).to_le_bytes())?;
        
        let bytes: Vec<u8> = self.vectors.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
        
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        for neighbors in &self.graph {
            file.write_all(&(neighbors.len() as u32).to_le_bytes())?;
            for &(id, dist) in neighbors {
                file.write_all(&id.to_le_bytes())?;
                file.write_all(&dist.to_le_bytes())?;
            }
        }
        
        // Write PQ codes if present
        let has_pq = self.pq_codes.is_some() as u32;
        file.write_all(&has_pq.to_le_bytes())?;
        if let Some(pq) = &self.pq_codes {
            file.write_all(&(pq.dims as u32).to_le_bytes())?;
            file.write_all(&(pq.codes.len() as u64).to_le_bytes())?;
            file.write_all(&pq.codes)?;
        }
        
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"DANN" {
            return Err(crate::api::KnowhereError::Codec("invalid magic".to_string()));
        }
        
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        if dim != self.dim {
            return Err(crate::api::KnowhereError::Codec("dimension mismatch".to_string()));
        }
        
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Read config (version 2+)
        if version >= 2 {
            let mut md_bytes = [0u8; 4];
            file.read_exact(&mut md_bytes)?;
            self.dann_config.max_degree = u32::from_le_bytes(md_bytes) as usize;
            
            file.read_exact(&mut md_bytes)?;
            self.dann_config.search_list_size = u32::from_le_bytes(md_bytes) as usize;
            
            file.read_exact(&mut md_bytes)?;
            self.dann_config.beamwidth = u32::from_le_bytes(md_bytes) as usize;
        }
        
        let mut vec_bytes = vec![0u8; count * dim * 4];
        file.read_exact(&mut vec_bytes)?;
        
        self.vectors.clear();
        for chunk in vec_bytes.chunks(4) {
            self.vectors.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        
        self.ids.clear();
        for _ in 0..count {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            self.ids.push(i64::from_le_bytes(id_bytes));
        }
        
        self.graph.clear();
        for _ in 0..count {
            let mut nc_bytes = [0u8; 4];
            file.read_exact(&mut nc_bytes)?;
            let nc = u32::from_le_bytes(nc_bytes) as usize;
            
            let mut neighbors = Vec::with_capacity(nc);
            for _ in 0..nc {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                let id = i64::from_le_bytes(id_bytes);
                
                let mut d_bytes = [0u8; 4];
                file.read_exact(&mut d_bytes)?;
                let dist = f32::from_le_bytes(d_bytes);
                
                neighbors.push((id, dist));
            }
            self.graph.push(neighbors);
        }
        
        // Read PQ codes (version 2+)
        if version >= 2 {
            let mut has_pq_bytes = [0u8; 4];
            file.read_exact(&mut has_pq_bytes)?;
            if u32::from_le_bytes(has_pq_bytes) != 0 {
                let mut dims_bytes = [0u8; 4];
                file.read_exact(&mut dims_bytes)?;
                let dims = u32::from_le_bytes(dims_bytes) as usize;
                
                let mut len_bytes = [0u8; 8];
                file.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;
                
                let mut codes = vec![0u8; len];
                file.read_exact(&mut codes)?;
                
                self.pq_codes = Some(PQCode { codes, dims });
            }
        }
        
        // Set entry point to first node
        if count > 0 {
            self.entry_point = Some(0);
        }
        
        self.trained = true;
        Ok(())
    }
}

/// Helper struct for priority queue (min-heap)
#[derive(Debug)]
struct ReverseOrderedFloat(f32, usize);

impl PartialEq for ReverseOrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for ReverseOrderedFloat {}

impl PartialOrd for ReverseOrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseOrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Iterator for streaming DiskANN search results
pub struct DiskAnnIterator<'a> {
    index: &'a DiskAnnIndex,
    results: Vec<(i64, f32)>,
    current: usize,
}

impl<'a> DiskAnnIterator<'a> {
    fn new(index: &'a DiskAnnIndex, query: &'a [f32], L: usize) -> Self {
        let results = index.beam_search(query, L);
        Self {
            index,
            results,
            current: 0,
        }
    }
    
    /// Get next result
    pub fn next(&mut self) -> Option<(i64, f32)> {
        if self.current < self.results.len() {
            let result = self.results[self.current];
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
    
    /// Get remaining count
    pub fn remaining(&self) -> usize {
        self.results.len() - self.current
    }
}

impl<'a> Iterator for DiskAnnIterator<'a> {
    type Item = (i64, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::IndexType;
    
    #[test]
    fn test_diskann() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = DiskAnnIndex::new(&config).unwrap();
        
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
    
    #[test]
    fn test_diskann_save_load() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        // Create and train index
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        
        // Save to temp file
        let temp_path = std::env::temp_dir().join("diskann_test.bin");
        index.save(&temp_path).unwrap();
        
        // Create new index and load
        let mut index2 = DiskAnnIndex::new(&config).unwrap();
        index2.load(&temp_path).unwrap();
        
        // Verify search results match
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result1 = index.search(&query, &req).unwrap();
        let result2 = index2.search(&query, &req).unwrap();
        
        // Results should be identical after reload
        assert_eq!(result1.ids, result2.ids);
        
        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
    
    #[test]
    fn test_diskann_range_search() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let results = index.range_search(&query, 1.0, 10).unwrap();
        
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_diskann_stats() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        
        let stats = index.get_stats();
        
        assert_eq!(stats.num_nodes, 4);
        assert!(stats.num_edges > 0);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.is_trained);
    }
    
    #[test]
    fn test_diskann_iterator() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let mut iter = index.search_iterator(&query, 10);
        
        let mut count = 0;
        while let Some((id, dist)) = iter.next() {
            count += 1;
            assert!(id >= 0);
        }
        
        assert!(count > 0);
    }
    
    #[test]
    fn test_diskann_config() {
        use crate::api::IndexParams;
        
        let params = IndexParams {
            max_degree: Some(64),
            search_list_size: Some(200),
            beamwidth: Some(16),
            ..Default::default()
        };
        
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            params,
        };
        
        let index = DiskAnnIndex::new(&config).unwrap();
        
        assert_eq!(index.dann_config.max_degree, 64);
        assert_eq!(index.dann_config.search_list_size, 200);
        assert_eq!(index.dann_config.beamwidth, 16);
    }
}

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

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::api::{IndexConfig, MetricType, Result, SearchRequest, SearchResult};
use crate::quantization::kmeans::KMeans;
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
    /// Candidate pool expansion ratio for PQ path (100 = off, 125 = +25%)
    pub pq_candidate_expand_pct: usize,
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
            pq_candidate_expand_pct: 125,
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
            pq_code_budget_gb: 0.0, // Not exposed in IndexParams yet
            build_dram_budget_gb: 0.0,
            disk_pq_dims: params.disk_pq_dims.unwrap_or(0),
            pq_candidate_expand_pct: params
                .disk_pq_candidate_expand_pct
                .unwrap_or(125)
                .clamp(100, 300),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskAnnScopeAudit {
    pub dim: usize,
    pub max_degree: usize,
    pub search_list_size: usize,
    pub beamwidth: usize,
    pub uses_vamana_graph: bool,
    pub uses_placeholder_pq: bool,
    pub has_flash_layout: bool,
    pub native_comparable: bool,
    pub comparability_reason: &'static str,
}

/// PQ-compressed vector placeholder.
///
/// This is *not* native DiskANN's full PQ/SSD pipeline: it only stores a coarse
/// per-subvector mean quantization so the Rust implementation can exercise a
/// compressed-distance branch. Treat it as an explicit simplification rather than
/// a performance-comparable implementation.
#[derive(Clone)]
struct PQCode {
    codes: Vec<u8>,
    dims: usize,
    ksub: usize,
    subdim_size: usize,
    codebooks: Vec<f32>, // [num_subdims][ksub][subdim_size]
}

impl PQCode {
    fn new(dims: usize) -> Self {
        Self {
            codes: Vec::new(),
            dims,
            ksub: 0,
            subdim_size: 0,
            codebooks: Vec::new(),
        }
    }

    fn encode(&mut self, vectors: &[f32], dim: usize) {
        if self.dims == 0 {
            return; // No compression
        }

        self.codes.clear();
        self.codebooks.clear();

        let n = vectors.len() / dim;
        if n == 0 {
            return;
        }

        let num_subdims = self.dims.min(dim);
        let subdim_size = dim / num_subdims;
        let ksub = n.min(256).max(1);

        self.ksub = ksub;
        self.subdim_size = subdim_size;
        self.codebooks = vec![0.0; num_subdims * ksub * subdim_size];

        // Train one codebook per sub-dimension.
        for sub_q in 0..num_subdims {
            let mut sub_vectors = Vec::with_capacity(n * subdim_size);
            for vec in vectors.chunks(dim) {
                let start = sub_q * subdim_size;
                let end = (start + subdim_size).min(dim);
                sub_vectors.extend_from_slice(&vec[start..end]);
            }

            let mut kmeans = KMeans::new(ksub, subdim_size);
            kmeans.set_max_iter(16);
            kmeans.train(&sub_vectors);
            let centroid_base = sub_q * ksub * subdim_size;
            self.codebooks[centroid_base..centroid_base + ksub * subdim_size]
                .copy_from_slice(&kmeans.centroids);
        }

        // Encode vectors with nearest centroid id for each sub-dimension.
        for vec in vectors.chunks(dim) {
            for sub_q in 0..num_subdims {
                let start = sub_q * subdim_size;
                let end = (start + subdim_size).min(dim);
                let subvec = &vec[start..end];
                let code = self.find_nearest_centroid(sub_q, subvec) as u8;
                self.codes.push(code);
            }
        }
    }

    fn find_nearest_centroid(&self, sub_q: usize, subvec: &[f32]) -> usize {
        if self.ksub == 0 || self.subdim_size == 0 {
            return 0;
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        let centroid_base = sub_q * self.ksub * self.subdim_size;
        for c in 0..self.ksub {
            let centroid_start = centroid_base + c * self.subdim_size;
            let centroid = &self.codebooks[centroid_start..centroid_start + self.subdim_size];
            let mut dist = 0.0f32;
            for d in 0..self.subdim_size {
                let diff = subvec[d] - centroid[d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }
        best_idx
    }

    fn distance(&self, query: &[f32], idx: usize, dim: usize) -> f32 {
        if self.dims == 0 || self.codes.is_empty() || self.codebooks.is_empty() {
            return f32::MAX; // Fallback to full distance
        }

        let query_table = self.build_query_distance_table(query, dim);
        self.distance_with_table(&query_table, idx, dim)
    }

    /// Build ADC table once per query.
    fn build_query_distance_table(&self, query: &[f32], dim: usize) -> Vec<f32> {
        if self.dims == 0 || self.ksub == 0 || self.subdim_size == 0 || self.codebooks.is_empty() {
            return Vec::new();
        }

        let num_subdims = self.dims.min(dim);
        let mut table = vec![0.0f32; num_subdims * self.ksub];

        for sub_q in 0..num_subdims {
            let start = sub_q * self.subdim_size;
            let end = (start + self.subdim_size).min(dim);
            let query_sub = &query[start..end];
            let centroid_base = sub_q * self.ksub * self.subdim_size;
            let table_base = sub_q * self.ksub;

            for c in 0..self.ksub {
                let centroid_start = centroid_base + c * self.subdim_size;
                let centroid = &self.codebooks[centroid_start..centroid_start + self.subdim_size];
                let mut dist = 0.0f32;
                for d in 0..self.subdim_size {
                    let diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                table[table_base + c] = dist;
            }
        }

        table
    }

    /// Distance using precomputed ADC table.
    fn distance_with_table(&self, query_table: &[f32], idx: usize, dim: usize) -> f32 {
        if self.dims == 0 || self.codes.is_empty() || self.ksub == 0 || self.codebooks.is_empty() {
            return f32::MAX;
        }

        let num_subdims = self.dims.min(dim);
        let mut dist = 0.0f32;

        for sub_q in 0..num_subdims {
            let code_idx = self.codes[idx * num_subdims + sub_q] as usize;
            let table_row = sub_q * self.ksub;
            let centroid_idx = code_idx.min(self.ksub - 1);
            dist += query_table[table_row + centroid_idx];
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
            self.vectors
                .extend_from_slice(&vectors[start..start + self.dim]);
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
        tracing::info!(
            "Built DiskANN graph with {} nodes, max_degree={}, search_list={}",
            n,
            self.dann_config.max_degree,
            self.dann_config.search_list_size
        );
        Ok(())
    }

    pub fn scope_audit(&self) -> DiskAnnScopeAudit {
        DiskAnnScopeAudit {
            dim: self.dim,
            max_degree: self.dann_config.max_degree,
            search_list_size: self.dann_config.search_list_size,
            beamwidth: self.dann_config.beamwidth,
            uses_vamana_graph: true,
            uses_placeholder_pq: true,
            has_flash_layout: false,
            native_comparable: false,
            comparability_reason: "DiskAnnIndex remains a simplified Vamana path with placeholder PQ rather than a native-comparable SSD DiskANN pipeline",
        }
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
    #[allow(non_snake_case)]
    fn build_vamana_graph(&mut self) {
        let n = self.ids.len();
        if n == 0 {
            return;
        }

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
    #[allow(non_snake_case)]
    fn vamana_search(
        &self,
        query: &[f32],
        L: usize,
        _R: usize,
        graph: &[Vec<(i64, f32)>],
    ) -> Vec<(usize, f32)> {
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
        results.into_iter().map(|(d, idx)| (idx, d)).collect()
    }

    /// Prune neighbors using Vamana pruning strategy
    #[allow(non_snake_case)]
    fn prune_neighbors(
        &self,
        _node_idx: usize,
        neighbors: &[(i64, f32)],
        R: usize,
    ) -> Vec<(i64, f32)> {
        if neighbors.len() <= R {
            return neighbors.to_vec();
        }

        // Sort by distance
        let mut sorted: Vec<(i64, f32)> = neighbors.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Vamana-style alpha pruning (triangle inequality occlusion):
        // keep candidate k only when max_j(distance(i,k) / distance(j,k)) <= alpha.
        let mut selected: Vec<(i64, f32)> = Vec::new();

        for &(id, dist) in &sorted {
            if selected.len() >= R {
                break;
            }

            let idx = id as usize;
            if idx >= self.vectors.len() / self.dim {
                continue;
            }

            let mut occluded = false;

            for &(sel_id, _) in &selected {
                let sel_idx = sel_id as usize;
                if sel_idx < self.vectors.len() / self.dim {
                    let d_jk = self.distance_between_nodes(sel_idx, idx);
                    if d_jk <= f32::EPSILON || dist / d_jk > self.alpha {
                        occluded = true;
                        break;
                    }
                }
            }

            if !occluded {
                selected.push((id, dist));
            }
        }

        selected
    }

    /// Refine graph with second pass
    #[allow(non_snake_case)]
    fn refine_graph(&mut self) {
        let n = self.ids.len();
        let R = self.dann_config.max_degree;

        // For each node, search again and update edges
        for i in 0..n {
            let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
            let neighbors =
                self.vamana_search(query, self.dann_config.search_list_size, R, &self.graph);

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
    fn l2_sqr(&self, a: &[f32], b_idx: usize) -> f32 {
        let start = b_idx * self.dim;
        let b = &self.vectors[start..start + self.dim];

        // Search/range gate on squared L2 directly to avoid an unnecessary sqrt+square roundtrip.
        simd::l2_distance_sq(a, b)
    }

    #[inline]
    fn distance_between_nodes(&self, a_idx: usize, b_idx: usize) -> f32 {
        if a_idx >= self.ids.len() || b_idx >= self.ids.len() {
            return f32::MAX;
        }
        let a_start = a_idx * self.dim;
        let a = &self.vectors[a_start..a_start + self.dim];
        self.compute_dist(a, b_idx)
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
            self.vectors
                .extend_from_slice(&vectors[start..start + self.dim]);

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

        #[allow(non_snake_case)]
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
    #[allow(non_snake_case)]
    fn beam_search(&self, query: &[f32], L: usize) -> Vec<(i64, f32)> {
        let n = self.ids.len();
        if n == 0 {
            return vec![];
        }

        let beamwidth = self.dann_config.beamwidth;
        let effective_l = if self.pq_codes.is_some() {
            let expanded = L
                .saturating_mul(self.dann_config.pq_candidate_expand_pct)
                .saturating_add(99)
                / 100;
            expanded.max(L).min(n)
        } else {
            L
        };
        let mut visited = vec![false; n];
        let mut candidates: BinaryHeap<ReverseOrderedFloat> = BinaryHeap::new();
        let mut results: Vec<(f32, usize)> = Vec::new();
        let pq_query_table = self.pq_codes.as_ref().and_then(|pq| {
            if pq.dims > 0 && pq.ksub > 0 {
                Some(pq.build_query_distance_table(query, self.dim))
            } else {
                None
            }
        });

        // Start from entry point or node 0
        let start = self.entry_point.unwrap_or(0);
        let dist = self.compute_dist(query, start);
        candidates.push(ReverseOrderedFloat(dist, start));
        visited[start] = true;

        // Early termination tracking
        let mut no_progress_count = 0;
        let mut best_dist = dist;

        // Beam search loop
        while !candidates.is_empty() && results.len() < effective_l {
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
            if no_progress_count > effective_l / 4 {
                break;
            }

            // Explore neighbors with beamwidth limit
            if let Some(nbrs) = self.graph.get(idx) {
                let mut nbr_dists: Vec<(f32, usize)> = Vec::new();

                for &(nbr_id, _) in nbrs {
                    let n_idx = nbr_id as usize;
                    if n_idx < n && !visited[n_idx] {
                        // Use PQ distance if available and node is not cached
                        let d = if let Some(pq_codes) = &self.pq_codes {
                            if !self.cached_nodes.contains(&n_idx) {
                                if let Some(table) = &pq_query_table {
                                    pq_codes.distance_with_table(table, n_idx, self.dim)
                                } else {
                                    pq_codes.distance(query, n_idx, self.dim)
                                }
                            } else {
                                self.compute_dist(query, n_idx)
                            }
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
                    // Keep PQ as screening score, but use exact distance for frontier updates
                    // so candidate ordering and final ranking stay faithful to the true metric.
                    let exact_d = if self.pq_codes.is_some() {
                        self.compute_dist(query, n_idx)
                    } else {
                        d
                    };
                    candidates.push(ReverseOrderedFloat(exact_d, n_idx));
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(L);

        results
            .into_iter()
            .map(|(d, idx)| (self.ids[idx], d))
            .collect()
    }

    /// Range search: find all vectors within radius
    pub fn range_search(
        &self,
        query: &[f32],
        radius: f32,
        max_results: usize,
    ) -> Result<Vec<(i64, f32)>> {
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
            let (idx, _dist) = candidates.pop_front().unwrap();

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
    #[allow(non_snake_case)]
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

        let avg_deg = if n > 0 {
            num_edges as f32 / n as f32
        } else {
            0.0
        };

        // Estimate memory usage
        let pq_memory = self
            .pq_codes
            .as_ref()
            .map(|p| p.codes.len() + p.codebooks.len() * 4)
            .unwrap_or(0);
        let memory = self.vectors.len() * 4  // vectors
            + self.ids.len() * 8             // ids
            + num_edges * 12                 // graph edges (8 byte id + 4 byte dist)
            + pq_memory;

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
        file.write_all(&3u32.to_le_bytes())?; // Version 3
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
            file.write_all(&(pq.ksub as u32).to_le_bytes())?;
            file.write_all(&(pq.subdim_size as u32).to_le_bytes())?;
            file.write_all(&(pq.codes.len() as u64).to_le_bytes())?;
            file.write_all(&pq.codes)?;
            file.write_all(&(pq.codebooks.len() as u64).to_le_bytes())?;
            for &v in &pq.codebooks {
                file.write_all(&v.to_le_bytes())?;
            }
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
            return Err(crate::api::KnowhereError::Codec(
                "invalid magic".to_string(),
            ));
        }

        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;

        if dim != self.dim {
            return Err(crate::api::KnowhereError::Codec(
                "dimension mismatch".to_string(),
            ));
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
            self.vectors
                .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
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

                let mut ksub = 0usize;
                let mut subdim_size = 0usize;
                if version >= 3 {
                    let mut val = [0u8; 4];
                    file.read_exact(&mut val)?;
                    ksub = u32::from_le_bytes(val) as usize;
                    file.read_exact(&mut val)?;
                    subdim_size = u32::from_le_bytes(val) as usize;
                }

                let mut len_bytes = [0u8; 8];
                file.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;

                let mut codes = vec![0u8; len];
                file.read_exact(&mut codes)?;

                let mut codebooks = Vec::new();
                if version >= 3 {
                    let mut cb_len_bytes = [0u8; 8];
                    file.read_exact(&mut cb_len_bytes)?;
                    let cb_len = u64::from_le_bytes(cb_len_bytes) as usize;
                    codebooks = vec![0.0f32; cb_len];
                    for item in codebooks.iter_mut().take(cb_len) {
                        let mut f_bytes = [0u8; 4];
                        file.read_exact(&mut f_bytes)?;
                        *item = f32::from_le_bytes(f_bytes);
                    }
                }

                self.pq_codes = Some(PQCode {
                    codes,
                    dims,
                    ksub,
                    subdim_size,
                    codebooks,
                });
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
#[allow(dead_code)]
pub struct DiskAnnIterator<'a> {
    index: &'a DiskAnnIndex,
    results: Vec<(i64, f32)>,
    current: usize,
}

impl<'a> DiskAnnIterator<'a> {
    #[allow(non_snake_case)]
    fn new(index: &'a DiskAnnIndex, query: &'a [f32], L: usize) -> Self {
        let results = index.beam_search(query, L);
        Self {
            index,
            results,
            current: 0,
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
        if self.current < self.results.len() {
            let result = self.results[self.current];
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::IndexType;
    use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
    use serde_json::Value;
    use std::fs;

    const DISKANN_FINAL_VERDICT_PATH: &str = "benchmark_results/diskann_p3_004_final_verdict.json";

    fn load_diskann_final_verdict() -> Value {
        let content = fs::read_to_string(DISKANN_FINAL_VERDICT_PATH)
            .expect("DiskANN family verdict artifact must exist for the library verdict lane");
        serde_json::from_str(&content).expect("DiskANN family verdict artifact must be valid JSON")
    }

    #[test]
    fn test_diskann() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();

        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
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
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        // Create and train index
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
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
    fn test_diskann_save_load_preserves_pq_codebook() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_pq_dims: Some(2),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        index.train(&vectors).unwrap();
        assert!(index.pq_codes.is_some());
        let before_pq = index.pq_codes.as_ref().unwrap();
        assert!(!before_pq.codebooks.is_empty());

        let temp_path = std::env::temp_dir().join("diskann_test_pq_v3.bin");
        index.save(&temp_path).unwrap();

        let mut index2 = DiskAnnIndex::new(&config).unwrap();
        index2.load(&temp_path).unwrap();
        let after_pq = index2.pq_codes.as_ref().unwrap();
        assert_eq!(before_pq.dims, after_pq.dims);
        assert_eq!(before_pq.ksub, after_pq.ksub);
        assert_eq!(before_pq.subdim_size, after_pq.subdim_size);
        assert_eq!(before_pq.codebooks.len(), after_pq.codebooks.len());
        assert_eq!(before_pq.codes.len(), after_pq.codes.len());

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
        assert_eq!(result1.ids, result2.ids);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_diskann_range_search() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
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
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
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
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let iter = index.search_iterator(&query, 10);

        let mut count = 0;
        for (id, _dist) in iter {
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
            disk_pq_dims: Some(2),
            disk_pq_candidate_expand_pct: Some(150),
            ..Default::default()
        };

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params,
        };

        let index = DiskAnnIndex::new(&config).unwrap();

        assert_eq!(index.dann_config.max_degree, 64);
        assert_eq!(index.dann_config.search_list_size, 200);
        assert_eq!(index.dann_config.beamwidth, 16);
        assert_eq!(index.dann_config.disk_pq_dims, 2);
        assert_eq!(index.dann_config.pq_candidate_expand_pct, 150);
    }

    #[test]
    fn test_diskann_config_clamps_pq_candidate_expand_pct() {
        use crate::api::IndexParams;

        let params = IndexParams {
            disk_pq_candidate_expand_pct: Some(20),
            ..Default::default()
        };

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params,
        };

        let index = DiskAnnIndex::new(&config).unwrap();
        assert_eq!(index.dann_config.pq_candidate_expand_pct, 100);
    }

    #[test]
    fn test_diskann_pq_table_distance_matches_direct_distance() {
        let mut pq = PQCode::new(2);
        let vectors = vec![
            0.1f32, -0.2, 0.3, -0.4, //
            0.5, -0.6, 0.7, -0.8,
        ];
        let dim = 4usize;
        pq.encode(&vectors, dim);

        let query = vec![0.11f32, -0.19, 0.29, -0.41];
        let table = pq.build_query_distance_table(&query, dim);
        let direct = pq.distance(&query, 0, dim);
        let cached = pq.distance_with_table(&table, 0, dim);

        assert!((direct - cached).abs() < 1e-6);
    }

    #[test]
    fn test_diskann_prune_neighbors_uses_alpha_occlusion() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        index.vectors = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            2.0, 0.0, // node 2
            3.0, 0.0, // node 3
        ];
        index.ids = vec![0, 1, 2, 3];

        let pruned = index.prune_neighbors(0, &[(1, 1.0), (2, 4.0), (3, 9.0)], 2);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].0, 1);
    }

    #[test]
    fn test_diskann_beam_search_with_pq_returns_exact_final_distances() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_pq_dims: Some(2),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.9, 0.0, 0.0, 0.0];
        let results = index.beam_search(&query, 4);

        for (id, dist) in results {
            let idx = id as usize;
            let exact = index.compute_dist(&query, idx);
            assert!(
                (dist - exact).abs() < 1e-6,
                "beam-search final distance should be exact when PQ is enabled"
            );
        }
    }

    #[test]
    fn test_diskann_index_trait() {
        use crate::dataset::Dataset;
        use crate::index::Index;

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();

        // Test Index trait methods
        assert_eq!(Index::index_type(&index), "DiskAnn");
        assert_eq!(Index::dim(&index), 4);
        assert_eq!(Index::count(&index), 0);
        assert!(!Index::is_trained(&index));

        // Train
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let dataset = Dataset::from_vectors(vectors.clone(), 4);
        Index::train(&mut index, &dataset).unwrap();

        assert!(Index::is_trained(&index));
        assert_eq!(Index::count(&index), 4);
        assert!(Index::has_raw_data(&index));

        // Search
        let query = Dataset::from_vectors(vec![0.1, 0.1, 0.1, 0.1], 4);
        let result = Index::search(&index, &query, 2).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.distances.len(), 2);

        // Get vector by IDs
        let vectors = Index::get_vector_by_ids(&index, &[0, 1]).unwrap();
        assert_eq!(vectors.len(), 8); // 2 vectors * 4 dim
    }

    #[test]
    fn test_diskann_l2_sqr_matches_simd_squared_distance() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        index.train(&vectors).unwrap();

        let query = vec![0.5, 1.5, 2.5, 3.5];
        let expected = simd::l2_distance_sq(&query, &vectors[4..8]);
        let actual = index.l2_sqr(&query, 1);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_diskann_pqcode_builds_codebook_and_codes() {
        let mut pq = PQCode::new(2);
        let vectors = vec![
            0.0, 2.0, 4.0, 6.0, //
            1.0, 1.0, 3.0, 7.0, //
            -1.0, -2.0, 2.0, 1.0,
        ];
        pq.encode(&vectors, 4);

        assert_eq!(pq.codes.len(), 6);
        assert!(pq.ksub > 0);
        assert_eq!(pq.subdim_size, 2);
        assert!(!pq.codebooks.is_empty());
    }

    #[test]
    fn test_diskann_raw_data_semantics_follow_metric_type() {
        use crate::index::Index;

        let l2_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();
        let cosine_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::Cosine,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();
        let ip_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::Ip,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();

        assert!(Index::has_raw_data(&l2_index));
        assert!(Index::has_raw_data(&cosine_index));
        assert!(!Index::has_raw_data(&ip_index));
        assert!(Index::get_vector_by_ids(&ip_index, &[0]).is_err());
    }

    #[test]
    fn test_diskann_get_vector_by_ids_rejects_missing_ids() {
        use crate::dataset::Dataset;
        use crate::index::{Index, IndexError};

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let dataset = Dataset::from_vectors(vec![0.0, 1.0, 2.0, 3.0], 4);
        Index::train(&mut index, &dataset).unwrap();

        let err = Index::get_vector_by_ids(&index, &[7]).unwrap_err();
        assert!(matches!(err, IndexError::Unsupported(_)));
    }

    #[test]
    fn test_diskann_scope_audit_locks_placeholder_boundary() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                max_degree: Some(16),
                search_list_size: Some(32),
                beamwidth: Some(4),
                ..Default::default()
            },
        };

        let index = DiskAnnIndex::new(&config).unwrap();
        let audit = index.scope_audit();

        assert_eq!(audit.dim, 4);
        assert_eq!(audit.max_degree, 16);
        assert_eq!(audit.search_list_size, 32);
        assert_eq!(audit.beamwidth, 4);
        assert!(audit.uses_vamana_graph);
        assert!(audit.uses_placeholder_pq);
        assert!(!audit.has_flash_layout);
        assert!(!audit.native_comparable);
        assert!(
            audit.comparability_reason.contains("placeholder PQ"),
            "audit should explain why DiskAnnIndex is not native-comparable"
        );
    }

    #[test]
    fn test_diskann_family_verdict_archives_constrained_classification() {
        let verdict = load_diskann_final_verdict();
        let diskann = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                max_degree: Some(16),
                search_list_size: Some(32),
                beamwidth: Some(4),
                ..Default::default()
            },
        })
        .unwrap();
        let diskann_audit = diskann.scope_audit();

        let mut pqflash = PQFlashIndex::new(
            AisaqConfig {
                max_degree: 2,
                search_list_size: 4,
                beamwidth: 2,
                disk_pq_dims: 2,
                num_entry_points: 1,
                ..AisaqConfig::default()
            },
            MetricType::L2,
            4,
        )
        .unwrap();
        pqflash
            .add(&[
                0.0, 0.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, //
                10.0, 10.0, 10.0, 10.0,
            ])
            .unwrap();
        let pqflash_audit = pqflash.scope_audit();

        assert_eq!(verdict["family"], "DiskANN");
        assert_eq!(verdict["classification"], "constrained");
        assert_eq!(
            verdict["leadership_verdict"],
            "no_go_for_native_comparable_benchmark"
        );
        assert_eq!(verdict["leadership_claim_allowed"], false);
        assert!(!diskann_audit.native_comparable);
        assert!(diskann_audit.uses_placeholder_pq);
        assert!(!pqflash_audit.native_comparable);
        assert!(pqflash_audit.uses_flash_layout);
    }
}

// ============================================================================
// Index trait implementation for DiskAnnIndex (PARITY-P1-003)
// ============================================================================

use crate::dataset::Dataset;
use crate::index::{AnnIterator, Index, IndexError};
use std::time::Instant;

/// Wrapper to adapt DiskAnnIterator to the AnnIterator trait
pub struct DiskAnnIteratorWrapper {
    results: Vec<(i64, f32)>,
    current: usize,
}

impl DiskAnnIteratorWrapper {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self {
            results,
            current: 0,
        }
    }
}

impl AnnIterator for DiskAnnIteratorWrapper {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.current < self.results.len() {
            let result = self.results[self.current];
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }

    fn buffer_size(&self) -> usize {
        self.results.len() - self.current
    }
}

impl Index for DiskAnnIndex {
    fn index_type(&self) -> &str {
        "DiskAnn"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.ntotal()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        let vectors = dataset.vectors().to_vec();
        DiskAnnIndex::train(self, &vectors).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        let vectors = dataset.vectors().to_vec();
        DiskAnnIndex::add(self, &vectors, None).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let query_vec = query.vectors().to_vec();
        let req = SearchRequest {
            top_k,
            nprobe: self.dann_config.beamwidth,
            filter: None,
            params: None,
            radius: None,
        };

        let start = Instant::now();
        let result = DiskAnnIndex::search(self, &query_vec, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        Ok(crate::index::SearchResult::new(
            result.ids,
            result.distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn range_search(
        &self,
        query: &Dataset,
        radius: f32,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let query_vec = query.vectors().to_vec();

        let start = Instant::now();
        let results = DiskAnnIndex::range_search(self, &query_vec, radius, self.dann_config.max_k)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = results.iter().map(|(_, dist)| *dist).collect();

        Ok(crate::index::SearchResult::new(
            ids,
            distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        if !self.has_raw_data() {
            return Err(IndexError::Unsupported(
                "get_vector_by_ids not supported for DiskANN without raw-data metric semantics"
                    .into(),
            ));
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(ids.len() * self.dim);

        for &id in ids {
            let idx =
                self.ids.iter().position(|&x| x == id).ok_or_else(|| {
                    IndexError::Unsupported(format!("ID {} not found in index", id))
                })?;
            let start = idx * self.dim;
            let end = start + self.dim;
            if end > self.vectors.len() {
                return Err(IndexError::Unsupported(format!(
                    "Vector data corrupted for ID {}",
                    id
                )));
            }
            result.extend_from_slice(&self.vectors[start..end]);
        }

        Ok(result)
    }

    fn has_raw_data(&self) -> bool {
        matches!(self.config.metric_type, MetricType::L2 | MetricType::Cosine)
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        _bitset: Option<&crate::bitset::BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        let query_vec = query.vectors().to_vec();
        let results = self.beam_search(&query_vec, self.dann_config.search_list_size);
        Ok(Box::new(DiskAnnIteratorWrapper::new(results)))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        DiskAnnIndex::save(self, std::path::Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        DiskAnnIndex::load(self, std::path::Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }
}

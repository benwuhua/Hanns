//! HNSW - High Performance Version with True Multi-Layer Support
//! 
//! Optimized HNSW with progressive sampling and full multi-layer graph structure.
//! Implements the original HNSW paper algorithm with:
//! - Random level generation using exponential distribution
//! - Layered neighbor connections (each node has connections at its levels)
//! - Greedy search from top layer to bottom layer

use std::collections::HashSet;
use std::sync::Arc;
use rand::Rng;

use crate::api::{IndexConfig, IndexType, MetricType, Predicate, RangeSearchResult, Result, SearchRequest, SearchResult};

/// Maximum number of layers in the HNSW graph
const MAX_LAYERS: usize = 16;

/// A single node's neighbor connections at a specific layer
#[derive(Clone, Debug)]
pub struct LayerNeighbors {
    /// Neighbor IDs and their distances at this layer
    pub neighbors: Vec<(i64, f32)>,
}

impl LayerNeighbors {
    fn new() -> Self {
        Self { neighbors: Vec::new() }
    }
    
    fn with_capacity(capacity: usize) -> Self {
        Self { neighbors: Vec::with_capacity(capacity) }
    }
}

/// Node information including its layer assignment and connections
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// The highest layer this node exists in (0 = only base layer)
    pub max_layer: usize,
    /// Neighbor connections per layer (index 0 = layer 0, etc.)
    /// Only stores connections up to max_layer
    pub layer_neighbors: Vec<LayerNeighbors>,
}

impl NodeInfo {
    fn new(max_layer: usize, m: usize) -> Self {
        let mut layer_neighbors = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            layer_neighbors.push(LayerNeighbors::with_capacity(m * 2));
        }
        Self { max_layer, layer_neighbors }
    }
}

/// HNSW index with true multi-layer graph structure
pub struct HnswIndex {
    config: IndexConfig,
    entry_point: Option<i64>,
    max_level: usize,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    id_to_idx: std::collections::HashMap<i64, usize>,
    node_info: Vec<NodeInfo>,
    next_id: i64,
    trained: bool,
    dim: usize,
    ef_construction: usize,
    ef_search: usize,
    m: usize,
    m_max0: usize,
    level_multiplier: f32,
    metric_type: MetricType,
}

impl HnswIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        // Get M parameter (default 16, typical range 5-48)
        let m = config.params.m.unwrap_or(16).max(2).min(64);
        
        // M_max0 is typically 2*M for layer 0 (denser connections at base)
        let m_max0 = m * 2;
        
        // Get ef_search parameter (default 64)
        let ef_search = config.params.ef_search.unwrap_or(64).max(1);
        
        // Get ef_construction parameter (default 200)
        let ef_construction = config.params.ef_construction.unwrap_or(200).max(1);
        
        // Calculate level multiplier: m_l = 1 / ln(M)
        let level_multiplier = 1.0 / (m as f32).ln().max(1.0);

        Ok(Self {
            config: config.clone(),
            entry_point: None,
            vectors: Vec::new(),
            ids: Vec::new(),
            id_to_idx: std::collections::HashMap::new(),
            node_info: Vec::new(),
            next_id: 0,
            trained: false,
            dim: config.dim,
            ef_construction,
            ef_search,
            m,
            m_max0,
            max_level: 0,
            level_multiplier,
            metric_type: config.metric_type,
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
    fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        self.trained = true;
        Ok(())
    }

    /// Add vectors to the index with proper multi-layer construction
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        let base_count = self.ids.len();
        self.vectors.reserve(n * self.dim);
        self.ids.reserve(n);
        self.node_info.reserve(n);

        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];

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

    /// Add a single vector to the index with optional explicit layer specification
    ///
    /// # Arguments
    /// * `vector` - The vector to add
    /// * `id` - Optional ID for the vector (auto-generated if None)
    /// * `layers` - Optional explicit layer specification. If Some, the vector will be
    ///              inserted at exactly these layers. If None, uses random level assignment.
    ///
    /// # Example
    /// ```ignore
    /// // Add with random layer (default behavior)
    /// index.add_vector(&[1.0, 2.0, 3.0, 4.0], None, None)?;
    ///
    /// // Add with explicit layers (insert at layers 0, 2, and 3)
    /// index.add_vector(&[1.0, 2.0, 3.0, 4.0], Some(42), Some(&[0, 2, 3]))?;
    /// ```
    pub fn add_vector(&mut self, vector: &[f32], id: Option<i64>, layers: Option<&[usize]>) -> Result<i64> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        if vector.len() != self.dim {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        let assigned_id = id.unwrap_or(self.next_id);
        self.next_id += 1;

        // Determine the node's level(s)
        let node_level = if let Some(spec) = layers {
            // Validate layer specifications
            if spec.is_empty() {
                return Err(crate::api::KnowhereError::InvalidArg(
                    "layers specification cannot be empty".to_string(),
                ));
            }
            let max_spec = *spec.iter().max().unwrap_or(&0);
            if max_spec >= MAX_LAYERS {
                return Err(crate::api::KnowhereError::InvalidArg(
                    format!("layer index {} exceeds maximum {}", max_spec, MAX_LAYERS - 1),
                ));
            }
            // Use the maximum layer from specification
            max_spec
        } else {
            // Use random level assignment
            self.random_level()
        };

        // Determine which layers to actually connect (may be subset of node's levels)
        let connect_layers: Vec<usize> = if let Some(spec) = layers {
            // Sort and deduplicate layer specifications
            let mut sorted: Vec<usize> = spec.iter().copied().collect();
            sorted.sort();
            sorted.dedup();
            // Only include layers up to node_level
            sorted.into_iter().filter(|&l| l <= node_level).collect()
        } else {
            // Connect at all layers from 0 to node_level (original behavior)
            (0..=node_level).collect()
        };

        // Create node info with appropriate layer structure
        let node_info = NodeInfo::new(node_level, self.m);

        // Store vector and metadata
        let idx = self.ids.len();
        self.ids.push(assigned_id);
        self.id_to_idx.insert(assigned_id, idx);
        self.vectors.extend_from_slice(vector);
        self.node_info.push(node_info);

        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(assigned_id);
            self.max_level = node_level;
            return Ok(assigned_id);
        }

        // Insert node into the graph for specified layers
        self.insert_node_at_layers(idx, vector, &connect_layers);

        // Update global max level and entry point if needed
        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(assigned_id);
        }

        Ok(assigned_id)
    }

    /// Insert a node into specific layers of the graph
    fn insert_node_at_layers(&mut self, new_idx: usize, new_vec: &[f32], layers: &[usize]) {
        let new_id = self.ids[new_idx];

        // Start from the top layer and work down
        let mut curr_ep = self.entry_point.unwrap();

        // Get the maximum layer we're connecting to
        let max_layer = layers.iter().max().copied().unwrap_or(0);

        // For each layer from max_layer down to 0
        for level in (0..=max_layer).rev() {
            // Check if this node should be connected at this layer
            if !layers.contains(&level) {
                // Skip this layer but still traverse to find entry point
                if level <= self.max_level {
                    let nearest_results = self.search_layer(new_vec, curr_ep, level, 1);
                    if !nearest_results.is_empty() {
                        curr_ep = nearest_results[0].0;
                    }
                }
                continue;
            }

            // Search for nearest neighbor at this layer
            let results = self.search_layer(new_vec, curr_ep, level, 1);

            if !results.is_empty() {
                curr_ep = results[0].0;
            }

            // Find efConstruction candidates at this layer
            let candidates = self.search_layer(new_vec, curr_ep, level, self.ef_construction);

            // Select best M neighbors using heuristic
            let m = if level == 0 { self.m_max0 } else { self.m };
            let selected = self.select_neighbors_heuristic(new_vec, &candidates, m);

            // Add bidirectional connections
            self.add_bidirectional_connections(new_idx, new_id, level, &selected);
        }
    }
    
    /// Insert a node into the multi-layer graph
    fn insert_node(&mut self, new_idx: usize, new_vec: &[f32], node_level: usize) {
        let new_id = self.ids[new_idx];
        
        // Start from the top layer and work down
        // Find entry point at the highest level
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_ep_idx = self.id_to_idx[&curr_ep];
        
        // For each layer from max_level down to 0
        for level in (0..=self.max_level.min(node_level)).rev() {
            // Search for nearest neighbor at this layer
            let nearest_results = self.search_layer(new_vec, curr_ep, level, 1);
            
            if !nearest_results.is_empty() {
                curr_ep = nearest_results[0].0;
                curr_ep_idx = self.id_to_idx[&curr_ep];
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
    fn search_layer(&self, query: &[f32], entry_id: i64, level: usize, ef: usize) -> Vec<(i64, f32)> {
        use std::collections::BinaryHeap;
        
        // Wrapper for f32 to implement Ord for BinaryHeap
        #[derive(Clone, Copy, PartialEq)]
        struct OrderedDist(f32);
        
        // Manual Eq implementation - f32 doesn't implement Eq due to NaN
        impl Eq for OrderedDist {}
        
        impl PartialOrd for OrderedDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                // Handle NaN by treating it as largest value
                if self.0.is_nan() {
                    return Some(std::cmp::Ordering::Greater);
                }
                if other.0.is_nan() {
                    return Some(std::cmp::Ordering::Less);
                }
                // For normal f32, compare directly but reverse for max-heap behavior
                other.0.partial_cmp(&self.0)
            }
        }
        
        impl Ord for OrderedDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::new();
        
        // Initialize with entry point
        let entry_idx = match self.id_to_idx.get(&entry_id) {
            Some(&idx) => idx,
            None => return vec![],
        };
        
        let entry_dist = self.distance(query, entry_idx);
        candidates.push((OrderedDist(entry_dist), entry_id));
        results.push((OrderedDist(entry_dist), entry_id));
        visited.insert(entry_id);
        
        while let Some((OrderedDist(cand_dist), cand_id)) = candidates.pop() {
            // Check if we can stop
            if results.len() >= ef {
                if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                    if cand_dist > worst_dist {
                        break;
                    }
                }
            }
            
            // Explore neighbors at this layer
            let cand_idx = match self.id_to_idx.get(&cand_id) {
                Some(&idx) => idx,
                None => continue,
            };
            
            let node_info = &self.node_info[cand_idx];
            if level > node_info.max_layer {
                continue;
            }
            
            for &(nbr_id, _) in &node_info.layer_neighbors[level].neighbors {
                if visited.insert(nbr_id) {
                    let nbr_idx = match self.id_to_idx.get(&nbr_id) {
                        Some(&idx) => idx,
                        None => continue,
                    };
                    
                    let nbr_dist = self.distance(query, nbr_idx);
                    
                    if results.len() < ef {
                        results.push((OrderedDist(nbr_dist), nbr_id));
                        candidates.push((OrderedDist(nbr_dist), nbr_id));
                    } else if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                        if nbr_dist < worst_dist {
                            results.pop();
                            results.push((OrderedDist(nbr_dist), nbr_id));
                            candidates.push((OrderedDist(nbr_dist), nbr_id));
                        }
                    }
                }
            }
        }
        
        // Convert to sorted vector
        let mut sorted: Vec<(i64, f32)> = results.into_sorted_vec()
            .into_iter()
            .map(|(OrderedDist(d), id)| (id, d))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted
    }
    
    /// Select neighbors using simple heuristic (closest M)
    fn select_neighbors_heuristic(&self, _query: &[f32], candidates: &[(i64, f32)], m: usize) -> Vec<(i64, f32)> {
        let mut selected: Vec<(i64, f32)> = candidates.iter().take(m).cloned().collect();
        selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        selected
    }
    
    /// Add bidirectional connections between nodes at a specific layer
    fn add_bidirectional_connections(&mut self, new_idx: usize, new_id: i64, level: usize, neighbors: &[(i64, f32)]) {
        let m_max = self.max_connections_for_layer(level);
        
        // Add forward connections from new node
        {
            let node_info = &mut self.node_info[new_idx];
            let layer_nbrs = &mut node_info.layer_neighbors[level].neighbors;
            
            for &(nbr_id, dist) in neighbors {
                layer_nbrs.push((nbr_id, dist));
            }
            
            // Prune if too many connections
            if layer_nbrs.len() > m_max {
                layer_nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                layer_nbrs.truncate(m_max);
            }
        }
        
        // Add reverse connections
        for &(nbr_id, dist) in neighbors {
            if let Some(&nbr_idx) = self.id_to_idx.get(&nbr_id) {
                let nbr_node_info = &mut self.node_info[nbr_idx];
                
                // Only add if this layer exists for the neighbor
                if level <= nbr_node_info.max_layer {
                    let nbr_layer_nbrs = &mut nbr_node_info.layer_neighbors[level].neighbors;
                    nbr_layer_nbrs.push((new_id, dist));
                    
                    // Prune if too many connections
                    if nbr_layer_nbrs.len() > m_max {
                        nbr_layer_nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        nbr_layer_nbrs.truncate(m_max);
                    }
                }
            }
        }
    }

    /// Calculate distance based on metric type
    #[inline]
    fn distance(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim;
        let stored = &self.vectors[start..start + self.dim];
        
        match self.metric_type {
            MetricType::L2 => {
                let mut sum = 0.0f32;
                for i in 0..self.dim {
                    let diff = query[i] - stored[i];
                    sum += diff * diff;
                }
                sum
            }
            MetricType::Ip => {
                let mut sum = 0.0f32;
                for i in 0..self.dim {
                    sum += query[i] * stored[i];
                }
                -sum
            }
            MetricType::Cosine => {
                let mut ip = 0.0f32;
                let mut q_norm = 0.0f32;
                let mut v_norm = 0.0f32;
                for i in 0..self.dim {
                    ip += query[i] * stored[i];
                    q_norm += query[i] * query[i];
                    v_norm += stored[i] * stored[i];
                }
                q_norm = q_norm.sqrt();
                v_norm = v_norm.sqrt();
                if q_norm > 0.0 && v_norm > 0.0 {
                    1.0 - ip / (q_norm * v_norm)
                } else {
                    1.0
                }
            }
            _ => {
                let mut sum = 0.0f32;
                for i in 0..self.dim {
                    let diff = query[i] - stored[i];
                    sum += diff * diff;
                }
                sum
            }
        }
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

        let ef = self.ef_search.max(req.nprobe.max(1));
        let k = req.top_k;
        let filter = req.filter.clone();
        
        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            let results = self.search_single(query_vec, ef, k, &filter);
            
            let offset = q_idx * k;
            for (i, item) in results.into_iter().enumerate().take(k) {
                all_ids[offset + i] = item.0;
                all_dists[offset + i] = item.1;
            }
        }
        
        // Finalize distances
        for i in 0..all_dists.len() {
            if all_ids[i] != -1 {
                match self.metric_type {
                    MetricType::L2 => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                    MetricType::Ip => {
                        all_dists[i] = -all_dists[i];
                    }
                    MetricType::Cosine => {}
                    _ => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn search_single(&self, query: &[f32], ef: usize, k: usize, filter: &Option<Arc<dyn Predicate>>) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        let filter_fn = |id: i64| {
            if let Some(f) = filter {
                f.evaluate(id)
            } else {
                true
            }
        };

        // Multi-layer search with layer-wise jumping: start from top layer
        let mut curr_ep = self.entry_point.unwrap();

        // Enhanced layer descent: use larger ef at higher layers for better jumping
        // As we descend, reduce ef to balance quality vs speed
        for level in (1..=self.max_level).rev() {
            // Higher layers use smaller ef for jumping (faster but less accurate)
            // Lower layers (closer to 0) use larger ef for better candidates
            let jump_ef = if level >= self.max_level / 2 {
                // Top half layers: use ef=1 for fast jumping
                1
            } else {
                // Bottom half layers: use ef=min(ef, 4) for better candidates
                ef.min(4)
            };

            let results = self.search_layer(query, curr_ep, level, jump_ef);

            // Find the best valid result for jumping
            let mut best_valid_id = curr_ep;
            let mut best_valid_dist = f32::MAX;

            for (id, dist) in results {
                if filter_fn(id) {
                    best_valid_id = id;
                    best_valid_dist = dist;
                    break;
                }
            }

            // Only jump if we found a valid better candidate
            if best_valid_id != curr_ep {
                curr_ep = best_valid_id;
            }
        }

        // Final search at layer 0 with full ef
        let results = self.search_layer(query, curr_ep, 0, ef);

        // Apply filter and return top k
        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (id, dist) in results {
            if filter_fn(id) {
                final_results.push((id, dist));
                if final_results.len() >= k {
                    break;
                }
            }
        }

        final_results
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        // Magic and version
        file.write_all(b"HNSW")?;
        file.write_all(&3u32.to_le_bytes())?; // Version 3: multi-layer support
        
        // Config
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.m_max0 as u32).to_le_bytes())?;
        file.write_all(&(self.ef_search as u32).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(self.max_level as u32).to_le_bytes())?;
        file.write_all(&(self.level_multiplier.to_bits()).to_le_bytes())?;
        
        // Metric type
        file.write_all(&(self.metric_type as u8).to_le_bytes())?;
        
        // Number of vectors
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        
        // Vectors
        for v in &self.vectors {
            file.write_all(&v.to_le_bytes())?;
        }
        
        // IDs
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        // Node info (layer assignments and connections)
        for node_info in &self.node_info {
            file.write_all(&(node_info.max_layer as u32).to_le_bytes())?;
            
            // Connections for each layer
            for layer_idx in 0..=node_info.max_layer {
                let layer_nbrs = &node_info.layer_neighbors[layer_idx].neighbors;
                file.write_all(&(layer_nbrs.len() as u32).to_le_bytes())?;
                
                for &(nbr_id, dist) in layer_nbrs {
                    file.write_all(&nbr_id.to_le_bytes())?;
                    file.write_all(&dist.to_le_bytes())?;
                }
            }
        }
        
        // Entry point
        if let Some(ep) = self.entry_point {
            file.write_all(&[1u8])?;
            file.write_all(&ep.to_le_bytes())?;
        } else {
            file.write_all(&[0u8])?;
        }
        
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(crate::api::KnowhereError::Codec("invalid magic".to_string()));
        }
        
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);
        
        if version != 3 {
            return Err(crate::api::KnowhereError::Codec(format!("unsupported version: {}", version)));
        }
        
        // Config
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        self.dim = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.m = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.m_max0 = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.ef_search = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.ef_construction = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.max_level = u32::from_le_bytes(buf4) as usize;
        
        file.read_exact(&mut buf4)?;
        self.level_multiplier = f32::from_bits(u32::from_le_bytes(buf4));
        
        // Metric type
        let mut buf1 = [0u8; 1];
        file.read_exact(&mut buf1)?;
        self.metric_type = MetricType::from_bytes(buf1[0]);
        
        // Number of vectors
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        
        // Vectors
        self.vectors = vec![0.0f32; count * self.dim];
        for i in 0..count * self.dim {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            self.vectors[i] = f32::from_le_bytes(buf);
        }
        
        // IDs
        self.ids = Vec::with_capacity(count);
        self.id_to_idx.clear();
        for i in 0..count {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            let id = i64::from_le_bytes(buf);
            self.ids.push(id);
            self.id_to_idx.insert(id, i);
        }
        
        // Node info
        self.node_info = Vec::with_capacity(count);
        for _ in 0..count {
            file.read_exact(&mut buf4)?;
            let max_layer = u32::from_le_bytes(buf4) as usize;
            
            let mut node_info = NodeInfo::new(max_layer, self.m);
            
            for layer_idx in 0..=max_layer {
                file.read_exact(&mut buf4)?;
                let nbr_count = u32::from_le_bytes(buf4) as usize;
                
                for _ in 0..nbr_count {
                    let mut id_buf = [0u8; 8];
                    let mut dist_buf = [0u8; 4];
                    file.read_exact(&mut id_buf)?;
                    file.read_exact(&mut dist_buf)?;
                    
                    let nbr_id = i64::from_le_bytes(id_buf);
                    let dist = f32::from_le_bytes(dist_buf);
                    node_info.layer_neighbors[layer_idx].neighbors.push((nbr_id, dist));
                }
            }
            
            self.node_info.push(node_info);
        }
        
        // Entry point
        file.read_exact(&mut buf1)?;
        if buf1[0] == 1 {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            self.entry_point = Some(i64::from_le_bytes(buf));
        } else {
            self.entry_point = None;
        }
        
        self.trained = true;
        Ok(())
    }

    /// Check if this index contains raw data
    /// 
    /// HNSW index stores raw vectors in the graph nodes
    pub fn has_raw_data(&self) -> bool {
        true
    }
}

/// Generate a random level for a new node using exponential distribution.
///
/// This implements the original HNSW algorithm for level selection:
/// level = floor(-ln(U) / ln(m)) where U ~ Uniform(0, 1]
///
/// The parameter `m` controls the expected number of connections per node:
/// - Higher m means fewer nodes at higher levels (steeper exponential decay)
/// - Lower m means more nodes at higher levels (shallower decay)
///
/// # Arguments
/// * `m` - The M parameter (number of connections), typically 5-48
/// * `rng` - Random number generator
///
/// # Returns
/// A random level (0 means only base layer, higher values mean more layers)
pub fn random_level(m: usize, rng: &mut impl Rng) -> usize {
    // Ensure m is at least 2 to avoid division issues
    let m = m.max(2);
    let r: f32 = rng.gen(); // Uniform [0, 1)

    // Inverse CDF of exponential distribution: -ln(U) / ln(m)
    // We use the natural logarithm
    let level = (-r.ln() / (m as f32).ln()) as usize;

    // Clamp to reasonable maximum
    level.min(MAX_LAYERS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hnsw() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = HnswIndex::new(&config).unwrap();
        
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
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
    fn test_hnsw_ip_metric() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Ip,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = HnswIndex::new(&config).unwrap();
        
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 0);
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Cosine,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = HnswIndex::new(&config).unwrap();
        
        let vectors = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        let query = vec![2.0, 0.2, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 0);
    }
    
    #[test]
    fn test_hnsw_search_with_filter() {
        use std::sync::Arc;
        use crate::api::search::IdsPredicate;
        
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = HnswIndex::new(&config).unwrap();
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            2.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 0.0, 0.0,
        ];
        let ids = vec![0, 1, 2, 3];
        
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();
        
        let query = vec![0.5, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 4,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 4);
        
        let ids_predicate = IdsPredicate { ids: vec![0, 2] };
        let req_filtered = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: Some(Arc::new(ids_predicate) as Arc<dyn Predicate>),
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req_filtered).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&0) || result.ids.contains(&2));
    }
    
    #[test]
    fn test_random_level_distribution() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams { m: Some(16), ..Default::default() },
        };
        
        let index = HnswIndex::new(&config).unwrap();
        
        // Test that random levels follow expected distribution
        let mut level_counts = vec![0usize; 10];
        for _ in 0..1000 {
            let level = index.random_level();
            if level < 10 {
                level_counts[level] += 1;
            }
        }
        
        // Level 0 should have most nodes (~50% for M=16)
        assert!(level_counts[0] > 400, "Level 0 should have ~50% of nodes");
        
        // Higher levels should have fewer nodes
        for i in 1..level_counts.len() {
            assert!(level_counts[i] <= level_counts[i-1], 
                "Level {} should have fewer nodes than level {}", i, i-1);
        }
    }
    
    #[test]
    fn test_multilayer_structure() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams { m: Some(8), ..Default::default() },
        };
        
        let mut index = HnswIndex::new(&config).unwrap();
        
        // Add many vectors to ensure multi-layer structure
        let mut vectors = Vec::new();
        for i in 0..100 {
            vectors.push(i as f32);
            vectors.push(0.0);
            vectors.push(0.0);
            vectors.push(0.0);
        }
        
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        // Verify that some nodes have multiple layers
        let mut max_layers_found = 0;
        for node_info in &index.node_info {
            max_layers_found = max_layers_found.max(node_info.max_layer);
        }
        
        // With 100 nodes, we should have some nodes at higher layers
        assert!(max_layers_found > 0, "Should have multi-layer structure");
        
        // Verify search still works
        let query = vec![50.0, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }
}

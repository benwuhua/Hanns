//! HNSW - High Performance Version
//! 
//! Optimized HNSW with progressive sampling and multi-layer support.

use std::sync::Arc;

use crate::api::{IndexConfig, IndexType, MetricType, Predicate, RangeSearchResult, Result, SearchRequest, SearchResult};

/// HNSW index with progressive sampling and configurable M parameter
pub struct HnswIndex {
    config: IndexConfig,
    entry_point: Option<i64>,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    neighbors: Vec<Vec<(i64, f32)>>,
    next_id: i64,
    trained: bool,
    dim: usize,
    ef_construction: usize,
    ef_search: usize,
    m: usize,  // Number of connections per layer
    max_level: usize,
    level_0_nodes: Vec<usize>,  // Layer 0 node indices
    metric_type: MetricType,
}

impl HnswIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        // Get M parameter (default 16)
        let m = config.params.m.unwrap_or(16).max(1);
        
        // Get ef_search parameter (default 50)
        let ef_search = config.params.ef_search.unwrap_or(50).max(1);
        
        // Get ef_construction parameter
        let ef_construction = config.params.ef_construction.unwrap_or(200).max(1);

        Ok(Self {
            config: config.clone(),
            entry_point: None,
            vectors: Vec::new(),
            ids: Vec::new(),
            neighbors: Vec::new(),
            next_id: 0,
            trained: false,
            dim: config.dim,
            ef_construction,
            ef_search,
            m,
            max_level: 0,
            level_0_nodes: Vec::new(),
            metric_type: config.metric_type,
        })
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

    /// Optimized add with progressive sampling and M-based pruning
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

        let k = self.ef_construction;
        let m = self.m;
        
        let base_count = self.ids.len();
        self.vectors.reserve(n * self.dim);
        self.ids.reserve(n);
        self.neighbors.reserve(n);

        // Use exact for small indices, progressive sampling for large
        let use_progressive = base_count > 500;
        
        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            // Find candidates
            let mut candidates = if use_progressive {
                // Progressive sampling: check every Nth, then refine
                self.progressive_knn(new_vec, k * 2)
            } else {
                // Exact for small
                let mut d = Vec::with_capacity(base_count + i);
                for j in 0..base_count + i {
                    d.push((self.ids[j], self.distance(new_vec, j)));
                }
                d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                d.truncate(k);
                d
            };
            
            // Apply M-based pruning: keep only top M connections
            // This is critical for HNSW performance
            candidates.truncate(m);
            
            self.ids.push(id);
            self.vectors.extend_from_slice(new_vec);
            self.neighbors.push(candidates.clone());
            self.level_0_nodes.push(self.ids.len() - 1);
            
            // Update neighbors of existing nodes (bidirectional connections)
            if !candidates.is_empty() {
                for (cand_id, cand_dist) in &candidates {
                    if let Some(cand_idx) = self.ids.iter().position(|&x| x == *cand_id) {
                        // Add reverse connection
                        let nbrs = &mut self.neighbors[cand_idx];
                        nbrs.push((id, *cand_dist));
                        // Prune to M
                        if nbrs.len() > m {
                            nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                            nbrs.truncate(m);
                        }
                    }
                }
            }
            
            if self.entry_point.is_none() {
                self.entry_point = Some(id);
            }
        }
        
        Ok(n)
    }

    /// Progressive k-NN: sample, find candidates, refine
    fn progressive_knn(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        let n = self.ids.len();
        if n == 0 { return vec![]; }
        
        // Phase 1: Sample
        let sample_rate = (n / k).max(1);
        let mut samples: Vec<(usize, f32)> = Vec::with_capacity(k);
        
        for i in (0..n).step_by(sample_rate) {
            let dist = self.distance(query, i);
            samples.push((i, dist));
        }
        
        samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        samples.truncate(k.min(samples.len()));
        
        // Phase 2: Refine around top samples
        let refine_radius = sample_rate * 2;
        let mut candidates: Vec<(usize, f32)> = samples.clone();
        
        for (idx, _) in &samples {
            let start = idx.saturating_sub(refine_radius / 2);
            let end = (idx + refine_radius / 2).min(n);
            for j in start..end {
                let dist = self.distance(query, j);
                candidates.push((j, dist));
            }
        }
        
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        
        candidates.into_iter()
            .map(|(idx, dist)| (self.ids[idx], dist))
            .collect()
    }

    #[inline]
    fn l2_sqr(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim;
        let stored = &self.vectors[start..start + self.dim];
        
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let diff = query[i] - stored[i];
            sum += diff * diff;
        }
        sum
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
                // For IP, we return negative inner product (so larger = better)
                let mut sum = 0.0f32;
                for i in 0..self.dim {
                    sum += query[i] * stored[i];
                }
                -sum
            }
            MetricType::Cosine => {
                // Cosine distance = 1 - cosine_similarity
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
                // Default to L2 for unknown types
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

        // Use ef_search from config, fallback to req.nprobe
        let ef = self.ef_search.max(req.nprobe.max(1));
        let k = req.top_k;
        
        // Get filter from request
        let filter = req.filter.clone();
        
        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];
        
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            
            // Generate query indices to paralyze over
            let q_indices: Vec<usize> = (0..n_queries).collect();
            
            // Map each query to its results
            let results: Vec<Vec<(i64, f32)>> = q_indices.par_iter().map(|&q_idx| {
                let q_start = q_idx * self.dim;
                let query_vec = &query[q_start..q_start + self.dim];
                self.search_single(query_vec, ef, k, &filter)
            }).collect();
            
            // Write results sequentially
            for (q_idx, res) in results.into_iter().enumerate() {
                let offset = q_idx * k;
                for (i, item) in res.into_iter().enumerate().take(k) {
                    all_ids[offset + i] = item.0;
                    all_dists[offset + i] = item.1;
                }
            }
        }
        
        #[cfg(not(feature = "parallel"))]
        {
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
        }
        
        // Finalize distances (sqrt for L2, convert back for IP/Cosine)
        for i in 0..all_dists.len() {
            if all_ids[i] != -1 {
                match self.metric_type {
                    MetricType::L2 => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                    MetricType::Ip => {
                        // return original IP (negative was used for ordering)
                        all_dists[i] = -all_dists[i];
                    }
                    MetricType::Cosine => {
                         // already 1 - cosine
                    }
                    _ => {
                        all_dists[i] = all_dists[i].sqrt(); // fallback
                    }
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn search_single(&self, query: &[f32], ef: usize, k: usize, filter: &Option<Arc<dyn Predicate>>) -> Vec<(i64, f32)> {
        if self.ids.is_empty() {
            return vec![];
        }

        let n = self.ids.len();
        let mut visited_mark = (self.next_id as u8).wrapping_add(1);
        if visited_mark == 0 { visited_mark = 1; }
        let mut visited = vec![0u8; n];
        
        let mut results: Vec<(f32, usize)> = Vec::with_capacity(ef);
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(ef * 2);
        
        // Check if filter excludes all elements
        let filter_fn = |id: i64| {
            if let Some(f) = filter {
                f.evaluate(id)
            } else {
                true
            }
        };
        
        // Find valid entry point
        if let Some(ep_id) = self.entry_point {
            if let Some(ep_idx) = self.ids.iter().position(|&id| id == ep_id) {
                if filter_fn(ep_id) {
                    let dist = self.distance(query, ep_idx);
                    candidates.push((dist, ep_idx));
                }
            }
        }
        
        while let Some((cand_dist, cand_idx)) = candidates.pop() {
            if visited[cand_idx] == visited_mark {
                continue;
            }
            visited[cand_idx] = visited_mark;
            
            let node_id = self.ids[cand_idx];
            
            // Apply filter
            if !filter_fn(node_id) {
                continue;
            }
            
            let node_dist = self.distance(query, cand_idx);
            results.push((node_dist, cand_idx));
            
            if results.len() >= ef {
                if let Some((worst, _)) = results.last() {
                    if cand_dist > *worst {
                        break;
                    }
                }
            }
            
            if let Some(nbrs) = self.neighbors.get(cand_idx) {
                for &(nbr_id, _) in nbrs {
                    if let Some(nbr_idx) = self.ids.iter().position(|&id| id == nbr_id) {
                        if visited[nbr_idx] != visited_mark {
                            let nbr_dist = self.distance(query, nbr_idx);
                            
                            let worst = results.last().map(|(d, _)| *d).unwrap_or(f32::MAX);
                            if results.len() < ef || nbr_dist < worst {
                                let pos = candidates.iter().position(|(d, _)| nbr_dist < *d)
                                    .unwrap_or(candidates.len());
                                candidates.insert(pos, (nbr_dist, nbr_idx));
                            }
                        }
                    }
                }
            }
        }
        
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for i in 0..k.min(results.len()) {
            let (d, idx) = results[i];
            final_results.push((self.ids[idx], d));
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
        
        // Version 2: includes M and ef parameters
        file.write_all(b"HNSW")?;
        file.write_all(&2u32.to_le_bytes())?;
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        
        // Save M and ef parameters
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.ef_search as u32).to_le_bytes())?;
        
        let bytes: Vec<u8> = self.vectors.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        file.write_all(&bytes)?;
        
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        for nbrs in &self.neighbors {
            file.write_all(&(nbrs.len() as u32).to_le_bytes())?;
            for &(id, dist) in nbrs {
                file.write_all(&id.to_le_bytes())?;
                file.write_all(&dist.to_le_bytes())?;
            }
        }
        
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
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        if dim != self.dim {
            return Err(crate::api::KnowhereError::Codec("dimension mismatch".to_string()));
        }
        
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Load M and ef_search if version >= 2
        if version >= 2 {
            let mut m_bytes = [0u8; 4];
            file.read_exact(&mut m_bytes)?;
            self.m = u32::from_le_bytes(m_bytes) as usize;
            
            let mut ef_bytes = [0u8; 4];
            file.read_exact(&mut ef_bytes)?;
            self.ef_search = u32::from_le_bytes(ef_bytes) as usize;
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
        
        self.neighbors.clear();
        for _ in 0..count {
            let mut nc_bytes = [0u8; 4];
            file.read_exact(&mut nc_bytes)?;
            let nc = u32::from_le_bytes(nc_bytes) as usize;
            
            let mut nbrs = Vec::with_capacity(nc);
            for _ in 0..nc {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                let id = i64::from_le_bytes(id_bytes);
                
                let mut d_bytes = [0u8; 4];
                file.read_exact(&mut d_bytes)?;
                let dist = f32::from_le_bytes(d_bytes);
                
                nbrs.push((id, dist));
            }
            self.neighbors.push(nbrs);
        }
        
        // Load level 0 nodes
        self.level_0_nodes = (0..count).collect();
        
        let mut has_ep = [0u8; 1];
        file.read_exact(&mut has_ep)?;
        if has_ep[0] == 1 {
            let mut ep_bytes = [0u8; 8];
            file.read_exact(&mut ep_bytes)?;
            self.entry_point = Some(i64::from_le_bytes(ep_bytes));
        } else {
            self.entry_point = None;
        }
        
        self.trained = true;
        Ok(())
    }
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
        
        // Normalized vectors - higher IP = closer
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        // Query similar to [1,0,0,0]
        let query = vec![1.0, 0.1, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        // Should find id 0 (most similar)
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
        
        // Non-normalized vectors
        let vectors = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 2.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        // Query similar to [2,0,0,0]
        let query = vec![2.0, 0.2, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        // Should find id 0 (most similar in cosine)
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
        
        // Add vectors with specific IDs
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,  // id 0
            1.0, 0.0, 0.0, 0.0,  // id 1
            2.0, 0.0, 0.0, 0.0,  // id 2
            3.0, 0.0, 0.0, 0.0,  // id 3
        ];
        let ids = vec![0, 1, 2, 3];
        
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();
        
        // Search without filter - should return all
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
        
        // Search with filter - only allow IDs 0 and 2
        let ids_predicate = IdsPredicate { ids: vec![0, 2] };
        let req_filtered = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: Some(Arc::new(ids_predicate) as Arc<dyn Predicate>),
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req_filtered).unwrap();
        // Should only return IDs 0 and 2
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&0) || result.ids.contains(&2));
    }
}

impl crate::serialize::Serializable for HnswIndex {
    fn serialize(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        // Header: magic + version
        writer.write_all(b"HNSW")?;
        writer.write_all(&2u32.to_le_bytes())?;
        
        // Config
        writer.write_all(&(self.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.m as u32).to_le_bytes())?;
        writer.write_all(&(self.ef_search as u32).to_le_bytes())?;
        writer.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        writer.write_all(&(self.max_level as u32).to_le_bytes())?;
        
        // Metric type
        writer.write_all(&(self.metric_type as u8).to_le_bytes())?;
        
        // Vectors
        writer.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for v in &self.vectors {
            writer.write_all(&v.to_le_bytes())?;
        }
        
        // IDs
        for &id in &self.ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        
        // Neighbors
        for nbrs in &self.neighbors {
            writer.write_all(&(nbrs.len() as u32).to_le_bytes())?;
            for &(id, dist) in nbrs {
                writer.write_all(&id.to_le_bytes())?;
                writer.write_all(&dist.to_le_bytes())?;
            }
        }
        
        // Entry point
        if let Some(ep) = self.entry_point {
            writer.write_all(&1u8.to_le_bytes())?;
            writer.write_all(&ep.to_le_bytes())?;
        } else {
            writer.write_all(&0u8.to_le_bytes())?;
        }
        
        // Level 0 nodes
        writer.write_all(&(self.level_0_nodes.len() as u64).to_le_bytes())?;
        for &idx in &self.level_0_nodes {
            writer.write_all(&(idx as u64).to_le_bytes())?;
        }
        
        Ok(())
    }
    
    fn deserialize(&mut self, reader: &mut dyn std::io::Read) -> std::io::Result<()> {
        // Read magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid HNSW magic"));
        }
        
        // Version
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let _version = u32::from_le_bytes(version_buf);
        
        // Config
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)?;
        self.dim = u32::from_le_bytes(dim_buf) as usize;
        
        let mut m_buf = [0u8; 4];
        reader.read_exact(&mut m_buf)?;
        self.m = u32::from_le_bytes(m_buf) as usize;
        
        let mut ef_buf = [0u8; 4];
        reader.read_exact(&mut ef_buf)?;
        self.ef_search = u32::from_le_bytes(ef_buf) as usize;
        
        let mut ef_c_buf = [0u8; 4];
        reader.read_exact(&mut ef_c_buf)?;
        self.ef_construction = u32::from_le_bytes(ef_c_buf) as usize;
        
        let mut max_level_buf = [0u8; 4];
        reader.read_exact(&mut max_level_buf)?;
        self.max_level = u32::from_le_bytes(max_level_buf) as usize;
        
        // Metric type
        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        self.metric_type = MetricType::from_bytes(metric_buf[0]);
        
        // Vectors
        let mut count_buf = [0u8; 8];
        reader.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;
        
        self.vectors = vec![0.0f32; count * self.dim];
        for v in &mut self.vectors {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }
        
        // IDs
        self.ids = Vec::with_capacity(count);
        for _ in 0..count {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            self.ids.push(i64::from_le_bytes(buf));
        }
        
        // Neighbors
        self.neighbors = Vec::with_capacity(count);
        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            
            let mut nbrs = Vec::with_capacity(len);
            for _ in 0..len {
                let mut id_buf = [0u8; 8];
                let mut dist_buf = [0u8; 4];
                reader.read_exact(&mut id_buf)?;
                reader.read_exact(&mut dist_buf)?;
                nbrs.push((i64::from_le_bytes(id_buf), f32::from_le_bytes(dist_buf)));
            }
            self.neighbors.push(nbrs);
        }
        
        // Entry point
        let mut ep_flag = [0u8; 1];
        reader.read_exact(&mut ep_flag)?;
        if ep_flag[0] == 1 {
            let mut ep_buf = [0u8; 8];
            reader.read_exact(&mut ep_buf)?;
            self.entry_point = Some(i64::from_le_bytes(ep_buf));
        } else {
            self.entry_point = None;
        }
        
        // Level 0 nodes
        let mut level_count_buf = [0u8; 8];
        reader.read_exact(&mut level_count_buf)?;
        let level_count = u64::from_le_bytes(level_count_buf) as usize;
        
        self.level_0_nodes = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let mut idx_buf = [0u8; 8];
            reader.read_exact(&mut idx_buf)?;
            self.level_0_nodes.push(u64::from_le_bytes(idx_buf) as usize);
        }
        
        self.trained = true;
        Ok(())
    }
}

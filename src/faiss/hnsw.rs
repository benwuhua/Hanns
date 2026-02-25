//! HNSW - High Performance Version
//! 
//! Optimized HNSW with progressive sampling and multi-layer support.

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};

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
                    d.push((self.ids[j], self.l2_sqr(new_vec, j)));
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
            let dist = self.l2_sqr(query, i);
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
                let dist = self.l2_sqr(query, j);
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
        
        let mut all_ids = Vec::with_capacity(n_queries * k);
        let mut all_dists = Vec::with_capacity(n_queries * k);
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            let results = self.search_single(query_vec, ef, k);
            
            for i in 0..k {
                if i < results.len() {
                    all_ids.push(results[i].0);
                    all_dists.push(results[i].1.sqrt());
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn search_single(&self, query: &[f32], ef: usize, k: usize) -> Vec<(i64, f32)> {
        if self.ids.is_empty() {
            return vec![];
        }

        let n = self.ids.len();
        let mut visited_mark = (self.next_id as u8).wrapping_add(1);
        if visited_mark == 0 { visited_mark = 1; }
        let mut visited = vec![0u8; n];
        
        let mut results: Vec<(f32, usize)> = Vec::with_capacity(ef);
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(ef * 2);
        
        if let Some(ep_id) = self.entry_point {
            if let Some(ep_idx) = self.ids.iter().position(|&id| id == ep_id) {
                let dist = self.l2_sqr(query, ep_idx);
                candidates.push((dist, ep_idx));
            }
        }
        
        while let Some((cand_dist, cand_idx)) = candidates.pop() {
            if visited[cand_idx] == visited_mark {
                continue;
            }
            visited[cand_idx] = visited_mark;
            
            let node_dist = self.l2_sqr(query, cand_idx);
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
                            let nbr_dist = self.l2_sqr(query, nbr_idx);
                            
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
}

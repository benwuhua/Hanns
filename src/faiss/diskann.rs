//! DiskANN-inspired index (Vamana algorithm) - Optimized
//! 
//! A graph-based index optimized for SSD storage.

use std::collections::HashMap;

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;
use crate::simd;

/// DiskANN-style graph index (Vamana) - Optimized
pub struct DiskAnnIndex {
    config: IndexConfig,
    dim: usize,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    graph: Vec<Vec<(i64, f32)>>,
    next_id: i64,
    trained: bool,
    pub search_l: usize,
    pub construction_l: usize,
    pub alpha: f32,
}

impl DiskAnnIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            next_id: 0,
            trained: false,
            search_l: 50,
            construction_l: 40,  // Reduced for speed
            alpha: 1.2,
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
        
        // Build graph with optimized method
        self.build_graph_optimized();
        
        self.trained = true;
        tracing::info!("Built DiskANN graph with {} nodes", n);
        Ok(())
    }

    /// Optimized graph building - use k-NN based approach
    fn build_graph_optimized(&mut self) {
        let n = self.ids.len();
        if n == 0 { return; }
        
        let L = self.construction_l;
        let R = (L / 2).max(4);  // Neighbor count
        
        // Pre-compute all distances in batches
        let batch_size = 100;
        
        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            
            // Compute distances for this batch
            for i in batch_start..batch_end {
                let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
                
                // Sample-based k-NN (for speed)
                let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(n.min(L * 4));
                
                let step = (n / (L * 4)).max(1);
                for j in (0..n).step_by(step) {
                    if j != i {
                        let dist = self.l2_sqr(query, j);
                        candidates.push((j, dist));
                    }
                }
                
                // Sort and take top L
                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                candidates.truncate(L);
                
                // Convert to graph format
                let neighbors: Vec<(i64, f32)> = candidates
                    .iter()
                    .take(R)
                    .map(|(idx, dist)| (self.ids[*idx], *dist))
                    .collect();
                
                self.graph.push(neighbors);
            }
        }
        
        // Ensure connectivity
        self.ensure_connectivity();
    }

    #[inline]
    fn l2_sqr(&self, a: &[f32], b_idx: usize) -> f32 {
        let start = b_idx * self.dim;
        let b = &self.vectors[start..start + self.dim];
        
        // Use SIMD for distance, then square
        let dist = simd::l2_distance(a, b);
        dist * dist
    }

    fn ensure_connectivity(&mut self) {
        let n = self.ids.len();
        if n < 3 { return; }
        
        // Simple: ensure each node has at least 1 edge
        for i in 0..n {
            if self.graph[i].is_empty() {
                let next = (i + 1) % n;
                let dist = self.l2_sqr(&self.vectors[i * self.dim..], next);
                self.graph[i].push((self.ids[next], dist));
            }
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

        let L = req.nprobe.max(10);
        let k = req.top_k;
        
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            let results = self.greedy_search(query_vec, L);
            
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

    /// Greedy search
    fn greedy_search(&self, query: &[f32], L: usize) -> Vec<(i64, f32)> {
        let n = self.ids.len();
        if n == 0 { return vec![]; }
        
        // Simple: find L nearest using linear scan (for small n)
        // or sample-based (for large n)
        let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(n);
        
        let step = if n > L * 10 { n / (L * 5) } else { 1 };
        
        for i in (0..n).step_by(step) {
            let dist = self.l2_sqr(query, i);
            candidates.push((dist, i));
        }
        
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(L.min(candidates.len()));
        
        // Refine using graph
        let mut visited = vec![false; n];
        let mut results: Vec<(f32, usize)> = Vec::with_capacity(L);
        
        // Start from best candidate
        if let Some((d, idx)) = candidates.first() {
            results.push((*d, *idx));
            visited[*idx] = true;
        }
        
        while results.len() < L {
            let mut best_dist = f32::MAX;
            let mut best_idx = 0;
            
            // Check neighbors of current results
            for &(_, idx) in &results {
                if let Some(nbrs) = self.graph.get(idx) {
                    for &(nbr_id, _) in nbrs {
                        let n_idx = nbr_id as usize;
                        if n_idx < n && !visited[n_idx] {
                            let dist = self.l2_sqr(query, n_idx);
                            if dist < best_dist {
                                best_dist = dist;
                                best_idx = n_idx;
                            }
                        }
                    }
                }
            }
            
            if best_dist == f32::MAX { break; }
            
            visited[best_idx] = true;
            results.push((best_dist, best_idx));
        }
        
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        results.into_iter()
            .map(|(d, idx)| (self.ids[idx], d))
            .collect()
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        file.write_all(b"DANN")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        
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
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        if dim != self.dim {
            return Err(crate::api::KnowhereError::Codec("dimension mismatch".to_string()));
        }
        
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
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
        
        self.trained = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
}

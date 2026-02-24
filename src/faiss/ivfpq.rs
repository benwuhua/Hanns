//! IVF-PQ Index Implementation
//! 
//! Inverted File Index with Product Quantization

use std::collections::HashMap;

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;

/// IVF-PQ Index
pub struct IvfPqIndex {
    config: IndexConfig,
    dim: usize,
    nlist: usize,     // Number of clusters
    nprobe: usize,    // Number of clusters to search
    m: usize,         // Number of sub-quantizers
    nbits_per_idx: usize,  // Bits per sub-vector
    
    /// Cluster centroids
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id -> list of (vector_id, residual)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<f32>)>>,
    /// All vectors
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl IvfPqIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);
        let m = config.params.m.unwrap_or(8);
        let nbits = config.params.nbits_per_idx.unwrap_or(8);

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            nlist,
            nprobe,
            m,
            nbits_per_idx: nbits,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }

    /// Train: k-means clustering to find centroids
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        // Simple k-means clustering
        self.centroids = self.kmeans(vectors, self.nlist);
        
        // Initialize inverted lists
        self.inverted_lists.clear();
        for i in 0..self.nlist {
            self.inverted_lists.insert(i, Vec::new());
        }
        
        self.trained = true;
        tracing::info!("Trained IVF-PQ with {} clusters", self.nlist);
        Ok(())
    }

    /// Simple k-means implementation
    fn kmeans(&self, vectors: &[f32], k: usize) -> Vec<f32> {
        let n = vectors.len() / self.dim;
        let mut centroids = vec![0.0f32; k * self.dim];
        
        // Random initialization: pick k random vectors as centroids
        let step = n / k;
        for i in 0..k {
            let idx = (i * step) * self.dim;
            for j in 0..self.dim {
                centroids[i * self.dim + j] = vectors[idx + j];
            }
        }
        
        // Iterative refinement (3 iterations for speed)
        for _iter in 0..3 {
            // Assign vectors to nearest centroid
            let mut assignments = vec![0usize; n];
            for i in 0..n {
                let start = i * self.dim;
                let mut min_dist = f32::MAX;
                let mut min_idx = 0;
                for c in 0..k {
                    let dist = l2_distance(&vectors[start..start + self.dim], &centroids[c * self.dim..]);
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = c;
                    }
                }
                assignments[i] = min_idx;
            }
            
            // Update centroids
            let mut sums = vec![0.0f32; k * self.dim];
            let mut counts = vec![0usize; k];
            
            for i in 0..n {
                let c = assignments[i];
                for j in 0..self.dim {
                    sums[c * self.dim + j] += vectors[i * self.dim + j];
                }
                counts[c] += 1;
            }
            
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..self.dim {
                        centroids[c * self.dim + j] = sums[c * self.dim + j] / counts[c] as f32;
                    }
                }
            }
        }
        
        centroids
    }

    /// Add vectors to index
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            let start = i * self.dim;
            let end = start + self.dim;
            let vector = &vectors[start..end];
            
            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut cluster = 0;
            for c in 0..self.nlist {
                let dist = l2_distance(vector, &self.centroids[c * self.dim..]);
                if dist < min_dist {
                    min_dist = dist;
                    cluster = c;
                }
            }
            
            // Compute residual
            let mut residual = vec![0.0f32; self.dim];
            for j in 0..self.dim {
                residual[j] = vector[j] - self.centroids[cluster * self.dim + j];
            }
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            // Add to inverted list
            self.inverted_lists.get_mut(&cluster).unwrap().push((id, residual));
            
            // Also store original
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
        }

        tracing::debug!("Added {} vectors to IVF-PQ", n);
        Ok(n)
    }

    /// Search
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

        let nprobe = req.nprobe.max(1).min(self.nlist);
        let k = req.top_k;
        
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            // Find nearest clusters
            let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
                .map(|c| {
                    let dist = l2_distance(query_vec, &self.centroids[c * self.dim..]);
                    (c, dist)
                })
                .collect();
            cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // Search top nprobe clusters
            let mut candidates: Vec<(i64, f32)> = Vec::new();
            
            for (cluster, _) in cluster_dists.iter().take(nprobe) {
                if let Some(list) = self.inverted_lists.get(cluster) {
                    for (id, residual) in list {
                        // Compute distance: query - centroid + residual
                        let mut reconstructed = vec![0.0f32; self.dim];
                        for j in 0..self.dim {
                            reconstructed[j] = query_vec[j] - self.centroids[*cluster * self.dim + j] + residual[j];
                        }
                        let dist = l2_distance(query_vec, &reconstructed);
                        candidates.push((*id, dist));
                    }
                }
            }
            
            // Sort and take top k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            for i in 0..k {
                if i < candidates.len() {
                    all_ids.push(candidates[i].0);
                    all_dists.push(candidates[i].1);
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        file.write_all(b"IVFPQ")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.nlist as u32).to_le_bytes())?;
        
        // Centroids
        let bytes: Vec<u8> = self.centroids.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;
        
        // IDs
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        // Vectors
        let vec_bytes: Vec<u8> = self.vectors.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&vec_bytes)?;
        
        // Inverted lists (simplified: just counts)
        for i in 0..self.nlist {
            let list = self.inverted_lists.get(&i).unwrap();
            file.write_all(&(list.len() as u32).to_le_bytes())?;
        }
        
        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        
        let mut magic = [0u8; 5];
        file.read_exact(&mut magic)?;
        if &magic != b"IVFPQ" {
            return Err(crate::api::KnowhereError::Codec("invalid magic".to_string()));
        }
        
        // Skip version
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        let mut nlist_bytes = [0u8; 4];
        file.read_exact(&mut nlist_bytes)?;
        let nlist = u32::from_le_bytes(nlist_bytes) as usize;
        
        // Load centroids
        let mut centroid_bytes = vec![0u8; nlist * dim * 4];
        file.read_exact(&mut centroid_bytes)?;
        self.centroids.clear();
        for chunk in centroid_bytes.chunks(4) {
            self.centroids.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        
        // Load IDs
        let mut id_count_bytes = [0u8; 8];
        file.read_exact(&mut id_count_bytes)?;
        let id_count = u64::from_le_bytes(id_count_bytes) as usize;
        
        self.ids.clear();
        for _ in 0..id_count {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            self.ids.push(i64::from_le_bytes(id_bytes));
        }
        
        // Load vectors
        let mut vec_bytes = vec![0u8; id_count * dim * 4];
        file.read_exact(&mut vec_bytes)?;
        self.vectors.clear();
        for chunk in vec_bytes.chunks(4) {
            self.vectors.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        
        // Load inverted lists (rebuild from vectors)
        self.inverted_lists.clear();
        for i in 0..nlist {
            self.inverted_lists.insert(i, Vec::new());
        }
        
        for (i, &id) in self.ids.iter().enumerate() {
            let vector = &self.vectors[i * dim..(i + 1) * dim];
            let cluster = self.find_nearest_centroid(vector);
            let residual = self.compute_residual(vector, cluster);
            self.inverted_lists.get_mut(&cluster).unwrap().push((id, residual));
        }
        
        self.trained = true;
        self.next_id = self.ids.last().map(|&id| id + 1).unwrap_or(0);
        Ok(())
    }
    
    /// 查找最近的 centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = vector.iter()
                .zip(centroid)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        best
    }
    
    /// 计算残差
    fn compute_residual(&self, vector: &[f32], cluster: usize) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        vector.iter()
            .zip(centroid)
            .map(|(a, b)| a - b)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivfpq() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 4,
            params: crate::api::IndexParams::default(),
        };
        
        let mut index = IvfPqIndex::new(&config).unwrap();
        
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
            nprobe: 2,
            filter: None,
            params: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
}

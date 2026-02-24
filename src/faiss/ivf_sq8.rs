//! IVF-SQ8 Index Implementation
//! 
//! IVF (Inverted File) + SQ8 (Scalar Quantization 8-bit)
//! 内存优化索引，适合大规模数据

use std::collections::HashMap;

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;
use crate::quantization::ScalarQuantizer;

/// IVF-SQ8 Index
pub struct IvfSq8Index {
    config: IndexConfig,
    dim: usize,
    nlist: usize,     // Number of clusters
    nprobe: usize,    // Number of clusters to search
    
    /// Cluster centroids
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id -> list of (vector_id, quantized residual)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
    /// Scalar quantizer for residuals
    quantizer: ScalarQuantizer,
    /// All vectors (for decoding)
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl IvfSq8Index {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            nlist,
            nprobe,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            quantizer: ScalarQuantizer::new(config.dim, 8),
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }
    
    /// Train the index (k-means for IVF, SQ for quantization)
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "empty training data".to_string(),
            ));
        }
        
        // Train scalar quantizer first
        self.quantizer.train(vectors);
        
        // Simple k-means for IVF
        self.train_ivf(vectors)?;
        
        self.trained = true;
        Ok(n)
    }
    
    /// Train IVF (clustering)
    fn train_ivf(&mut self, vectors: &[f32]) -> Result<()> {
        use crate::quantization::KMeans;
        
        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(vectors);
        
        self.centroids = km.centroids().to_vec();
        Ok(())
    }
    
    /// Add vectors
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        
        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            
            // Find nearest centroid
            let cluster = self.find_nearest_centroid(vector);
            
            // Compute residual
            let residual = self.compute_residual(vector, cluster);
            
            // Quantize residual
            let quantized = self.quantizer.encode(&residual);
            
            // Get ID
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            // Store
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
            
            self.inverted_lists
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push((id, quantized));
        }
        
        Ok(n)
    }
    
    /// Search
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        
        if self.ids.is_empty() {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        
        let n_queries = query.len() / self.dim;
        if n_queries * self.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }
        
        let k = req.top_k;
        let nprobe = req.nprobe.min(self.nlist);
        
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            
            // Find nearest clusters
            let clusters = self.search_clusters(query_vec, nprobe);
            
            // Search in each cluster
            let mut candidates: Vec<(i64, f32)> = Vec::new();
            
            for &cluster_id in &clusters {
                if let Some(list) = self.inverted_lists.get(&cluster_id) {
                    for &(id, ref quantized) in list {
                        // Decode quantized residual
                        let residual = self.quantizer.decode(quantized);
                        
                        // Reconstruct vector: centroid + residual
                        let reconstructed = self.reconstruct(query_vec, cluster_id, &residual);
                        
                        // Compute distance
                        let dist = l2_distance(query_vec, &reconstructed);
                        candidates.push((id, dist));
                    }
                }
            }
            
            // Sort and take top-k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            candidates.truncate(k);
            
            for (id, dist) in candidates {
                all_ids.push(id);
                all_dists.push(dist);
            }
            
            // Fill if not enough
            while all_ids.len() < (q_idx + 1) * k {
                all_ids.push(-1);
                all_dists.push(f32::MAX);
            }
        }
        
        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }
    
    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = l2_distance(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        
        best
    }
    
    /// Search clusters
    fn search_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.nlist)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let dist = l2_distance(query, centroid);
                (i, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }
    
    /// Compute residual
    fn compute_residual(&self, vector: &[f32], cluster: usize) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        vector.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }
    
    /// Reconstruct vector from centroid + residual
    fn reconstruct(&self, _query: &[f32], cluster: usize, residual: &[f32]) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        centroid.iter().zip(residual).map(|(c, r)| c + r).collect()
    }
    
    /// Save index
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        // Magic
        file.write_all(b"IVFSQ8")?;
        
        // Dim
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        
        // Nlist
        file.write_all(&(self.nlist as u32).to_le_bytes())?;
        
        // Centroids
        let centroid_bytes: Vec<u8> = self.centroids.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        file.write_all(&centroid_bytes)?;
        
        // Quantizer params
        file.write_all(&self.quantizer.min_val.to_le_bytes())?;
        file.write_all(&self.quantizer.max_val.to_le_bytes())?;
        file.write_all(&self.quantizer.scale.to_le_bytes())?;
        
        // IDs
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        // Vectors
        let vec_bytes: Vec<u8> = self.vectors.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        file.write_all(&vec_bytes)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexParams, IndexType, MetricType, SearchRequest};
    
    #[test]
    fn test_ivf_sq8_new() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::default(),
        };
        
        let index = IvfSq8Index::new(&config).unwrap();
        assert_eq!(index.dim, 4);
    }
    
    #[test]
    fn test_ivf_sq8_train_add_search() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::default(),
        };
        
        let mut index = IvfSq8Index::new(&config).unwrap();
        
        // Training data: 4 clusters
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            0.1, 0.1, 0.1, 0.1,
            10.0, 10.0, 10.0, 10.0,
            10.1, 10.1, 10.1, 10.1,
        ];
        
        index.train(&vectors).unwrap();
        
        // Add vectors
        index.add(&vectors, None).unwrap();
        
        // Search
        let query = vec![0.05, 0.05, 0.05, 0.05];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
}

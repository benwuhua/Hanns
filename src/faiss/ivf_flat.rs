//! IVF-Flat Index Implementation
//! 
//! IVF (Inverted File) + Flat (no quantization)
//! 内存索引，倒排列表中存储原始向量

use std::collections::HashMap;

use crate::api::{IndexConfig, IndexType, MetricType, IndexParams, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;

/// IVF-Flat Index - stores raw vectors in inverted lists
pub struct IvfFlatIndex {
    config: IndexConfig,
    dim: usize,
    nlist: usize,     // Number of clusters
    nprobe: usize,    // Number of clusters to search
    
    /// Cluster centroids
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id -> list of (vector_id, raw_vector)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<f32>)>>,
    /// All vectors (for reference)
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl IvfFlatIndex {
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
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }
    
    /// Train the index (k-means for IVF)
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "empty training data".to_string(),
            ));
        }
        
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
            let cluster_id = self.find_nearest_centroid(vector);
            
            // Get ID
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            // Store in inverted list (store raw vector)
            let entry = self.inverted_lists.entry(cluster_id).or_insert_with(Vec::new);
            entry.push((id, vector.to_vec()));
            
            // Also store in flat array for reference
            self.vectors.extend_from_slice(vector);
            self.ids.push(id);
        }
        
        Ok(n)
    }
    
    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for c in 0..self.nlist {
            let dist = l2_distance(vector, &self.centroids[c * self.dim..]);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }
        
        best
    }
    
    /// Search
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        
        let top_k = req.top_k;
        let nprobe = if req.nprobe > 0 { req.nprobe } else { self.nprobe };
        
        // Find nearest nprobe clusters
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, l2_distance(query, &self.centroids[c * self.dim..])))
            .collect();
        
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_dists.truncate(nprobe);
        
        // Search in selected clusters
        let mut all_results: Vec<(i64, f32)> = Vec::new();
        
        for (cluster_id, _) in cluster_dists {
            if let Some(list) = self.inverted_lists.get(&cluster_id) {
                for (id, vector) in list {
                    let dist = l2_distance(query, vector);
                    all_results.push((*id, dist));
                }
            }
        }
        
        // Sort and take top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(top_k);
        
        let ids: Vec<i64> = all_results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = all_results.iter().map(|(_, d)| *d).collect();
        
        Ok(SearchResult {
            ids,
            distances,
            elapsed_ms: 0.0,
            num_visited: all_results.len(),
        })
    }
    
    /// Get number of vectors
    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get raw vectors by IDs
    pub fn get_vectors(&self, ids: &[i64]) -> Vec<Option<Vec<f32>>> {
        // Build ID -> vector map from inverted lists
        let mut id_to_vec: HashMap<i64, Vec<f32>> = HashMap::new();
        for (_, list) in &self.inverted_lists {
            for (id, vec) in list {
                id_to_vec.insert(*id, vec.clone());
            }
        }
        
        ids.iter()
            .map(|id| id_to_vec.get(id).cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivf_flat_new() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 128,
            params: IndexParams::ivf(100, 10),
        };
        
        let index = IvfFlatIndex::new(&config).unwrap();
        assert_eq!(index.dim(), 128);
    }
    
    #[test]
    fn test_ivf_flat_train_add_search() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::ivf(2, 1),
        };
        
        let mut index = IvfFlatIndex::new(&config).unwrap();
        
        // Training data
        let train_data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0,
        ];
        
        index.train(&train_data).unwrap();
        
        // Add vectors
        index.add(&train_data, None).unwrap();
        
        // Search
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
}

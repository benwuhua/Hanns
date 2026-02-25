//! Pure Rust Index Implementation (Fallback when Faiss not available)

use std::path::Path;

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;

/// In-memory vector index (pure Rust implementation)
#[derive(Clone)]
pub struct MemIndex {
    config: IndexConfig,
    vectors: Vec<Vec<f32>>,
    ids: Vec<i64>,
    trained: bool,
}

impl MemIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }
        
        Ok(Self {
            config: config.clone(),
            vectors: Vec::new(),
            ids: Vec::new(),
            trained: false,
        })
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.config.dim;
        if n * self.config.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        
        // For flat index, training is a no-op
        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained && self.config.index_type != IndexType::Flat {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }
        
        let n = vectors.len() / self.config.dim;
        if n * self.config.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        
        for i in 0..n {
            let start = i * self.config.dim;
            let end = start + self.config.dim;
            self.vectors.push(vectors[start..end].to_vec());
            
            if let Some(ids) = ids {
                self.ids.push(ids[i]);
            } else {
                self.ids.push(self.ids.len() as i64);
            }
        }
        
        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }
        
        let n = query.len() / self.config.dim;
        if n * self.config.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }
        
        let k = req.top_k.min(self.vectors.len());
        
        let mut results: Vec<(i64, f32)> = Vec::new();
        
        // For each query vector
        for q_idx in 0..n {
            let q_start = q_idx * self.config.dim;
            let q_end = q_start + self.config.dim;
            let query_vec = &query[q_start..q_end];
            
            // Compute distances to all vectors
            let mut distances: Vec<(i64, f32)> = self.vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist = match self.config.metric_type {
                        MetricType::L2 => l2_distance(query_vec, v),
                        MetricType::Ip => -inner_product(query_vec, v), // Negative because we want max
                        MetricType::Cosine => -cosine_similarity(query_vec, v),
                    };
                    (self.ids[i], dist)
                })
                .collect();
            
            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // Take top k
            for i in 0..k.min(distances.len()) {
                results.push(distances[i].clone());
            }
        }
        
        let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
        
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    pub fn ntotal(&self) -> usize {
        self.vectors.len()
    }

    /// Get vectors by their IDs
    /// Returns vectors in the order of the requested IDs
    /// Missing IDs are skipped (not included in result)
    pub fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build ID to index mapping
        let id_to_idx: std::collections::HashMap<i64, usize> = self.ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Collect vectors in the order of requested IDs
        let mut result = Vec::with_capacity(ids.len() * self.config.dim);
        let mut found_count = 0;

        for &id in ids {
            if let Some(&idx) = id_to_idx.get(&id) {
                result.extend_from_slice(&self.vectors[idx]);
                found_count += 1;
            }
        }

        if found_count == 0 {
            return Err(crate::api::KnowhereError::NotFound(
                "none of the requested IDs found".to_string(),
            ));
        }

        Ok(result)
    }

    /// Range search: find all vectors within radius
    pub fn range_search(&self, query: &[f32], radius: f32) -> Result<(Vec<i64>, Vec<f32>)> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n = query.len() / self.config.dim;
        if n * self.config.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let mut result_ids = Vec::new();
        let mut result_distances = Vec::new();

        // For each query vector
        for q_idx in 0..n {
            let q_start = q_idx * self.config.dim;
            let q_end = q_start + self.config.dim;
            let query_vec = &query[q_start..q_end];

            // Compute distances to all vectors
            for (i, v) in self.vectors.iter().enumerate() {
                let dist = match self.config.metric_type {
                    MetricType::L2 => l2_distance(query_vec, v),
                    MetricType::Ip => -inner_product(query_vec, v), // Negative because IP is max
                    MetricType::Cosine => -cosine_similarity(query_vec, v),
                };

                // For L2: dist <= radius; for IP/Cosine: -dist >= radius (since we negated)
                let within_radius = if self.config.metric_type == MetricType::L2 {
                    dist <= radius
                } else {
                    -dist <= radius
                };

                if within_radius {
                    result_ids.push(self.ids[i]);
                    result_distances.push(dist);
                }
            }
        }

        Ok((result_ids, result_distances))
    }

    /// Range search with predicate filter
    pub fn range_search_with_filter(
        &self,
        query: &[f32],
        radius: f32,
        filter: &dyn crate::api::Predicate,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        if self.vectors.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n = query.len() / self.config.dim;
        if n * self.config.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let mut result_ids = Vec::new();
        let mut result_distances = Vec::new();

        // For each query vector
        for q_idx in 0..n {
            let q_start = q_idx * self.config.dim;
            let q_end = q_start + self.config.dim;
            let query_vec = &query[q_start..q_end];

            // Compute distances to all vectors
            for (i, v) in self.vectors.iter().enumerate() {
                // Apply predicate filter
                if !filter.evaluate(self.ids[i]) {
                    continue;
                }

                let dist = match self.config.metric_type {
                    MetricType::L2 => l2_distance(query_vec, v),
                    MetricType::Ip => -inner_product(query_vec, v),
                    MetricType::Cosine => -cosine_similarity(query_vec, v),
                };

                let within_radius = if self.config.metric_type == MetricType::L2 {
                    dist <= radius
                } else {
                    -dist <= radius
                };

                if within_radius {
                    result_ids.push(self.ids[i]);
                    result_distances.push(dist);
                }
            }
        }

        Ok((result_ids, result_distances))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        // Write header
        file.write_all(b"KWIX")?; // Magic
        file.write_all(&1u32.to_le_bytes())?; // Version
        file.write_all(&(self.config.dim as u32).to_le_bytes())?;
        file.write_all(&(self.vectors.len() as u64).to_le_bytes())?;
        
        // Write vectors
        for v in &self.vectors {
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes)?;
        }
        
        // Write IDs
        for id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
        
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};
        
        let mut file = File::open(path)?;
        
        // Read header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"KWIX" {
            return Err(crate::api::KnowhereError::Codec("invalid magic".to_string()));
        }
        
        let mut version = [0u8; 4];
        file.read_exact(&mut version)?;
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        if dim != self.config.dim {
            return Err(crate::api::KnowhereError::Codec("dimension mismatch".to_string()));
        }
        
        let mut num_bytes = [0u8; 8];
        file.read_exact(&mut num_bytes)?;
        let num = u64::from_le_bytes(num_bytes) as usize;
        
        // Read vectors
        self.vectors.clear();
        self.ids.clear();
        
        for _ in 0..num {
            let mut vec = vec![0.0f32; dim];
            // Read raw bytes
            let mut buffer = vec![0u8; dim * 4];
            file.read_exact(&mut buffer)?;
            // Convert to f32
            for (i, chunk) in buffer.chunks(4).enumerate() {
                vec[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            self.vectors.push(vec);
        }
        
        // Read IDs
        for _ in 0..num {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            self.ids.push(i64::from_le_bytes(id_bytes));
        }
        
        self.trained = true;
        
        Ok(())
    }
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = inner_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mem_index() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        // Add some vectors
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0,  // id=0
            0.0, 0.0, 1.0, 0.0,  // id=1
            0.0, 1.0, 0.0, 0.0,  // id=2
            1.0, 0.0, 0.0, 0.0,  // id=3
        ];
        index.add(&vectors, None).unwrap();
        
        // Search
        let query = vec![0.1, 0.1, 0.1, 0.1];
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

    #[test]
    fn test_mem_index_with_ids() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![1.0, 0.0, 0.0, 0.0];
        let ids = vec![100i64];
        index.add(&vectors, Some(&ids)).unwrap();
        
        let query = vec![0.0, 0.0, 0.0, 1.0];
        let req = SearchRequest {
            top_k: 1,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 100);
    }

    #[test]
    fn test_mem_index_empty() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let index = MemIndex::new(&config).unwrap();
        
        let query = vec![0.0, 0.0, 0.0, 1.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req);
        assert!(result.is_err());
    }

    #[test]
    fn test_mem_index_invalid_dim() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 0);
        let result = MemIndex::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mem_index_dimension_mismatch() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        // Wrong dimension
        let vectors = vec![1.0, 2.0, 3.0]; // dim=3 instead of 4
        let result = index.add(&vectors, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mem_index_search_dimension_mismatch() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![1.0, 0.0, 0.0, 0.0];
        index.add(&vectors, None).unwrap();
        
        // Wrong query dimension
        let query = vec![0.0, 0.0, 0.0]; // dim=3 instead of 4
        let req = SearchRequest {
            top_k: 1,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req);
        assert!(result.is_err());
    }

    #[test]
    fn test_mem_index_inner_product() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::Ip, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        index.add(&vectors, None).unwrap();
        
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 1,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 0); // Should return first vector (highest IP)
    }

    #[test]
    fn test_mem_index_serialize() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![1.0, 0.0, 0.0, 0.0];
        let ids = vec![42i64];
        index.add(&vectors, Some(&ids)).unwrap();
        
        // Save
        let path = std::env::temp_dir().join("test_flat_idx");
        index.save(&path).unwrap();
        
        // Load
        let mut loaded = MemIndex::new(&config).unwrap();
        loaded.load(&path).unwrap();
        
        assert_eq!(loaded.ntotal(), 1);
        
        // Clean up
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_get_vector_by_ids() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        // Add vectors with specific IDs
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // id=10
            0.0, 1.0, 0.0, 0.0,  // id=20
            0.0, 0.0, 1.0, 0.0,  // id=30
            0.0, 0.0, 0.0, 1.0,  // id=40
        ];
        let ids = vec![10i64, 20, 30, 40];
        index.add(&vectors, Some(&ids)).unwrap();
        
        // Get vectors by IDs
        let result = index.get_vector_by_ids(&[20, 40, 10]).unwrap();
        
        // Should return vectors in order: id=20, id=40, id=10
        assert_eq!(result.len(), 12); // 3 vectors * 4 dim
        assert_eq!(&result[0..4], &[0.0, 1.0, 0.0, 0.0]); // id=20
        assert_eq!(&result[4..8], &[0.0, 0.0, 0.0, 1.0]); // id=40
        assert_eq!(&result[8..12], &[1.0, 0.0, 0.0, 0.0]); // id=10
    }

    #[test]
    fn test_get_vector_by_ids_missing() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![1.0, 0.0, 0.0, 0.0];
        let ids = vec![100i64];
        index.add(&vectors, Some(&ids)).unwrap();
        
        // Request non-existent ID - should skip it
        let result = index.get_vector_by_ids(&[999, 100]).unwrap();
        
        // Should return only the found vector
        assert_eq!(result.len(), 4);
        assert_eq!(&result, &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_get_vector_by_ids_empty_index() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let index = MemIndex::new(&config).unwrap();
        
        let result = index.get_vector_by_ids(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_search_l2() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        // Vectors at distance 1.0 from origin on each axis
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // dist=1.0
            0.0, 1.0, 0.0, 0.0,  // dist=1.0
            0.0, 0.0, 1.0, 0.0,  // dist=1.0
            2.0, 0.0, 0.0, 0.0,  // dist=2.0
        ];
        index.add(&vectors, None).unwrap();
        
        // Query at origin, radius=1.5
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let (ids, distances) = index.range_search(&query, 1.5).unwrap();
        
        // Should find vectors with distance <= 1.5
        assert_eq!(ids.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_range_search_inner_product() {
        let config = IndexConfig::new(IndexType::Flat, MetricType::Ip, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        // Unit vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // IP=1.0
            0.0, 1.0, 0.0, 0.0,  // IP=0.0
            1.0, 1.0, 0.0, 0.0,  // IP=1.0 (not normalized)
        ];
        index.add(&vectors, None).unwrap();
        
        // Query same as first vector
        let query = vec![1.0, 0.0, 0.0, 0.0];
        // For IP, we use -IP internally, so radius=-0.5 means IP >= 0.5
        let (ids, distances) = index.range_search(&query, -0.5).unwrap();
        
        // Should find vectors with -IP <= -0.5, i.e., IP >= 0.5
        // IDs 0 and 2 have IP >= 0.5 with query
        assert!(ids.len() >= 2);
    }

    #[test]
    fn test_range_search_with_filter() {
        use crate::api::{IdsPredicate, Predicate};
        
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, 4);
        let mut index = MemIndex::new(&config).unwrap();
        
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // id=0
            2.0, 0.0, 0.0, 0.0,  // id=1
            3.0, 0.0, 0.0, 0.0,  // id=2
        ];
        index.add(&vectors, None).unwrap();
        
        // Filter: only allow id=0 and id=2
        let filter = IdsPredicate { ids: vec![0, 2] };
        
        // Query at origin, radius=3.5 (would find all without filter)
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let (ids, _) = index.range_search_with_filter(&query, 3.5, &filter).unwrap();
        
        // Should only return id=0 and id=2
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
    }
}

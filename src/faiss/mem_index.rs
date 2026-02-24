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
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
}

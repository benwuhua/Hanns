//! Index trait implementation for MinHashLSHIndex
//!
//! Provides unified Index trait interface for MinHash-LSH index.

use super::minhash_lsh::MinHashLSHIndex;
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{AnnIterator, Index, IndexError, SearchResult};

/// AnnIterator wrapper for MinHash LSH
pub struct MinHashAnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl MinHashAnnIterator {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl AnnIterator for MinHashAnnIterator {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.pos >= self.results.len() {
            return None;
        }
        let result = self.results[self.pos];
        self.pos += 1;
        Some(result)
    }
}

impl Index for MinHashLSHIndex {
    fn index_type(&self) -> &str {
        "MinHashLSH"
    }

    fn dim(&self) -> usize {
        // MinHash dimension is vec_length * element_size in bytes
        self.mh_vec_length * self.mh_vec_element_size
    }

    fn count(&self) -> usize {
        MinHashLSHIndex::count(self)
    }

    fn is_trained(&self) -> bool {
        // MinHash is treated as trained once build() has produced internal state.
        self.count() > 0 && self.dim() > 0
    }

    fn train(&mut self, _dataset: &Dataset) -> Result<(), IndexError> {
        // MinHash requires byte-level data, not f32 vectors
        // Use build() method directly with byte data
        Err(IndexError::Unsupported(
            "MinHash LSH requires byte data. Use build() method with u8 data.".into(),
        ))
    }

    fn add(&mut self, _dataset: &Dataset) -> Result<usize, IndexError> {
        // MinHash requires byte-level data, not f32 vectors
        // Use build() method directly with byte data
        Err(IndexError::Unsupported(
            "MinHash LSH requires byte data. Use build() method with u8 data.".into(),
        ))
    }

    fn search(&self, query: &Dataset, top_k: usize) -> Result<SearchResult, IndexError> {
        // Convert f32 query to bytes
        let query_f32 = query.vectors();
        
        // Reinterpret f32 bytes as u8 (zero-copy)
        let query_bytes = unsafe {
            std::slice::from_raw_parts(
                query_f32.as_ptr() as *const u8,
                query_f32.len() * std::mem::size_of::<f32>(),
            )
        };

        // Perform search
        let (ids, distances) = self
            .search(query_bytes, top_k, None)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        Ok(SearchResult {
            ids,
            distances,
            elapsed_ms: 0.0,
        })
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult, IndexError> {
        // Convert f32 query to bytes
        let query_f32 = query.vectors();
        let query_bytes = unsafe {
            std::slice::from_raw_parts(
                query_f32.as_ptr() as *const u8,
                query_f32.len() * std::mem::size_of::<f32>(),
            )
        };

        // Perform search with bitset filter
        let (ids, distances) = self
            .search(query_bytes, top_k, Some(bitset))
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        Ok(SearchResult {
            ids,
            distances,
            elapsed_ms: 0.0,
        })
    }

    fn range_search(&self, _query: &Dataset, _radius: f32) -> Result<SearchResult, IndexError> {
        // MinHash LSH doesn't support range search
        Err(IndexError::Unsupported(
            "range_search not supported for MinHash LSH".into(),
        ))
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> Result<Vec<f32>, IndexError> {
        // Get byte data from MinHash index
        let byte_data = self
            .get_vector_by_ids(ids)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        // Convert bytes to f32 (reinterpret)
        let f32_count = byte_data.len() / std::mem::size_of::<f32>();
        let mut result = Vec::with_capacity(f32_count);
        
        for i in 0..f32_count {
            let offset = i * std::mem::size_of::<f32>();
            let f32_bytes: [u8; 4] = byte_data[offset..offset + 4]
                .try_into()
                .map_err(|_| IndexError::Unsupported("Invalid byte data".into()))?;
            result.push(f32::from_le_bytes(f32_bytes));
        }

        Ok(result)
    }

    fn has_raw_data(&self) -> bool {
        MinHashLSHIndex::has_raw_data(self)
    }

    fn save(&self, path: &str) -> Result<(), IndexError> {
        MinHashLSHIndex::save(self, path).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> Result<(), IndexError> {
        MinHashLSHIndex::load(self, path).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        bitset: Option<&BitsetView>,
    ) -> Result<Box<dyn AnnIterator>, IndexError> {
        // Convert f32 query to bytes
        let query_f32 = query.vectors();
        let query_bytes = unsafe {
            std::slice::from_raw_parts(
                query_f32.as_ptr() as *const u8,
                query_f32.len() * std::mem::size_of::<f32>(),
            )
        };

        // Get all results (we'll iterate over them)
        let top_k = self.count(); // Get all candidates
        let (ids, distances) = match bitset {
            Some(b) => self
                .search(query_bytes, top_k, Some(b))
                .map_err(|e| IndexError::Unsupported(e.to_string()))?,
            None => self
                .search(query_bytes, top_k, None)
                .map_err(|e| IndexError::Unsupported(e.to_string()))?,
        };

        // Create iterator from results
        let results: Vec<(i64, f32)> = ids.into_iter().zip(distances.into_iter()).collect();
        Ok(Box::new(MinHashAnnIterator::new(results)))
    }

    fn serialize_to_memory(&self) -> Result<Vec<u8>, IndexError> {
        // MinHash uses file-based serialization
        // In-memory serialization not directly supported
        Err(IndexError::Unsupported(
            "serialize_to_memory not implemented for MinHash LSH. Use save() instead.".into(),
        ))
    }

    fn deserialize_from_memory(&mut self, _data: &[u8]) -> Result<(), IndexError> {
        // MinHash uses file-based deserialization
        // In-memory deserialization not directly supported
        Err(IndexError::Unsupported(
            "deserialize_from_memory not implemented for MinHash LSH. Use load() instead.".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::minhash_lsh::MinHashLSHConfig;

    #[test]
    fn test_minhash_index_trait_basic() {
        let config = MinHashLSHConfig {
            bands: 4,
            band_size: 2,
            use_shared_bloom: false,
            expected_elements: 1000,
            bloom_false_positive_prob: 0.01,
        };

        let mut index = MinHashLSHIndex::with_config(&config);

        // Create test byte data
        let vec_len = 8;
        let elem_size = 8;
        let num_vectors = 20;
        let mut data = Vec::new();

        for i in 0..num_vectors {
            for j in 0..vec_len * elem_size {
                data.push(((i * 10 + j) % 256) as u8);
            }
        }

        // Build index
        index
            .build(&data, vec_len, elem_size, 4, true)
            .unwrap();

        // Test Index trait methods
        assert_eq!(index.index_type(), "MinHashLSH");
        assert_eq!(index.count(), 20);
        assert_eq!(index.dim(), vec_len * elem_size);
        assert!(index.is_trained());
        assert!(index.has_raw_data());

        // Test train (should return Unsupported)
        let dataset = Dataset::from_vectors(vec![0.0; 64], 64);
        assert!(index.train(&dataset).is_err());

        // Test add (should return Unsupported)
        assert!(index.add(&dataset).is_err());
    }

    #[test]
    fn test_minhash_index_trait_search() {
        let config = MinHashLSHConfig {
            bands: 4,
            band_size: 2,
            use_shared_bloom: false,
            expected_elements: 1000,
            bloom_false_positive_prob: 0.01,
        };

        let mut index = MinHashLSHIndex::with_config(&config);

        // Create test byte data
        let vec_len = 8;
        let elem_size = 8;
        let num_vectors = 50;
        let mut data = Vec::new();

        for i in 0..num_vectors {
            for j in 0..vec_len * elem_size {
                data.push(((i * 10 + j) % 256) as u8);
            }
        }

        index
            .build(&data, vec_len, elem_size, 4, true)
            .unwrap();

        // Create query dataset (reinterpret bytes as f32)
        let query_start = 5 * vec_len * elem_size;
        let query_end = query_start + vec_len * elem_size;
        let query_bytes = &data[query_start..query_end];
        
        let query_f32: Vec<f32> = query_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let query_dim = query_f32.len();
        let query_dataset = Dataset::from_vectors(query_f32, query_dim);

        // Test search through Index trait (avoid calling MinHashLSH inherent search)
        let result = Index::search(&index, &query_dataset, 5).unwrap();
        assert!(result.ids.len() <= 5);
        assert_eq!(result.ids.len(), result.distances.len());
    }

    #[test]
    fn test_minhash_index_trait_get_vector_by_ids() {
        let config = MinHashLSHConfig {
            bands: 4,
            band_size: 2,
            use_shared_bloom: false,
            expected_elements: 1000,
            bloom_false_positive_prob: 0.01,
        };

        let mut index = MinHashLSHIndex::with_config(&config);

        let vec_len = 8;
        let elem_size = 8;
        let num_vectors = 10;
        let mut data = Vec::new();

        for i in 0..num_vectors {
            for j in 0..vec_len * elem_size {
                data.push(((i * 10 + j) % 256) as u8);
            }
        }

        index
            .build(&data, vec_len, elem_size, 4, true)
            .unwrap();

        // Test get_vector_by_ids
        let ids = vec![0, 1, 2];
        let vectors = Index::get_vector_by_ids(&index, &ids).unwrap();
        
        // Should return dim * num_ids f32 values
        let expected_len = vec_len * elem_size / 4 * ids.len(); // bytes / 4 = f32 count per vector
        assert_eq!(vectors.len(), expected_len);
    }

    #[test]
    fn test_minhash_index_trait_save_load() {
        let config = MinHashLSHConfig {
            bands: 4,
            band_size: 2,
            use_shared_bloom: false,
            expected_elements: 1000,
            bloom_false_positive_prob: 0.01,
        };

        let mut index = MinHashLSHIndex::with_config(&config);

        let vec_len = 8;
        let elem_size = 8;
        let num_vectors = 10;
        let mut data = Vec::new();

        for i in 0..num_vectors {
            for j in 0..vec_len * elem_size {
                data.push(((i * 10 + j) % 256) as u8);
            }
        }

        index
            .build(&data, vec_len, elem_size, 4, true)
            .unwrap();

        // Save
        let temp_path = "/tmp/test_minhash_index_trait.bin";
        index.save(temp_path).unwrap();

        // Load into new index
        let mut index2 = MinHashLSHIndex::new();
        index2.load(temp_path).unwrap();

        // Verify
        assert_eq!(index2.count(), 10);
        assert_eq!(index2.mh_vec_length, vec_len);
        assert_eq!(index2.mh_vec_element_size, elem_size);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_minhash_index_trait_ann_iterator() {
        let config = MinHashLSHConfig {
            bands: 4,
            band_size: 2,
            use_shared_bloom: false,
            expected_elements: 1000,
            bloom_false_positive_prob: 0.01,
        };

        let mut index = MinHashLSHIndex::with_config(&config);

        let vec_len = 8;
        let elem_size = 8;
        let num_vectors = 20;
        let mut data = Vec::new();

        for i in 0..num_vectors {
            for j in 0..vec_len * elem_size {
                data.push(((i * 10 + j) % 256) as u8);
            }
        }

        index
            .build(&data, vec_len, elem_size, 4, true)
            .unwrap();

        // Create query
        let query_bytes = &data[0..vec_len * elem_size];
        let query_f32: Vec<f32> = query_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let query_dim = query_f32.len();
        let query_dataset = Dataset::from_vectors(query_f32, query_dim);

        // Create iterator
        let mut iter = Index::create_ann_iterator(&index, &query_dataset, None).unwrap();

        // Should be able to iterate
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert!(count > 0);
    }
}

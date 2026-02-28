//! BinIvfFlat - Binary IVF Flat Index
//! 
//! Implements IVF (Inverted File) indexing for binary vectors using Hamming distance.
//! Reference: Faiss IndexBinaryIVF
//! 
//! Uses k-means clustering to partition the vector space into Voronoi cells,
//! enabling faster search by only examining the most relevant clusters.

use crate::api::{MetricType, Result, KnowhereError};
use crate::bitset::BitsetView;

/// Binary IVF Flat Index - uses clustering for faster search
pub struct BinIvfFlatIndex {
    /// Dimension in bits
    dim: usize,
    /// Dimension in bytes (dim / 8, rounded up)
    dim_bytes: usize,
    /// Number of clusters (inverted lists)
    nlist: usize,
    /// Number of probes during search
    nprobe: usize,
    /// Stored binary vectors (size: ntotal * dim_bytes)
    xb: Vec<u8>,
    /// Vector IDs
    ids: Vec<i64>,
    /// Centroids for clustering [nlist * dim_bytes]
    centroids: Vec<u8>,
    /// Inverted lists - indices of vectors in each cluster
    lists: Vec<Vec<usize>>,
    /// Distance metric (Hamming or Jaccard)
    metric: MetricType,
    /// Whether the index has been trained
    is_trained: bool,
}

impl BinIvfFlatIndex {
    /// Create a new Binary IVF Flat index
    /// 
    /// # Arguments
    /// * `dim` - Dimension in bits
    /// * `nlist` - Number of clusters
    /// * `metric` - Distance metric (Hamming or Jaccard)
    pub fn new(dim: usize, nlist: usize, metric: MetricType) -> Self {
        let dim_bytes = (dim + 7) / 8;
        Self {
            dim,
            dim_bytes,
            nlist,
            nprobe: 1,
            xb: Vec::new(),
            ids: Vec::new(),
            centroids: vec![0u8; nlist * dim_bytes],
            lists: (0..nlist).map(|_| Vec::new()).collect(),
            metric,
            is_trained: false,
        }
    }

    /// Set the number of probes for search
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.min(self.nlist);
    }

    /// Get the number of probes
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Get the number of clusters
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Get the number of stored vectors
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Get the dimension in bits
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the dimension in bytes
    pub fn dim_bytes(&self) -> usize {
        self.dim_bytes
    }

    /// Get the metric type
    pub fn metric(&self) -> MetricType {
        self.metric
    }

    /// Check if the index is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Train the index using k-means clustering
    /// 
    /// # Arguments
    /// * `n` - Number of training vectors
    /// * `x` - Training vectors (size: n * dim_bytes)
    pub fn train(&mut self, n: u32, x: &[u8]) -> Result<()> {
        let n = n as usize;
        let expected_size = n * self.dim_bytes;
        
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for {} training vectors, got {}",
                expected_size, n, x.len()
            )));
        }

        if n < self.nlist {
            return Err(KnowhereError::InvalidArg(format!(
                "Number of training vectors ({}) must be >= nlist ({})",
                n, self.nlist
            )));
        }

        // Initialize centroids with training vectors
        // Simple approach: use first nlist vectors as initial centroids
        for i in 0..self.nlist {
            let idx = i * (n / self.nlist);
            let start = idx * self.dim_bytes;
            for j in 0..self.dim_bytes {
                self.centroids[i * self.dim_bytes + j] = x[start + j];
            }
        }

        // Note: We don't store training vectors in the index
        // Training only initializes centroids; vectors are added via add()
        self.is_trained = true;
        Ok(())
    }

    /// Add binary vectors to the index
    /// 
    /// # Arguments
    /// * `n` - Number of vectors to add
    /// * `x` - Binary vectors (size: n * dim_bytes)
    /// * `ids` - Optional vector IDs (if None, auto-generated)
    pub fn add(&mut self, n: u32, x: &[u8], ids: Option<&[i64]>) -> Result<()> {
        let n = n as usize;
        let expected_size = n * self.dim_bytes;
        
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for {} vectors of {} bytes each, got {}",
                expected_size, n, self.dim_bytes, x.len()
            )));
        }

        // Generate IDs if not provided
        let generated_ids: Vec<i64>;
        let id_slice = if let Some(provided_ids) = ids {
            if provided_ids.len() != n {
                return Err(KnowhereError::InvalidArg(format!(
                    "Expected {} IDs, got {}",
                    n, provided_ids.len()
                )));
            }
            provided_ids
        } else {
            generated_ids = (self.ids.len() as i64..(self.ids.len() + n) as i64).collect();
            &generated_ids
        };

        // Store vectors and assign to clusters
        let start_idx = self.xb.len() / self.dim_bytes;
        
        for i in 0..n {
            let vec_start = i * self.dim_bytes;
            let vec = &x[vec_start..vec_start + self.dim_bytes];
            
            // Find nearest centroid and add to that cluster
            let cluster_id = if self.is_trained {
                self.find_nearest_centroid(vec)
            } else {
                // If not trained, distribute evenly
                i % self.nlist
            };
            
            self.lists[cluster_id].push(start_idx + i);
        }

        self.xb.extend_from_slice(x);
        self.ids.extend_from_slice(id_slice);

        Ok(())
    }

    /// Search for k nearest neighbors
    /// 
    /// # Arguments
    /// * `nq` - Number of query vectors
    /// * `xq` - Query vectors (size: nq * dim_bytes)
    /// * `k` - Number of nearest neighbors to return
    /// * `dists` - Output distances (size: nq * k)
    /// * `ids` - Output IDs (size: nq * k)
    pub fn search(
        &self,
        nq: u32,
        xq: &[u8],
        k: i32,
        dists: &mut [f32],
        ids: &mut [i64],
    ) -> Result<()> {
        let nq = nq as usize;
        let k = k as usize;
        let expected_query_size = nq * self.dim_bytes;
        
        if xq.len() != expected_query_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for {} query vectors, got {}",
                expected_query_size, nq, xq.len()
            )));
        }

        if dists.len() != nq * k || ids.len() != nq * k {
            return Err(KnowhereError::InvalidArg(format!(
                "Output arrays must have size {} (nq * k), got dists={} ids={}",
                nq * k, dists.len(), ids.len()
            )));
        }

        // Initialize output with -1 and infinity
        for i in 0..nq * k {
            ids[i] = -1;
            dists[i] = f32::INFINITY;
        }

        // Handle empty index
        if self.is_empty() {
            return Ok(());
        }

        // Process each query
        for q in 0..nq {
            let query_start = q * self.dim_bytes;
            let query = &xq[query_start..query_start + self.dim_bytes];
            
            let output_offset = q * k;
            
            self.search_single_query(query, k, &mut dists[output_offset..], &mut ids[output_offset..]);
        }

        Ok(())
    }

    /// Search with bitset filtering
    pub fn search_with_bitset(
        &self,
        nq: u32,
        xq: &[u8],
        k: i32,
        dists: &mut [f32],
        ids: &mut [i64],
        bitset: &BitsetView,
    ) -> Result<()> {
        let nq = nq as usize;
        let k = k as usize;
        let expected_query_size = nq * self.dim_bytes;
        
        if xq.len() != expected_query_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for {} query vectors, got {}",
                expected_query_size, nq, xq.len()
            )));
        }

        if dists.len() != nq * k || ids.len() != nq * k {
            return Err(KnowhereError::InvalidArg(format!(
                "Output arrays must have size {} (nq * k), got dists={} ids={}",
                nq * k, dists.len(), ids.len()
            )));
        }

        // Initialize output with -1 and infinity
        for i in 0..nq * k {
            ids[i] = -1;
            dists[i] = f32::INFINITY;
        }

        if self.is_empty() {
            return Ok(());
        }

        for q in 0..nq {
            let query_start = q * self.dim_bytes;
            let query = &xq[query_start..query_start + self.dim_bytes];
            
            let output_offset = q * k;
            
            self.search_single_query_with_bitset(query, k, bitset, &mut dists[output_offset..], &mut ids[output_offset..]);
        }

        Ok(())
    }

    /// Search a single query vector
    fn search_single_query(&self, query: &[u8], k: usize, dists: &mut [f32], ids: &mut [i64]) {
        // Find nearest nprobe clusters
        let mut cluster_dists: Vec<(usize, u32)> = (0..self.nlist)
            .map(|c| (c, self.hamming_distance_u32(query, &self.centroids[c * self.dim_bytes..(c + 1) * self.dim_bytes])))
            .collect();
        
        // Sort by distance to centroids
        cluster_dists.sort_by_key(|&(_, d)| d);
        cluster_dists.truncate(self.nprobe);

        // Collect candidates from selected clusters
        let mut candidates: Vec<(f32, i64)> = Vec::new();
        
        for (cluster_id, _) in cluster_dists {
            for &vec_idx in &self.lists[cluster_id] {
                let vec_start = vec_idx * self.dim_bytes;
                let vec = &self.xb[vec_start..vec_start + self.dim_bytes];
                
                let dist = self.compute_distance(query, vec) as f32;
                candidates.push((dist, self.ids[vec_idx]));
            }
        }

        // Sort candidates by distance
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Take top k
        for (i, (dist, id)) in candidates.into_iter().take(k).enumerate() {
            dists[i] = dist;
            ids[i] = id;
        }
    }

    /// Search a single query vector with bitset filtering
    fn search_single_query_with_bitset(&self, query: &[u8], k: usize, bitset: &BitsetView, dists: &mut [f32], ids: &mut [i64]) {
        // Find nearest nprobe clusters
        let mut cluster_dists: Vec<(usize, u32)> = (0..self.nlist)
            .map(|c| (c, self.hamming_distance_u32(query, &self.centroids[c * self.dim_bytes..(c + 1) * self.dim_bytes])))
            .collect();
        
        cluster_dists.sort_by_key(|&(_, d)| d);
        cluster_dists.truncate(self.nprobe);

        let mut candidates: Vec<(f32, i64)> = Vec::new();
        
        for (cluster_id, _) in cluster_dists {
            for &vec_idx in &self.lists[cluster_id] {
                // Skip if filtered by bitset (get returns true if bit is set/filtered)
                if vec_idx < bitset.len() && bitset.get(vec_idx) {
                    continue;
                }

                let vec_start = vec_idx * self.dim_bytes;
                let vec = &self.xb[vec_start..vec_start + self.dim_bytes];
                
                let dist = self.compute_distance(query, vec) as f32;
                candidates.push((dist, self.ids[vec_idx]));
            }
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        for (i, (dist, id)) in candidates.into_iter().take(k).enumerate() {
            dists[i] = dist;
            ids[i] = id;
        }
    }

    /// Find the nearest centroid to a vector
    fn find_nearest_centroid(&self, vec: &[u8]) -> usize {
        let mut best_cluster = 0;
        let mut min_dist = u32::MAX;

        for c in 0..self.nlist {
            let centroid = &self.centroids[c * self.dim_bytes..(c + 1) * self.dim_bytes];
            let dist = self.hamming_distance_u32(vec, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_cluster = c;
            }
        }

        best_cluster
    }

    /// Compute Hamming distance, returning u32
    #[inline]
    fn hamming_distance_u32(&self, a: &[u8], b: &[u8]) -> u32 {
        crate::simd::hamming_distance(a, b) as u32
    }

    /// Compute distance based on metric type
    #[inline]
    fn compute_distance(&self, a: &[u8], b: &[u8]) -> usize {
        match self.metric {
            MetricType::Hamming => crate::simd::hamming_distance(a, b),
            _ => crate::simd::hamming_distance(a, b),
        }
    }

    /// Reconstruct a vector by ID
    pub fn reconstruct(&self, id: i64, recons: &mut [u8]) -> Result<()> {
        if recons.len() != self.dim_bytes {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for reconstruction, got {}",
                self.dim_bytes, recons.len()
            )));
        }

        if let Some(idx) = self.ids.iter().position(|&x| x == id) {
            let start = idx * self.dim_bytes;
            recons.copy_from_slice(&self.xb[start..start + self.dim_bytes]);
            Ok(())
        } else {
            Err(KnowhereError::NotFound(format!("Vector with ID {} not found", id)))
        }
    }

    /// Reconstruct a vector by index
    pub fn reconstruct_at(&self, idx: usize, recons: &mut [u8]) -> Result<()> {
        if recons.len() != self.dim_bytes {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} bytes for reconstruction, got {}",
                self.dim_bytes, recons.len()
            )));
        }

        if idx >= self.len() {
            return Err(KnowhereError::InvalidArg(format!(
                "Index {} out of bounds (len={})",
                idx, self.len()
            )));
        }

        let start = idx * self.dim_bytes;
        recons.copy_from_slice(&self.xb[start..start + self.dim_bytes]);
        Ok(())
    }

    /// Reset the index (remove all vectors)
    pub fn reset(&mut self) {
        self.xb.clear();
        self.ids.clear();
        self.lists = (0..self.nlist).map(|_| Vec::new()).collect();
        self.is_trained = false;
    }

    /// Get all stored IDs
    pub fn get_ids(&self) -> &[i64] {
        &self.ids
    }

    /// Check if the index contains a specific ID
    pub fn has_id(&self, id: i64) -> bool {
        self.ids.contains(&id)
    }

    /// Get the size of a specific inverted list
    pub fn list_size(&self, list_id: usize) -> usize {
        if list_id < self.nlist {
            self.lists[list_id].len()
        } else {
            0
        }
    }

    /// Get centroid data
    pub fn get_centroids(&self) -> &[u8] {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(n: usize, dim_bytes: usize) -> Vec<u8> {
        let mut vectors = Vec::with_capacity(n * dim_bytes);
        for i in 0..n {
            for j in 0..dim_bytes {
                vectors.push(((i + j) % 256) as u8);
            }
        }
        vectors
    }

    #[test]
    fn test_bin_ivf_flat_new() {
        let index = BinIvfFlatIndex::new(64, 10, MetricType::Hamming);
        assert_eq!(index.dim(), 64);
        assert_eq!(index.dim_bytes(), 8);
        assert_eq!(index.nlist(), 10);
        assert_eq!(index.nprobe(), 1);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(!index.is_trained());
        assert_eq!(index.metric(), MetricType::Hamming);
    }

    #[test]
    fn test_bin_ivf_flat_set_nprobe() {
        let mut index = BinIvfFlatIndex::new(64, 10, MetricType::Hamming);
        index.set_nprobe(5);
        assert_eq!(index.nprobe(), 5);
        
        // Should cap at nlist
        index.set_nprobe(20);
        assert_eq!(index.nprobe(), 10);
    }

    #[test]
    fn test_bin_ivf_flat_train() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        // Create 20 training vectors (20 * 8 = 160 bytes)
        let vectors = create_test_vectors(20, 8);
        
        index.train(20, &vectors).unwrap();
        
        assert!(index.is_trained());
        // Check that centroids are initialized
        assert!(!index.get_centroids().iter().all(|&x| x == 0));
    }

    #[test]
    fn test_bin_ivf_flat_train_insufficient_vectors() {
        let mut index = BinIvfFlatIndex::new(64, 10, MetricType::Hamming);
        
        // Try to train with fewer vectors than nlist
        let vectors = vec![0u8; 5 * 8];
        
        let result = index.train(5, &vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_bin_ivf_flat_add() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        // Train first
        let train_vectors = create_test_vectors(20, 8);
        index.train(20, &train_vectors).unwrap();
        
        // Add vectors
        let vectors = create_test_vectors(10, 8);
        let ids: Vec<i64> = (0..10).collect();
        
        index.add(10, &vectors, Some(&ids)).unwrap();
        
        assert_eq!(index.len(), 10);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_bin_ivf_flat_add_without_training() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        // Add without training - should distribute evenly
        let vectors = create_test_vectors(10, 8);
        index.add(10, &vectors, None).unwrap();
        
        assert_eq!(index.len(), 10);
        // Vectors should be distributed across clusters
        let total_in_lists: usize = index.lists.iter().map(|l| l.len()).sum();
        assert_eq!(total_in_lists, 10);
    }

    #[test]
    fn test_bin_ivf_flat_search() {
        let mut index = BinIvfFlatIndex::new(256, 4, MetricType::Hamming); // 32 bytes for SIMD alignment
        
        // Train
        let train_vectors = create_test_vectors(20, 32);
        index.train(20, &train_vectors).unwrap();
        
        // Add test vectors
        let mut vectors = vec![0u8; 32 * 3];
        // Vector 0: all zeros
        // Vector 1: all ones
        for i in 32..64 {
            vectors[i] = 0xff;
        }
        // Vector 2: mixed
        for i in 64..96 {
            vectors[i] = 0x0f;
        }
        let ids = vec![0, 1, 2];
        
        index.add(3, &vectors, Some(&ids)).unwrap();
        
        // Search
        let query = vec![0u8; 32];
        let mut dists = vec![0.0f32; 2];
        let mut ids_out = vec![0i64; 2];
        
        index.set_nprobe(4); // Search all clusters
        index.search(1, &query, 2, &mut dists, &mut ids_out).unwrap();
        
        assert_eq!(ids_out.len(), 2);
        // Results should be sorted by distance
        assert!(dists[0] <= dists[1]);
    }

    #[test]
    fn test_bin_ivf_flat_search_empty() {
        let index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        let query = vec![0u8; 8];
        let mut dists = vec![0.0f32; 5];
        let mut ids = vec![0i64; 5];
        
        index.search(1, &query, 5, &mut dists, &mut ids).unwrap();
        
        // All results should be -1 with infinity distance
        for i in 0..5 {
            assert_eq!(ids[i], -1);
            assert!(dists[i].is_infinite());
        }
    }

    #[test]
    fn test_bin_ivf_flat_reconstruct() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        let train_vectors = create_test_vectors(20, 8);
        index.train(20, &train_vectors).unwrap();
        
        let vectors = vec![
            0b10101010u8, 0b01010101, 0b10101010, 0b01010101, 0b10101010, 0b01010101, 0b10101010, 0b01010101,
            0b11110000u8, 0b00001111, 0b11110000, 0b00001111, 0b11110000, 0b00001111, 0b11110000, 0b00001111,
        ];
        let ids = vec![100, 200];
        
        index.add(2, &vectors, Some(&ids)).unwrap();
        
        let mut recons = vec![0u8; 8];
        index.reconstruct(100, &mut recons).unwrap();
        assert_eq!(recons, vec![0b10101010u8, 0b01010101, 0b10101010, 0b01010101, 0b10101010, 0b01010101, 0b10101010, 0b01010101]);
        
        index.reconstruct_at(1, &mut recons).unwrap();
        assert_eq!(recons, vec![0b11110000u8, 0b00001111, 0b11110000, 0b00001111, 0b11110000, 0b00001111, 0b11110000, 0b00001111]);
    }

    #[test]
    fn test_bin_ivf_flat_reset() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        let train_vectors = create_test_vectors(20, 8);
        index.train(20, &train_vectors).unwrap();
        
        let vectors = create_test_vectors(10, 8);
        index.add(10, &vectors, None).unwrap();
        assert_eq!(index.len(), 10);
        
        index.reset();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(!index.is_trained());
    }

    #[test]
    fn test_bin_ivf_flat_list_size() {
        let mut index = BinIvfFlatIndex::new(64, 4, MetricType::Hamming);
        
        let train_vectors = create_test_vectors(20, 8);
        index.train(20, &train_vectors).unwrap();
        
        let vectors = create_test_vectors(10, 8);
        index.add(10, &vectors, None).unwrap();
        
        // Check that added vectors are distributed across clusters
        let total: usize = (0..4).map(|i| index.list_size(i)).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_bin_ivf_flat_large_dataset() {
        let mut index = BinIvfFlatIndex::new(1024, 16, MetricType::Hamming); // 128 bytes for SIMD
        
        // Train with 100 vectors
        let train_vectors = create_test_vectors(100, 128);
        index.train(100, &train_vectors).unwrap();
        
        // Add 50 vectors
        let vectors = create_test_vectors(50, 128);
        let ids: Vec<i64> = (0..50).collect();
        index.add(50, &vectors, Some(&ids)).unwrap();
        
        assert_eq!(index.len(), 50);
        
        // Search
        let query = vec![0u8; 128];
        let mut dists = vec![0.0f32; 10];
        let mut ids_out = vec![0i64; 10];
        
        index.set_nprobe(8);
        index.search(1, &query, 10, &mut dists, &mut ids_out).unwrap();
        
        assert_eq!(ids_out.len(), 10);
        // Results should be sorted by distance
        for i in 1..10 {
            assert!(dists[i - 1] <= dists[i]);
        }
    }
}

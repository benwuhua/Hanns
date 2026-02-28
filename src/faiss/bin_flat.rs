//! BinFlat - Binary Flat Index (IDMAP)
//! 
//! Implements exhaustive search for binary vectors using Hamming distance.
//! Reference: Faiss IndexBinaryFlat
//! 
//! This is the simplest binary index - stores all vectors and performs brute-force search.
//! Suitable for small datasets or as a baseline for accuracy comparison.

use crate::api::{MetricType, Result, KnowhereError};
use crate::bitset::BitsetView;

/// Binary Flat Index - stores full binary vectors and performs exhaustive search
pub struct BinFlatIndex {
    /// Dimension in bits
    dim: usize,
    /// Dimension in bytes (dim / 8, rounded up)
    dim_bytes: usize,
    /// Stored binary vectors (size: ntotal * dim_bytes)
    xb: Vec<u8>,
    /// Vector IDs
    ids: Vec<i64>,
    /// Distance metric (Hamming or Jaccard)
    metric: MetricType,
    /// Whether to use heap for top-k selection (vs counting sort)
    use_heap: bool,
}

impl BinFlatIndex {
    /// Create a new Binary Flat index
    pub fn new(dim: usize, metric: MetricType) -> Self {
        let dim_bytes = (dim + 7) / 8;
        Self {
            dim,
            dim_bytes,
            xb: Vec::new(),
            ids: Vec::new(),
            metric,
            use_heap: true,
        }
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

    /// Train the index (no-op for Flat index)
    pub fn train(&mut self, _dim: usize, _n: u32, _x: &[u8]) -> Result<()> {
        // Flat index doesn't require training
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

        // Store vectors and IDs
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

        let ntotal = self.len();

        // Process each query
        for q in 0..nq {
            let query_start = q * self.dim_bytes;
            let query = &xq[query_start..query_start + self.dim_bytes];
            
            let output_offset = q * k;
            
            // Compute distances to all vectors
            if self.use_heap && k < ntotal / 2 {
                // Use heap for small k
                self.search_with_heap(query, k, &mut dists[output_offset..], &mut ids[output_offset..]);
            } else {
                // Use counting sort for large k
                self.search_full_sort(query, k, &mut dists[output_offset..], &mut ids[output_offset..]);
            }
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

        let ntotal = self.len();

        for q in 0..nq {
            let query_start = q * self.dim_bytes;
            let query = &xq[query_start..query_start + self.dim_bytes];
            
            let output_offset = q * k;
            
            // Compute distances with bitset filtering
            self.search_with_heap_and_bitset(query, k, bitset, &mut dists[output_offset..], &mut ids[output_offset..]);
        }

        Ok(())
    }

    /// Search using a heap to maintain top-k results
    fn search_with_heap(&self, query: &[u8], k: usize, dists: &mut [f32], ids: &mut [i64]) {
        use std::collections::BinaryHeap;
        
        // Wrapper for f32 to implement Ord for BinaryHeap
        #[derive(Clone, Copy, PartialEq)]
        struct OrderedFloat(f32);
        
        impl Eq for OrderedFloat {}
        
        impl PartialOrd for OrderedFloat {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        
        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse order for min-heap behavior (we want smallest distances)
                other.0.partial_cmp(&self.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let ntotal = self.len();
        let mut heap: BinaryHeap<(OrderedFloat, i64)> = BinaryHeap::with_capacity(k);

        for i in 0..ntotal {
            let vec_start = i * self.dim_bytes;
            let vec = &self.xb[vec_start..vec_start + self.dim_bytes];
            
            let dist = self.compute_distance(query, vec) as f32;
            
            if heap.len() < k {
                heap.push((OrderedFloat(dist), self.ids[i]));
            } else if let Some(&(top_dist, _)) = heap.peek() {
                if dist < top_dist.0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), self.ids[i]));
                }
            }
        }

        // Extract results (sorted by distance ascending)
        let mut results: Vec<(f32, i64)> = heap.into_iter().map(|(d, id)| (d.0, id)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (i, (dist, id)) in results.into_iter().enumerate() {
            if i < k {
                dists[i] = dist;
                ids[i] = id;
            }
        }
    }

    /// Search with full sorting (better for large k)
    fn search_full_sort(&self, query: &[u8], k: usize, dists: &mut [f32], ids: &mut [i64]) {
        let ntotal = self.len();
        
        let mut all_results: Vec<(f32, i64)> = Vec::with_capacity(ntotal);
        
        for i in 0..ntotal {
            let vec_start = i * self.dim_bytes;
            let vec = &self.xb[vec_start..vec_start + self.dim_bytes];
            
            let dist = self.compute_distance(query, vec) as f32;
            all_results.push((dist, self.ids[i]));
        }

        // Sort by distance (ascending)
        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Take top k
        for (i, (dist, id)) in all_results.into_iter().take(k).enumerate() {
            dists[i] = dist;
            ids[i] = id;
        }
    }

    /// Search with heap and bitset filtering
    fn search_with_heap_and_bitset(&self, query: &[u8], k: usize, bitset: &BitsetView, dists: &mut [f32], ids: &mut [i64]) {
        use std::collections::BinaryHeap;
        
        // Wrapper for f32 to implement Ord for BinaryHeap
        #[derive(Clone, Copy, PartialEq)]
        struct OrderedFloat(f32);
        
        impl Eq for OrderedFloat {}
        
        impl PartialOrd for OrderedFloat {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        
        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse order for min-heap behavior (we want smallest distances)
                other.0.partial_cmp(&self.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        let ntotal = self.len();
        let mut heap: BinaryHeap<(OrderedFloat, i64)> = BinaryHeap::with_capacity(k);

        for i in 0..ntotal {
            // Skip if bitset filters out this vector (get returns true if bit is set/filtered)
            if bitset.get(i) {
                continue;
            }

            let vec_start = i * self.dim_bytes;
            let vec = &self.xb[vec_start..vec_start + self.dim_bytes];
            
            let dist = self.compute_distance(query, vec) as f32;
            
            if heap.len() < k {
                heap.push((OrderedFloat(dist), self.ids[i]));
            } else if let Some(&(top_dist, _)) = heap.peek() {
                if dist < top_dist.0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), self.ids[i]));
                }
            }
        }

        // Extract results
        let mut results: Vec<(f32, i64)> = heap.into_iter().map(|(d, id)| (d.0, id)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (i, (dist, id)) in results.into_iter().enumerate() {
            if i < k {
                dists[i] = dist;
                ids[i] = id;
            }
        }
    }

    /// Compute distance based on metric type
    #[inline]
    fn compute_distance(&self, a: &[u8], b: &[u8]) -> usize {
        match self.metric {
            MetricType::Hamming => crate::simd::hamming_distance(a, b),
            _ => {
                // Default to Hamming for binary vectors
                crate::simd::hamming_distance(a, b)
            }
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

    /// Remove vectors matching the given IDs
    pub fn remove_ids(&mut self, ids_to_remove: &[i64]) -> usize {
        let mut removed = 0;
        
        // Create a set of IDs to remove for O(1) lookup
        let remove_set: std::collections::HashSet<i64> = ids_to_remove.iter().cloned().collect();
        
        // Filter out vectors with matching IDs
        let mut new_xb = Vec::with_capacity(self.xb.len());
        let mut new_ids = Vec::with_capacity(self.ids.len());
        
        for i in 0..self.ids.len() {
            if !remove_set.contains(&self.ids[i]) {
                let start = i * self.dim_bytes;
                new_xb.extend_from_slice(&self.xb[start..start + self.dim_bytes]);
                new_ids.push(self.ids[i]);
            } else {
                removed += 1;
            }
        }
        
        self.xb = new_xb;
        self.ids = new_ids;
        
        removed
    }

    /// Reset the index (remove all vectors)
    pub fn reset(&mut self) {
        self.xb.clear();
        self.ids.clear();
    }

    /// Get all stored IDs
    pub fn get_ids(&self) -> &[i64] {
        &self.ids
    }

    /// Check if the index contains a specific ID
    pub fn has_id(&self, id: i64) -> bool {
        self.ids.contains(&id)
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
    fn test_bin_flat_new() {
        let index = BinFlatIndex::new(64, MetricType::Hamming);
        assert_eq!(index.dim(), 64);
        assert_eq!(index.dim_bytes(), 8);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.metric(), MetricType::Hamming);
    }

    #[test]
    fn test_bin_flat_add() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        let vectors = create_test_vectors(10, 8);
        let ids: Vec<i64> = (0..10).collect();
        
        index.add(10, &vectors, Some(&ids)).unwrap();
        
        assert_eq!(index.len(), 10);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_bin_flat_add_auto_ids() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        let vectors = create_test_vectors(5, 8);
        index.add(5, &vectors, None).unwrap();
        
        assert_eq!(index.len(), 5);
        let ids = index.get_ids();
        assert_eq!(ids, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_bin_flat_search() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        // Add some test vectors (flattened: 3 vectors * 8 bytes = 24 bytes)
        let vectors = vec![
            0b00000000u8, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, // All zeros
            0b11111111u8, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, // All ones
            0b00001111u8, 0b00001111, 0b00001111, 0b00001111, 0b00001111, 0b00001111, 0b00001111, 0b00001111, // Half ones
        ];
        let ids = vec![0, 1, 2];
        
        index.add(3, &vectors, Some(&ids)).unwrap();
        
        // Search with a query similar to first vector
        let query = vec![0b00000001u8, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000];
        let mut dists = vec![0.0f32; 2];
        let mut ids_out = vec![0i64; 2];
        
        index.search(1, &query, 2, &mut dists, &mut ids_out).unwrap();
        
        assert_eq!(ids_out.len(), 2);
        // First result should be vector 0 (closest)
        assert_eq!(ids_out[0], 0);
        // Distance should be 1 (one bit differs)
        assert!((dists[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hamming_distance_correctness() {
        let mut index = BinFlatIndex::new(16, MetricType::Hamming);
        
        // Vector with all zeros
        let v1 = vec![0b00000000u8, 0b00000000];
        // Vector with all ones
        let v2 = vec![0b11111111u8, 0b11111111];
        // Vector with 8 ones
        let v3 = vec![0b11111111u8, 0b00000000];
        
        index.add(3, &[v1, v2, v3].concat(), Some(&[0, 1, 2])).unwrap();
        
        // Query: all zeros
        let query = vec![0b00000000u8, 0b00000000];
        let mut dists = vec![0.0f32; 3];
        let mut ids = vec![0i64; 3];
        
        index.search(1, &query, 3, &mut dists, &mut ids).unwrap();
        
        // Results should be sorted by distance: v1(0), v3(8), v2(16)
        assert_eq!(ids[0], 0); // Distance 0
        assert!((dists[0] - 0.0).abs() < 0.01);
        
        assert_eq!(ids[1], 2); // Distance 8
        assert!((dists[1] - 8.0).abs() < 0.01);
        
        assert_eq!(ids[2], 1); // Distance 16
        assert!((dists[2] - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_bin_flat_reconstruct() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        // 2 vectors * 8 bytes = 16 bytes
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
    fn test_bin_flat_reset() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        let vectors = create_test_vectors(10, 8);
        index.add(10, &vectors, None).unwrap();
        assert_eq!(index.len(), 10);
        
        index.reset();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_bin_flat_remove_ids() {
        let mut index = BinFlatIndex::new(64, MetricType::Hamming);
        
        let vectors = create_test_vectors(5, 8);
        let ids = vec![10, 20, 30, 40, 50];
        index.add(5, &vectors, Some(&ids)).unwrap();
        
        let removed = index.remove_ids(&[20, 40]);
        assert_eq!(removed, 2);
        assert_eq!(index.len(), 3);
        
        assert!(!index.has_id(20));
        assert!(!index.has_id(40));
        assert!(index.has_id(10));
        assert!(index.has_id(30));
        assert!(index.has_id(50));
    }

    #[test]
    fn test_bin_flat_empty_search() {
        let index = BinFlatIndex::new(64, MetricType::Hamming);
        
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
    fn test_bin_flat_large_vectors() {
        let mut index = BinFlatIndex::new(512, MetricType::Hamming); // 64 bytes
        
        let n = 100;
        let vectors = create_test_vectors(n, 64);
        let ids: Vec<i64> = (0..n as i64).collect();
        
        index.add(n as u32, &vectors, Some(&ids)).unwrap();
        assert_eq!(index.len(), n);
        
        // Search
        let query = vec![0u8; 64];
        let mut dists = vec![0.0f32; 10];
        let mut ids_out = vec![0i64; 10];
        
        index.search(1, &query, 10, &mut dists, &mut ids_out).unwrap();
        
        assert_eq!(ids_out.len(), 10);
        // Results should be sorted by distance
        for i in 1..10 {
            assert!(dists[i - 1] <= dists[i]);
        }
    }
}

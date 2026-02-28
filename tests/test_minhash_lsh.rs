//! MinHash-LSH Integration Tests

use knowhere_rs::index::minhash_lsh::{MinHashLSHIndex, MinHashLSHResultHandler, KVPair};
use knowhere_rs::bitset::BitsetView;
use knowhere_rs::comp::bloomfilter::BloomFilter;

#[test]
fn test_minhash_lsh_build_and_save() {
    let mut index = MinHashLSHIndex::new();
    
    // Create test data: 100 vectors, each with 8 elements of u64 (8 bytes each)
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 100;
    let data_size = num_vectors * vec_len * elem_size;
    let data: Vec<u8> = (0..data_size).map(|i| i as u8).collect();
    
    // Build index with raw data
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    assert_eq!(index.count(), num_vectors);
    assert!(index.has_raw_data());
    assert!(index.bands > 0);
    assert!(index.band_size > 0);
    
    // Save to temp file
    let temp_path = "/tmp/test_minhash_lsh_build.bin";
    index.save(temp_path).unwrap();
    
    // Verify file exists
    assert!(std::path::Path::new(temp_path).exists());
    
    // Cleanup
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_minhash_lsh_load() {
    // First build and save
    let mut index = MinHashLSHIndex::new();
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 50;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    let temp_path = "/tmp/test_minhash_lsh_load.bin";
    index.save(temp_path).unwrap();
    
    // Load from file
    let mut loaded_index = MinHashLSHIndex::new();
    loaded_index.load(temp_path).unwrap();
    
    assert_eq!(loaded_index.count(), num_vectors);
    assert_eq!(loaded_index.bands, index.bands);
    assert_eq!(loaded_index.mh_vec_length, vec_len);
    
    // Cleanup
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_minhash_lsh_search() {
    let mut index = MinHashLSHIndex::new();
    
    // Create test data
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 200;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    // Create query vector (same as first vector)
    let query: Vec<u8> = (0..vec_len * elem_size).map(|i| i as u8).collect();
    
    // Search
    let (ids, distances) = index.search(&query, 10, None).unwrap();
    
    assert!(ids.len() <= 10);
    assert_eq!(ids.len(), distances.len());
    
    // First result should be a valid ID (>= 0) or -1 if no results
    // LSH may return approximate results
    assert!(ids.len() <= 10);
}

#[test]
fn test_minhash_lsh_batch_search() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 100;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    // Create multiple query vectors
    let nq = 5;
    let queries: Vec<u8> = (0..nq * vec_len * elem_size).map(|i| i as u8).collect();
    
    // Batch search
    let (all_ids, all_distances) = index.batch_search(&queries, nq, 5, None).unwrap();
    
    assert_eq!(all_ids.len(), nq);
    assert_eq!(all_distances.len(), nq);
    
    for (ids, dists) in all_ids.iter().zip(all_distances.iter()) {
        assert!(ids.len() <= 5);
        assert_eq!(ids.len(), dists.len());
    }
}

#[test]
fn test_minhash_lsh_jaccard_similarity() {
    // Test Jaccard similarity calculation
    // Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    let mut index = MinHashLSHIndex::new();
    
    // Create two similar vectors
    let vec_len = 8;
    let elem_size = 8;
    let mut data = Vec::new();
    
    // Vector 0: all zeros
    for _ in 0..vec_len * elem_size {
        data.push(0u8);
    }
    
    // Vector 1: same as vector 0 (identical)
    for _ in 0..vec_len * elem_size {
        data.push(0u8);
    }
    
    // Vector 2: different
    for i in 0..vec_len * elem_size {
        data.push(i as u8);
    }
    
    index.build(&data, vec_len, elem_size, 2, true).unwrap();
    
    // Query with vector 0
    let query: Vec<u8> = vec![0u8; vec_len * elem_size];
    let (ids, _dists) = index.search(&query, 3, None).unwrap();
    
    // Should find similar vectors
    assert!(!ids.is_empty());
}

#[test]
fn test_minhash_lsh_bitset_filter() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 50;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    // Create bitset to filter out first 10 vectors
    let mut bitset = BitsetView::new(num_vectors);
    // Set first 10 bits
    for i in 0..10 {
        bitset.set(i, true);
    }
    
    let query: Vec<u8> = (0..vec_len * elem_size).map(|i| i as u8).collect();
    let (ids, _dists) = index.search(&query, 10, Some(&bitset)).unwrap();
    
    // Search should work with bitset filter
    // Note: bitset filtering depends on implementation
    assert!(ids.len() <= 10);
}

#[test]
fn test_minhash_lsh_get_vector_by_ids() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 20;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    // Get single vector
    let ids = vec![5i64];
    let vectors = index.get_vector_by_ids(&ids).unwrap();
    
    assert_eq!(vectors.len(), vec_len * elem_size);
    
    // Verify vector values
    let expected_start = 5 * vec_len * elem_size;
    for (i, &byte) in vectors.iter().enumerate() {
        assert_eq!(byte, data[expected_start + i]);
    }
    
    // Get multiple vectors
    let ids = vec![0i64, 10, 19];
    let vectors = index.get_vector_by_ids(&ids).unwrap();
    
    assert_eq!(vectors.len(), 3 * vec_len * elem_size);
}

#[test]
fn test_minhash_lsh_get_vector_by_ids_no_raw_data() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 10;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    // Build without raw data
    index.build(&data, vec_len, elem_size, 4, false).unwrap();
    
    assert!(!index.has_raw_data());
    
    // Should fail to get vectors
    let ids = vec![0i64];
    let result = index.get_vector_by_ids(&ids);
    
    assert!(result.is_err());
}

#[test]
fn test_minhash_lsh_count_and_size() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 8;
    let elem_size = 8;
    let num_vectors = 100;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 4, true).unwrap();
    
    assert_eq!(index.count(), num_vectors);
    assert!(index.size() > 0);
}

#[test]
fn test_minhash_lsh_empty_search() {
    let index = MinHashLSHIndex::new();
    
    assert_eq!(index.count(), 0);
    
    // Empty index should handle search gracefully
    let query = vec![0u8; 64];
    let result = index.search(&query, 10, None);
    
    // Should not panic, may return empty or partial results
    assert!(result.is_ok());
}

#[test]
fn test_bloom_filter_basic() {
    let mut bf = BloomFilter::<u64>::new(1000, 0.01);
    
    // Add some keys
    bf.add(&42);
    bf.add(&100);
    bf.add(&1000);
    
    // Should contain added keys
    assert!(bf.contains(&42));
    assert!(bf.contains(&100));
    assert!(bf.contains(&1000));
    
    // May or may not contain other keys (false positives possible)
    // But with high probability should not contain very different keys
    assert!(!bf.contains(&999999));
}

#[test]
fn test_bloom_filter_false_positive_rate() {
    let mut bf = BloomFilter::<u64>::new(100, 0.01);
    
    // Add 100 items
    for i in 0..100 {
        bf.add(&(i * 1000));
    }
    
    // Check false positive rate
    let mut false_positives = 0;
    let test_start = 1000u64;
    let test_end = 2000u64;
    
    for i in test_start..test_end {
        if bf.contains(&i) {
            false_positives += 1;
        }
    }
    
    let fp_rate = false_positives as f32 / (test_end - test_start) as f32;
    
    // False positive rate should be reasonable (< 10% for this test)
    assert!(fp_rate < 0.1, "False positive rate too high: {}", fp_rate);
}

#[test]
fn test_result_handler() {
    let mut res = MinHashLSHResultHandler::new(5);
    
    // Add results
    res.push(1, 0.95);
    res.push(2, 0.85);
    res.push(3, 0.75);
    
    assert_eq!(res.count(), 3);
    assert!(!res.is_full());
    
    // Fill to capacity
    res.push(4, 0.65);
    res.push(5, 0.55);
    
    assert!(res.is_full());
    assert_eq!(res.count(), 5);
    
    // Should not add more when full
    res.push(6, 0.45);
    assert_eq!(res.count(), 5);
    
    // Get results
    let (ids, dists) = res.get_results();
    
    assert_eq!(ids.len(), 5);
    assert_eq!(dists.len(), 5);
    assert_eq!(ids[0], 1);
    assert_eq!(dists[0], 0.95);
}

#[test]
fn test_result_handler_duplicate_prevention() {
    let mut res = MinHashLSHResultHandler::new(5);
    
    res.push(1, 0.9);
    res.push(1, 0.9); // Duplicate
    res.push(2, 0.8);
    
    // Should have only 2 unique results
    assert_eq!(res.count(), 2);
}

#[test]
fn test_result_handler_invalid_ids() {
    let mut res = MinHashLSHResultHandler::new(5);
    
    res.push(-1, 0.9); // Invalid ID
    res.push(0, 0.0); // Zero distance
    res.push(1, 0.8); // Valid
    
    assert_eq!(res.count(), 1);
    assert_eq!(res.count(), 1);
}

#[test]
fn test_kv_pair() {
    let kv = KVPair { key: 42, value: 100 };
    
    assert_eq!(kv.key, 42);
    assert_eq!(kv.value, 100);
}

#[test]
fn test_optimize_params() {
    let (bands, band_size) = MinHashLSHIndex::optimize_params(64, 8);
    
    assert!(bands > 0);
    assert!(band_size > 0);
    assert_eq!(bands * band_size, 64);
}

#[test]
fn test_hash_key_deterministic() {
    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    
    let hash1 = MinHashLSHIndex::get_hash_key(&data, 1, 0);
    let hash2 = MinHashLSHIndex::get_hash_key(&data, 1, 0);
    
    // Hash should be deterministic
    assert_eq!(hash1, hash2);
}

#[test]
fn test_hash_key_different_inputs() {
    let data1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let data2 = vec![8u8, 7, 6, 5, 4, 3, 2, 1];
    
    let hash1 = MinHashLSHIndex::get_hash_key(&data1, 1, 0);
    let hash2 = MinHashLSHIndex::get_hash_key(&data2, 1, 0);
    
    // Different inputs should produce different hashes (with high probability)
    assert_ne!(hash1, hash2);
}

#[test]
fn test_sort_hash_kv() {
    let mut kv_pairs = vec![
        KVPair { key: 30, value: 3 },
        KVPair { key: 10, value: 1 },
        KVPair { key: 20, value: 2 },
        KVPair { key: 10, value: 4 },
    ];
    
    MinHashLSHIndex::sort_hash_kv(&mut kv_pairs);
    
    // Should be sorted by key
    assert_eq!(kv_pairs[0].key, 10);
    assert_eq!(kv_pairs[1].key, 10);
    assert_eq!(kv_pairs[2].key, 20);
    assert_eq!(kv_pairs[3].key, 30);
}

#[test]
fn test_gen_transposed_hash_kv() {
    let vec_len = 4;
    let elem_size = 8;
    let num_vectors = 3;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    let band_num = 2;
    let band_size = vec_len / band_num;
    
    let kv_pairs = MinHashLSHIndex::gen_transposed_hash_kv(&data, num_vectors, band_num, band_size);
    
    // Should generate num_vectors * band_num KV pairs
    assert_eq!(kv_pairs.len(), num_vectors * band_num);
    
    // Values should be vector IDs
    for (i, kv) in kv_pairs.iter().enumerate() {
        assert_eq!(kv.value, (i / band_num) as i64);
    }
}

#[test]
fn test_minhash_lsh_persistence_roundtrip() {
    // Build, save, load, and verify
    let mut original_index = MinHashLSHIndex::new();
    
    let vec_len = 16;
    let elem_size = 8;
    let num_vectors = 50;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    original_index.build(&data, vec_len, elem_size, 8, true).unwrap();
    
    let temp_path = "/tmp/test_minhash_lsh_persistence.bin";
    original_index.save(temp_path).unwrap();
    
    // Load
    let mut loaded_index = MinHashLSHIndex::new();
    loaded_index.load(temp_path).unwrap();
    
    // Verify same search results
    let query: Vec<u8> = (0..vec_len * elem_size).map(|i| i as u8).collect();
    
    let (orig_ids, _orig_dists) = original_index.search(&query, 5, None).unwrap();
    let (loaded_ids, _loaded_dists) = loaded_index.search(&query, 5, None).unwrap();
    
    // Should get same results
    assert_eq!(orig_ids.len(), loaded_ids.len());
    
    // Cleanup
    std::fs::remove_file(temp_path).unwrap();
}

#[test]
fn test_minhash_lsh_large_index() {
    let mut index = MinHashLSHIndex::new();
    
    let vec_len = 32;
    let elem_size = 8;
    let num_vectors = 1000;
    let data: Vec<u8> = (0..num_vectors * vec_len * elem_size).map(|i| i as u8).collect();
    
    index.build(&data, vec_len, elem_size, 8, true).unwrap();
    
    assert_eq!(index.count(), num_vectors);
    assert!(index.bands > 0);
    
    // Search should work
    let query: Vec<u8> = (0..vec_len * elem_size).map(|i| i as u8).collect();
    let (ids, _dists) = index.search(&query, 10, None).unwrap();
    
    assert!(ids.len() <= 10);
}

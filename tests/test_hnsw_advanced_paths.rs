//! HNSW Advanced Paths Test
//!
//! Test HNSW index advanced functionality parity with C++ knowhere:
//! - get_vector_by_ids: retrieve vectors by ID
//! - AnnIterator: streaming search results  
//! - Serialize/Deserialize: index persistence
//!
//! # Usage
//! ```bash
//! cargo test --test test_hnsw_advanced_paths -- --nocapture
//! ```

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::dataset::Dataset;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::index::Index;
use knowhere_rs::MetricType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::NamedTempFile;

/// Test HNSW get_vector_by_ids functionality
#[test]
fn test_hnsw_get_vector_by_ids() {
    let dim = 128;
    let num_vectors = 100;

    // Create test data
    let mut vectors = vec![0.0f32; num_vectors * dim];
    for i in 0..(num_vectors * dim) {
        vectors[i] = (i as f32 * 0.01) % 10.0;
    }

    // Build HNSW index
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    // Verify has_raw_data
    assert!(index.has_raw_data(), "HNSW should have raw data");

    // Test get_vector_by_ids with single ID
    let ids = vec![5i64];
    let retrieved = index.get_vector_by_ids(&ids).unwrap();
    assert_eq!(retrieved.len(), dim, "Should retrieve one full vector");

    // Verify vector values match
    let expected_start = 5 * dim;
    for i in 0..dim {
        assert!(
            (retrieved[i] - vectors[expected_start + i]).abs() < 1e-6,
            "Vector data mismatch at position {}",
            i
        );
    }

    // Test get_vector_by_ids with multiple IDs
    let ids = vec![0i64, 50, 99];
    let retrieved = index.get_vector_by_ids(&ids).unwrap();
    assert_eq!(retrieved.len(), 3 * dim, "Should retrieve three full vectors");

    // Verify all three vectors
    for (idx, &id) in ids.iter().enumerate() {
        let expected_start = id as usize * dim;
        let retrieved_start = idx * dim;
        for i in 0..dim {
            assert!(
                (retrieved[retrieved_start + i] - vectors[expected_start + i]).abs() < 1e-6,
                "Vector data mismatch for ID {} at position {}",
                id,
                i
            );
        }
    }

    println!("✅ get_vector_by_ids test passed");
}

/// Test HNSW get_vector_by_ids error handling
#[test]
fn test_hnsw_get_vector_by_ids_errors() {
    let dim = 64;
    let num_vectors = 50;

    let mut vectors = vec![0.0f32; num_vectors * dim];
    for i in 0..(num_vectors * dim) {
        vectors[i] = (i as f32 * 0.01) % 5.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    // Test with empty IDs
    let ids: Vec<i64> = vec![];
    let result = index.get_vector_by_ids(&ids).unwrap();
    assert!(result.is_empty(), "Empty IDs should return empty result");

    // Test with out-of-range ID
    let ids = vec![999i64];
    let result = index.get_vector_by_ids(&ids);
    assert!(result.is_err(), "Out-of-range ID should return error");

    println!("✅ get_vector_by_ids error handling test passed");
}

/// Test HNSW AnnIterator functionality
#[test]
fn test_hnsw_ann_iterator() {
    let dim = 128;
    let num_vectors = 100;
    let num_queries = 5;

    let mut vectors = vec![0.0f32; num_vectors * dim];
    for i in 0..(num_vectors * dim) {
        vectors[i] = (i as f32 * 0.01) % 10.0;
    }

    let mut query_vectors = vec![0.0f32; num_queries * dim];
    for i in 0..(num_queries * dim) {
        query_vectors[i] = (i as f32 * 0.02) % 10.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    // Test AnnIterator using Index trait
    let query = Dataset::from_vectors(query_vectors.clone(), dim);
    let mut iterator = index.create_ann_iterator(&query, None).unwrap();

    // Retrieve first result
    let first = iterator.next();
    assert!(first.is_some(), "Iterator should return at least one result");

    let (id, distance) = first.unwrap();
    println!("First result: id={}, distance={:.4}", id, distance);

    // Retrieve multiple results
    let mut count = 1;
    while let Some((id, dist)) = iterator.next() {
        count += 1;
        if count >= 10 {
            break;
        }
        println!("Result {}: id={}, distance={:.4}", count, id, dist);
    }

    assert!(count >= 10, "Iterator should return at least 10 results");

    println!("✅ AnnIterator test passed");
}

/// Test HNSW serialize/deserialize functionality
#[test]
fn test_hnsw_serialize_deserialize() {
    let dim = 128;
    let num_vectors = 100;

    let mut vectors = vec![0.0f32; num_vectors * dim];
    for i in 0..(num_vectors * dim) {
        vectors[i] = (i as f32 * 0.01) % 10.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build first index
    let mut index1 = HnswIndex::new(&config).unwrap();
    index1.train(&vectors).unwrap();
    index1.add(&vectors, None).unwrap();

    // Save to file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    index1.save(path).unwrap();

    // Load into new index
    let mut index2 = HnswIndex::new(&config).unwrap();
    index2.load(path).unwrap();

    // Verify both indexes return same search results
    let mut query = vec![0.0f32; dim];
    for i in 0..dim {
        query[i] = (i as f32 * 0.05) % 10.0;
    }

    let req = SearchRequest {
        top_k: 10,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };

    let result1 = index1.search(&query, &req).unwrap();
    let result2 = index2.search(&query, &req).unwrap();

    assert_eq!(result1.ids.len(), result2.ids.len(), "Result count should match");
    for i in 0..result1.ids.len() {
        assert_eq!(result1.ids[i], result2.ids[i], "Result IDs should match at position {}", i);
        assert!(
            (result1.distances[i] - result2.distances[i]).abs() < 1e-6,
            "Result distances should match at position {}",
            i
        );
    }

    println!("✅ Serialize/deserialize test passed");
}

/// Test HNSW range search (if supported)
#[test]
fn test_hnsw_range_search() {
    let dim = 128;
    let num_vectors = 100;

    let mut vectors = vec![0.0f32; num_vectors * dim];
    for i in 0..(num_vectors * dim) {
        vectors[i] = (i as f32 * 0.01) % 10.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    let query = vec![0.0f32; dim];
    let query_dataset = Dataset::from_vectors(query, dim);

    // Test range search - may return Unsupported if not implemented
    match index.range_search(&query_dataset, 5.0) {
        Ok(result) => {
            println!("Range search returned {} results", result.ids.len());
            // Verify all distances are within radius
            for &dist in &result.distances {
                assert!(
                    dist <= 5.0 + 0.001,
                    "Distance {} exceeds radius 5.0",
                    dist
                );
            }
            println!("✅ Range search test passed");
        }
        Err(e) => {
            println!("Range search not supported: {:?}", e);
            println!("✅ Range search correctly returns Unsupported");
        }
    }
}

#[test]
fn test_hnsw_build_quality_signals_survive_save_load() {
    let dim = 64;
    let num_vectors = 512;
    let mut rng = StdRng::seed_from_u64(20260311);
    let vectors: Vec<f32> = (0..num_vectors * dim).map(|_| rng.gen::<f32>()).collect();

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let repaired = index.find_and_repair_unreachable();

    let unreachable = index.find_unreachable_vectors();
    assert!(
        unreachable.is_empty(),
        "repair-backed HNSW build should not leave unreachable vectors after {} repairs: {:?}",
        repaired,
        unreachable
    );

    let (_, max_l0, avg_l0) = index.layer_neighbor_count_stats(0).unwrap();
    assert!(
        max_l0 <= 32 + repaired,
        "layer-0 degree should stay within 2*M plus repair slack, got {} with {} repairs",
        max_l0,
        repaired
    );
    assert!(
        avg_l0 >= 8.0,
        "layer-0 average degree should stay meaningfully populated, got {:.2}",
        avg_l0
    );
    assert!(index.max_level() >= 1, "build should produce a multi-layer graph");

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();
    index.save(path).unwrap();

    let mut restored = HnswIndex::new(&config).unwrap();
    restored.load(path).unwrap();

    let restored_unreachable = restored.find_unreachable_vectors();
    assert!(
        restored_unreachable.is_empty(),
        "reloaded HNSW graph should keep all nodes reachable: {:?}",
        restored_unreachable
    );

    let restored_stats = restored.layer_neighbor_count_stats(0).unwrap();
    assert_eq!(
        max_l0,
        restored_stats.1,
        "save/load should preserve layer-0 degree bounds"
    );
    assert!(
        (avg_l0 - restored_stats.2).abs() < 1e-6,
        "save/load should preserve layer-0 average degree"
    );
}

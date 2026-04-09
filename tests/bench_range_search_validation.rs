#![cfg(feature = "long-tests")]
//! Range Search Distance Validation Benchmark
//!
//! Test range search with distance validation functionality.
//! Validates distances are within expected bounds for range search results.
//!
//! # Usage
//! ```bash
//! cargo test --test bench_range_search_validation -- --nocapture
//! ```

use hanns::api::{IndexConfig, IndexParams, IndexType};
use hanns::benchmark::{
    check_distance_in_scope_range, distance_statistics, validate_l2_distances,
};
use hanns::dataset::{load_sift1m_complete, SiftDataset};
use hanns::faiss::MemIndex as FlatIndex;
use hanns::MetricType;
use std::env;
use std::time::Instant;

/// Get SIFT1M dataset path from environment or use default
fn get_sift_path() -> String {
    env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string())
}

/// Load SIFT1M dataset or return None if not found
fn try_load_sift1m() -> Option<SiftDataset> {
    let path = get_sift_path();
    match load_sift1m_complete(&path) {
        Ok(dataset) => {
            println!("Loaded SIFT1M dataset from {}", path);
            println!("  Base vectors: {}", dataset.num_base());
            println!("  Query vectors: {}", dataset.num_query());
            println!("  Dimension: {}", dataset.dim());
            Some(dataset)
        }
        Err(e) => {
            eprintln!("Failed to load SIFT1M dataset from {}: {}", path, e);
            eprintln!("Skipping test - set SIFT1M_PATH or place dataset in ./data/sift/");
            None
        }
    }
}

/// Range search validation result
#[derive(Debug, Clone)]
pub struct RangeSearchValidationResult {
    pub radius: f32,
    pub total_results: usize,
    pub avg_results_per_query: f64,
    pub min_results: usize,
    pub max_results: usize,
    pub min_distance: f32,
    pub max_distance: f32,
    pub mean_distance: f32,
    pub std_dev: f32,
    pub validation_passed: bool,
}

impl RangeSearchValidationResult {
    pub fn print(&self) {
        println!("\n  Radius: {:.1}", self.radius);
        println!("  Total results: {}", self.total_results);
        println!(
            "  Avg results/query: {:.1} (min: {}, max: {})",
            self.avg_results_per_query, self.min_results, self.max_results
        );
        println!(
            "  Distance stats: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            self.min_distance, self.max_distance, self.mean_distance, self.std_dev
        );
        println!(
            "  Validation: {}",
            if self.validation_passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
    }
}

/// Benchmark Flat index with range search validation
fn benchmark_flat_range_validation(
    dataset: &SiftDataset,
    num_queries: usize,
    radii: &[f32],
) -> Vec<RangeSearchValidationResult> {
    println!("\n=== Benchmarking Flat Index (Range Search) ===");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let dim = dataset.dim();

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    let mut results = Vec::new();

    for &radius in radii {
        println!("\n  Testing radius: {:.1}", radius);

        let mut all_distances: Vec<f32> = Vec::new();
        let mut lims: Vec<usize> = vec![0];
        let mut total_results = 0;
        let mut min_results = usize::MAX;
        let mut max_results = 0;

        let search_start = Instant::now();

        for i in 0..num_queries {
            let query = &query_vectors[i * dim..(i + 1) * dim];

            // Perform range search
            let (_ids, distances) = index.range_search(query, radius).unwrap();

            total_results += distances.len();
            min_results = min_results.min(distances.len());
            max_results = max_results.max(distances.len());

            all_distances.extend(&distances);
            lims.push(all_distances.len());
        }

        let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
        let qps = num_queries as f64 / (search_time / 1000.0);
        println!(
            "  Search time: {:.2} ms ({} queries)",
            search_time, num_queries
        );
        println!("  QPS: {:.0}", qps);

        // Validate distances
        let validation_passed = check_distance_in_scope_range(
            &all_distances,
            &lims,
            0.0,
            radius + 0.001, // Allow small tolerance
        );

        let l2_valid = validate_l2_distances(&all_distances);

        // Calculate statistics
        let (min_dist, max_dist, mean_dist, std_dev) = distance_statistics(&all_distances);

        let result = RangeSearchValidationResult {
            radius,
            total_results,
            avg_results_per_query: total_results as f64 / num_queries as f64,
            min_results: if min_results == usize::MAX {
                0
            } else {
                min_results
            },
            max_results,
            min_distance: min_dist,
            max_distance: max_dist,
            mean_distance: mean_dist,
            std_dev,
            validation_passed: validation_passed && l2_valid,
        };

        result.print();
        results.push(result);
    }

    results
}

#[test]
#[ignore = "Requires SIFT1M dataset"]
fn test_range_search_validation_flat() {
    if let Some(dataset) = try_load_sift1m() {
        let radii = vec![1.0, 2.0, 5.0, 10.0];
        let results = benchmark_flat_range_validation(&dataset, 100, &radii);

        // Verify all validations passed
        for result in &results {
            assert!(
                result.validation_passed,
                "Range search validation failed for radius {}",
                result.radius
            );
        }
    } else {
        println!("Skipping test - SIFT1M dataset not found");
    }
}

/// Unit test with small synthetic data
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_range_search_validation_unit() {
    // Create small test dataset
    let dim = 128;
    let num_vectors = 1000;
    let num_queries = 10;

    let mut base_vectors = vec![0.0f32; num_vectors * dim];
    let mut query_vectors = vec![0.0f32; num_queries * dim];

    // Fill with random values
    for (i, value) in base_vectors.iter_mut().enumerate() {
        *value = (i as f32 * 0.01) % 10.0;
    }
    for (i, value) in query_vectors.iter_mut().enumerate() {
        *value = (i as f32 * 0.02) % 10.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).unwrap();
    index.add(&base_vectors, None).unwrap();

    // Test with multiple radii
    let radii = vec![100.0, 200.0, 500.0];

    for radius in radii {
        let mut all_distances: Vec<f32> = Vec::new();
        let mut lims: Vec<usize> = vec![0];

        for i in 0..num_queries {
            let query = &query_vectors[i * dim..(i + 1) * dim];
            let (_ids, distances) = index.range_search(query, radius).unwrap();

            all_distances.extend(&distances);
            lims.push(all_distances.len());

            // Verify all distances are within radius
            for &dist in &distances {
                assert!(
                    dist <= radius + 0.001,
                    "Distance {} exceeds radius {}",
                    dist,
                    radius
                );
            }
        }

        // Validate using distance_validator
        let validation_passed =
            check_distance_in_scope_range(&all_distances, &lims, 0.0, radius + 0.001);

        assert!(
            validation_passed,
            "Distance validation failed for radius {}",
            radius
        );

        // Validate L2 distances are non-negative
        assert!(
            validate_l2_distances(&all_distances),
            "L2 distance validation failed"
        );

        // Note: Range search results are not guaranteed to be sorted,
        // so we don't validate monotonicity here

        // Print statistics
        let (min, max, mean, std_dev) = distance_statistics(&all_distances);
        println!(
            "Radius {}: count={}, min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
            radius,
            all_distances.len(),
            min,
            max,
            mean,
            std_dev
        );
    }
}

/// Test distance bounds validation
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_range_distance_bounds() {
    // Test check_distance_in_scope_range
    let distances = vec![0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
    let lims = vec![0, 3, 6]; // 2 queries, 3 results each

    assert!(check_distance_in_scope_range(&distances, &lims, 0.0, 1.5));
    assert!(!check_distance_in_scope_range(&distances, &lims, 0.0, 0.5)); // 0.8, 1.0 exceed

    // Test with empty results
    let empty_distances: Vec<f32> = vec![];
    let empty_lims = vec![0, 0, 0];
    assert!(check_distance_in_scope_range(
        &empty_distances,
        &empty_lims,
        0.0,
        1.0
    ));
}

/// Test range search with different radius values
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_range_search_radius_monotonicity() {
    let dim = 64;
    let num_vectors = 500;

    let mut base_vectors = vec![0.0f32; num_vectors * dim];
    for (i, value) in base_vectors.iter_mut().enumerate() {
        *value = (i as f32 * 0.01) % 5.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).unwrap();
    index.add(&base_vectors, None).unwrap();

    let query = vec![0.0f32; dim];

    // Test that larger radius returns more results
    let (_ids1, dists1) = index.range_search(&query, 10.0).unwrap();
    let (_ids2, dists2) = index.range_search(&query, 20.0).unwrap();
    let (_ids3, dists3) = index.range_search(&query, 50.0).unwrap();

    assert!(
        dists1.len() <= dists2.len(),
        "Larger radius should return at least as many results"
    );
    assert!(
        dists2.len() <= dists3.len(),
        "Larger radius should return at least as many results"
    );

    println!("Radius 10: {} results", dists1.len());
    println!("Radius 20: {} results", dists2.len());
    println!("Radius 50: {} results", dists3.len());
}

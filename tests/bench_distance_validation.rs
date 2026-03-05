//! Distance Validation Benchmark Test
//!
//! Test distance validation functionality on search results.
//! Validates distances are within expected bounds and properly sorted.
//!
//! # Usage
//! ```bash
//! cargo test --test bench_distance_validation -- --nocapture
//! ```

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, distance_statistics, validate_distance_monotonicity,
    validate_l2_distances, DistanceValidationReport,
};
use knowhere_rs::dataset::{load_sift1m_complete, SiftDataset};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use std::env;
use std::time::Instant;

/// Get SIFT1M dataset path from environment or use default
fn get_sift_path() -> String {
    env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string())
}

/// Load or return None if dataset not found
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
            eprintln!("Set SIFT1M_PATH environment variable or place dataset in ./data/sift/");
            None
        }
    }
}

/// Benchmark Flat index with distance validation
fn benchmark_flat_validation(dataset: &SiftDataset, num_queries: usize, top_k: usize) {
    println!("\n=== Benchmarking Flat Index ===");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;
    let dim = dataset.dim();

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    // Search and collect results
    let mut all_distances: Vec<f32> = Vec::new();
    let mut all_results: Vec<Vec<i64>> = Vec::new();
    let search_start = Instant::now();

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_distances.extend(&result.distances);
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Validate distances
    println!("\n  Distance Validation:");
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        top_k,
        true,      // L2 metric
        0.0,       // low bound
        1000000.0, // high bound
    );
    report.print();

    // Calculate recall
    let recall_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("  Recall@1: {:.4}", recall_1);
    println!("  Recall@10: {:.4}", recall_10);
    println!("  Recall@100: {:.4}", recall_100);

    // Summary
    println!("\n  Summary:");
    println!(
        "    All validations passed: {}",
        if report.all_passed() {
            "✅ YES"
        } else {
            "❌ NO"
        }
    );
    println!(
        "    Build: {:.2} ms, Search: {:.2} ms, QPS: {:.0}",
        build_time, search_time, qps
    );
    println!(
        "    Recall@1/10/100: {:.3}/{:.3}/{:.3}",
        recall_1, recall_10, recall_100
    );
}

/// Benchmark HNSW index with distance validation
fn benchmark_hnsw_validation(dataset: &SiftDataset, num_queries: usize, top_k: usize) {
    println!("\n=== Benchmarking HNSW Index ===");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;
    let dim = dataset.dim();

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::hnsw(16, 64, 0.5),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    // Search and collect results
    let mut all_distances: Vec<f32> = Vec::new();
    let mut all_results: Vec<Vec<i64>> = Vec::new();
    let search_start = Instant::now();

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe: 64,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_distances.extend(&result.distances);
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Validate distances
    println!("\n  Distance Validation:");
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        top_k,
        true,
        0.0,
        1000000.0,
    );
    report.print();

    // Calculate recall
    let recall_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("  Recall@1: {:.4}", recall_1);
    println!("  Recall@10: {:.4}", recall_10);
    println!("  Recall@100: {:.4}", recall_100);
}

/// Benchmark IVF-Flat index with distance validation
fn benchmark_ivf_flat_validation(dataset: &SiftDataset, num_queries: usize, top_k: usize) {
    println!("\n=== Benchmarking IVF-Flat Index ===");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;
    let dim = dataset.dim();

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::ivf(256, 20),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    // Search and collect results
    let mut all_distances: Vec<f32> = Vec::new();
    let mut all_results: Vec<Vec<i64>> = Vec::new();
    let search_start = Instant::now();

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe: 20,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_distances.extend(&result.distances);
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Validate distances
    println!("\n  Distance Validation:");
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        top_k,
        true,
        0.0,
        1000000.0,
    );
    report.print();

    // Calculate recall
    let recall_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("  Recall@1: {:.4}", recall_1);
    println!("  Recall@10: {:.4}", recall_10);
    println!("  Recall@100: {:.4}", recall_100);
}

#[test]
#[ignore = "Requires SIFT1M dataset"]
fn test_distance_validation_flat() {
    if let Some(dataset) = try_load_sift1m() {
        benchmark_flat_validation(&dataset, 100, 100);
    } else {
        println!("Skipping test - SIFT1M dataset not found");
    }
}

#[test]
#[ignore = "Requires SIFT1M dataset"]
fn test_distance_validation_hnsw() {
    if let Some(dataset) = try_load_sift1m() {
        benchmark_hnsw_validation(&dataset, 100, 100);
    } else {
        println!("Skipping test - SIFT1M dataset not found");
    }
}

#[test]
#[ignore = "Requires SIFT1M dataset"]
fn test_distance_validation_ivf_flat() {
    if let Some(dataset) = try_load_sift1m() {
        benchmark_ivf_flat_validation(&dataset, 100, 100);
    } else {
        println!("Skipping test - SIFT1M dataset not found");
    }
}

/// Test distance validation on small synthetic data
#[test]
fn test_distance_validation_unit() {
    // Create small test dataset
    let dim = 128;
    let num_vectors = 1000;
    let num_queries = 10;
    let top_k = 10;

    let mut base_vectors = vec![0.0f32; num_vectors * dim];
    let mut query_vectors = vec![0.0f32; num_queries * dim];

    // Fill with random values
    for i in 0..(num_vectors * dim) {
        base_vectors[i] = (i as f32 * 0.01) % 10.0;
    }
    for i in 0..(num_queries * dim) {
        query_vectors[i] = (i as f32 * 0.02) % 10.0;
    }

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).unwrap();
    index.add(&base_vectors, None).unwrap();

    // Search and collect distances
    let mut all_distances: Vec<f32> = Vec::new();
    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_distances.extend(&result.distances);
    }

    // Validate
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        top_k,
        true, // L2
        0.0,
        10000.0,
    );

    report.print();
    assert!(report.all_passed(), "Distance validation should pass");

    // Additional checks
    assert!(
        validate_l2_distances(&all_distances),
        "L2 distances should be non-negative"
    );
    assert!(
        validate_distance_monotonicity(&all_distances, num_queries, top_k, true),
        "Distances should be sorted (non-decreasing for L2)"
    );

    let (min, max, mean, std_dev) = distance_statistics(&all_distances);
    println!(
        "Distance stats: min={:.4}, max={:.4}, mean={:.4}, std_dev={:.4}",
        min, max, mean, std_dev
    );
    assert!(min >= 0.0, "Minimum distance should be non-negative");
}

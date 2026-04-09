#![cfg(feature = "long-tests")]
//! Random100K Benchmark Test
//!
//! Benchmark Flat, HNSW, and IVF-Flat indexes on 100K random vectors.
//! This is a medium-scale test that doesn't require external datasets.
//!
//! # Usage
//! ```bash
//! cargo test --test bench_random100k -- --nocapture
//! ```

mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_vector_memory, BenchmarkResult, DistanceValidationReport,
    MemoryTracker,
};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

/// Generate random dataset
fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-1.0..1.0));
    }

    data
}

/// Generate ground truth for random dataset (brute-force)


/// Benchmark Flat index
fn benchmark_flat(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\nBenchmarking Flat index...");

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(base.len() / dim, dim);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = (base.len() * 4) as u64;
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Use only the first num_queries ground truth entries
    let gt_slice = &ground_truth[..num_queries];
    let recall_at_1 = average_recall_at_k(&all_results, gt_slice, 1);
    let recall_at_10 = average_recall_at_k(&all_results, gt_slice, 10);
    let recall_at_100 = average_recall_at_k(&all_results, gt_slice, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark HNSW index
fn benchmark_hnsw(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\nBenchmarking HNSW index...");

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(base.len() / dim, dim);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::hnsw(16, 200, 0.5),
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = (base.len() * 16 * 8) as u64;
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Use only the first num_queries ground truth entries
    let gt_slice = &ground_truth[..num_queries];
    let recall_at_1 = average_recall_at_k(&all_results, gt_slice, 1);
    let recall_at_10 = average_recall_at_k(&all_results, gt_slice, 10);
    let recall_at_100 = average_recall_at_k(&all_results, gt_slice, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "HNSW".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark IVF-Flat index
fn benchmark_ivf_flat(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\nBenchmarking IVF-Flat index...");

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(base.len() / dim, dim);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::ivf(100, 20),
    };

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = (base.len() * 8) as u64;
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 20,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Use only the first num_queries ground truth entries
    let gt_slice = &ground_truth[..num_queries];
    let recall_at_1 = average_recall_at_k(&all_results, gt_slice, 1);
    let recall_at_10 = average_recall_at_k(&all_results, gt_slice, 10);
    let recall_at_100 = average_recall_at_k(&all_results, gt_slice, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "IVF-Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_random100k_benchmark() {
    const NUM_BASE: usize = 100_000;
    const NUM_QUERY: usize = 100; // Reduced for faster execution
    const DIM: usize = 128;
    const K: usize = 100;

    println!("=== Random100K Benchmark ===");
    println!("Base vectors: {}", NUM_BASE);
    println!("Query vectors: {}", NUM_QUERY);
    println!("Dimension: {}", DIM);
    println!("Top-K: {}", K);

    // Generate datasets
    println!("\nGenerating random dataset...");
    let base = generate_random_dataset(NUM_BASE, DIM);
    let query = generate_random_dataset(NUM_QUERY, DIM);

    // Compute ground truth
    println!("Computing ground truth (brute-force)...");
    let gt_start = Instant::now();
    let ground_truth = common::compute_ground_truth(&base, &query, NUM_QUERY, DIM, K);
    let gt_time = gt_start.elapsed().as_secs_f64();
    println!("  Ground truth time: {:.2}s", gt_time);

    // Run benchmarks
    let flat_result = benchmark_flat(&base, &query, &ground_truth, NUM_QUERY, DIM);
    let hnsw_result = benchmark_hnsw(&base, &query, &ground_truth, NUM_QUERY, DIM);
    let ivf_result = benchmark_ivf_flat(&base, &query, &ground_truth, NUM_QUERY, DIM);

    // Print summary
    println!("\n=== Benchmark Summary ===");
    BenchmarkResult::print_table(&[flat_result.clone(), hnsw_result.clone(), ivf_result.clone()]);
    BenchmarkResult::print_markdown_table(
        &[flat_result.clone(), hnsw_result.clone(), ivf_result.clone()],
        "Random100K",
    );

    assert!(
        flat_result.recall_at_100 > 0.99,
        "Flat index should have near-perfect recall"
    );
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_random100k_quick() {
    // Quick test with fewer queries
    const NUM_BASE: usize = 100_000;
    const NUM_QUERY: usize = 100;
    const DIM: usize = 128;

    println!("=== Random100K Quick Test ===");

    let base = generate_random_dataset(NUM_BASE, DIM);
    let query = generate_random_dataset(NUM_QUERY, DIM);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).unwrap();
    index.add(&base, None).unwrap();

    let req = SearchRequest {
        top_k: 10,
        ..Default::default()
    };

    let result = index.search(&query[0..DIM], &req).unwrap();
    assert_eq!(result.ids.len(), 10);

    println!("Quick test passed!");
}

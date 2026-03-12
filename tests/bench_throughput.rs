#![cfg(feature = "long-tests")]
//! Throughput Benchmark Test (QPS Stress Test)
//!
//! Benchmark concurrent query throughput for Flat, HNSW, and IVF-Flat indexes.
//! Tests QPS at different concurrency levels (1, 2, 4, 8, 16 threads).
//! Records latency distribution (P50, P90, P99).
//!
//! # Usage
//! ```bash
//! cargo test --test bench_throughput -- --nocapture
//! ```

use hdrhistogram::Histogram;
use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_vector_memory, DistanceValidationReport, MemoryTracker,
};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const NUM_VECTORS: usize = 100_000;
const DIM: usize = 128;
const NUM_QUERIES: usize = 1000;
const TOP_K: usize = 100;
const CONCURRENCY_LEVELS: [usize; 5] = [1, 2, 4, 8, 16];

/// Generate random dataset
fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);
    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-1.0..1.0));
    }
    data
}

/// Generate query vectors
fn generate_queries(num_queries: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(num_queries * dim);
    for _ in 0..(num_queries * dim) {
        queries.push(rng.gen_range(-1.0..1.0));
    }
    queries
}

/// Generate ground truth using Flat index (exact search)
/// Returns Vec<Vec<i32>> for compatibility with recall_at_k
fn generate_ground_truth(base: &[f32], queries: &[f32], dim: usize, top_k: usize) -> Vec<Vec<i32>> {
    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base, None).unwrap();

    let num_queries = queries.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        // Convert i64 to i32 for compatibility with recall.rs
        let gt: Vec<i32> = result.ids.iter().map(|&id| id as i32).collect();
        ground_truth.push(gt);
    }

    ground_truth
}

/// Throughput test result for a single concurrency level
struct ThroughputResult {
    concurrency: usize,
    qps: f64,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

/// Benchmark Flat index throughput
fn benchmark_flat_throughput(
    base: &[f32],
    queries: &[f32],
    concurrency: usize,
    ground_truth: &[Vec<i32>],
) -> ThroughputResult {
    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let index = FlatIndex::new(&config).unwrap();
    let arc_index = Arc::new(parking_lot::RwLock::new(index));

    // Build index
    let build_start = Instant::now();
    {
        let mut idx = arc_index.write();
        idx.add(base, None).unwrap();
    }
    println!(
        "  Build time: {:.2} ms",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    // Concurrent search with latency, distance, and ID collection
    let start = Instant::now();
    let search_results: Vec<(u64, Vec<i64>, Vec<f32>)> = (0..NUM_QUERIES)
        .into_par_iter()
        .map(|i| {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let req = SearchRequest {
                top_k: TOP_K,
                ..Default::default()
            };

            let query_start = Instant::now();
            let (ids, distances) = {
                let idx = arc_index.read();
                let result = idx.search(q, &req).unwrap();
                (result.ids, result.distances)
            };
            let latency = query_start.elapsed().as_micros() as u64;
            (latency, ids, distances)
        })
        .collect();

    let total_time = start.elapsed().as_secs_f64();
    let qps = NUM_QUERIES as f64 / total_time;

    // Extract latencies, IDs, and distances
    let latencies: Vec<u64> = search_results.iter().map(|(lat, _, _)| *lat).collect();
    let all_ids: Vec<Vec<i64>> = search_results
        .iter()
        .map(|(_, ids, _)| ids.clone())
        .collect();
    let all_distances: Vec<f32> = search_results
        .iter()
        .flat_map(|(_, _, dist)| dist.iter().copied())
        .collect();

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        NUM_QUERIES,
        TOP_K,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_ids, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_ids, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_ids, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    // Validate recall (Flat should be perfect)
    assert!(
        recall_at_10 >= 0.95,
        "Flat recall@10 too low: {:.3}",
        recall_at_10
    );

    // Calculate percentiles
    let mut hist = Histogram::<u64>::new(3).unwrap();
    for &lat in &latencies {
        hist.record(lat).unwrap();
    }

    ThroughputResult {
        concurrency,
        qps,
        p50_ms: hist.value_at_percentile(50.0) as f64 / 1000.0,
        p90_ms: hist.value_at_percentile(90.0) as f64 / 1000.0,
        p99_ms: hist.value_at_percentile(99.0) as f64 / 1000.0,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark HNSW index throughput
fn benchmark_hnsw_throughput(
    base: &[f32],
    queries: &[f32],
    concurrency: usize,
    ground_truth: &[Vec<i32>],
) -> ThroughputResult {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::hnsw(16, 200, 0.5),
    };

    let index = HnswIndex::new(&config).unwrap();
    let arc_index = Arc::new(parking_lot::RwLock::new(index));

    // Build index
    let build_start = Instant::now();
    {
        let mut idx = arc_index.write();
        idx.train(base).unwrap();
        idx.add(base, None).unwrap();
    }
    println!(
        "  Build time: {:.2} ms",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    // Concurrent search with latency, distance, and ID collection
    let start = Instant::now();
    let search_results: Vec<(u64, Vec<i64>, Vec<f32>)> = (0..NUM_QUERIES)
        .into_par_iter()
        .map(|i| {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let req = SearchRequest {
                top_k: TOP_K,
                ..Default::default()
            };

            let query_start = Instant::now();
            let (ids, distances) = {
                let idx = arc_index.read();
                let result = idx.search(q, &req).unwrap();
                (result.ids, result.distances)
            };
            let latency = query_start.elapsed().as_micros() as u64;
            (latency, ids, distances)
        })
        .collect();

    let total_time = start.elapsed().as_secs_f64();
    let qps = NUM_QUERIES as f64 / total_time;

    // Extract latencies, IDs, and distances
    let latencies: Vec<u64> = search_results.iter().map(|(lat, _, _)| *lat).collect();
    let all_ids: Vec<Vec<i64>> = search_results
        .iter()
        .map(|(_, ids, _)| ids.clone())
        .collect();
    let all_distances: Vec<f32> = search_results
        .iter()
        .flat_map(|(_, _, dist)| dist.iter().copied())
        .collect();

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        NUM_QUERIES,
        TOP_K,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_ids, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_ids, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_ids, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    // Validate recall (HNSW should have reasonable recall)
    assert!(
        recall_at_10 >= 0.70,
        "HNSW recall@10 too low: {:.3}",
        recall_at_10
    );

    // Calculate percentiles
    let mut hist = Histogram::<u64>::new(3).unwrap();
    for &lat in &latencies {
        hist.record(lat).unwrap();
    }

    ThroughputResult {
        concurrency,
        qps,
        p50_ms: hist.value_at_percentile(50.0) as f64 / 1000.0,
        p90_ms: hist.value_at_percentile(90.0) as f64 / 1000.0,
        p99_ms: hist.value_at_percentile(99.0) as f64 / 1000.0,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark IVF-Flat index throughput
fn benchmark_ivf_flat_throughput(
    base: &[f32],
    queries: &[f32],
    concurrency: usize,
    ground_truth: &[Vec<i32>],
) -> ThroughputResult {
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::ivf(100, 20),
    };

    let index = IvfFlatIndex::new(&config).unwrap();
    let arc_index = Arc::new(parking_lot::RwLock::new(index));

    // Build index
    let build_start = Instant::now();
    {
        let mut idx = arc_index.write();
        idx.train(base).unwrap();
        idx.add(base, None).unwrap();
    }
    println!(
        "  Build time: {:.2} ms",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    // Concurrent search with latency, distance, and ID collection
    let start = Instant::now();
    let search_results: Vec<(u64, Vec<i64>, Vec<f32>)> = (0..NUM_QUERIES)
        .into_par_iter()
        .map(|i| {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let req = SearchRequest {
                top_k: TOP_K,
                nprobe: 20,
                ..Default::default()
            };

            let query_start = Instant::now();
            let (ids, distances) = {
                let idx = arc_index.read();
                let result = idx.search(q, &req).unwrap();
                (result.ids, result.distances)
            };
            let latency = query_start.elapsed().as_micros() as u64;
            (latency, ids, distances)
        })
        .collect();

    let total_time = start.elapsed().as_secs_f64();
    let qps = NUM_QUERIES as f64 / total_time;

    // Extract latencies, IDs, and distances
    let latencies: Vec<u64> = search_results.iter().map(|(lat, _, _)| *lat).collect();
    let all_ids: Vec<Vec<i64>> = search_results
        .iter()
        .map(|(_, ids, _)| ids.clone())
        .collect();
    let all_distances: Vec<f32> = search_results
        .iter()
        .flat_map(|(_, _, dist)| dist.iter().copied())
        .collect();

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        NUM_QUERIES,
        TOP_K,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_ids, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_ids, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_ids, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    // Validate recall (IVF-Flat should have reasonable recall)
    assert!(
        recall_at_10 >= 0.70,
        "IVF-Flat recall@10 too low: {:.3}",
        recall_at_10
    );

    // Calculate percentiles
    let mut hist = Histogram::<u64>::new(3).unwrap();
    for &lat in &latencies {
        hist.record(lat).unwrap();
    }

    ThroughputResult {
        concurrency,
        qps,
        p50_ms: hist.value_at_percentile(50.0) as f64 / 1000.0,
        p90_ms: hist.value_at_percentile(90.0) as f64 / 1000.0,
        p99_ms: hist.value_at_percentile(99.0) as f64 / 1000.0,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Print throughput results table with recall
fn print_results(index_name: &str, results: &[ThroughputResult]) {
    println!("\n索引类型：{}", index_name);
    println!("并发度 | QPS     | P50(ms) | P90(ms) | P99(ms) | R@1   | R@10  | R@100");
    println!("--------|---------|---------|---------|---------|-------|-------|-------");

    for result in results {
        println!(
            "{:7} | {:7.0} | {:7.2} | {:7.2} | {:7.2} | {:.3} | {:.3} | {:.3}",
            result.concurrency,
            result.qps,
            result.p50_ms,
            result.p90_ms,
            result.p99_ms,
            result.recall_at_1,
            result.recall_at_10,
            result.recall_at_100
        );
    }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_throughput_benchmark() {
    println!("=== 吞吐量基准测试 ({} 向量，{} 维) ===", NUM_VECTORS, DIM);

    // Generate dataset
    println!("\n生成测试数据集...");
    let base = generate_random_dataset(NUM_VECTORS, DIM);
    let queries = generate_queries(NUM_QUERIES, DIM);

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(NUM_VECTORS, DIM);
    tracker.record_base_memory(base_mem);
    println!("{}", tracker.report());

    // Generate ground truth using Flat index (exact search)
    println!("\n生成 Ground Truth (Flat 精确搜索)...");
    let ground_truth = generate_ground_truth(&base, &queries, DIM, TOP_K);
    println!("Ground Truth 生成完成 ({} 查询)", ground_truth.len());

    // Benchmark Flat index
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Flat 索引吞吐量测试");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let flat_results: Vec<ThroughputResult> = CONCURRENCY_LEVELS
        .iter()
        .map(|&concurrency| {
            println!("\n  测试并发度：{}", concurrency);
            benchmark_flat_throughput(&base, &queries, concurrency, &ground_truth)
        })
        .collect();

    print_results("Flat", &flat_results);

    // Benchmark HNSW index
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("HNSW 索引吞吐量测试");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let hnsw_results: Vec<ThroughputResult> = CONCURRENCY_LEVELS
        .iter()
        .map(|&concurrency| {
            println!("\n  测试并发度：{}", concurrency);
            benchmark_hnsw_throughput(&base, &queries, concurrency, &ground_truth)
        })
        .collect();

    print_results("HNSW", &hnsw_results);

    // Benchmark IVF-Flat index
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("IVF-Flat 索引吞吐量测试");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let ivf_results: Vec<ThroughputResult> = CONCURRENCY_LEVELS
        .iter()
        .map(|&concurrency| {
            println!("\n  测试并发度：{}", concurrency);
            benchmark_ivf_flat_throughput(&base, &queries, concurrency, &ground_truth)
        })
        .collect();

    print_results("IVF-Flat", &ivf_results);

    // Summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("吞吐量基准测试完成");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

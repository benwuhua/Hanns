#![cfg(feature = "long-tests")]
//! GIST1M Benchmark Test
//!
//! Benchmark Flat, HNSW, and IVF-Flat indexes on GIST1M dataset.
//!
//! # Usage
//! ```bash
//! cargo test --test bench_gist1m -- --nocapture
//! ```
//!
//! # Dataset
//! Download GIST1M from: http://corpus-texmex.irisa.fr/
//! - base.fvecs: 1M base vectors (960D)
//! - query.fvecs: 1K query vectors (960D)
//! - groundtruth.ivecs: 1K x 100 ground truth neighbors

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_hnsw_overhead, estimate_ivf_overhead, estimate_vector_memory,
    BenchmarkResult, DistanceValidationReport, MemoryTracker,
};
use knowhere_rs::dataset::{load_gist1m_complete, GistDataset};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use std::env;
use std::time::Instant;

/// Get GIST1M dataset path from environment or use default
fn get_gist1m_path() -> String {
    env::var("GIST1M_PATH").unwrap_or_else(|_| "./data/gist".to_string())
}

/// Load or return None if dataset not found
fn try_load_gist1m() -> Option<GistDataset> {
    let path = get_gist1m_path();
    match load_gist1m_complete(&path) {
        Ok(dataset) => {
            println!("Loaded GIST1M dataset from {}", path);
            println!("  Base vectors: {}", dataset.num_base());
            println!("  Query vectors: {}", dataset.num_query());
            println!("  Dimension: {}", dataset.dim());
            Some(dataset)
        }
        Err(e) => {
            eprintln!("Failed to load GIST1M dataset from {}: {}", path, e);
            eprintln!("Set GIST1M_PATH environment variable or place dataset in ./data/gist/");
            None
        }
    }
}

/// Benchmark Flat index
fn benchmark_flat(dataset: &GistDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking Flat index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record index overhead (Flat index has minimal overhead)
    let overhead = (dataset.num_base() * 4) as u64; // Just ID storage
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 40,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
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

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
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
fn benchmark_hnsw(dataset: &GistDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking HNSW index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(32), // Higher M for high-dimensional data
            ef_construction: Some(200),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record HNSW overhead
    let overhead = estimate_hnsw_overhead(dataset.num_base(), dataset.dim(), 32);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 64,
            params: Some(r#"{"ef": 64}"#.to_string()),
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
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

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
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
fn benchmark_ivf_flat(dataset: &GistDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking IVF-Flat index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 20; // Higher nprobe for high-dimensional data

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record IVF overhead
    let overhead = estimate_ivf_overhead(dataset.num_base(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  nlist: {}, nprobe: {}", nlist, nprobe);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
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

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
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
fn test_gist1m_benchmark() {
    // Try to load dataset
    let dataset = match try_load_gist1m() {
        Some(ds) => ds,
        None => {
            println!("\nSkipping benchmark - GIST1M dataset not found");
            println!("To run benchmark:");
            println!("1. Download GIST1M from http://corpus-texmex.irisa.fr/");
            println!("2. Extract to ./data/gist/ or set GIST1M_PATH env var");
            println!("3. Run: cargo test --test bench_gist1m -- --nocapture");
            return;
        }
    };

    // Use subset of queries for faster testing (or all 1K for full benchmark)
    let num_queries = env::var("GIST_NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("\nRunning GIST1M benchmark with {} queries", num_queries);

    // Run benchmarks
    let results = vec![
        benchmark_flat(&dataset, num_queries),
        benchmark_hnsw(&dataset, num_queries),
        benchmark_ivf_flat(&dataset, num_queries),
    ];

    // Print summary table
    BenchmarkResult::print_table(&results);
    BenchmarkResult::print_markdown_table(&results, "GIST1M");

    // Save JSON if requested
    if let Ok(json_path) = env::var("JSON_OUTPUT") {
        match BenchmarkResult::save_json(&results, "GIST1M", &json_path) {
            Ok(_) => println!("✓ JSON results saved to: {}", json_path),
            Err(e) => eprintln!("Failed to save JSON: {}", e),
        }
    }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_gist1m_quick() {
    // Quick test with small dataset subset
    let dataset = match try_load_gist1m() {
        Some(ds) => ds,
        None => {
            println!("Skipping quick test - GIST1M dataset not found");
            return;
        }
    };

    let num_queries = 50;
    println!("\nRunning quick GIST1M test with {} queries", num_queries);

    let results = vec![
        benchmark_flat(&dataset, num_queries),
        benchmark_hnsw(&dataset, num_queries),
        benchmark_ivf_flat(&dataset, num_queries),
    ];

    BenchmarkResult::print_table(&results);

    // Save JSON if requested
    if let Ok(json_path) = env::var("JSON_OUTPUT") {
        match BenchmarkResult::save_json(&results, "GIST1M", &json_path) {
            Ok(_) => println!("✓ JSON results saved to: {}", json_path),
            Err(e) => eprintln!("Failed to save JSON: {}", e),
        }
    }
}

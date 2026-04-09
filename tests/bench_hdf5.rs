#![cfg(feature = "long-tests")]
//! HDF5 Dataset Benchmark Test
//!
//! Benchmark Flat, HNSW, and IVF indexes on HDF5 datasets from ann-benchmarks.
//!
//! # Supported Datasets
//! - GloVe-100, GloVe-200
//! - SIFT-128
//! - Deep-96
//! - GIST-960
//!
//! # Usage
//! ```bash
//! # Enable hdf5 feature and set path to HDF5 file
//! export HDF5_DATASET_PATH="./data/glove-100.hdf5"
//! cargo test --features hdf5 --test bench_hdf5 -- --nocapture
//! ```
//!
//! # Dataset Format
//! HDF5 files from ann-benchmarks contain:
//! - `train`: Training vectors (N x D)
//! - `test`: Query vectors (Q x D)
//! - `neighbors`: Ground truth neighbor IDs (Q x K)
//! - `distances`: Ground truth distances (Q x K)
//!
//! # Requirements
//! - HDF5 C library installed: `brew install hdf5` (macOS) or `apt-get install libhdf5-dev` (Ubuntu)

#![cfg(feature = "hdf5")]

use hanns::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use hanns::benchmark::{
    average_recall_at_k, estimate_hnsw_overhead, estimate_ivf_overhead, estimate_vector_memory,
    BenchmarkResult, MemoryTracker,
};
use hanns::dataset::{load_hdf5_dataset, Hdf5Dataset, Hdf5LoaderError};
use hanns::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use hanns::MetricType;
use std::env;
use std::time::Instant;

/// Get HDF5 dataset path from environment
fn get_hdf5_path() -> Option<String> {
    env::var("HDF5_DATASET_PATH").ok()
}

/// Try to load HDF5 dataset
fn try_load_hdf5_dataset() -> Option<Hdf5Dataset> {
    let path = get_hdf5_path()?;

    match load_hdf5_dataset(&path) {
        Ok(dataset) => {
            println!("Loaded HDF5 dataset from {}", path);
            println!("  Train vectors: {}", dataset.num_train());
            println!("  Test vectors: {}", dataset.num_test());
            println!("  Dimension: {}", dataset.dim());
            println!("  Ground truth neighbors: {}", dataset.num_neighbors());
            Some(dataset)
        }
        Err(e) => {
            eprintln!("Failed to load HDF5 dataset from {}: {}", path, e);
            eprintln!("Set HDF5_DATASET_PATH environment variable to HDF5 file path");
            eprintln!("Supported formats: GloVe, SIFT, Deep, GIST in HDF5");
            None
        }
    }
}

/// Benchmark Flat index
fn benchmark_flat(dataset: &Hdf5Dataset, num_queries: usize) -> BenchmarkResult {
    println!("\n=== Benchmarking Flat Index ===");

    let base_vectors = dataset.train.vectors();
    let test_vectors = dataset.test.vectors();
    let ground_truth: &[Vec<i32>] = &dataset.neighbors;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_train(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record index overhead
    let overhead = (dataset.num_train() * 4) as u64; // ID storage
    tracker.record_index_overhead(overhead);

    println!("Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &test_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 40,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("QPS: {:.0}", qps);

    // Calculate recall

    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("Recall@1: {:.3}", recall_at_1);
    println!("Recall@10: {:.3}", recall_at_10);
    println!("Recall@100: {:.3}", recall_at_100);

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
fn benchmark_hnsw(dataset: &Hdf5Dataset, num_queries: usize) -> BenchmarkResult {
    println!("\n=== Benchmarking HNSW Index ===");

    let base_vectors = dataset.train.vectors();
    let test_vectors = dataset.test.vectors();
    let ground_truth: &[Vec<i32>] = &dataset.neighbors;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_train(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(16),
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

    // Estimate HNSW overhead
    let overhead = estimate_hnsw_overhead(dataset.num_train(), dataset.dim(), 16);
    tracker.record_index_overhead(overhead);

    println!("Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &test_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 64,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("QPS: {:.0}", qps);

    // Calculate recall

    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("Recall@1: {:.3}", recall_at_1);
    println!("Recall@10: {:.3}", recall_at_10);
    println!("Recall@100: {:.3}", recall_at_100);

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
fn benchmark_ivf_flat(dataset: &Hdf5Dataset, num_queries: usize) -> BenchmarkResult {
    println!("\n=== Benchmarking IVF-Flat Index ===");

    let base_vectors = dataset.train.vectors();
    let test_vectors = dataset.test.vectors();
    let ground_truth: &[Vec<i32>] = &dataset.neighbors;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_train(), dataset.dim());
    tracker.record_base_memory(base_mem);

    // Calculate number of clusters (sqrt of base vectors)
    let nlist = (dataset.num_train() as f64).sqrt() as usize;
    let nlist = nlist.clamp(16, 4096); // Clamp to reasonable range

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some((nlist as f64 * 0.05) as usize), // 5% of clusters
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Estimate IVF overhead
    let overhead = estimate_ivf_overhead(dataset.num_train(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("Build time: {:.2} ms (nlist={})", build_time, nlist);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &test_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: (nlist as f64 * 0.05) as usize,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("QPS: {:.0}", qps);

    // Calculate recall

    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!("Recall@1: {:.3}", recall_at_1);
    println!("Recall@10: {:.3}", recall_at_10);
    println!("Recall@100: {:.3}", recall_at_100);

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

/// Print benchmark summary
fn print_summary(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Index", "Build(ms)", "Search(ms)", "QPS", "R@1", "R@10", "R@100"
    );
    println!("{}", "-".repeat(80));

    for result in results {
        println!(
            "{:<12} {:>10.1} {:>10.1} {:>10.0} {:>10.3} {:>10.3} {:>10.3}",
            result.index_name,
            result.build_time_ms,
            result.search_time_ms,
            result.qps,
            result.recall_at_1,
            result.recall_at_10,
            result.recall_at_100
        );
    }

    println!("{}", "=".repeat(80));
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hdf5_loader() {
    println!("\n{}", "=".repeat(80));
    println!("HDF5 Dataset Loader Test");
    println!("{}", "=".repeat(80));

    // Test loading dataset
    match try_load_hdf5_dataset() {
        Some(dataset) => {
            println!("\nDataset loaded successfully!");
            println!(
                "  Train: {} vectors x {} dims",
                dataset.num_train(),
                dataset.dim()
            );
            println!("  Test: {} vectors", dataset.num_test());
            println!(
                "  Ground truth: {} queries x {} neighbors",
                dataset.neighbors.len(),
                dataset.num_neighbors()
            );

            // Run benchmarks with 100 queries for quick test
            let num_queries = (100).min(dataset.num_test());
            println!("\nRunning benchmarks with {} queries...", num_queries);

            let results = vec![
                benchmark_flat(&dataset, num_queries),
                benchmark_hnsw(&dataset, num_queries),
                benchmark_ivf_flat(&dataset, num_queries),
            ];

            // Print summary
            print_summary(&results);
        }
        None => {
            println!("\nDataset not available. Set HDF5_DATASET_PATH to run full benchmark.");
            println!("Example: export HDF5_DATASET_PATH=\"./data/glove-100.hdf5\"");

            // Test error handling
            println!("\nTesting error handling...");
            let result = load_hdf5_dataset("/nonexistent/file.hdf5");
            assert!(matches!(result, Err(Hdf5LoaderError::FileNotFound(_))));
            println!("  ✓ FileNotFound error handled correctly");
        }
    }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hdf5_dataset_structure() {
    // Test that Hdf5Dataset struct works correctly
    use hanns::dataset::Dataset;

    let train = Dataset::from_vectors(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
    let test = Dataset::from_vectors(vec![7.0, 8.0, 9.0, 10.0], 2);
    let neighbors = vec![vec![0i32, 1], vec![1, 0]];
    let distances = vec![vec![0.0, 1.0], vec![0.5, 1.5]];

    let dataset = Hdf5Dataset::new(train, test, neighbors.clone(), distances.clone());

    assert_eq!(dataset.num_train(), 3);
    assert_eq!(dataset.num_test(), 2);
    assert_eq!(dataset.dim(), 2);
    assert_eq!(dataset.num_neighbors(), 2);
    assert_eq!(dataset.neighbors, neighbors);
    assert_eq!(dataset.distances, distances);

    println!("✓ Hdf5Dataset structure test passed");
}

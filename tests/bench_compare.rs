//! Unified Benchmark Comparison Tool
//!
//! Supports multiple datasets (SIFT1M, Deep1M, GIST1M) with consistent output format.
//! Outputs both human-readable tables and JSON for analysis.
//!
//! # Usage
//! ```bash
//! # Run with JSON output
//! DATASET=sift1m JSON_OUTPUT=output.json cargo test --test bench_compare test_bench_compare_all -- --nocapture
//!
//! # Run specific dataset
//! DATASET=deep1m cargo test --test bench_compare test_bench_compare_deep1m -- --nocapture
//!
//! # Run with custom query count
//! DATASET=gist1m NUM_QUERIES=100 cargo test --test bench_compare test_bench_compare_gist1m -- --nocapture
//! ```

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{average_recall_at_k, BenchmarkResult};
use knowhere_rs::dataset::{
    load_deep1m_complete, load_gist1m_complete, load_sift1m_complete, DeepDataset, GistDataset,
    SiftDataset,
};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use std::env;
use std::time::Instant;

#[test]
fn compare_lane_excludes_diskann_until_it_is_native_comparable() {
    let indexes = compare_lane_index_names();

    assert!(indexes.contains(&"Flat"));
    assert!(indexes.contains(&"HNSW"));
    assert!(indexes.contains(&"IVF-Flat"));
    assert!(
        !indexes.contains(&"DiskANN"),
        "DiskANN must stay out of the compare lane until the implementation is native-comparable"
    );
}

/// Dataset type enum
#[derive(Debug, Clone, Copy)]
enum DatasetType {
    Sift1m,
    Deep1m,
    Gist1m,
}

fn compare_lane_index_names() -> Vec<&'static str> {
    vec!["Flat", "HNSW", "IVF-Flat"]
}

impl DatasetType {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "deep1m" => DatasetType::Deep1m,
            "gist1m" => DatasetType::Gist1m,
            _ => DatasetType::Sift1m,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            DatasetType::Sift1m => "SIFT1M",
            DatasetType::Deep1m => "DEEP1M",
            DatasetType::Gist1m => "GIST1M",
        }
    }
}

/// Unified dataset wrapper
enum UnifiedDataset {
    Sift(SiftDataset),
    Deep(DeepDataset),
    Gist(GistDataset),
}

impl UnifiedDataset {
    fn dim(&self) -> usize {
        match self {
            UnifiedDataset::Sift(ds) => ds.dim(),
            UnifiedDataset::Deep(ds) => ds.dim(),
            UnifiedDataset::Gist(ds) => ds.dim(),
        }
    }

    fn num_base(&self) -> usize {
        match self {
            UnifiedDataset::Sift(ds) => ds.num_base(),
            UnifiedDataset::Deep(ds) => ds.num_base(),
            UnifiedDataset::Gist(ds) => ds.num_base(),
        }
    }

    fn num_query(&self) -> usize {
        match self {
            UnifiedDataset::Sift(ds) => ds.num_query(),
            UnifiedDataset::Deep(ds) => ds.num_query(),
            UnifiedDataset::Gist(ds) => ds.num_query(),
        }
    }

    fn base_vectors(&self) -> &[f32] {
        match self {
            UnifiedDataset::Sift(ds) => ds.base.vectors(),
            UnifiedDataset::Deep(ds) => ds.base.vectors(),
            UnifiedDataset::Gist(ds) => ds.base.vectors(),
        }
    }

    fn query_vectors(&self) -> &[f32] {
        match self {
            UnifiedDataset::Sift(ds) => ds.query.vectors(),
            UnifiedDataset::Deep(ds) => ds.query.vectors(),
            UnifiedDataset::Gist(ds) => ds.query.vectors(),
        }
    }

    fn ground_truth(&self) -> &[Vec<i32>] {
        match self {
            UnifiedDataset::Sift(ds) => &ds.ground_truth,
            UnifiedDataset::Deep(ds) => &ds.ground_truth,
            UnifiedDataset::Gist(ds) => &ds.ground_truth,
        }
    }
}

/// Load dataset based on type
fn load_dataset(dataset_type: DatasetType) -> Option<UnifiedDataset> {
    let base_path = env::var("DATASET_PATH").unwrap_or_else(|_| "./data".to_string());

    match dataset_type {
        DatasetType::Sift1m => {
            let path = format!("{}/sift", base_path);
            match load_sift1m_complete(&path) {
                Ok(ds) => {
                    println!("Loaded SIFT1M dataset from {}", path);
                    println!(
                        "  Base: {}, Query: {}, Dim: {}",
                        ds.num_base(),
                        ds.num_query(),
                        ds.dim()
                    );
                    Some(UnifiedDataset::Sift(ds))
                }
                Err(e) => {
                    eprintln!("Failed to load SIFT1M: {}", e);
                    None
                }
            }
        }
        DatasetType::Deep1m => {
            let path = format!("{}/deep", base_path);
            match load_deep1m_complete(&path) {
                Ok(ds) => {
                    println!("Loaded DEEP1M dataset from {}", path);
                    println!(
                        "  Base: {}, Query: {}, Dim: {}",
                        ds.num_base(),
                        ds.num_query(),
                        ds.dim()
                    );
                    Some(UnifiedDataset::Deep(ds))
                }
                Err(e) => {
                    eprintln!("Failed to load DEEP1M: {}", e);
                    None
                }
            }
        }
        DatasetType::Gist1m => {
            let path = format!("{}/gist", base_path);
            match load_gist1m_complete(&path) {
                Ok(ds) => {
                    println!("Loaded GIST1M dataset from {}", path);
                    println!(
                        "  Base: {}, Query: {}, Dim: {}",
                        ds.num_base(),
                        ds.num_query(),
                        ds.dim()
                    );
                    Some(UnifiedDataset::Gist(ds))
                }
                Err(e) => {
                    eprintln!("Failed to load GIST1M: {}", e);
                    None
                }
            }
        }
    }
}

/// Benchmark Flat index
fn benchmark_flat(dataset: &UnifiedDataset, num_queries: usize) -> BenchmarkResult {
    let index_name = "Flat";
    println!("\nBenchmarking {} index...", index_name);

    let base_vectors = dataset.base_vectors();
    let query_vectors = dataset.query_vectors();
    let ground_truth = dataset.ground_truth();
    let dim = dataset.dim();

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
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
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    BenchmarkResult {
        index_name: index_name.to_string(),
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
fn benchmark_hnsw(dataset: &UnifiedDataset, num_queries: usize) -> BenchmarkResult {
    let index_name = "HNSW";
    println!("\nBenchmarking {} index...", index_name);

    let base_vectors = dataset.base_vectors();
    let query_vectors = dataset.query_vectors();
    let ground_truth = dataset.ground_truth();
    let dim = dataset.dim();

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

    // Build index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Build time: {:.2} ms", build_time);

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 64,
            params: Some(r#"{"ef": 64}"#.to_string()),
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    BenchmarkResult {
        index_name: index_name.to_string(),
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
fn benchmark_ivf_flat(dataset: &UnifiedDataset, num_queries: usize) -> BenchmarkResult {
    let index_name = "IVF-Flat";
    println!("\nBenchmarking {} index...", index_name);

    let base_vectors = dataset.base_vectors();
    let query_vectors = dataset.query_vectors();
    let ground_truth = dataset.ground_truth();
    let dim = dataset.dim();
    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 10;

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
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
    println!(
        "  Build time: {:.2} ms (nlist={}, nprobe={})",
        build_time, nlist, nprobe
    );

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);
    println!(
        "  Recall@1: {:.3}, Recall@10: {:.3}, Recall@100: {:.3}",
        recall_at_1, recall_at_10, recall_at_100
    );

    BenchmarkResult {
        index_name: index_name.to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Run all benchmarks for a dataset
fn run_benchmarks(dataset_type: DatasetType, num_queries: usize, json_output: Option<&str>) {
    println!("\n{}", "=".repeat(70));
    println!("Knowhere-rs Benchmark: {}", dataset_type.name());
    println!("Queries: {}, Dataset: {}", num_queries, dataset_type.name());
    println!("{}\n", "=".repeat(70));

    // Load dataset
    let dataset = match load_dataset(dataset_type) {
        Some(ds) => ds,
        None => {
            eprintln!("\nDataset not found. Please download and place in ./data/");
            eprintln!("  SIFT1M: http://corpus-texmex.irisa.fr/");
            eprintln!("  DEEP1M: https://github.com/arbabenko/GNOIMI");
            eprintln!("  GIST1M: http://corpus-texmex.irisa.fr/");
            return;
        }
    };

    // Run benchmarks
    let results = vec![
        benchmark_flat(&dataset, num_queries),
        benchmark_hnsw(&dataset, num_queries),
        benchmark_ivf_flat(&dataset, num_queries),
    ];

    // Print results
    BenchmarkResult::print_table(&results);
    BenchmarkResult::print_markdown_table(&results, dataset_type.name());

    // Save JSON if requested
    if let Some(json_path) = json_output {
        match BenchmarkResult::save_json(&results, dataset_type.name(), json_path) {
            Ok(_) => println!("✓ JSON results saved to: {}", json_path),
            Err(e) => eprintln!("Failed to save JSON: {}", e),
        }
    }
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_bench_compare_all() {
    let dataset_type_str = env::var("DATASET").unwrap_or_else(|_| "sift1m".to_string());
    let dataset_type = DatasetType::from_str(&dataset_type_str);

    let num_queries = env::var("NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let json_output = env::var("JSON_OUTPUT").ok();

    run_benchmarks(dataset_type, num_queries, json_output.as_deref());
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_bench_compare_sift1m() {
    let num_queries = env::var("NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let json_output = env::var("JSON_OUTPUT").ok();

    run_benchmarks(DatasetType::Sift1m, num_queries, json_output.as_deref());
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_bench_compare_deep1m() {
    let num_queries = env::var("NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let json_output = env::var("JSON_OUTPUT").ok();

    run_benchmarks(DatasetType::Deep1m, num_queries, json_output.as_deref());
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_bench_compare_gist1m() {
    let num_queries = env::var("NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let json_output = env::var("JSON_OUTPUT").ok();

    run_benchmarks(DatasetType::Gist1m, num_queries, json_output.as_deref());
}

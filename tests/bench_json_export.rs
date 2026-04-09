//! Benchmark JSON Export Test
//!
//! Test and validate JSON export functionality for benchmark results.
//! Tests Flat, HNSW, and IVF-Flat indexes with Random10K dataset.
//!
//! # Usage
//! ```bash
//! # Standard mode (10K vectors, ~5 seconds)
//! cargo test --release --test bench_json_export -- --nocapture
//!
//! # Quick mode (1K vectors, <1 second)
//! QUICK=1 cargo test --release --test bench_json_export -- --nocapture
//!
//! # Detailed mode (50K vectors, ~30 seconds)
//! DETAILED=1 cargo test --release --test bench_json_export -- --nocapture
//! ```
//!
//! # Environment Variables
//! - `JSON_OUTPUT_DIR`: Custom output directory (default: /Users/ryan/.openclaw/workspace-builder/benchmark_results/)
//! - `QUICK`: Set to enable quick mode (1000 vectors, 10 queries)
//! - `DETAILED`: Set to enable detailed mode (50000 vectors, 500 queries)

mod common;
#[cfg(feature = "long-tests")]
use hanns::api::{IndexConfig, IndexParams, SearchRequest};
#[cfg(feature = "long-tests")]
use hanns::benchmark::average_recall_at_k;
use hanns::benchmark::BenchmarkResult;
#[cfg(feature = "long-tests")]
use hanns::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
#[cfg(feature = "long-tests")]
use hanns::IndexType;
#[cfg(feature = "long-tests")]
use hanns::MetricType;
#[cfg(feature = "long-tests")]
use rand::Rng;
#[cfg(feature = "long-tests")]
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "long-tests")]
use std::time::Instant;

/// Test mode configuration
#[derive(Debug, Clone, Copy)]
struct TestConfig {
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    #[cfg(feature = "long-tests")]
    k: usize,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10000,
            num_queries: 100,
            dim: 128,
            #[cfg(feature = "long-tests")]
            k: 100,
        }
    }
}

/// Generate random dataset
#[cfg(feature = "long-tests")]
fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-1.0..1.0));
    }

    data
}

/// Generate ground truth for random dataset (brute-force)
#[cfg(feature = "long-tests")]

#[cfg(feature = "long-tests")]

/// Benchmark Flat index
#[cfg(feature = "long-tests")]
fn benchmark_flat(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\n🔍 Benchmarking Flat index...");

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    println!("  Build time: {:.2} ms", build_time);

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

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
#[cfg(feature = "long-tests")]
fn benchmark_hnsw(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\n🔍 Benchmarking HNSW index...");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::hnsw(16, 64, 0.5),
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    println!("  Build time: {:.2} ms", build_time);

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

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
#[cfg(feature = "long-tests")]
fn benchmark_ivf_flat(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
) -> BenchmarkResult {
    println!("\n🔍 Benchmarking IVF-Flat index...");

    // Use reasonable nlist for dataset size (sqrt of num_vectors is a good rule of thumb)
    let num_vectors = base.len() / dim;
    let nlist = ((num_vectors as f32).sqrt() as u32).clamp(16, 256);
    let nprobe = (nlist as f32 * 0.1).max(1.0) as u32;

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::ivf(nlist as usize, nprobe as usize),
    };

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    println!("  Build time: {:.2} ms", build_time);

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

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

/// Get output directory from environment or use default
#[cfg(feature = "long-tests")]
fn get_output_dir() -> String {
    env::var("JSON_OUTPUT_DIR").unwrap_or_else(|_| {
        "/Users/ryan/.openclaw/workspace-builder/benchmark_results/".to_string()
    })
}

/// Generate JSON filename with timestamp
#[cfg(feature = "long-tests")]
fn generate_filename() -> String {
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    format!("benchmark_results_{}.json", timestamp)
}

fn dataset_name(config: TestConfig) -> String {
    format!("Random{}", config.num_vectors / 100)
}

/// Ensure output directory exists
#[cfg(feature = "long-tests")]
fn ensure_output_dir(dir: &str) -> std::io::Result<()> {
    if !Path::new(dir).exists() {
        fs::create_dir_all(dir)?;
        println!("📁 Created output directory: {}", dir);
    }
    Ok(())
}

/// Parse test mode from environment variables
#[cfg(feature = "long-tests")]
fn parse_test_mode() -> TestConfig {
    // Check environment variables for test mode
    if env::var("QUICK").is_ok() {
        println!("⚡ Quick mode: 1000 vectors, 10 queries");
        return TestConfig {
            num_vectors: 1000,
            num_queries: 10,
            dim: 128,
            #[cfg(feature = "long-tests")]
            k: 100,
        };
    }

    if env::var("DETAILED").is_ok() {
        println!("📊 Detailed mode: 50000 vectors, 500 queries");
        return TestConfig {
            num_vectors: 50000,
            num_queries: 500,
            dim: 128,
            #[cfg(feature = "long-tests")]
            k: 100,
        };
    }

    println!("📈 Standard mode: 10000 vectors, 100 queries");
    TestConfig::default()
}

fn sample_contract_results() -> Vec<BenchmarkResult> {
    vec![
        BenchmarkResult {
            index_name: "Flat".to_string(),
            build_time_ms: 1.2,
            search_time_ms: 2.5,
            num_queries: 10,
            qps: 4000.0,
            recall_at_1: 1.0,
            recall_at_10: 1.0,
            recall_at_100: 1.0,
        },
        BenchmarkResult {
            index_name: "HNSW".to_string(),
            build_time_ms: 3.4,
            search_time_ms: 1.1,
            num_queries: 10,
            qps: 9000.0,
            recall_at_1: 0.98,
            recall_at_10: 0.97,
            recall_at_100: 0.97,
        },
        BenchmarkResult {
            index_name: "IVF-Flat".to_string(),
            build_time_ms: 2.0,
            search_time_ms: 1.8,
            num_queries: 10,
            qps: 5500.0,
            recall_at_1: 0.96,
            recall_at_10: 0.95,
            recall_at_100: 0.95,
        },
    ]
}

fn build_json_report(config: &TestConfig, results: &[BenchmarkResult]) -> serde_json::Value {
    let fastest = results
        .iter()
        .min_by(|a, b| a.search_time_ms.partial_cmp(&b.search_time_ms).unwrap())
        .unwrap();
    let best_recall = results
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
        .unwrap();
    let total_time: f64 = results
        .iter()
        .map(|r| r.build_time_ms + r.search_time_ms)
        .sum();

    serde_json::json!({
        "dataset": dataset_name(*config),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "config": {
            "dim": config.dim,
            "num_vectors": config.num_vectors,
            "num_queries": config.num_queries,
            "metric_type": "L2"
        },
        "results": results,
        "summary": {
            "fastest_index": fastest.index_name,
            "best_recall_index": best_recall.index_name,
            "total_test_time_ms": total_time
        }
    })
}

fn write_json_report(
    output_dir: &Path,
    filename: &str,
    config: &TestConfig,
    results: &[BenchmarkResult],
) -> std::io::Result<PathBuf> {
    fs::create_dir_all(output_dir)?;
    let json_obj = build_json_report(config, results);
    let json_string =
        serde_json::to_string_pretty(&json_obj).expect("Failed to serialize benchmark JSON report");
    let output_path = output_dir.join(filename);
    fs::write(&output_path, json_string)?;
    Ok(output_path)
}

#[test]
fn test_json_export_contract_fast_lane() {
    let config = TestConfig {
        num_vectors: 1000,
        num_queries: 10,
        dim: 8,
        #[cfg(feature = "long-tests")]
        k: 10,
    };
    let output_dir = std::env::temp_dir().join(format!(
        "hanns_json_export_contract_{}",
        std::process::id()
    ));
    let output_path = write_json_report(
        &output_dir,
        "benchmark_results_contract.json",
        &config,
        &sample_contract_results(),
    )
    .expect("contract JSON report should be written");

    let content = fs::read_to_string(&output_path).expect("Failed to read JSON file");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse JSON");
    let results_array = parsed["results"].as_array().unwrap();

    assert_eq!(parsed["dataset"], "Random10");
    assert_eq!(parsed["config"]["dim"], 8);
    assert_eq!(parsed["summary"]["fastest_index"], "HNSW");
    assert_eq!(parsed["summary"]["best_recall_index"], "Flat");
    assert_eq!(results_array.len(), 3, "Should have results for 3 indexes");
    assert_eq!(results_array[0]["index_name"], "Flat");
    assert_eq!(results_array[1]["index_name"], "HNSW");
    assert_eq!(results_array[2]["index_name"], "IVF-Flat");

    let _ = fs::remove_dir_all(output_dir);
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_json_export() {
    println!("\n{}", "=".repeat(60));
    println!("🚀 Benchmark JSON Export Test");
    println!("{}", "=".repeat(60));

    // Parse test mode
    let config = parse_test_mode();

    // Get output directory
    let output_dir = get_output_dir();
    println!("📁 Output directory: {}", output_dir);

    // Ensure output directory exists
    ensure_output_dir(&output_dir).expect("Failed to create output directory");

    // Generate dataset
    println!(
        "\n📊 Generating Random{} dataset...",
        config.num_vectors / 100
    );
    let base = generate_random_dataset(config.num_vectors, config.dim);
    let query = generate_random_dataset(config.num_queries, config.dim);

    // Compute ground truth
    println!("🎯 Computing ground truth...");
    let gt_start = Instant::now();
    let ground_truth =
        common::compute_ground_truth(&base, &query, config.num_queries, config.dim, config.k);
    let gt_time = gt_start.elapsed().as_secs_f64();
    println!("  Ground truth computed in {:.2}s", gt_time);

    // Run benchmarks
    let mut results = Vec::new();

    let flat_result = benchmark_flat(&base, &query, &ground_truth, config.num_queries, config.dim);
    results.push(flat_result);

    let hnsw_result = benchmark_hnsw(&base, &query, &ground_truth, config.num_queries, config.dim);
    results.push(hnsw_result);

    let ivf_result =
        benchmark_ivf_flat(&base, &query, &ground_truth, config.num_queries, config.dim);
    results.push(ivf_result);

    // Print summary table
    BenchmarkResult::print_table(&results);

    // Write to file
    let filename = generate_filename();
    let output_path = write_json_report(Path::new(&output_dir), &filename, &config, &results)
        .expect("Failed to write JSON file");

    let total_time: f64 = results
        .iter()
        .map(|r| r.build_time_ms + r.search_time_ms)
        .sum();

    println!("\n{}", "=".repeat(60));
    println!("✅ JSON Export Complete!");
    println!("📄 Output file: {}", output_path.display());
    println!("📊 Total test time: {:.2} ms", total_time);
    println!("{}", "=".repeat(60));

    // Verify file was created
    assert!(output_path.exists(), "JSON file should exist");

    // Verify JSON is valid by reading it back
    let content = fs::read_to_string(&output_path).expect("Failed to read JSON file");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse JSON");

    // Verify structure
    assert!(parsed["dataset"].is_string(), "dataset should be a string");
    assert!(
        parsed["timestamp"].is_string(),
        "timestamp should be a string"
    );
    assert!(parsed["config"].is_object(), "config should be an object");
    assert!(parsed["results"].is_array(), "results should be an array");
    assert!(parsed["summary"].is_object(), "summary should be an object");

    let results_array = parsed["results"].as_array().unwrap();
    assert_eq!(results_array.len(), 3, "Should have results for 3 indexes");

    println!("\n✅ JSON validation passed!");
}

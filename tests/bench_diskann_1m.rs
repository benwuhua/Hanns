//! BENCH-051: DiskANN 1M benchmark
//!
//! 用法:
//! ```bash
//! cargo test --release --test bench_diskann_1m -- --nocapture
//! ```
//!
//! 环境变量:
//! - `SIFT1M_PATH`: 数据集路径（默认 `./data/sift1m`）
//! - `SIFT_NUM_QUERIES`: 查询数量（默认 `100`）
//! - `SIFT_BASE_SIZE`: base 向量数量（默认 `1000000`）

use knowhere_rs::api::{MetricType, SearchResult};
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use knowhere_rs::benchmark::average_recall_at_k;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

const TOP_K: usize = 10;
const DEFAULT_BASE_SIZE: usize = 1_000_000;
const DEFAULT_NUM_QUERIES: usize = 100;
const REPORT_PATH: &str = "BENCH-051_DISKANN_1M.md";

#[derive(Debug, Clone)]
struct BenchResult {
    config_name: String,
    config: AisaqConfig,
    build_time_s: f64,
    search_time_s: f64,
    qps: f64,
    recall_at_10: f64,
}

fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, top_k: usize) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for q_idx in 0..num_queries {
        let query = &queries[q_idx * dim..(q_idx + 1) * dim];
        let mut distances: Vec<(f32, i32)> = (0..num_base)
            .map(|i| {
                let v = &base[i * dim..(i + 1) * dim];
                let dist: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                (dist, i as i32)
            })
            .collect();
        distances.select_nth_unstable_by(top_k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(top_k);
        ground_truth.push(distances.iter().map(|(_, id)| *id).collect());
    }

    ground_truth
}

fn load_sift1m_subset(base_size: usize) -> Option<(Vec<f32>, Vec<f32>, usize)> {
    let path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift1m".to_string());
    let base_file = format!("{}/sift_base.fvecs", path);
    let query_file = format!("{}/sift_query.fvecs", path);

    if !std::path::Path::new(&base_file).exists() {
        eprintln!("SIFT1M dataset not found at {}", base_file);
        return None;
    }

    // Load base vectors (subset)
    let mut base_reader = BufReader::new(File::open(&base_file).ok()?);
    let base_full = knowhere_rs::dataset::read_fvecs(&mut base_reader).ok()?;
    
    // Determine dimension by reading first 4 bytes
    let mut dim_reader = BufReader::new(File::open(&base_file).ok()?);
    let mut dim_buf = [0u8; 4];
    dim_reader.read_exact(&mut dim_buf).ok()?;
    let dim = u32::from_le_bytes(dim_buf) as usize;
    
    let base_count = base_full.len() / dim;
    let actual_size = base_count.min(base_size);
    let base = base_full[..actual_size * dim].to_vec();

    // Load query vectors
    let mut query_reader = BufReader::new(File::open(&query_file).ok()?);
    let queries = knowhere_rs::dataset::read_fvecs(&mut query_reader).ok()?;

    Some((base, queries, dim))
}

fn run_benchmark(
    base: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    config: AisaqConfig,
    config_name: &str,
) -> BenchResult {
    let num_queries = queries.len() / dim;

    // Build index
    let build_start = Instant::now();
    let mut index = PQFlashIndex::new(config.clone(), MetricType::L2, dim)
        .expect("Failed to create DiskANN index");
    index.add(base).expect("Failed to add vectors");
    let build_time_s = build_start.elapsed().as_secs_f64();

    // Search
    let search_start = Instant::now();
    let mut results = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result: SearchResult = index.search(query, TOP_K).expect("Search failed");
        results.push(result.ids);
    }
    let search_time_s = search_start.elapsed().as_secs_f64();

    // Compute metrics
    let qps = num_queries as f64 / search_time_s;
    let recall = average_recall_at_k(&results, ground_truth, TOP_K);

    BenchResult {
        config_name: config_name.to_string(),
        config,
        build_time_s,
        search_time_s,
        qps,
        recall_at_10: recall,
    }
}

fn generate_report(results: &[BenchResult], base_size: usize, num_queries: usize) -> String {
    let mut report = String::new();
    report.push_str("# BENCH-051: DiskANN 1M Benchmark\n\n");
    report.push_str(&format!("**Date**: {}\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    report.push_str(&format!("**Dataset**: SIFT1M ({} base, {} queries)\n\n", base_size, num_queries));
    
    report.push_str("## Results\n\n");
    report.push_str("| Config | MaxDegree | BeamWidth | Build (s) | Search (s) | QPS | R@10 |\n");
    report.push_str("|--------|-----------|-----------|-----------|------------|-----|------|\n");
    
    for r in results {
        report.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.3} | {:.0} | {:.3} |\n",
            r.config_name,
            r.config.max_degree,
            r.config.beamwidth,
            r.build_time_s,
            r.search_time_s,
            r.qps,
            r.recall_at_10
        ));
    }

    report.push_str("\n## Configuration Details\n\n");
    for r in results {
        report.push_str(&format!(
            "### {}\n- max_degree: {}\n- search_list_size: {}\n- beamwidth: {}\n- num_entry_points: {}\n\n",
            r.config_name,
            r.config.max_degree,
            r.config.search_list_size,
            r.config.beamwidth,
            r.config.num_entry_points
        ));
    }

    report.push_str("## Notes\n\n");
    report.push_str("- Ground truth computed via brute-force L2 distance\n");
    report.push_str("- DiskANN uses PQ compression + beam search\n");
    report.push_str("- Benchmark uses in-memory mode (Phase 1-2 implementation)\n");
    report.push_str("- This harness exercises a constrained Rust AISAQ skeleton, not a native-comparable SSD DiskANN pipeline\n");

    report
}

#[test]
fn diskann_benchmark_report_states_constrained_scope() {
    let report = generate_report(&[], 1024, 16);

    assert!(
        report.contains("Benchmark uses in-memory mode"),
        "report must disclose that this benchmark is not exercising a native SSD path"
    );
    assert!(
        report.contains("not a native-comparable SSD DiskANN pipeline"),
        "report must explicitly block parity/leadership interpretation"
    );
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_diskann_1m_benchmark() {
    let base_size = env::var("SIFT_BASE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BASE_SIZE);
    
    let num_queries = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);

    println!("BENCH-051: DiskANN {}M Benchmark", base_size / 1_000_000);
    println!("Loading dataset...");

    let (base, queries_full, dim) = match load_sift1m_subset(base_size) {
        Some(data) => data,
        None => {
            println!("Skipping: SIFT1M dataset not available");
            println!("To run this benchmark, download SIFT1M and set SIFT1M_PATH");
            return;
        }
    };

    let queries = if queries_full.len() / dim > num_queries {
        queries_full[..num_queries * dim].to_vec()
    } else {
        queries_full
    };

    let actual_queries = queries.len() / dim;
    println!("Dataset loaded: {} base, {} queries, dim={}", 
             base.len() / dim, actual_queries, dim);

    // Compute ground truth
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&base, &queries, dim, TOP_K);

    // Test configurations
    let configs = vec![
        ("R=32-B=4", AisaqConfig {
            max_degree: 32,
            search_list_size: 64,
            beamwidth: 4,
            num_entry_points: 1,
            ..AisaqConfig::default()
        }),
        ("R=48-B=8", AisaqConfig {
            max_degree: 48,
            search_list_size: 128,
            beamwidth: 8,
            num_entry_points: 1,
            ..AisaqConfig::default()
        }),
        ("R=64-B=16", AisaqConfig {
            max_degree: 64,
            search_list_size: 256,
            beamwidth: 16,
            num_entry_points: 2,
            ..AisaqConfig::default()
        }),
    ];

    let mut results = Vec::new();
    for (name, config) in configs {
        println!("Testing {} (R={}, B={})...", name, config.max_degree, config.beamwidth);
        let result = run_benchmark(&base, &queries, &ground_truth, dim, config, name);
        println!("  QPS: {:.0}, R@10: {:.3}", result.qps, result.recall_at_10);
        results.push(result);
    }

    // Generate report
    let report = generate_report(&results, base_size, actual_queries);
    std::fs::write(REPORT_PATH, &report).expect("Failed to write report");
    println!("\nReport saved to {}", REPORT_PATH);
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore]
fn test_diskann_1m_full() {
    // Full 1M benchmark (long running)
    std::env::set_var("SIFT_BASE_SIZE", "1000000");
    std::env::set_var("SIFT_NUM_QUERIES", "100");
    test_diskann_1m_benchmark();
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_diskann_100k_quick() {
    // Quick 100K benchmark for validation
    std::env::set_var("SIFT_BASE_SIZE", "100000");
    std::env::set_var("SIFT_NUM_QUERIES", "100");
    test_diskann_1m_benchmark();
}

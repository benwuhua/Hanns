#![cfg(feature = "long-tests")]
//! BENCH-024: HNSW 参数优化 Benchmark
//!
//! 测试不同 M / ef_construction / ef_search 组合对构建时间、搜索 QPS、
//! 召回率 (R@1 / R@10 / R@100) 以及内存占用的影响，并识别最佳配置。
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_hnsw_param_optimization -- --nocapture
//! ```
//!
//! # Environment Variables
//! - `QUICK`: 启用快速模式 (10K 向量，缩小参数空间)
//! - `JSON_OUTPUT_DIR`: 自定义输出目录 (默认: benchmark_results/)

mod common;
use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::benchmark::{average_recall_at_k, estimate_vector_memory, MemoryTracker};
use hanns::faiss::HnswIndex;
use hanns::IndexType;
use hanns::MetricType;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParamResult {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    build_time_s: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    memory_mb: f64,
}

#[derive(Debug, Clone)]
struct BenchConfig {
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    k: usize,
    m_values: Vec<usize>,
    ef_construction_values: Vec<usize>,
    ef_search_values: Vec<usize>,
}

impl BenchConfig {
    fn standard() -> Self {
        Self {
            num_vectors: 100_000,
            num_queries: 100,
            dim: 128,
            k: 100,
            m_values: vec![8, 16, 32, 48, 64],
            ef_construction_values: vec![100, 200, 400, 600, 800],
            ef_search_values: vec![100, 200, 400],
        }
    }

    fn quick() -> Self {
        Self {
            num_vectors: 10_000,
            num_queries: 20,
            dim: 128,
            k: 100,
            m_values: vec![8, 16, 32],
            ef_construction_values: vec![100, 200, 400],
            ef_search_values: vec![100, 200],
        }
    }

    fn from_env() -> Self {
        if env::var("QUICK").is_ok() {
            println!("⚡ Quick mode: 10K vectors, reduced parameter space");
            Self::quick()
        } else {
            println!("📊 Standard mode: 100K vectors, full parameter sweep");
            Self::standard()
        }
    }

    fn total_configs(&self) -> usize {
        self.m_values.len() * self.ef_construction_values.len() * self.ef_search_values.len()
    }
}

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

fn generate_gaussian_dataset(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim)
        .map(|_| {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark core
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn benchmark_single(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> ParamResult {
    let num_vectors = base.len() / dim;

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()),
            ..Default::default()
        },
    };

    // Memory tracking
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(num_vectors, dim);
    tracker.record_base_memory(base_mem);

    // Build
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    // Estimate HNSW graph overhead: each vector has ~2*M neighbors stored as u32/u64
    let graph_overhead = (num_vectors * 2 * m * 8) as u64;
    tracker.record_index_overhead(graph_overhead);
    let memory_mb = (base_mem + graph_overhead) as f64 / (1024.0 * 1024.0);

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: ef_search,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time_s = search_start.elapsed().as_secs_f64();
    let qps = num_queries as f64 / search_time_s;

    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);

    ParamResult {
        m,
        ef_construction,
        ef_search,
        build_time_s,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        memory_mb,
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_results_table(results: &[ParamResult]) {
    println!("\n{}", "=".repeat(130));
    println!("BENCH-024: HNSW 参数优化结果");
    println!("{}", "=".repeat(130));
    println!(
        "| {:>4} | {:>6} | {:>6} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} |",
        "M", "ef_C", "ef_S", "Build(s)", "QPS", "R@1", "R@10", "R@100", "Mem(MB)"
    );
    println!("{}", "-".repeat(130));

    for r in results {
        println!(
            "| {:>4} | {:>6} | {:>6} | {:>10.3} | {:>10.0} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.1} |",
            r.m, r.ef_construction, r.ef_search, r.build_time_s, r.qps,
            r.recall_at_1, r.recall_at_10, r.recall_at_100, r.memory_mb
        );
    }
    println!("{}", "=".repeat(130));
}

fn print_sensitivity_analysis(results: &[ParamResult], cfg: &BenchConfig) {
    println!("\n{}", "=".repeat(90));
    println!("参数敏感性分析");
    println!("{}", "=".repeat(90));

    // M impact
    println!("\n--- M 参数影响 (平均值) ---");
    println!(
        "| {:>4} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} |",
        "M", "Build(s)", "QPS", "R@1", "R@10", "R@100"
    );
    for &m in &cfg.m_values {
        let filtered: Vec<_> = results.iter().filter(|r| r.m == m).collect();
        if filtered.is_empty() {
            continue;
        }
        let n = filtered.len() as f64;
        println!(
            "| {:>4} | {:>10.3} | {:>10.0} | {:>8.4} | {:>8.4} | {:>8.4} |",
            m,
            filtered.iter().map(|r| r.build_time_s).sum::<f64>() / n,
            filtered.iter().map(|r| r.qps).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_1).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_100).sum::<f64>() / n,
        );
    }

    // ef_construction impact
    println!("\n--- ef_construction 参数影响 (平均值) ---");
    println!(
        "| {:>6} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} |",
        "ef_C", "Build(s)", "QPS", "R@1", "R@10", "R@100"
    );
    for &ef_c in &cfg.ef_construction_values {
        let filtered: Vec<_> = results
            .iter()
            .filter(|r| r.ef_construction == ef_c)
            .collect();
        if filtered.is_empty() {
            continue;
        }
        let n = filtered.len() as f64;
        println!(
            "| {:>6} | {:>10.3} | {:>10.0} | {:>8.4} | {:>8.4} | {:>8.4} |",
            ef_c,
            filtered.iter().map(|r| r.build_time_s).sum::<f64>() / n,
            filtered.iter().map(|r| r.qps).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_1).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_100).sum::<f64>() / n,
        );
    }

    // ef_search impact
    println!("\n--- ef_search 参数影响 (平均值) ---");
    println!(
        "| {:>6} | {:>10} | {:>8} | {:>8} | {:>8} |",
        "ef_S", "QPS", "R@1", "R@10", "R@100"
    );
    for &ef_s in &cfg.ef_search_values {
        let filtered: Vec<_> = results.iter().filter(|r| r.ef_search == ef_s).collect();
        if filtered.is_empty() {
            continue;
        }
        let n = filtered.len() as f64;
        println!(
            "| {:>6} | {:>10.0} | {:>8.4} | {:>8.4} | {:>8.4} |",
            ef_s,
            filtered.iter().map(|r| r.qps).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_1).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / n,
            filtered.iter().map(|r| r.recall_at_100).sum::<f64>() / n,
        );
    }
}

fn find_best_configs(results: &[ParamResult]) {
    println!("\n{}", "=".repeat(90));
    println!("最佳配置推荐");
    println!("{}", "=".repeat(90));

    // 1. Highest R@10 with fastest QPS
    println!("\n[1] R@10 > 0.95 的最高 QPS 配置:");
    let mut high_recall: Vec<_> = results.iter().filter(|r| r.recall_at_10 > 0.95).collect();
    high_recall.sort_by(|a, b| b.qps.partial_cmp(&a.qps).unwrap());
    if let Some(best) = high_recall.first() {
        println!(
            "    M={}, ef_C={}, ef_S={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "    QPS={:.0}, R@1={:.4}, R@10={:.4}, R@100={:.4}, Build={:.3}s, Mem={:.1}MB",
            best.qps,
            best.recall_at_1,
            best.recall_at_10,
            best.recall_at_100,
            best.build_time_s,
            best.memory_mb
        );
    } else {
        println!("    (无配置满足 R@10 > 0.95)");
    }

    // 2. Best balance: R@10 × QPS
    println!("\n[2] 最佳平衡配置 (R@10 x QPS):");
    let mut balanced: Vec<_> = results.iter().collect();
    balanced.sort_by(|a, b| {
        let sa = a.recall_at_10 * a.qps;
        let sb = b.recall_at_10 * b.qps;
        sb.partial_cmp(&sa).unwrap()
    });
    if let Some(best) = balanced.first() {
        let score = best.recall_at_10 * best.qps;
        println!(
            "    M={}, ef_C={}, ef_S={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "    Score={:.0}, QPS={:.0}, R@10={:.4}, Build={:.3}s, Mem={:.1}MB",
            score, best.qps, best.recall_at_10, best.build_time_s, best.memory_mb
        );
    }

    // 3. Lowest memory with R@10 > 0.90
    println!("\n[3] R@10 > 0.90 的最低内存配置:");
    let mut low_mem: Vec<_> = results.iter().filter(|r| r.recall_at_10 > 0.90).collect();
    low_mem.sort_by(|a, b| a.memory_mb.partial_cmp(&b.memory_mb).unwrap());
    if let Some(best) = low_mem.first() {
        println!(
            "    M={}, ef_C={}, ef_S={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "    Mem={:.1}MB, QPS={:.0}, R@10={:.4}, Build={:.3}s",
            best.memory_mb, best.qps, best.recall_at_10, best.build_time_s
        );
    } else {
        println!("    (无配置满足 R@10 > 0.90)");
    }

    // 4. Fastest build with R@10 > 0.90
    println!("\n[4] R@10 > 0.90 的最快构建配置:");
    let mut fast_build: Vec<_> = results.iter().filter(|r| r.recall_at_10 > 0.90).collect();
    fast_build.sort_by(|a, b| a.build_time_s.partial_cmp(&b.build_time_s).unwrap());
    if let Some(best) = fast_build.first() {
        println!(
            "    M={}, ef_C={}, ef_S={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "    Build={:.3}s, QPS={:.0}, R@10={:.4}, Mem={:.1}MB",
            best.build_time_s, best.qps, best.recall_at_10, best.memory_mb
        );
    } else {
        println!("    (无配置满足 R@10 > 0.90)");
    }
}

fn export_json(results: &[ParamResult], cfg: &BenchConfig) {
    let output_dir = env::var("JSON_OUTPUT_DIR").unwrap_or_else(|_| {
        "/Users/ryan/.openclaw/workspace-builder/benchmark_results/".to_string()
    });

    if !Path::new(&output_dir).exists() {
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    }

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let filename = format!("bench024_hnsw_param_optimization_{}.json", timestamp);
    let output_path = Path::new(&output_dir).join(&filename);

    let json_obj = serde_json::json!({
        "benchmark": "BENCH-024: HNSW Parameter Optimization",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "dataset": {
            "num_vectors": cfg.num_vectors,
            "num_queries": cfg.num_queries,
            "dimension": cfg.dim,
            "metric_type": "L2",
        },
        "parameter_ranges": {
            "M": cfg.m_values,
            "ef_construction": cfg.ef_construction_values,
            "ef_search": cfg.ef_search_values,
        },
        "total_configurations": results.len(),
        "results": results,
        "summary": {
            "best_recall_at_10": results.iter().map(|r| r.recall_at_10).fold(0.0f64, f64::max),
            "best_qps": results.iter().map(|r| r.qps).fold(0.0f64, f64::max),
            "min_build_time_s": results.iter().map(|r| r.build_time_s).fold(f64::INFINITY, f64::min),
            "min_memory_mb": results.iter().map(|r| r.memory_mb).fold(f64::INFINITY, f64::min),
        }
    });

    let json_str = serde_json::to_string_pretty(&json_obj).unwrap();
    fs::write(&output_path, &json_str).expect("Failed to write JSON");
    println!("\n📄 JSON exported: {}", output_path.display());
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_param_optimization() {
    println!("\n{}", "=".repeat(90));
    println!("BENCH-024: HNSW 参数优化 Benchmark");
    println!("{}", "=".repeat(90));

    let cfg = BenchConfig::from_env();

    println!("\n配置:");
    println!("  向量数: {}", cfg.num_vectors);
    println!("  查询数: {}", cfg.num_queries);
    println!("  维度: {}", cfg.dim);
    println!("  M: {:?}", cfg.m_values);
    println!("  ef_construction: {:?}", cfg.ef_construction_values);
    println!("  ef_search: {:?}", cfg.ef_search_values);
    println!("  总配置数: {}", cfg.total_configs());

    // Generate dataset
    println!("\n生成高斯分布数据集...");
    let base = generate_gaussian_dataset(cfg.num_vectors, cfg.dim);
    let query = generate_gaussian_dataset(cfg.num_queries, cfg.dim);

    // Ground truth
    println!("计算 ground truth (brute-force)...");
    let gt_start = Instant::now();
    let ground_truth = common::compute_ground_truth(&base, &query, cfg.num_queries, cfg.dim, cfg.k);
    println!("  完成: {:.2}s", gt_start.elapsed().as_secs_f64());

    // Parameter sweep
    let total = cfg.total_configs();
    let mut results = Vec::with_capacity(total);
    let mut idx = 0;

    for &m in &cfg.m_values {
        for &ef_c in &cfg.ef_construction_values {
            for &ef_s in &cfg.ef_search_values {
                idx += 1;
                println!(
                    "\n[{}/{}] M={}, ef_C={}, ef_S={}",
                    idx, total, m, ef_c, ef_s
                );

                let r = benchmark_single(
                    &base,
                    &query,
                    &ground_truth,
                    cfg.num_queries,
                    cfg.dim,
                    m,
                    ef_c,
                    ef_s,
                );

                println!(
                    "  Build={:.3}s  QPS={:.0}  R@1={:.4}  R@10={:.4}  R@100={:.4}  Mem={:.1}MB",
                    r.build_time_s,
                    r.qps,
                    r.recall_at_1,
                    r.recall_at_10,
                    r.recall_at_100,
                    r.memory_mb
                );

                results.push(r);
            }
        }
    }

    // Output
    print_results_table(&results);
    print_sensitivity_analysis(&results, &cfg);
    find_best_configs(&results);
    export_json(&results, &cfg);

    println!("\nBENCH-024 完成! 共测试 {} 种配置。", results.len());
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_param_optimization_quick() {
    env::set_var("QUICK", "1");
    test_hnsw_param_optimization();
}

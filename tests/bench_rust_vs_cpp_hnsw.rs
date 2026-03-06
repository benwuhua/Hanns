//! Rust vs C++ knowhere HNSW 性能对比
//!
//! 公平对比：使用相同的数据集、相同的 HNSW 参数 (M, ef_construction, ef_search)
//! 对比指标：QPS、召回率、内存占用、构建时间

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::dataset::{load_deep1m_complete, load_sift1m_complete};
use knowhere_rs::faiss::HnswIndex;
use std::path::Path;
use std::time::Instant;

/// 测试结果结构
#[derive(Debug, Clone)]
struct CompareResult {
    dataset: String,
    rust_build_time_ms: f64,
    rust_search_time_ms: f64,
    rust_qps: f64,
    rust_recall_at_1: f64,
    rust_recall_at_10: f64,
    rust_recall_at_100: f64,
    rust_memory_mb: f64,
    cpp_build_time_ms: f64,
    cpp_search_time_ms: f64,
    cpp_qps: f64,
    cpp_recall_at_1: f64,
    cpp_recall_at_10: f64,
    cpp_recall_at_100: f64,
    cpp_memory_mb: f64,
}

/// 加载 SIFT1M 数据集
fn load_sift1m() -> Option<(Vec<f32>, Vec<f32>, Vec<Vec<i32>>)> {
    let base_path = "/Users/ryan/.openclaw/workspace-builder/datasets/sift1m";

    match load_sift1m_complete(base_path) {
        Ok(dataset) => {
            let base = dataset.base.vectors().to_vec();
            let query = dataset.query.vectors().to_vec();
            let ground_truth = dataset.ground_truth.iter().map(|v| v.to_vec()).collect();
            Some((base, query, ground_truth))
        }
        Err(e) => {
            eprintln!("Failed to load SIFT1M: {}", e);
            None
        }
    }
}

/// 加载 Deep1M 数据集
fn load_deep1m() -> Option<(Vec<f32>, Vec<f32>, Vec<Vec<i32>>)> {
    let base_path = "/Users/ryan/.openclaw/workspace-builder/datasets/deep1m";

    match load_deep1m_complete(base_path) {
        Ok(dataset) => {
            let base = dataset.base.vectors().to_vec();
            let query = dataset.query.vectors().to_vec();
            let ground_truth = dataset.ground_truth.iter().map(|v| v.to_vec()).collect();
            Some((base, query, ground_truth))
        }
        Err(e) => {
            eprintln!("Failed to load Deep1M: {}", e);
            None
        }
    }
}

/// 计算召回率
fn compute_recall_at_k(results: &[Vec<i64>], ground_truth: &[Vec<i32>], k: usize) -> f64 {
    let mut total_recall = 0.0;
    let num_queries = results.len();

    for (i, result) in results.iter().enumerate() {
        if i >= ground_truth.len() {
            break;
        }

        let gt = &ground_truth[i];
        let gt_k = std::cmp::min(k, gt.len());
        let mut matches = 0;

        for &id in result.iter().take(k) {
            if gt[..gt_k].contains(&(id as i32)) {
                matches += 1;
            }
        }

        total_recall += matches as f64 / gt_k as f64;
    }

    total_recall / num_queries as f64
}

/// 运行 Rust HNSW benchmark
fn run_rust_benchmark(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    num_queries: usize,
    k: usize,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let num_base = base.len() / dim;

    // HNSW 参数 (与 C++ knowhere 保持一致)
    let m = 16;
    let ef_construction = 200;
    let ef_search = 128;

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: knowhere_rs::MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::hnsw(ef_construction, ef_search, m as f32),
    };

    // 构建索引
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).expect("Failed to create HNSW index");
    index.add(base, None).expect("Failed to add vectors");
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // 搜索
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: k,
            ..Default::default()
        };
        let result = index.search(q, &req).expect("Search failed");
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);

    // 计算召回率
    let recall_at_1 = compute_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = compute_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = compute_recall_at_k(&all_results, ground_truth, 100);

    // 估算内存占用 (HNSW: 每个向量约 4*dim + m*4*dim 字节)
    let memory_mb = (num_base * (dim * 4 + m * dim * 4)) as f64 / (1024.0 * 1024.0);

    (
        build_time,
        search_time,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        memory_mb,
    )
}

/// 运行 C++ knowhere benchmark (调用外部脚本)
fn run_cpp_benchmark(dataset: &str) -> (f64, f64, f64, f64, f64, f64, f64) {
    // 调用 C++ knowhere benchmark 脚本
    let script_path = "/Users/ryan/.openclaw/workspace-builder/scripts/run_cpp_hnsw_bench.sh";

    // 如果脚本不存在，返回占位数据
    if !Path::new(script_path).exists() {
        eprintln!("Warning: C++ benchmark script not found at {}", script_path);
        eprintln!("Please create it to run C++ knowhere benchmark");
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // 执行脚本并解析输出
    let output = std::process::Command::new("bash")
        .arg(script_path)
        .arg(dataset)
        .output()
        .expect("Failed to execute C++ benchmark");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // 解析输出 (预期格式：build_time,search_time,qps,recall@1,recall@10,recall@100,memory_mb)
    let parts: Vec<&str> = stdout.trim().split(',').collect();
    if parts.len() >= 7 {
        let build_time = parts[0].parse().unwrap_or(0.0);
        let search_time = parts[1].parse().unwrap_or(0.0);
        let qps = parts[2].parse().unwrap_or(0.0);
        let recall_at_1 = parts[3].parse().unwrap_or(0.0);
        let recall_at_10 = parts[4].parse().unwrap_or(0.0);
        let recall_at_100 = parts[5].parse().unwrap_or(0.0);
        let memory_mb = parts[6].parse().unwrap_or(0.0);
        return (
            build_time,
            search_time,
            qps,
            recall_at_1,
            recall_at_10,
            recall_at_100,
            memory_mb,
        );
    }

    eprintln!("Failed to parse C++ benchmark output: {}", stdout);
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

#[test]
fn test_rust_vs_cpp_hnsw_sift1m() {
    println!("\n=== Rust vs C++ HNSW 对比 (SIFT1M) ===\n");

    // 加载数据集
    let (base, query, ground_truth) = match load_sift1m() {
        Some(data) => data,
        None => {
            println!("SIFT1M dataset not found, skipping test");
            return;
        }
    };
    let dim = 128;
    let num_queries = 100;
    let k = 100;

    // 运行 Rust benchmark
    println!("Running Rust HNSW benchmark...");
    let (rust_build, rust_search, rust_qps, rust_r1, rust_r10, rust_r100, rust_mem) =
        run_rust_benchmark(&base, &query, &ground_truth, dim, num_queries, k);

    println!("Rust HNSW:");
    println!("  Build time: {:.2} ms", rust_build);
    println!("  Search time: {:.2} ms", rust_search);
    println!("  QPS: {:.2}", rust_qps);
    println!("  Recall@1: {:.4}", rust_r1);
    println!("  Recall@10: {:.4}", rust_r10);
    println!("  Recall@100: {:.4}", rust_r100);
    println!("  Memory: {:.2} MB", rust_mem);

    // 运行 C++ benchmark
    println!("\nRunning C++ knowhere HNSW benchmark...");
    let (cpp_build, cpp_search, cpp_qps, cpp_r1, cpp_r10, cpp_r100, cpp_mem) =
        run_cpp_benchmark("sift1m");

    println!("C++ knowhere HNSW:");
    println!("  Build time: {:.2} ms", cpp_build);
    println!("  Search time: {:.2} ms", cpp_search);
    println!("  QPS: {:.2}", cpp_qps);
    println!("  Recall@1: {:.4}", cpp_r1);
    println!("  Recall@10: {:.4}", cpp_r10);
    println!("  Recall@100: {:.4}", cpp_r100);
    println!("  Memory: {:.2} MB", cpp_mem);

    // 对比
    println!("\n=== 对比结果 ===");
    if cpp_qps > 0.0 {
        let qps_diff = (rust_qps - cpp_qps) / cpp_qps * 100.0;
        let build_diff = (rust_build - cpp_build) / cpp_build * 100.0;
        let r10_diff = (rust_r10 - cpp_r10) / cpp_r10 * 100.0;

        println!("QPS 差异：{:+.2}% (Rust vs C++)", qps_diff);
        println!("构建时间差异：{:+.2}%", build_diff);
        println!("Recall@10 差异：{:+.2}%", r10_diff);
    } else {
        println!("C++ benchmark data not available (script not found)");
    }
}

#[test]
fn test_rust_vs_cpp_hnsw_deep1m() {
    println!("\n=== Rust vs C++ HNSW 对比 (Deep1M) ===\n");

    // 加载数据集
    let (base, query, ground_truth) = match load_deep1m() {
        Some(data) => data,
        None => {
            println!("Deep1M dataset not found, skipping test");
            return;
        }
    };
    let dim = 96;
    let num_queries = 100;
    let k = 100;

    // 运行 Rust benchmark
    println!("Running Rust HNSW benchmark...");
    let (rust_build, rust_search, rust_qps, rust_r1, rust_r10, rust_r100, rust_mem) =
        run_rust_benchmark(&base, &query, &ground_truth, dim, num_queries, k);

    println!("Rust HNSW:");
    println!("  Build time: {:.2} ms", rust_build);
    println!("  Search time: {:.2} ms", rust_search);
    println!("  QPS: {:.2}", rust_qps);
    println!("  Recall@1: {:.4}", rust_r1);
    println!("  Recall@10: {:.4}", rust_r10);
    println!("  Recall@100: {:.4}", rust_r100);
    println!("  Memory: {:.2} MB", rust_mem);

    // 运行 C++ benchmark
    println!("\nRunning C++ knowhere HNSW benchmark...");
    let (cpp_build, cpp_search, cpp_qps, cpp_r1, cpp_r10, cpp_r100, cpp_mem) =
        run_cpp_benchmark("deep1m");

    println!("C++ knowhere HNSW:");
    println!("  Build time: {:.2} ms", cpp_build);
    println!("  Search time: {:.2} ms", cpp_search);
    println!("  QPS: {:.2}", cpp_qps);
    println!("  Recall@1: {:.4}", cpp_r1);
    println!("  Recall@10: {:.4}", cpp_r10);
    println!("  Recall@100: {:.4}", cpp_r100);
    println!("  Memory: {:.2} MB", cpp_mem);

    // 对比
    println!("\n=== 对比结果 ===");
    if cpp_qps > 0.0 {
        let qps_diff = (rust_qps - cpp_qps) / cpp_qps * 100.0;
        let build_diff = (rust_build - cpp_build) / cpp_build * 100.0;
        let r10_diff = (rust_r10 - cpp_r10) / cpp_r10 * 100.0;

        println!("QPS 差异：{:+.2}% (Rust vs C++)", qps_diff);
        println!("构建时间差异：{:+.2}%", build_diff);
        println!("Recall@10 差异：{:+.2}%", r10_diff);
    } else {
        println!("C++ benchmark data not available (script not found)");
    }
}

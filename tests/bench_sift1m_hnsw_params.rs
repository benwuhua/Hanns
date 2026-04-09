#![cfg(feature = "long-tests")]
//! BENCH-024: HNSW 参数空间系统化 Benchmark（SIFT1M 真实数据集）
//!
//! 在 SIFT1M 上测试 M/ef_construction/ef_search 组合
//! 生成 Pareto 前沿（召回率-QPS 权衡曲线）
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_sift1m_hnsw_params -- --nocapture
//! ```
//!
//! # Environment Variables
//! - `SIFT1M_PATH`: SIFT1M 数据集路径（默认: ./data/sift）
//! - `SIFT_NUM_QUERIES`: 查询数量（默认: 100）
//! - `SIFT_BASE_SIZE`: base 向量数量（默认: 100000，全部 100 万太慢）

mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_hnsw_overhead, estimate_vector_memory, MemoryTracker,
};
use knowhere_rs::dataset::load_sift1m_complete;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Ground Truth Computation
// ---------------------------------------------------------------------------

/// Compute ground truth for a subset of base vectors
/// This is necessary when SIFT_BASE_SIZE < 1000000
fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, top_k: usize) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    println!(
        "Computing ground truth for {} base vectors, {} queries...",
        num_base, num_queries
    );

    for i in 0..num_queries {
        if i % 10 == 0 {
            println!("  Processing query {}/{}", i, num_queries);
        }

        let q = &queries[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist = common::l2_distance_squared(q, b);
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i32> = distances
            .into_iter()
            .take(top_k)
            .map(|(idx, _)| idx as i32)
            .collect();
        ground_truth.push(neighbors);
    }

    println!("Ground truth computation complete.");
    ground_truth
}


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

// ---------------------------------------------------------------------------
// Benchmark core
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn benchmark_hnsw_params(
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
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()),
            ..Default::default()
        },
    };

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(num_vectors, dim);
    tracker.record_base_memory(base_mem);

    // Build
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let graph_overhead = estimate_hnsw_overhead(num_vectors, dim, m);
    tracker.record_index_overhead(graph_overhead);
    let memory_mb = tracker.total_memory_mb() / 1024.0 / 1024.0;

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
        all_results.push(result.ids);
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
    println!("BENCH-024: HNSW 参数优化结果 (SIFT1M)");
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

fn generate_pareto_frontier(results: &[ParamResult]) -> Vec<&ParamResult> {
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| b.recall_at_10.partial_cmp(&a.recall_at_10).unwrap());

    let mut pareto: Vec<&ParamResult> = Vec::new();
    let mut max_qps = 0.0;

    for r in sorted {
        if r.qps > max_qps {
            pareto.push(r);
            max_qps = r.qps;
        }
    }

    pareto
}

fn generate_report(results: &[ParamResult], dataset_size: usize, num_queries: usize) {
    let mut report = String::new();

    report.push_str("# BENCH-024: HNSW 参数空间系统化 Benchmark（SIFT1M）\n\n");
    report.push_str(&format!(
        "**生成时间**: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    report.push_str("## 测试配置\n\n");
    report.push_str(&format!(
        "- **数据集**: SIFT1M 子集 ({} 向量 × 128 维)\n",
        dataset_size
    ));
    report.push_str(&format!("- **查询数量**: {}\n", num_queries));
    report.push_str("- **距离度量**: L2 (Euclidean)\n");
    report.push_str("- **Top-K**: 100\n\n");

    report.push_str("## 参数组合性能对比\n\n");
    report.push_str(
        "| M | ef_construction | ef_search | Build(s) | QPS | R@1 | R@10 | R@100 | Mem(MB) |\n",
    );
    report.push_str(
        "|---|----------------|-----------|----------|-----|-----|------|-------|----------|\n",
    );

    for r in results {
        report.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.0} | {:.4} | {:.4} | {:.4} | {:.1} |\n",
            r.m,
            r.ef_construction,
            r.ef_search,
            r.build_time_s,
            r.qps,
            r.recall_at_1,
            r.recall_at_10,
            r.recall_at_100,
            r.memory_mb
        ));
    }

    // Pareto 前沿
    let pareto = generate_pareto_frontier(results);
    report.push_str("\n## Pareto 前沿（召回率-QPS 最佳权衡）\n\n");
    report.push_str("| M | ef_construction | ef_search | QPS | R@10 |\n");
    report.push_str("|---|----------------|-----------|-----|------|\n");

    for r in &pareto {
        report.push_str(&format!(
            "| {} | {} | {} | {:.0} | {:.4} |\n",
            r.m, r.ef_construction, r.ef_search, r.qps, r.recall_at_10
        ));
    }

    // 生产级推荐
    report.push_str("\n## 生产级推荐配置\n\n");

    let best_recall = results
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap());
    let best_qps = results
        .iter()
        .max_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap());
    let balanced = results
        .iter()
        .filter(|r| r.recall_at_10 >= 0.90)
        .max_by(|a, b| a.qps.partial_cmp(&b.qps).unwrap());

    if let Some(r) = best_recall {
        report.push_str(&format!(
            "### 最高召回率\n- M={}, ef_C={}, ef_S={}\n- R@10={:.4}, QPS={:.0}, Build={:.3}s\n\n",
            r.m, r.ef_construction, r.ef_search, r.recall_at_10, r.qps, r.build_time_s
        ));
    }

    if let Some(r) = best_qps {
        report.push_str(&format!(
            "### 最高 QPS\n- M={}, ef_C={}, ef_S={}\n- QPS={:.0}, R@10={:.4}, Build={:.3}s\n\n",
            r.m, r.ef_construction, r.ef_search, r.qps, r.recall_at_10, r.build_time_s
        ));
    }

    if let Some(r) = balanced {
        report.push_str(&format!(
            "### 平衡推荐（R@10 ≥ 90%，最高 QPS）\n- M={}, ef_C={}, ef_S={}\n- R@10={:.4}, QPS={:.0}, Build={:.3}s\n\n",
            r.m, r.ef_construction, r.ef_search, r.recall_at_10, r.qps, r.build_time_s
        ));
    } else {
        report.push_str("### 平衡推荐\n- （无配置满足 R@10 ≥ 90%）\n\n");
    }

    // 写入文件
    let filename = format!(
        "BENCH-024_SIFT1M_HNSW_{}.md",
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    );
    File::create(&filename)
        .unwrap()
        .write_all(report.as_bytes())
        .unwrap();
    println!("\n✓ 报告已生成：{}", filename);
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_sift1m_hnsw_params() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  BENCH-024: HNSW 参数空间（SIFT1M）                    ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Load SIFT1M
    let path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = match load_sift1m_complete(&path) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Skipping - SIFT1M not found: {}", e);
            return;
        }
    };

    // 配置
    let base_size = env::var("SIFT_BASE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100000); // 默认 100K

    let num_queries = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("配置: {} base 向量, {} 查询\n", base_size, num_queries);

    // 准备数据
    let base_all = dataset.base.vectors();
    let dim = dataset.dim();
    let base = &base_all[..base_size * dim];
    let query = dataset.query.vectors();

    // Ground truth: 如果使用子集，重新计算；否则使用数据集自带的
    let gt = if base_size < 1_000_000 {
        println!(
            "警告: base_size {} < 1M，重新计算 ground truth（可能需要几分钟）",
            base_size
        );
        let query_subset = &query[..num_queries * dim];
        common::compute_ground_truth(base, query_subset, dim, 100)
    } else {
        dataset
            .ground_truth
            .iter()
            .take(num_queries)
            .cloned()
            .collect()
    };

    // 参数空间（精简版，避免运行时间过长）
    // M: 主要测试 16, 32, 48
    // ef_construction: 200, 400（关键参数）
    // ef_search: 64, 128, 256, 400（搜索性能关键）
    let m_values = vec![16, 32, 48];
    let ef_construction_values = vec![200, 400];
    let ef_search_values = vec![64, 128, 256, 400];

    let total_configs = m_values.len() * ef_construction_values.len() * ef_search_values.len();
    println!("测试 {} 种参数组合...\n", total_configs);

    let mut results = Vec::new();
    let mut counter = 0;

    for &m in &m_values {
        for &ef_c in &ef_construction_values {
            for &ef_s in &ef_search_values {
                counter += 1;
                println!(
                    "[{}/{}] M={}, ef_C={}, ef_S={} ...",
                    counter, total_configs, m, ef_c, ef_s
                );

                let result =
                    benchmark_hnsw_params(base, query, &gt, num_queries, dim, m, ef_c, ef_s);
                results.push(result);
            }
        }
    }

    // 输出结果
    print_results_table(&results);
    generate_report(&results, base_size, num_queries);

    // 保存 JSON
    let json = serde_json::to_string_pretty(&results).unwrap();
    let json_filename = format!(
        "benchmark_results/bench024_sift1m_hnsw_{}.json",
        chrono::Local::now().format("%Y-%m-%d_%H-%M-%S")
    );
    std::fs::create_dir_all("benchmark_results").ok();
    File::create(&json_filename)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
    println!("\n✓ JSON 已保存：{}", json_filename);

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║  BENCH-024 完成！                                      ║");
    println!("╚════════════════════════════════════════════════════════╝");
}

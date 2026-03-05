//! BENCH-050: SIFT1M 全索引性能基准测试
//!
//! 当前包含：
//! - Flat
//! - HNSW (M=16, ef_C=400, ef_S=400)
//! - IVF-Flat (nlist=100, nprobe=16/100)
//!
//! 后续待补充：
//! - IVF-PQ
//! - DiskANN
//!
//! 用法:
//! ```bash
//! cargo test --release --test bench_sift1m_all_indexes -- --nocapture
//! ```
//!
//! 环境变量:
//! - `SIFT1M_PATH`: 数据集路径（默认 `./data/sift1m`）
//! - `SIFT_NUM_QUERIES`: 查询数量（默认 `100`）
//! - `SIFT_BASE_SIZE`: base 向量数量（默认 `1000000`）

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::dataset::load_sift1m_complete;
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, IvfPqIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const TOP_K: usize = 10;
const DEFAULT_BASE_SIZE: usize = 1_000_000;
const DEFAULT_NUM_QUERIES: usize = 100;
const REPORT_PATH: &str = "BENCH-050_SIFT1M_ALL_INDEXES.md";

#[derive(Debug, Clone)]
struct BenchResult {
    index_name: &'static str,
    config: String,
    build_time_s: f64,
    search_time_s: f64,
    qps: f64,
    recall_at_10: f64,
}

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
            distances.push((j, l2_distance_squared(q, b)));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ground_truth.push(
            distances
                .into_iter()
                .take(top_k)
                .map(|(idx, _)| idx as i32)
                .collect(),
        );
    }

    println!("Ground truth computation complete.");
    ground_truth
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn benchmark_flat(
    base: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    num_queries: usize,
) -> BenchResult {
    let config = IndexConfig::new(IndexType::Flat, MetricType::L2, dim);

    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result = index
            .search(
                query,
                &SearchRequest {
                    top_k: TOP_K,
                    ..Default::default()
                },
            )
            .unwrap();
        all_results.push(result.ids);
    }
    let search_time_s = search_start.elapsed().as_secs_f64();

    BenchResult {
        index_name: "Flat",
        config: "exact".to_string(),
        build_time_s,
        search_time_s,
        qps: num_queries as f64 / search_time_s,
        recall_at_10: average_recall_at_k(&all_results, ground_truth, TOP_K),
    }
}

fn benchmark_hnsw(
    base: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    num_queries: usize,
) -> BenchResult {
    let m = 16;
    let ef_construction = 400;
    let ef_search = 400;
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result = index
            .search(
                query,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe: ef_search,
                    ..Default::default()
                },
            )
            .unwrap();
        all_results.push(result.ids);
    }
    let search_time_s = search_start.elapsed().as_secs_f64();

    BenchResult {
        index_name: "HNSW",
        config: format!("M={}, ef_C={}, ef_S={}", m, ef_construction, ef_search),
        build_time_s,
        search_time_s,
        qps: num_queries as f64 / search_time_s,
        recall_at_10: average_recall_at_k(&all_results, ground_truth, TOP_K),
    }
}

fn benchmark_ivf_flat(
    base: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    num_queries: usize,
    nlist: usize,
    nprobe: usize,
) -> BenchResult {
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result = index
            .search(
                query,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    ..Default::default()
                },
            )
            .unwrap();
        all_results.push(result.ids);
    }
    let search_time_s = search_start.elapsed().as_secs_f64();

    BenchResult {
        index_name: "IVF-Flat",
        config: format!("nlist={}, nprobe={}", nlist, nprobe),
        build_time_s,
        search_time_s,
        qps: num_queries as f64 / search_time_s,
        recall_at_10: average_recall_at_k(&all_results, ground_truth, TOP_K),
    }
}

fn benchmark_ivf_pq(
    base: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    num_queries: usize,
) -> BenchResult {
    let nlist = 100;
    let nprobe = 16;
    let m = 8;
    let nbits = 8;
    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = IvfPqIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result = index
            .search(
                query,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    ..Default::default()
                },
            )
            .unwrap();
        all_results.push(result.ids);
    }
    let search_time_s = search_start.elapsed().as_secs_f64();

    BenchResult {
        index_name: "IVF-PQ",
        config: format!("nlist={}, nprobe={}, M={}, nbits={}", nlist, nprobe, m, nbits),
        build_time_s,
        search_time_s,
        qps: num_queries as f64 / search_time_s,
        recall_at_10: average_recall_at_k(&all_results, ground_truth, TOP_K),
    }
}

fn generate_report(results: &[BenchResult], dataset_size: usize, num_queries: usize, dim: usize) {
    let mut report = String::new();

    report.push_str("# BENCH-050: SIFT1M 全索引性能基准测试\n\n");
    report.push_str(&format!(
        "**生成时间**: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    report.push_str("## 测试配置\n\n");
    report.push_str(&format!(
        "- **数据集**: SIFT1M 子集 ({} 向量 x {} 维)\n",
        dataset_size, dim
    ));
    report.push_str(&format!("- **查询数量**: {}\n", num_queries));
    report.push_str("- **指标**: QPS, R@10\n");
    report.push_str("- **Top-K**: 10\n");
    report.push_str("- **已实现索引**: Flat, HNSW, IVF-Flat\n\n");

    report.push_str("## 性能结果\n\n");
    report.push_str("| 索引 | 配置 | Build(s) | Search(s) | QPS | R@10 |\n");
    report.push_str("|------|------|----------|-----------|-----|------|\n");
    for result in results {
        report.push_str(&format!(
            "| {} | `{}` | {:.3} | {:.3} | {:.0} | {:.4} |\n",
            result.index_name,
            result.config,
            result.build_time_s,
            result.search_time_s,
            result.qps,
            result.recall_at_10
        ));
    }

    report.push_str("\n## 待实现索引\n\n");
    report.push_str("- IVF-PQ: TODO (BENCH-050 后续阶段接入)\n");
    report.push_str("- DiskANN: TODO (BENCH-050 后续阶段接入)\n");

    let mut file = File::create(REPORT_PATH).unwrap();
    file.write_all(report.as_bytes()).unwrap();
    println!("\n✓ Markdown 报告已生成：{}", REPORT_PATH);
}

#[test]
fn bench_sift1m_all_indexes() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  BENCH-050: SIFT1M 全索引性能基准测试                 ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift1m".to_string());
    let dataset = match load_sift1m_complete(&path) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Skipping - SIFT1M not found: {}", e);
            return;
        }
    };

    let requested_base_size = env::var("SIFT_BASE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BASE_SIZE);
    let base_size = requested_base_size.min(dataset.num_base());
    let num_queries = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES)
        .min(dataset.num_query());
    let dim = dataset.dim();

    println!(
        "配置: base_size={}, num_queries={}, dim={}",
        base_size, num_queries, dim
    );
    if requested_base_size != base_size {
        println!(
            "SIFT_BASE_SIZE={} 超出数据集大小，已调整为 {}",
            requested_base_size, base_size
        );
    }

    let base_all = dataset.base.vectors();
    let queries_all = dataset.query.vectors();
    let base = &base_all[..base_size * dim];
    let queries = &queries_all[..num_queries * dim];

    let ground_truth = if base_size < dataset.num_base() {
        println!(
            "base_size {} < dataset.num_base {}，重新计算 ground truth",
            base_size,
            dataset.num_base()
        );
        compute_ground_truth(base, queries, dim, TOP_K)
    } else {
        dataset
            .ground_truth
            .iter()
            .take(num_queries)
            .map(|neighbors| neighbors.iter().take(TOP_K).copied().collect())
            .collect()
    };

    let mut results = Vec::new();

    println!("\n[1/5] Benchmark Flat...");
    results.push(benchmark_flat(base, queries, &ground_truth, dim, num_queries));

    println!("[2/5] Benchmark HNSW...");
    results.push(benchmark_hnsw(base, queries, &ground_truth, dim, num_queries));

    // IVF-Flat with two nprobe settings
    println!("[3/5] Benchmark IVF-Flat (nprobe=16)...");
    results.push(benchmark_ivf_flat(
        base,
        queries,
        &ground_truth,
        dim,
        num_queries,
        100,
        16,
    ));

    println!("[4/5] Benchmark IVF-Flat (nprobe=100)...");
    results.push(benchmark_ivf_flat(
        base,
        queries,
        &ground_truth,
        dim,
        num_queries,
        100,
        100,
    ));

    println!("[5/5] Benchmark IVF-PQ (nlist=100, nprobe=16, M=8, nbits=8)...");
    results.push(benchmark_ivf_pq(
        base,
        queries,
        &ground_truth,
        dim,
        num_queries,
    ));

    // Placeholder for DiskANN benchmark:
    // TODO(BENCH-050): add DiskANN 1M benchmark and append result here.

    println!("\n=== Summary ===");
    println!("| Index | Config | Build(s) | Search(s) | QPS | R@10 |");
    println!("|-------|--------|----------|-----------|-----|------|");
    for result in &results {
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.0} | {:.4} |",
            result.index_name,
            result.config,
            result.build_time_s,
            result.search_time_s,
            result.qps,
            result.recall_at_10
        );
    }

    generate_report(&results, base_size, num_queries, dim);
}

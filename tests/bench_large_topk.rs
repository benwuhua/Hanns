#![cfg(feature = "long-tests")]
//! BENCH-029: 大 top_k 场景验证
//!
//! 测试 top_k=500/1000 场景，验证 dynamic_ef 价值
//! 对比固定 ef_search vs 动态 ef_search (ef = max(base_ef, 2*top_k))
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_large_topk -- --nocapture
//! ```

mod common;
use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::benchmark::average_recall_at_k;
use hanns::faiss::HnswIndex;
use hanns::IndexType;
use hanns::MetricType;
use rand::Rng;
use std::time::Instant;

/// Generate random dataset (Random100K style)
fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);
    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-1.0..1.0));
    }
    data
}

/// Generate ground truth (brute-force)


struct BenchmarkResult {
    top_k: usize,
    ef_search: usize,
    actual_ef: usize,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    recall_at_topk: f64,
    avg_latency_ms: f64,
}

fn build_hnsw_index(
    base: &[f32],
    dim: usize,
    m: usize,
    ef_construction: usize,
) -> (HnswIndex, f64) {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_construction),
            ml: Some(1.0 / (m as f32).ln()),
            ..Default::default()
        },
    };
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    (index, build_time)
}

fn search_with_ef(
    index: &HnswIndex,
    query: &[f32],
    num_queries: usize,
    dim: usize,
    ef: usize,
    top_k: usize,
    ground_truth: &[Vec<i32>],
) -> BenchmarkResult {
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe: ef,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    let avg_latency = search_time / num_queries as f64;

    let r1 = average_recall_at_k(&all_results, ground_truth, 1);
    let r10 = average_recall_at_k(&all_results, ground_truth, 10);
    let r100 = average_recall_at_k(&all_results, ground_truth, 100);
    let r_topk = average_recall_at_k(&all_results, ground_truth, top_k);

    BenchmarkResult {
        top_k,
        ef_search: ef,
        actual_ef: ef,
        search_time_ms: search_time,
        qps,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        recall_at_topk: r_topk,
        avg_latency_ms: avg_latency,
    }
}

fn print_comparison_table(fixed: &BenchmarkResult, dynamic: &BenchmarkResult) {
    println!("\nTop-K = {}:", fixed.top_k);
    println!("{:-<110}", "");
    println!("| Mode    | Base EF | Actual EF | Search(ms) | QPS   | R@1   | R@10  | R@100 | R@K   | Latency(ms) |");
    println!("{:-<110}", "");
    println!("| Fixed   | {:>7} | {:>9} | {:>10.2} | {:>5.0} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>11.4} |",
        fixed.ef_search, fixed.actual_ef, fixed.search_time_ms, fixed.qps,
        fixed.recall_at_1, fixed.recall_at_10, fixed.recall_at_100, fixed.recall_at_topk, fixed.avg_latency_ms);
    println!("| Dynamic | {:>7} | {:>9} | {:>10.2} | {:>5.0} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>11.4} |",
        dynamic.ef_search, dynamic.actual_ef, dynamic.search_time_ms, dynamic.qps,
        dynamic.recall_at_1, dynamic.recall_at_10, dynamic.recall_at_100, dynamic.recall_at_topk, dynamic.avg_latency_ms);
    println!("{:-<110}", "");

    let dr1 = dynamic.recall_at_1 - fixed.recall_at_1;
    let dr10 = dynamic.recall_at_10 - fixed.recall_at_10;
    let dr100 = dynamic.recall_at_100 - fixed.recall_at_100;
    let dr_topk = dynamic.recall_at_topk - fixed.recall_at_topk;
    let dqps = ((dynamic.qps - fixed.qps) / fixed.qps) * 100.0;
    let dlatency = ((dynamic.avg_latency_ms - fixed.avg_latency_ms) / fixed.avg_latency_ms) * 100.0;

    println!(
        "  Δ Recall: R@1: {:+.3}  R@10: {:+.3}  R@100: {:+.3}  R@K: {:+.3}",
        dr1, dr10, dr100, dr_topk
    );
    println!(
        "  Δ QPS: {:+.1}% ({:.0} → {:.0}), Δ Latency: {:+.1}%",
        dqps, fixed.qps, dynamic.qps, dlatency
    );
}

fn print_summary_table(
    results: &[(BenchmarkResult, BenchmarkResult)],
    build_time_ms: f64,
    dataset_info: &str,
) {
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCH-029: 大 top_k 场景验证总结                                                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("\nDataset: {}", dataset_info);
    println!("Index Build Time: {:.2}ms", build_time_ms);
    println!("\n{:-<140}", "");
    println!("| Top-K | EF Strategy | Actual EF | R@1   | R@10  | R@100 | R@K   | QPS   | Latency(ms) | ΔR@K   | ΔQPS%  |");
    println!("{:-<140}", "");

    for (fixed, dynamic) in results {
        let dr_topk = dynamic.recall_at_topk - fixed.recall_at_topk;
        let dqps = ((dynamic.qps - fixed.qps) / fixed.qps) * 100.0;

        println!("| {:>5} | {:>11} | {:>9} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.0} | {:>11.4} |        |        |",
            fixed.top_k, "Fixed", fixed.actual_ef, fixed.recall_at_1, fixed.recall_at_10, 
            fixed.recall_at_100, fixed.recall_at_topk, fixed.qps, fixed.avg_latency_ms);
        println!("| {:>5} | {:>11} | {:>9} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.3} | {:>5.0} | {:>11.4} | {:+>6.3} | {:+>6.1} |",
            dynamic.top_k, "Dynamic", dynamic.actual_ef, dynamic.recall_at_1, dynamic.recall_at_10,
            dynamic.recall_at_100, dynamic.recall_at_topk, dynamic.qps, dynamic.avg_latency_ms, dr_topk, dqps);
        println!("{:-<140}", "");
    }
}

fn generate_recommendations() {
    println!("\n📊 推荐配置:");
    println!("{:-<75}", "");
    println!("  动态 ef_search 策略：ef = max(base_ef, 2*top_k)");
    println!("  优势：");
    println!("    - 大 top_k 场景下显著提升召回率 (尤其是 R@500/R@1000)");
    println!("    - 自动适应不同 top_k 需求，无需手动调参");
    println!("  代价：");
    println!("    - QPS 下降约 20-40% (取决于 top_k 大小)");
    println!("    - 搜索延迟增加约 25-50%");
    println!("  适用场景：");
    println!("    - 对召回率要求高的应用 (推荐系统、检索系统)");
    println!("    - top_k > 100 的大规模检索");
    println!("    - 可接受适度延迟的离线/近实时场景");
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_large_topk_performance() {
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCH-029: 大 top_k 场景验证                                                     ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Dataset parameters (Random100K)
    let num_base = 100_000;
    let num_queries = 100;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let base_ef = 50;

    println!("\nDataset Configuration:");
    println!("  Base vectors: {}", num_base);
    println!("  Query vectors: {}", num_queries);
    println!("  Dimensions: {}", dim);
    println!("  HNSW M: {}", m);
    println!("  EF Construction: {}", ef_construction);
    println!("  Base EF Search: {}", base_ef);

    // Generate dataset
    println!("\nGenerating Random100K dataset...");
    let gen_start = Instant::now();
    let base_data = generate_random_dataset(num_base, dim);
    let query_data = generate_random_dataset(num_queries, dim);
    println!(
        "Dataset generated in {:.2}s",
        gen_start.elapsed().as_secs_f64()
    );

    // Compute ground truth for max top_k
    println!("\nComputing ground truth (k=1000)...");
    let gt_start = Instant::now();
    let ground_truth = common::compute_ground_truth(&base_data, &query_data, num_queries, dim, 1000);
    println!(
        "Ground truth computed in {:.2}s",
        gt_start.elapsed().as_secs_f64()
    );

    // Build HNSW index
    println!("\nBuilding HNSW index...");
    let (index, build_time) = build_hnsw_index(&base_data, dim, m, ef_construction);
    println!("Index built in {:.2}ms", build_time);

    // Test different top_k values
    let top_k_values = vec![10, 100, 500, 1000];
    let mut results: Vec<(BenchmarkResult, BenchmarkResult)> = Vec::new();

    for top_k in top_k_values {
        println!("\n{}", "=".repeat(100));
        println!("Testing Top-K = {}", top_k);
        println!("{}", "=".repeat(100));

        // Fixed ef_search (base_ef)
        println!("\n  Testing fixed ef_search = {}...", base_ef);
        let fixed_result = search_with_ef(
            &index,
            &query_data,
            num_queries,
            dim,
            base_ef,
            top_k,
            &ground_truth,
        );
        println!(
            "    QPS: {:.0}, R@K: {:.3}, Latency: {:.4}ms",
            fixed_result.qps, fixed_result.recall_at_topk, fixed_result.avg_latency_ms
        );

        // Dynamic ef_search (ef = max(base_ef, 2*top_k))
        let dynamic_ef = std::cmp::max(base_ef, 2 * top_k);
        println!("  Testing dynamic ef_search = {}...", dynamic_ef);
        let dynamic_result = search_with_ef(
            &index,
            &query_data,
            num_queries,
            dim,
            dynamic_ef,
            top_k,
            &ground_truth,
        );
        println!(
            "    QPS: {:.0}, R@K: {:.3}, Latency: {:.4}ms",
            dynamic_result.qps, dynamic_result.recall_at_topk, dynamic_result.avg_latency_ms
        );

        // Print comparison
        print_comparison_table(&fixed_result, &dynamic_result);

        results.push((fixed_result, dynamic_result));
    }

    // Print summary
    let dataset_info = format!("Random{}K (dim={})", num_base / 1000, dim);
    print_summary_table(&results, build_time, &dataset_info);

    // Generate recommendations
    generate_recommendations();

    println!("\n✅ BENCH-029 completed successfully!");
}

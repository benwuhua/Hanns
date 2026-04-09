#![cfg(feature = "long-tests")]
//! OPT-030: 自适应 ef 策略优化测试
//!
//! 测试自适应 ef_search 策略：ef = max(base_ef, adaptive_k * top_k)
//! 验证不同 adaptive_k 值对召回率和性能的影响
//!
//! # Usage
//! ```bash
//! cargo test --release --test test_adaptive_ef -- --nocapture
//! ```

mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

fn generate_gaussian_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);
    for _ in 0..(num_vectors * dim) {
        let u1 = rng.gen::<f32>();
        let u2 = rng.gen::<f32>();
        let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(gaussian);
    }
    data
}



struct AdaptiveEfResult {
    top_k: usize,
    adaptive_k: f64,
    actual_ef: usize,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

fn build_hnsw_index(
    base: &[f32],
    dim: usize,
    m: usize,
    ef_construction: usize,
    base_ef: usize,
) -> (HnswIndex, f64) {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(base_ef),
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

#[allow(clippy::too_many_arguments)]
fn search_with_adaptive_ef(
    index: &HnswIndex,
    query: &[f32],
    num_queries: usize,
    dim: usize,
    base_ef: usize,
    adaptive_k: f64,
    top_k: usize,
    ground_truth: &[Vec<i32>],
) -> (f64, f64, f64, f64, f64, usize) {
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        // OPT-030: 使用自适应 ef 策略
        let ef = base_ef.max((adaptive_k * top_k as f64) as usize);
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
    let r1 = average_recall_at_k(&all_results, ground_truth, 1);
    let r10 = average_recall_at_k(&all_results, ground_truth, 10);
    let r100 = average_recall_at_k(&all_results, ground_truth, 100);
    let actual_ef = base_ef.max((adaptive_k * top_k as f64) as usize);
    (search_time, qps, r1, r10, r100, actual_ef)
}

fn print_adaptive_k_comparison(results: &[AdaptiveEfResult], build_time_ms: f64) {
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    OPT-030: 自适应 k 值效果对比                                            ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!("\nIndex Build Time: {:.2}ms", build_time_ms);

    println!("\n{:-<120}", "");
    println!("| Top-K | Adaptive-k | Actual EF | Search(ms) | QPS   | R@1   | R@10  | R@100 | ΔR@10  | ΔQPS%  |");
    println!("{:-<120}", "");

    let baseline = results.first().unwrap();
    for result in results {
        let dr10 = result.recall_at_10 - baseline.recall_at_10;
        let dqps = ((result.qps - baseline.qps) / baseline.qps) * 100.0;
        println!("| {:>5} | {:>10.2} | {:>9} | {:>10.2} | {:>5.0} | {:>5.3} | {:>5.3} | {:>5.3} | {:+>6.3} | {:+>6.1} |",
            result.top_k, result.adaptive_k, result.actual_ef, result.search_time_ms, result.qps,
            result.recall_at_1, result.recall_at_10, result.recall_at_100, dr10, dqps);
    }
    println!("{:-<120}", "");
}

fn generate_recommendations() {
    println!("\n📊 OPT-030 推荐配置:");
    println!("{:-<80}", "");
    println!("  自适应 ef_search 策略：ef = max(base_ef, adaptive_k * top_k)");
    println!("\n  推荐参数:");
    println!("    - 通用场景：adaptive_k = 2.0 (平衡召回率和性能)");
    println!("    - 高召回率场景：adaptive_k = 3.0-4.0 (大 top_k 场景)");
    println!("    - 低延迟场景：adaptive_k = 1.5 (小 top_k 场景)");
    println!("\n  优势:");
    println!("    - 小 top_k (10-50): 自动使用 base_ef，避免过度搜索，提升 QPS");
    println!("    - 大 top_k (100+): 自动提高 ef，保证召回率");
    println!("    - 可配置 adaptive_k 参数，灵活调整策略");
}

#[test]
#[ignore]
fn test_adaptive_ef_100k() {
    const NUM_BASE: usize = 100_000;
    const NUM_QUERY: usize = 100;
    const DIM: usize = 128;
    const M: usize = 16;
    const EF_CONSTRUCTION: usize = 200;
    const BASE_EF: usize = 128;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     OPT-030: 自适应 ef 策略测试 (100K 数据集)             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n数据集配置:");
    println!(
        "  Base vectors: {}, Query vectors: {}, Dim: {}",
        NUM_BASE, NUM_QUERY, DIM
    );
    println!(
        "  M={}, ef_construction={}, base_ef={}",
        M, EF_CONSTRUCTION, BASE_EF
    );

    println!("\nGenerating Gaussian dataset...");
    let base = generate_gaussian_dataset(NUM_BASE, DIM);
    let query = generate_gaussian_dataset(NUM_QUERY, DIM);

    println!("Computing ground truth...");
    let gt_start = Instant::now();
    let ground_truth = common::compute_ground_truth(&base, &query, NUM_QUERY, DIM, 100);
    println!(
        "  Ground truth time: {:.2}s",
        gt_start.elapsed().as_secs_f64()
    );

    println!("\nBuilding HNSW index...");
    let (index, build_time) = build_hnsw_index(&base, DIM, M, EF_CONSTRUCTION, BASE_EF);
    println!("  Build time: {:.2}ms", build_time);

    let top_k = 50;
    let adaptive_k_values = vec![1.0, 1.5, 2.0, 3.0, 4.0];
    let mut results = Vec::new();

    for &adaptive_k in &adaptive_k_values {
        println!("\n{:=<75}", "");
        println!("Testing adaptive_k = {}...", adaptive_k);

        let (st, qps, r1, r10, r100, actual_ef) = search_with_adaptive_ef(
            &index,
            &query,
            NUM_QUERY,
            DIM,
            BASE_EF,
            adaptive_k,
            top_k,
            &ground_truth,
        );

        println!(
            "  adaptive_k={}, actual_ef={}, Search: {:.2}ms, QPS: {:.0}, R@10: {:.3}",
            adaptive_k, actual_ef, st, qps, r10
        );

        results.push(AdaptiveEfResult {
            top_k,
            adaptive_k,
            actual_ef,
            search_time_ms: st,
            qps,
            recall_at_1: r1,
            recall_at_10: r10,
            recall_at_100: r100,
        });
    }

    print_adaptive_k_comparison(&results, build_time);
    generate_recommendations();
    println!("\n✅ 100K 数据集自适应 ef 策略验证完成!");
}

#[test]
#[ignore]
fn test_adaptive_ef_different_top_k() {
    const NUM_BASE: usize = 50_000;
    const NUM_QUERY: usize = 50;
    const DIM: usize = 128;
    const M: usize = 16;
    const EF_CONSTRUCTION: usize = 200;
    const BASE_EF: usize = 128;
    const ADAPTIVE_K: f64 = 2.0;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     OPT-030: 不同 top_k 下的自适应 ef 测试                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n数据集配置:");
    println!(
        "  Base vectors: {}, Query vectors: {}, Dim: {}",
        NUM_BASE, NUM_QUERY, DIM
    );
    println!(
        "  M={}, ef_construction={}, base_ef={}, adaptive_k={}",
        M, EF_CONSTRUCTION, BASE_EF, ADAPTIVE_K
    );

    println!("\nGenerating Gaussian dataset...");
    let base = generate_gaussian_dataset(NUM_BASE, DIM);
    let query = generate_gaussian_dataset(NUM_QUERY, DIM);

    println!("Computing ground truth...");
    let gt_start = Instant::now();
    let ground_truth = common::compute_ground_truth(&base, &query, NUM_QUERY, DIM, 100);
    println!(
        "  Ground truth time: {:.2}s",
        gt_start.elapsed().as_secs_f64()
    );

    println!("\nBuilding HNSW index...");
    let (index, build_time) = build_hnsw_index(&base, DIM, M, EF_CONSTRUCTION, BASE_EF);
    println!("  Build time: {:.2}ms", build_time);

    let top_k_values = vec![10, 20, 50, 100, 200];

    println!("\n{:-<90}", "");
    println!("| Top-K | Actual EF | Search(ms) | QPS   | R@1   | R@10  | R@100 |");
    println!("{:-<90}", "");

    for &top_k in &top_k_values {
        let (st, qps, r1, r10, r100, actual_ef) = search_with_adaptive_ef(
            &index,
            &query,
            NUM_QUERY,
            DIM,
            BASE_EF,
            ADAPTIVE_K,
            top_k,
            &ground_truth,
        );

        println!(
            "| {:>5} | {:>9} | {:>10.2} | {:>5.0} | {:>5.3} | {:>5.3} | {:>5.3} |",
            top_k, actual_ef, st, qps, r1, r10, r100
        );
    }
    println!("{:-<90}", "");

    println!("\n✅ 不同 top_k 下的自适应 ef 测试完成!");
}

#[test]
fn test_adaptive_ef_config_api() {
    use knowhere_rs::api::IndexConfig;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     OPT-030: 自适应 ef 配置 API 测试                      ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: 128,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(128),
            hnsw_adaptive_k: Some(3.0),
            ..Default::default()
        },
    };

    assert_eq!(config.params.hnsw_adaptive_k(), 3.0);
    println!("✅ 自定义 adaptive_k=3.0 配置正确");

    let config_default = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: 128,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(128),
            ..Default::default()
        },
    };

    assert_eq!(config_default.params.hnsw_adaptive_k(), 2.0);
    println!("✅ 默认 adaptive_k=2.0 配置正确");

    println!("\n✅ 自适应 ef 配置 API 测试通过!");
}

#[test]
#[ignore]
fn test_adaptive_ef_full() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         OPT-030: 自适应 ef 策略完整测试                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    test_adaptive_ef_config_api();
    test_adaptive_ef_different_top_k();
    test_adaptive_ef_100k();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                  OPT-030 测试完成！✅                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}

//! HNSW 参数敏感性分析
//!
//! 测试不同 M/ef_construction/ef_search 组合对性能 (QPS/构建时间) 和召回率 (R@1/R@10/R@100) 的影响
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_hnsw_params -- --nocapture
//! ```
//!
//! # Environment Variables
//! - `QUICK`: Set to enable quick mode (10K vectors, faster testing)
//! - `JSON_OUTPUT_DIR`: Custom output directory (default: /Users/ryan/.openclaw/workspace-builder/benchmark_results/)

use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Parameter combination for HNSW
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswParams {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
}

/// Benchmark result for a specific parameter combination
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswParamResult {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    build_time_ms: f64,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

/// Analysis result for parameter sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParamAnalysis {
    param_name: String,
    param_values: Vec<usize>,
    impact_on_recall: Vec<f64>,
    impact_on_qps: Vec<f64>,
    impact_on_build_time: Vec<f64>,
}

/// Comprehensive benchmark configuration
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    k: usize,
    m_values: Vec<usize>,
    ef_construction_values: Vec<usize>,
    ef_search_values: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 100_000,
            num_queries: 100,
            dim: 128,
            k: 100,
            m_values: vec![8, 16, 32, 64],
            ef_construction_values: vec![100, 200, 400, 600],
            ef_search_values: vec![32, 64, 128, 256, 512],
        }
    }
}

/// Generate random dataset with Gaussian distribution
fn generate_gaussian_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        // Box-Muller transform for Gaussian distribution
        let u1 = rng.gen::<f32>();
        let u2 = rng.gen::<f32>();
        let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(gaussian);
    }

    data
}

/// Compute ground truth for random dataset (brute-force)
fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist = l2_distance_squared(q, b);
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i32> = distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx as i32)
            .collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Benchmark HNSW with specific parameter combination
fn benchmark_hnsw_params(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> HnswParamResult {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()), // Optimal level multiplier
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Search with explicit ef parameter
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: ef_search, // Use nprobe to control ef during search
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);

    let recall_at_1 = average_recall_at_k(&all_results, ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, ground_truth, 100);

    HnswParamResult {
        m,
        ef_construction,
        ef_search,
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Print detailed parameter table
fn print_param_table(results: &[HnswParamResult]) {
    println!("\n{}", "=".repeat(140));
    println!("HNSW 参数敏感性分析结果");
    println!("{}", "=".repeat(140));
    println!(
        "| {:>4} | {:>12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>10} |",
        "M",
        "ef_construction",
        "ef_search",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "Config"
    );
    println!("{}", "=".repeat(140));

    for r in results {
        let config_str = format!("M{}-efC{}-efS{}", r.m, r.ef_construction, r.ef_search);
        println!(
            "| {:>4} | {:>12} | {:>10} | {:>10.2} | {:>10.2} | {:>8.0} | {:>8.3} | {:>8.3} | {:>8.3} | {:>10} |",
            r.m, r.ef_construction, r.ef_search, r.build_time_ms, r.search_time_ms, r.qps,
            r.recall_at_1, r.recall_at_10, r.recall_at_100, config_str
        );
    }
    println!("{}", "=".repeat(140));
}

/// Analyze parameter sensitivity
fn analyze_parameter_sensitivity(results: &[HnswParamResult]) -> Vec<ParamAnalysis> {
    let mut analyses = Vec::new();

    // Analyze M impact (average over other params)
    let m_values = vec![8, 16, 32, 64];
    let mut m_recall = Vec::new();
    let mut m_qps = Vec::new();
    let mut m_build = Vec::new();

    for &m in &m_values {
        let filtered: Vec<_> = results.iter().filter(|r| r.m == m).collect();
        if !filtered.is_empty() {
            let avg_recall =
                filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / filtered.len() as f64;
            let avg_qps = filtered.iter().map(|r| r.qps).sum::<f64>() / filtered.len() as f64;
            let avg_build =
                filtered.iter().map(|r| r.build_time_ms).sum::<f64>() / filtered.len() as f64;
            m_recall.push(avg_recall);
            m_qps.push(avg_qps);
            m_build.push(avg_build);
        }
    }

    analyses.push(ParamAnalysis {
        param_name: "M".to_string(),
        param_values: m_values,
        impact_on_recall: m_recall,
        impact_on_qps: m_qps,
        impact_on_build_time: m_build,
    });

    // Analyze ef_construction impact
    let ef_c_values = vec![100, 200, 400, 600];
    let mut ef_c_recall = Vec::new();
    let mut ef_c_qps = Vec::new();
    let mut ef_c_build = Vec::new();

    for &ef_c in &ef_c_values {
        let filtered: Vec<_> = results
            .iter()
            .filter(|r| r.ef_construction == ef_c)
            .collect();
        if !filtered.is_empty() {
            let avg_recall =
                filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / filtered.len() as f64;
            let avg_qps = filtered.iter().map(|r| r.qps).sum::<f64>() / filtered.len() as f64;
            let avg_build =
                filtered.iter().map(|r| r.build_time_ms).sum::<f64>() / filtered.len() as f64;
            ef_c_recall.push(avg_recall);
            ef_c_qps.push(avg_qps);
            ef_c_build.push(avg_build);
        }
    }

    analyses.push(ParamAnalysis {
        param_name: "ef_construction".to_string(),
        param_values: ef_c_values,
        impact_on_recall: ef_c_recall,
        impact_on_qps: ef_c_qps,
        impact_on_build_time: ef_c_build,
    });

    // Analyze ef_search impact
    let ef_s_values = vec![32, 64, 128, 256, 512];
    let mut ef_s_recall = Vec::new();
    let mut ef_s_qps = Vec::new();

    for &ef_s in &ef_s_values {
        let filtered: Vec<_> = results.iter().filter(|r| r.ef_search == ef_s).collect();
        if !filtered.is_empty() {
            let avg_recall =
                filtered.iter().map(|r| r.recall_at_10).sum::<f64>() / filtered.len() as f64;
            let avg_qps = filtered.iter().map(|r| r.qps).sum::<f64>() / filtered.len() as f64;
            ef_s_recall.push(avg_recall);
            ef_s_qps.push(avg_qps);
        }
    }

    analyses.push(ParamAnalysis {
        param_name: "ef_search".to_string(),
        param_values: ef_s_values,
        impact_on_recall: ef_s_recall,
        impact_on_qps: ef_s_qps,
        impact_on_build_time: vec![], // ef_search doesn't affect build time
    });

    analyses
}

/// Print analysis summary
fn print_analysis_summary(analyses: &[ParamAnalysis]) {
    println!("\n{}", "=".repeat(100));
    println!("参数敏感性分析总结");
    println!("{}", "=".repeat(100));

    for analysis in analyses {
        println!("\n📊 {} 的影响:", analysis.param_name);
        println!("  参数值：{:?}", analysis.param_values);

        if !analysis.impact_on_recall.is_empty() {
            let recall_range = analysis
                .impact_on_recall
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });
            println!(
                "  R@10 范围：{:.3} - {:.3} (Δ={:.3})",
                recall_range.0,
                recall_range.1,
                recall_range.1 - recall_range.0
            );
        }

        if !analysis.impact_on_qps.is_empty() {
            let qps_range = analysis
                .impact_on_qps
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });
            println!(
                "  QPS 范围：{:.0} - {:.0} (Δ={:.0})",
                qps_range.0,
                qps_range.1,
                qps_range.1 - qps_range.0
            );
        }

        if !analysis.impact_on_build_time.is_empty() {
            let build_range = analysis
                .impact_on_build_time
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });
            println!(
                "  构建时间范围：{:.2} - {:.2}ms (Δ={:.2}ms)",
                build_range.0,
                build_range.1,
                build_range.1 - build_range.0
            );
        }
    }
}

/// Find best configurations based on criteria
fn find_best_configurations(results: &[HnswParamResult]) {
    println!("\n{}", "=".repeat(100));
    println!("最佳配置分析");
    println!("{}", "=".repeat(100));

    // Find fastest config with R@10 > 0.95
    println!("\n🎯 R@10 > 0.95 的最快配置:");
    let mut high_recall_results: Vec<_> =
        results.iter().filter(|r| r.recall_at_10 > 0.95).collect();
    high_recall_results.sort_by(|a, b| a.search_time_ms.partial_cmp(&b.search_time_ms).unwrap());

    if let Some(best) = high_recall_results.first() {
        println!(
            "  ⭐ 最佳配置：M={}, ef_construction={}, ef_search={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "     QPS: {:.0}, R@10: {:.3}, 构建时间：{:.2}ms",
            best.qps, best.recall_at_10, best.build_time_ms
        );
    } else {
        println!("  ⚠️  无配置满足 R@10 > 0.95");
    }

    // Find highest recall with build time < 1s
    println!("\n🎯 构建时间 < 1s 的最高召回率配置:");
    let mut fast_build_results: Vec<_> = results
        .iter()
        .filter(|r| r.build_time_ms < 1000.0)
        .collect();
    fast_build_results.sort_by(|a, b| b.recall_at_10.partial_cmp(&a.recall_at_10).unwrap());

    if let Some(best) = fast_build_results.first() {
        println!(
            "  ⭐ 最佳配置：M={}, ef_construction={}, ef_search={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "     R@10: {:.3}, QPS: {:.0}, 构建时间：{:.2}ms",
            best.recall_at_10, best.qps, best.build_time_ms
        );
    } else {
        println!("  ⚠️  无配置满足构建时间 < 1s");
    }

    // Find best balance (highest R@10 * QPS product)
    println!("\n🎯 最佳性价比配置 (R@10 × QPS 最大化):");
    let mut balanced_results: Vec<_> = results.iter().collect();
    balanced_results.sort_by(|a, b| {
        let score_a = a.recall_at_10 * a.qps;
        let score_b = b.recall_at_10 * b.qps;
        score_b.partial_cmp(&score_a).unwrap()
    });

    if let Some(best) = balanced_results.first() {
        let score = best.recall_at_10 * best.qps;
        println!(
            "  ⭐ 最佳配置：M={}, ef_construction={}, ef_search={}",
            best.m, best.ef_construction, best.ef_search
        );
        println!(
            "     综合得分：{:.0} (R@10={:.3} × QPS={:.0})",
            score, best.recall_at_10, best.qps
        );
        println!(
            "     构建时间：{:.2}ms, 搜索时间：{:.2}ms",
            best.build_time_ms, best.search_time_ms
        );
    }
}

/// Provide parameter selection recommendations
fn provide_recommendations(results: &[HnswParamResult], analyses: &[ParamAnalysis]) {
    println!("\n{}", "=".repeat(100));
    println!("参数选择建议");
    println!("{}", "=".repeat(100));

    println!("\n💡 M (图连接度) 选择建议:");
    if let Some(m_analysis) = analyses.iter().find(|a| a.param_name == "M") {
        let best_m_idx = m_analysis
            .impact_on_recall
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_m = m_analysis.param_values[best_m_idx];
        println!("  - 推荐 M = {} (在测试范围内召回率最高)", best_m);
        println!("  - M 越大，图连接越好，召回率越高，但构建时间和内存占用也越大");
        println!("  - 对于 128 维数据，M=32 或 M=64 通常是很好的选择");
    }

    println!("\n💡 ef_construction (构建时搜索范围) 选择建议:");
    if let Some(ef_c_analysis) = analyses.iter().find(|a| a.param_name == "ef_construction") {
        let best_ef_c_idx = ef_c_analysis
            .impact_on_recall
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_ef_c = ef_c_analysis.param_values[best_ef_c_idx];
        println!(
            "  - 推荐 ef_construction = {} (在测试范围内召回率最高)",
            best_ef_c
        );
        println!("  - ef_construction 越大，构建质量越好，但构建时间线性增长");
        println!("  - 建议设置为 ef_search 的 2-4 倍以获得更好的图质量");
    }

    println!("\n💡 ef_search (搜索时探索范围) 选择建议:");
    if let Some(ef_s_analysis) = analyses.iter().find(|a| a.param_name == "ef_search") {
        // Find the point of diminishing returns
        let recalls = &ef_s_analysis.impact_on_recall;
        let values = &ef_s_analysis.param_values;

        if recalls.len() >= 2 {
            let mut best_idx = 0;
            let mut max_improvement = 0.0;

            for i in 1..recalls.len() {
                let improvement = recalls[i] - recalls[i - 1];
                let cost_increase = (values[i] - values[i - 1]) as f64 / values[i - 1] as f64;
                let efficiency = improvement / cost_increase.max(0.001);

                if efficiency > max_improvement {
                    max_improvement = efficiency;
                    best_idx = i;
                }
            }

            println!("  - 推荐 ef_search = {} (性价比最优)", values[best_idx]);
            println!("  - ef_search 越大，召回率越高，但 QPS 下降");
            println!("  - 实时搜索场景：ef_search=64-128 (平衡 QPS 和召回率)");
            println!("  - 离线批处理场景：ef_search=256-512 (追求最高召回率)");
        }
    }

    println!("\n💡 综合推荐配置:");
    println!("  - 高性能场景 (QPS 优先): M=16, ef_construction=200, ef_search=64");
    println!("  - 高召回率场景 (精度优先): M=64, ef_construction=600, ef_search=512");
    println!("  - 平衡场景 (推荐): M=32, ef_construction=400, ef_search=128");
}

/// Export results to JSON
fn export_to_json(
    results: &[HnswParamResult],
    analyses: &[ParamAnalysis],
    config: &BenchmarkConfig,
) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let json_obj = serde_json::json!({
        "benchmark_type": "HNSW Parameter Sensitivity Analysis",
        "timestamp": timestamp,
        "dataset": {
            "num_vectors": config.num_vectors,
            "num_queries": config.num_queries,
            "dimension": config.dim,
            "metric_type": "L2"
        },
        "parameter_ranges": {
            "M": config.m_values,
            "ef_construction": config.ef_construction_values,
            "ef_search": config.ef_search_values
        },
        "results": results,
        "analysis": analyses,
        "summary": {
            "total_configurations": results.len(),
            "best_recall_at_10": results.iter().map(|r| r.recall_at_10).fold(0.0f64, f64::max),
            "best_qps": results.iter().map(|r| r.qps).fold(0.0f64, f64::max),
            "min_build_time_ms": results.iter().map(|r| r.build_time_ms).fold(f64::INFINITY, f64::min),
        }
    });

    serde_json::to_string_pretty(&json_obj).unwrap_or_default()
}

/// Get output directory from environment or use default
fn get_output_dir() -> String {
    env::var("JSON_OUTPUT_DIR").unwrap_or_else(|_| {
        "/Users/ryan/.openclaw/workspace-builder/benchmark_results/".to_string()
    })
}

/// Ensure output directory exists
fn ensure_output_dir(dir: &str) -> std::io::Result<()> {
    if !Path::new(dir).exists() {
        fs::create_dir_all(dir)?;
        println!("📁 Created output directory: {}", dir);
    }
    Ok(())
}

/// Parse test mode from environment variables
fn parse_test_mode() -> BenchmarkConfig {
    // Check environment variables for test mode
    if env::var("QUICK").is_ok() {
        println!("⚡ Quick mode: 10K vectors, 20 queries");
        return BenchmarkConfig {
            num_vectors: 10_000,
            num_queries: 20,
            dim: 128,
            k: 100,
            m_values: vec![8, 16, 32, 64],
            ef_construction_values: vec![100, 200, 400, 600],
            ef_search_values: vec![32, 64, 128, 256, 512],
        };
    }

    println!("📊 Standard mode: 100K vectors, 100 queries");
    BenchmarkConfig::default()
}

#[test]
fn test_hnsw_parameter_sensitivity() {
    println!("\n{}", "=".repeat(100));
    println!("🚀 BENCH-025: HNSW 参数敏感性分析");
    println!("{}", "=".repeat(100));

    // Parse test mode
    let config = parse_test_mode();

    println!("\n📋 测试配置:");
    println!("  - 数据集大小：{} 向量", config.num_vectors);
    println!("  - 查询数量：{}", config.num_queries);
    println!("  - 维度：{}", config.dim);
    println!("  - M 测试范围：{:?}", config.m_values);
    println!(
        "  - ef_construction 测试范围：{:?}",
        config.ef_construction_values
    );
    println!("  - ef_search 测试范围：{:?}", config.ef_search_values);

    // Generate dataset
    println!("\n📊 生成高斯分布数据集...");
    let base = generate_gaussian_dataset(config.num_vectors, config.dim);
    let query = generate_gaussian_dataset(config.num_queries, config.dim);

    // Compute ground truth
    println!("🎯 计算 ground truth (brute-force)...");
    let gt_start = Instant::now();
    let ground_truth =
        compute_ground_truth(&base, &query, config.num_queries, config.dim, config.k);
    let gt_time = gt_start.elapsed().as_secs_f64();
    println!("  Ground truth 计算完成：{:.2}s", gt_time);

    // Run parameter sweep
    let total_configs =
        config.m_values.len() * config.ef_construction_values.len() * config.ef_search_values.len();
    println!("\n🔬 开始参数敏感性测试 (共 {} 种配置)...", total_configs);

    let mut results = Vec::new();
    let mut current_config = 0;

    for &m in &config.m_values {
        for &ef_c in &config.ef_construction_values {
            for &ef_s in &config.ef_search_values {
                current_config += 1;
                println!(
                    "\n[{}/{}] Testing M={}, ef_construction={}, ef_search={}",
                    current_config, total_configs, m, ef_c, ef_s
                );

                let result = benchmark_hnsw_params(
                    &base,
                    &query,
                    &ground_truth,
                    config.num_queries,
                    config.dim,
                    m,
                    ef_c,
                    ef_s,
                );

                println!(
                    "  Build: {:.2}ms, Search: {:.2}ms, QPS: {:.0}",
                    result.build_time_ms, result.search_time_ms, result.qps
                );
                println!(
                    "  R@1: {:.3}, R@10: {:.3}, R@100: {:.3}",
                    result.recall_at_1, result.recall_at_10, result.recall_at_100
                );

                results.push(result);
            }
        }
    }

    // Print detailed results table
    print_param_table(&results);

    // Analyze parameter sensitivity
    let analyses = analyze_parameter_sensitivity(&results);
    print_analysis_summary(&analyses);

    // Find best configurations
    find_best_configurations(&results);

    // Provide recommendations
    provide_recommendations(&results, &analyses);

    // Export to JSON
    let output_dir = get_output_dir();
    ensure_output_dir(&output_dir).expect("Failed to create output directory");

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let filename = format!("hnsw_param_sensitivity_{}.json", timestamp);
    let output_path = Path::new(&output_dir).join(&filename);

    let json_content = export_to_json(&results, &analyses, &config);
    fs::write(&output_path, &json_content).expect("Failed to write JSON file");

    println!("\n{}", "=".repeat(100));
    println!("✅ JSON 导出完成");
    println!("📄 输出文件：{}", output_path.display());
    println!("{}", "=".repeat(100));

    println!("\n📈 BENCH-025: HNSW 参数敏感性分析完成!");
}

#[test]
fn test_hnsw_params_quick() {
    // Quick test for CI/CD or rapid iteration
    env::set_var("QUICK", "1");
    test_hnsw_parameter_sensitivity();
}

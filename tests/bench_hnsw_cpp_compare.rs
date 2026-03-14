//! BENCH-021: HNSW Rust vs C++ 公平对比
//!
//! 目标：使用相同参数（M=16, ef_construction=200, ef_search=64）
//! 对比 Rust vs C++ 的 QPS/召回率，确认 -21% QPS 差距的真实原因
//!
//! 数据集：SIFT1M (前 100K 向量子集，加速测试)
//! 参数：M=16, ef_construction=200, ef_search=64
//!
//! 运行：cargo test --release --test bench_hnsw_cpp_compare -- --nocapture

#[cfg(feature = "long-tests")]
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
#[cfg(feature = "long-tests")]
use knowhere_rs::benchmark::average_recall_at_k;
#[cfg(feature = "long-tests")]
use knowhere_rs::dataset::{load_sift1m_complete, SiftDataset};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::HnswIndex;
#[cfg(feature = "long-tests")]
use knowhere_rs::MetricType;
use serde_json::Value;
#[cfg(feature = "long-tests")]
use std::env;
use std::fs;
#[cfg(feature = "long-tests")]
use std::time::Instant;

const HNSW_FINAL_VERDICT_PATH: &str = "benchmark_results/hnsw_p3_002_final_verdict.json";
const FINAL_PERFORMANCE_LEADERSHIP_PROOF_PATH: &str =
    "benchmark_results/final_performance_leadership_proof.json";

fn load_hnsw_final_verdict() -> Value {
    let content = fs::read_to_string(HNSW_FINAL_VERDICT_PATH)
        .expect("family verdict artifact must exist for the HNSW compare lane");
    serde_json::from_str(&content).expect("family verdict artifact must be valid JSON")
}

fn load_final_performance_leadership_proof() -> Value {
    let content = fs::read_to_string(FINAL_PERFORMANCE_LEADERSHIP_PROOF_PATH)
        .expect("final leadership proof artifact must exist for the HNSW compare lane");
    serde_json::from_str(&content).expect("final leadership proof artifact must be valid JSON")
}

fn find_family<'a>(artifact: &'a Value, family: &str) -> &'a Value {
    artifact["families"]
        .as_array()
        .expect("families must be an array")
        .iter()
        .find(|entry| entry["family"] == family)
        .unwrap_or_else(|| panic!("family entry for {family} must exist"))
}

fn assert_close(actual: &Value, expected: f64) {
    let actual = actual
        .as_f64()
        .expect("artifact metric fields must be numeric");
    let delta = (actual - expected).abs();
    assert!(
        delta < 1e-9,
        "expected {expected} but found {actual} (delta={delta})"
    );
}

#[test]
fn hnsw_compare_lane_blocks_leadership_claims_until_native_gap_closes() {
    let verdict = load_hnsw_final_verdict();

    assert_eq!(verdict["family"], "HNSW");
    assert_eq!(verdict["classification"], "functional-but-not-leading");
    assert_eq!(
        verdict["leadership_verdict"],
        "no_go_for_performance_leadership"
    );
    assert_eq!(verdict["leadership_claim_allowed"], false);
    assert_close(
        &verdict["evidence"]["rust_recall_at_10"],
        0.9879999999999989,
    );
    assert_close(&verdict["evidence"]["rust_qps"], 8564.172172815608);
    assert_close(
        &verdict["evidence"]["native_over_rust_qps_ratio"],
        1.228938511232638,
    );
    assert!(
        verdict["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("1.2x"),
        "summary must disclose the current throughput gap that blocks leadership claims"
    );
}

#[test]
fn final_performance_proof_artifact_records_the_unmet_completion_criterion() {
    let proof = load_final_performance_leadership_proof();
    let hnsw = find_family(&proof, "HNSW");
    let ivfpq = find_family(&proof, "IVF-PQ");
    let diskann = find_family(&proof, "DiskANN");

    assert_eq!(proof["task_id"], "FINAL-PERFORMANCE-LEADERSHIP-PROOF");
    assert_eq!(proof["authority_scope"], "remote_x86_only");
    assert_eq!(proof["criterion_met"], false);
    assert_eq!(
        proof["baseline_stop_go_source"],
        "benchmark_results/baseline_p3_001_stop_go_verdict.json"
    );
    assert_eq!(
        proof["final_core_path_classification_source"],
        "benchmark_results/final_core_path_classification.json"
    );

    assert_eq!(hnsw["classification"], "functional-but-not-leading");
    assert_eq!(
        hnsw["leadership_status"],
        "trusted_but_blocked_by_native_qps_gap"
    );
    assert_eq!(hnsw["leadership_claim_allowed"], false);
    assert_close(&hnsw["evidence"]["rust_recall_at_10"], 0.9879999999999989);
    assert_close(&hnsw["evidence"]["rust_qps"], 8564.172172815608);
    assert_close(
        &hnsw["evidence"]["native_over_rust_qps_ratio"],
        1.228938511232638,
    );

    assert_eq!(ivfpq["classification"], "no-go");
    assert_eq!(ivfpq["leadership_status"], "family_no_go");
    assert_eq!(ivfpq["leadership_claim_allowed"], false);

    assert_eq!(diskann["classification"], "constrained");
    assert_eq!(
        diskann["leadership_status"],
        "constrained_non_comparable_lane"
    );
    assert_eq!(diskann["leadership_claim_allowed"], false);
}

/// 从 SIFT1M 加载数据集（支持子集）
#[cfg(feature = "long-tests")]
fn load_dataset_with_subset(max_vectors: Option<usize>) -> Option<SiftDataset> {
    let path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());

    match load_sift1m_complete(&path) {
        Ok(ds) => {
            if let Some(n) = max_vectors {
                if n < ds.num_base() {
                    // 创建子集
                    let dim = ds.dim();
                    let base: Vec<f32> = ds.base.vectors()[..n * dim].to_vec();
                    let query = ds.query.vectors().to_vec();
                    let gt = ds.ground_truth.iter().take(n.min(10000)).cloned().collect();

                    println!("✓ Loaded SIFT1M subset from {}", path);
                    println!(
                        "  Base: {}/{} vectors, Query: {} vectors, Dim: {}",
                        n,
                        ds.num_base(),
                        ds.num_query(),
                        dim
                    );

                    // 重建 SiftDataset
                    use knowhere_rs::dataset::Dataset;
                    return Some(SiftDataset {
                        base: Dataset::from_vectors(base, dim),
                        query: Dataset::from_vectors(query, dim),
                        ground_truth: gt,
                    });
                }
            }
            println!("✓ Loaded SIFT1M from {}", path);
            println!(
                "  Base: {} vectors, Query: {} vectors, Dim: {}",
                ds.num_base(),
                ds.num_query(),
                ds.dim()
            );
            Some(ds)
        }
        Err(e) => {
            eprintln!("Failed to load SIFT1M: {}", e);
            None
        }
    }
}

/// 从 SIFT1M 加载数据集
#[cfg(feature = "long-tests")]
fn load_dataset() -> Option<SiftDataset> {
    load_dataset_with_subset(None)
}

/// HNSW Benchmark 结果
#[cfg(feature = "long-tests")]
struct HnswResult {
    build_time_ms: f64,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    _params: String,
}

/// 运行 HNSW benchmark
#[cfg(feature = "long-tests")]
fn run_hnsw_benchmark(
    dataset: &SiftDataset,
    num_queries: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    top_k: usize,
) -> HnswResult {
    let base = dataset.base.vectors();
    let query = dataset.query.vectors();
    let gt = &dataset.ground_truth;
    let dim = dataset.dim();

    // 配置
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ..Default::default()
        },
    };

    // 构建
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).expect("Failed to create HNSW");
    index.train(base).expect("Train failed");
    index.add(base, None).expect("Add failed");
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // 搜索
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe: ef_search,
            params: Some(format!(r#"{{"ef": {}}}"#, ef_search)),
            ..Default::default()
        };
        let result = index.search(q, &req).expect("Search failed");
        all_results.push(result.ids);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);

    // 召回率
    let gt_subset: Vec<_> = gt.iter().take(num_queries).cloned().collect();
    let r1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let r10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let r100 = average_recall_at_k(&all_results, &gt_subset, 100);

    HnswResult {
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        _params: format!("M={}, ef_c={}, ef_s={}", m, ef_construction, ef_search),
    }
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_vs_cpp_compare() {
    println!("\n=== BENCH-021: HNSW Rust vs C++ 公平对比 ===\n");

    let dataset = match load_dataset() {
        Some(ds) => ds,
        None => {
            println!("SKIP: SIFT1M dataset not found");
            return;
        }
    };

    let num_queries = 1000; // 使用 1000 查询
    let top_k = 100;

    // 测试配置组 (按 TASK_QUEUE 描述)
    let configs = vec![
        // 基准配置 (TASK_QUEUE 中提到的配置)
        (16, 200, 64, "基准配置"),
        // 召回率优先配置
        (16, 200, 128, "ef_search=128"),
        (16, 200, 256, "ef_search=256"),
        (16, 400, 256, "高召回配置"),
        // 高 M 配置
        (32, 200, 128, "M=32"),
        (32, 400, 400, "高 M + 高 ef"),
    ];

    let mut results = Vec::new();

    for (m, ef_c, ef_s, desc) in configs {
        println!(
            "\n--- 测试: {} (M={}, ef_c={}, ef_s={}) ---",
            desc, m, ef_c, ef_s
        );

        let result = run_hnsw_benchmark(&dataset, num_queries, m, ef_c, ef_s, top_k);

        println!("  Build: {:.2} ms", result.build_time_ms);
        println!(
            "  Search: {:.2} ms ({} queries)",
            result.search_time_ms, num_queries
        );
        println!("  QPS: {:.0}", result.qps);
        println!(
            "  R@1: {:.3}, R@10: {:.3}, R@100: {:.3}",
            result.recall_at_1, result.recall_at_10, result.recall_at_100
        );

        results.push((desc.to_string(), result));
    }

    // 输出对比表格
    println!("\n=== HNSW Rust Benchmark 结果 ===\n");
    println!("| 配置 | Build(ms) | Search(ms) | QPS | R@1 | R@10 | R@100 |");
    println!("|------|-----------|------------|-----|-----|------|-------|");
    for (desc, r) in &results {
        println!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.3} | {:.3} | {:.3} |",
            desc,
            r.build_time_ms,
            r.search_time_ms,
            r.qps,
            r.recall_at_1,
            r.recall_at_10,
            r.recall_at_100
        );
    }

    // C++ 对比参考值 (来自历史 benchmark 数据)
    println!("\n=== C++ knowhere 参考值 (历史数据) ===\n");
    println!("| 配置 | QPS | R@10 | 说明 |");
    println!("|------|-----|------|------|");
    println!("| M=16, ef_c=200, ef_s=64 | ~1050 | ~0.96 | 基准配置 |");
    println!("| M=16, ef_c=200, ef_s=128 | ~850 | ~0.97 | ef_search 提升 |");

    // 差距分析
    println!("\n=== 差距分析 ===\n");
    if let Some((_, base_result)) = results.first() {
        let cpp_qps = 1050.0; // C++ 参考值
        let cpp_r10 = 0.96;
        let qps_gap = (base_result.qps - cpp_qps) / cpp_qps * 100.0;
        let r10_gap = (base_result.recall_at_10 - cpp_r10) / cpp_r10 * 100.0;

        println!("Rust (M=16, ef_c=200, ef_s=64):");
        println!(
            "  QPS: {:.0} (vs C++ {:.0}, 差距 {:+.1}%)",
            base_result.qps, cpp_qps, qps_gap
        );
        println!(
            "  R@10: {:.3} (vs C++ {:.2}, 差距 {:+.1}%)",
            base_result.recall_at_10, cpp_r10, r10_gap
        );

        // 可能的差距原因
        println!("\n可能差距原因：");
        println!("1. 并行构建：Rust 版本是否使用并行构建？");
        println!("2. SIMD 优化：Rust 版本是否启用 AVX2/NEON？");
        println!("3. 内存分配：Rust HashMap vs C++ std::vector");
        println!("4. 邻居遍历：Rust 版本的邻居访问效率");
    }
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_quick_compare() {
    // 快速测试 - 使用完整数据集，测试多个 ef_search 配置
    println!("\n=== HNSW Quick Compare (SIFT1M) ===\n");

    // 加载完整数据集
    let dataset = match load_dataset() {
        Some(ds) => ds,
        None => {
            println!("SKIP: SIFT1M dataset not found");
            return;
        }
    };

    let base = dataset.base.vectors();
    let query = dataset.query.vectors();
    let gt = &dataset.ground_truth;
    let dim = dataset.dim();
    let num_queries = 100;

    // 构建一次索引，测试多个 ef_search
    let m = 16;
    let ef_construction = 200;

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(800), // 高 ef_search 用于构建
            ..Default::default()
        },
    };

    println!("Building HNSW (M={}, ef_c={})...", m, ef_construction);
    let build_start = std::time::Instant::now();
    let mut index = HnswIndex::new(&config).expect("Failed to create HNSW");
    index.train(base).expect("Train failed");
    index.add(base, None).expect("Add failed");
    let build_time = build_start.elapsed().as_secs_f64();
    println!("Build time: {:.1}s\n", build_time);

    // 测试不同 ef_search
    println!("| ef_search | Search(ms) | QPS | R@1 | R@10 |");
    println!("|-----------|------------|-----|-----|------|");

    for ef_search in [64, 128, 256, 400, 800] {
        let search_start = std::time::Instant::now();
        let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

        for i in 0..num_queries {
            let q = &query[i * dim..(i + 1) * dim];
            let req = SearchRequest {
                top_k: 100,
                nprobe: ef_search,
                params: Some(format!(r#"{{"ef": {}}}"#, ef_search)),
                ..Default::default()
            };
            let result = index.search(q, &req).expect("Search failed");
            all_results.push(result.ids);
        }

        let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
        let qps = num_queries as f64 / (search_time / 1000.0);

        // 计算召回率
        let gt_subset: Vec<_> = gt.iter().take(num_queries).cloned().collect();
        let r1 = average_recall_at_k(&all_results, &gt_subset, 1);
        let r10 = average_recall_at_k(&all_results, &gt_subset, 10);

        println!(
            "| {} | {:.1} | {:.0} | {:.3} | {:.3} |",
            ef_search, search_time, qps, r1, r10
        );
    }

    // C++ 参考
    println!("\n=== C++ knowhere 参考 ===");
    println!("| ef_search | QPS | R@10 |");
    println!("|-----------|-----|------|");
    println!("| 64 | ~1050 | ~0.96 |");
    println!("| 128 | ~850 | ~0.97 |");
}

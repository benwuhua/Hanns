#![cfg(feature = "long-tests")]
//! 量化索引精度损失分析 Benchmark
//!
//! BENCH-033: 量化索引精度损失分析 - PQ/SQ 的召回率 vs 压缩比权衡
//!
//! 测试目标:
//! 1. 对比不同量化方法 (PQ, SQ8, RaBitQ) 的召回率
//! 2. 分析压缩比与精度损失的关系
//! 3. 识别不同场景下的最佳量化配置

use hanns::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::benchmark::recall::recall_at_k;
use hanns::faiss::{IvfPqIndex, IvfSq8Index, MemIndex as FlatIndex};
use rand::Rng;
use std::time::Instant;

/// 生成随机测试数据集
fn generate_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..num_vectors * dim)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0) // [-1, 1] 范围
        .collect()
}

/// 运行 Flat benchmark
fn run_flat_bench(
    train_data: &[f32],
    base_data: &[f32],
    query_data: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    top_k: usize,
) -> BenchmarkResult {
    println!("\n  测试 Flat (基准线)...");

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut index = FlatIndex::new(&config).expect("Flat index creation failed");

    // 训练 (Flat 不需要真正训练)
    let _ = index.train(train_data);

    // 添加向量
    let build_start = Instant::now();
    index.add(base_data, None).expect("Flat add failed");
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // 搜索
    let num_queries = query_data.len() / dim;
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = index.search(query, &req).expect("Flat search failed");
        all_results.push(result.ids);
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // 计算召回率
    let (r1, r10, r100) = calculate_recall(&all_results, ground_truth, top_k);

    BenchmarkResult {
        name: "Flat",
        compression_ratio: 1.0,
        build_time_ms: build_time,
        search_time_ms: search_time / num_queries as f64,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        error: None,
    }
}

/// 运行 SQ8 benchmark
#[allow(clippy::too_many_arguments)]
fn run_sq8_bench(
    train_data: &[f32],
    base_data: &[f32],
    query_data: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    top_k: usize,
    nlist: usize,
    nprobe: usize,
) -> BenchmarkResult {
    println!(
        "\n  测试 SQ8 (4x 压缩，nlist={}, nprobe={})...",
        nlist, nprobe
    );

    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    let mut index = IvfSq8Index::new(&config).expect("SQ8 index creation failed");

    // 训练
    let train_start = Instant::now();
    index.train(train_data).expect("SQ8 train failed");
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    // 添加向量
    let build_start = Instant::now();
    index.add(base_data, None).expect("SQ8 add failed");
    let build_time = train_time + build_start.elapsed().as_secs_f64() * 1000.0;

    // 搜索
    let num_queries = query_data.len() / dim;
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).expect("SQ8 search failed");
        all_results.push(result.ids);
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // 计算召回率
    let (r1, r10, r100) = calculate_recall(&all_results, ground_truth, top_k);

    BenchmarkResult {
        name: "SQ8",
        compression_ratio: 4.0,
        build_time_ms: build_time,
        search_time_ms: search_time / num_queries as f64,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        error: None,
    }
}

/// 运行 PQ benchmark
#[allow(clippy::too_many_arguments)]
fn run_pq_bench(
    train_data: &[f32],
    base_data: &[f32],
    query_data: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    top_k: usize,
    nlist: usize,
    nprobe: usize,
    pq_m: usize,
) -> BenchmarkResult {
    let compression_ratio = (dim / pq_m) as f32; // 假设 nbits=8
    println!(
        "\n  测试 PQ m={} ({:.0}x 压缩，nlist={}, nprobe={})...",
        pq_m, compression_ratio, nlist, nprobe
    );

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(pq_m),
            nbits_per_idx: Some(8),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config).expect("PQ index creation failed");

    // 训练
    let train_start = Instant::now();
    index.train(train_data).expect("PQ train failed");
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    // 添加向量
    let build_start = Instant::now();
    index.add(base_data, None).expect("PQ add failed");
    let build_time = train_time + build_start.elapsed().as_secs_f64() * 1000.0;

    // 搜索
    let num_queries = query_data.len() / dim;
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe,
            ..Default::default()
        };
        match index.search(query, &req) {
            Ok(result) => all_results.push(result.ids),
            Err(e) => {
                return BenchmarkResult {
                    name: "PQ (error)",
                    compression_ratio,
                    build_time_ms: build_time,
                    search_time_ms: 0.0,
                    recall_at_1: 0.0,
                    recall_at_10: 0.0,
                    recall_at_100: 0.0,
                    error: Some(format!("搜索失败：{}", e)),
                };
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // 计算召回率
    let (r1, r10, r100) = calculate_recall(&all_results, ground_truth, top_k);

    BenchmarkResult {
        name: match pq_m {
            16 => "PQ m=16",
            8 => "PQ m=8",
            4 => "PQ m=4",
            _ => "PQ",
        },
        compression_ratio,
        build_time_ms: build_time,
        search_time_ms: search_time / num_queries as f64,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        error: None,
    }
}

/// 计算召回率
fn calculate_recall(
    results: &[Vec<i64>],
    ground_truth: &[Vec<i32>],
    top_k: usize,
) -> (f32, f32, f32) {
    let mut r1_sum = 0.0;
    let mut r10_sum = 0.0;
    let mut r100_sum = 0.0;

    for (result, gt) in results.iter().zip(ground_truth.iter()) {
        r1_sum += recall_at_k(result, gt, 1) as f32;
        r10_sum += recall_at_k(result, gt, 10.min(top_k)) as f32;
        r100_sum += recall_at_k(result, gt, 100.min(top_k)) as f32;
    }

    let num_queries = results.len() as f32;
    (
        r1_sum / num_queries,
        r10_sum / num_queries,
        r100_sum / num_queries,
    )
}

/// Benchmark 结果
struct BenchmarkResult {
    name: &'static str,
    compression_ratio: f32,
    build_time_ms: f64,
    search_time_ms: f64,
    recall_at_1: f32,
    recall_at_10: f32,
    recall_at_100: f32,
    error: Option<String>,
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn bench_quantization_accuracy() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BENCH-033: 量化索引精度损失分析                              ║");
    println!("║  Quantization Index Accuracy Loss Analysis                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 测试参数
    let dim = 128;
    let num_train = 10_000;
    let num_base = 50_000; // 减小规模以加快测试
    let num_queries = 500;
    let top_k = 100;
    let nlist = 256;
    let nprobe = 32;

    println!("测试配置:");
    println!("  维度：{}", dim);
    println!("  训练集：{} 向量", num_train);
    println!("  基数据集：{} 向量", num_base);
    println!("  查询集：{} 向量", num_queries);
    println!("  Top-K: {}", top_k);
    println!("  IVF nlist: {}, nprobe: {}", nlist, nprobe);
    println!();

    // 生成数据集
    println!("生成数据集...");
    let train_data = generate_dataset(num_train, dim);
    let base_data = generate_dataset(num_base, dim);
    let query_data = generate_dataset(num_queries, dim);

    // 使用 Flat index 计算 ground truth
    println!("计算 Ground Truth (Flat index)...");
    let flat_config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams::default(),
    };
    let mut flat_index = FlatIndex::new(&flat_config).expect("Flat index failed");
    let _ = flat_index.train(&train_data);
    flat_index.add(&base_data, None).expect("Flat add failed");

    let mut ground_truth: Vec<Vec<i32>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = flat_index.search(query, &req).expect("Flat search failed");
        ground_truth.push(result.ids.iter().map(|&id| id as i32).collect());
    }
    println!("Ground Truth 计算完成 (R@100=1.0 基准)\n");

    // 运行 benchmark
    let mut results = Vec::new();

    // Flat (基准)
    let flat_result = run_flat_bench(
        &train_data,
        &base_data,
        &query_data,
        &ground_truth,
        dim,
        top_k,
    );
    results.push(flat_result);

    // SQ8
    let sq8_result = run_sq8_bench(
        &train_data,
        &base_data,
        &query_data,
        &ground_truth,
        dim,
        top_k,
        nlist,
        nprobe,
    );
    results.push(sq8_result);

    // PQ with different m values
    for pq_m in [16, 8, 4] {
        if dim % pq_m == 0 {
            let pq_result = run_pq_bench(
                &train_data,
                &base_data,
                &query_data,
                &ground_truth,
                dim,
                top_k,
                nlist,
                nprobe,
                pq_m,
            );
            results.push(pq_result);
        }
    }

    // 打印汇总报告
    println!("\n{}", "═".repeat(70));
    println!("📊 量化索引精度损失分析报告");
    println!("{}\n", "═".repeat(70));

    // 召回率对比表
    println!("┌────────────────────┬────────────┬──────────┬──────────┬──────────┐");
    println!("│ 量化方法           │ 压缩比     │  R@1     │  R@10    │  R@100   │");
    println!("├────────────────────┼────────────┼──────────┼──────────┼──────────┤");
    for result in &results {
        if let Some(error) = result.error.as_deref() {
            println!(
                "│ {:<18} │ {:>8.1}x │ {:>17} │",
                result.name,
                result.compression_ratio,
                format!("❌ {}", error),
            );
        } else {
            println!(
                "│ {:<18} │ {:>8.1}x │ {:>8.3} │ {:>8.3} │ {:>8.3} │",
                result.name,
                result.compression_ratio,
                result.recall_at_1,
                result.recall_at_10,
                result.recall_at_100,
            );
        }
    }
    println!("└────────────────────┴────────────┴──────────┴──────────┴──────────┘");

    // 性能对比表
    println!("\n┌────────────────────┬────────────┬────────────┬────────────┐");
    println!("│ 量化方法           │ 压缩比     │ 构建 (ms)  │ 搜索 (ms)  │");
    println!("├────────────────────┼────────────┼────────────┼────────────┤");
    for result in &results {
        if result.error.is_none() {
            println!(
                "│ {:<18} │ {:>8.1}x │ {:>10.1} │ {:>10.2} │",
                result.name, result.compression_ratio, result.build_time_ms, result.search_time_ms,
            );
        }
    }
    println!("└────────────────────┴────────────┴────────────┴────────────┘");

    // 精度损失分析
    println!("\n📈 精度损失分析 (相对于 Flat):");
    println!();
    for result in &results {
        if result.error.is_none() && result.name != "Flat" {
            let recall_loss_r10 = (1.0 - result.recall_at_10 as f64) * 100.0;
            let recall_loss_r100 = (1.0 - result.recall_at_100 as f64) * 100.0;
            println!(
                "  {}: R@10 损失 {:.1}%, R@100 损失 {:.1}%",
                result.name, recall_loss_r10, recall_loss_r100
            );
        }
    }

    // 推荐配置
    println!("\n💡 推荐配置:");
    println!();

    // 找到 R@10 > 0.9 的最佳压缩比
    let high_accuracy: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|r| r.error.is_none() && r.recall_at_10 > 0.90)
        .collect();

    if let Some(best) = high_accuracy.iter().max_by(|a, b| {
        a.compression_ratio
            .partial_cmp(&b.compression_ratio)
            .unwrap()
    }) {
        println!("  高准确度场景 (R@10 > 90%): {}", best.name);
        println!("    - 压缩比：{:.1}x", best.compression_ratio);
        println!("    - R@10: {:.3}", best.recall_at_10);
        println!("    - 适用：搜索质量优先的场景");
    }

    // 找到平衡配置
    let balanced: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|r| r.error.is_none() && r.recall_at_10 > 0.70 && r.recall_at_10 < 0.90)
        .collect();

    if let Some(best) = balanced.iter().max_by(|a, b| {
        a.compression_ratio
            .partial_cmp(&b.compression_ratio)
            .unwrap()
    }) {
        println!("\n  平衡场景 (R@10 70-90%): {}", best.name);
        println!("    - 压缩比：{:.1}x", best.compression_ratio);
        println!("    - R@10: {:.3}", best.recall_at_10);
        println!("    - 适用：性能和精度平衡的场景");
    }

    // 找到最高压缩比
    let high_compression: Vec<&BenchmarkResult> = results
        .iter()
        .filter(|r| r.error.is_none() && r.compression_ratio >= 16.0)
        .collect();

    if let Some(best) = high_compression
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
    {
        println!("\n  高压缩场景 (压缩比 >= 16x): {}", best.name);
        println!("    - 压缩比：{:.1}x", best.compression_ratio);
        println!("    - R@10: {:.3}", best.recall_at_10);
        println!("    - 适用：内存受限的场景");
    }

    println!("\n{}", "═".repeat(70));
    println!("Benchmark 完成!");
    println!("{}\n", "═".repeat(70));
}

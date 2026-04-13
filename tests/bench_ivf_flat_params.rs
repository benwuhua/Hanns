#![cfg(feature = "long-tests")]
//! OPT-018: IVF-Flat 参数调优 Benchmark
//!
//! 问题：IVF-Flat Fast 版本召回率下降严重：R@100 从 1.0 降至 0.347
//! 目标：分析参数差异，调整配置恢复召回率（目标 R@100 >= 0.95）
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_ivf_flat_params -- --nocapture
//! ```

mod common;
use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::faiss::IvfFlatIndex;
use hanns::IndexType;
use hanns::MetricType;
use rand::Rng;
use std::time::Instant;

/// 生成带聚类结构的测试数据
fn generate_clustered_vectors(n: usize, dim: usize, num_clusters: usize) -> (Vec<f32>, Vec<i32>) {
    let mut rng = rand::thread_rng();
    let mut vectors = vec![0.0f32; n * dim];
    let mut labels = vec![0i32; n];

    // 生成簇中心 - 使用更大的间隔确保簇间分离
    let mut centroids = vec![0.0f32; num_clusters * dim];
    for c in 0..num_clusters {
        for d in 0..dim {
            centroids[c * dim + d] = (c as f32) * 5.0 + rng.gen_range(-1.0..1.0);
        }
    }

    // 生成簇内向量
    let vectors_per_cluster = n / num_clusters;
    for c in 0..num_clusters {
        for i in 0..vectors_per_cluster {
            let idx = c * vectors_per_cluster + i;
            labels[idx] = c as i32;

            for d in 0..dim {
                vectors[idx * dim + d] = centroids[c * dim + d] + rng.gen_range(-0.3..0.3);
            }
        }
    }

    (vectors, labels)
}

/// 生成查询向量 (从基础向量中采样并添加噪声)
fn generate_queries_from_base(
    base_vectors: &[f32],
    base_labels: &[i32],
    num_queries: usize,
) -> (Vec<f32>, Vec<i32>) {
    let mut rng = rand::thread_rng();
    let dim = base_vectors.len() / base_labels.len();
    let mut queries = vec![0.0f32; num_queries * dim];
    let mut query_labels = vec![0i32; num_queries];

    for i in 0..num_queries {
        let base_idx = rng.gen_range(0..base_labels.len());
        query_labels[i] = base_labels[base_idx];

        for d in 0..dim {
            queries[i * dim + d] = base_vectors[base_idx * dim + d] + rng.gen_range(-0.1..0.1);
        }
    }

    (queries, query_labels)
}

/// 计算 ground truth (基于暴力搜索)
fn compute_ground_truth(base: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<Vec<i64>> {
    let num_base = base.len() / dim;
    let num_query = query.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_query);

    for i in 0..num_query {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(i64, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist = common::l2_distance_squared(q, b);
            distances.push((j as i64, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i64> = distances.into_iter().take(k).map(|(id, _)| id).collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

/// 计算召回率
fn calculate_recall(results: &[Vec<i64>], ground_truth: &[Vec<i64>], k: usize) -> f64 {
    let mut total_recall = 0.0;

    for (result, gt) in results.iter().zip(ground_truth.iter()) {
        let result_set: std::collections::HashSet<i64> = result.iter().take(k).cloned().collect();
        let gt_set: std::collections::HashSet<i64> = gt.iter().take(k).cloned().collect();

        let intersection = result_set.intersection(&gt_set).count();
        total_recall += intersection as f64 / k.min(gt_set.len()) as f64;
    }

    total_recall / results.len() as f64
}

/// Benchmark 结果
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    _nlist: usize,
    _nprobe: usize,
    _kmeans_type: String,
    train_time_ms: f64,
    build_time_ms: f64,
    _search_time_ms: f64,
    qps: f64,
    recall_at_10: f64,
    recall_at_50: f64,
    recall_at_100: f64,
}

/// 测试 IVF-Flat 配置
#[allow(clippy::too_many_arguments)]
fn benchmark_ivf_flat_config(
    name: &str,
    vectors: &[f32],
    queries: &[f32],
    ground_truth: &[Vec<i64>],
    dim: usize,
    nlist: usize,
    nprobe: usize,
    params: IndexParams,
) -> BenchmarkResult {
    let kmeans_type = if params.use_elkan.unwrap_or(false) {
        "Elkan".to_string()
    } else if params.use_kmeans_pp.unwrap_or(false) {
        "KMeans++".to_string()
    } else if params.use_mini_batch.unwrap_or(false) {
        "MiniBatch".to_string()
    } else {
        "Standard".to_string()
    };

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params,
    };

    // Train
    let train_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(vectors).unwrap();
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    // Build (add vectors)
    let build_start = Instant::now();
    index.add(vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Search
    let num_queries = queries.len() / dim;
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            filter: None,
            params: None,
            radius: None,
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall
    let recall_at_10 = calculate_recall(&all_results, ground_truth, 10);
    let recall_at_50 = calculate_recall(&all_results, ground_truth, 50);
    let recall_at_100 = calculate_recall(&all_results, ground_truth, 100);

    BenchmarkResult {
        name: name.to_string(),
        _nlist: nlist,
        _nprobe: nprobe,
        _kmeans_type: kmeans_type,
        train_time_ms: train_time,
        build_time_ms: build_time,
        _search_time_ms: search_time,
        qps: num_queries as f64 / (search_time / 1000.0),
        recall_at_10,
        recall_at_50,
        recall_at_100,
    }
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_ivf_flat_fast_vs_standard() {
    println!("\n🔬 OPT-018: IVF-Flat Fast vs Standard 召回率对比");
    println!("==================================================\n");

    let n = 10_000;
    let dim = 128;
    let num_clusters = 100;
    let nlist = 100;
    let nprobe = 10;

    println!("📊 数据集配置:");
    println!("   - 向量数：{}", n);
    println!("   - 维度：{}", dim);
    println!("   - 簇数：{}", num_clusters);
    println!("   - nlist: {}", nlist);
    println!("   - nprobe: {}", nprobe);
    println!();

    let (vectors, _labels) = generate_clustered_vectors(n, dim, num_clusters);
    let (queries, _query_labels) = generate_queries_from_base(&vectors, &_labels, 100);
    let ground_truth = common::compute_ground_truth(&vectors, &queries, dim, 100);

    // 测试 1: 标准 IVF-Flat (基准)
    println!("📍 测试 1: 标准 IVF-Flat (基准)");
    let result_std = benchmark_ivf_flat_config(
        "Standard IVF-Flat",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf(nlist, nprobe),
    );
    println!(
        "   Train: {:.2}ms, Build: {:.2}ms, QPS: {:.0}",
        result_std.train_time_ms, result_std.build_time_ms, result_std.qps
    );
    println!(
        "   Recall@10/50/100: {:.3}/{:.3}/{:.3}",
        result_std.recall_at_10, result_std.recall_at_50, result_std.recall_at_100
    );
    println!();

    // 测试 2: Fast IVF-Flat (修复后版本)
    println!("📍 测试 2: Fast IVF-Flat (修复后版本)");
    let result_fast = benchmark_ivf_flat_config(
        "Fast IVF-Flat",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf_flat_fast(nlist, nprobe),
    );
    println!(
        "   Train: {:.2}ms, Build: {:.2}ms, QPS: {:.0}",
        result_fast.train_time_ms, result_fast.build_time_ms, result_fast.qps
    );
    println!(
        "   Recall@10/50/100: {:.3}/{:.3}/{:.3}",
        result_fast.recall_at_10, result_fast.recall_at_50, result_fast.recall_at_100
    );
    println!();

    // 测试 3: Elkan K-Means (问题版本)
    println!("📍 测试 3: Elkan K-Means (问题版本 - 原 Fast 配置)");
    let result_elkan = benchmark_ivf_flat_config(
        "Elkan K-Means",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf_elkan(nlist, nprobe, 5, 1e-3, 42),
    );
    println!(
        "   Train: {:.2}ms, Build: {:.2}ms, QPS: {:.0}",
        result_elkan.train_time_ms, result_elkan.build_time_ms, result_elkan.qps
    );
    println!(
        "   Recall@10/50/100: {:.3}/{:.3}/{:.3}",
        result_elkan.recall_at_10, result_elkan.recall_at_50, result_elkan.recall_at_100
    );
    println!();

    // 对比分析
    println!("\n📊 对比分析:");
    println!("==================================================");
    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "配置", "R@10", "R@50", "R@100"
    );
    println!("--------------------------------------------------");
    println!(
        "{:<20} {:>10.3} {:>10.3} {:>10.3}",
        result_std.name, result_std.recall_at_10, result_std.recall_at_50, result_std.recall_at_100
    );
    println!(
        "{:<20} {:>10.3} {:>10.3} {:>10.3}",
        result_fast.name,
        result_fast.recall_at_10,
        result_fast.recall_at_50,
        result_fast.recall_at_100
    );
    println!(
        "{:<20} {:>10.3} {:>10.3} {:>10.3}",
        result_elkan.name,
        result_elkan.recall_at_10,
        result_elkan.recall_at_50,
        result_elkan.recall_at_100
    );
    println!();

    // 验证修复效果
    let recall_drop = result_std.recall_at_100 - result_elkan.recall_at_100;
    let recall_recovery = result_std.recall_at_100 - result_fast.recall_at_100;

    println!("📈 召回率分析:");
    println!(
        "   - Elkan 版本召回率下降：{:.3} (从 {:.3} 降至 {:.3})",
        recall_drop, result_std.recall_at_100, result_elkan.recall_at_100
    );
    println!("   - 修复后召回率差距：{:.3}", recall_recovery);

    if result_fast.recall_at_100 >= 0.95 {
        println!("   ✅ 修复成功！Fast 版本 R@100 >= 0.95");
    } else if recall_recovery < 0.05 {
        println!("   ✅ 修复有效！召回率差距 < 0.05");
    } else {
        println!("   ⚠️  需要进一步优化");
    }
    println!();
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_kmeans_algorithms_comparison() {
    println!("\n🔬 OPT-018: K-Means 算法对比");
    println!("==================================================\n");

    let n = 10_000;
    let dim = 128;
    let num_clusters = 100;
    let nlist = 100;
    let nprobe = 10;

    let (vectors, _labels) = generate_clustered_vectors(n, dim, num_clusters);
    let (queries, _query_labels) = generate_queries_from_base(&vectors, &_labels, 100);
    let ground_truth = common::compute_ground_truth(&vectors, &queries, dim, 100);

    // 测试不同的 k-means 算法
    println!("📍 测试 1: 标准 K-Means");
    let result_std = benchmark_ivf_flat_config(
        "Standard K-Means",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf(nlist, nprobe),
    );
    println!(
        "   Train: {:.2}ms, R@100: {:.3}",
        result_std.train_time_ms, result_std.recall_at_100
    );

    println!("\n📍 测试 2: K-Means++");
    let result_pp = benchmark_ivf_flat_config(
        "K-Means++",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf_pp(nlist, nprobe, 25, 1e-4, 42),
    );
    println!(
        "   Train: {:.2}ms, R@100: {:.3}",
        result_pp.train_time_ms, result_pp.recall_at_100
    );

    println!("\n📍 测试 3: Elkan K-Means (快速模式)");
    let result_elkan = benchmark_ivf_flat_config(
        "Elkan K-Means",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf_elkan(nlist, nprobe, 5, 1e-3, 42),
    );
    println!(
        "   Train: {:.2}ms, R@100: {:.3}",
        result_elkan.train_time_ms, result_elkan.recall_at_100
    );

    println!("\n📍 测试 4: Mini-Batch K-Means");
    let result_mb = benchmark_ivf_flat_config(
        "Mini-Batch K-Means",
        &vectors,
        &queries,
        &ground_truth,
        dim,
        nlist,
        nprobe,
        IndexParams::ivf_mini_batch(nlist, nprobe, 5000, 100, 1e-4),
    );
    println!(
        "   Train: {:.2}ms, R@100: {:.3}",
        result_mb.train_time_ms, result_mb.recall_at_100
    );

    println!("\n📊 K-Means 算法对比:");
    println!("==================================================");
    println!("{:<20} {:>12} {:>10}", "算法", "Train(ms)", "R@100");
    println!("--------------------------------------------------");
    println!(
        "{:<20} {:>12.2} {:>10.3}",
        result_std.name, result_std.train_time_ms, result_std.recall_at_100
    );
    println!(
        "{:<20} {:>12.2} {:>10.3}",
        result_pp.name, result_pp.train_time_ms, result_pp.recall_at_100
    );
    println!(
        "{:<20} {:>12.2} {:>10.3}",
        result_elkan.name, result_elkan.train_time_ms, result_elkan.recall_at_100
    );
    println!(
        "{:<20} {:>12.2} {:>10.3}",
        result_mb.name, result_mb.train_time_ms, result_mb.recall_at_100
    );
    println!();

    println!("📝 推荐:");
    if result_std.recall_at_100 >= 0.95 {
        println!("   ✅ 标准 K-Means: 召回率最高，适合对精度要求高的场景");
    }
    if result_elkan.recall_at_100 < 0.90 {
        println!("   ⚠️  Elkan (快速模式): 召回率较低，不推荐用于 Fast 版本");
    }
    println!();
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_opt018_summary() {
    println!("\n📊 OPT-018: IVF-Flat 参数调优总结");
    println!("==================================================\n");

    println!("问题描述:");
    println!("  IVF-Flat Fast 版本召回率下降严重：R@100 从 1.0 降至 0.347");
    println!();

    println!("根本原因:");
    println!("  - 原 Fast 版本使用 Elkan K-Means (max_iter=5, tol=1e-3)");
    println!("  - 迭代次数过少导致聚类质量差");
    println!("  - 聚类中心不准确导致向量分配错误");
    println!();

    println!("修复方案:");
    println!("  - 修改 IndexParams::ivf_flat_fast() 不使用 Elkan K-Means");
    println!("  - 使用标准 K-Means 默认配置");
    println!("  - 速度优化来自并行化的 add() 方法，而非降低聚类质量");
    println!();

    println!("预期效果:");
    println!("  - 召回率恢复：R@100 >= 0.95");
    println!("  - 构建时间优化：保持 <2s (从 5.2s 减少 60%+)");
    println!();

    println!("运行完整测试:");
    println!("  cargo test --release --test bench_ivf_flat_params -- --nocapture");
    println!();
}

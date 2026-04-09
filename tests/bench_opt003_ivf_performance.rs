#![cfg(feature = "long-tests")]
//! OPT-003 Phase 3: IVF 内存布局优化性能基准测试
//!
//! 验证 HashMap → Vec 优化后的 QPS 提升
//! 目标: IVF-Flat QPS 从 100-250 → 2,500+ (50% C++)
//!       IVF-PQ QPS 从 160-400 → 4,000+ (50% C++)
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_opt003_ivf_performance -- --nocapture
//! ```

use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::faiss::{IvfFlatIndex, IvfPqIndex};
use hanns::IndexType;
use hanns::MetricType;
use std::time::Instant;

/// 生成随机测试数据
fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// 计算 Recall@k
fn calculate_recall(predictions: &[i64], ground_truth: &[i64], k: usize) -> f32 {
    let pred_set: std::collections::HashSet<i64> = predictions.iter().copied().take(k).collect();
    let gt_set: std::collections::HashSet<i64> = ground_truth.iter().copied().take(k).collect();

    let intersection = pred_set.intersection(&gt_set).count();
    intersection as f32 / k.min(gt_set.len()) as f32
}

/// 生成 ground truth (brute force)
fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i64>> {
    let n = base.len() / dim;
    let q = queries.len() / dim;

    (0..q)
        .map(|i| {
            let query = &queries[i * dim..(i + 1) * dim];
            let mut distances: Vec<(f32, i64)> = (0..n)
                .map(|j| {
                    let vec = &base[j * dim..(j + 1) * dim];
                    let dist: f32 = query
                        .iter()
                        .zip(vec.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (dist, j as i64)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances.iter().take(k).map(|(_, id)| *id).collect()
        })
        .collect()
}

/// IVF-Flat 性能测试
#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_ivf_flat_performance() {
    println!("\n🚀 OPT-003 Phase 3: IVF-Flat 性能基准测试");
    println!("==============================================");

    let dim = 128;
    let n_base = 50_000; // 50K 向量
    let n_queries = 100;
    let k = 10;
    let nlist = 100;
    let nprobe = 16;

    println!("\n📊 配置:");
    println!("   Base vectors: {}", n_base);
    println!("   Queries: {}", n_queries);
    println!("   Dimension: {}", dim);
    println!("   nlist: {}, nprobe: {}", nlist, nprobe);
    println!("   k: {}", k);

    // 生成数据
    println!("\n⏳ 生成数据...");
    let base = generate_random_vectors(n_base, dim);
    let queries = generate_random_vectors(n_queries, dim);

    // 计算 ground truth
    println!("⏳ 计算 ground truth...");
    let start = Instant::now();
    let ground_truth = compute_ground_truth(&base, &queries, dim, k);
    println!(
        "   Ground truth 计算耗时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 创建索引
    let mut config = IndexConfig::new(IndexType::IvfFlat, MetricType::L2, dim);
    config.params = IndexParams::ivf(nlist, nprobe);

    let mut index = IvfFlatIndex::new(&config).expect("Failed to create index");

    // 训练
    println!("\n⏳ 训练 IVF-Flat...");
    let start = Instant::now();
    index.train(&base).expect("Train failed");
    let train_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("   训练耗时: {:.2}ms", train_time);

    // 添加向量
    println!("⏳ 添加向量...");
    let start = Instant::now();
    index.add(&base, None).expect("Add failed");
    let add_time = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   添加耗时: {:.2}ms ({:.0} vec/s)",
        add_time,
        n_base as f64 / (add_time / 1000.0)
    );

    // 搜索 + QPS 测试
    println!("\n⏳ 性能测试 (预热 10 次 + 正式 100 次)...");

    // 预热
    for i in 0..10 {
        let query = &queries[i % n_queries * dim..(i % n_queries + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };
        let _ = index.search(query, &request);
    }

    // 正式测试
    let iterations = 100;
    let start = Instant::now();
    let mut total_recall = 0.0;

    for i in 0..iterations {
        let query_idx = i % n_queries;
        let query = &queries[query_idx * dim..(query_idx + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &request).expect("Search failed");

        let recall = calculate_recall(&result.ids, &ground_truth[query_idx], k);
        total_recall += recall;
    }

    let search_time = start.elapsed().as_secs_f64() * 1000.0;
    let qps = (iterations as f64) / (search_time / 1000.0);
    let avg_recall = total_recall / iterations as f32;

    println!("\n📈 IVF-Flat 性能结果:");
    println!("================================================");
    println!("   搜索总耗时: {:.2}ms ({} 次)", search_time, iterations);
    println!("   QPS: {:.0}", qps);
    println!("   Recall@{}: {:.3}", k, avg_recall);
    println!("================================================");

    // 对比 C++ 性能（估算）
    let cpp_qps_estimate = 5000.0; // C++ knowhere IVF-Flat 估计 QPS
    let performance_ratio = (qps / cpp_qps_estimate) * 100.0;
    println!("\n🎯 vs C++ (估算 {} QPS):", cpp_qps_estimate);
    println!("   性能比例: {:.1}%", performance_ratio);

    if performance_ratio >= 50.0 {
        println!("   ✅ 达成目标 (≥ 50%)");
    } else {
        println!("   ⚠️  未达标 (< 50%)");
    }

    // 验证召回率
    assert!(
        avg_recall >= 0.80,
        "Recall@{} too low: {:.3}",
        k,
        avg_recall
    );
}

/// IVF-PQ 性能测试
#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_ivf_pq_performance() {
    println!("\n🚀 OPT-003 Phase 3: IVF-PQ 性能基准测试");
    println!("============================================");

    let dim = 128;
    let n_base = 10_000; // 10K 向量（减少以加快测试）
    let n_queries = 50;
    let k = 10;
    let nlist = 50;
    let nprobe = 8;
    let m = 8; // PQ 子向量数
    let nbits = 8;

    println!("\n📊 配置:");
    println!("   Base vectors: {}", n_base);
    println!("   Queries: {}", n_queries);
    println!("   Dimension: {}", dim);
    println!("   nlist: {}, nprobe: {}", nlist, nprobe);
    println!("   m: {}, nbits: {}", m, nbits);
    println!("   k: {}", k);

    // 生成数据
    println!("\n⏳ 生成数据...");
    let base = generate_random_vectors(n_base, dim);
    let queries = generate_random_vectors(n_queries, dim);

    // 计算 ground truth
    println!("⏳ 计算 ground truth...");
    let start = Instant::now();
    let ground_truth = compute_ground_truth(&base, &queries, dim, k);
    println!(
        "   Ground truth 计算耗时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 创建索引
    let mut config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);
    config.params = IndexParams {
        nlist: Some(nlist),
        nprobe: Some(nprobe),
        m: Some(m),
        nbits_per_idx: Some(nbits),
        ..Default::default()
    };

    let mut index = IvfPqIndex::new(&config).expect("Failed to create index");

    // 训练
    println!("\n⏳ 训练 IVF-PQ...");
    let start = Instant::now();
    index.train(&base).expect("Train failed");
    let train_time = start.elapsed().as_secs_f64() * 1000.0;
    println!("   训练耗时: {:.2}ms", train_time);

    // 添加向量
    println!("⏳ 添加向量...");
    let start = Instant::now();
    index.add(&base, None).expect("Add failed");
    let add_time = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "   添加耗时: {:.2}ms ({:.0} vec/s)",
        add_time,
        n_base as f64 / (add_time / 1000.0)
    );

    // 搜索 + QPS 测试
    println!("\n⏳ 性能测试 (预热 10 次 + 正式 100 次)...");

    // 预热
    for i in 0..10 {
        let query = &queries[i % n_queries * dim..(i % n_queries + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };
        let _ = index.search(query, &request);
    }

    // 正式测试
    let iterations = 100;
    let start = Instant::now();
    let mut total_recall = 0.0;

    for i in 0..iterations {
        let query_idx = i % n_queries;
        let query = &queries[query_idx * dim..(query_idx + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &request).expect("Search failed");

        let recall = calculate_recall(&result.ids, &ground_truth[query_idx], k);
        total_recall += recall;
    }

    let search_time = start.elapsed().as_secs_f64() * 1000.0;
    let qps = (iterations as f64) / (search_time / 1000.0);
    let avg_recall = total_recall / iterations as f32;

    println!("\n📈 IVF-PQ 性能结果:");
    println!("================================================");
    println!("   搜索总耗时: {:.2}ms ({} 次)", search_time, iterations);
    println!("   QPS: {:.0}", qps);
    println!("   Recall@{}: {:.3}", k, avg_recall);
    println!("================================================");

    // 对比 C++ 性能（估算）
    let cpp_qps_estimate = 8000.0; // C++ knowhere IVF-PQ 估计 QPS
    let performance_ratio = (qps / cpp_qps_estimate) * 100.0;
    println!("\n🎯 vs C++ (估算 {} QPS):", cpp_qps_estimate);
    println!("   性能比例: {:.1}%", performance_ratio);

    if performance_ratio >= 50.0 {
        println!("   ✅ 达成目标 (≥ 50%)");
    } else {
        println!("   ⚠️  未达标 (< 50%)");
    }

    // 验证召回率
    assert!(
        avg_recall >= 0.75,
        "Recall@{} too low: {:.3}",
        k,
        avg_recall
    );
}

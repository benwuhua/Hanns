//! HNSW Shuffle 构建 Benchmark
//!
//! 测试随机插入顺序对 HNSW 图质量的影响
//!
//! # 理论背景
//! HNSW 构建时，向量插入顺序会影响最终图的质量：
//! - 有序插入（如按 ID 顺序）：可能导致图结构偏向早期插入的向量
//! - 随机插入（Shuffle）：打乱插入顺序，使图结构更均匀
//!
//! # 预期效果
//! - 召回率提升：R@10 提升 2-5%
//! - 构建时间：基本不变（仅增加 shuffle 开销）
//! - 内存占用：不变
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_hnsw_shuffle -- --nocapture
//! ```

use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::prelude::*;
use std::time::Instant;

/// 生成高斯分布数据集
fn generate_gaussian_dataset(num_vectors: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        let u1 = rng.gen::<f32>();
        let u2 = rng.gen::<f32>();
        let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(gaussian);
    }

    data
}

/// 计算 ground truth（暴力搜索）
fn compute_ground_truth(queries: &[f32], data: &[f32], dim: usize, k: usize) -> Vec<Vec<i64>> {
    let num_queries = queries.len() / dim;
    let num_vectors = data.len() / dim;
    let mut results = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_vectors);

        for j in 0..num_vectors {
            let vec = &data[j * dim..(j + 1) * dim];
            let dist = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        results.push(distances.iter().map(|(idx, _)| *idx as i64).collect());
    }

    results
}

/// 创建索引配置
fn create_hnsw_params(m: usize, ef_construction: usize, ef_search: usize) -> IndexParams {
    IndexParams::hnsw(ef_construction, ef_search, 1.0 / (m as f32).ln())
}

/// Benchmark 主函数
fn benchmark_shuffle_build() {
    const NUM_VECTORS: usize = 50_000;
    const NUM_QUERIES: usize = 100;
    const DIM: usize = 128;
    const K: usize = 100;
    const M: usize = 32;
    const EF_CONSTRUCTION: usize = 200;
    const EF_SEARCH: usize = 128;

    println!("=== HNSW Shuffle 构建 Benchmark ===");
    println!("数据集：{} 向量 x {} 维", NUM_VECTORS, DIM);
    println!("查询数：{}", NUM_QUERIES);
    println!(
        "HNSW 参数：M={}, ef_construction={}, ef_search={}",
        M, EF_CONSTRUCTION, EF_SEARCH
    );
    println!();

    // 生成数据集
    println!("生成数据集...");
    let data = generate_gaussian_dataset(NUM_VECTORS, DIM, 42);
    let queries = generate_gaussian_dataset(NUM_QUERIES, DIM, 123);

    // 计算 ground truth
    println!("计算 ground truth...");
    let gt = compute_ground_truth(&queries, &data, DIM, K);

    // 创建索引置换（shuffle 版本）
    let mut rng = StdRng::seed_from_u64(456);
    let mut indices: Vec<usize> = (0..NUM_VECTORS).collect();
    indices.shuffle(&mut rng);

    let shuffled_data: Vec<f32> = indices
        .iter()
        .flat_map(|&idx| data[idx * DIM..(idx + 1) * DIM].iter().copied())
        .collect();

    // 测试 1: 原始顺序构建
    println!("\n[测试 1] 原始顺序构建...");
    let params = create_hnsw_params(M, EF_CONSTRUCTION, EF_SEARCH);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };
    let mut index_original = HnswIndex::new(&config).unwrap();

    let train_start = Instant::now();
    index_original.train(&data).unwrap();
    let train_time = train_start.elapsed();

    let build_start = Instant::now();
    index_original.add(&data, None).unwrap();
    let build_time_original = build_start.elapsed();

    // 搜索测试
    let search_start = Instant::now();
    let mut results_original = Vec::new();
    for i in 0..NUM_QUERIES {
        let query = &queries[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: K,
            ..Default::default()
        };
        let result = index_original.search(query, &req).unwrap();
        results_original.push(result);
    }
    let search_time_original = search_start.elapsed();

    // 计算召回率
    let recalls_original: Vec<f64> = results_original
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let result_ids: Vec<i64> = result.ids.iter().copied().collect();
            // average_recall_at_k expects &[Vec<i64>] for ground truth and &[Vec<i32>] for results
            // We need to call it differently
            compute_recall(&gt[i], &result_ids)
        })
        .collect();
    let avg_recall_original = recalls_original.iter().sum::<f64>() / recalls_original.len() as f64;

    let qps_original = NUM_QUERIES as f64 / search_time_original.as_secs_f64();

    println!("  训练时间：{:?}", train_time);
    println!(
        "  构建时间：{:?} ({:.2} ms/向量)",
        build_time_original,
        build_time_original.as_secs_f64() * 1000.0 / NUM_VECTORS as f64
    );
    println!("  搜索时间：{:?}", search_time_original);
    println!("  QPS: {:.2}", qps_original);
    println!("  平均召回率@{}: {:.4}", K, avg_recall_original);

    // 测试 2: Shuffle 顺序构建
    println!("\n[测试 2] Shuffle 顺序构建...");
    let mut index_shuffled = HnswIndex::new(&config).unwrap();

    let train_start = Instant::now();
    index_shuffled.train(&shuffled_data).unwrap();
    let train_time = train_start.elapsed();

    let build_start = Instant::now();
    index_shuffled.add(&shuffled_data, None).unwrap();
    let build_time_shuffled = build_start.elapsed();

    // 搜索测试（使用原始查询，需要映射回原始 ID）
    let search_start = Instant::now();
    let mut results_shuffled = Vec::new();
    for i in 0..NUM_QUERIES {
        let query = &queries[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: K,
            ..Default::default()
        };
        let result = index_shuffled.search(query, &req).unwrap();
        // 映射回原始 ID
        let mapped_ids: Vec<i64> = result
            .ids
            .iter()
            .map(|&shuffled_idx| indices[shuffled_idx as usize] as i64)
            .collect();
        results_shuffled.push(mapped_ids);
    }
    let search_time_shuffled = search_start.elapsed();

    // 计算召回率
    let recalls_shuffled: Vec<f64> = results_shuffled
        .iter()
        .enumerate()
        .map(|(i, result_ids)| compute_recall(&gt[i], result_ids))
        .collect();
    let avg_recall_shuffled = recalls_shuffled.iter().sum::<f64>() / recalls_shuffled.len() as f64;

    let qps_shuffled = NUM_QUERIES as f64 / search_time_shuffled.as_secs_f64();

    println!("  训练时间：{:?}", train_time);
    println!(
        "  构建时间：{:?} ({:.2} ms/向量)",
        build_time_shuffled,
        build_time_shuffled.as_secs_f64() * 1000.0 / NUM_VECTORS as f64
    );
    println!("  搜索时间：{:?}", search_time_shuffled);
    println!("  QPS: {:.2}", qps_shuffled);
    println!("  平均召回率@{}: {:.4}", K, avg_recall_shuffled);

    // 对比分析
    println!("\n=== 对比分析 ===");
    let recall_improvement = (avg_recall_shuffled - avg_recall_original) * 100.0;
    let build_time_diff =
        (build_time_shuffled.as_secs_f64() - build_time_original.as_secs_f64()) * 1000.0;

    println!("召回率变化：{:+.2}%", recall_improvement);
    println!("构建时间变化：{:+.2} ms", build_time_diff);
    println!("QPS 变化：{:+.2}", qps_shuffled - qps_original);

    if recall_improvement > 0.0 {
        println!("✅ Shuffle 构建提升了召回率！");
    } else if recall_improvement < 0.0 {
        println!("⚠️  Shuffle 构建降低了召回率（可能是随机波动）");
    } else {
        println!("✓ Shuffle 构建对召回率无显著影响");
    }
}

/// 计算单个查询的召回率
fn compute_recall(ground_truth: &[i64], results: &[i64]) -> f64 {
    let k = results.len();
    let mut matches = 0;
    for &id in results {
        if ground_truth.contains(&id) {
            matches += 1;
        }
    }
    matches as f64 / k as f64
}

#[test]
fn test_hnsw_shuffle_benchmark() {
    benchmark_shuffle_build();
}

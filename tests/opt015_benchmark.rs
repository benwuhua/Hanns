//! OPT-015: HNSW 构建优化 - Benchmark 对比
//!
//! 测试不同参数配置对构建时间的影响
//!
//! 参数矩阵：
//! - M: 8, 16, 32
//! - ef_construction: 100, 200, 400
//! - level_multiplier: 1/ln(M)

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

struct BenchmarkResult {
    m: usize,
    ef_construction: usize,
    build_time_ms: f64,
    recall_at_10: f64,
}

fn run_benchmark(n: usize, dim: usize, m: usize, ef_construction: usize) -> BenchmarkResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(10, dim);

    // Compute ground truth (brute force)
    let mut ground_truth: Vec<Vec<usize>> = Vec::new();
    for i in 0..10 {
        let query = &queries[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for j in 0..n {
            let base = &vectors[j * dim..(j + 1) * dim];
            let dist: f32 = query
                .iter()
                .zip(base.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            distances.push((j, dist));
        }
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: Vec<usize> = distances.into_iter().take(10).map(|(idx, _)| idx).collect();
        ground_truth.push(gt);
    }

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    let start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall
    let mut recall_sum = 0.0;
    for i in 0..10 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 10,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();

        let gt_set: std::collections::HashSet<usize> = ground_truth[i].iter().copied().collect();
        let mut hits = 0;
        for &id in &result.ids {
            if gt_set.contains(&(id as usize)) {
                hits += 1;
            }
        }
        recall_sum += hits as f64 / 10.0;
    }
    let recall = recall_sum / 10.0;

    BenchmarkResult {
        m,
        ef_construction,
        build_time_ms: build_time,
        recall_at_10: recall,
    }
}

#[test]
fn test_hnsw_parameter_sweep() {
    println!("\n=== OPT-015: HNSW 参数优化 Benchmark ===\n");
    println!("数据集：10K 向量，128 维 (缩小规模用于快速测试)\n");

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Test different parameter combinations
    let m_values = vec![8, 16, 32];
    let ef_values = vec![100, 200, 400];

    println!(
        "Testing {} parameter combinations...\n",
        m_values.len() * ef_values.len()
    );

    for &m in &m_values {
        for &ef in &ef_values {
            let result = run_benchmark(10_000, 128, m, ef);
            println!(
                "M={:2}, ef={:3} -> Build: {:8.2}ms, R@10: {:.3}",
                m, ef, result.build_time_ms, result.recall_at_10
            );
            results.push(result);
        }
    }

    println!("\n📊 优化建议:");
    println!("   1. 减小 M 可显著降低构建时间 (但可能降低召回率)");
    println!("   2. 减小 ef_construction 可加速构建 (但图质量下降)");
    println!("   3. 对于 100K 向量，建议 M=16, ef_construction=200");
    println!("   4. 关键优化：移除 HashMap，使用直接索引");
    println!("   5. 关键优化：批量插入时预分配存储");

    // Find best trade-off (fastest with recall > 0.8)
    let best = results
        .iter()
        .filter(|r| r.recall_at_10 > 0.8)
        .min_by(|a, b| a.build_time_ms.partial_cmp(&b.build_time_ms).unwrap());

    if let Some(best) = best {
        println!(
            "\n✅ 推荐配置：M={}, ef_construction={}",
            best.m, best.ef_construction
        );
        println!("   构建时间：{:.2}ms (10K 向量)", best.build_time_ms);
        println!(
            "   预估 100K 向量：{:.2}s",
            best.build_time_ms * 10.0 / 1000.0
        );
    }
}

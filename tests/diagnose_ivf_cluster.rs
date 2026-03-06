//! 诊断 IVF-Flat 聚类分布和性能关系
//!
//! 目的：找出为什么不同数据分布导致性能差异 5-10 倍

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::IvfFlatIndex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const DIM: usize = 128;
const NLIST: usize = 100;
const NPROBE: usize = 16;

/// 确定性模式数据（quick_ivf_perf 使用）
fn gen_pattern_data(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| ((i % 100) as f32 / 100.0 - 0.5) * 2.0)
        .collect()
}

/// 真正伪随机数据（BENCH-048 使用）
fn gen_random_data(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn analyze_distribution(name: &str, data: &[f32], dim: usize) {
    let n = data.len() / dim;

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams::ivf(NLIST, NPROBE),
    };

    let mut index = IvfFlatIndex::new(&config).unwrap();

    println!("\n=== {} ({} vectors) ===", name, n);

    // 训练
    let train_start = Instant::now();
    index.train(data).unwrap();
    println!("Training: {:.2}s", train_start.elapsed().as_secs_f64());

    // 添加
    let add_start = Instant::now();
    index.add(data, None).unwrap();
    println!("Add: {:.2}s", add_start.elapsed().as_secs_f64());

    // 分析聚类分布（通过反射访问内部状态）
    // 由于 invlist_ids 是私有字段，我们通过搜索性能推断

    // 搜索性能测试
    let num_queries = 100;
    let queries = &data[0..num_queries * dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: NPROBE,
        ..Default::default()
    };

    // Warmup
    let _ = index.search(&queries[0..dim], &req).unwrap();

    // Benchmark
    let search_start = Instant::now();
    for i in 0..num_queries {
        let q = &queries[i * dim..(i + 1) * dim];
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = search_start.elapsed().as_secs_f64();
    let qps = num_queries as f64 / search_time;

    println!("Search QPS: {:.0}", qps);
    println!(
        "Latency: {:.2}ms",
        search_time * 1000.0 / num_queries as f64
    );

    // 计算每个查询访问的向量数（推断）
    // nprobe 个簇，每个簇平均 n/NLIST 个向量
    let avg_vectors_per_probe = (n as f64 / NLIST as f64) * NPROBE as f64;
    println!(
        "Avg vectors per query (estimated): {:.0}",
        avg_vectors_per_probe
    );
}

#[test]
fn diagnose_cluster_distribution() {
    println!("\n========================================");
    println!("IVF-Flat 性能波动诊断");
    println!("========================================");

    // 测试 10K
    println!("\n--- 10K Scale ---");
    let pattern_10k = gen_pattern_data(10_000, DIM);
    let random_10k = gen_random_data(42, 10_000, DIM);

    analyze_distribution("Pattern 10K", &pattern_10k, DIM);
    analyze_distribution("Random 10K", &random_10k, DIM);

    // 测试 50K
    println!("\n--- 50K Scale ---");
    let pattern_50k = gen_pattern_data(50_000, DIM);
    let random_50k = gen_random_data(42, 50_000, DIM);

    analyze_distribution("Pattern 50K", &pattern_50k, DIM);
    analyze_distribution("Random 50K", &random_50k, DIM);

    println!("\n========================================");
    println!("诊断结论:");
    println!("如果 Pattern 数据 QPS 远低于 Random 数据,");
    println!("说明模式数据的聚类分布不均匀，导致某些簇过大。");
    println!("========================================");
}

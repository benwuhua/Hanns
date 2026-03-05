//! OPT-015: HNSW 构建性能优化测试
//!
//! 问题分析：
//! - 原始版本：34.7s (100K 向量)
//! - 瓶颈：HashMap 查找、逐个插入、过度分配
//!
//! 优化策略：
//! 1. 移除 HashMap，使用直接索引
//! 2. 两阶段构建：先存向量，再建图
//! 3. 减少 search_layer 中的分配
//! 4. 简化邻居选择

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

#[test]
fn test_hnsw_build_performance() {
    println!("\n=== OPT-015: HNSW 构建性能测试 ===\n");
    println!("数据集：100K 向量，128 维");
    println!("目标：构建时间 <500ms\n");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: 128,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    let vectors = generate_vectors(100_000, 128);

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    println!("Index         Build(ms)");
    println!("------------------------");
    println!("HNSW          {:>10.2}", build_time);

    println!("\n📊 优化目标:");
    println!("   当前构建时间：{:.2}ms", build_time);
    println!("   目标构建时间：<500ms");

    if build_time < 500.0 {
        println!("\n✅ 验收通过：构建时间 <500ms");
    } else {
        println!("\n❌ 验收失败：构建时间 {:.2}ms > 500ms", build_time);
        println!("   需要进一步优化");
    }
}

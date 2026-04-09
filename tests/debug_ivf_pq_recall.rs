#![cfg(feature = "long-tests")]
//! 诊断 IVF-PQ 召回率问题
//! 这个入口显式走 `src/faiss/ivfpq.rs` 的 residual-PQ hot path，
//! 不代表 `src/faiss/ivf.rs` 的简化 coarse-assignment scaffold。
//!
//! 运行: cargo test --release --test debug_ivf_pq_recall -- --nocapture

use hanns::api::{IndexConfig, IndexParams, SearchRequest};
use hanns::faiss::IvfPqIndex;
use hanns::{IndexType, MetricType};

fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

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

#[test]
fn debug_ivf_pq_recall() {
    println!("\n🔍 诊断 IVF-PQ 召回率问题");
    println!("==================================");
    println!("   hot path: src/faiss/ivfpq.rs (residual PQ)");
    println!("   placeholder scaffold: src/faiss/ivf.rs");

    let dim = 128;
    let n_base = 1000; // 小数据集便于调试
    let n_queries = 10;
    let k = 10;
    let nlist = 10;
    let nprobe = 4;
    let m = 8;
    let nbits = 8;

    println!("\n📊 配置:");
    println!("   Base: {}, Queries: {}", n_base, n_queries);
    println!("   nlist: {}, nprobe: {}", nlist, nprobe);
    println!("   m: {}, nbits: {}", m, nbits);

    // 生成数据
    let base = generate_random_vectors(n_base, dim);
    let queries = generate_random_vectors(n_queries, dim);

    // Ground truth
    let ground_truth = compute_ground_truth(&base, &queries, dim, k);

    // 创建索引
    let mut config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);
    config.params = IndexParams::ivf(nlist, nprobe);
    config.params.m = Some(m);
    config.params.nbits_per_idx = Some(nbits);

    let mut index = IvfPqIndex::new(&config).expect("Failed to create index");

    // 训练
    println!("\n⏳ 训练...");
    index.train(&base).expect("Train failed");

    // 添加
    println!("⏳ 添加向量...");
    index.add(&base, None).expect("Add failed");

    // 搜索并检查每个查询
    println!("\n🔍 检查前 5 个查询的召回率:");
    let mut total_recall = 0.0;

    for i in 0..5.min(n_queries) {
        let query = &queries[i * dim..(i + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };

        let result = index.search(query, &request).expect("Search failed");

        // 计算召回率
        let pred_set: std::collections::HashSet<i64> = result.ids.iter().copied().take(k).collect();
        let gt_set: std::collections::HashSet<i64> = ground_truth[i].iter().copied().collect();

        let intersection = pred_set.intersection(&gt_set).count();
        let recall = intersection as f32 / k as f32;
        total_recall += recall;

        println!("\n   Query {}:", i);
        println!(
            "     预测 IDs: {:?}",
            &result.ids[..k.min(result.ids.len())]
        );
        println!("     真实 IDs: {:?}", &ground_truth[i][..k]);
        println!(
            "     距离: {:?}",
            &result.distances[..k.min(result.distances.len())]
        );
        println!("     召回率: {:.3}", recall);

        // 检查距离是否异常
        if !result.distances.is_empty() {
            let min_dist = result
                .distances
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max_dist = result
                .distances
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            println!("     距离范围: [{:.3}, {:.3}]", min_dist, max_dist);
        }
    }

    let avg_recall = total_recall / 5.0;
    println!("\n📊 平均召回率 (前 5 个查询): {:.3}", avg_recall);

    // 完整测试
    println!("\n⏳ 完整 {} 次查询测试...", n_queries);
    let mut full_recall = 0.0;
    for i in 0..n_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let request = SearchRequest {
            top_k: k,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &request).expect("Search failed");

        let pred_set: std::collections::HashSet<i64> = result.ids.iter().copied().take(k).collect();
        let gt_set: std::collections::HashSet<i64> = ground_truth[i].iter().copied().collect();

        let intersection = pred_set.intersection(&gt_set).count();
        full_recall += intersection as f32 / k as f32;
    }

    println!("\n✅ 最终召回率: {:.3}", full_recall / n_queries as f32);
}

//! 诊断 IVF-PQ 召回率问题 - 测试不同 nprobe 和配置
//!
//! 运行: cargo test --release --test debug_ivf_pq_nprobe -- --nocapture

use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::faiss::IvfPqIndex;
use knowhere_rs::{IndexType, MetricType};

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
fn debug_ivf_pq_nprobe_sweep() {
    println!("\n🔍 诊断 IVF-PQ 召回率 - Nprobe 扫描");
    println!("=========================================");

    let dim = 128;
    let n_base = 1000;
    let n_queries = 10;
    let k = 10;
    let nlist = 10;
    let m = 8;
    let nbits = 8;

    // 生成数据
    let base = generate_random_vectors(n_base, dim);
    let queries = generate_random_vectors(n_queries, dim);
    let ground_truth = compute_ground_truth(&base, &queries, dim, k);

    // 创建索引
    let mut config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);
    config.params = IndexParams::ivf(nlist, 4);
    config.params.m = Some(m);
    config.params.nbits_per_idx = Some(nbits);

    let mut index = IvfPqIndex::new(&config).expect("Failed to create index");
    index.train(&base).expect("Train failed");
    index.add(&base, None).expect("Add failed");

    // 测试不同 nprobe
    println!("\n📊 测试不同 nprobe:");
    for nprobe in [1, 2, 4, 6, 8, 10] {
        let mut total_recall = 0.0;

        for i in 0..n_queries {
            let query = &queries[i * dim..(i + 1) * dim];
            let request = SearchRequest {
                top_k: k,
                nprobe,
                ..Default::default()
            };
            let result = index.search(query, &request).expect("Search failed");

            let pred_set: std::collections::HashSet<i64> =
                result.ids.iter().copied().take(k).collect();
            let gt_set: std::collections::HashSet<i64> = ground_truth[i].iter().copied().collect();

            let intersection = pred_set.intersection(&gt_set).count();
            total_recall += intersection as f32 / k as f32;
        }

        let avg_recall = total_recall / n_queries as f32;
        println!("   nprobe={:2}: Recall@10 = {:.3}", nprobe, avg_recall);
    }

    // 测试不同 nlist
    println!("\n📊 测试不同 nlist (nprobe=nlist):");
    for nlist_test in [5, 10, 20, 50] {
        let mut config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);
        config.params = IndexParams::ivf(nlist_test, nlist_test);
        config.params.m = Some(m);
        config.params.nbits_per_idx = Some(nbits);

        let mut index = IvfPqIndex::new(&config).expect("Failed to create index");
        index.train(&base).expect("Train failed");
        index.add(&base, None).expect("Add failed");

        let mut total_recall = 0.0;
        for i in 0..n_queries {
            let query = &queries[i * dim..(i + 1) * dim];
            let request = SearchRequest {
                top_k: k,
                nprobe: nlist_test,
                ..Default::default()
            };
            let result = index.search(query, &request).expect("Search failed");

            let pred_set: std::collections::HashSet<i64> =
                result.ids.iter().copied().take(k).collect();
            let gt_set: std::collections::HashSet<i64> = ground_truth[i].iter().copied().collect();

            let intersection = pred_set.intersection(&gt_set).count();
            total_recall += intersection as f32 / k as f32;
        }

        let avg_recall = total_recall / n_queries as f32;
        println!("   nlist={:2}: Recall@10 = {:.3}", nlist_test, avg_recall);
    }
}

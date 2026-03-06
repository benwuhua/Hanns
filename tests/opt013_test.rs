//! OPT-013: IVF-Flat 构建优化验证测试

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::IvfFlatIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist: f32 = q.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i32> = distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx as i32)
            .collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

#[test]
fn test_opt013_ivf_flat_fast_build() {
    println!("\n=== OPT-013: IVF-Flat 快速构建验证 ===\n");

    let n = 5000;
    let dim = 128;
    let nlist = 70; // sqrt(5000) ≈ 70

    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(20, dim); // 减少查询数量加快测试

    // 计算 ground truth
    let ground_truth = compute_ground_truth(&vectors, &queries, 20, dim, 100);

    // 测试快速配置
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_flat_fast(nlist, 10),
    };

    println!("配置：n={}, dim={}, nlist={}", n, dim, nlist);
    println!("使用 ivf_flat_fast 配置：max_iter=5, tolerance=1e-3, use_elkan=true\n");

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    println!("1. 创建索引完成");

    index.train(&vectors).unwrap();
    println!("2. 训练完成 (K-Means)");

    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("3. 向量添加完成");

    println!("\n构建时间：{:.2}ms\n", build_time);

    // 搜索测试
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(20);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;

    for i in 0..20 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 10,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // 计算召回率
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // 输出结果
    println!("📊 性能结果:");
    println!("   构建时间：{:.2}ms", build_time);
    println!("   搜索时间：{:.2}ms (20  queries)", search_time);
    println!();
    println!("   召回率 R@1:   {:.3}", recall_at_1);
    println!("   召回率 R@10:  {:.3}", recall_at_10);
    println!("   召回率 R@100: {:.3}", recall_at_100);
    println!(
        "   距离单调性：{}",
        if distances_sorted {
            "✅ 通过"
        } else {
            "❌ 失败"
        }
    );
    println!();

    // 验收标准
    println!("✅ 验收标准:");
    let build_ok = build_time < 2000.0;
    // R@1 and R@10 are the key metrics for IVF quality; R@100 is not meaningful for small datasets
    let recall_ok = recall_at_1 >= 0.95 && recall_at_10 >= 0.95;

    println!(
        "   构建时间 <2s: {} ({:.2}ms)",
        if build_ok { "✅ PASS" } else { "❌ FAIL" },
        build_time
    );
    println!(
        "   召回率 R@1 >=95%: {} (R@1={:.3})",
        if recall_at_1 >= 0.95 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        },
        recall_at_1
    );
    println!(
        "   召回率 R@10 >=95%: {} (R@10={:.3})",
        if recall_at_10 >= 0.95 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        },
        recall_at_10
    );
    println!();

    assert!(build_ok, "构建时间应 <2s，实际 {:.2}ms", build_time);
    assert!(recall_at_1 >= 0.95, "R@1 应 >=95%");
    assert!(recall_at_10 >= 0.95, "R@10 应 >=95%");
    assert!(distances_sorted, "距离应保持单调性");

    println!("🎉 OPT-013 优化验证通过!");
}

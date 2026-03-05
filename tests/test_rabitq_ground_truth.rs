//! RaBitQ Ground Truth 验证测试
//! 使用真正的 brute-force ground truth，而不是 IVF-Flat 的结果

use knowhere_rs::{IvfRaBitqConfig, IvfRaBitqIndex, SearchRequest};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn generate_random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * dim];
    for val in data.iter_mut() {
        *val = rng.gen::<f32>();
    }
    data
}

/// 计算真正的 brute-force ground truth
fn compute_brute_force_ground_truth(
    database: &[f32],
    query: &[f32],
    dim: usize,
    k: usize,
) -> Vec<Vec<i64>> {
    let n_database = database.len() / dim;
    let n_query = query.len() / dim;

    let mut ground_truth = Vec::with_capacity(n_query);

    for i in 0..n_query {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_database);

        // Brute-force 计算所有距离
        for j in 0..n_database {
            let b = &database[j * dim..(j + 1) * dim];
            let dist = l2_distance_squared(q, b);
            distances.push((j, dist));
        }

        // 排序并取 top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i64> = distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx as i64)
            .collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_recall(approx_ids: &[i64], ground_truth_ids: &[i64], k: usize) -> f32 {
    let approx_set: std::collections::HashSet<i64> = approx_ids.iter().copied().collect();
    let mut matched = 0;
    for &id in ground_truth_ids.iter().take(k) {
        if approx_set.contains(&id) {
            matched += 1;
        }
    }
    matched as f32 / k.min(ground_truth_ids.len()) as f32
}

#[test]
fn test_rabitq_with_true_ground_truth() {
    println!("\n=== RaBitQ Ground Truth 验证测试 ===");

    let dim = 128;
    let nlist = 50;
    let n_database = 10000;
    let n_query = 100;

    let mut rng = StdRng::seed_from_u64(42);

    // 生成数据集
    println!(
        "生成数据集：{} 向量，{} 维，{} 查询",
        n_database, dim, n_query
    );
    let database = generate_random_vectors(&mut rng, n_database, dim);
    let queries = generate_random_vectors(&mut rng, n_query, dim);

    // 计算真正的 brute-force ground truth
    println!("计算 brute-force ground truth...");
    let ground_truth = compute_brute_force_ground_truth(&database, &queries, dim, 100);

    // 构建 RaBitQ 索引
    println!("构建 RaBitQ 索引...");
    let rabitq_config = IvfRaBitqConfig::new(dim, nlist);
    let mut rabitq_index = IvfRaBitqIndex::new(rabitq_config);
    rabitq_index.train(&database).expect("RaBitQ 训练失败");
    rabitq_index.add(&database, None).expect("RaBitQ 添加失败");

    // 计算压缩比
    let raw_size = n_database * dim * std::mem::size_of::<f32>();
    let rabitq_size = rabitq_index.size();
    let compression_ratio = raw_size as f64 / rabitq_size as f64;

    println!("\n压缩比：{:.2}x", compression_ratio);
    println!("原始大小：{:.2} MB", raw_size as f64 / 1024.0 / 1024.0);
    println!(
        "RaBitQ 大小：{:.2} MB\n",
        rabitq_size as f64 / 1024.0 / 1024.0
    );

    // 测试不同 nprobe 的召回率
    let nprobe_values = [5, 10, 20, 30, 40, 50];

    println!("nprobe | R@10 | R@100");
    println!("-------|------|-------");

    for &nprobe in &nprobe_values {
        let mut total_r10 = 0.0;
        let mut total_r100 = 0.0;

        for i in 0..n_query {
            let query = &queries[i * dim..(i + 1) * dim];

            // RaBitQ 搜索 (top-100)
            let rabitq_req = SearchRequest {
                top_k: 100,
                nprobe,
                filter: None,
                params: None,
                radius: None,
            };
            let rabitq_results = rabitq_index
                .search(query, &rabitq_req)
                .expect("RaBitQ 搜索失败");

            let r10 = compute_recall(&rabitq_results.ids, &ground_truth[i], 10);
            let r100 = compute_recall(&rabitq_results.ids, &ground_truth[i], 100);

            total_r10 += r10;
            total_r100 += r100;
        }

        let avg_r10 = total_r10 / n_query as f32;
        let avg_r100 = total_r100 / n_query as f32;

        println!("{:6} | {:.4} | {:.4}", nprobe, avg_r10, avg_r100);

        // 验证召回率是否达到预期
        if nprobe >= 20 {
            // 目标：R@10 > 80%
            assert!(avg_r10 > 0.80, "R@10={:.4} 未达到预期 (>0.80)", avg_r10);
        }
    }

    println!("\n✅ RaBitQ Ground Truth 验证通过！");
}

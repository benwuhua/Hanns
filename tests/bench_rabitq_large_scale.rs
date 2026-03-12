#![cfg(feature = "long-tests")]
//! RaBitQ 大规模召回率测试
//! 验证 OPT-038 在更大规模数据集上的效果
//! 目标：R@10 > 80%

use knowhere_rs::{
    faiss::IvfFlatIndex, IndexConfig, IndexType, IvfRaBitqConfig, IvfRaBitqIndex, MetricType,
    SearchRequest,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn generate_random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * dim];
    for val in data.iter_mut() {
        *val = rng.gen::<f32>();
    }
    data
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
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_rabitq_large_scale_recall() {
    println!("\n=== RaBitQ 大规模召回率测试 (BENCH-037) ===");

    let dim = 128;
    let nlist = 50; // 增加聚类数
    let n_database = 10000; // 增加到 10K
    let n_query = 100;

    let mut rng = StdRng::seed_from_u64(42);

    // 生成数据集
    println!(
        "生成数据集：{} 向量，{} 维，{} 查询",
        n_database, dim, n_query
    );
    let database = generate_random_vectors(&mut rng, n_database, dim);
    let queries = generate_random_vectors(&mut rng, n_query, dim);

    // 构建 Ground Truth (IVF-Flat)
    println!("构建 Ground Truth (IVF-Flat)...");
    let mut flat_config = IndexConfig::new(IndexType::IvfFlat, MetricType::L2, dim);
    flat_config.params.nlist = Some(nlist);
    let mut flat_index = IvfFlatIndex::new(&flat_config).expect("Flat 创建失败");
    flat_index.train(&database).expect("Flat 训练失败");
    flat_index.add(&database, None).expect("Flat 添加失败");

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

            // Ground Truth 搜索 (top-100)
            let gt_req = SearchRequest {
                top_k: 100,
                nprobe: nlist,
                filter: None,
                params: None,
                radius: None,
            };
            let gt_results = flat_index.search(query, &gt_req).expect("Flat 搜索失败");

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

            let r10 = compute_recall(&rabitq_results.ids, &gt_results.ids, 10);
            let r100 = compute_recall(&rabitq_results.ids, &gt_results.ids, 100);

            total_r10 += r10;
            total_r100 += r100;
        }

        let avg_r10 = total_r10 / n_query as f32;
        let avg_r100 = total_r100 / n_query as f32;

        println!("{:6} | {:.4} | {:.4}", nprobe, avg_r10, avg_r100);
    }

    println!("\n✅ RaBitQ 大规模召回率测试完成！");
    println!("对比：");
    println!("  - 小数据集 (2K 向量): R@10 ≈ 70.5%");
    println!("  - 大数据集 (10K 向量): 见上表");
    println!("  - 目标：R@10 > 80% (nprobe >= 20)");
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_rabitq_optimal_config() {
    println!("\n=== RaBitQ 最优配置探索 ===");

    let dim = 128;
    let n_database = 5000;
    let n_query = 50;

    let mut rng = StdRng::seed_from_u64(123);

    // 生成数据集
    let database = generate_random_vectors(&mut rng, n_database, dim);
    let queries = generate_random_vectors(&mut rng, n_query, dim);

    // 测试不同 nlist 配置
    let nlist_values = [20, 50, 100];

    for &nlist in &nlist_values {
        println!("\nnlist = {}", nlist);

        // Ground Truth
        let mut flat_config = IndexConfig::new(IndexType::IvfFlat, MetricType::L2, dim);
        flat_config.params.nlist = Some(nlist);
        let mut flat_index = IvfFlatIndex::new(&flat_config).expect("Flat 创建失败");
        flat_index.train(&database).expect("Flat 训练失败");
        flat_index.add(&database, None).expect("Flat 添加失败");

        // RaBitQ
        let rabitq_config = IvfRaBitqConfig::new(dim, nlist);
        let mut rabitq_index = IvfRaBitqIndex::new(rabitq_config);
        rabitq_index.train(&database).expect("RaBitQ 训练失败");
        rabitq_index.add(&database, None).expect("RaBitQ 添加失败");

        // 测试最优 nprobe (nprobe = nlist / 2)
        let nprobe = nlist / 2;
        let mut total_r10 = 0.0;

        for i in 0..n_query {
            let query = &queries[i * dim..(i + 1) * dim];

            let gt_req = SearchRequest {
                top_k: 10,
                nprobe: nlist,
                filter: None,
                params: None,
                radius: None,
            };
            let gt_results = flat_index.search(query, &gt_req).expect("Flat 搜索失败");

            let rabitq_req = SearchRequest {
                top_k: 10,
                nprobe,
                filter: None,
                params: None,
                radius: None,
            };
            let rabitq_results = rabitq_index
                .search(query, &rabitq_req)
                .expect("RaBitQ 搜索失败");

            let r10 = compute_recall(&rabitq_results.ids, &gt_results.ids, 10);
            total_r10 += r10;
        }

        let avg_r10 = total_r10 / n_query as f32;
        println!("  nprobe={} (nlist/2): R@10={:.4}", nprobe, avg_r10);
    }

    println!("\n✅ 最优配置探索完成！");
}

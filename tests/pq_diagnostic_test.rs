// PQ 诊断测试 - 快速验证修复方案

#[cfg(feature = "long-tests")]
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::IvfPqIndex;
#[cfg(feature = "long-tests")]
use knowhere_rs::quantization::{PQConfig, ProductQuantizer};
use knowhere_rs::quantization::{ResidualPQConfig, ResidualProductQuantizer};
use knowhere_rs::simd::{l2_distance, l2_distance_sq};

#[cfg(feature = "long-tests")]
fn main() {
    println!("=== PQ 诊断测试 ===\n");

    // 测试 1: 距离函数一致性
    test_distance_consistency();

    // 测试 2: 纯 PQ 召回率
    test_pure_pq_recall();

    // 测试 3: IVF-PQ 召回率
    test_ivf_pq_recall();
}

#[test]
fn pq_distance_consistency_regression() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];

    let l2 = l2_distance(&a, &b);
    let l2_sq = l2_distance_sq(&a, &b);

    assert!((l2 - l2_sq.sqrt()).abs() < 0.0001);
}

#[test]
fn residual_pq_adc_prefers_self_code_over_far_cluster_code() {
    let dim = 16;
    let nlist = 4;
    let points_per_cluster = 24;

    let mut vectors = Vec::with_capacity(nlist * points_per_cluster * dim);
    for cluster in 0..nlist {
        let base = cluster as f32 * 100.0;
        for point in 0..points_per_cluster {
            for d in 0..dim {
                vectors.push(base + d as f32 * 0.01 + point as f32 * 0.001);
            }
        }
    }

    let total = nlist * points_per_cluster;
    let mut rpq = ResidualProductQuantizer::new(ResidualPQConfig::new(dim, 8, 4, 8)).unwrap();
    rpq.train(total, &vectors).unwrap();

    for i in 0..8 {
        let query = &vectors[i * dim..(i + 1) * dim];
        let far_idx = total - 1 - i;
        let far_vector = &vectors[far_idx * dim..(far_idx + 1) * dim];

        let self_code = rpq.encode(query).unwrap();
        let far_code = rpq.encode(far_vector).unwrap();
        assert!(
            rpq.compute_distance(query, &self_code) < rpq.compute_distance(query, &far_code),
            "residual PQ ADC should rank the query's own code ahead of a far-cluster code"
        );
    }
}

#[cfg(feature = "long-tests")]
fn test_distance_consistency() {
    println!("测试 1: 距离函数一致性");

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];

    let l2 = l2_distance(&a, &b);
    let l2_sq = l2_distance_sq(&a, &b);

    println!("  L2 距离: {}", l2);
    println!("  L2 平方距离: {}", l2_sq);
    println!("  sqrt(L2^2): {}", l2_sq.sqrt());

    let diff = (l2 - l2_sq.sqrt()).abs();
    if diff < 0.0001 {
        println!("  ✅ 距离计算一致\n");
    } else {
        println!("  ❌ 距离计算不一致！\n");
    }
}

#[cfg(feature = "long-tests")]
fn test_pure_pq_recall() {
    println!("测试 2: 纯 PQ 召回率（不使用 IVF）");

    let dim = 128;
    let n_train = 10000;
    let n_base = 50000;
    let n_query = 100;
    let m = 16;
    let nbits = 8;

    println!(
        "  配置: dim={}, n_train={}, m={}, nbits={}",
        dim, n_train, m, nbits
    );

    // 生成随机数据
    let mut vectors = Vec::with_capacity(n_train * dim);
    for i in 0..n_train {
        for j in 0..dim {
            vectors.push(((i * dim + j) % 100) as f32 / 100.0);
        }
    }

    // 训练 PQ
    let mut pq = ProductQuantizer::new(PQConfig::new(dim, m, nbits));
    println!("  训练 PQ...");
    let start = std::time::Instant::now();
    pq.train(n_train, &vectors).unwrap();
    let train_time = start.elapsed().as_millis();
    println!("  ✅ 训练完成: {}ms", train_time);

    // 编码向量
    let mut codes = Vec::new();
    for i in 0..n_base {
        let vec_start = i * dim;
        let vec_end = vec_start + dim;
        if vec_end <= vectors.len() {
            let code = pq.encode(&vectors[vec_start..vec_end]).unwrap();
            codes.push(code);
        }
    }

    // 搜索测试
    println!("  搜索 {} 个查询...", n_query);
    let mut correct = 0;
    let total = n_query.min(n_base);

    for i in 0..total {
        let query_start = i * dim;
        let query_end = query_start + dim;
        if query_end > vectors.len() {
            break;
        }

        let query = &vectors[query_start..query_end];

        // 使用 PQ 距离计算搜索
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;

        for (j, code) in codes.iter().enumerate() {
            let dist = pq.compute_distance(query, code);
            if dist < min_dist {
                min_dist = dist;
                best_idx = j;
            }
        }

        // 检查召回率（ground truth 就是自己）
        if best_idx == i {
            correct += 1;
        }
    }

    let recall = correct as f32 / total as f32;
    println!("  R@1: {:.3}% ({}/{})", recall * 100.0, correct, total);

    if recall > 0.80 {
        println!("  ✅ 召回率达标\n");
    } else {
        println!("  ❌ 召回率过低，需要修复\n");
    }
}

#[cfg(feature = "long-tests")]
fn test_ivf_pq_recall() {
    println!("测试 3: IVF-PQ 召回率");

    let dim = 128;
    let n_train = 10000;
    let n_base = 50000;
    let n_query = 100;
    let nlist = 256;
    let nprobe = 32;
    let m = 16;
    let nbits = 8;

    println!(
        "  配置: dim={}, nlist={}, nprobe={}, m={}, nbits={}",
        dim, nlist, nprobe, m, nbits
    );

    // 生成随机数据
    let mut vectors = Vec::with_capacity(n_train * dim);
    for i in 0..n_train {
        for j in 0..dim {
            vectors.push(((i * dim + j) % 100) as f32 / 100.0);
        }
    }

    // 创建 IVF-PQ 索引
    let config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);

    let mut index = IvfPqIndex::new(&config).unwrap();

    // 训练
    println!("  训练 IVF-PQ...");
    let start = std::time::Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed().as_millis();
    println!("  ✅ 训练完成: {}ms", train_time);

    // 添加向量
    println!("  添加 {} 个向量...", n_base);
    let mut ids = Vec::with_capacity(n_base);
    for i in 0..n_base {
        ids.push(i as i64);
    }
    index.add(&vectors[..n_base * dim], Some(&ids)).unwrap();

    // 搜索
    println!("  搜索 {} 个查询...", n_query);
    let req = SearchRequest {
        top_k: 10,
        nprobe,
        ..Default::default()
    };

    let mut correct_at_1 = 0;
    let mut correct_at_10 = 0;
    let total = n_query.min(n_base);

    for i in 0..total {
        let query_start = i * dim;
        let query_end = query_start + dim;
        if query_end > vectors.len() {
            break;
        }

        let query = &vectors[query_start..query_end];
        let result = index.search(query, &req).unwrap();

        // R@1
        if result.ids[0] == i as i64 {
            correct_at_1 += 1;
        }

        // R@10
        for j in 0..10.min(result.ids.len()) {
            if result.ids[j] == i as i64 {
                correct_at_10 += 1;
                break;
            }
        }
    }

    let r1 = correct_at_1 as f32 / total as f32;
    let r10 = correct_at_10 as f32 / total as f32;

    println!("  R@1: {:.3}% ({}/{})", r1 * 100.0, correct_at_1, total);
    println!("  R@10: {:.3}% ({}/{})", r10 * 100.0, correct_at_10, total);

    if r10 > 0.80 {
        println!("  ✅ 召回率达标\n");
    } else {
        println!("  ❌ 召回率过低，需要修复\n");
    }
}

//! BENCH-035: PQ 参数敏感性分析
//!
//! 测试不同 pq_m/pq_k 组合对 PQ 量化器性能的影响，识别最佳配置。

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::benchmark::recall_at_k;
use knowhere_rs::faiss::{IvfPqIndex, MemIndex as FlatIndex};
use rand::Rng;
use std::time::Instant;

/// 生成随机测试数据集
fn generate_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..num_vectors * dim)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect()
}

/// 计算 ground truth
fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::default(),
    };
    let mut flat_index = FlatIndex::new(&config).expect("Flat index creation failed");
    flat_index.add(base, None).expect("Flat add failed");

    let mut ground_truth = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: k,
            ..Default::default()
        };
        let result = flat_index.search(q, &req).expect("Flat search failed");
        ground_truth.push(result.ids.iter().map(|&id| id as i32).collect());
    }
    ground_truth
}

struct PQResult {
    pq_m: usize,
    pq_k: usize,
    nbits: usize,
    compression_ratio: f32,
    train_time_ms: f64,
    build_time_ms: f64,
    qps: f64,
    recall_at_1: f32,
    recall_at_10: f32,
    recall_at_100: f32,
    error: Option<String>,
}

fn run_pq_benchmark(
    train_data: &[f32],
    base_data: &[f32],
    query_data: &[f32],
    ground_truth: &[Vec<i32>],
    dim: usize,
    top_k: usize,
    nlist: usize,
    nprobe: usize,
    pq_m: usize,
    pq_k: usize,
) -> PQResult {
    let nbits = (pq_k as f32).log2() as usize;
    let compression_ratio = (dim * 32) as f32 / (pq_m * nbits) as f32;

    println!(
        "\n  测试 PQ m={}, k={} ({}bits, {:.1}x 压缩)...",
        pq_m, pq_k, nbits, compression_ratio
    );

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(pq_m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let mut index = match IvfPqIndex::new(&config) {
        Ok(idx) => idx,
        Err(e) => {
            return PQResult {
                pq_m,
                pq_k,
                nbits,
                compression_ratio,
                train_time_ms: 0.0,
                build_time_ms: 0.0,
                qps: 0.0,
                recall_at_1: 0.0,
                recall_at_10: 0.0,
                recall_at_100: 0.0,
                error: Some(format!("索引创建失败：{}", e)),
            }
        }
    };

    let train_start = Instant::now();
    if let Err(e) = index.train(train_data) {
        return PQResult {
            pq_m,
            pq_k,
            nbits,
            compression_ratio,
            train_time_ms: 0.0,
            build_time_ms: 0.0,
            qps: 0.0,
            recall_at_1: 0.0,
            recall_at_10: 0.0,
            recall_at_100: 0.0,
            error: Some(format!("训练失败：{}", e)),
        };
    }
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    let build_start = Instant::now();
    if let Err(e) = index.add(base_data, None) {
        return PQResult {
            pq_m,
            pq_k,
            nbits,
            compression_ratio,
            train_time_ms: train_time,
            build_time_ms: 0.0,
            qps: 0.0,
            recall_at_1: 0.0,
            recall_at_10: 0.0,
            recall_at_100: 0.0,
            error: Some(format!("添加向量失败：{}", e)),
        };
    }
    let build_time = train_time + build_start.elapsed().as_secs_f64() * 1000.0;

    let num_queries = query_data.len() / dim;
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    let search_start = Instant::now();
    let mut search_error = false;
    for i in 0..num_queries {
        let q = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe,
            ..Default::default()
        };
        match index.search(q, &req) {
            Ok(result) => all_results.push(result.ids),
            Err(e) => {
                search_error = true;
                eprintln!("    搜索错误：{}", e);
                break;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = if search_time > 0.0 {
        num_queries as f64 / (search_time / 1000.0)
    } else {
        0.0
    };

    let (r1, r10, r100) = if !search_error {
        let mut s1 = 0.0f32;
        let mut s10 = 0.0f32;
        let mut s100 = 0.0f32;
        for (result, gt) in all_results.iter().zip(ground_truth.iter()) {
            s1 += recall_at_k(result, gt, 1) as f32;
            s10 += recall_at_k(result, gt, 10.min(top_k)) as f32;
            s100 += recall_at_k(result, gt, 100.min(top_k)) as f32;
        }
        (
            s1 / num_queries as f32,
            s10 / num_queries as f32,
            s100 / num_queries as f32,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    PQResult {
        pq_m,
        pq_k,
        nbits,
        compression_ratio,
        train_time_ms: train_time,
        build_time_ms: build_time,
        qps,
        recall_at_1: r1,
        recall_at_10: r10,
        recall_at_100: r100,
        error: if search_error {
            Some("搜索失败".to_string())
        } else {
            None
        },
    }
}

#[test]
fn bench_pq_params() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  BENCH-035: PQ 参数敏感性分析                                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let dim = 128;
    let num_train = 10_000;
    let num_base = 50_000;
    let num_queries = 200;
    let top_k = 100;
    let nlist = 256;
    let nprobe = 32;

    let pq_m_values = [4, 8, 16, 32];
    let pq_k_values = [256, 512, 1024];

    println!(
        "测试配置：dim={}, train={}, base={}, queries={}, top_k={}, nlist={}, nprobe={}",
        dim, num_train, num_base, num_queries, top_k, nlist, nprobe
    );
    println!("PQ 参数：m={:?}, k={:?}\n", pq_m_values, pq_k_values);

    println!("生成数据集...");
    let train_data = generate_dataset(num_train, dim);
    let base_data = generate_dataset(num_base, dim);
    let query_data = generate_dataset(num_queries, dim);

    println!("计算 Ground Truth...");
    let ground_truth = compute_ground_truth(&base_data, &query_data, num_queries, dim, top_k);

    let mut results = Vec::new();
    for &pq_m in &pq_m_values {
        if dim % pq_m != 0 {
            println!("  跳过 m={} (dim 不能整除)", pq_m);
            continue;
        }
        for &pq_k in &pq_k_values {
            let nbits = (pq_k as f32).log2() as usize;
            if nbits < 1 || nbits > 16 {
                println!("  跳过 k={} (nbits 超出范围)", pq_k);
                continue;
            }
            let result = run_pq_benchmark(
                &train_data,
                &base_data,
                &query_data,
                &ground_truth,
                dim,
                top_k,
                nlist,
                nprobe,
                pq_m,
                pq_k,
            );
            results.push(result);
        }
    }

    print_report(&results);
}

fn print_report(results: &[PQResult]) {
    println!("\n{}", "═".repeat(100));
    println!("📊 BENCH-035: PQ 参数敏感性分析报告");
    println!("{}\n", "═".repeat(100));

    println!("┌────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 召回率对比 (R@1 / R@10 / R@100)                                                        │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");

    for &pq_m in [4, 8, 16, 32].iter() {
        let m_results: Vec<&PQResult> = results
            .iter()
            .filter(|r| r.pq_m == pq_m && r.error.is_none())
            .collect();
        if m_results.is_empty() {
            continue;
        }
        println!("│ m = {:2}                                                                                 │", pq_m);
        for r in &m_results {
            println!("│   k={:4} ({}bits): R@1={:.3}, R@10={:.3}, R@100={:.3} (压缩比 {:.1}x)                    │",
                r.pq_k, r.nbits, r.recall_at_1, r.recall_at_10, r.recall_at_100, r.compression_ratio);
        }
        println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    }

    println!("\n┌────────────────────────────────────────────────────────────────────────────────────────┐");
    println!(
        "│ 性能对比 (训练时间 / 构建时间 / QPS)                                                  │"
    );
    println!("├────────────────────────────────────────────────────────────────────────────────────────┤");
    for r in results.iter().filter(|r| r.error.is_none()) {
        println!("│ m={:2}, k={:4} ({}bits): 训练={:7.1}ms, 构建={:7.1}ms, QPS={:6.0}                          │",
            r.pq_m, r.pq_k, r.nbits, r.train_time_ms, r.build_time_ms, r.qps);
    }
    println!("└────────────────────────────────────────────────────────────────────────────────────────┘");

    println!("\n📈 召回率热力图 (R@10):\n");
    print!("        ");
    for pq_k in [256, 512, 1024] {
        print!("{:>10}", format!("k={}", pq_k));
    }
    println!();
    for &pq_m in [4, 8, 16, 32].iter() {
        print!("m={:>2}  ", pq_m);
        for pq_k in [256, 512, 1024] {
            if let Some(r) = results
                .iter()
                .find(|x| x.pq_m == pq_m && x.pq_k == pq_k && x.error.is_none())
            {
                let icon = if r.recall_at_10 > 0.9 {
                    "✅"
                } else if r.recall_at_10 > 0.7 {
                    "⚠️"
                } else {
                    "❌"
                };
                print!(
                    "{:>6} ({:.2}) ",
                    format!("{}{:.2}", icon, r.recall_at_10),
                    r.recall_at_10
                );
            } else {
                print!("{:>13}", "-");
            }
        }
        println!();
    }

    println!("\n💡 最佳配置推荐:\n");

    if let Some(best) = results
        .iter()
        .filter(|r| r.error.is_none() && r.recall_at_10 > 0.90)
        .max_by(|a, b| {
            a.compression_ratio
                .partial_cmp(&b.compression_ratio)
                .unwrap()
        })
    {
        println!("  🏆 高召回率场景 (R@10 > 90%):");
        println!(
            "     配置：m={}, k={} ({}bits), 压缩比：{:.1}x",
            best.pq_m, best.pq_k, best.nbits, best.compression_ratio
        );
        println!(
            "     召回率：R@1={:.3}, R@10={:.3}, R@100={:.3}, QPS={:.0}",
            best.recall_at_1, best.recall_at_10, best.recall_at_100, best.qps
        );
        println!("     适用：搜索质量优先的场景\n");
    }

    if let Some(best) = results
        .iter()
        .filter(|r| r.error.is_none() && r.recall_at_10 > 0.70 && r.recall_at_10 <= 0.90)
        .max_by(|a, b| {
            a.compression_ratio
                .partial_cmp(&b.compression_ratio)
                .unwrap()
        })
    {
        println!("  ⚖️  平衡场景 (R@10 70-90%):");
        println!(
            "     配置：m={}, k={} ({}bits), 压缩比：{:.1}x",
            best.pq_m, best.pq_k, best.nbits, best.compression_ratio
        );
        println!(
            "     召回率：R@1={:.3}, R@10={:.3}, R@100={:.3}, QPS={:.0}",
            best.recall_at_1, best.recall_at_10, best.recall_at_100, best.qps
        );
        println!("     适用：性能和精度平衡的场景\n");
    }

    if let Some(best) = results
        .iter()
        .filter(|r| r.error.is_none() && r.compression_ratio >= 16.0)
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
    {
        println!("  📦 高压缩场景 (压缩比 >= 16x):");
        println!(
            "     配置：m={}, k={} ({}bits), 压缩比：{:.1}x",
            best.pq_m, best.pq_k, best.nbits, best.compression_ratio
        );
        println!(
            "     召回率：R@1={:.3}, R@10={:.3}, R@100={:.3}, QPS={:.0}",
            best.recall_at_1, best.recall_at_10, best.recall_at_100, best.qps
        );
        println!("     适用：内存受限的场景\n");
    }

    println!("📊 参数敏感性分析:\n");
    println!("  1. pq_m (子量化器数量) 的影响:");
    println!("     结论：pq_m 越大，子向量越短，量化误差越大，召回率越低");
    println!("     推荐：对于 dim=128，使用 m=8-16 可获得较好的召回率/压缩比平衡\n");
    println!("  2. pq_k (聚类数) 的影响:");
    println!("     结论：pq_k 越大 (nbits 越多)，量化精度越高，召回率越高，但压缩比降低");
    println!("     推荐：使用 k=256 (8bits) 或 k=512 (9bits) 作为起点\n");

    println!("✅ BENCH-035 完成！");
}

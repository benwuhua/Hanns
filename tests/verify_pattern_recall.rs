//! 验证 Pattern 数据 vs Random 数据的召回率
//!
//! 目的：确认 Pattern 数据是否因向量重复导致虚假的高性能

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::{IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIM: usize = 128;
const NLIST: usize = 100;
const NPROBE: usize = 16;
const NQ: usize = 100;
const TOP_K: usize = 10;

fn gen_pattern_data(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| ((i % 100) as f32 / 100.0 - 0.5) * 2.0)
        .collect()
}

fn gen_random_data(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn test_recall(name: &str, nbase: usize, data: &[f32], queries: &[f32]) {
    let dim = DIM;

    // Ground Truth (Flat index)
    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, dim);
    let mut flat = FlatIndex::new(&flat_cfg).unwrap();
    flat.add(data, None).unwrap();

    let mut gt_ids: Vec<Vec<i32>> = Vec::with_capacity(NQ);
    for i in 0..NQ {
        let q = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: TOP_K,
            ..Default::default()
        };
        let gt = flat.search(q, &req).unwrap();
        gt_ids.push(gt.ids.iter().map(|&x| x as i32).collect());
    }

    // IVF-Flat
    let ivf_cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams::ivf(NLIST, NPROBE),
    };

    let mut ivf = IvfFlatIndex::new(&ivf_cfg).unwrap();
    ivf.train(data).unwrap();
    ivf.add(data, None).unwrap();

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: NPROBE,
        ..Default::default()
    };

    let mut pred_ids: Vec<Vec<i64>> = Vec::with_capacity(NQ);
    for i in 0..NQ {
        let q = &queries[i * dim..(i + 1) * dim];
        let ret = ivf.search(q, &req).unwrap();
        pred_ids.push(ret.ids);
    }

    let recall = average_recall_at_k(&pred_ids, &gt_ids, TOP_K);

    println!("{}: R@{} = {:.4}", name, TOP_K, recall);
}

#[test]
fn verify_recall_comparison() {
    println!("\n========================================");
    println!("Pattern vs Random 数据召回率对比");
    println!("========================================");

    // 10K
    println!("\n--- 10K Scale ---");
    let pattern_10k = gen_pattern_data(10_000, DIM);
    let random_10k = gen_random_data(42, 10_000, DIM);
    let queries_pattern = gen_pattern_data(1000, DIM); // 从 pattern 数据中取查询
    let queries_random = gen_random_data(999_999, 100, DIM);

    test_recall(
        "Pattern 10K (pattern queries)",
        10_000,
        &pattern_10k,
        &queries_pattern[0..100 * DIM],
    );
    test_recall(
        "Random 10K (random queries)",
        10_000,
        &random_10k,
        &queries_random,
    );

    // 50K
    println!("\n--- 50K Scale ---");
    let pattern_50k = gen_pattern_data(50_000, DIM);
    let random_50k = gen_random_data(42, 50_000, DIM);
    let queries_pattern_50k = gen_pattern_data(1000, DIM);
    let queries_random_50k = gen_random_data(999_999, 100, DIM);

    test_recall(
        "Pattern 50K (pattern queries)",
        50_000,
        &pattern_50k,
        &queries_pattern_50k[0..100 * DIM],
    );
    test_recall(
        "Random 50K (random queries)",
        50_000,
        &random_50k,
        &queries_random_50k,
    );

    println!("\n========================================");
    println!("结论:");
    println!("如果 Pattern 数据召回率显著低于 Random 数据,");
    println!("说明 Pattern 数据的高性能是虚假的（因向量重复）。");
    println!("========================================");
}

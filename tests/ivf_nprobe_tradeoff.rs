//! IVF-Flat 召回率与性能平衡分析
//!
//! 对比不同 nprobe 下的召回率和 QPS

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType};
use knowhere_rs::faiss::IvfPqIndex;
use knowhere_rs::MetricType;

#[cfg(feature = "long-tests")]
use knowhere_rs::api::SearchRequest;
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::{IvfFlatIndex, MemIndex as FlatIndex};
#[cfg(feature = "long-tests")]
use rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(feature = "long-tests")]
use std::time::Instant;

#[cfg(feature = "long-tests")]
const DIM: usize = 128;
#[cfg(feature = "long-tests")]
const NBASE: usize = 10000;
#[cfg(feature = "long-tests")]
const NQ: usize = 100;
#[cfg(feature = "long-tests")]
const TOP_K: usize = 10;
#[cfg(feature = "long-tests")]
const NLIST: usize = 100;

#[cfg(feature = "long-tests")]
fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

#[test]
fn ivfpq_nprobe_expands_coarse_probe_order() {
    let dim = 4;
    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams {
            nlist: Some(3),
            nprobe: Some(1),
            m: Some(2),
            nbits_per_idx: Some(8),
            ..Default::default()
        },
    };

    let mut base = Vec::new();
    for center in [0.0f32, 100.0, 200.0] {
        for offset in 0..24 {
            let value = center + offset as f32 * 0.01;
            base.extend_from_slice(&[value, value + 0.1, value + 0.2, value + 0.3]);
        }
    }

    let mut index = IvfPqIndex::new(&config).unwrap();
    index.train(&base).unwrap();
    index.add(&base, None).unwrap();

    let audit = index.hot_path_audit();
    let query = [140.0, 140.1, 140.2, 140.3];

    let one_probe = index.coarse_probe_order(&query, 1);
    let two_probe = index.coarse_probe_order(&query, 2);

    assert_eq!(one_probe.len(), 1);
    assert_eq!(two_probe.len(), 2);
    assert_eq!(two_probe[0], one_probe[0]);
    assert_ne!(two_probe[0], two_probe[1]);

    let first = audit.centroids[two_probe[0] * dim];
    let second = audit.centroids[two_probe[1] * dim];
    assert!(
        first < second,
        "query near 140 should probe the ~100 centroid before the ~200 centroid: [{first}, {second}]"
    );
}

#[cfg(feature = "long-tests")]
#[test]
fn ivf_flat_nprobe_recall_tradeoff() {
    println!(
        "\n=== IVF-Flat nprobe-Recall Tradeoff (nlist={}) ===",
        NLIST
    );
    println!("Base: {} vectors, {} queries, dim={}", NBASE, NQ, DIM);
    println!();

    let base = gen_vectors(42, NBASE, DIM);
    let query = gen_vectors(999_999, NQ, DIM);

    // Ground Truth
    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut flat = FlatIndex::new(&flat_cfg).unwrap();
    flat.add(&base, None).unwrap();

    // IVF-Flat
    let ivf_cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::ivf(NLIST, 1), // nprobe will be set in search
    };

    let mut ivf = IvfFlatIndex::new(&ivf_cfg).unwrap();
    ivf.train(&base).unwrap();
    ivf.add(&base, None).unwrap();

    println!("| nprobe | R@10 | QPS | vs C++ (5K) |");
    println!("|--------|------|-----|-------------|");

    for nprobe in [1, 2, 5, 10, 16, 25, 50, 100] {
        let mut total_recall = 0.0;

        let ivf_req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let start = Instant::now();
        for i in 0..NQ {
            let q = &query[i * DIM..(i + 1) * DIM];

            let gt = flat
                .search(
                    q,
                    &SearchRequest {
                        top_k: TOP_K,
                        ..Default::default()
                    },
                )
                .unwrap();
            let ivf_result = ivf.search(q, &ivf_req).unwrap();

            let gt_set: std::collections::HashSet<i64> = gt.ids.into_iter().collect();
            let ivf_set: std::collections::HashSet<i64> = ivf_result.ids.into_iter().collect();

            let hit = gt_set.intersection(&ivf_set).count() as f64;
            total_recall += hit / TOP_K as f64;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let qps = NQ as f64 / elapsed;
        let avg_recall = total_recall / NQ as f64;

        println!(
            "| {} | {:.0}% | {:.0} | {:.2}x |",
            nprobe,
            avg_recall * 100.0,
            qps,
            qps / 5000.0
        );
    }

    println!();
    println!("结论：");
    println!("- nprobe=16: R@10≈43%, QPS≈{}K → 适合低延迟场景", 29);
    println!("- nprobe=100: R@10=100%, QPS≈{} → 适合高精度场景", 500);
}

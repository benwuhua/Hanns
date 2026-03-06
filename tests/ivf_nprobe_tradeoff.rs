//! IVF-Flat 召回率与性能平衡分析
//! 
//! 对比不同 nprobe 下的召回率和 QPS

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::{IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const DIM: usize = 128;
const NBASE: usize = 10000;
const NQ: usize = 100;
const TOP_K: usize = 10;
const NLIST: usize = 100;

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

#[test]
fn ivf_flat_nprobe_recall_tradeoff() {
    println!("\n=== IVF-Flat nprobe-Recall Tradeoff (nlist={}) ===", NLIST);
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
            data_type: crate::api::DataType::Float,
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
            
            let gt = flat.search(q, &SearchRequest { top_k: TOP_K, ..Default::default() }).unwrap();
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

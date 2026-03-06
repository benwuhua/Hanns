//! 诊断 IVF-Flat 召回率问题

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::{IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIM: usize = 128;
const NBASE: usize = 1000;
const NQ: usize = 10;
const TOP_K: usize = 10;

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

#[test]
fn diagnose_ivf_recall() {
    println!("\n=== IVF-Flat 召回率诊断 ===");
    
    let base = gen_vectors(42, NBASE, DIM);
    let query = gen_vectors(999, NQ, DIM);

    // Flat 索引作为 Ground Truth
    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut flat = FlatIndex::new(&flat_cfg).unwrap();
    flat.add(&base, None).unwrap();

    // IVF-Flat 索引
    for nlist in [10, 50, 100] {
        for nprobe in [1, 2, 5, 10, nlist] {
            let ivf_cfg = IndexConfig {
                index_type: IndexType::IvfFlat,
                dim: DIM,
                metric_type: MetricType::L2,
                data_type: knowhere_rs::api::DataType::Float,
                params: IndexParams::ivf(nlist, nprobe),
            };

            let mut ivf = IvfFlatIndex::new(&ivf_cfg).unwrap();
            ivf.train(&base).unwrap();
            ivf.add(&base, None).unwrap();

            let mut total_recall = 0.0;
            
            for i in 0..NQ {
                let q = &query[i * DIM..(i + 1) * DIM];
                
                // Ground truth
                let gt_req = SearchRequest {
                    top_k: TOP_K,
                    ..Default::default()
                };
                let gt = flat.search(q, &gt_req).unwrap();
                
                // IVF 搜索
                let ivf_req = SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    ..Default::default()
                };
                let ivf_result = ivf.search(q, &ivf_req).unwrap();

                // 计算召回率
                let gt_set: std::collections::HashSet<i64> = gt.ids.into_iter().collect();
                let ivf_set: std::collections::HashSet<i64> = ivf_result.ids.into_iter().collect();
                
                let hit = gt_set.intersection(&ivf_set).count() as f64;
                total_recall += hit / TOP_K as f64;
            }

            let avg_recall = total_recall / NQ as f64;
            println!(
                "nlist={}, nprobe={}: R@{} = {:.2}%",
                nlist, nprobe, TOP_K, avg_recall * 100.0
            );
        }
        println!();
    }
}

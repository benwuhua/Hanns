use std::collections::HashSet;
use std::fs;

use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfRaBitqConfig, IvfRaBitqIndex, MemIndex};
use knowhere_rs::quantization::RefineType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 100_000;
const DIM: usize = 128;
const NQ: usize = 1000;
const TOP_K: usize = 10;
const NLIST: usize = 256;
const NPROBE: usize = 256;
const SEED: u64 = 42;
const STATUS_PATH: &str = "/tmp/codex_status_b.txt";

const REFINE_KS: &[usize] = &[10, 40, 100, 200, 500, 1000];

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn overlap_at_k(a: &[i64], b: &[i64]) -> usize {
    let gt_set: HashSet<i64> = a.iter().copied().collect();
    b.iter().filter(|&&id| gt_set.contains(&id)).count()
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-RaBitQ refine_k sweep (n=100K, nprobe=256, full scan) ===");

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let queries = gen_vectors(&mut rng, NQ, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(&base, None)?;
    let gt_req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };
    let gt_top10: Vec<Vec<i64>> = (0..NQ)
        .map(|i| {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let res = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, knowhere_rs::api::KnowhereError>(res.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut best_recall = -1.0f64;
    let mut best_refine_k = 0usize;

    for &refine_k in REFINE_KS {
        let cfg = IvfRaBitqConfig::new(DIM, NLIST)
            .with_metric(MetricType::L2)
            .with_refine(RefineType::DataView, refine_k);
        let mut index = IvfRaBitqIndex::new(cfg);
        index.train(&base)?;
        index.add(&base, None)?;

        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: NPROBE,
            ..Default::default()
        };

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            let hit = overlap_at_k(gt_ids, &res.ids) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = round3(recall_sum / NQ as f64);
        println!("refine_k={:>5}  recall@10={:.3}", refine_k, recall);

        if recall > best_recall {
            best_recall = recall;
            best_refine_k = refine_k;
        }
    }

    fs::write(
        STATUS_PATH,
        format!(
            "DONE: max_recall={:.3} at refine_k={}\n",
            best_recall, best_refine_k
        ),
    )?;

    Ok(())
}

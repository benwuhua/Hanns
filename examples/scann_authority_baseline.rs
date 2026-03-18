use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{MemIndex, ScaNNConfig, ScaNNIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 100_000;
const DIM: usize = 128;
const SEED: u64 = 42;
const TOP_K: usize = 10;
const NUM_PARTITIONS: usize = 16;
const NUM_CENTROIDS: usize = 256;
const RECALL_QUERIES: usize = 1_000;
const QPS_QUERIES: usize = 1_000;
const WARMUP_QUERIES: usize = 200;
const REORDER_K_SWEEP: [usize; 3] = [400, 800, 1600];
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn overlap_at_k(gt: &[i64], pred: &[i64]) -> usize {
    let gt_set: HashSet<i64> = gt.iter().copied().collect();
    pred.iter().filter(|&&id| gt_set.contains(&id)).count()
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn run() -> Result<(f64, u64), Box<dyn std::error::Error>> {
    println!("=== ScaNN authority baseline ===");
    println!(
        "base={} dim={} seed={} partitions={} centroids={} recall_q={} qps_q={} warmup={}",
        BASE_SIZE,
        DIM,
        SEED,
        NUM_PARTITIONS,
        NUM_CENTROIDS,
        RECALL_QUERIES,
        QPS_QUERIES,
        WARMUP_QUERIES
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let recall_queries = gen_vectors(&mut rng, RECALL_QUERIES, DIM);
    let qps_queries = gen_vectors(&mut rng, QPS_QUERIES, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(&base, None)?;

    let gt_req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };

    let gt_topk: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let gt = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, knowhere_rs::api::KnowhereError>(gt.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("reorder_k | recall@10 | qps");
    println!("----------|-----------|-----");

    let mut recall_1600 = 0.0f64;
    let mut qps_1600 = 0u64;

    for &reorder_k in &REORDER_K_SWEEP {
        let cfg = ScaNNConfig::new(NUM_PARTITIONS, NUM_CENTROIDS, reorder_k);
        let mut index = ScaNNIndex::new(DIM, cfg)?;
        index.train(&base, Some(&recall_queries));
        let _ = index.add(&base, None);

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_topk.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, TOP_K);
            let ids: Vec<i64> = res.into_iter().map(|(id, _)| id).collect();
            recall_sum += overlap_at_k(gt_ids, &ids) as f64 / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        for i in 0..WARMUP_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, TOP_K);
        }

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, TOP_K);
        }
        let elapsed = start.elapsed();
        let qps = (QPS_QUERIES as f64 / elapsed.as_secs_f64()).round() as u64;

        println!("{:>9} | {:>9.3} | {}", reorder_k, recall, qps);

        if reorder_k == 1600 {
            recall_1600 = recall;
            qps_1600 = qps;
        }
    }

    Ok((recall_1600, qps_1600))
}

fn main() {
    match run() {
        Ok((recall, qps)) => {
            let status = format!("DONE: recall@10={:.3}, QPS={} (reorder_k=1600)\n", recall, qps);
            if let Err(err) = fs::write(STATUS_FILE, status) {
                eprintln!("failed to write status file: {}", err);
            }
        }
        Err(err) => {
            let _ = fs::write(STATUS_FILE, format!("ERROR: {}\n", err));
            eprintln!("{}", err);
            std::process::exit(1);
        }
    }
}

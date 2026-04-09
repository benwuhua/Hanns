use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use hanns::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use hanns::faiss::{MemIndex, ScaNNConfig, ScaNNIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 100_000;
const DIM: usize = 128;
const SEED: u64 = 42;
const RECALL_QUERIES: usize = 1_000;
const QPS_QUERIES: usize = 200;
const TOP_K: usize = 10;
const NUM_PARTITIONS: usize = 16;
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

#[derive(Clone, Copy)]
struct SweepCase {
    centroids: usize,
    reorder_k: usize,
}

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
    println!("=== ScaNN large sweep (n=100K, full scan) ===");

    let cases: Vec<SweepCase> = vec![
        SweepCase {
            centroids: 256,
            reorder_k: 200,
        },
        SweepCase {
            centroids: 256,
            reorder_k: 400,
        },
        SweepCase {
            centroids: 256,
            reorder_k: 800,
        },
        SweepCase {
            centroids: 256,
            reorder_k: 1600,
        },
        SweepCase {
            centroids: 512,
            reorder_k: 200,
        },
        SweepCase {
            centroids: 512,
            reorder_k: 400,
        },
        SweepCase {
            centroids: 512,
            reorder_k: 800,
        },
        SweepCase {
            centroids: 1024,
            reorder_k: 200,
        },
        SweepCase {
            centroids: 1024,
            reorder_k: 400,
        },
    ];

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
    let gt_top10: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let gt = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, hanns::api::KnowhereError>(gt.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut best_recall = -1.0f64;
    let mut best_centroids = 0usize;
    let mut best_reorder_k = 0usize;

    for case in cases {
        let config = ScaNNConfig::new(NUM_PARTITIONS, case.centroids, case.reorder_k);
        let mut index = ScaNNIndex::new(DIM, config)?;
        index.train(&base, Some(&recall_queries));
        let _ = index.add(&base, None);

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, TOP_K);
            let ids: Vec<i64> = res.into_iter().map(|(id, _)| id).collect();
            let hit = overlap_at_k(gt_ids, &ids) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        // Warmup before timed QPS
        for i in 0..QPS_QUERIES {
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

        println!(
            "centroids={}  reorder_k={:>4}  recall@10={:.3}  qps={}",
            case.centroids, case.reorder_k, recall, qps
        );

        if recall > best_recall {
            best_recall = recall;
            best_centroids = case.centroids;
            best_reorder_k = case.reorder_k;
        }
    }

    fs::write(
        STATUS_FILE,
        format!(
            "DONE: max_recall={:.3} config=centroids={},reorder_k={}\n",
            best_recall, best_centroids, best_reorder_k
        ),
    )?;

    Ok(())
}

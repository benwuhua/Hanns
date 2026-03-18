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
const QPS_QUERIES: usize = 1_000;
const RECALL_QUERIES: usize = 100;
const TOP_K: usize = 10;
const NUM_PARTITIONS: usize = 16;
const NUM_CENTROIDS: usize = 256;
const REORDER_K_SWEEP: [usize; 5] = [10, 20, 40, 80, 160];
const OUT_JSON: &str = "/tmp/scann_sweep_result.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

#[derive(Debug, Clone)]
struct SweepRow {
    reorder_k: usize,
    recall_at_10: f64,
    qps: u64,
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
    println!("=== ScaNN reorder_k sweep ===");
    println!(
        "base={} dim={} metric=L2 seed={} partitions={} centroids={}",
        BASE_SIZE, DIM, SEED, NUM_PARTITIONS, NUM_CENTROIDS
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let qps_queries = gen_vectors(&mut rng, QPS_QUERIES, DIM);
    let recall_queries = gen_vectors(&mut rng, RECALL_QUERIES, DIM);

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
            Ok::<Vec<i64>, knowhere_rs::api::KnowhereError>(gt.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("reorder_k | recall@10 | qps");
    println!("----------|-----------|------");

    let mut rows = Vec::with_capacity(REORDER_K_SWEEP.len());

    for &reorder_k in &REORDER_K_SWEEP {
        let config = ScaNNConfig::new(NUM_PARTITIONS, NUM_CENTROIDS, reorder_k);
        let mut index = ScaNNIndex::new(DIM, config)?;
        index.train(&base, Some(&recall_queries));
        let _ = index.add(&base, None);

        let qps_start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, TOP_K);
        }
        let qps_secs = qps_start.elapsed().as_secs_f64();
        let qps = (QPS_QUERIES as f64 / qps_secs).round() as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, TOP_K);
            let ids: Vec<i64> = res.into_iter().map(|(id, _)| id).collect();
            let hit = overlap_at_k(gt_ids, &ids) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("{:>9} | {:>9.3} | {}", reorder_k, recall, qps);
        rows.push(SweepRow {
            reorder_k,
            recall_at_10: recall,
            qps,
        });
    }

    let mut best: Option<&SweepRow> = None;
    for row in &rows {
        if best.is_none_or(|b| row.recall_at_10 > b.recall_at_10) {
            best = Some(row);
        }
    }

    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "reorder_k": r.reorder_k,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    fs::write(OUT_JSON, serde_json::to_string_pretty(&rows_json)?)?;
    println!("saved {}", OUT_JSON);

    if let Some(best_row) = best {
        let status = format!(
            "DONE: scann sweep done. max recall: {:.3}, qps at max recall: {}. results /tmp/scann_sweep_result.json\n",
            best_row.recall_at_10, best_row.qps
        );
        fs::write(STATUS_FILE, status)?;
    } else {
        fs::write(
            STATUS_FILE,
            "DONE: scann sweep done. max recall: 0.000, qps at max recall: 0. results /tmp/scann_sweep_result.json\n",
        )?;
    }

    Ok(())
}

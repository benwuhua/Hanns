use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N: usize = 100_000;
const DIM: usize = 128;
const TOP_K: usize = 10;
const QPS_QUERIES: usize = 2_000;
const RECALL_QUERIES: usize = 200;
const SEED: u64 = 42;
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const LEVEL_MULTIPLIER: f32 = 0.5;
const EF_SWEEP: [usize; 7] = [16, 32, 50, 64, 100, 128, 200];
const OUT_JSON: &str = "/tmp/hnsw_ef_sweep_result.json";

#[derive(Clone, Debug)]
struct SweepRow {
    ef: usize,
    recall_at_10: f64,
    qps: u64,
}

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn overlap_at_k(gt: &[i64], got: &[i64], k: usize) -> usize {
    let set: HashSet<i64> = gt.iter().take(k).copied().collect();
    got.iter().take(k).filter(|&&id| set.contains(&id)).count()
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HNSW ef sweep baseline ===");
    println!(
        "n={} dim={} metric=L2 seed={} m={} ef_construction={}",
        N, DIM, SEED, M, EF_CONSTRUCTION
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, N, DIM);
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
            let res = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, hanns::api::KnowhereError>(res.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut params = IndexParams::hnsw(EF_CONSTRUCTION, 64, LEVEL_MULTIPLIER);
    params.m = Some(M);
    let hnsw_cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = HnswIndex::new(&hnsw_cfg)?;
    index.train(&base)?;
    index.add(&base, None)?;

    let mut rows = Vec::with_capacity(EF_SWEEP.len());
    for &ef in &EF_SWEEP {
        index.set_ef_search(ef);
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            params: Some(format!(r#"{{"ef": {ef}}}"#)),
            ..Default::default()
        };

        let qps_start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, &req)?;
        }
        let qps = (QPS_QUERIES as f64 / qps_start.elapsed().as_secs_f64())
            .round()
            .max(0.0) as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            let hit = overlap_at_k(gt_ids, &res.ids, TOP_K) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("ef={:>4} recall@10={:.3} qps={}", ef, recall, qps);
        rows.push(SweepRow {
            ef,
            recall_at_10: recall,
            qps,
        });
    }

    let json_rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "ef": r.ef,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    fs::write(OUT_JSON, serde_json::to_string_pretty(&json_rows)?)?;
    println!("saved {}", OUT_JSON);

    Ok(())
}

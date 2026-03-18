use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfSq8Index, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 100_000;
const DIM: usize = 128;
const SEED: u64 = 42;
const NLIST: usize = 256;
const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 200;
const QPS_QUERIES: usize = 1_000;
const NPROBE_SWEEP: [usize; 6] = [8, 16, 32, 64, 128, 256];
const RECALL_GATE: f64 = 0.95;
const OUT_JSON: &str = "/tmp/ivf_sq8_authority_result.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

#[derive(Debug, Clone)]
struct SweepRow {
    nprobe: usize,
    recall_at_10: f64,
    qps: u64,
}

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

fn run() -> Result<Option<(usize, u64)>, Box<dyn std::error::Error>> {
    println!("=== IVF-SQ8 authority baseline (100K) ===");
    println!(
        "base={} dim={} metric=L2 seed={} nlist={} top_k={} recall_q={} qps_q={}",
        BASE_SIZE, DIM, SEED, NLIST, TOP_K, RECALL_QUERIES, QPS_QUERIES
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let recall_queries = gen_vectors(&mut rng, RECALL_QUERIES, DIM);
    let qps_queries = gen_vectors(&mut rng, QPS_QUERIES, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(&base, None)?;

    let ivf_cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams::ivf(NLIST, 8),
    };
    let mut index = IvfSq8Index::new(&ivf_cfg)?;
    index.train(&base)?;
    index.add(&base, None)?;

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

    let mut rows = Vec::with_capacity(NPROBE_SWEEP.len());

    for &nprobe in &NPROBE_SWEEP {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_topk.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            let hit = overlap_at_k(gt_ids, &res.ids) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, &req)?;
        }
        let elapsed = start.elapsed();
        let qps = (QPS_QUERIES as f64 / elapsed.as_secs_f64()).round() as u64;

        println!("nprobe={:>3} recall@10={:.3} qps={}", nprobe, recall, qps);

        rows.push(SweepRow {
            nprobe,
            recall_at_10: recall,
            qps,
        });
    }

    let min_gate = rows.iter().find(|row| row.recall_at_10 >= RECALL_GATE);
    if let Some(row) = min_gate {
        println!(
            "summary: min nprobe for recall>=0.95 is {} (qps={})",
            row.nprobe, row.qps
        );
    } else {
        println!("summary: min nprobe for recall>=0.95 is N/A (qps=N/A)");
    }

    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "nprobe": r.nprobe,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    fs::write(OUT_JSON, serde_json::to_string_pretty(&rows_json)?)?;

    Ok(min_gate.map(|r| (r.nprobe, r.qps)))
}

fn main() {
    match run() {
        Ok(Some((nprobe, qps))) => {
            let status = format!(
                "DONE: ivf_sq8_authority_baseline created and runs. min_nprobe_for_0.95={} qps={}\n",
                nprobe, qps
            );
            if let Err(err) = fs::write(STATUS_FILE, status) {
                eprintln!("failed to write status file: {}", err);
            }
        }
        Ok(None) => {
            let status =
                "DONE: ivf_sq8_authority_baseline created and runs. min_nprobe_for_0.95=N/A qps=N/A\n";
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

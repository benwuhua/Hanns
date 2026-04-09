use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{IvfFlatIndex, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 100_000;
const DIM: usize = 128;
const SEED: u64 = 42;
const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 200;
const QPS_QUERIES: usize = 2_000;
const NLIST_SWEEP: [usize; 3] = [64, 128, 256];
const NPROBE_SWEEP: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const OUT_JSON: &str = "/tmp/ivf_flat_recall_vs_nprobe.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

#[derive(Debug, Clone)]
struct SweepRow {
    nlist: usize,
    nprobe: usize,
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

fn run() -> Result<Vec<SweepRow>, Box<dyn std::error::Error>> {
    println!("=== IVF-Flat recall vs nprobe (SIFT-like random 100K) ===");
    println!(
        "base={} dim={} metric=L2 seed={} top_k={} recall_q={} qps_q={}",
        BASE_SIZE, DIM, SEED, TOP_K, RECALL_QUERIES, QPS_QUERIES
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
    let gt_top10: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let gt = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, hanns::api::KnowhereError>(gt.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut rows = Vec::new();

    for &nlist in &NLIST_SWEEP {
        println!("=== nlist={} ===", nlist);

        let ivf_cfg = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            data_type: DataType::Float,
            dim: DIM,
            params: IndexParams::ivf(nlist, 8),
        };
        let mut index = IvfFlatIndex::new(&ivf_cfg)?;
        index.train(&base)?;
        index.add(&base, None)?;

        for &nprobe in &NPROBE_SWEEP {
            if nprobe > nlist {
                continue;
            }

            let req = SearchRequest {
                top_k: TOP_K,
                nprobe,
                ..Default::default()
            };

            let mut recall_sum = 0.0f64;
            for (i, gt_ids) in gt_top10.iter().enumerate() {
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
                nlist,
                nprobe,
                recall_at_10: recall,
                qps,
            });
        }
    }

    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "nlist": r.nlist,
                "nprobe": r.nprobe,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    fs::write(OUT_JSON, serde_json::to_string_pretty(&rows_json)?)?;

    Ok(rows)
}

fn main() {
    match run() {
        Ok(rows) => {
            let selected = rows.iter().find(|r| r.nlist == 256 && r.nprobe == 32);
            let status = if let Some(r) = selected {
                format!(
                    "DONE: ivf_flat_recall_vs_nprobe done. nlist=256 nprobe=32: recall={:.3} qps={}\n",
                    r.recall_at_10, r.qps
                )
            } else {
                "DONE: ivf_flat_recall_vs_nprobe done. nlist=256 nprobe=32: recall=N/A qps=N/A\n"
                    .to_string()
            };
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

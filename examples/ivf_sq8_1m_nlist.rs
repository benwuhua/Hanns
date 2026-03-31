use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfSq8Index, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 1_000_000;
const DIM: usize = 128;
const SEED: u64 = 42;
const NLIST_SWEEP: [usize; 2] = [1024, 4096];
const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 200;
const QPS_QUERIES: usize = 500;
const NPROBE_SWEEP: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const RECALL_GATE: f64 = 0.95;
const OUT_JSON: &str = "/tmp/ivf_sq8_1m_nlist_result.json";

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

fn overlap_at_k(gt: &[i64], pred: &[i64]) -> usize {
    let gt_set: HashSet<i64> = gt.iter().copied().collect();
    pred.iter().filter(|&&id| gt_set.contains(&id)).count()
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-SQ8 1M nlist sweep ===");
    println!(
        "base={} dim={} metric=L2 seed={} top_k={} recall_q={} qps_q={} nlist={:?} nprobe={:?}",
        BASE_SIZE, DIM, SEED, TOP_K, RECALL_QUERIES, QPS_QUERIES, NLIST_SWEEP, NPROBE_SWEEP
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

    let mut rows: Vec<SweepRow> = Vec::new();
    let mut gate_summary: Vec<(usize, Option<(usize, u64)>)> = Vec::new();

    for &nlist in &NLIST_SWEEP {
        println!("--- nlist={} ---", nlist);

        let ivf_cfg = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            data_type: DataType::Float,
            dim: DIM,
            params: IndexParams::ivf(nlist, 8),
        };
        let mut index = IvfSq8Index::new(&ivf_cfg)?;

        let train_start = Instant::now();
        index.train(&base)?;
        println!(
            "nlist={} train_done secs={:.2}",
            nlist,
            train_start.elapsed().as_secs_f64()
        );

        let add_start = Instant::now();
        index.add(&base, None)?;
        println!(
            "nlist={} add_done secs={:.2}",
            nlist,
            add_start.elapsed().as_secs_f64()
        );

        let mut first_gate: Option<(usize, u64)> = None;
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

            if first_gate.is_none() && recall >= RECALL_GATE {
                first_gate = Some((nprobe, qps));
                println!(
                    "nlist={} nprobe={} recall@10={:.3} QPS={} <- min for gate",
                    nlist, nprobe, recall, qps
                );
            } else {
                println!(
                    "nlist={} nprobe={} recall@10={:.3} QPS={}",
                    nlist, nprobe, recall, qps
                );
            }

            rows.push(SweepRow {
                nlist,
                nprobe,
                recall_at_10: recall,
                qps,
            });
        }

        match first_gate {
            Some((np, qps)) => {
                println!(
                    "nlist={} summary: min_nprobe_gate={} qps={}",
                    nlist, np, qps
                );
            }
            None => {
                println!("nlist={} summary: min_nprobe_gate=N/A qps=N/A", nlist);
            }
        }
        gate_summary.push((nlist, first_gate));
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
    println!("saved {}", OUT_JSON);

    println!("done");
    for (nlist, gate) in gate_summary {
        match gate {
            Some((np, qps)) => println!("nlist={} min_nprobe_gate={} qps={}", nlist, np, qps),
            None => println!("nlist={} min_nprobe_gate=N/A qps=N/A", nlist),
        }
    }

    Ok(())
}

use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfOpqIndexWrapper, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;
const BASE_SIZE: usize = 1_000_000;
const NLIST: usize = 1024;
const M: usize = 16;
const NBITS: usize = 8;
const TOP_K: usize = 10;
const QPS_QUERIES: usize = 500;
const RECALL_QUERIES: usize = 200;
const SEED: u64 = 42;
const NPROBE_SWEEP: [usize; 5] = [16, 32, 64, 128, 256];
const RECALL_GATE: f64 = 0.95;
const OUTPUT_JSON: &str = "/tmp/ivf_opq_1m_result.json";

#[derive(Debug, Clone)]
struct SweepResult {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-OPQ 1M sweep ===");
    println!(
        "base={} dim={} metric=L2 nlist={} m={} nbits={} seed={}",
        BASE_SIZE, DIM, NLIST, M, NBITS, SEED
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let qps_queries = gen_vectors(&mut rng, QPS_QUERIES, DIM);
    let recall_queries = gen_vectors(&mut rng, RECALL_QUERIES, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(&base, None)?;

    let cfg = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams {
            nlist: Some(NLIST),
            m: Some(M),
            nbits_per_idx: Some(NBITS),
            ..Default::default()
        },
    };
    let mut index = IvfOpqIndexWrapper::new(&cfg)?;
    index.train(&base)?;
    index.add(&base, None)?;

    let gt_req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };
    let gt_top10: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, knowhere_rs::api::KnowhereError>(res.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("nprobe | recall@10 | qps");
    println!("-------|-----------|------");

    let mut rows = Vec::with_capacity(NPROBE_SWEEP.len());
    for &nprobe in &NPROBE_SWEEP {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let qps_start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, &req)?;
        }
        let qps_secs = qps_start.elapsed().as_secs_f64();
        let qps = (QPS_QUERIES as f64 / qps_secs).round() as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            let hit = overlap_at_k(gt_ids, &res.ids) as f64;
            recall_sum += hit / TOP_K as f64;
        }
        let recall = recall_sum / RECALL_QUERIES as f64;
        let recall_rounded = round3(recall);

        println!("{:>6} | {:>9.3} | {}", nprobe, recall_rounded, qps);

        rows.push(SweepResult {
            nprobe,
            recall_at_10: recall_rounded,
            qps,
        });
    }

    let min_gate = rows.iter().find(|row| row.recall_at_10 >= RECALL_GATE);
    if let Some(best) = min_gate {
        println!(
            "summary: min nprobe for recall>=0.95 is {} (qps={})",
            best.nprobe, best.qps
        );
    } else {
        println!("summary: no nprobe reached recall>=0.95");
    }

    let json_rows: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "nprobe": r.nprobe,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    fs::write(OUTPUT_JSON, serde_json::to_string_pretty(&json_rows)?)?;
    println!("saved {}", OUTPUT_JSON);

    Ok(())
}

use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfSq8Index, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;
const BASE_SIZE: usize = 100_000;
const NLIST: usize = 256;
const TOP_K: usize = 10;
const QPS_QUERIES: usize = 1_000;
const RECALL_QUERIES: usize = 100;
const SEED: u64 = 42;
const NPROBE_SWEEP: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];
const OUTPUT_JSON: &str = "/tmp/ivf_sq8_sweep_result.json";
const STATUS_PATH: &str = "/tmp/codex_status_b.txt";
const RECALL_GATE: f64 = 0.95;

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
    println!("=== IVF-SQ8 nprobe sweep ===");
    println!(
        "base={} dim={} metric=L2 nlist={} seed={}",
        BASE_SIZE, DIM, NLIST, SEED
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let qps_queries = gen_vectors(&mut rng, QPS_QUERIES, DIM);
    let recall_queries = gen_vectors(&mut rng, RECALL_QUERIES, DIM);

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(&base, None)?;

    let ivf_cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params: IndexParams::ivf(NLIST, 8),
    };
    let mut ivf_sq8 = IvfSq8Index::new(&ivf_cfg)?;
    ivf_sq8.train(&base)?;
    ivf_sq8.add(&base, None)?;

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
            let _ = ivf_sq8.search(q, &req)?;
        }
        let qps_secs = qps_start.elapsed().as_secs_f64();
        let qps = (QPS_QUERIES as f64 / qps_secs).round() as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = ivf_sq8.search(q, &req)?;
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

    #[cfg(feature = "parallel")]
    {
        println!("\n=== Batch parallel mode (search_parallel) ===");
        let test_nprobes = vec![32, 64, min_gate.map(|r| r.nprobe).unwrap_or(256)];
        let test_nprobes: Vec<usize> = test_nprobes
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        let batch_queries: Vec<f32> = qps_queries[..QPS_QUERIES * DIM].to_vec();
        for &np in &test_nprobes {
            let req_batch = SearchRequest {
                top_k: TOP_K,
                nprobe: np,
                ..Default::default()
            };
            let batch_start = Instant::now();
            let _ = ivf_sq8.search_parallel(&batch_queries, &req_batch, 0)?;
            let batch_secs = batch_start.elapsed().as_secs_f64();
            let batch_qps = (QPS_QUERIES as f64 / batch_secs).round() as u64;
            println!("  nprobe={}: {} QPS", np, batch_qps);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("\n=== Batch parallel mode (search_parallel) ===");
        println!("batch parallel: skipped (feature \"parallel\" not enabled)");
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

    let status_line = if let Some(best) = min_gate {
        format!(
            "DONE: ivf-sq8 nprobe sweep done. min nprobe for recall>=0.95: {} (qps={}). results /tmp/ivf_sq8_sweep_result.json\n",
            best.nprobe, best.qps
        )
    } else {
        "DONE: ivf-sq8 nprobe sweep done. min nprobe for recall>=0.95: N/A (qps=N/A). results /tmp/ivf_sq8_sweep_result.json\n".to_string()
    };
    fs::write(STATUS_PATH, status_line)?;

    Ok(())
}

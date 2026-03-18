use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfRaBitqConfig, IvfRaBitqIndex, MemIndex};
use knowhere_rs::quantization::RefineType;
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
const OUTPUT_JSON: &str = "/tmp/ivf_rabitq_sweep_result.json";
const STATUS_PATH: &str = "/tmp/codex_status_b.txt";

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

fn run_sweep(
    name: &str,
    index: &IvfRaBitqIndex,
    qps_queries: &[f32],
    recall_queries: &[f32],
    gt_top10: &[Vec<i64>],
) -> Result<Vec<SweepResult>, Box<dyn std::error::Error>> {
    println!("--- {} ---", name);
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
        let recall = round3(recall_sum / RECALL_QUERIES as f64);
        println!("{:>6} | {:>9.3} | {}", nprobe, recall, qps);

        rows.push(SweepResult {
            nprobe,
            recall_at_10: recall,
            qps,
        });
    }

    Ok(rows)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-RaBitQ nprobe sweep (no_refine vs with_refine) ===");
    println!(
        "base={} dim={} metric=L2 nlist={} seed={} refine_k={}",
        BASE_SIZE,
        DIM,
        NLIST,
        SEED,
        TOP_K * 4
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
            let res = gt_index.search(q, &gt_req)?;
            Ok::<Vec<i64>, knowhere_rs::api::KnowhereError>(res.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let no_refine_cfg = IvfRaBitqConfig::new(DIM, NLIST).with_metric(MetricType::L2);
    let mut no_refine = IvfRaBitqIndex::new(no_refine_cfg);
    no_refine.train(&base)?;
    no_refine.add(&base, None)?;

    let with_refine_cfg = IvfRaBitqConfig::new(DIM, NLIST)
        .with_metric(MetricType::L2)
        .with_refine(RefineType::DataView, TOP_K * 4);
    let mut with_refine = IvfRaBitqIndex::new(with_refine_cfg);
    with_refine.train(&base)?;
    with_refine.add(&base, None)?;

    let no_refine_rows = run_sweep(
        "no_refine",
        &no_refine,
        &qps_queries,
        &recall_queries,
        &gt_top10,
    )?;
    let with_refine_rows = run_sweep(
        "with_refine(dataview)",
        &with_refine,
        &qps_queries,
        &recall_queries,
        &gt_top10,
    )?;

    let no_refine_max = no_refine_rows
        .iter()
        .map(|r| r.recall_at_10)
        .fold(0.0f64, f64::max);
    let with_refine_max = with_refine_rows
        .iter()
        .map(|r| r.recall_at_10)
        .fold(0.0f64, f64::max);

    println!("summary: no_refine max recall={:.3}", no_refine_max);
    println!("summary: with_refine max recall={:.3}", with_refine_max);

    let no_refine_json: Vec<serde_json::Value> = no_refine_rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "nprobe": r.nprobe,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    let with_refine_json: Vec<serde_json::Value> = with_refine_rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "nprobe": r.nprobe,
                "recall_at_10": r.recall_at_10,
                "qps": r.qps
            })
        })
        .collect();
    let output = serde_json::json!({
        "no_refine": no_refine_json,
        "with_refine": with_refine_json
    });
    fs::write(OUTPUT_JSON, serde_json::to_string_pretty(&output)?)?;
    println!("saved {}", OUTPUT_JSON);

    let status_line = format!(
        "DONE: rabitq sweep done. no_refine max recall: {:.3}, with_refine max recall: {:.3}. results /tmp/ivf_rabitq_sweep_result.json\n",
        no_refine_max, with_refine_max
    );
    fs::write(STATUS_PATH, status_line)?;

    Ok(())
}

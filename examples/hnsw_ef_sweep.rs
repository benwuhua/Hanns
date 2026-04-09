use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::HnswIndex;
use rayon::prelude::*;
use serde::Serialize;
use std::env;
use std::fs;
use std::time::Instant;

const DEFAULT_SIFT_PATH: &str = "/data/work/hanns-src/data/sift1m";
const NUM_QUERIES: usize = 1000;
const TOP_K: usize = 10;
const RECALL_K: usize = 10;
const BATCH_SIZE: usize = 32;
const NUM_THREADS: usize = 8;

#[derive(Debug, Serialize, Clone)]
struct SweepRow {
    ef: usize,
    recall_at_10: f64,
    qps: f64,
}

fn recall_at_k(results: &[Vec<i64>], gt: &[Vec<i32>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (row, gt_row) in results.iter().zip(gt.iter()) {
        let take_k = k.min(gt_row.len()).min(row.len());
        total += take_k;
        for &gt_id in gt_row.iter().take(take_k) {
            if row.iter().take(k).any(|&id| id == gt_id as i64) {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn run_sweep(
    index: &HnswIndex,
    queries: &[f32],
    dim: usize,
    gt: &[Vec<i32>],
    ef: usize,
) -> SweepRow {
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: ef,
        filter: None,
        params: Some(format!(r#"{{"ef": {ef}}}"#)),
        radius: None,
    };

    let started = Instant::now();
    let chunk_size = BATCH_SIZE * dim;
    let rows: Vec<Vec<Vec<i64>>> = queries
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut out = Vec::with_capacity(chunk.len() / dim);
            for q in chunk.chunks(dim) {
                let res = index
                    .search(q, &req)
                    .expect("hnsw search failed during sweep");
                out.push(res.ids);
            }
            out
        })
        .collect();

    let mut results = Vec::with_capacity(NUM_QUERIES);
    for block in rows {
        results.extend(block);
    }
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let qps = NUM_QUERIES as f64 / elapsed;
    let recall = recall_at_k(&results, gt, RECALL_K);

    println!("ef={ef} recall@10={recall:.4} qps={qps:.0}");
    SweepRow {
        ef,
        recall_at_10: recall,
        qps,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(NUM_THREADS)
        .build_global();

    let sift_path = env::var("SIFT_PATH").unwrap_or_else(|_| DEFAULT_SIFT_PATH.to_string());
    println!("Loading SIFT-1M from: {sift_path}");
    let dataset = load_sift1m_complete(&sift_path)?;
    let dim = dataset.dim();

    println!(
        "Loaded base={} query={} dim={}",
        dataset.num_base(),
        dataset.num_query(),
        dim
    );

    let base = dataset.base.vectors();
    let queries_all = dataset.query.vectors();
    let gt_all = dataset.ground_truth;
    if queries_all.len() < NUM_QUERIES * dim || gt_all.len() < NUM_QUERIES {
        return Err(format!("insufficient query/ground-truth rows for {NUM_QUERIES}").into());
    }
    let queries = &queries_all[..NUM_QUERIES * dim];
    let gt = &gt_all[..NUM_QUERIES];

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::BFloat16,
        dim,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(100),
            ef_search: Some(20),
            num_threads: Some(NUM_THREADS),
            ..Default::default()
        },
    };

    println!("Building HNSW: M=16 efConstruction=100 data_type=BFloat16 threads={NUM_THREADS}");
    let mut index = HnswIndex::new(&config)?;
    index.train(base)?;
    index.add(base, None)?;
    println!("Build complete.");

    let ef_sweep = [20usize, 30, 40, 50, 60, 70, 80, 100, 138, 200];
    let mut rows = Vec::with_capacity(ef_sweep.len());
    for ef in ef_sweep {
        index.set_ef_search(ef);
        rows.push(run_sweep(&index, queries, dim, gt, ef));
    }

    let out_path = "/tmp/hnsw_ef_sweep_result.json";
    let json = serde_json::to_string_pretty(&rows)?;
    fs::write(out_path, json)?;
    println!("Saved sweep JSON to {out_path}");

    Ok(())
}

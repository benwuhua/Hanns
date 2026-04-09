use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{
    HnswIndex, HnswPqConfig, HnswPqIndex, HnswPrqConfig, HnswPrqIndex, HnswSqIndex, MemIndex,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 10_000;
const DIM: usize = 128;
const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 200;
const QPS_QUERIES: usize = 1_000;
const SEED: u64 = 42;
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const LEVEL_MULTIPLIER: f32 = 0.5;
const EF_SWEEP: [usize; 6] = [16, 32, 50, 64, 100, 128];
const OUT_JSON: &str = "/tmp/hnsw_quantized_recall.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";
const RECALL_GATE: f64 = 0.95;

#[derive(Clone, Debug)]
struct Row {
    index_name: String,
    ef: usize,
    recall_at_10: f64,
    qps: u64,
}

#[derive(Clone, Debug)]
struct SummaryRow {
    index_name: String,
    min_ef_095: Option<usize>,
    qps_at_min_ef_095: Option<u64>,
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

fn eval_hnsw_flat(
    base: &[f32],
    recall_queries: &[f32],
    qps_queries: &[f32],
    gt_top10: &[Vec<i64>],
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut params = IndexParams::hnsw(EF_CONSTRUCTION, 64, LEVEL_MULTIPLIER);
    params.m = Some(M);
    let cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = HnswIndex::new(&cfg)?;
    index.train(base)?;
    index.add(base, None)?;

    let mut rows = Vec::with_capacity(EF_SWEEP.len());
    for &ef in &EF_SWEEP {
        index.set_ef_search(ef);
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            params: Some(format!(r#"{{"ef": {ef}}}"#)),
            ..Default::default()
        };

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, &req)?;
        }
        let qps = (QPS_QUERIES as f64 / start.elapsed().as_secs_f64()).round() as u64;

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            recall_sum += overlap_at_k(gt_ids, &res.ids, TOP_K) as f64 / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("HnswFlat  ef={:>3} recall@10={:.3} qps={}", ef, recall, qps);
        rows.push(Row {
            index_name: "HnswFlat".to_string(),
            ef,
            recall_at_10: recall,
            qps,
        });
    }
    Ok(rows)
}

fn eval_hnsw_sq8(
    base: &[f32],
    recall_queries: &[f32],
    qps_queries: &[f32],
    gt_top10: &[Vec<i64>],
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut index = HnswSqIndex::new(DIM);
    index.train(base)?;
    index.add(base, None)?;

    let mut rows = Vec::with_capacity(EF_SWEEP.len());
    for &ef in &EF_SWEEP {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            ..Default::default()
        };

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, &req)?;
        }
        let qps = (QPS_QUERIES as f64 / start.elapsed().as_secs_f64()).round() as u64;

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req)?;
            recall_sum += overlap_at_k(gt_ids, &res.ids, TOP_K) as f64 / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("HnswSQ8   ef={:>3} recall@10={:.3} qps={}", ef, recall, qps);
        rows.push(Row {
            index_name: "HnswSQ8".to_string(),
            ef,
            recall_at_10: recall,
            qps,
        });
    }
    Ok(rows)
}

fn eval_hnsw_pq8(
    base: &[f32],
    recall_queries: &[f32],
    qps_queries: &[f32],
    gt_top10: &[Vec<i64>],
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut rows = Vec::with_capacity(EF_SWEEP.len());
    for &ef in &EF_SWEEP {
        let cfg = HnswPqConfig::new(DIM)
            .with_m(M)
            .with_ef_construction(EF_CONSTRUCTION)
            .with_ef_search(ef)
            .with_pq_params(8, 256)
            .with_metric_type(MetricType::L2);
        let mut index = HnswPqIndex::new(cfg)?;
        index.train(base)?;
        index.add(base, None)?;

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, TOP_K, None)?;
        }
        let qps = (QPS_QUERIES as f64 / start.elapsed().as_secs_f64()).round() as u64;

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, TOP_K, None)?;
            recall_sum += overlap_at_k(gt_ids, &res.ids, TOP_K) as f64 / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("HnswPQ8   ef={:>3} recall@10={:.3} qps={}", ef, recall, qps);
        rows.push(Row {
            index_name: "HnswPQ8".to_string(),
            ef,
            recall_at_10: recall,
            qps,
        });
    }
    Ok(rows)
}

fn eval_hnsw_prq2(
    base: &[f32],
    recall_queries: &[f32],
    qps_queries: &[f32],
    gt_top10: &[Vec<i64>],
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut rows = Vec::with_capacity(EF_SWEEP.len());
    for &ef in &EF_SWEEP {
        let cfg = HnswPrqConfig::new(DIM)
            .with_m(M)
            .with_ef_construction(EF_CONSTRUCTION)
            .with_ef_search(ef)
            .with_prq_params(2, 4, 8)
            .with_metric_type(MetricType::L2);
        let mut index = HnswPrqIndex::new(cfg)?;
        index.train(base)?;
        index.add(base, None)?;

        let start = Instant::now();
        for i in 0..QPS_QUERIES {
            let q = &qps_queries[i * DIM..(i + 1) * DIM];
            let _ = index.search(q, TOP_K, None)?;
        }
        let qps = (QPS_QUERIES as f64 / start.elapsed().as_secs_f64()).round() as u64;

        let mut recall_sum = 0.0f64;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, TOP_K, None)?;
            recall_sum += overlap_at_k(gt_ids, &res.ids, TOP_K) as f64 / TOP_K as f64;
        }
        let recall = round3(recall_sum / RECALL_QUERIES as f64);

        println!("HnswPRQ2  ef={:>3} recall@10={:.3} qps={}", ef, recall, qps);
        rows.push(Row {
            index_name: "HnswPRQ2".to_string(),
            ef,
            recall_at_10: recall,
            qps,
        });
    }
    Ok(rows)
}

fn summarize(rows: &[Row], name: &str) -> SummaryRow {
    let hit = rows
        .iter()
        .filter(|r| r.index_name == name && r.recall_at_10 >= RECALL_GATE)
        .min_by_key(|r| r.ef);
    SummaryRow {
        index_name: name.to_string(),
        min_ef_095: hit.map(|r| r.ef),
        qps_at_min_ef_095: hit.map(|r| r.qps),
    }
}

fn main() {
    let result: Result<(), Box<dyn std::error::Error>> = (|| {
        println!("=== HNSW quantized variants recall sweep ===");
        println!(
            "base={} dim={} metric=L2 seed={} m={} ef_construction={} top_k={}",
            BASE_SIZE, DIM, SEED, M, EF_CONSTRUCTION, TOP_K
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
                let res = gt_index.search(q, &gt_req)?;
                Ok::<Vec<i64>, hanns::api::KnowhereError>(res.ids)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut rows = Vec::new();
        rows.extend(eval_hnsw_flat(
            &base,
            &recall_queries,
            &qps_queries,
            &gt_top10,
        )?);
        rows.extend(eval_hnsw_sq8(
            &base,
            &recall_queries,
            &qps_queries,
            &gt_top10,
        )?);
        rows.extend(eval_hnsw_pq8(
            &base,
            &recall_queries,
            &qps_queries,
            &gt_top10,
        )?);
        rows.extend(eval_hnsw_prq2(
            &base,
            &recall_queries,
            &qps_queries,
            &gt_top10,
        )?);

        let summary = vec![
            summarize(&rows, "HnswFlat"),
            summarize(&rows, "HnswSQ8"),
            summarize(&rows, "HnswPQ8"),
            summarize(&rows, "HnswPRQ2"),
        ];

        println!("summary:");
        for s in &summary {
            match (s.min_ef_095, s.qps_at_min_ef_095) {
                (Some(ef), Some(qps)) => {
                    println!("  {:<9} min_ef_0.95={} qps={}", s.index_name, ef, qps)
                }
                _ => println!("  {:<9} min_ef_0.95=N/A qps=N/A", s.index_name),
            }
        }

        let json_rows: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "index_name": r.index_name,
                    "ef": r.ef,
                    "recall_at_10": r.recall_at_10,
                    "qps": r.qps
                })
            })
            .collect();
        fs::write(OUT_JSON, serde_json::to_string_pretty(&json_rows)?)?;
        println!("saved {}", OUT_JSON);

        let sq = summary.iter().find(|s| s.index_name == "HnswSQ8");
        let pq = summary.iter().find(|s| s.index_name == "HnswPQ8");
        let prq = summary.iter().find(|s| s.index_name == "HnswPRQ2");
        let status = format!(
            "DONE: hnsw_quantized done. sq8_min_ef_0.95={} sq8_qps={} pq8_min_ef_0.95={} pq8_qps={} prq2_min_ef_0.95={} prq2_qps={}\n",
            sq.and_then(|s| s.min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
            sq.and_then(|s| s.qps_at_min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
            pq.and_then(|s| s.min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
            pq.and_then(|s| s.qps_at_min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
            prq.and_then(|s| s.min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
            prq.and_then(|s| s.qps_at_min_ef_095.map(|v| v.to_string()))
                .unwrap_or_else(|| "N/A".to_string()),
        );
        fs::write(STATUS_FILE, status)?;

        Ok(())
    })();

    if let Err(err) = result {
        let _ = fs::write(STATUS_FILE, format!("ERROR: {}\n", err));
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

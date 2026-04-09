use std::collections::HashSet;
use std::fs;

use hanns::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{IvfPqIndex, MemIndex};
use hanns::quantization::pq::{PQConfig, ProductQuantizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BASE_SIZE: usize = 10_000;
const DIM: usize = 128;
const NLIST: usize = 64;
const NQ: usize = 100;
const TOP_K: usize = 10;
const SEED: u64 = 42;
const NPROBE_SWEEP: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const M_SWEEP: [usize; 5] = [2, 4, 8, 16, 32];
const OUT_JSON: &str = "/tmp/ivf_pq_diag_result.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn overlap_at_k(a: &[i64], b: &[i64]) -> usize {
    let gt_set: HashSet<i64> = a.iter().copied().collect();
    b.iter().filter(|&&id| gt_set.contains(&id)).count()
}

fn compute_gt_top10(
    base: &[f32],
    queries: &[f32],
) -> Result<Vec<Vec<i64>>, Box<dyn std::error::Error>> {
    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg)?;
    gt_index.add(base, None)?;
    let req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };
    let gt_top10: Vec<Vec<i64>> = (0..NQ)
        .map(|i| {
            let q = &queries[i * DIM..(i + 1) * DIM];
            let res = gt_index.search(q, &req)?;
            Ok::<Vec<i64>, hanns::api::KnowhereError>(res.ids)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(gt_top10)
}

fn recall_for_index(
    index: &IvfPqIndex,
    queries: &[f32],
    gt_top10: &[Vec<i64>],
    nprobe: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe,
        ..Default::default()
    };
    let mut recall_sum = 0.0f64;
    for (i, gt_ids) in gt_top10.iter().enumerate() {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let res = index.search(q, &req)?;
        let hit = overlap_at_k(gt_ids, &res.ids) as f64;
        recall_sum += hit / TOP_K as f64;
    }
    Ok(recall_sum / NQ as f64)
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-PQ diagnostic ===");
    println!(
        "base={} dim={} nlist={} nbits={} queries={} top_k={}",
        BASE_SIZE, DIM, NLIST, 8, NQ, TOP_K
    );

    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let queries = gen_vectors(&mut rng, NQ, DIM);
    let gt_top10 = compute_gt_top10(&base, &queries)?;

    // 1) nprobe sweep (m=8, nbits=8)
    println!("\n[nprobe sweep] m=8 nbits=8");
    let mut nprobe_rows = Vec::with_capacity(NPROBE_SWEEP.len());
    {
        let cfg = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            data_type: hanns::api::DataType::Float,
            dim: DIM,
            params: IndexParams {
                nlist: Some(NLIST),
                nprobe: Some(1),
                m: Some(8),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };
        let mut index = IvfPqIndex::new(&cfg)?;
        index.train(&base)?;
        index.add(&base, None)?;

        for &nprobe in &NPROBE_SWEEP {
            let recall = round3(recall_for_index(&index, &queries, &gt_top10, nprobe)?);
            println!("nprobe={:>2} recall@10={:.3}", nprobe, recall);
            nprobe_rows.push(serde_json::json!({
                "nprobe": nprobe,
                "recall_at_10": recall
            }));
        }
    }

    // 2) m sweep (fixed nprobe=nlist=64)
    println!("\n[m sweep] nprobe=64 nbits=8");
    let mut m_rows = Vec::with_capacity(M_SWEEP.len());
    for &m in &M_SWEEP {
        let cfg = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            data_type: hanns::api::DataType::Float,
            dim: DIM,
            params: IndexParams {
                nlist: Some(NLIST),
                nprobe: Some(NLIST),
                m: Some(m),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };
        let mut index = IvfPqIndex::new(&cfg)?;
        index.train(&base)?;
        index.add(&base, None)?;

        let recall = round3(recall_for_index(&index, &queries, &gt_top10, NLIST)?);
        println!("m={:>2} recall@10={:.3}", m, recall);
        m_rows.push(serde_json::json!({
            "m": m,
            "recall_at_10": recall
        }));
    }

    // 3) ADC sanity (small test)
    println!("\n[adc sanity] n=100 dim=16 nlist=4 m=4 nprobe=4");
    let mut rng_small = StdRng::seed_from_u64(SEED + 7);
    let dim_small = 16usize;
    let n_small = 100usize;
    let data_small = gen_vectors(&mut rng_small, n_small, dim_small);
    let query_small = gen_vectors(&mut rng_small, 1, dim_small);
    let q = &query_small[..dim_small];

    let mut pq = ProductQuantizer::new(PQConfig::new(dim_small, 4, 8));
    pq.train(n_small, &data_small)?;

    let mut adc_sanity = Vec::new();
    for vid in [0usize, 1usize, 2usize] {
        let v = &data_small[vid * dim_small..(vid + 1) * dim_small];
        let code = pq.encode(v)?;
        let true_l2 = l2_sqr(q, v);
        let adc_approx_dist = pq.compute_distance(q, &code);
        println!(
            "vid={} true_L2={:.6} adc_approx_dist={:.6}",
            vid, true_l2, adc_approx_dist
        );
        adc_sanity.push(serde_json::json!({
            "vector_id": vid,
            "true_l2": true_l2,
            "adc_approx_dist": adc_approx_dist
        }));
    }

    let output = serde_json::json!({
        "nprobe_sweep": nprobe_rows,
        "m_sweep": m_rows,
        "adc_sanity": adc_sanity
    });
    fs::write(OUT_JSON, serde_json::to_string_pretty(&output)?)?;
    println!("\nsaved {}", OUT_JSON);

    let nprobe_max = output["nprobe_sweep"]
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .filter_map(|r| r["recall_at_10"].as_f64())
                .max_by(|a, b| a.total_cmp(b))
        })
        .unwrap_or(0.0);
    let m_max = output["m_sweep"]
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .filter_map(|r| r["recall_at_10"].as_f64())
                .max_by(|a, b| a.total_cmp(b))
        })
        .unwrap_or(0.0);

    let status = format!(
        "DONE: ivf-pq-diag. nprobe sweep max recall={:.3}, m sweep max recall={:.3}. see /tmp/ivf_pq_diag_result.json\n",
        nprobe_max, m_max
    );
    fs::write(STATUS_FILE, status)?;

    Ok(())
}

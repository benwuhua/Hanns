use std::collections::HashSet;
use std::fs;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{AisaqConfig, MemIndex, PQFlashIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;

const DIMS: [usize; 6] = [4, 16, 32, 64, 96, 128];
const N: usize = 2_000;
const NQ: usize = 100;
const TOP_K: usize = 10;
const MAX_DEGREE: usize = 32;
const SEARCH_LIST_SIZE: usize = 64;
const BEAMWIDTH: usize = 8;
const SEED: u64 = 42;
const OUT_JSON: &str = "/tmp/diskann_dim_diag.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

fn compute_recall_at_10(gt_top10: &[Vec<i64>], pred_top10: &[Vec<i64>]) -> f64 {
    let mut hits = 0usize;
    for (gt, pred) in gt_top10.iter().zip(pred_top10.iter()) {
        let gt_set: HashSet<i64> = gt.iter().copied().collect();
        for id in pred.iter().take(TOP_K) {
            if gt_set.contains(id) {
                hits += 1;
            }
        }
    }
    hits as f64 / (gt_top10.len() * TOP_K) as f64
}

fn find_medoid(vectors: &[f32], dim: usize) -> usize {
    let n = vectors.len() / dim;
    let mut centroid = vec![0.0f32; dim];
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        for d in 0..dim {
            centroid[d] += v[d];
        }
    }
    for c in &mut centroid {
        *c /= n as f32;
    }

    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        let d = l2_sqr(&centroid, v);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DiskANN dim diagnostic ===");
    println!(
        "n={} queries={} top_k={} max_degree={} search_list_size={} beamwidth={} seed={}",
        N, NQ, TOP_K, MAX_DEGREE, SEARCH_LIST_SIZE, BEAMWIDTH, SEED
    );

    let mut result_rows = Vec::new();
    let mut status_rows = Vec::new();

    for &dim in &DIMS {
        let base = gen_vectors(SEED, N, dim);
        let queries = gen_vectors(SEED.wrapping_add(1), NQ, dim);

        let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, dim);
        let mut gt_index = MemIndex::new(&flat_cfg)?;
        gt_index.add(&base, None)?;

        let params = IndexParams {
            max_degree: Some(MAX_DEGREE),
            search_list_size: Some(SEARCH_LIST_SIZE),
            beamwidth: Some(BEAMWIDTH),
            ..Default::default()
        };
        let disk_cfg = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            data_type: knowhere_rs::api::DataType::Float,
            dim,
            params,
        };
        let mut disk = PQFlashIndex::new(
            AisaqConfig::from_index_config(&disk_cfg),
            disk_cfg.metric_type,
            disk_cfg.dim,
        )?;
        disk.train(&base)?;
        disk.add(&base)?;

        let req = SearchRequest {
            top_k: TOP_K,
            ..Default::default()
        };

        let mut gt_top10 = Vec::with_capacity(NQ);
        let mut disk_top10 = Vec::with_capacity(NQ);
        for qi in 0..NQ {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt = gt_index.search(q, &req)?;
            let pred = disk.search(q, req.top_k)?;
            gt_top10.push(gt.ids.into_iter().take(TOP_K).collect());
            disk_top10.push(pred.ids.into_iter().take(TOP_K).collect());
        }
        let recall = compute_recall_at_10(&gt_top10, &disk_top10);
        println!(
            "dim={:>3} recall@10={:.4} ntotal={}",
            dim,
            recall,
            disk.len()
        );

        let mut extra = json!({});
        if dim == 64 {
            let entry_idx = find_medoid(&base, dim);
            let query0 = &queries[0..dim];
            let entry_vec = &base[entry_idx * dim..(entry_idx + 1) * dim];
            extra = json!({
                "entry_point_inferred_id": entry_idx,
                "entry_distance": l2_sqr(query0, entry_vec).sqrt(),
                "effective_search_l": SEARCH_LIST_SIZE,
            });
        }

        result_rows.push(json!({
            "dim": dim,
            "recall": recall,
            "ntotal": disk.len(),
            "search_list_size": SEARCH_LIST_SIZE,
            "beamwidth": BEAMWIDTH,
            "diag": extra
        }));
        status_rows.push(format!("dim={}: recall@10={:.4}", dim, recall));
    }

    let output = json!({
        "config": {
            "n": N,
            "nq": NQ,
            "top_k": TOP_K,
            "max_degree": MAX_DEGREE,
            "search_list_size": SEARCH_LIST_SIZE,
            "beamwidth": BEAMWIDTH,
            "seed": SEED
        },
        "results": result_rows
    });
    fs::write(OUT_JSON, serde_json::to_string_pretty(&output)?)?;
    println!("saved {}", OUT_JSON);

    let mut status = String::new();
    status.push_str("DONE: DiskANN dim diagnostic complete. ");
    status.push_str(&status_rows.join("; "));
    status.push('\n');
    fs::write(STATUS_FILE, status)?;
    println!("saved {}", STATUS_FILE);

    Ok(())
}

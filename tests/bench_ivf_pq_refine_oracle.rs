//! IVF-PQ exact-refine oracle diagnostic harness.
//!
//! This is intentionally an ignored diagnostic test. It verifies the contract
//! required by the IVF-PQ optimization plan: use the same ADC candidate set as
//! the production search path, then rerank that set with exact L2 over original
//! vectors to distinguish candidate/ranking loss from PQ quantization loss.
//!
//! Local smoke:
//!   cargo test --test bench_ivf_pq_refine_oracle -- --ignored --nocapture
//!
//! Remote authority runs should set:
//!   IVFPQ_ORACLE_OUTPUT_DIR=docs/parity

use hanns::api::{IndexConfig, IndexParams};
use hanns::faiss::IvfPqIndex;
use hanns::{IndexType, MetricType};
use rand::{Rng, SeedableRng};
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn ground_truth(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i64>> {
    let n = base.len() / dim;
    let nq = queries.len() / dim;
    (0..nq)
        .map(|q| {
            let query = &queries[q * dim..(q + 1) * dim];
            let mut scored = (0..n)
                .map(|i| {
                    let vector = &base[i * dim..(i + 1) * dim];
                    (l2(query, vector), i as i64)
                })
                .collect::<Vec<_>>();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            scored.into_iter().take(k).map(|(_, id)| id).collect()
        })
        .collect()
}

fn recall_at(predictions: &[Vec<i64>], truth: &[Vec<i64>], k: usize) -> f64 {
    let total = predictions
        .iter()
        .zip(truth.iter())
        .map(|(pred, gt)| {
            let pred_set = pred.iter().take(k).copied().collect::<HashSet<_>>();
            let gt_set = gt.iter().take(k).copied().collect::<HashSet<_>>();
            pred_set.intersection(&gt_set).count() as f64 / k as f64
        })
        .sum::<f64>();
    total / predictions.len().max(1) as f64
}

fn exact_refine_from_adc_candidates(
    base: &[f32],
    query: &[f32],
    dim: usize,
    adc_candidates: &[i64],
    k: usize,
) -> Vec<i64> {
    let mut scored = adc_candidates
        .iter()
        .copied()
        .filter(|&id| id >= 0)
        .map(|id| {
            let offset = id as usize * dim;
            let vector = &base[offset..offset + dim];
            (l2(query, vector), id)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
#[ignore]
fn bench_ivf_pq_refine_oracle() {
    let dim = 32;
    let n_base = 2_000;
    let n_queries = 25;
    let k = 10;
    let candidate_pool = 100;
    let nlist = 32;
    let nprobe = 16;
    let m = 8;
    let nbits = 8;

    let base = generate_vectors(n_base, dim, 42);
    let queries = generate_vectors(n_queries, dim, 7);
    let truth = ground_truth(&base, &queries, dim, k);

    let mut config = IndexConfig::new(IndexType::IvfPq, MetricType::L2, dim);
    config.params = IndexParams::ivf(nlist, nprobe);
    config.params.m = Some(m);
    config.params.nbits_per_idx = Some(nbits);

    let build_start = Instant::now();
    let mut index = IvfPqIndex::new(&config).expect("create IVF-PQ");
    index.train(&base).expect("train IVF-PQ");
    index.add(&base, None).expect("add IVF-PQ");
    let build_s = build_start.elapsed().as_secs_f64();

    let adc_start = Instant::now();
    let mut adc_topk = Vec::with_capacity(n_queries);
    let mut adc_pooled = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let query = &queries[q * dim..(q + 1) * dim];
        let pooled = index
            .diagnostic_adc_candidate_ids(query, nprobe, candidate_pool)
            .expect("ADC candidate search");
        adc_topk.push(pooled.iter().take(k).copied().collect::<Vec<_>>());
        adc_pooled.push(pooled);
    }
    let adc_s = adc_start.elapsed().as_secs_f64();

    let refine_start = Instant::now();
    let mut refined_topk = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let query = &queries[q * dim..(q + 1) * dim];
        refined_topk.push(exact_refine_from_adc_candidates(
            &base,
            query,
            dim,
            &adc_pooled[q],
            k,
        ));
    }
    let refine_s = refine_start.elapsed().as_secs_f64();

    let adc_recall = recall_at(&adc_topk, &truth, k);
    let refine_recall = recall_at(&refined_topk, &truth, k);
    let root_cause = if refine_recall > adc_recall + 0.05 {
        "candidate_selection_loss"
    } else if refine_recall < 0.8 {
        "pq_quantization_loss"
    } else {
        "parameter_ceiling"
    };

    let run_set_id = format!(
        "ivfpq-refine-oracle-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let payload = json!({
        "artifact_type": "ivfpq_refine_oracle",
        "run_set_id": run_set_id,
        "authority_surface": std::env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "hanns_commit": std::env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "top_k": k,
        "nlist": nlist,
        "nprobe": nprobe,
        "m": m,
        "nbits": nbits,
        "candidate_pool": candidate_pool,
        "refine_multiplier": candidate_pool / k,
        "build_s": build_s,
        "adc": {
            "recall_at_10": adc_recall,
            "qps": n_queries as f64 / adc_s.max(1e-9)
        },
        "exact_refine": {
            "recall_at_10": refine_recall,
            "qps": n_queries as f64 / (adc_s + refine_s).max(1e-9),
            "rerank_only_s": refine_s
        },
        "root_cause_classification": root_cause,
        "note": "Diagnostic only: exact-refine oracle over the same ADC candidate set; not a production verdict."
    });

    let output_dir = std::env::var("IVFPQ_ORACLE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/parity"));
    fs::create_dir_all(&output_dir).expect("create output dir");
    let stem = payload["run_set_id"].as_str().expect("run_set_id");
    let json_path = output_dir.join(format!("hanns-knowhere-{stem}.json"));
    let md_path = output_dir.join(format!("hanns-knowhere-{stem}.md"));
    fs::write(&json_path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
    fs::write(
        &md_path,
        format!(
            "# IVF-PQ exact-refine oracle\n\n- ADC R@10: {:.4}\n- Exact-refine R@10: {:.4}\n- Root cause: `{}`\n- JSON: `{}`\n",
            adc_recall,
            refine_recall,
            root_cause,
            json_path.display()
        ),
    )
    .unwrap();

    println!("wrote {}", json_path.display());
    println!("wrote {}", md_path.display());
}

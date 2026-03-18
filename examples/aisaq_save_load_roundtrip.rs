use knowhere_rs::api::MetricType;
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;

const N: usize = 3000;
const DIM: usize = 64;
const NQ: usize = 100;
const TOP_K: usize = 10;

#[inline]
fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn brute_force_topk_external_ids(
    vectors: &[f32],
    queries: &[f32],
    ids: &[i64],
    dim: usize,
    k: usize,
) -> Vec<Vec<i64>> {
    let mut all = Vec::with_capacity(queries.len() / dim);
    for q in queries.chunks(dim) {
        let mut scored: Vec<(usize, f32)> = (0..ids.len())
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                (i, l2_sqr(q, v))
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.push(
            scored
                .iter()
                .take(k)
                .map(|(i, _)| ids[*i])
                .collect::<Vec<i64>>(),
        );
    }
    all
}

fn recall_at_k(results: &[Vec<i64>], gt: &[Vec<i64>], k: usize) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (r, g) in results.iter().zip(gt.iter()) {
        for id in g.iter().take(k) {
            total += 1;
            if r.iter().take(k).any(|x| x == id) {
                hit += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..N * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NQ * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let ext_ids: Vec<i64> = (10_000..10_000 + N as i64).collect();

    let config = AisaqConfig {
        disk_pq_dims: 0,
        max_degree: 32,
        search_list_size: 128,
        ..AisaqConfig::default()
    };

    let mut index = PQFlashIndex::new(config, MetricType::L2, DIM)?;
    index.train(&vectors)?;
    index.add_with_ids(&vectors, Some(&ext_ids))?;

    let dir = "/tmp/aisaq_test_index";
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir)?;
    index.save(dir)?;

    let loaded = PQFlashIndex::load(dir)?;

    let mut all_results = Vec::with_capacity(NQ);
    let mut external_id_ok = true;
    for q in queries.chunks(DIM).take(NQ) {
        let r = loaded.search(q, TOP_K)?;
        if r.ids.len() != TOP_K {
            external_id_ok = false;
        }
        for &id in &r.ids {
            if !(10_000..13_000).contains(&id) {
                external_id_ok = false;
            }
        }
        all_results.push(r.ids);
    }

    let gt = brute_force_topk_external_ids(&vectors, &queries, &ext_ids, DIM, TOP_K);
    let recall = recall_at_k(&all_results, &gt, TOP_K);

    println!(
        "AISAQ save/load roundtrip: external_id_ok={} recall@10={:.3}",
        external_id_ok, recall
    );

    if !external_id_ok {
        return Err("external id validation failed".into());
    }
    if recall < 0.90 {
        return Err(format!("recall@10 too low: {:.3}", recall).into());
    }

    Ok(())
}

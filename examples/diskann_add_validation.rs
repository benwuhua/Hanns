use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs;
use std::hash::{Hash, Hasher};

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{AisaqConfig, PQFlashIndex};

const N: usize = 2000;
const DIM: usize = 64;
const NQ: usize = 200;
const TOP_K: usize = 10;
const SEED: u64 = 42;
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n * dim {
        let mut h = DefaultHasher::new();
        (seed, i as u64).hash(&mut h);
        let v = (h.finish() as f32) / (u64::MAX as f32);
        out.push(v);
    }
    out
}

fn brute_force_top_k(base: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<i64> {
    let n = base.len() / dim;
    let mut dists: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let v = &base[i * dim..(i + 1) * dim];
            let d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists
        .into_iter()
        .take(k)
        .map(|(idx, _)| idx as i64)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DiskANN add() validation ===");
    println!(
        "n={} dim={} nq={} top_k={} seed={}",
        N, DIM, NQ, TOP_K, SEED
    );

    let base = gen_vectors(SEED, N, DIM);
    let queries = gen_vectors(SEED + 1, NQ, DIM);

    let cfg = IndexConfig {
        index_type: IndexType::DiskAnn,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams {
            max_degree: Some(32),
            search_list_size: Some(64),
            construction_l: Some(64),
            beamwidth: Some(8),
            ..Default::default()
        },
    };

    let mut index = PQFlashIndex::new(
        AisaqConfig::from_index_config(&cfg),
        cfg.metric_type,
        cfg.dim,
    )?;

    let train_data = &base[..1000 * DIM];
    let add_data = &base[1000 * DIM..];

    index.train(train_data)?;
    index.add(train_data)?;
    index.add(add_data)?;
    let added = add_data.len() / DIM;

    println!("train_count=1000 add_count={}", added);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 64,
        ..Default::default()
    };

    let mut recall_sum = 0.0f64;
    for i in 0..NQ {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let gt = brute_force_top_k(&base, q, DIM, TOP_K);
        let pred = index.search(q, req.top_k)?;
        let pred_set: HashSet<i64> = pred.ids.into_iter().collect();
        let hit = gt.iter().filter(|&&id| pred_set.contains(&id)).count();
        recall_sum += hit as f64 / TOP_K as f64;
    }

    let recall = recall_sum / NQ as f64;

    println!("recall@10={:.3}", recall);
    println!("ntotal={}", index.len());

    if recall > 0.80 {
        fs::write(
            STATUS_FILE,
            format!("DONE: recall={:.3} ntotal={}\n", recall, index.len(),),
        )?;
    } else {
        let msg = format!("ERROR: recall too low: {:.3}\n", recall);
        fs::write(STATUS_FILE, &msg)?;
        return Err(msg.into());
    }

    assert!(recall > 0.80, "recall too low: {}", recall);
    Ok(())
}

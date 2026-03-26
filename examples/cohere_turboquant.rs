//! Wikipedia-Cohere 1M TurboQuant benchmark (IP, dim=768)
//! Usage:
//!   cargo run --example cohere_turboquant -- <optional_data_dir>

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::api::{MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfTurboQuantConfig, IvfTurboQuantIndex};
use rayon::prelude::*;

const DEFAULT_DATA_DIR: &str = "/data/work/datasets/wikipedia-cohere-1m";
const TOP_K: usize = 10;
const IVF_NLIST: usize = 1024;
const IVF_TRAIN_SIZE: usize = 100_000;
const TURBO_BITS_SWEEP: [u8; 3] = [4, 6, 8];
const IVF_NPROBE_SWEEP: [usize; 4] = [8, 16, 32, 64];
const EXPECTED_DIM: usize = 768;

fn read_u32_le(file: &mut File) -> Result<u32, Box<dyn Error>> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_fbin(path: &Path) -> Result<(usize, usize, Vec<f32>), Box<dyn Error>> {
    let mut file = File::open(path)?;
    let n = read_u32_le(&mut file)? as usize;
    let dim = read_u32_le(&mut file)? as usize;
    let total = n
        .checked_mul(dim)
        .ok_or_else(|| format!("overflow in n*dim for {}", path.display()))?;

    let mut bytes = vec![0u8; total * std::mem::size_of::<f32>()];
    file.read_exact(&mut bytes)?;

    let mut vectors = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(4) {
        vectors.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((n, dim, vectors))
}

fn read_ibin(path: &Path) -> Result<(usize, usize, Vec<i32>), Box<dyn Error>> {
    let mut file = File::open(path)?;
    let n = read_u32_le(&mut file)? as usize;
    let k = read_u32_le(&mut file)? as usize;
    let total = n
        .checked_mul(k)
        .ok_or_else(|| format!("overflow in n*k for {}", path.display()))?;

    let mut bytes = vec![0u8; total * std::mem::size_of::<i32>()];
    file.read_exact(&mut bytes)?;

    let mut neighbors = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(4) {
        neighbors.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((n, k, neighbors))
}

fn rows_i32(flat: &[i32], width: usize) -> Vec<Vec<i32>> {
    flat.chunks_exact(width).map(|c| c.to_vec()).collect()
}

fn normalize_ids(ids: &[i64], top_k: usize) -> Vec<i64> {
    let mut row = Vec::with_capacity(top_k);
    row.extend(ids.iter().take(top_k).copied());
    while row.len() < top_k {
        row.push(-1);
    }
    row
}

fn compute_recall(results: &[Vec<i64>], gt: &[Vec<i32>], top_k: usize) -> f32 {
    let n = results.len().min(gt.len());
    if n == 0 || top_k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    for i in 0..n {
        let r = &results[i];
        let g = &gt[i];
        for &gid in g.iter().take(top_k.min(g.len())) {
            if r.iter().take(top_k.min(r.len())).any(|&rid| rid == gid as i64) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir_arg = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DATA_DIR.to_string());
    let data_dir = Path::new(&data_dir_arg);
    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.ibin");

    println!("=== Wikipedia-Cohere 1M TurboQuant benchmark ===");
    println!("data_dir={}", data_dir.display());

    let t_load = Instant::now();
    let (base_n, base_dim, base) = read_fbin(&base_path)?;
    let (query_n, query_dim, queries) = read_fbin(&query_path)?;
    let (gt_n, gt_k, gt_flat) = read_ibin(&gt_path)?;
    let gt = rows_i32(&gt_flat, gt_k);
    println!(
        "loaded: base_n={} query_n={} gt_n={} dim={} gt_k={} (load {:.2}s)",
        base_n,
        query_n,
        gt_n,
        base_dim,
        gt_k,
        t_load.elapsed().as_secs_f64()
    );

    if base_dim == 0 || base_dim != query_dim {
        return Err(format!(
            "dim mismatch: base_dim={} query_dim={}",
            base_dim, query_dim
        )
        .into());
    }
    if gt_n != query_n {
        return Err(format!("gt rows {} != query rows {}", gt_n, query_n).into());
    }
    if base_dim != EXPECTED_DIM {
        println!(
            "warning: expected dim={} but got {}, continuing",
            EXPECTED_DIM, base_dim
        );
    }
    if gt_k < TOP_K {
        return Err(format!("gt_k={} < top_k={}", gt_k, TOP_K).into());
    }

    let train_n = IVF_TRAIN_SIZE.min(base_n);
    let train = &base[..train_n * base_dim];
    let eval_n = query_n.min(gt.len());

    println!("setup: train_size={} eval_queries={}", train_n, eval_n);

    for &bits_per_dim in &TURBO_BITS_SWEEP {
        let mut index = IvfTurboQuantIndex::new(
            IvfTurboQuantConfig::new(base_dim, IVF_NLIST, bits_per_dim)
                .with_metric(MetricType::Ip),
        );

        let t_build = Instant::now();
        index.train(train)?;
        index.add(&base, None)?;
        println!(
            "[TurboQuant] bits={} build_time={:.2}s",
            bits_per_dim,
            t_build.elapsed().as_secs_f64()
        );

        for &nprobe in &IVF_NPROBE_SWEEP {
            let req = SearchRequest {
                top_k: TOP_K,
                nprobe,
                ..Default::default()
            };

            let t_batch = Instant::now();
            queries.par_chunks_exact(base_dim).for_each(|q| {
                let _ = index.search(q, &req);
            });
            let qps = query_n as f64 / t_batch.elapsed().as_secs_f64().max(f64::EPSILON);

            let mut results = Vec::with_capacity(eval_n);
            for q in queries.chunks_exact(base_dim).take(eval_n) {
                let res = index.search(q, &req)?;
                results.push(normalize_ids(&res.ids, TOP_K));
            }

            let recall = compute_recall(&results, &gt[..eval_n], TOP_K);
            println!(
                "[TurboQuant] bits={} nprobe={}: recall@10={:.4}, QPS={:.0}",
                bits_per_dim, nprobe, recall, qps
            );
        }
    }

    Ok(())
}

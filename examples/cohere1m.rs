//! Wikipedia-Cohere 1M benchmark (IP, dim=768)
//! Usage:
//!   cargo run --example cohere1m --release -- <data_dir>
//! data_dir must contain: base.fbin, query.fbin, gt.ibin

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, IvfSq8Index};
use rayon::prelude::*;

const TOP_K: usize = 10;
const HNSW_M: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_LEVEL_MULTIPLIER: f32 = 0.5;
const HNSW_EF_SWEEP: [usize; 6] = [16, 32, 50, 60, 100, 138];
const IVF_NLIST: usize = 1024;
const IVF_TRAIN_SIZE: usize = 100_000;
const IVF_NPROBE_SWEEP: [usize; 5] = [4, 8, 16, 32, 64];
const SINGLE_QPS_QUERIES: usize = 1000;
const EXPECTED_DIM: usize = 768;
const EXPECTED_GT_K: usize = 100;

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
    for &id in ids.iter().take(top_k) {
        row.push(id);
    }
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
        let rk = top_k.min(r.len());
        let gk = top_k.min(g.len());
        for &gid in g.iter().take(gk) {
            if r.iter().take(rk).any(|&rid| rid == gid as i64) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir_arg = env::args().nth(1).unwrap_or_default();
    if data_dir_arg.is_empty() {
        return Err(
            "missing <data_dir>; expected base.fbin/query.fbin/gt.ibin in that directory".into(),
        );
    }
    let data_dir = Path::new(&data_dir_arg);
    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.ibin");

    println!("=== Wikipedia-Cohere 1M benchmark ===");
    println!("data_dir={}", data_dir.display());
    println!(
        "files: base={}, query={}, gt={}",
        base_path.display(),
        query_path.display(),
        gt_path.display()
    );

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
    if gt_k != EXPECTED_GT_K {
        println!(
            "warning: expected gt_k={} but got {}, continuing",
            EXPECTED_GT_K, gt_k
        );
    }

    let train_n = IVF_TRAIN_SIZE.min(base_n);
    let train = &base[..train_n * base_dim];
    let qps_n = SINGLE_QPS_QUERIES.min(query_n);
    let eval_n = query_n.min(gt.len());

    println!(
        "setup: train_size={} single_qps_queries={} eval_queries={}",
        train_n, qps_n, eval_n
    );

    println!("\n--- HNSW (IP, M=16, ef_construction=200) ---");
    let mut hnsw_params = IndexParams::hnsw(
        HNSW_EF_CONSTRUCTION,
        HNSW_EF_SWEEP[0],
        HNSW_LEVEL_MULTIPLIER,
    );
    hnsw_params.m = Some(HNSW_M);
    let hnsw_cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Ip,
        data_type: DataType::Float,
        dim: base_dim,
        params: hnsw_params,
    };
    let mut hnsw = HnswIndex::new(&hnsw_cfg)?;

    let t = Instant::now();
    hnsw.train(&base)?;
    hnsw.add(&base, None)?;
    println!("build_time={:.2}s", t.elapsed().as_secs_f64());

    for &ef in &HNSW_EF_SWEEP {
        hnsw.set_ef_search(ef);
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            params: Some(format!(r#"{{\"ef\": {ef}}}"#)),
            ..Default::default()
        };

        let t_single = Instant::now();
        for q in queries.chunks_exact(base_dim).take(qps_n) {
            let _ = hnsw.search(q, &req)?;
        }
        let qps_single = qps_n as f64 / t_single.elapsed().as_secs_f64().max(f64::EPSILON);

        let t_batch = Instant::now();
        queries.par_chunks_exact(base_dim).for_each(|q| {
            let _ = hnsw.search(q, &req);
        });
        let qps_batch = query_n as f64 / t_batch.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut results = Vec::with_capacity(eval_n);
        for q in queries.chunks_exact(base_dim).take(eval_n) {
            let res = hnsw.search(q, &req)?;
            results.push(normalize_ids(&res.ids, TOP_K));
        }
        let recall = compute_recall(&results, &gt[..eval_n], TOP_K);

        println!(
            "ef={:>4} recall@10={:.4} single_qps={:.0} batch_qps={:.0}",
            ef, recall, qps_single, qps_batch
        );
    }

    println!("\n--- IVF-Flat (IP, nlist=1024, train_size=100000) ---");
    let mut ivf_flat_params = IndexParams::default();
    ivf_flat_params.nlist = Some(IVF_NLIST);
    ivf_flat_params.nprobe = Some(IVF_NPROBE_SWEEP[0]);
    let ivf_flat_cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::Ip,
        data_type: DataType::Float,
        dim: base_dim,
        params: ivf_flat_params,
    };
    let mut ivf_flat = IvfFlatIndex::new(&ivf_flat_cfg)?;
    let t = Instant::now();
    ivf_flat.train(train)?;
    ivf_flat.add(&base, None)?;
    println!("build_time={:.2}s", t.elapsed().as_secs_f64());

    for &nprobe in &IVF_NPROBE_SWEEP {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let t_batch = Instant::now();
        queries.par_chunks_exact(base_dim).for_each(|q| {
            let _ = ivf_flat.search(q, &req);
        });
        let qps_batch = query_n as f64 / t_batch.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut results = Vec::with_capacity(eval_n);
        for q in queries.chunks_exact(base_dim).take(eval_n) {
            let res = ivf_flat.search(q, &req)?;
            results.push(normalize_ids(&res.ids, TOP_K));
        }
        let recall = compute_recall(&results, &gt[..eval_n], TOP_K);
        println!(
            "nprobe={:>3} recall@10={:.4} batch_qps={:.0}",
            nprobe, recall, qps_batch
        );
    }

    println!("\n--- IVF-SQ8 (IP, nlist=1024, train_size=100000) ---");
    let mut ivf_sq8_params = IndexParams::default();
    ivf_sq8_params.nlist = Some(IVF_NLIST);
    ivf_sq8_params.nprobe = Some(IVF_NPROBE_SWEEP[0]);
    let ivf_sq8_cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::Ip,
        data_type: DataType::Float,
        dim: base_dim,
        params: ivf_sq8_params,
    };
    let mut ivf_sq8 = IvfSq8Index::new(&ivf_sq8_cfg)?;
    let t = Instant::now();
    ivf_sq8.train(train)?;
    ivf_sq8.add(&base, None)?;
    println!("build_time={:.2}s", t.elapsed().as_secs_f64());

    for &nprobe in &IVF_NPROBE_SWEEP {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let t_batch = Instant::now();
        queries.par_chunks_exact(base_dim).for_each(|q| {
            let _ = ivf_sq8.search(q, &req);
        });
        let qps_batch = query_n as f64 / t_batch.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut results = Vec::with_capacity(eval_n);
        for q in queries.chunks_exact(base_dim).take(eval_n) {
            let res = ivf_sq8.search(q, &req)?;
            results.push(normalize_ids(&res.ids, TOP_K));
        }
        let recall = compute_recall(&results, &gt[..eval_n], TOP_K);
        println!(
            "nprobe={:>3} recall@10={:.4} batch_qps={:.0}",
            nprobe, recall, qps_batch
        );
    }

    Ok(())
}

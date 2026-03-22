//! HNSW SIFT-1M authority benchmark
//! Usage:
//!   cargo run --example hnsw_sift1m --release -- <data_dir>

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use rayon::prelude::*;

const TOP_K: usize = 10;
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const LEVEL_MULTIPLIER: f32 = 0.5;
const EF_SWEEP: [usize; 6] = [16, 32, 50, 60, 100, 138];
const QPS_QUERIES: usize = 1000;
const RECALL_QUERIES: usize = 200;

fn resolve_file(data_dir: &Path, names: &[&str]) -> Result<PathBuf, Box<dyn Error>> {
    for name in names {
        let p = data_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(format!("missing file in {}: tried {:?}", data_dir.display(), names).into())
}

fn read_fvecs(path: &Path) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut data = Vec::<f32>::new();
    let mut n = 0usize;
    let mut dim_opt: Option<usize> = None;
    let mut raw = Vec::<u8>::new();

    loop {
        let mut dbytes = [0u8; 4];
        match reader.read_exact(&mut dbytes) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        let dim = i32::from_le_bytes(dbytes);
        if dim <= 0 {
            return Err(format!("invalid dim {} in {}", dim, path.display()).into());
        }
        let dim = dim as usize;

        if let Some(expected) = dim_opt {
            if expected != dim {
                return Err(format!(
                    "inconsistent dim in {}: expected {}, got {} at vector {}",
                    path.display(),
                    expected,
                    dim,
                    n
                )
                .into());
            }
        } else {
            dim_opt = Some(dim);
            raw.resize(dim * 4, 0);
        }

        reader.read_exact(&mut raw)?;
        for c in raw.chunks_exact(4) {
            data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        n += 1;
    }

    Ok((data, dim_opt.unwrap_or(0), n))
}

fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut out: Vec<Vec<i32>> = Vec::new();

    loop {
        let mut dbytes = [0u8; 4];
        match reader.read_exact(&mut dbytes) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        let dim = i32::from_le_bytes(dbytes);
        if dim <= 0 {
            return Err(format!("invalid ivecs dim {} in {}", dim, path.display()).into());
        }
        let dim = dim as usize;

        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut ib = [0u8; 4];
            reader.read_exact(&mut ib)?;
            row.push(i32::from_le_bytes(ib));
        }
        out.push(row);
    }

    Ok(out)
}

fn recall_at_k(gt_row: &[i32], result_ids: &[i64], k: usize) -> f64 {
    let k = k.min(gt_row.len()).min(result_ids.len());
    if k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    for &gid in gt_row.iter().take(k) {
        if result_ids.iter().take(k).any(|&rid| rid == gid as i64) {
            hits += 1;
        }
    }
    hits as f64 / k as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example hnsw_sift1m --release -- <data_dir>");
        std::process::exit(2);
    }

    let data_dir = Path::new(&args[1]);
    let base_path = resolve_file(data_dir, &["sift_base.fvecs", "base.fvecs"])?;
    let query_path = resolve_file(data_dir, &["sift_query.fvecs", "query.fvecs"])?;
    let gt_path = resolve_file(data_dir, &["sift_groundtruth.ivecs", "groundtruth.ivecs"])?;

    println!("=== HNSW SIFT-1M authority benchmark ===");
    println!("data_dir={}", data_dir.display());
    println!(
        "files: base={}, query={}, gt={}",
        base_path.display(),
        query_path.display(),
        gt_path.display()
    );

    let t_load = Instant::now();
    let (base, base_dim, base_n) = read_fvecs(&base_path)?;
    let (queries, query_dim, query_n) = read_fvecs(&query_path)?;
    let gt = read_ivecs(&gt_path)?;
    println!(
        "loaded: base_n={} query_n={} gt_n={} dim={} (load {:.2}s)",
        base_n,
        query_n,
        gt.len(),
        base_dim,
        t_load.elapsed().as_secs_f64()
    );

    if base_dim == 0 || base_dim != query_dim {
        return Err(format!(
            "dim mismatch: base_dim={} query_dim={}",
            base_dim, query_dim
        )
        .into());
    }

    let recall_n = RECALL_QUERIES.min(query_n).min(gt.len());
    let qps_n = QPS_QUERIES.min(query_n);

    let mut params = IndexParams::hnsw(EF_CONSTRUCTION, 60, LEVEL_MULTIPLIER);
    params.m = Some(M);

    let cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: base_dim,
        params,
    };

    let mut index = HnswIndex::new(&cfg)?;

    let t0 = Instant::now();
    index.train(&base)?;
    index.add(&base, None)?;
    let build_s = t0.elapsed().as_secs_f64();

    println!(
        "build: {:.2}s (M={}, ef_construction={})",
        build_s, M, EF_CONSTRUCTION
    );
    println!("ef sweep (qps on {} queries, recall on {} queries):", qps_n, recall_n);

    let mut ef60: Option<(u64, f64)> = None;
    let mut ef138: Option<(u64, f64)> = None;

    for &ef in &EF_SWEEP {
        index.set_ef_search(ef);
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            params: Some(format!(r#"{{\"ef\": {ef}}}"#)),
            ..Default::default()
        };

        let t_qps = Instant::now();
        for i in 0..qps_n {
            let q = &queries[i * base_dim..(i + 1) * base_dim];
            let _ = index.search(q, &req)?;
        }
        let qps = (qps_n as f64 / t_qps.elapsed().as_secs_f64().max(f64::EPSILON)).round() as u64;

        let mut recall_sum = 0.0;
        for i in 0..recall_n {
            let q = &queries[i * base_dim..(i + 1) * base_dim];
            let res = index.search(q, &req)?;
            recall_sum += recall_at_k(&gt[i], &res.ids, TOP_K);
        }
        let recall = recall_sum / recall_n as f64;

        println!("ef={:>4} recall@10={:.4} qps={}", ef, recall, qps);

        if ef == 60 {
            ef60 = Some((qps, recall));
        } else if ef == 138 {
            ef138 = Some((qps, recall));
        }
    }

    if let Some((qps, recall)) = ef60 {
        println!("marker ef=60: qps={} recall@10={:.4}", qps, recall);
    }
    if let Some((qps, recall)) = ef138 {
        println!("marker ef=138: qps={} recall@10={:.4}", qps, recall);
    }

    println!("batch parallel sweep (all {} queries):", query_n);
    let mut ef60_parallel_qps: Option<u64> = None;
    let mut ef138_parallel_qps: Option<u64> = None;
    for &ef in &[60usize, 138usize] {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: ef,
            params: Some(format!(r#"{{\"ef\": {ef}}}"#)),
            ..Default::default()
        };

        let t = Instant::now();
        queries
            .par_chunks(base_dim)
            .for_each(|q| {
                let _ = index.search(q, &req);
            });
        let qps = (query_n as f64 / t.elapsed().as_secs_f64().max(f64::EPSILON)).round() as u64;
        println!("batch ef={:>4} qps={}", ef, qps);

        if ef == 60 {
            ef60_parallel_qps = Some(qps);
        } else if ef == 138 {
            ef138_parallel_qps = Some(qps);
        }
    }

    if let Some(qps) = ef60_parallel_qps {
        println!("marker batch ef=60: qps={}", qps);
    }
    if let Some(qps) = ef138_parallel_qps {
        println!("marker batch ef=138: qps={}", qps);
    }

    Ok(())
}

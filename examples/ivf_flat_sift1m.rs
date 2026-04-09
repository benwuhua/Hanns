//! IVF-Flat SIFT-1M authority benchmark
//! Usage:
//!   cargo run --example ivf_flat_sift1m --release -- <data_dir>
//! data_dir must contain either:
//!   - sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs
//! or:
//!   - base.fvecs, query.fvecs, groundtruth.ivecs

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::IvfFlatIndex;

const TOP_K: usize = 10;
const NLIST: usize = 1024;
const NPROBES: [usize; 6] = [1, 4, 8, 16, 32, 64];

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

    let mut data: Vec<f32> = Vec::new();
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
        eprintln!("Usage: cargo run --example ivf_flat_sift1m --release -- <data_dir>");
        std::process::exit(2);
    }

    let data_dir = Path::new(&args[1]);
    let base_path = resolve_file(data_dir, &["sift_base.fvecs", "base.fvecs"])?;
    let query_path = resolve_file(data_dir, &["sift_query.fvecs", "query.fvecs"])?;
    let gt_path = resolve_file(data_dir, &["sift_groundtruth.ivecs", "groundtruth.ivecs"])?;

    println!("=== IVF-Flat SIFT-1M Benchmark ===");
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

    let recall_queries = query_n.min(gt.len());
    let qps_queries = query_n;

    let mut params = IndexParams::default();
    params.nlist = Some(NLIST);
    params.nprobe = Some(32);

    let cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: base_dim,
        params,
    };

    let mut index = IvfFlatIndex::new(&cfg)?;

    let t0 = Instant::now();
    index.train(&base)?;
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(&base, None)?;
    let add_s = t1.elapsed().as_secs_f64();

    println!(
        "build: train={:.2}s add={:.2}s total={:.2}s",
        train_s,
        add_s,
        train_s + add_s
    );

    println!("single-query sweep:");
    println!("nprobe | recall@10 | qps");
    println!("-------|-----------|------");

    for &nprobe in &NPROBES {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let qps_start = Instant::now();
        for i in 0..qps_queries {
            let q = &queries[i * base_dim..(i + 1) * base_dim];
            let _ = index.search(q, &req)?;
        }
        let qps = qps_queries as f64 / qps_start.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut recall_sum = 0.0;
        for i in 0..recall_queries {
            let q = &queries[i * base_dim..(i + 1) * base_dim];
            let res = index.search(q, &req)?;
            recall_sum += recall_at_k(&gt[i], &res.ids, TOP_K);
        }
        let recall = recall_sum / recall_queries as f64;

        println!("{:>6} | {:>9.3} | {:.0}", nprobe, recall, qps);
    }

    #[cfg(feature = "parallel")]
    {
        let t_batch = Instant::now();
        let batch = index.search_parallel(&queries, TOP_K, 32)?;
        let batch_qps = qps_queries as f64 / t_batch.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut recall_sum = 0.0;
        for i in 0..recall_queries {
            let ids: Vec<i64> = batch[i]
                .iter()
                .take(TOP_K)
                .map(|(id, _)| *id as i64)
                .collect();
            recall_sum += recall_at_k(&gt[i], &ids, TOP_K);
        }
        let recall = recall_sum / recall_queries as f64;

        println!(
            "batch parallel (nprobe=32): recall@10={:.3}, qps={:.0}",
            recall, batch_qps
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("batch parallel (nprobe=32): skipped (feature parallel not enabled)");
    }

    Ok(())
}

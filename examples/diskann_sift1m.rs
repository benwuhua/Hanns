//! DiskANN/PQFlashIndex SIFT-1M authority benchmark
//! Usage:
//!   cargo run --example diskann_sift1m --release -- <data_dir>
//!
//! data_dir must contain:
//!   - sift_learn.fvecs
//!   - sift_base.fvecs
//!   - sift_query.fvecs
//!   - sift_groundtruth.ivecs

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use knowhere_rs::api::MetricType;
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};

const TOP_K: usize = 10;
const RECALL_QUERIES: usize = 1_000;

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

#[derive(Debug, Clone)]
struct RunMetrics {
    train_s: f64,
    add_s: f64,
    save_s: f64,
    load_s: f64,
    recall: f64,
    qps: f64,
}

fn unique_temp_dir(tag: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    std::env::temp_dir().join(format!("diskann_sift1m_{}_{}", tag, ts))
}

fn run_one_config(
    cfg: AisaqConfig,
    label: &str,
    dim: usize,
    learn: &[f32],
    base: &[f32],
    queries: &[f32],
    gt: &[Vec<i32>],
) -> Result<RunMetrics, Box<dyn Error>> {
    let mut index = PQFlashIndex::new(cfg.clone(), MetricType::L2, dim)?;

    let t0 = Instant::now();
    index.train(learn)?;
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(base)?;
    let add_s = t1.elapsed().as_secs_f64();

    let save_dir = unique_temp_dir(label);
    let t2 = Instant::now();
    let _ = index.save(&save_dir)?;
    let save_s = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let loaded = PQFlashIndex::load(&save_dir)?;
    let load_s = t3.elapsed().as_secs_f64();

    let query_n = queries.len() / dim;
    let t4 = Instant::now();
    let batch = loaded.search_batch(queries, TOP_K)?;
    let qps = query_n as f64 / t4.elapsed().as_secs_f64().max(f64::EPSILON);

    let recall_n = RECALL_QUERIES.min(query_n).min(gt.len());
    let mut recall_sum = 0.0;
    for (i, gt_row) in gt.iter().enumerate().take(recall_n) {
        let s = i * TOP_K;
        let e = s + TOP_K;
        recall_sum += recall_at_k(gt_row, &batch.ids[s..e], TOP_K);
    }
    let recall = if recall_n > 0 {
        recall_sum / recall_n as f64
    } else {
        0.0
    };

    let _ = std::fs::remove_dir_all(&save_dir);

    Ok(RunMetrics {
        train_s,
        add_s,
        save_s,
        load_s,
        recall,
        qps,
    })
}

fn search_at_l(
    index: &mut PQFlashIndex,
    queries: &[f32],
    gt: &[Vec<i32>],
    dim: usize,
    l: usize,
) -> Result<(f64, f64), Box<dyn Error>> {
    index.set_search_list_size(l);

    let query_n = queries.len() / dim;
    let t = Instant::now();
    let batch = index.search_batch(queries, TOP_K)?;
    let qps = query_n as f64 / t.elapsed().as_secs_f64().max(f64::EPSILON);

    let recall_n = RECALL_QUERIES.min(query_n).min(gt.len());
    let mut recall_sum = 0.0;
    for (i, gt_row) in gt.iter().enumerate().take(recall_n) {
        let s = i * TOP_K;
        let e = s + TOP_K;
        recall_sum += recall_at_k(gt_row, &batch.ids[s..e], TOP_K);
    }
    let recall = if recall_n > 0 {
        recall_sum / recall_n as f64
    } else {
        0.0
    };

    Ok((recall, qps))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example diskann_sift1m --release -- <data_dir>");
        std::process::exit(2);
    }

    let data_dir = Path::new(&args[1]);
    let learn_path = data_dir.join("sift_learn.fvecs");
    let base_path = data_dir.join("sift_base.fvecs");
    let query_path = data_dir.join("sift_query.fvecs");
    let gt_path = data_dir.join("sift_groundtruth.ivecs");

    println!("=== DiskANN/PQFlashIndex SIFT-1M Benchmark ===");
    println!(
        "files: learn={}, base={}, query={}, gt={}",
        learn_path.display(),
        base_path.display(),
        query_path.display(),
        gt_path.display()
    );

    let t_load = Instant::now();
    let (learn, learn_dim, learn_n) = read_fvecs(&learn_path)?;
    let (base, base_dim, base_n) = read_fvecs(&base_path)?;
    let (queries, query_dim, query_n) = read_fvecs(&query_path)?;
    let gt = read_ivecs(&gt_path)?;
    println!(
        "loaded: learn_n={} base_n={} query_n={} gt_n={} dim={} (load {:.2}s)",
        learn_n,
        base_n,
        query_n,
        gt.len(),
        base_dim,
        t_load.elapsed().as_secs_f64()
    );

    if base_dim == 0 || learn_dim != base_dim || query_dim != base_dim {
        return Err(format!(
            "dim mismatch: learn={} base={} query={}",
            learn_dim, base_dim, query_dim
        )
        .into());
    }

    let dim = base_dim;

    println!("--- NoPQ (disk_pq_dims=0) ---");
    let nopq_cfg = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 128,
        cache_all_on_load: true,
        ..AisaqConfig::default()
    };

    let mut build_cfg = nopq_cfg.clone();
    build_cfg.search_list_size = 128;
    let mut index = PQFlashIndex::new(build_cfg, MetricType::L2, dim)?;

    let t0 = Instant::now();
    index.train(&learn)?;
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(&base)?;
    let add_s = t1.elapsed().as_secs_f64();

    let save_dir = unique_temp_dir("nopq_shared_build");
    let t2 = Instant::now();
    let _ = index.save(&save_dir)?;
    let save_s = t2.elapsed().as_secs_f64();

    let t3 = Instant::now();
    let mut loaded = PQFlashIndex::load(&save_dir)?;
    let load_s = t3.elapsed().as_secs_f64();

    println!(
        "build: train={:.2}s add={:.2}s total={:.2}s",
        train_s,
        add_s,
        train_s + add_s
    );
    println!("save: {:.2}s  load: {:.2}s", save_s, load_s);

    for &l in &[64usize, 128usize, 200usize] {
        let (recall, qps) = search_at_l(&mut loaded, &queries, &gt, dim, l)?;
        println!("L={}:  recall@10={:.3} QPS={:.0}", l, recall, qps);
    }
    let _ = std::fs::remove_dir_all(&save_dir);

    println!("--- PQ32 (disk_pq_dims=32) ---");
    let pq32_cfg = AisaqConfig {
        disk_pq_dims: 32,
        search_list_size: 128,
        run_refine_pass: true,
        rerank_expand_pct: 200,
        cache_all_on_load: true,
        ..AisaqConfig::default()
    };
    let pq32 = run_one_config(pq32_cfg, "pq32_l128", dim, &learn, &base, &queries, &gt)?;
    println!(
        "build: train={:.2}s add={:.2}s total={:.2}s",
        pq32.train_s,
        pq32.add_s,
        pq32.train_s + pq32.add_s
    );
    println!("save: {:.2}s  load: {:.2}s", pq32.save_s, pq32.load_s);
    println!("L=128: recall@10={:.3} QPS={:.0}", pq32.recall, pq32.qps);

    println!("--- NoPQ+SQ8 (use_sq8_prefilter=true) ---");
    let sq8_cfg = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 128,
        cache_all_on_load: true,
        use_sq8_prefilter: true,
        ..AisaqConfig::default()
    };
    let mut sq8_idx = PQFlashIndex::new(sq8_cfg, MetricType::L2, dim)?;

    let t0 = Instant::now();
    sq8_idx.train(&learn)?;
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    sq8_idx.add(&base)?;
    let add_s = t1.elapsed().as_secs_f64();
    println!(
        "build: train={:.2}s add={:.2}s total={:.2}s",
        train_s,
        add_s,
        train_s + add_s
    );
    let sq8_dir = unique_temp_dir("nopq_sq8");
    sq8_idx.save(&sq8_dir)?;
    let mut sq8_loaded = PQFlashIndex::load(&sq8_dir)?;

    for &l in &[64usize, 128usize, 200usize] {
        let (recall, qps) = search_at_l(&mut sq8_loaded, &queries, &gt, dim, l)?;
        println!("L={}:  recall@10={:.3} QPS={:.0}", l, recall, qps);
    }
    let _ = std::fs::remove_dir_all(&sq8_dir);

    Ok(())
}

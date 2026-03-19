use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfFlatIndex, IvfPqIndex, IvfSq8Index};

const DEFAULT_SIFT_ROOT: &str = "/data/work/datasets/sift-1m";
const DEFAULT_TRAIN_SIZE: usize = 100_000;
const DEFAULT_BASE_LIMIT: usize = 1_000_000;
const DEFAULT_QUERY_LIMIT: usize = 10_000;
const DIM_EXPECTED: usize = 128;
const TOP_K: usize = 10;
const NLIST: usize = 256;

fn read_u32_le(file: &mut File) -> IoResult<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_fbin(path: &str) -> IoResult<(usize, usize, Vec<f32>)> {
    let mut file = File::open(path)?;
    let n = read_u32_le(&mut file)? as usize;
    let dim = read_u32_le(&mut file)? as usize;

    let mut bytes = vec![0u8; n * dim * std::mem::size_of::<f32>()];
    file.read_exact(&mut bytes)?;

    let mut data = Vec::with_capacity(n * dim);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((n, dim, data))
}

fn read_ibin(path: &str) -> IoResult<(usize, usize, Vec<i32>)> {
    let mut file = File::open(path)?;
    let n = read_u32_le(&mut file)? as usize;
    let k = read_u32_le(&mut file)? as usize;

    let mut bytes = vec![0u8; n * k * std::mem::size_of::<i32>()];
    file.read_exact(&mut bytes)?;

    let mut data = Vec::with_capacity(n * k);
    for chunk in bytes.chunks_exact(4) {
        data.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((n, k, data))
}

fn recall_at_k(gt: &[i32], results: &[i64], k: usize, n_queries: usize, gt_width: usize) -> f32 {
    let mut hits = 0usize;
    for q in 0..n_queries {
        let gt_row = &gt[q * gt_width..q * gt_width + k.min(gt_width)];
        let result_row = &results[q * k..q * k + k.min(results.len().saturating_sub(q * k))];
        for &gt_id in gt_row {
            if result_row.iter().any(|&id| id == gt_id as i64) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n_queries * k) as f32
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn dataset_root() -> String {
    std::env::var("SIFT1M_ROOT").unwrap_or_else(|_| DEFAULT_SIFT_ROOT.to_string())
}

fn make_cfg(index_type: IndexType, nlist: usize, nprobe: usize, m: Option<usize>) -> IndexConfig {
    let mut cfg = IndexConfig::new(index_type, MetricType::L2, DIM_EXPECTED);
    cfg.params = IndexParams {
        nlist: Some(nlist),
        nprobe: Some(nprobe),
        m,
        nbits_per_idx: Some(8),
        ..IndexParams::default()
    };
    cfg
}

fn print_result(
    name: &str,
    nlist: usize,
    nprobe: usize,
    m: Option<usize>,
    recall: f32,
    qps: f64,
) {
    match m {
        Some(m) => println!(
            "{} nlist={} nprobe={} m={}: recall@10={:.3}, QPS={:.0}",
            name, nlist, nprobe, m, recall, qps
        ),
        None => println!(
            "{} nlist={} nprobe={}: recall@10={:.3}, QPS={:.0}",
            name, nlist, nprobe, recall, qps
        ),
    }
}

fn run_ivf_flat(
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    gt: &[i32],
    n_queries: usize,
    gt_width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = make_cfg(IndexType::IvfFlat, NLIST, 1, None);
    let mut index = IvfFlatIndex::new(&cfg)?;
    index.train(train)?;
    index.add(base, None)?;

    for &nprobe in &[16usize, 32, 64, 128, 256] {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let start = Instant::now();
        let mut all_ids = Vec::with_capacity(n_queries * TOP_K);
        for q in 0..n_queries {
            let query = &queries[q * DIM_EXPECTED..(q + 1) * DIM_EXPECTED];
            let result = index.search(query, &req)?;
            all_ids.extend_from_slice(&result.ids[..TOP_K.min(result.ids.len())]);
            for _ in result.ids.len()..TOP_K {
                all_ids.push(-1);
            }
        }
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = recall_at_k(gt, &all_ids, TOP_K, n_queries, gt_width);
        let qps = n_queries as f64 / elapsed;
        print_result("IVF-Flat", NLIST, nprobe, None, recall, qps);
    }
    Ok(())
}

fn run_ivf_sq8(
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    gt: &[i32],
    n_queries: usize,
    gt_width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = make_cfg(IndexType::IvfSq8, NLIST, 1, None);
    let mut index = IvfSq8Index::new(&cfg)?;
    index.train(train)?;
    index.add(base, None)?;

    for &nprobe in &[32usize, 64, 128, 256] {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let start = Instant::now();
        let result = index.search(queries, &req)?;
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = recall_at_k(gt, &result.ids, TOP_K, n_queries, gt_width);
        let qps = n_queries as f64 / elapsed;
        print_result("IVF-SQ8", NLIST, nprobe, None, recall, qps);
    }
    Ok(())
}

fn run_ivf_pq(
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    gt: &[i32],
    n_queries: usize,
    gt_width: usize,
    m: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = make_cfg(IndexType::IvfPq, NLIST, 1, Some(m));
    let mut index = IvfPqIndex::new(&cfg)?;
    index.train(train)?;
    index.add(base, None)?;

    for &nprobe in &[32usize, 64, 128, 256] {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let start = Instant::now();
        let result = index.search(queries, &req)?;
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = recall_at_k(gt, &result.ids, TOP_K, n_queries, gt_width);
        let qps = n_queries as f64 / elapsed;
        print_result("IVF-PQ", NLIST, nprobe, Some(m), recall, qps);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = dataset_root();
    let train_size = env_usize("SIFT1M_TRAIN_SIZE", DEFAULT_TRAIN_SIZE);
    let base_limit = env_usize("SIFT1M_BASE_LIMIT", DEFAULT_BASE_LIMIT);
    let query_limit = env_usize("SIFT1M_QUERY_LIMIT", DEFAULT_QUERY_LIMIT);

    let base_path = format!("{root}/base.fbin");
    let query_path = format!("{root}/query.fbin");
    let gt_path = format!("{root}/gt.ibin");

    println!("=== SIFT1M IVF recall benchmark ===");
    println!("root={root}");
    println!(
        "requested limits: train_size={}, base_limit={}, query_limit={}",
        train_size, base_limit, query_limit
    );

    let (base_n, base_dim, base_all) = read_fbin(&base_path)?;
    let (query_n, query_dim, query_all) = read_fbin(&query_path)?;
    let (gt_n, gt_width, gt_all) = read_ibin(&gt_path)?;

    if base_dim != DIM_EXPECTED || query_dim != DIM_EXPECTED {
        return Err(format!(
            "unexpected dim: base_dim={}, query_dim={}, expected={}",
            base_dim, query_dim, DIM_EXPECTED
        )
        .into());
    }
    if gt_n != query_n {
        return Err(format!("gt query count {} != query count {}", gt_n, query_n).into());
    }

    let used_base = base_limit.min(base_n);
    let used_queries = query_limit.min(query_n);
    let used_train = train_size.min(used_base);

    let base = &base_all[..used_base * DIM_EXPECTED];
    let train = &base_all[..used_train * DIM_EXPECTED];
    let queries = &query_all[..used_queries * DIM_EXPECTED];
    let gt = &gt_all[..used_queries * gt_width];

    println!(
        "loaded: base={} query={} gt_queries={} gt_width={}",
        used_base, used_queries, used_queries, gt_width
    );

    run_ivf_flat(base, train, queries, gt, used_queries, gt_width)?;
    run_ivf_sq8(base, train, queries, gt, used_queries, gt_width)?;
    for &m in &[8usize, 16, 32] {
        run_ivf_pq(base, train, queries, gt, used_queries, gt_width, m)?;
    }

    Ok(())
}

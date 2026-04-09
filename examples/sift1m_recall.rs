use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::time::Instant;

use hanns::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{IvfFlatIndex, IvfPqIndex, IvfSq8Index};
use hanns::quantization::pq::{PQConfig, ProductQuantizer};

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

fn print_result(name: &str, nlist: usize, nprobe: usize, m: Option<usize>, recall: f32, qps: f64) {
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

        #[cfg(feature = "parallel")]
        {
            let start = Instant::now();
            let batch_results = index.search_parallel(queries, TOP_K, nprobe)?;
            let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);

            let mut all_ids_parallel = Vec::with_capacity(n_queries * TOP_K);
            for row in batch_results.iter().take(n_queries) {
                for &(id, _) in row.iter().take(TOP_K) {
                    all_ids_parallel.push(id as i64);
                }
                for _ in row.len()..TOP_K {
                    all_ids_parallel.push(-1);
                }
            }
            let recall_parallel = recall_at_k(gt, &all_ids_parallel, TOP_K, n_queries, gt_width);
            let qps_parallel = n_queries as f64 / elapsed;
            println!(
                "IVF-Flat-Parallel nlist={} nprobe={}: recall@10={:.3}, QPS={:.0}",
                NLIST, nprobe, recall_parallel, qps_parallel
            );
        }
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
    let train_start = Instant::now();
    index.train(train)?;
    let train_elapsed = train_start.elapsed().as_secs_f64();
    let add_start = Instant::now();
    index.add(base, None)?;
    let add_elapsed = add_start.elapsed().as_secs_f64();
    println!("IVF-PQ m={m} train={train_elapsed:.1}s add={add_elapsed:.1}s");

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

fn run_ivf_pq_plain_diagnostic(
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    gt: &[i32],
    n_queries: usize,
    gt_width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Plain IVF-PQ (IVFPQ_NO_OPQ=1) diagnostic ---");
    unsafe { std::env::set_var("IVFPQ_NO_OPQ", "1") };

    for &m in &[8usize, 16, 32] {
        let cfg = make_cfg(IndexType::IvfPq, NLIST, 1, Some(m));
        let mut index = IvfPqIndex::new(&cfg)?;
        index.train(train)?;
        index.add(base, None)?;

        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: 64,
            ..Default::default()
        };

        let start = Instant::now();
        let result = index.search(queries, &req)?;
        let elapsed = start.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = recall_at_k(gt, &result.ids, TOP_K, n_queries, gt_width);
        let qps = n_queries as f64 / elapsed;
        print_result("IVF-PQ-Plain", NLIST, 64, Some(m), recall, qps);
    }

    unsafe { std::env::remove_var("IVFPQ_NO_OPQ") };
    Ok(())
}

fn run_pq_only_diagnostic(
    base: &[f32],
    train: &[f32],
    queries: &[f32],
    gt: &[i32],
    n_queries: usize,
    gt_width: usize,
    m: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let base_n = base.len() / DIM_EXPECTED;
    let diag_base_n = base_n; // Use full 1M base so GT is valid
    let diag_queries = n_queries.min(100); // 100 × 1M ADC comparisons
    let diag_base = &base[..diag_base_n * DIM_EXPECTED];
    let diag_query = &queries[..diag_queries * DIM_EXPECTED];
    let diag_gt = &gt[..diag_queries * gt_width];

    let config = PQConfig::new(DIM_EXPECTED, m, 8);
    let mut pq = ProductQuantizer::new(config);
    pq.train(train.len() / DIM_EXPECTED, train)?;

    // Reconstruction MSE on first 100 training vectors
    let sample_n = (train.len() / DIM_EXPECTED).min(100);
    let mut mse_sum = 0.0f64;
    let mut mse_count = 0usize;
    for i in 0..sample_n {
        let x = &train[i * DIM_EXPECTED..(i + 1) * DIM_EXPECTED];
        let code = pq.encode(x)?;
        let recon = pq.decode(&code)?;
        for d in 0..DIM_EXPECTED {
            let diff = x[d] - recon[d];
            mse_sum += (diff * diff) as f64;
            mse_count += 1;
        }
    }
    let mse = if mse_count == 0 {
        0.0
    } else {
        (mse_sum / mse_count as f64) as f32
    };

    let codes = pq.encode_batch(diag_base_n, diag_base)?;
    let code_size = pq.code_size();
    let mut all_ids = Vec::with_capacity(diag_queries * TOP_K);

    for q in 0..diag_queries {
        let qv = &diag_query[q * DIM_EXPECTED..(q + 1) * DIM_EXPECTED];
        let mut scored = Vec::with_capacity(diag_base_n);
        for i in 0..diag_base_n {
            let code = &codes[i * code_size..(i + 1) * code_size];
            let dist = pq.compute_distance(qv, code);
            scored.push((i, dist));
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (idx, _) in scored.into_iter().take(TOP_K) {
            all_ids.push(idx as i64);
        }
    }

    let recall = recall_at_k(diag_gt, &all_ids, TOP_K, diag_queries, gt_width);
    println!(
        "PQ-Only m={} reconstruction_mse={:.4}: recall@10={:.3}",
        m, mse, recall
    );
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

    let ivfpq_only = std::env::var("IVFPQ_ONLY").is_ok();
    if !ivfpq_only {
        run_ivf_flat(base, train, queries, gt, used_queries, gt_width)?;
        run_ivf_sq8(base, train, queries, gt, used_queries, gt_width)?;
    }
    for &m in &[8usize, 16, 32] {
        run_ivf_pq(base, train, queries, gt, used_queries, gt_width, m)?;
    }
    if !ivfpq_only {
        run_ivf_pq_plain_diagnostic(base, train, queries, gt, used_queries, gt_width)?;
    } else {
        for &m in &[8usize, 16, 32] {
            run_pq_only_diagnostic(base, train, queries, gt, used_queries, gt_width, m)?;
        }
    }

    Ok(())
}

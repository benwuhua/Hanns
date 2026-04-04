//! Wikipedia-Cohere 1M IVF-PQ benchmark (IP, dim=768)
//! Usage:
//!   cargo run --example ivf_pq_cohere1m --release -- <optional_data_dir>

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::IvfPqIndex;

const DEFAULT_DATA_DIR: &str = "/data/work/datasets/wikipedia-cohere-1m";
const TOP_K: usize = 10;
const EXPECTED_DIM: usize = 768;
const NLIST: usize = 1024;
const TRAIN_SIZE: usize = 100_000;
const QPS_QUERIES: usize = 1_000;
const M_SWEEP: [usize; 2] = [32, 48];
const NPROBE_SWEEP: [usize; 2] = [32, 64];

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
    flat.chunks_exact(width).map(|row| row.to_vec()).collect()
}

fn measure_qps(
    index: &IvfPqIndex,
    queries: &[f32],
    dim: usize,
    nprobe: usize,
    qps_queries: usize,
) -> Result<f64, Box<dyn Error>> {
    if qps_queries == 0 {
        return Ok(0.0);
    }

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe,
        ..Default::default()
    };

    let warmup = &queries[..dim];
    let _ = index.search(warmup, &req)?;

    let start = Instant::now();
    for query in queries.chunks_exact(dim).take(qps_queries) {
        let _ = index.search(query, &req)?;
    }
    let secs = start.elapsed().as_secs_f64().max(f64::EPSILON);
    Ok(qps_queries as f64 / secs)
}

fn measure_recall(
    index: &IvfPqIndex,
    queries: &[f32],
    gt: &[Vec<i32>],
    dim: usize,
    nprobe: usize,
) -> Result<f32, Box<dyn Error>> {
    let eval_n = (queries.len() / dim).min(gt.len());
    if eval_n == 0 {
        return Ok(0.0);
    }

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe,
        ..Default::default()
    };

    let mut hits = 0usize;
    for (qi, gt_row) in gt.iter().take(eval_n).enumerate() {
        let query = &queries[qi * dim..(qi + 1) * dim];
        let result = index.search(query, &req)?;
        let rk = TOP_K.min(result.ids.len());
        let gk = TOP_K.min(gt_row.len());
        for &gid in gt_row.iter().take(gk) {
            if result.ids.iter().take(rk).any(|&rid| rid == gid as i64) {
                hits += 1;
            }
        }
    }

    Ok(hits as f32 / (eval_n * TOP_K) as f32)
}

fn make_pq_config(dim: usize, m: usize) -> IndexConfig {
    IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::Ip,
        data_type: DataType::Float,
        dim,
        params: IndexParams {
            nlist: Some(NLIST),
            nprobe: Some(NPROBE_SWEEP[0]),
            m: Some(m),
            nbits_per_idx: Some(8),
            ..Default::default()
        },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir_arg = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DATA_DIR.to_string());
    let data_dir = Path::new(&data_dir_arg);
    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.ibin");

    if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
        return Err(format!(
            "missing dataset files under {} (need base.fbin/query.fbin/gt.ibin)",
            data_dir.display()
        )
        .into());
    }

    println!("=== Cohere 1M IVF-PQ benchmark ===");
    println!("data_dir={}", data_dir.display());
    println!("metric=IP nlist={} train_size={}", NLIST, TRAIN_SIZE);

    let load_start = Instant::now();
    let (base_n, base_dim, base) = read_fbin(&base_path)?;
    let (query_n, query_dim, queries) = read_fbin(&query_path)?;
    let (gt_n, gt_k, gt_flat) = read_ibin(&gt_path)?;
    let gt = rows_i32(&gt_flat, gt_k);
    println!(
        "loaded: base_n={} query_n={} gt_n={} dim={} gt_k={} load_s={:.2}",
        base_n,
        query_n,
        gt_n,
        base_dim,
        gt_k,
        load_start.elapsed().as_secs_f64()
    );

    if base_dim == 0 || base_dim != query_dim {
        return Err(format!("dim mismatch: base_dim={} query_dim={}", base_dim, query_dim).into());
    }
    if gt_n != query_n {
        return Err(format!("gt rows {} != query rows {}", gt_n, query_n).into());
    }
    if gt_k < TOP_K {
        return Err(format!("gt_k={} < top_k={}", gt_k, TOP_K).into());
    }
    if base_dim != EXPECTED_DIM {
        println!(
            "warning: expected dim={} but got {}, continuing",
            EXPECTED_DIM, base_dim
        );
    }

    let train_n = TRAIN_SIZE.min(base_n);
    let train = &base[..train_n * base_dim];
    let qps_queries = QPS_QUERIES.min(query_n);
    let eval_queries = query_n.min(gt.len());

    println!(
        "setup: train_n={} eval_queries={} qps_queries={}",
        train_n, eval_queries, qps_queries
    );

    for &m in &M_SWEEP {
        if base_dim % m != 0 {
            return Err(format!("invalid m={} for dim={}", m, base_dim).into());
        }

        let config = make_pq_config(base_dim, m);
        let mut index = IvfPqIndex::new(&config)?;

        let build_start = Instant::now();
        index.train(train)?;
        index.add(&base, None)?;
        println!("build: m={} build_s={:.2}", m, build_start.elapsed().as_secs_f64());

        for &nprobe in &NPROBE_SWEEP {
            let recall = measure_recall(
                &index,
                &queries[..eval_queries * base_dim],
                &gt[..eval_queries],
                base_dim,
                nprobe,
            )?;
            let qps = measure_qps(&index, &queries, base_dim, nprobe, qps_queries)?;
            println!("m={} nprobe={} recall={:.3} qps={:.0}", m, nprobe, recall, qps);
        }
    }

    Ok(())
}

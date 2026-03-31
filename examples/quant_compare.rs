use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{
    IvfExRaBitqConfig, IvfExRaBitqIndex, IvfFlatIndex, IvfHvqConfig, IvfHvqIndex, IvfPqIndex,
};
use rayon::prelude::*;

const DEFAULT_DATA_DIR: &str = "/data/work/datasets/wikipedia-cohere-1m";
const EXPECTED_DIM: usize = 768;
const NLIST: usize = 256;
const TOP_K: usize = 10;
const TRAIN_SIZE: usize = 1_000_000;
const EVAL_QUERIES: usize = 5_000;
const NPROBES: [usize; 3] = [10, 32, 64];

struct Tier {
    label: &'static str,
    pq_m: usize,
    pq_nbits: usize,
    hvq_bits: u8,
    exrabitq_bits: usize,
}

const TIERS: [Tier; 3] = [
    Tier {
        label: "32x",
        pq_m: 96,
        pq_nbits: 8,
        hvq_bits: 1,
        exrabitq_bits: 3,
    },
    Tier {
        label: "8x",
        pq_m: 384,
        pq_nbits: 8,
        hvq_bits: 4,
        exrabitq_bits: 5,
    },
    Tier {
        label: "4x",
        pq_m: 768,
        pq_nbits: 8,
        hvq_bits: 8,
        exrabitq_bits: 9,
    },
];

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
            if r.iter()
                .take(top_k.min(r.len()))
                .any(|&rid| rid == gid as i64)
            {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn normalize_vectors(data: &mut [f32], dim: usize) {
    for vec in data.chunks_exact_mut(dim) {
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
}

fn print_row(method: &str, tier: &str, nprobe: usize, build_s: f64, recall: f32, qps: f64) {
    println!(
        "{:<10} {:>4} {:>7} {:>8.2} {:>10.4} {:>11.0}",
        method, tier, nprobe, build_s, recall, qps
    );
}

fn make_pq_config(dim: usize, nlist: usize, nprobe: usize, m: usize, nbits: usize) -> IndexConfig {
    IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::Ip,
        dim,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..IndexParams::default()
        },
    }
}

fn evaluate_queries<F>(
    queries: &[f32],
    dim: usize,
    eval_queries: usize,
    search_one: F,
) -> Vec<Vec<i64>>
where
    F: Sync + Fn(&[f32]) -> Vec<i64>,
{
    (0..eval_queries)
        .into_par_iter()
        .map(|qi| {
            let query = &queries[qi * dim..(qi + 1) * dim];
            search_one(query)
        })
        .collect()
}

fn collect_batch_ids(ids: Vec<i64>, top_k: usize, eval_queries: usize) -> Vec<Vec<i64>> {
    ids.chunks(top_k)
        .take(eval_queries)
        .map(|row| row.iter().copied().take(top_k).collect())
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = Path::new(DEFAULT_DATA_DIR);
    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.cosine.ibin");

    if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
        println!("missing dataset under {}", data_dir.display());
        return Ok(());
    }

    let (base_n, base_dim, mut base) = read_fbin(&base_path)?;
    let (query_n, query_dim, mut queries) = read_fbin(&query_path)?;
    let (gt_n, gt_k, gt_flat) = read_ibin(&gt_path)?;
    let gt = rows_i32(&gt_flat, gt_k);

    if base_dim != EXPECTED_DIM || query_dim != EXPECTED_DIM {
        return Err(format!(
            "expected dim {}, got base_dim={} query_dim={}",
            EXPECTED_DIM, base_dim, query_dim
        )
        .into());
    }
    if gt_n != query_n {
        return Err(format!("gt rows {} != query rows {}", gt_n, query_n).into());
    }
    if gt_k < TOP_K {
        return Err(format!("gt_k={} < top_k={}", gt_k, TOP_K).into());
    }

    normalize_vectors(&mut base, base_dim);
    normalize_vectors(&mut queries, query_dim);

    let train_n = TRAIN_SIZE.min(base_n);
    let train = &base[..train_n * base_dim];
    let eval_queries = EVAL_QUERIES.min(query_n).min(gt.len());

    println!("=== IVF Quantizer Comparison: Cohere 768D ===");
    println!(
        "{:<10} {:>4} {:>7} {:>8} {:>10} {:>11}",
        "method", "tier", "nprobe", "build_s", "recall@10", "search_qps"
    );

    let args: Vec<String> = std::env::args().collect();
    let enabled_methods: Vec<&str> = if args.len() > 1 {
        args[1].split(',').collect()
    } else {
        vec!["FLAT", "PQ", "HVQ", "EXRABITQ"]
    };
    let enabled_tiers: Vec<&str> = if args.len() > 2 {
        args[2].split(',').collect()
    } else {
        vec!["8x"]
    };

    for tier in &TIERS {
        if !enabled_tiers
            .iter()
            .any(|t| t.eq_ignore_ascii_case(tier.label))
        {
            continue;
        }

        if enabled_methods
            .iter()
            .any(|m| m.eq_ignore_ascii_case("FLAT"))
        {
            let config = IndexConfig {
                index_type: IndexType::IvfFlat,
                metric_type: MetricType::Ip,
                dim: EXPECTED_DIM,
                data_type: knowhere_rs::api::DataType::Float,
                params: IndexParams {
                    nlist: Some(NLIST),
                    nprobe: Some(NPROBES[0]),
                    ..IndexParams::default()
                },
            };
            let mut index = IvfFlatIndex::new(&config)?;
            let t_build = Instant::now();
            index.train(train)?;
            index.add(&base, None)?;
            let build_s = t_build.elapsed().as_secs_f64();

            for &nprobe in &NPROBES {
                let req = SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    filter: None,
                    params: None,
                    radius: None,
                };
                let t_search = Instant::now();
                let batch_queries = &queries[..eval_queries * EXPECTED_DIM];
                let results = index
                    .search(batch_queries, &req)
                    .map(|r| collect_batch_ids(r.ids, TOP_K, eval_queries))
                    .unwrap_or_default();
                let search_qps =
                    eval_queries as f64 / t_search.elapsed().as_secs_f64().max(f64::EPSILON);
                let recall = compute_recall(&results, &gt[..eval_queries], TOP_K);
                print_row("IVF-FLAT", tier.label, nprobe, build_s, recall, search_qps);
            }
        }

        if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("PQ")) {
            let config = make_pq_config(EXPECTED_DIM, NLIST, NPROBES[0], tier.pq_m, tier.pq_nbits);
            let mut index = IvfPqIndex::new(&config)?;
            let t_build = Instant::now();
            index.train(train)?;
            index.add(&base, None)?;
            let build_s = t_build.elapsed().as_secs_f64();

            for &nprobe in &NPROBES {
                let req = SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    filter: None,
                    params: None,
                    radius: None,
                };
                let t_search = Instant::now();
                let batch_queries = &queries[..eval_queries * EXPECTED_DIM];
                let results = index
                    .search(batch_queries, &req)
                    .map(|r| collect_batch_ids(r.ids, TOP_K, eval_queries))
                    .unwrap_or_default();
                let search_qps =
                    eval_queries as f64 / t_search.elapsed().as_secs_f64().max(f64::EPSILON);
                let recall = compute_recall(&results, &gt[..eval_queries], TOP_K);
                print_row("IVF-PQ", tier.label, nprobe, build_s, recall, search_qps);
            }
        }

        if enabled_methods
            .iter()
            .any(|m| m.eq_ignore_ascii_case("HVQ"))
        {
            let config = IvfHvqConfig::new(EXPECTED_DIM, NLIST, tier.hvq_bits)
                .with_metric(MetricType::Ip)
                .with_seed(42)
                .with_nprobe(NPROBES[0]);
            let mut index = IvfHvqIndex::new(config);
            let t_build = Instant::now();
            index.train(train)?;
            index.add(&base, None)?;
            let build_s = t_build.elapsed().as_secs_f64();

            for &nprobe in &NPROBES {
                let req = SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    filter: None,
                    params: None,
                    radius: None,
                };
                let t_search = Instant::now();
                let results = evaluate_queries(&queries, EXPECTED_DIM, eval_queries, |query| {
                    index
                        .search(query, &req)
                        .map(|r| r.ids.into_iter().take(TOP_K).collect())
                        .unwrap_or_default()
                });
                let search_qps =
                    eval_queries as f64 / t_search.elapsed().as_secs_f64().max(f64::EPSILON);
                let recall = compute_recall(&results, &gt[..eval_queries], TOP_K);
                print_row("IVF-HVQ", tier.label, nprobe, build_s, recall, search_qps);
            }
        }

        if enabled_methods
            .iter()
            .any(|m| m.eq_ignore_ascii_case("EXRABITQ"))
        {
            let config = IvfExRaBitqConfig::new(EXPECTED_DIM, NLIST, tier.exrabitq_bits)
                .with_metric(MetricType::L2)
                .with_rotation_seed(42)
                .with_rerank_k(100)
                .with_nprobe(NPROBES[0]);
            let mut index = IvfExRaBitqIndex::new(config);
            let t_build = Instant::now();
            index.train(train)?;
            index.add(&base, None)?;
            let build_s = t_build.elapsed().as_secs_f64();

            for &nprobe in &NPROBES {
                let req = SearchRequest {
                    top_k: TOP_K,
                    nprobe,
                    filter: None,
                    params: None,
                    radius: None,
                };
                let t_search = Instant::now();
                let results = evaluate_queries(&queries, EXPECTED_DIM, eval_queries, |query| {
                    index
                        .search(query, &req)
                        .map(|r| r.ids.into_iter().take(TOP_K).collect())
                        .unwrap_or_default()
                });
                let search_qps =
                    eval_queries as f64 / t_search.elapsed().as_secs_f64().max(f64::EPSILON);
                let recall = compute_recall(&results, &gt[..eval_queries], TOP_K);
                print_row("IVF-EXRQ", tier.label, nprobe, build_s, recall, search_qps);
            }
        }
    }

    Ok(())
}

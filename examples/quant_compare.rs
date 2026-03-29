use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::quantization::{
    HvqConfig, HvqIndex, HvqQuantizer, PQConfig, ProductQuantizer, TurboQuantConfig,
    TurboQuantMse,
};
use rayon::prelude::*;

const DEFAULT_DATA_DIR: &str = "/data/work/datasets/wikipedia-cohere-1m";
const EXPECTED_DIM: usize = 768;
const TOP_K: usize = 10;
const TRAIN_SIZE: usize = 100_000;
const EVAL_QUERIES: usize = 5_000;

struct Tier {
    label: &'static str,
    pq_m: usize,
    pq_nbits: usize,
    tq_bits: u8,
    hvq_bits: u8,
}

const TIERS: [Tier; 3] = [
    Tier {
        label: "32x",
        pq_m: 96,
        pq_nbits: 8,
        tq_bits: 1,
        hvq_bits: 1,
    },
    Tier {
        label: "8x",
        pq_m: 384,
        pq_nbits: 8,
        tq_bits: 3,
        hvq_bits: 4,
    },
    Tier {
        label: "4x",
        pq_m: 768,
        pq_nbits: 8,
        tq_bits: 6,
        hvq_bits: 8,
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
struct MinScored {
    id: i64,
    score: f32,
}

impl Eq for MinScored {}

impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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
            if r.iter().take(top_k.min(r.len())).any(|&rid| rid == gid as i64) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn compute_recall_from_tuples(results: &[Vec<(usize, f32)>], gt: &[Vec<i32>], top_k: usize) -> f32 {
    let n = results.len().min(gt.len());
    if n == 0 || top_k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    for i in 0..n {
        let r = &results[i];
        let g = &gt[i];
        for &gid in g.iter().take(top_k.min(g.len())) {
            if r.iter().take(top_k.min(r.len())).any(|&(rid, _)| rid == gid as usize) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn push_topk(heap: &mut BinaryHeap<MinScored>, top_k: usize, candidate: MinScored) {
    if heap.len() < top_k {
        heap.push(candidate);
        return;
    }
    if let Some(worst) = heap.peek() {
        if candidate.score > worst.score
            || (candidate.score == worst.score && candidate.id < worst.id)
        {
            heap.pop();
            heap.push(candidate);
        }
    }
}

fn finalize_topk(mut heap: BinaryHeap<MinScored>, top_k: usize) -> Vec<i64> {
    let mut items = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        items.push(item);
    }
    items.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.id.cmp(&b.id))
    });
    items.truncate(top_k);
    items.into_iter().map(|item| item.id).collect()
}

fn scan_topk_codes<F>(flat_codes: &[u8], code_size: usize, top_k: usize, score_fn: F) -> Vec<i64>
where
    F: Sync + Send + Fn(usize, &[u8]) -> f32,
{
    let heap = flat_codes
        .par_chunks_exact(code_size)
        .enumerate()
        .fold(
            || BinaryHeap::with_capacity(top_k + 1),
            |mut local_heap, (idx, code)| {
                let score = score_fn(idx, code);
                push_topk(
                    &mut local_heap,
                    top_k,
                    MinScored {
                        id: idx as i64,
                        score,
                    },
                );
                local_heap
            },
        )
        .reduce(
            || BinaryHeap::with_capacity(top_k + 1),
            |mut left, right| {
                for item in right {
                    push_topk(&mut left, top_k, item);
                }
                left
            },
        );

    finalize_topk(heap, top_k)
}

fn evaluate_queries<F>(
    queries: &[f32],
    dim: usize,
    eval_queries: usize,
    top_k: usize,
    search_one: F,
) -> Vec<Vec<i64>>
where
    F: Sync + Fn(&[f32]) -> Vec<i64>,
{
    let mut results = Vec::with_capacity(eval_queries);
    for query in queries.chunks_exact(dim).take(eval_queries) {
        results.push(search_one(query));
    }
    if results.len() < top_k {
        results.shrink_to_fit();
    }
    results
}

fn tq_code_bytes(bits: u8) -> usize {
    EXPECTED_DIM.next_power_of_two() * bits as usize / 8
}

fn hvq_code_bytes(bits: u8) -> usize {
    12 + (EXPECTED_DIM * bits as usize).div_ceil(8)
}

fn print_row(method: &str, tier: &str, code_bytes: usize, build_s: f64, recall: f32, qps: f64) {
    println!(
        "{:<8} {:>4} {:>11} {:>8.2} {:>10.4} {:>10.0}",
        method, tier, code_bytes, build_s, recall, qps
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = Path::new(DEFAULT_DATA_DIR);
    let base_path = data_dir.join("base.fbin");
    let query_path = data_dir.join("query.fbin");
    let gt_path = data_dir.join("gt.ibin");

    if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
        println!("missing dataset under {}", data_dir.display());
        return Ok(());
    }

    let (base_n, base_dim, base) = read_fbin(&base_path)?;
    let (query_n, query_dim, queries) = read_fbin(&query_path)?;
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

    let train_n = TRAIN_SIZE.min(base_n);
    let train = &base[..train_n * base_dim];
    let eval_queries = EVAL_QUERIES.min(query_n).min(gt.len());

    println!("=== Quantizer Comparison: Cohere 768D, IP metric ===");
    println!("=== (TQ uses normalized vectors; PQ/HVQ use raw vectors) ===");
    println!(
        "{:<8} {:>4} {:>11} {:>8} {:>10} {:>10}",
        "method", "tier", "code_bytes", "build_s", "recall@10", "scan_qps"
    );

    let args: Vec<String> = std::env::args().collect();
    let enabled_methods: Vec<&str> = if args.len() > 1 {
        args[1].split(',').collect()
    } else {
        vec!["PQ", "TQ", "HVQ", "HVQ2"]
    };

    for tier in &TIERS {
        if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("PQ")) {
        let pq_config = PQConfig::new(EXPECTED_DIM, tier.pq_m, tier.pq_nbits);
        let mut pq = ProductQuantizer::new(pq_config);
        let t_build = Instant::now();
        pq.train(train_n, train)?;
        let pq_codes = pq.encode_batch(base_n, &base)?;
        let build_s = t_build.elapsed().as_secs_f64();

        let t_scan = Instant::now();
        let pq_results = evaluate_queries(&queries, EXPECTED_DIM, eval_queries, TOP_K, |query| {
            let table = pq.build_distance_table_ip(query);
            scan_topk_codes(&pq_codes, pq.code_size(), TOP_K, |_idx, code| {
                pq.compute_distance_with_table(&table, code)
            })
        });
        let scan_qps = eval_queries as f64 / t_scan.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = compute_recall(&pq_results, &gt[..eval_queries], TOP_K);
        print_row(
            "PQ",
            tier.label,
            pq.code_size(),
            build_s,
            recall,
            scan_qps,
        );
        }

        if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("TQ")) {
        let tq_config = TurboQuantConfig::new(EXPECTED_DIM, tier.tq_bits)
            .with_hadamard()
            .with_normalize_for_cosine(true);
        let tq = TurboQuantMse::new(tq_config.clone());
        let t_build = Instant::now();
        let tq_codes = tq.encode_batch(base_n, &base);
        let build_s = t_build.elapsed().as_secs_f64();
        let tq_code_size = tq.code_size_bytes();

        let t_scan = Instant::now();
        let tq_results = evaluate_queries(&queries, EXPECTED_DIM, eval_queries, TOP_K, |query| {
            let q_rot = tq.rotate_query(query);
            let adc = (tier.tq_bits <= 6).then(|| tq.precompute_adc_table(&q_rot));
            scan_topk_codes(&tq_codes, tq_code_size, TOP_K, |_idx, code| match &adc {
                Some(table) => tq.score_ip_adc(table, code),
                None => tq.score_ip(&q_rot, code),
            })
        });
        let scan_qps = eval_queries as f64 / t_scan.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = compute_recall(&tq_results, &gt[..eval_queries], TOP_K);
        print_row(
            "TQ",
            tier.label,
            tq_code_bytes(tier.tq_bits),
            build_s,
            recall,
            scan_qps,
        );
        }

        if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("HVQ")) {
        let mut hvq = HvqQuantizer::new(
            HvqConfig {
                dim: EXPECTED_DIM,
                nbits: tier.hvq_bits,
            },
            42,
        );
        let t_build = Instant::now();
        hvq.train(train_n, train);
        let hvq_codes = hvq.encode_batch(base_n, &base, 4);
        let build_s = t_build.elapsed().as_secs_f64();
        let hvq_storage_size = hvq.code_size_bytes();

        let t_scan = Instant::now();
        let hvq_results = evaluate_queries(&queries, EXPECTED_DIM, eval_queries, TOP_K, |query| {
            let q_rot = hvq.rotate_query(query);
            let state = hvq.precompute_query_state(&q_rot);
            scan_topk_codes(&hvq_codes, hvq_storage_size, TOP_K, |_idx, code| {
                hvq.score_code(&state, code)
            })
        });
        let scan_qps = eval_queries as f64 / t_scan.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = compute_recall(&hvq_results, &gt[..eval_queries], TOP_K);
        print_row(
            "HVQ",
            tier.label,
            hvq_code_bytes(tier.hvq_bits),
            build_s,
            recall,
            scan_qps,
        );
        }

        if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("HVQ2")) {
        let mut hvq = HvqQuantizer::new(
            HvqConfig {
                dim: EXPECTED_DIM,
                nbits: tier.hvq_bits,
            },
            42,
        );
        hvq.train(train_n, train);

        let build_start = Instant::now();
        let hvq_index = HvqIndex::build(&hvq, &base, base_n);
        let build_s = build_start.elapsed().as_secs_f64();

        let t_scan = Instant::now();
        let hvq2_results: Vec<Vec<(usize, f32)>> = (0..eval_queries)
            .into_par_iter()
            .map(|qi| {
                let query = &queries[qi * EXPECTED_DIM..(qi + 1) * EXPECTED_DIM];
                hvq_index.search(query, TOP_K, 10)
            })
            .collect();
        let scan_qps = eval_queries as f64 / t_scan.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = compute_recall_from_tuples(&hvq2_results, &gt[..eval_queries], TOP_K);
        let code_bytes = hvq.code_size_bytes() + EXPECTED_DIM.div_ceil(8);
        print_row("HVQ2", tier.label, code_bytes, build_s, recall, scan_qps);
        }
    }

    Ok(())
}

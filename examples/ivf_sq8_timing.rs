use std::cmp::Ordering;
use std::time::Instant;

use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType};
use knowhere_rs::faiss::IvfSq8Index;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 128;
const BASE_SIZE: usize = 100_000;
const NLIST: usize = 256;
const NPROBE: usize = 32;
const TOP_K: usize = 10;
const QUERIES: usize = 1_000;
const SEED: u64 = 42;

#[derive(Clone, Copy, Debug)]
struct Hit {
    id: i64,
    dist: f32,
}

#[derive(Default)]
struct Timing {
    total_ns: u128,
    search_clusters_ns: u128,
    precompute_query_ns: u128,
    distance_compute_ns: u128,
    topk_accumulate_ns: u128,
    merge_ns: u128,
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn gen_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn is_better(candidate: Hit, current: Hit) -> bool {
    match candidate.dist.total_cmp(&current.dist) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => candidate.id < current.id,
    }
}

fn push_topk(topk: &mut Vec<Hit>, limit: usize, candidate: Hit) {
    if limit == 0 {
        return;
    }
    if topk.len() == limit && !is_better(candidate, topk[topk.len() - 1]) {
        return;
    }

    let pos = topk
        .binary_search_by(|h| {
            h.dist
                .total_cmp(&candidate.dist)
                .then_with(|| h.id.cmp(&candidate.id))
        })
        .unwrap_or_else(|p| p);

    if topk.len() < limit {
        topk.insert(pos, candidate);
    } else if pos < limit {
        topk.insert(pos, candidate);
        topk.truncate(limit);
    }
}

fn print_row(name: &str, ns: u128, total_ns: u128) {
    let total_ms = ns as f64 / 1_000_000.0;
    let per_query_us = ns as f64 / QUERIES as f64 / 1_000.0;
    let pct = if total_ns == 0 {
        0.0
    } else {
        ns as f64 * 100.0 / total_ns as f64
    };
    println!(
        "{:<18} | {:>10.3} | {:>14.3} | {:>8.2}%",
        name, total_ms, per_query_us, pct
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let base = gen_vectors(&mut rng, BASE_SIZE, DIM);
    let queries = gen_vectors(&mut rng, QUERIES, DIM);

    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params: IndexParams::ivf(NLIST, NPROBE),
    };

    let mut index = IvfSq8Index::new(&config)?;
    index.train(&base)?;
    index.add(&base, None)?;

    let mut timing = Timing::default();
    let mut checksum: i64 = 0;

    for q in queries.chunks(DIM) {
        let total_start = Instant::now();

        let c_start = Instant::now();
        let mut dists: Vec<(usize, f32)> = index
            .centroids()
            .chunks(index.dim())
            .enumerate()
            .map(|(i, centroid)| (i, l2_sq(q, centroid)))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        let clusters: Vec<usize> = dists.iter().take(NPROBE).map(|(id, _)| *id).collect();
        timing.search_clusters_ns += c_start.elapsed().as_nanos();

        let mut global_topk: Vec<Hit> = Vec::with_capacity(TOP_K);

        let dim = index.dim();
        for &cluster_id in &clusters {
            let centroid = &index.centroids()[cluster_id * dim..(cluster_id + 1) * dim];

            let pre_start = Instant::now();
            let mut q_residual = vec![0.0f32; dim];
            for i in 0..dim {
                q_residual[i] = q[i] - centroid[i];
            }
            let mut q_precomputed = vec![0i16; dim];
            index
                .quantizer()
                .precompute_query_into(&q_residual, &mut q_precomputed);
            timing.precompute_query_ns += pre_start.elapsed().as_nanos();

            let dist_start = Instant::now();
            let mut local_distances: Vec<Hit> = Vec::new();
            if let Some((ids, codes)) = index.inverted_lists().get(&cluster_id) {
                let n = ids.len().min(codes.len() / dim);
                local_distances.reserve(n);
                for i in 0..n {
                    let code = &codes[i * dim..(i + 1) * dim];
                    let dist = index
                        .quantizer()
                        .sq_l2_precomputed(&q_precomputed, code);
                    local_distances.push(Hit { id: ids[i], dist });
                }
            }
            timing.distance_compute_ns += dist_start.elapsed().as_nanos();

            let topk_start = Instant::now();
            for hit in local_distances {
                push_topk(&mut global_topk, TOP_K, hit);
            }
            timing.topk_accumulate_ns += topk_start.elapsed().as_nanos();
        }

        if let Some(first) = global_topk.first() {
            checksum ^= first.id;
        }

        timing.total_ns += total_start.elapsed().as_nanos();
    }

    let accounted = timing.search_clusters_ns
        + timing.precompute_query_ns
        + timing.distance_compute_ns
        + timing.topk_accumulate_ns
        + timing.merge_ns;
    let overhead_ns = timing.total_ns.saturating_sub(accounted);

    println!(
        "IVF-SQ8 Timing Breakdown ({} queries, nprobe={}, {} vectors)",
        QUERIES, NPROBE, BASE_SIZE
    );
    println!("checksum={}", checksum);
    println!("Phase              | Total (ms) | Per-query (us) | % of total");
    println!("-------------------|------------|----------------|----------");
    print_row("search_clusters", timing.search_clusters_ns, timing.total_ns);
    print_row("precompute_query", timing.precompute_query_ns, timing.total_ns);
    print_row("distance_compute", timing.distance_compute_ns, timing.total_ns);
    print_row("topk_accumulate", timing.topk_accumulate_ns, timing.total_ns);
    print_row("merge", timing.merge_ns, timing.total_ns);
    print_row("overhead", overhead_ns, timing.total_ns);
    print_row("total", timing.total_ns, timing.total_ns);

    Ok(())
}

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::fs::File;
use std::io::Read;

use knowhere_rs::quantization::pq::{PQConfig, ProductQuantizer};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

const DIM: usize = 128;
const NLIST: usize = 256;
const M: usize = 32;
const NBITS: usize = 8;
const TRAIN_SIZE: usize = 100_000;
const RECON_SAMPLE_COUNT: usize = 1_000;
const BRUTE_QUERY_COUNT: usize = 100;
const TOP_K: usize = 10;

fn dataset_root() -> String {
    env::var("SIFT1M_ROOT").unwrap_or_else(|_| "/data/work/datasets/sift-1m".to_string())
}

fn read_fbin(path: &str) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let n = u32::from_le_bytes(header[0..4].try_into()?) as usize;
    let dim = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    let mut bytes = vec![0u8; n * dim * 4];
    file.read_exact(&mut bytes)?;
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((n, dim, data))
}

fn read_ibin(path: &str) -> Result<(usize, usize, Vec<i32>), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let n = u32::from_le_bytes(header[0..4].try_into()?) as usize;
    let width = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    let mut bytes = vec![0u8; n * width * 4];
    file.read_exact(&mut bytes)?;
    let data = bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok((n, width, data))
}

#[inline]
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_sq(a, b).sqrt()
}

#[derive(Copy, Clone, Debug)]
struct OrdF32(f32);

impl PartialEq for OrdF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn nearest_centroid(vector: &[f32], centroids: &[f32], dim: usize) -> usize {
    let nlist = centroids.len() / dim;
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for c in 0..nlist {
        let start = c * dim;
        let dist = l2_distance_sq(vector, &centroids[start..start + dim]);
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

fn coarse_kmeans(vectors: &[f32], dim: usize, k: usize) -> Vec<f32> {
    let n = vectors.len() / dim;
    let mut centroids = vec![0.0f32; k * dim];
    let mut rng = rand::thread_rng();

    let first_idx = rng.gen_range(0..n) * dim;
    centroids[..dim].copy_from_slice(&vectors[first_idx..first_idx + dim]);

    let mut weights = vec![0.0f32; n];
    for c in 1..k {
        for i in 0..n {
            let vector = &vectors[i * dim..(i + 1) * dim];
            let mut min_dist = f32::MAX;
            for existing in 0..c {
                let centroid = &centroids[existing * dim..(existing + 1) * dim];
                min_dist = min_dist.min(l2_distance_sq(vector, centroid));
            }
            weights[i] = min_dist.max(1e-10);
        }

        let sum: f32 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        let dist = WeightedIndex::new(&weights).unwrap();
        let next_idx = dist.sample(&mut rng) * dim;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&vectors[next_idx..next_idx + dim]);
    }

    let max_iter = if n < 10_000 {
        10
    } else if n < 100_000 {
        25
    } else {
        50
    };

    let convergence_threshold = 1e-4;
    let mut prev_centroids = centroids.clone();
    for iter in 0..max_iter {
        let mut assignments = vec![0usize; n];
        for (i, assignment) in assignments.iter_mut().enumerate().take(n) {
            let vector = &vectors[i * dim..(i + 1) * dim];
            let mut min_dist = f32::MAX;
            let mut min_idx = 0usize;
            for c in 0..k {
                let centroid = &centroids[c * dim..(c + 1) * dim];
                let dist = l2_distance_sq(vector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = c;
                }
            }
            *assignment = min_idx;
        }

        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            for j in 0..dim {
                sums[c * dim + j] += vectors[i * dim + j];
            }
            counts[c] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c * dim + j] = sums[c * dim + j] / counts[c] as f32;
                }
            }
        }

        if iter > 0 {
            let mut total_change = 0.0f32;
            for i in 0..k * dim {
                let diff = centroids[i] - prev_centroids[i];
                total_change += diff * diff;
            }
            let avg_change = total_change / (k * dim) as f32;
            if avg_change < convergence_threshold {
                break;
            }
        }

        prev_centroids = centroids.clone();
    }

    centroids
}

fn compute_residuals(data: &[f32], count: usize, dim: usize, centroids: &[f32]) -> Vec<f32> {
    let mut residuals = Vec::with_capacity(count * dim);
    for i in 0..count {
        let vector = &data[i * dim..(i + 1) * dim];
        let cluster = nearest_centroid(vector, centroids, dim);
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        for d in 0..dim {
            residuals.push(vector[d] - centroid[d]);
        }
    }
    residuals
}

fn recall_at_k(gt: &[i32], results: &[usize], k: usize, n_queries: usize, gt_width: usize) -> f32 {
    let mut hits = 0usize;
    for q in 0..n_queries {
        let gt_row = &gt[q * gt_width..q * gt_width + k];
        let res_row = &results[q * k..q * k + k];
        for &res in res_row {
            if gt_row.iter().any(|&gt_idx| gt_idx as usize == res) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n_queries * k) as f32
}

fn brute_force_top_k(base: &[f32], queries: &[f32], dim: usize, nq: usize, k: usize) -> Vec<usize> {
    let nbase = base.len() / dim;
    let mut all = Vec::with_capacity(nq * k);

    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut heap: BinaryHeap<(OrdF32, usize)> = BinaryHeap::with_capacity(k + 1);

        for i in 0..nbase {
            let vector = &base[i * dim..(i + 1) * dim];
            let dist = l2_distance_sq(query, vector);
            if heap.len() < k {
                heap.push((OrdF32(dist), i));
            } else if let Some(&(worst, _)) = heap.peek() {
                if dist < worst.0 {
                    heap.pop();
                    heap.push((OrdF32(dist), i));
                }
            }
        }

        let mut row: Vec<(f32, usize)> = heap.into_iter().map(|(d, i)| (d.0, i)).collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, idx) in row {
            all.push(idx);
        }
    }

    all
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = dataset_root();
    let (base_n, base_dim, base) = read_fbin(&format!("{root}/base.fbin"))?;
    let (query_n, query_dim, queries) = read_fbin(&format!("{root}/query.fbin"))?;
    let (gt_q, gt_width, gt) = read_ibin(&format!("{root}/gt.ibin"))?;

    if base_dim != DIM || query_dim != DIM {
        return Err(format!(
            "unexpected dim: base={}, query={}, expected={DIM}",
            base_dim, query_dim
        )
        .into());
    }
    if gt_q != query_n {
        return Err(format!("gt queries {} != query count {}", gt_q, query_n).into());
    }

    let train = &base[..TRAIN_SIZE * DIM];
    let coarse_centroids = coarse_kmeans(train, DIM, NLIST);

    let residuals = compute_residuals(train, TRAIN_SIZE, DIM, &coarse_centroids);
    let mut pq = ProductQuantizer::new(PQConfig::new(DIM, M, NBITS));
    pq.train(TRAIN_SIZE, &residuals)?;

    let sample_count = RECON_SAMPLE_COUNT.min(base_n);
    let mut total_err = 0.0f64;
    let mut total_rel = 0.0f64;

    println!("=== IVF-PQ encode/decode diagnostic ===");
    println!(
        "base={} train_size={} dim={} nlist={} m={} nbits={} sample_count={}",
        base_n, TRAIN_SIZE, DIM, NLIST, M, NBITS, sample_count
    );

    for i in 0..sample_count {
        let vector = &base[i * DIM..(i + 1) * DIM];
        let cluster = nearest_centroid(vector, &coarse_centroids, DIM);
        let centroid = &coarse_centroids[cluster * DIM..(cluster + 1) * DIM];
        let mut residual = vec![0.0f32; DIM];
        for d in 0..DIM {
            residual[d] = vector[d] - centroid[d];
        }

        let code = pq.encode(&residual)?;
        let reconstructed = pq.decode(&code)?;
        let err = l2_distance(&residual, &reconstructed);
        let norm = l2_distance(&residual, &vec![0.0f32; DIM]).max(f32::EPSILON);
        let rel = err / norm;

        total_err += err as f64;
        total_rel += rel as f64;

        if i < 5 {
            println!(
                "sample={} residual[:4]=[{:.4}, {:.4}, {:.4}, {:.4}] reconstructed[:4]=[{:.4}, {:.4}, {:.4}, {:.4}] err={:.4} rel_err={:.2}%",
                i,
                residual[0], residual[1], residual[2], residual[3],
                reconstructed[0], reconstructed[1], reconstructed[2], reconstructed[3],
                err,
                rel * 100.0
            );
        }
    }

    let avg_err = total_err / sample_count as f64;
    let avg_rel = total_rel / sample_count as f64;
    println!("avg_l2_err={:.6}", avg_err);
    println!("avg_rel_err={:.2}%", avg_rel * 100.0);

    let nq = BRUTE_QUERY_COUNT.min(query_n);
    let exact_results = brute_force_top_k(&base, &queries[..nq * DIM], DIM, nq, TOP_K);
    let brute_recall = recall_at_k(&gt, &exact_results, TOP_K, nq, gt_width);
    println!("brute_force_recall@10={:.3}", brute_recall);

    let conclusion = if brute_recall < 0.99 {
        "eval_bug"
    } else if avg_rel > 0.08 {
        "encode_bug"
    } else {
        "ok"
    };
    println!("conclusion={}", conclusion);

    Ok(())
}

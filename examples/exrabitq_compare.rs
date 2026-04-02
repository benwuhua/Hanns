use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::error::Error;
use std::time::Instant;

use knowhere_rs::api::{MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfUsqConfig, IvfUsqIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaxScored {
    id: i64,
    distance: f32,
}

impl Eq for MaxScored {}

impl Ord for MaxScored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for MaxScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * dim];
    for value in &mut data {
        *value = rng.gen_range(-1.0f32..1.0f32);
    }
    data
}

fn l2_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}

fn brute_force_topk(base: &[f32], dim: usize, query: &[f32], top_k: usize) -> Vec<i64> {
    let mut heap = BinaryHeap::with_capacity(top_k + 1);
    for (idx, vector) in base.chunks_exact(dim).enumerate() {
        let candidate = MaxScored {
            id: idx as i64,
            distance: l2_distance(vector, query),
        };
        if heap.len() < top_k {
            heap.push(candidate);
            continue;
        }
        if let Some(worst) = heap.peek() {
            if candidate.distance < worst.distance
                || (candidate.distance == worst.distance && candidate.id < worst.id)
            {
                heap.pop();
                heap.push(candidate);
            }
        }
    }

    let mut out = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        out.push(item);
    }
    out.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.id.cmp(&b.id))
    });
    out.into_iter().map(|item| item.id).collect()
}

fn compute_recall(results: &[Vec<i64>], gt: &[Vec<i64>], top_k: usize) -> f32 {
    let n = results.len().min(gt.len());
    if n == 0 || top_k == 0 {
        return 0.0;
    }

    let mut hits = 0usize;
    for (result, expected) in results.iter().zip(gt.iter()).take(n) {
        for &gt_id in expected.iter().take(top_k) {
            if result.iter().take(top_k).any(|&rid| rid == gt_id) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n * top_k) as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let dim = env_usize("EXRABITQ_DIM", 64);
    let train_n = env_usize("EXRABITQ_TRAIN", 2_048);
    let base_n = env_usize("EXRABITQ_BASE", 8_192).max(train_n);
    let nq = env_usize("EXRABITQ_NQ", 200);
    let nlist = env_usize("EXRABITQ_NLIST", 64).min(train_n);
    let nprobe = env_usize("EXRABITQ_NPROBE", 8).min(nlist.max(1));
    let bits = env_usize("EXRABITQ_BITS", 4);
    let top_k = env_usize("EXRABITQ_TOPK", 10);
    let rerank_k = env_usize("EXRABITQ_RERANK_K", top_k.max(64));
    let use_high_accuracy = env_bool("EXRABITQ_HIGH_ACCURACY", false);

    let base = random_vectors(base_n, dim, 7);
    let queries = random_vectors(nq, dim, 17);
    let ids: Vec<i64> = (0..base_n as i64).collect();

    let config = IvfUsqConfig::new(dim, nlist.max(1), bits)
        .with_metric(MetricType::L2)
        .with_nprobe(nprobe.max(1))
        .with_rotation_seed(29)
        .with_rerank_k(rerank_k.max(top_k))
        .with_high_accuracy_scan(use_high_accuracy);    let mut index = IvfUsqIndex::new(config);
    let build_start = Instant::now();
    index.train(&base[..train_n * dim])?;
    index.add(&base, Some(&ids))?;
    let build_s = build_start.elapsed().as_secs_f64();

    let gt: Vec<Vec<i64>> = queries
        .chunks_exact(dim)
        .map(|query| brute_force_topk(&base, dim, query, top_k))
        .collect();

    let request = SearchRequest {
        top_k,
        nprobe,
        filter: None,
        params: None,
        radius: None,
    };

    let search_start = Instant::now();
    let mut results = Vec::with_capacity(nq);
    for query in queries.chunks_exact(dim) {
        let result = index.search(query, &request)?;
        results.push(result.ids.into_iter().take(top_k).collect::<Vec<_>>());
    }
    let search_s = search_start.elapsed().as_secs_f64();
    let search_qps = nq as f64 / search_s.max(1e-9);
    let recall = compute_recall(&results, &gt, top_k);

    println!(
        "{:<12} {:>4} {:>6} {:>7} {:>10} {:>10} {:>12}",
        "method", "bits", "nlist", "nprobe", "build_s", "recall@10", "search_qps"
    );
    println!(
        "{:<12} {:>4} {:>6} {:>7} {:>10.2} {:>10.4} {:>12.1}",
        if use_high_accuracy {
            "USQ-HA"
        } else {
            "USQ"
        },
        bits,
        nlist,
        nprobe,
        build_s,
        recall,
        search_qps
    );

    Ok(())
}

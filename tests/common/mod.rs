//! Shared test utilities for integration tests.

use rand::{Rng, SeedableRng};

/// Generate random f32 vectors.
pub fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

/// Generate seeded random f32 vectors (deterministic).
pub fn generate_vectors_seeded(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

/// Squared L2 distance between two vectors.
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Brute-force ground truth: for each query, return top-k nearest indices (i32).
pub fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let n = base.len() / dim;
    (0..num_queries)
        .map(|q| {
            let qv = &query[q * dim..(q + 1) * dim];
            let mut dists: Vec<(usize, f32)> = (0..n)
                .map(|i| (i, l2_distance_squared(qv, &base[i * dim..(i + 1) * dim])))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(idx, _)| *idx as i32).collect()
        })
        .collect()
}

/// Brute-force ground truth (auto-derives num_queries from slice length).
pub fn compute_ground_truth_batch(
    base: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let num_queries = queries.len() / dim;
    compute_ground_truth(base, queries, num_queries, dim, k)
}

/// Recall@k: fraction of ground truth found in results.
pub fn compute_recall(results: &[Vec<i64>], ground_truth: &[Vec<i32>], k: usize) -> f64 {
    let mut total = 0usize;
    let mut hits = 0usize;
    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let gt_set: std::collections::HashSet<i32> = gt.iter().take(k).copied().collect();
        for &id in res.iter().take(k) {
            if gt_set.contains(&(id as i32)) {
                hits += 1;
            }
        }
        total += k.min(gt.len());
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

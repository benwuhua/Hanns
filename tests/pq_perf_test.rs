//! Performance tests for PQ quantization improvements
//!
//! Tests verify that:
//! 1. OPQ + PQ achieves R@10 > 50% on SIFT1M-like data
//! 2. Residual PQ improves recall by > 20% over standard PQ

use hanns::quantization::{
    PQConfig, ProductQuantizer, ResidualPQConfig, ResidualProductQuantizer,
};

#[cfg(feature = "long-tests")]
use hanns::api::SearchRequest;
#[cfg(feature = "long-tests")]
use hanns::faiss::{IvfOpqConfig, IvfOpqIndex};
#[cfg(feature = "long-tests")]
use hanns::quantization::OPQConfig;
#[cfg(feature = "long-tests")]
use hanns::quantization::OptimizedProductQuantizer;
#[cfg(feature = "long-tests")]
use rand::rngs::StdRng;
#[cfg(feature = "long-tests")]
use rand::Rng;
#[cfg(feature = "long-tests")]
use rand::SeedableRng;

/// Generate SIFT-like random data (Gaussian distribution, normalized)
#[cfg(feature = "long-tests")]
fn generate_sift_like_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n * dim);

    for _ in 0..n * dim {
        // SIFT features are approximately Gaussian distributed
        let val = rng.gen::<f32>();
        data.push(val);
    }

    // Normalize each vector
    for i in 0..n {
        let start = i * dim;
        let end = start + dim;
        let norm: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in data.iter_mut().take(end).skip(start) {
                *value /= norm;
            }
        }
    }

    data
}

/// Compute ground truth nearest neighbors using brute force
#[cfg(feature = "long-tests")]
fn compute_ground_truth(
    queries: &[f32],
    database: &[f32],
    nq: usize,
    n: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<usize>> {
    let mut ground_truth = Vec::with_capacity(nq);

    for q_idx in 0..nq {
        let q_offset = q_idx * dim;
        let query = &queries[q_offset..q_offset + dim];

        // Compute distances to all database vectors
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..n {
            let d_offset = i * dim;
            let vec = &database[d_offset..d_offset + dim];

            let dist: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            distances.push((i, dist));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Take top k
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

/// Compute Recall@k
#[cfg(feature = "long-tests")]
fn compute_recall_at_k(results: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f32 {
    let mut total_recall = 0.0f32;

    for (result, gt) in results.iter().zip(ground_truth.iter()) {
        let result_set: std::collections::HashSet<_> = result.iter().take(k).cloned().collect();
        let gt_set: std::collections::HashSet<_> = gt.iter().take(k).cloned().collect();

        let intersection: Vec<_> = result_set.intersection(&gt_set).collect();
        total_recall += intersection.len() as f32 / k as f32;
    }

    total_recall / results.len() as f32
}

#[test]
fn residual_pq_reconstruction_error_beats_standard_pq_on_clustered_data() {
    let dim = 32;
    let n_clusters = 4;
    let points_per_cluster = 32;
    let total = n_clusters * points_per_cluster;

    let mut data = Vec::with_capacity(total * dim);
    for cluster in 0..n_clusters {
        let base = cluster as f32 * 50.0;
        for point in 0..points_per_cluster {
            for d in 0..dim {
                data.push(base + d as f32 * 0.01 + point as f32 * 0.001);
            }
        }
    }

    let mut pq = ProductQuantizer::new(PQConfig::new(dim, 4, 8));
    pq.train(total, &data).unwrap();

    let mut rpq = ResidualProductQuantizer::new(ResidualPQConfig::new(dim, 8, 4, 8)).unwrap();
    rpq.train(total, &data).unwrap();

    let mut pq_mse = 0.0f32;
    let mut rpq_mse = 0.0f32;

    for i in 0..16 {
        let vector = &data[i * dim..(i + 1) * dim];

        let pq_code = pq.encode(vector).unwrap();
        let pq_recon = pq.decode(&pq_code).unwrap();
        let rpq_code = rpq.encode(vector).unwrap();
        let rpq_recon = rpq.decode(&rpq_code).unwrap();

        for j in 0..dim {
            let pq_diff = vector[j] - pq_recon[j];
            pq_mse += pq_diff * pq_diff;

            let rpq_diff = vector[j] - rpq_recon[j];
            rpq_mse += rpq_diff * rpq_diff;
        }
    }

    pq_mse /= (16 * dim) as f32;
    rpq_mse /= (16 * dim) as f32;

    assert!(
        rpq_mse < pq_mse,
        "residual PQ should reconstruct clustered data better than plain PQ (rpq_mse={rpq_mse}, pq_mse={pq_mse})"
    );
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore]
fn test_opq_recall_improvement() {
    let dim = 128;
    let n_train = 10000;
    let n_test = 100;
    let k = 10;

    println!("\n=== OPQ Recall Test ===");
    println!("Dataset: {} train, {} test, dim={}", n_train, n_test, dim);

    // Generate data
    let train_data = generate_sift_like_data(n_train, dim, 42);
    let test_data = generate_sift_like_data(n_test, dim, 123);

    // Compute ground truth
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&test_data, &train_data, n_test, n_train, dim, k);

    // Test standard PQ
    println!("Training standard PQ...");
    let pq_config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(pq_config);
    pq.train(n_train, &train_data).unwrap();

    // Encode database
    let pq_codes = pq.encode_batch(n_train, &train_data).unwrap();

    // Search with PQ
    let mut pq_results = Vec::with_capacity(n_test);
    for q_idx in 0..n_test {
        let q_offset = q_idx * dim;
        let query = &test_data[q_offset..q_offset + dim];

        // Compute distances to all encoded vectors
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let code_offset = i * pq.code_size();
            let code = &pq_codes[code_offset..code_offset + pq.code_size()];
            let dist = pq.compute_distance(query, code);
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        pq_results.push(neighbors);
    }

    let pq_recall = compute_recall_at_k(&pq_results, &ground_truth, k);
    println!("Standard PQ R@{}: {:.2}%", k, pq_recall * 100.0);

    // Test OPQ
    println!("Training OPQ...");
    let opq_config = OPQConfig::new(dim, 8, 8);
    let mut opq = OptimizedProductQuantizer::new(opq_config).unwrap();
    opq.train(n_train, &train_data).unwrap();

    // Encode database
    let opq_codes = opq.encode_batch(n_train, &train_data).unwrap();

    // Search with OPQ
    let mut opq_results = Vec::with_capacity(n_test);
    for q_idx in 0..n_test {
        let q_offset = q_idx * dim;
        let query = &test_data[q_offset..q_offset + dim];

        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let code_offset = i * opq.code_size();
            let code = &opq_codes[code_offset..code_offset + opq.code_size()];
            let dist = opq.compute_distance(query, code);
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        opq_results.push(neighbors);
    }

    let opq_recall = compute_recall_at_k(&opq_results, &ground_truth, k);
    println!("OPQ R@{}: {:.2}%", k, opq_recall * 100.0);

    // Verify improvement
    let improvement = (opq_recall - pq_recall) / pq_recall * 100.0;
    println!("Improvement: {:.1}%", improvement);

    // OPQ should achieve R@10 > 50% and improve over PQ
    assert!(
        opq_recall > 0.50,
        "OPQ R@10 should be > 50%, got {:.2}%",
        opq_recall * 100.0
    );
    assert!(
        opq_recall >= pq_recall,
        "OPQ should perform at least as well as PQ"
    );
}

#[test]
#[cfg(feature = "long-tests")]
#[ignore]
fn test_residual_pq_recall_improvement() {
    let dim = 128;
    let n_train = 10000;
    let n_test = 100;
    let k = 10;

    println!("\n=== Residual PQ Recall Test ===");
    println!("Dataset: {} train, {} test, dim={}", n_train, n_test, dim);

    // Generate data
    let train_data = generate_sift_like_data(n_train, dim, 42);
    let test_data = generate_sift_like_data(n_test, dim, 123);

    // Compute ground truth
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&test_data, &train_data, n_test, n_train, dim, k);

    // Test standard PQ
    println!("Training standard PQ...");
    let pq_config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(pq_config);
    pq.train(n_train, &train_data).unwrap();

    let pq_codes = pq.encode_batch(n_train, &train_data).unwrap();

    let mut pq_results = Vec::with_capacity(n_test);
    for q_idx in 0..n_test {
        let q_offset = q_idx * dim;
        let query = &test_data[q_offset..q_offset + dim];

        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let code_offset = i * pq.code_size();
            let code = &pq_codes[code_offset..code_offset + pq.code_size()];
            let dist = pq.compute_distance(query, code);
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        pq_results.push(neighbors);
    }

    let pq_recall = compute_recall_at_k(&pq_results, &ground_truth, k);
    println!("Standard PQ R@{}: {:.2}%", k, pq_recall * 100.0);

    // Test Residual PQ
    println!("Training Residual PQ...");
    let rpq_config = ResidualPQConfig::new(dim, 256, 8, 8);
    let mut rpq = ResidualProductQuantizer::new(rpq_config).unwrap();
    rpq.train(n_train, &train_data).unwrap();

    let rpq_codes = rpq.encode_batch(n_train, &train_data).unwrap();

    let mut rpq_results = Vec::with_capacity(n_test);
    for q_idx in 0..n_test {
        let q_offset = q_idx * dim;
        let query = &test_data[q_offset..q_offset + dim];

        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n_train);
        for i in 0..n_train {
            let code_offset = i * rpq.code_size();
            let code = &rpq_codes[code_offset..code_offset + rpq.code_size()];
            let dist = rpq.compute_distance(query, code);
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
        rpq_results.push(neighbors);
    }

    let rpq_recall = compute_recall_at_k(&rpq_results, &ground_truth, k);
    println!("Residual PQ R@{}: {:.2}%", k, rpq_recall * 100.0);

    // Verify improvement
    let improvement = (rpq_recall - pq_recall) / pq_recall * 100.0;
    println!("Improvement: {:.1}%", improvement);

    // Residual PQ should improve recall by > 20%
    assert!(
        improvement > 20.0,
        "Residual PQ should improve recall by > 20%, got {:.1}%",
        improvement
    );
}

#[test]
#[cfg(feature = "long-tests")]
#[ignore]
fn test_ivf_opq_index() {
    let dim = 128;
    let n_train = 10000;
    let n_add = 5000;
    let n_test = 50;
    let k = 10;

    println!("\n=== IVF-OPQ Index Test ===");
    println!(
        "Dataset: {} train, {} add, {} test, dim={}",
        n_train, n_add, n_test, dim
    );

    // Generate data
    let train_data = generate_sift_like_data(n_train, dim, 42);
    let add_data = generate_sift_like_data(n_add, dim, 456);
    let test_data = generate_sift_like_data(n_test, dim, 789);

    // Compute ground truth (brute force on added data)
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&test_data, &add_data, n_test, n_add, dim, k);

    // Create and train IVF-OPQ index
    println!("Training IVF-OPQ index...");
    let config = IvfOpqConfig::new(dim, 256, 8, 8);
    let mut index = IvfOpqIndex::new(config).unwrap();

    index.train(n_train, &train_data).unwrap();
    assert!(index.is_trained());

    // Add vectors
    println!("Adding vectors...");
    let ids: Vec<i64> = (0..n_add as i64).collect();
    index.add(n_add, &add_data, Some(&ids)).unwrap();
    assert_eq!(index.ntotal(), n_add);

    // Search
    println!("Searching...");
    let req = SearchRequest {
        top_k: k,
        nprobe: 32,
        filter: None,
        params: None,
        radius: None,
    };

    let result = index.search(n_test, &test_data, &req).unwrap();

    // Parse results
    let mut ivf_results: Vec<Vec<usize>> = Vec::with_capacity(n_test);
    for q_idx in 0..n_test {
        let start = q_idx * k;
        let end = start + k;
        let ids: Vec<usize> = result.ids[start..end]
            .iter()
            .filter(|&&id| id >= 0)
            .map(|id| *id as usize)
            .collect();
        ivf_results.push(ids);
    }

    // Compute recall
    let recall = compute_recall_at_k(&ivf_results, &ground_truth, k);
    println!("IVF-OPQ R@{}: {:.2}%", k, recall * 100.0);

    // IVF-OPQ should achieve reasonable recall
    assert!(
        recall > 0.30,
        "IVF-OPQ R@10 should be > 30%, got {:.2}%",
        recall * 100.0
    );
}

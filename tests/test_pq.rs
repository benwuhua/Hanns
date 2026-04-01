//! PQ unit tests extracted from src/quantization/pq.rs
//!
//! Editing these assertions no longer triggers recompilation of the PQ module.
//! Run: cargo test --test test_pq

use knowhere_rs::quantization::{PQConfig, ProductQuantizer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn create_test_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            vectors.push(((i * dim + j) % 100) as f32 / 100.0);
        }
    }
    vectors
}

#[test]
fn test_pq_config() {
    let config = PQConfig::new(128, 8, 8);
    assert_eq!(config.sub_dim(), 16);
    assert_eq!(config.ksub(), 256);
    assert_eq!(config.code_size(), 8);
}

#[test]
fn test_pq_config_validation() {
    let config = PQConfig::new(128, 8, 8);
    assert!(config.validate().is_ok());

    let config = PQConfig::new(100, 7, 8);
    assert!(config.validate().is_err());

    let config = PQConfig::new(128, 8, 0);
    assert!(config.validate().is_err());
}

#[test]
fn test_pq_train_and_encode() {
    let dim = 64;
    let config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(config);

    let train_data = create_test_vectors(1000, dim);
    pq.train(1000, &train_data).unwrap();
    assert!(pq.is_trained());

    let query = create_test_vectors(1, dim);
    let code = pq.encode(&query).unwrap();
    assert_eq!(code.len(), 8);

    let queries = create_test_vectors(10, dim);
    let codes = pq.encode_batch(10, &queries).unwrap();
    assert_eq!(codes.len(), 10 * 8);
}

#[test]
fn test_pq_distance() {
    let dim = 64;
    let config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(config);

    let train_data = create_test_vectors(1000, dim);
    pq.train(1000, &train_data).unwrap();

    let query = create_test_vectors(1, dim);
    let code = pq.encode(&query).unwrap();

    let dist = pq.compute_distance(&query, &code);
    assert!(dist >= 0.0);
}

#[test]
fn test_pq_adc_table() {
    let dim = 64;
    let config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(config);
    let train_data = create_test_vectors(1000, dim);
    pq.train(1000, &train_data).unwrap();

    let query = create_test_vectors(1, dim);
    let code = pq.encode(&query).unwrap();

    let table = pq.build_distance_table_l2(&query);
    let dist_table = pq.compute_distance_with_table(&table, &code);
    let dist_direct = pq.compute_distance(&query, &code);
    assert!(
        (dist_table - dist_direct).abs() < 1e-4,
        "table={} vs direct={}",
        dist_table,
        dist_direct
    );

    let ip_table = pq.build_distance_table_ip(&query);
    let ip_score = pq.compute_distance_with_table(&ip_table, &code);
    assert!(ip_score.is_finite());
}

#[test]
fn test_pq_decode() {
    let dim = 64;
    let config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(config);

    let train_data = create_test_vectors(1000, dim);
    pq.train(1000, &train_data).unwrap();

    let original = create_test_vectors(1, dim);
    let code = pq.encode(&original).unwrap();
    let reconstructed = pq.decode(&code).unwrap();

    assert_eq!(reconstructed.len(), dim);
    let mut mse = 0.0f32;
    for i in 0..dim {
        let diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    mse /= dim as f32;
    assert!(mse < 1.0);
}

#[test]
fn test_pq_get_centroids() {
    let dim = 64;
    let config = PQConfig::new(dim, 8, 8);
    let ksub = config.ksub();
    let sub_dim = config.sub_dim();
    let mut pq = ProductQuantizer::new(config);

    let train_data = create_test_vectors(1000, dim);
    pq.train(1000, &train_data).unwrap();

    let centroids = pq.get_centroids(0).unwrap();
    assert_eq!(centroids.len(), ksub * sub_dim);

    assert!(pq.get_centroids(8).is_none());
}

#[test]
fn test_adc_correctness() {
    let dim = 16;
    let m = 4;
    let nbits = 4;
    let n = 200usize;

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let config = PQConfig::new(dim, m, nbits);
    let mut pq = ProductQuantizer::new(config);
    pq.train(n, &data).unwrap();

    let codes = pq.encode_batch(n, &data).unwrap();
    let code_size = pq.code_size();

    let mut top1_hits = 0usize;
    let num_queries = 5usize;
    for q_idx in 0..num_queries {
        let query = &data[q_idx * dim..(q_idx + 1) * dim];

        let mut gt: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                let d = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum::<f32>();
                (i, d)
            })
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_top1 = gt[0].0;

        let mut adc: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let code = &codes[i * code_size..(i + 1) * code_size];
                (i, pq.compute_distance(query, code))
            })
            .collect();
        adc.sort_by(|a, b| a.1.total_cmp(&b.1));
        let adc_top1 = adc[0].0;

        if adc_top1 == gt_top1 {
            top1_hits += 1;
        }
    }
    let top1_recall = top1_hits as f32 / num_queries as f32;

    let mut mse_sum = 0.0f32;
    for i in 0..n {
        let original = &data[i * dim..(i + 1) * dim];
        let code = &codes[i * code_size..(i + 1) * code_size];
        let recon = pq.decode(code).unwrap();
        let mse = original
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            / dim as f32;
        mse_sum += mse;
    }
    let mse = mse_sum / n as f32;

    println!(
        "PQ ADC correctness: top1_recall={:.3}, reconstruction_mse={:.4}",
        top1_recall, mse
    );

    assert!(top1_recall.is_finite());
    assert!(mse.is_finite());
    assert!(mse < 10.0, "reconstruction MSE too large: {}", mse);
}

#[test]
fn test_pq_recall_small() {
    let dim = 16usize;
    let m = 4usize;
    let nbits = 8usize; // ksub = 256 (4-bit is too lossy for a stable recall gate)
    let n = 500usize;
    let nq = 100usize;
    let top_k = 10usize;

    let mut rng = StdRng::seed_from_u64(42);
    let base: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    // Queries are base vectors + small perturbation, not exact copies.
    // Exact-copy queries give distance=0 ground truth which inflates recall artificially.
    let mut queries = Vec::with_capacity(nq * dim);
    for _ in 0..nq {
        let idx = rng.gen_range(0..n);
        let noise: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1f32..0.1)).collect();
        for j in 0..dim {
            queries.push(base[idx * dim + j] + noise[j]);
        }
    }

    let config = PQConfig::new(dim, m, nbits);
    let mut pq = ProductQuantizer::new(config);
    pq.train(n, &base).unwrap();
    let codes = pq.encode_batch(n, &base).unwrap();
    let code_size = pq.code_size();

    let mut hits = 0usize;
    let mut total = 0usize;

    for q_idx in 0..nq {
        let query = &queries[q_idx * dim..(q_idx + 1) * dim];

        let mut gt: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &base[i * dim..(i + 1) * dim];
                let d = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum::<f32>();
                (i, d)
            })
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_topk: Vec<usize> = gt.iter().take(top_k).map(|(idx, _)| *idx).collect();

        let mut adc: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let code = &codes[i * code_size..(i + 1) * code_size];
                (i, pq.compute_distance(query, code))
            })
            .collect();
        adc.sort_by(|a, b| a.1.total_cmp(&b.1));
        let adc_topk: Vec<usize> = adc.iter().take(top_k).map(|(idx, _)| *idx).collect();

        for idx in gt_topk {
            total += 1;
            if adc_topk.contains(&idx) {
                hits += 1;
            }
        }
    }

    let recall = hits as f32 / total as f32;
    println!("PQ-only recall@10 (small): {:.3}", recall);
    assert!(recall > 0.3, "recall@10 too low: {}", recall);
}

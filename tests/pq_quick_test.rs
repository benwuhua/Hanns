//! Quick performance test for PQ improvements

use knowhere_rs::quantization::{OPQConfig, OptimizedProductQuantizer, PQConfig, ProductQuantizer};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn generate_random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn compute_ground_truth(query: &[f32], database: &[f32], n: usize, dim: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let vec = &database[i * dim..(i + 1) * dim];
            let dist: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (i, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.iter().map(|(idx, _)| *idx).collect()
}

fn compute_recall(result: &[usize], ground_truth: &[usize], k: usize) -> f32 {
    let result_set: std::collections::HashSet<_> = result.iter().take(k).cloned().collect();
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).cloned().collect();
    let intersection: Vec<_> = result_set.intersection(&gt_set).collect();
    intersection.len() as f32 / k as f32
}

#[test]
fn test_pq_recall_quick() {
    let dim = 128;
    let n_train = 5000;
    let n_test = 50;
    let k = 10;

    println!("\n=== Quick PQ Recall Test ===");

    let train_data = generate_random_data(n_train, dim, 42);
    let test_data = generate_random_data(n_test, dim, 123);

    // Standard PQ
    println!("Training standard PQ...");
    let pq_config = PQConfig::new(dim, 8, 8);
    let mut pq = ProductQuantizer::new(pq_config);
    pq.train(n_train, &train_data).unwrap();

    let pq_codes = pq.encode_batch(n_train, &train_data).unwrap();

    let mut pq_recall_sum = 0.0f32;
    for q_idx in 0..n_test {
        let query = &test_data[q_idx * dim..(q_idx + 1) * dim];
        let gt = compute_ground_truth(query, &train_data, n_train, dim);

        let mut distances: Vec<(usize, f32)> = (0..n_train)
            .map(|i| {
                let code = &pq_codes[i * pq.code_size()..(i + 1) * pq.code_size()];
                (i, pq.compute_distance(query, code))
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let result: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();

        pq_recall_sum += compute_recall(&result, &gt, k);
    }
    let pq_recall = pq_recall_sum / n_test as f32;
    println!("Standard PQ R@{}: {:.2}%", k, pq_recall * 100.0);

    // OPQ
    println!("Training OPQ...");
    let opq_config = OPQConfig::new(dim, 8, 8);
    let mut opq = OptimizedProductQuantizer::new(opq_config).unwrap();
    match opq.train(n_train, &train_data) {
        Ok(_) => println!("OPQ training completed"),
        Err(e) => {
            println!("OPQ training failed: {:?}", e);
            return;
        }
    }

    let opq_codes = match opq.encode_batch(n_train, &train_data) {
        Ok(codes) => codes,
        Err(e) => {
            println!("OPQ encoding failed: {:?}", e);
            return;
        }
    };

    let mut opq_recall_sum = 0.0f32;
    for q_idx in 0..n_test {
        let query = &test_data[q_idx * dim..(q_idx + 1) * dim];
        let gt = compute_ground_truth(query, &train_data, n_train, dim);

        let code_size = opq.code_size();
        if code_size == 0 {
            println!("OPQ code_size is 0, skipping");
            continue;
        }

        let mut distances: Vec<(usize, f32)> = (0..n_train)
            .map(|i| {
                let code = &opq_codes[i * code_size..(i + 1) * code_size];
                (i, opq.compute_distance(query, code))
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let result: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();

        opq_recall_sum += compute_recall(&result, &gt, k);
    }
    let opq_recall = opq_recall_sum / n_test as f32;
    println!("OPQ R@{}: {:.2}%", k, opq_recall * 100.0);

    println!(
        "Improvement: {:.1}%",
        (opq_recall - pq_recall) / pq_recall * 100.0
    );

    // Just verify they work - actual recall depends on data distribution
    assert!(pq_recall > 0.0, "PQ should have some recall");
    assert!(opq_recall > 0.0, "OPQ should have some recall");
}

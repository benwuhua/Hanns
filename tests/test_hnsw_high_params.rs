/// Test HNSW with higher M and ef_search parameters
///
/// This test checks if increasing M and ef_search improves recall.
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::index::Index;
use rand::Rng;
use std::collections::HashSet;

fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_ground_truth(base: &[f32], query: &[f32], k: usize, dim: usize) -> Vec<i64> {
    let mut distances: Vec<(usize, f32)> = base
        .chunks(dim)
        .enumerate()
        .map(|(idx, vec)| (idx, l2_distance_sq(query, vec)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances.into_iter().map(|(idx, _)| idx as i64).collect()
}

#[test]
fn test_hnsw_high_params() {
    let dim = 128;
    let num_base = 10000;
    let num_queries = 10;

    println!("\n=== HNSW High Parameters Test ===");
    println!("Testing if higher M and ef_search improve recall");

    // Generate random data
    let mut rng = rand::thread_rng();
    let base: Vec<f32> = (0..num_base * dim).map(|_| rng.gen::<f32>()).collect();
    let queries: Vec<f32> = (0..num_queries * dim).map(|_| rng.gen::<f32>()).collect();

    // Test with different parameter combinations
    let test_cases = vec![
        ("M=16, ef=200", 16, 200),
        ("M=32, ef=400", 32, 400),
        ("M=48, ef=600", 48, 600),
        ("M=64, ef=800", 64, 800),
    ];

    for (name, m, ef_search) in test_cases {
        println!("\n--- Testing {} ---", name);

        // Build HNSW index
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            dim,
            metric_type: MetricType::L2,
            params: IndexParams::hnsw(m, ef_search, 0.5),
        };

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&base).unwrap();
        index.add(&base, None).unwrap();

        // Search
        let mut total_r1 = 0.0;
        let mut total_r10 = 0.0;
        let mut total_r100 = 0.0;

        for q_idx in 0..num_queries {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];

            // Ground truth
            let gt = compute_ground_truth(&base, q, 100, dim);

            // HNSW search
            let req = SearchRequest {
                top_k: 100,
                ..Default::default()
            };
            let result = index.search(q, &req).unwrap();

            // Calculate recall
            let gt_set: HashSet<i64> = gt.into_iter().collect();
            let matched_1 = result
                .ids
                .iter()
                .take(1)
                .filter(|&id| gt_set.contains(id))
                .count();
            let matched_10 = result
                .ids
                .iter()
                .take(10)
                .filter(|&id| gt_set.contains(id))
                .count();
            let matched_100 = result
                .ids
                .iter()
                .take(100)
                .filter(|&id| gt_set.contains(id))
                .count();

            total_r1 += matched_1 as f64;
            total_r10 += matched_10 as f64;
            total_r100 += matched_100 as f64;
        }

        let avg_r1 = total_r1 / num_queries as f64;
        let avg_r10 = total_r10 / num_queries as f64 / 10.0;
        let avg_r100 = total_r100 / num_queries as f64 / 100.0;

        println!("  R@1:   {:.3}", avg_r1);
        println!("  R@10:  {:.3}", avg_r10);
        println!("  R@100: {:.3}", avg_r100);
    }
}

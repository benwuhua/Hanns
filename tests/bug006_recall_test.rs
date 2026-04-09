// BUG-006: Test to verify HNSW recall improvement after heuristic fix
use hanns::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use hanns::faiss::HnswIndex;

fn generate_random_data(n: usize, dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn calculate_recall_at_k(results: &[i64], ground_truth: &[i64], k: usize) -> f32 {
    let result_set: std::collections::HashSet<i64> = results.iter().take(k).cloned().collect();
    let gt_set: std::collections::HashSet<i64> = ground_truth.iter().take(k).cloned().collect();
    let intersection = result_set.intersection(&gt_set).count();
    intersection as f32 / k as f32
}

#[test]
fn test_hnsw_recall_after_heuristic_fix() {
    let n = 1000;
    let dim = 64;
    let k = 10;

    // Generate random data
    let data = generate_random_data(n, dim);

    // Build HNSW index
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        dim,
        params: Default::default(),
    };

    let mut index = HnswIndex::new(&config).expect("Failed to create index");

    // Train and add
    index.train(&data).expect("Failed to train");
    index.add(&data, None).expect("Failed to add"); // Use sequential IDs

    // Search with low ef to test heuristic
    let search_req = SearchRequest {
        top_k: k,
        nprobe: 64, // ef_search
        ..Default::default()
    };

    // Test multiple queries
    let mut total_recall = 0.0;
    let num_queries = 10;

    for i in 0..num_queries {
        let query_idx = i * 100 + 50;
        let query = &data[query_idx * dim..(query_idx + 1) * dim];

        // Search using HNSW
        let results = index.search(query, &search_req).expect("Failed to search");
        let hnsw_ids = &results.ids;

        // Calculate ground truth (brute force)
        let mut distances: Vec<(i64, f32)> = (0..n as i64)
            .map(|id| {
                let start = id as usize * dim;
                let vec = &data[start..start + dim];
                let dist: f32 = query
                    .iter()
                    .zip(vec.iter())
                    .map(|(q, v)| (q - v).powi(2))
                    .sum();
                (id, dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: Vec<i64> = distances.iter().take(k).map(|(id, _)| *id).collect();

        let recall = calculate_recall_at_k(hnsw_ids, &ground_truth, k);
        total_recall += recall;

        println!("Query {}: R@{} = {:.3}", i, k, recall);
    }

    let avg_recall = total_recall / num_queries as f32;
    println!("\n=== BUG-006 Test Results ===");
    println!(
        "Average R@{}: {:.3} ({:.1}%)",
        k,
        avg_recall,
        avg_recall * 100.0
    );
    println!("Target: > 0.85 (85%)");

    // After heuristic fix, recall should be significantly improved
    // With random data, we expect >85% recall (much better than the 32% before fix)
    assert!(
        avg_recall > 0.85,
        "Recall too low: {:.3} (expected > 0.85)",
        avg_recall
    );

    println!("✅ BUG-006 fix verified: HNSW heuristic algorithm working correctly!");
}

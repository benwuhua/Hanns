use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::collections::HashSet;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[test]
fn debug_hnsw_scale_test() {
    let n = 10000;
    let dim = 128;
    let vectors = generate_vectors(n, dim);
    let query = generate_vectors(1, dim);

    // Compute ground truth
    let q = &query[0..dim];
    let mut gt_distances: Vec<(usize, f32)> = (0..n)
        .map(|j| {
            let b = &vectors[j * dim..(j + 1) * dim];
            (j, l2_distance_squared(q, b))
        })
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let gt_top10: HashSet<usize> = gt_distances.iter().take(10).map(|(id, _)| *id).collect();
    let gt_top100: HashSet<usize> = gt_distances.iter().take(100).map(|(id, _)| *id).collect();

    println!("Testing n={}, dim={}", n, dim);

    // Build HNSW
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams {
            m: Some(32),
            ef_construction: Some(400),
            ef_search: Some(400),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    // Search with k=100
    let req = SearchRequest {
        top_k: 100,
        ..Default::default()
    };
    let result = index.search(q, &req).unwrap();

    println!("HNSW returned {} results", result.ids.len());

    // Check R@10
    let hnsw_top10: HashSet<usize> = result.ids.iter().take(10).map(|id| *id as usize).collect();
    let matched_10 = hnsw_top10.intersection(&gt_top10).count();
    println!(
        "R@10: {}/10 = {:.1}%",
        matched_10,
        matched_10 as f64 / 10.0 * 100.0
    );

    // Check R@100
    let hnsw_top100: HashSet<usize> = result.ids.iter().take(100).map(|id| *id as usize).collect();
    let matched_100 = hnsw_top100.intersection(&gt_top100).count();
    println!(
        "R@100: {}/100 = {:.1}%",
        matched_100,
        matched_100 as f64 / 100.0 * 100.0
    );

    // Show HNSW results
    println!("\nHNSW top-10:");
    for i in 0..10.min(result.ids.len()) {
        let id = result.ids[i] as usize;
        let dist = result.distances[i];
        let in_gt10 = if gt_top10.contains(&id) { "✓" } else { "✗" };
        let gt_rank = gt_distances
            .iter()
            .position(|(gt_id, _)| *gt_id == id)
            .map(|p| p + 1)
            .unwrap_or(999);
        println!(
            "  {}: id={}, dist={:.4} {} (GT rank: {})",
            i + 1,
            id,
            dist,
            in_gt10,
            gt_rank
        );
    }
}

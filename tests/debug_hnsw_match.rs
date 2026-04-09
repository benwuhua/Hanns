#![cfg(feature = "long-tests")]
mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::collections::HashSet;



#[test]
fn debug_hnsw_match_analysis() {
    let n = 1000;
    let dim = 128;
    let vectors = common::generate_vectors(n, dim);
    let query = common::generate_vectors(1, dim);

    // Compute ground truth
    let q = &query[0..dim];
    let mut gt_distances: Vec<(usize, f32)> = (0..n)
        .map(|j| {
            let b = &vectors[j * dim..(j + 1) * dim];
            (j, common::l2_distance_squared(q, b))
        })
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let gt_top10: HashSet<usize> = gt_distances.iter().take(10).map(|(id, _)| *id).collect();

    println!("Ground truth top-10:");
    for (i, (id, dist)) in gt_distances.iter().take(10).enumerate() {
        println!(
            "  {}: id={}, dist^2={:.4}, dist={:.4}",
            i + 1,
            id,
            dist,
            dist.sqrt()
        );
    }

    // Build HNSW
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
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

    // Search
    let req = SearchRequest {
        top_k: 10,
        ..Default::default()
    };
    let result = index.search(q, &req).unwrap();

    println!("\nHNSW top-10:");
    for i in 0..10.min(result.ids.len()) {
        let id = result.ids[i] as usize;
        let dist = result.distances[i];
        let in_gt = if gt_top10.contains(&id) { "✓" } else { "✗" };

        // Compute actual distance
        let b = &vectors[id * dim..(id + 1) * dim];
        let actual_dist_sq = common::l2_distance_squared(q, b);

        println!(
            "  {}: id={}, dist={:.4}, dist^2={:.4} {} (GT rank: {})",
            i + 1,
            id,
            dist,
            actual_dist_sq,
            in_gt,
            gt_distances
                .iter()
                .position(|(gt_id, _)| *gt_id == id)
                .map(|p| p + 1)
                .unwrap_or(999)
        );
    }

    // Count matches
    let hnsw_set: HashSet<usize> = result.ids.iter().take(10).map(|id| *id as usize).collect();
    let matched = hnsw_set.intersection(&gt_top10).count();
    println!("\nMatched: {}/10", matched);

    // Check if HNSW results are sorted correctly
    let mut sorted = true;
    for i in 1..result.distances.len() {
        if result.distances[i] < result.distances[i - 1] - 1e-6 {
            sorted = false;
            println!(
                "Sorting issue: dist[{}]={:.4} < dist[{}]={:.4}",
                i,
                result.distances[i],
                i - 1,
                result.distances[i - 1]
            );
        }
    }
    println!("Distances sorted: {}", sorted);
}

#![cfg(feature = "long-tests")]
mod common;
use hanns::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use hanns::faiss::HnswIndex;
use hanns::MetricType;
use rand::Rng;
use std::collections::HashSet;



#[test]
fn debug_entry_point_test() {
    let n = 10000;
    let dim = 128;
    let vectors = common::generate_vectors(n, dim);
    let query = common::generate_vectors(1, dim);
    let q = &query[0..dim];

    // Compute ground truth
    let mut gt_distances: Vec<(usize, f32)> = (0..n)
        .map(|j| {
            let b = &vectors[j * dim..(j + 1) * dim];
            (j, common::l2_distance_squared(q, b))
        })
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let gt_top10: Vec<(usize, f32)> = gt_distances.iter().take(10).cloned().collect();

    println!("Query distances to GT top-10:");
    for (i, (id, dist)) in gt_top10.iter().enumerate() {
        println!(
            "  Rank {}: id={}, dist^2={:.4}, dist={:.4}",
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
        data_type: hanns::api::DataType::Float,
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

    let (max_layer, _, avg_neighbors) = index.get_graph_stats();
    println!(
        "\nGraph stats: max_layer={}, avg_neighbors_l0={:.1}",
        max_layer, avg_neighbors
    );

    // Search with detailed analysis
    println!("\nSearching with ef=400, k=100:");
    let req = SearchRequest {
        top_k: 100,
        ..Default::default()
    };
    let result = index.search(q, &req).unwrap();

    println!("HNSW top-10 results:");
    let gt_set: HashSet<usize> = gt_top10.iter().map(|(id, _)| *id).collect();
    for i in 0..10.min(result.ids.len()) {
        let id = result.ids[i] as usize;
        let dist = result.distances[i];
        let in_gt = if gt_set.contains(&id) { "✓" } else { "✗" };
        let gt_rank = gt_distances
            .iter()
            .position(|(gt_id, _)| *gt_id == id)
            .map(|p| p + 1)
            .unwrap_or(999);

        // Compute actual dist^2
        let b = &vectors[id * dim..(id + 1) * dim];
        let actual_dist_sq = common::l2_distance_squared(q, b);

        println!(
            "  {}: id={}, dist={:.4}, dist^2={:.4} {} (GT rank: {})",
            i + 1,
            id,
            dist,
            actual_dist_sq,
            in_gt,
            gt_rank
        );
    }

    // Key question: why is GT rank 2-8 not found?
    // Let's check if they're close to GT rank 1 in the graph
    println!("\nChecking GT rank 1 (id={}) neighbors:", gt_top10[0].0);
    // We can't access internal neighbors, but we can check if searching from GT nodes helps

    // Test: use each GT top-10 as query - if HNSW can find them as top-1
    println!("\nCan HNSW find each GT top-10 as top-1 when using them as query?");
    for (i, (gt_id, _)) in gt_top10.iter().enumerate() {
        let gt_vec = &vectors[*gt_id * dim..(*gt_id + 1) * dim];
        let req = SearchRequest {
            top_k: 1,
            ..Default::default()
        };
        let result = index.search(gt_vec, &req).unwrap();

        let top1_id = result.ids.first().map(|id| *id as usize);
        let found = top1_id == Some(*gt_id);
        println!(
            "  GT rank {} (id={}): found as top-1 = {}",
            i + 1,
            gt_id,
            found
        );
    }

    // Test: distance from GT rank 1 to GT rank 2-10
    println!(
        "\nDistance from GT rank 1 (id={}) to other GT nodes:",
        gt_top10[0].0
    );
    let gt1_vec = &vectors[gt_top10[0].0 * dim..(gt_top10[0].0 + 1) * dim];
    for (i, (gt_id, _)) in gt_top10.iter().skip(1).enumerate() {
        let gt_vec = &vectors[*gt_id * dim..(*gt_id + 1) * dim];
        let dist = common::l2_distance_squared(gt1_vec, gt_vec).sqrt();
        println!(
            "  GT rank {} (id={}): dist from GT1 = {:.4}",
            i + 2,
            gt_id,
            dist
        );
    }
}

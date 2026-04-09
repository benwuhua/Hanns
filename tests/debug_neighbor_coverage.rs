#![cfg(feature = "long-tests")]
mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;



#[test]
fn debug_neighbor_coverage() {
    let n = 10000;
    let dim = 128;
    let vectors = common::generate_vectors(n, dim);
    let query = common::generate_vectors(1, dim);
    let q = &query[0..dim];

    // Compute ground truth
    let mut gt_distances: Vec<(usize, f32)> = (0..n)
        .map(|j| (j, common::l2_distance_squared(q, &vectors[j * dim..(j + 1) * dim])))
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let gt_top10: Vec<usize> = gt_distances.iter().take(10).map(|(id, _)| *id).collect();

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

    // Check: for each GT top-10, what is its distance to other GT top-10 nodes?
    println!("GT top-10 inter-node distances:");
    print!("        ");
    for j in 0..10 {
        print!("GT{:2}   ", j + 1);
    }
    println!();

    for i in 0..10 {
        print!("GT{:2}: ", i + 1);
        let vi = &vectors[gt_top10[i] * dim..(gt_top10[i] + 1) * dim];
        for j in 0..10 {
            let vj = &vectors[gt_top10[j] * dim..(gt_top10[j] + 1) * dim];
            let dist = common::l2_distance_squared(vi, vj).sqrt();
            print!("{:.2}  ", dist);
        }
        println!();
    }

    // The key insight: HNSW relies on graph edges to navigate
    // If GT nodes are far apart in the graph, they won't be found together

    // Check how many hops between GT rank 1 and other GT nodes
    // (We can't directly access the graph, but we can simulate)

    // Alternative approach: check if using GT rank 1 as entry point improves recall
    println!(
        "\n\nSimulating search starting from GT rank 1 (id={}):",
        gt_top10[0]
    );

    // First, find the distance from query to GT rank 1
    let gt1_vec = &vectors[gt_top10[0] * dim..(gt_top10[0] + 1) * dim];
    let q_to_gt1 = common::l2_distance_squared(q, gt1_vec).sqrt();
    println!("Query to GT rank 1 distance: {:.4}", q_to_gt1);

    // Search and check what was found
    let req = SearchRequest {
        top_k: 100,
        ..Default::default()
    };
    let result = index.search(q, &req).unwrap();

    // Check if each GT top-10 is in the results
    println!("\nGT top-10 found in HNSW results:");
    let result_set: std::collections::HashSet<i64> = result.ids.iter().cloned().collect();
    for (i, &gt_id) in gt_top10.iter().enumerate() {
        let found_at = result.ids.iter().position(|id| *id as usize == gt_id);
        match found_at {
            Some(pos) => println!(
                "  GT rank {} (id={}): found at position {}",
                i + 1,
                gt_id,
                pos + 1
            ),
            None => println!("  GT rank {} (id={}): NOT FOUND in top 100", i + 1, gt_id),
        }
    }

    // Summary
    let found_count = gt_top10
        .iter()
        .filter(|id| result_set.contains(&(**id as i64)))
        .count();
    println!(
        "\nSummary: {}/10 GT top-10 found in HNSW top 100",
        found_count
    );
}

#![cfg(feature = "long-tests")]
mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;



#[test]
fn debug_hnsw_recall() {
    let n = 1000;
    let dim = 128;
    let vectors = common::generate_vectors(n, dim);
    let queries = common::generate_vectors(10, dim);

    // Compute ground truth for first query
    let q = &queries[0..dim];
    let mut distances: Vec<(usize, f32)> = (0..n)
        .map(|j| {
            let b = &vectors[j * dim..(j + 1) * dim];
            (j, common::l2_distance_squared(q, b))
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let gt_top10: Vec<i32> = distances
        .iter()
        .take(10)
        .map(|(id, _)| *id as i32)
        .collect();
    let gt_top100: Vec<i32> = distances
        .iter()
        .take(100)
        .map(|(id, _)| *id as i32)
        .collect();

    println!("Ground truth top-10: {:?}", gt_top10);
    println!("Ground truth top-100 first 10: {:?}", &gt_top100[..10]);

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
        top_k: 100,
        ..Default::default()
    };
    let result = index.search(q, &req).unwrap();

    println!("\nHNSW returned {} results", result.ids.len());
    println!(
        "HNSW top-10 IDs: {:?}",
        &result.ids[..10.min(result.ids.len())]
    );
    println!(
        "HNSW top-10 distances: {:?}",
        &result.distances[..10.min(result.distances.len())]
    );

    // Calculate recall
    use std::collections::HashSet;
    let gt_set_10: HashSet<i32> = gt_top10.iter().copied().collect();
    let gt_set_100: HashSet<i32> = gt_top100.iter().copied().collect();

    let matched_10 = result
        .ids
        .iter()
        .take(10)
        .filter(|id| gt_set_10.contains(&(**id as i32)))
        .count();
    let matched_100 = result
        .ids
        .iter()
        .take(100)
        .filter(|id| gt_set_100.contains(&(**id as i32)))
        .count();

    println!(
        "\nRecall@10: {}/10 = {:.3}",
        matched_10,
        matched_10 as f64 / 10.0
    );
    println!(
        "Recall@100: {}/{} = {:.3}",
        matched_100,
        result.ids.len().min(100),
        matched_100 as f64 / result.ids.len().min(100) as f64
    );
}

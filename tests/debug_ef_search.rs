#![cfg(feature = "long-tests")]
mod common;
use hanns::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use hanns::faiss::HnswIndex;
use hanns::MetricType;
use rand::Rng;
use std::collections::HashSet;

#[test]
fn debug_ef_search_test() {
    let n = 10000;
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
    let gt_top100: HashSet<usize> = gt_distances.iter().take(100).map(|(id, _)| *id).collect();

    // Build HNSW once
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(32),
            ef_construction: Some(400),
            ef_search: Some(400), // Will be overridden
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    // Test different ef_search values
    for ef in [64, 128, 256, 400, 800, 1600, 3200] {
        index.set_ef_search(ef);

        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();

        let hnsw_top10: HashSet<usize> =
            result.ids.iter().take(10).map(|id| *id as usize).collect();
        let hnsw_top100: HashSet<usize> =
            result.ids.iter().take(100).map(|id| *id as usize).collect();

        let r10 = hnsw_top10.intersection(&gt_top10).count();
        let r100 = hnsw_top100.intersection(&gt_top100).count();

        println!(
            "ef={:4}: R@10={}/10 ({:.0}%), R@100={}/100 ({:.0}%)",
            ef,
            r10,
            r10 as f64 * 10.0,
            r100,
            r100 as f64
        );
    }
}

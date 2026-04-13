#![cfg(feature = "long-tests")]

mod common;
use hanns::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use hanns::faiss::HnswIndex;
use hanns::MetricType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn debug_graph_connectivity() {
    let n = 4000;
    let dim = 64;
    let vectors = common::generate_vectors(20260311, n, dim);
    let query = common::generate_vectors(20260312, 1, dim);
    let q = &query[0..dim];

    let mut gt_distances: Vec<(usize, f32)> = (0..n)
        .map(|j| {
            let start = j * dim;
            (
                j,
                common::l2_distance_squared(q, &vectors[start..start + dim]),
            )
        })
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let gt_top10: Vec<usize> = gt_distances.iter().take(10).map(|(id, _)| *id).collect();

    println!("Ground truth top-10: {:?}", gt_top10);

    for m in [8, 16, 32] {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            dim,
            metric_type: MetricType::L2,
            data_type: hanns::api::DataType::Float,
            params: IndexParams {
                m: Some(m),
                ef_construction: Some(400),
                ef_search: Some(400),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        let repaired = index.find_and_repair_unreachable();

        let unreachable = index.find_unreachable_vectors();
        assert!(
            unreachable.is_empty(),
            "M={} repair-backed build should not leave unreachable vectors after {} repairs: {:?}",
            m,
            repaired,
            unreachable
        );

        let (max_layer, _, avg_neighbors) = index.get_graph_stats();
        let (_, max_l0, avg_l0) = index.layer_neighbor_count_stats(0).unwrap();

        println!(
            "M={}, repairs={}, max_layer={}, max_neighbors_l0={}, avg_neighbors_l0={:.1}",
            m, repaired, max_layer, max_l0, avg_neighbors
        );

        assert!(max_layer >= 1, "M={} should build more than one layer", m);
        assert!(
            max_l0 <= m * 2 + repaired,
            "M={} layer-0 degree should stay within 2*M plus repair slack, got {} after {} repairs",
            m,
            max_l0,
            repaired
        );
        assert!(
            avg_l0 >= m as f32,
            "M={} layer-0 average degree should stay populated, got {:.2}",
            m,
            avg_l0
        );

        let req = SearchRequest {
            top_k: 1,
            ..Default::default()
        };
        let mut found_self = 0usize;
        for &gt_id in &gt_top10 {
            let start = gt_id * dim;
            let gt_vec = &vectors[start..start + dim];
            let result = index.search(gt_vec, &req).unwrap();
            if result.ids.first().map(|id| *id as usize) == Some(gt_id) {
                found_self += 1;
            }
        }
        assert_eq!(
            found_self, 10,
            "M={} should allow each GT top-10 vector to retrieve itself",
            m
        );
    }
}

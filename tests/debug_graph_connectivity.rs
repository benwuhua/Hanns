use knowhere_rs::api::{IndexConfig, IndexParams, IndexType};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::collections::{HashSet, VecDeque};

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// BFS to check connectivity from entry point
fn count_reachable_nodes(index: &HnswIndex, start_idx: usize) -> usize {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(start_idx);
    queue.push_back(start_idx);

    while let Some(curr_idx) = queue.pop_front() {
        // Get neighbors at layer 0
        let stats = index.get_graph_stats();
        // We need to access internal node_info, but it's private...
        // Instead, let's use a different approach
    }

    visited.len()
}

#[test]
fn debug_graph_connectivity() {
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

    let gt_top10: Vec<usize> = gt_distances.iter().take(10).map(|(id, _)| *id).collect();

    println!("Ground truth top-10: {:?}", gt_top10);

    // Build HNSW with different M values
    for m in [8, 16, 32] {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            dim,
            metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
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

        let (max_layer, layer_dist, avg_neighbors) = index.get_graph_stats();

        // Check connectivity at layer 0 using search from each GT node
        println!(
            "\nM={}, max_layer={}, avg_neighbors_l0={:.1}",
            m, max_layer, avg_neighbors
        );

        // Try searching from each GT top-10 node to see if they can be found
        for (i, &gt_id) in gt_top10.iter().enumerate() {
            let gt_vec = &vectors[gt_id * dim..(gt_id + 1) * dim];
            let req = knowhere_rs::api::SearchRequest {
                top_k: 1,
                ..Default::default()
            };
            let result = index.search(gt_vec, &req).unwrap();

            let found_self = result.ids.first().map(|id| *id as usize) == Some(gt_id);
            println!(
                "  GT#{} (id={}): can find itself = {}",
                i + 1,
                gt_id,
                found_self
            );
        }
    }
}

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::collections::HashSet;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[test]
fn debug_ef_construction_test() {
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

    // Test different ef_construction values
    for ef_c in [100, 200, 400, 800] {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            dim,
            metric_type: MetricType::L2,
            params: IndexParams {
                m: Some(32),
                ef_construction: Some(ef_c),
                ef_search: Some(400),
                ..Default::default()
            },
        };

        let build_start = Instant::now();
        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        let build_time = build_start.elapsed().as_millis();

        // Test with ef_search=400
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

        // Also test with ef_search=1600
        index.set_ef_search(1600);
        let result2 = index.search(q, &req).unwrap();
        let hnsw_top10_2: HashSet<usize> =
            result2.ids.iter().take(10).map(|id| *id as usize).collect();
        let hnsw_top100_2: HashSet<usize> = result2
            .ids
            .iter()
            .take(100)
            .map(|id| *id as usize)
            .collect();
        let r10_2 = hnsw_top10_2.intersection(&gt_top10).count();
        let r100_2 = hnsw_top100_2.intersection(&gt_top100).count();

        println!("ef_c={:3} (build {}ms): ef_s=400  R@10={}/10 ({:.0}%), R@100={}/100 ({:.0}%) | ef_s=1600 R@10={}/10 ({:.0}%) R@100={}/100 ({:.0}%)",
            ef_c, build_time, r10, r10 as f64 * 10.0, r100, r100 as f64, r10_2, r10_2 as f64 * 10.0, r100_2, r100_2 as f64);
    }
}

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::{HnswIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::Rng;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist = l2_distance_squared(q, b);
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<i32> = distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx as i32)
            .collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

#[test]
fn debug_hnsw_perf_test() {
    let n = 10000;
    let dim = 128;
    let num_queries = 100;

    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(num_queries, dim);
    let ground_truth = compute_ground_truth(&vectors, &queries, num_queries, dim, 100);

    println!("Ground truth computed: {} queries, k=100", num_queries);
    println!("First query GT top-10: {:?}", &ground_truth[0][..10]);

    // Build HNSW with same config as perf_test.rs
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

    // Search with same config as perf_test.rs
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    println!("\nFirst query HNSW top-10: {:?}", &all_results[0][..10]);

    // Calculate recall
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    println!("\nRecall@1: {:.3}", recall_at_1);
    println!("Recall@10: {:.3}", recall_at_10);
    println!("Recall@100: {:.3}", recall_at_100);

    // Also test Flat index for comparison
    let flat_config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    let mut flat_index = FlatIndex::new(&flat_config).unwrap();
    flat_index.add(&vectors, None).unwrap();

    let mut flat_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = flat_index.search(query, &req).unwrap();
        flat_results.push(result.ids.clone());
    }

    let flat_recall_at_10 = average_recall_at_k(&flat_results, &ground_truth, 10);
    let flat_recall_at_100 = average_recall_at_k(&flat_results, &ground_truth, 100);

    println!("\nFlat Recall@10: {:.3}", flat_recall_at_10);
    println!("Flat Recall@100: {:.3}", flat_recall_at_100);
}

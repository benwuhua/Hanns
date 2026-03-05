//! Diagnose HNSW graph structure

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use rand::Rng;

fn main() {
    println!("\n=== HNSW Graph Diagnostics ===\n");

    let num_base = 1000;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let ef_search = 64;

    // Generate random data
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_base * dim);
    for _ in 0..(num_base * dim) {
        data.push(rng.gen_range(-100.0..100.0));
    }

    // Build index
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&data).unwrap();
    index.add(&data, None).unwrap();

    // Print graph stats
    println!(
        "Graph built with M={}, ef_construction={}",
        m, ef_construction
    );
    println!("\nGraph statistics will be printed via debug output...");

    // Search test
    let query = &data[0..dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: ef_search,
        ..Default::default()
    };
    let result = index.search(query, &req).unwrap();

    println!("\nSearch result for first vector (should return ID 0):");
    for i in 0..10 {
        println!(
            "  #{}: ID={} dist={:.4}",
            i + 1,
            result.ids[i],
            result.distances[i]
        );
    }
}

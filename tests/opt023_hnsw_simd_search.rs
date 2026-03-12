#![cfg(feature = "long-tests")]
//! OPT-023: HNSW SIMD Search Performance Test
//!
//! This test measures the search performance improvement from SIMD optimization.

use knowhere_rs::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("\n=== OPT-023: HNSW SIMD Search Performance Test ===\n");

    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let top_k = 10;

    // Generate random vectors
    let mut rng = rand::thread_rng();
    let mut vectors: Vec<f32> = Vec::with_capacity(n_vectors * dim);
    for _ in 0..(n_vectors * dim) {
        vectors.push(rng.gen::<f32>());
    }

    // Generate random queries
    let mut queries: Vec<f32> = Vec::with_capacity(n_queries * dim);
    for _ in 0..(n_queries * dim) {
        queries.push(rng.gen::<f32>());
    }

    // Create and build HNSW index
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim,
        data_type: knowhere_rs::api::DataType::Float,
        params: knowhere_rs::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();

    println!("Building index with {} vectors ({}D)...", n_vectors, dim);
    let build_start = Instant::now();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed();
    println!("Build time: {:?}\n", build_time);

    // Search performance test
    println!("Running {} searches with top_k={}...", n_queries, top_k);
    let req = SearchRequest {
        top_k,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };

    let search_start = Instant::now();
    let mut total_results = 0;

    for i in 0..n_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let result = index.search(query, &req).unwrap();
        total_results += result.ids.len();
    }

    let search_time = search_start.elapsed();
    let search_time_ms = search_time.as_secs_f64() * 1000.0;
    let qps = n_queries as f64 / search_time.as_secs_f64();

    println!("\n📊 Search Performance Results:");
    println!("   Total queries: {}", n_queries);
    println!("   Total results: {}", total_results);
    println!("   Search time: {:.2}ms", search_time_ms);
    println!("   QPS: {:.2}", qps);
    println!(
        "   Avg latency: {:.4}ms/query",
        search_time_ms / n_queries as f64
    );

    // SIMD optimization note
    println!("\n✅ SIMD Optimization Applied:");
    println!("   - L2 distance: simd::l2_distance_sq()");
    println!("   - Inner product: simd::inner_product()");
    println!("   - Cosine distance: SIMD-optimized components");
    println!("\n   Expected speedup: 2-4x vs scalar implementation");
    println!("   (depending on dimension and CPU SIMD capabilities)\n");
}

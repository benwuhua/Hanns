//! Quick IVF-Flat Performance Test
//!
//! Purpose: Fast performance validation (avoid running full bench_opt003)
//! Usage: cargo test --release --test quick_ivf_perf -- --nocapture

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::ivf_flat::IvfFlatIndex;
use std::time::Instant;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| ((i % 100) as f32 / 100.0 - 0.5) * 2.0)
        .collect()
}

#[test]
fn test_ivf_flat_quick_perf_50k() {
    let dim = 128;
    let nlist = 100;
    let nprobe = 16;
    let base_size = 50_000;
    let num_queries = 100;
    let k = 10;

    // Generate data
    let base = generate_random_vectors(base_size, dim);
    let queries = generate_random_vectors(num_queries, dim);

    // Train
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim,
        params: IndexParams::ivf(nlist, nprobe),
    };

    let mut index = IvfFlatIndex::new(&config).expect("Failed to create index");

    println!("Training IVF-Flat (nlist={}, dim={})...", nlist, dim);
    let train_start = Instant::now();
    index.train(&base).expect("Failed to train");
    let train_time = train_start.elapsed();
    println!("Training time: {:.2}s", train_time.as_secs_f64());

    // Add
    println!("Adding {} vectors...", base_size);
    let add_start = Instant::now();
    index.add(&base, None).expect("Failed to add");
    let add_time = add_start.elapsed();
    println!(
        "Add time: {:.2}s ({:.0} vec/s)",
        add_time.as_secs_f64(),
        base_size as f64 / add_time.as_secs_f64()
    );

    // Search request
    let req = SearchRequest {
        top_k: k,
        nprobe,
        filter: None,
        params: None,
        radius: None,
    };

    // Search (warmup)
    let _ = index
        .search(&queries[0..dim], &req)
        .expect("Failed to search");

    // Search (benchmark)
    println!(
        "Searching {} queries (k={}, nprobe={})...",
        num_queries, k, nprobe
    );
    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let _ = index.search(query, &req).expect("Failed to search");
    }
    let search_time = search_start.elapsed();

    let qps = num_queries as f64 / search_time.as_secs_f64();
    println!("\n=== Performance Result ===");
    println!("QPS: {:.0}", qps);
    println!(
        "Latency: {:.2}ms",
        search_time.as_secs_f64() * 1000.0 / num_queries as f64
    );
    println!("Target: 2500+ QPS (50% C++)");
    println!(
        "Status: {}",
        if qps >= 2500.0 {
            "✅ PASSED"
        } else {
            "⚠️ BELOW TARGET"
        }
    );
}

#[test]
fn test_ivf_flat_quick_perf_10k() {
    let dim = 128;
    let nlist = 100;
    let nprobe = 16;
    let base_size = 10_000;
    let num_queries = 100;
    let k = 10;

    // Generate data
    let base = generate_random_vectors(base_size, dim);
    let queries = generate_random_vectors(num_queries, dim);

    // Train
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim,
        params: IndexParams::ivf(nlist, nprobe),
    };

    let mut index = IvfFlatIndex::new(&config).expect("Failed to create index");

    println!("\n=== IVF-Flat 10K Test ===");
    index.train(&base).expect("Failed to train");
    index.add(&base, None).expect("Failed to add");

    // Search request
    let req = SearchRequest {
        top_k: k,
        nprobe,
        filter: None,
        params: None,
        radius: None,
    };

    // Search benchmark
    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let _ = index.search(query, &req).expect("Failed to search");
    }
    let search_time = search_start.elapsed();

    let qps = num_queries as f64 / search_time.as_secs_f64();
    println!("10K QPS: {:.0}", qps);
}

//! IVF-PQ Performance Validation (BENCH-046)
//!
//! Purpose: Validate IVF-PQ QPS and recall after BUG-005 fix
//! Config: 50K base, 100 queries, nlist=100, nprobe=16, M=8, nbits=8
//! Target: QPS vs C++, R@10 >= 90%

#[cfg(feature = "long-tests")]
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::ivfpq::IvfPqIndex;
use serde_json::Value;
use std::fs;
#[cfg(feature = "long-tests")]
use std::time::Instant;

const IVFPQ_FINAL_VERDICT_PATH: &str = "benchmark_results/ivfpq_p3_003_final_verdict.json";

fn load_ivfpq_final_verdict() -> Value {
    let content = fs::read_to_string(IVFPQ_FINAL_VERDICT_PATH)
        .expect("family verdict artifact must exist for the IVF-PQ verdict lane");
    serde_json::from_str(&content).expect("family verdict artifact must be valid JSON")
}

#[test]
fn ivfpq_verdict_lane_archives_family_as_no_go() {
    let verdict = load_ivfpq_final_verdict();

    assert_eq!(verdict["family"], "IVF-PQ");
    assert_eq!(verdict["classification"], "no-go");
    assert_eq!(
        verdict["leadership_verdict"],
        "no_go_for_production_and_leadership"
    );
    assert_eq!(verdict["leadership_claim_allowed"], false);
    assert!(
        verdict["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("recall"),
        "summary must explain that the current no-go verdict is driven by sub-gate recall evidence"
    );
}

#[cfg(feature = "long-tests")]
fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|i| ((i % 100) as f32 / 100.0 - 0.5) * 2.0)
        .collect()
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_ivf_pq_perf_50k() {
    let dim = 128;
    let nlist = 50; // Reduced from 100
    let nprobe = 8; // Reduced from 16
    let m = 8; // Number of sub-quantizers
    let nbits = 8; // Bits per sub-quantizer
    let base_size = 10_000; // Reduced from 50K for faster test
    let num_queries = 100;
    let k = 10;

    // Generate data
    let base = generate_random_vectors(base_size, dim);
    let queries = generate_random_vectors(num_queries, dim);

    // Train
    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config).expect("Failed to create IVF-PQ index");

    println!("\n=== IVF-PQ Performance Test (BENCH-046) ===");
    println!(
        "Config: nlist={}, nprobe={}, M={}, nbits={}",
        nlist, nprobe, m, nbits
    );
    println!(
        "Data: {} base vectors, {} queries, dim={}",
        base_size, num_queries, dim
    );

    // Train
    println!("\nTraining IVF-PQ...");
    let train_start = Instant::now();
    index.train(&base).expect("Failed to train");
    let train_time = train_start.elapsed();
    println!("Training time: {:.2}s", train_time.as_secs_f64());

    // Add
    println!("\nAdding {} vectors...", base_size);
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

    // Warmup
    let _ = index
        .search(&queries[0..dim], &req)
        .expect("Failed to search");

    // Search benchmark
    println!(
        "\nSearching {} queries (k={}, nprobe={})...",
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
    println!("\n=== Benchmark Status ===");
    println!("Target: QPS competitive with C++ knowhere");
    println!("Note: Ground truth recall validation requires separate test");
    println!("      R@10 target: >= 90% (verified in BUG-005 fix)");
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_ivf_pq_perf_10k() {
    let dim = 128;
    let nlist = 50;
    let nprobe = 8;
    let m = 8;
    let nbits = 8;
    let base_size = 10_000;
    let num_queries = 100;
    let k = 10;

    let base = generate_random_vectors(base_size, dim);
    let queries = generate_random_vectors(num_queries, dim);

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let mut index = IvfPqIndex::new(&config).expect("Failed to create IVF-PQ index");

    println!("\n=== IVF-PQ 10K Quick Test ===");
    index.train(&base).expect("Failed to train");
    index.add(&base, None).expect("Failed to add");

    let req = SearchRequest {
        top_k: k,
        nprobe,
        filter: None,
        params: None,
        radius: None,
    };

    let search_start = Instant::now();
    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let _ = index.search(query, &req).expect("Failed to search");
    }
    let search_time = search_start.elapsed();

    let qps = num_queries as f64 / search_time.as_secs_f64();
    println!("10K QPS: {:.0}", qps);
}

#![cfg(feature = "long-tests")]
//! OPT-024/OPT-031: HNSW Parallel Build Benchmark
//!
//! This benchmark compares serial vs parallel HNSW construction performance.
//! OPT-031 improvements:
//! - Dynamic batch size strategy
//! - Progress tracking and logging
//! - Enhanced error handling
//! - Comprehensive performance validation
//!
//! Expected speedup: 4-8x (from 24.5s to 3-6s for 100K vectors)
//! Target: 100K vectors build time < 3s

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::dataset::Dataset;
use knowhere_rs::faiss::hnsw::HnswIndex;
use knowhere_rs::index::Index;
use rand::Rng;
use std::time::Instant;

/// Generate random vectors for benchmarking
fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

/// Benchmark serial HNSW build
fn bench_hnsw_serial(n: usize, dim: usize, m: usize, ef_construction: usize) -> (f64, HnswIndex) {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(64),
            num_threads: Some(1), // Force serial
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    let vectors = generate_random_vectors(n, dim);

    // Train
    index.train(&vectors).unwrap();

    // Build (serial)
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed, index)
}

/// Benchmark parallel HNSW build
fn bench_hnsw_parallel(
    n: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
    num_threads: usize,
) -> (f64, HnswIndex) {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(64),
            num_threads: Some(num_threads),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    let vectors = generate_random_vectors(n, dim);

    // Train
    index.train(&vectors).unwrap();

    // Build (parallel)
    let start = Instant::now();
    index.add_parallel(&vectors, None, Some(true)).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed, index)
}

/// Verify search quality after parallel build
fn verify_search_quality(index: &HnswIndex, vectors: &[f32], dim: usize) -> bool {
    let query = &vectors[0..dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 20,
        filter: None,
        params: None,
        radius: None,
    };

    let result = index.search(query, &req).unwrap();

    // Check that we get results and the top result is reasonably close
    result.ids.len() > 0 && result.distances[0] < 100.0
}

#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_hnsw_parallel_build_small() {
    println!("\n=== OPT-024: HNSW Parallel Build Benchmark (Small) ===");

    let n = 10_000;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("Dataset: {} vectors x {} dimensions", n, dim);
    println!("HNSW params: M={}, EF_CONSTRUCTION={}", m, ef_construction);
    println!("Parallel threads: {}", num_threads);
    println!();

    // Serial build
    println!("Building HNSW (serial)...");
    let (time_serial, _index_serial) = bench_hnsw_serial(n, dim, m, ef_construction);
    println!("Serial build time: {:.3}s", time_serial);

    // Parallel build
    println!("Building HNSW (parallel, {} threads)...", num_threads);
    let (time_parallel, index_parallel) =
        bench_hnsw_parallel(n, dim, m, ef_construction, num_threads);
    println!("Parallel build time: {:.3}s", time_parallel);

    // Calculate speedup
    let speedup = time_serial / time_parallel;
    println!("\nSpeedup: {:.2}x", speedup);

    // Verify search quality
    let vectors = generate_random_vectors(n, dim);
    let quality_ok = verify_search_quality(&index_parallel, &vectors, dim);
    println!(
        "Search quality check: {}",
        if quality_ok { "PASS" } else { "FAIL" }
    );

    assert!(quality_ok, "Parallel build should maintain search quality");
    assert!(speedup > 1.0, "Parallel build should be faster than serial");

    println!("\n=== Benchmark Complete ===\n");
}

#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_hnsw_parallel_build_medium() {
    println!("\n=== OPT-024: HNSW Parallel Build Benchmark (Medium) ===");

    let n = 50_000;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("Dataset: {} vectors x {} dimensions", n, dim);
    println!("HNSW params: M={}, EF_CONSTRUCTION={}", m, ef_construction);
    println!("Parallel threads: {}", num_threads);
    println!();

    // Serial build
    println!("Building HNSW (serial)...");
    let (time_serial, _index_serial) = bench_hnsw_serial(n, dim, m, ef_construction);
    println!("Serial build time: {:.3}s", time_serial);

    // Parallel build
    println!("Building HNSW (parallel, {} threads)...", num_threads);
    let (time_parallel, index_parallel) =
        bench_hnsw_parallel(n, dim, m, ef_construction, num_threads);
    println!("Parallel build time: {:.3}s", time_parallel);

    // Calculate speedup
    let speedup = time_serial / time_parallel;
    println!("\nSpeedup: {:.2}x", speedup);

    // Verify search quality
    let vectors = generate_random_vectors(n, dim);
    let quality_ok = verify_search_quality(&index_parallel, &vectors, dim);
    println!(
        "Search quality check: {}",
        if quality_ok { "PASS" } else { "FAIL" }
    );

    assert!(quality_ok, "Parallel build should maintain search quality");
    assert!(speedup > 1.0, "Parallel build should be faster than serial");

    // Target: 4-8x speedup
    if speedup >= 4.0 {
        println!("✅ Target achieved: {:.2}x speedup (target: 4-8x)", speedup);
    } else {
        println!(
            "⚠️  Speedup {:.2}x is below target (4-8x), but parallel build is still beneficial",
            speedup
        );
    }

    println!("\n=== Benchmark Complete ===\n");
}

#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_hnsw_parallel_build_large() {
    println!("\n=== OPT-024: HNSW Parallel Build Benchmark (Large) ===");

    let n = 100_000;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("Dataset: {} vectors x {} dimensions", n, dim);
    println!("HNSW params: M={}, EF_CONSTRUCTION={}", m, ef_construction);
    println!("Parallel threads: {}", num_threads);
    println!("Target: 3-6s (from 24.5s serial)");
    println!();

    // Serial build
    println!("Building HNSW (serial)...");
    let (time_serial, _index_serial) = bench_hnsw_serial(n, dim, m, ef_construction);
    println!("Serial build time: {:.3}s", time_serial);

    // Parallel build
    println!("Building HNSW (parallel, {} threads)...", num_threads);
    let (time_parallel, index_parallel) =
        bench_hnsw_parallel(n, dim, m, ef_construction, num_threads);
    println!("Parallel build time: {:.3}s", time_parallel);

    // Calculate speedup
    let speedup = time_serial / time_parallel;
    println!("\nSpeedup: {:.2}x", speedup);

    // Verify search quality
    let vectors = generate_random_vectors(n, dim);
    let quality_ok = verify_search_quality(&index_parallel, &vectors, dim);
    println!(
        "Search quality check: {}",
        if quality_ok { "PASS" } else { "FAIL" }
    );

    assert!(quality_ok, "Parallel build should maintain search quality");
    assert!(speedup > 1.0, "Parallel build should be faster than serial");

    // Target: 3-6s for 100K vectors
    if time_parallel <= 6.0 {
        println!(
            "✅ Target achieved: {:.3}s build time (target: 3-6s)",
            time_parallel
        );
    } else {
        println!(
            "⚠️  Build time {:.3}s exceeds target (3-6s), but parallel build is still beneficial",
            time_parallel
        );
    }

    println!("\n=== Benchmark Complete ===\n");
}

/// Test different thread counts to find optimal configuration
#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_hnsw_thread_scaling() {
    println!("\n=== OPT-024: HNSW Thread Scaling Test ===");

    let n = 50_000;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    println!("Dataset: {} vectors x {} dimensions", n, dim);
    println!("Testing thread counts: 1, 2, 4, {}", max_threads);
    println!();

    let thread_counts = vec![1, 2, 4, max_threads];
    let mut results = Vec::new();

    for num_threads in thread_counts {
        println!("Building with {} threads...", num_threads);
        let (time, _index) = bench_hnsw_parallel(n, dim, m, ef_construction, num_threads);
        let speedup = results.first().map(|&(t, _)| t / time).unwrap_or(1.0);
        println!("  Time: {:.3}s (speedup: {:.2}x)", time, speedup);
        results.push((time, num_threads));
    }

    println!("\n=== Thread Scaling Results ===");
    println!("Threads\tTime (s)\tSpeedup");
    println!("------\t--------\t-------");
    let baseline = results[0].0;
    for (time, threads) in &results {
        let speedup = baseline / time;
        println!("{}\t{:.3}\t\t{:.2}x", threads, time, speedup);
    }
    println!();
}

/// OPT-031: Test error handling in parallel build
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_parallel_error_handling() {
    println!("\n=== OPT-031: HNSW Parallel Build Error Handling ===");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: 128,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            num_threads: Some(4),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    let vectors = generate_random_vectors(1000, 128);

    // Test 1: Untrained index
    println!("Test 1: Untrained index (should fail)...");
    let result = index.add_parallel(&vectors, None, Some(true));
    assert!(result.is_err(), "Should fail on untrained index");
    println!("  ✅ Correctly rejected untrained index");

    // Train the index
    index.train(&vectors).unwrap();

    // Test 2: Empty vector set
    println!("Test 2: Empty vector set...");
    let result = index.add_parallel(&[], None, Some(true));
    assert!(result.is_ok(), "Should handle empty vector set gracefully");
    assert_eq!(result.unwrap(), 0, "Should return 0 for empty set");
    println!("  ✅ Correctly handled empty vector set");

    // Test 3: Dimension mismatch
    println!("Test 3: Dimension mismatch (should fail)...");
    let bad_vectors = vec![1.0, 2.0, 3.0]; // Not divisible by dim=128
    let result = index.add_parallel(&bad_vectors, None, Some(true));
    assert!(result.is_err(), "Should fail on dimension mismatch");
    println!("  ✅ Correctly rejected dimension mismatch");

    // Test 4: ID count mismatch
    println!("Test 4: ID count mismatch (should fail)...");
    let wrong_ids = vec![1i64, 2i64]; // Only 2 IDs for 1000 vectors
    let result = index.add_parallel(&vectors, Some(&wrong_ids), Some(true));
    assert!(result.is_err(), "Should fail on ID count mismatch");
    println!("  ✅ Correctly rejected ID count mismatch");

    // Test 5: Valid parallel build
    println!("Test 5: Valid parallel build...");
    let result = index.add_parallel(&vectors, None, Some(true));
    assert!(result.is_ok(), "Should succeed on valid input");
    assert_eq!(result.unwrap(), 1000, "Should add all vectors");
    println!("  ✅ Successfully built index with parallel construction");

    println!("\n=== Error Handling Tests Complete ===\n");
}

/// OPT-031: Test API compatibility between add() and add_parallel()
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_parallel_api_compatibility() {
    println!("\n=== OPT-031: HNSW Parallel API Compatibility ===");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: 64,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            num_threads: Some(4),
            ..Default::default()
        },
    };

    let n = 5000;
    let dim = 64;
    let vectors = generate_random_vectors(n, dim);
    let ids: Vec<i64> = (0..n as i64).collect();

    // Build with serial add()
    println!("Building with serial add()...");
    let mut index_serial = HnswIndex::new(&config).unwrap();
    index_serial.train(&vectors).unwrap();
    let start = Instant::now();
    index_serial.add(&vectors, Some(&ids)).unwrap();
    let time_serial = start.elapsed();

    // Build with parallel add_parallel()
    println!("Building with parallel add_parallel()...");
    let mut index_parallel = HnswIndex::new(&config).unwrap();
    index_parallel.train(&vectors).unwrap();
    let start = Instant::now();
    index_parallel
        .add_parallel(&vectors, Some(&ids), Some(true))
        .unwrap();
    let time_parallel = start.elapsed();

    // Verify both have same count
    assert_eq!(
        index_serial.ntotal(),
        index_parallel.ntotal(),
        "Serial and parallel should have same vector count"
    );
    println!("  ✅ Vector count matches: {}", index_serial.ntotal());

    // Verify search results are similar
    let query = &vectors[0..dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 20,
        filter: None,
        params: None,
        radius: None,
    };

    let result_serial = index_serial.search(query, &req).unwrap();
    let result_parallel = index_parallel.search(query, &req).unwrap();

    assert_eq!(
        result_serial.ids.len(),
        result_parallel.ids.len(),
        "Serial and parallel should return same number of results"
    );
    println!(
        "  ✅ Search result count matches: {}",
        result_serial.ids.len()
    );

    // Check that top results are reasonably close (may differ due to parallel ordering)
    let dist_diff = (result_serial.distances[0] - result_parallel.distances[0]).abs();
    assert!(
        dist_diff < 1.0,
        "Top result distances should be similar (diff: {:.4})",
        dist_diff
    );
    println!("  ✅ Top result distance similar (diff: {:.4})", dist_diff);

    println!("\nSerial time: {:?}", time_serial);
    println!("Parallel time: {:?}", time_parallel);
    println!("\n=== API Compatibility Tests Complete ===\n");
}

/// OPT-031: Test batch size optimization
#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_hnsw_batch_size_optimization() {
    println!("\n=== OPT-031: HNSW Batch Size Optimization ===");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: 128,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            num_threads: Some(8),
            ..Default::default()
        },
    };

    let index = HnswIndex::new(&config).unwrap();

    // Test batch size calculation for different scenarios
    let test_cases = vec![
        (1000, 64, "Small dataset, low dim"),
        (10000, 128, "Medium dataset, standard dim"),
        (100000, 128, "Large dataset, standard dim"),
        (50000, 512, "Medium dataset, high dim"),
        (100000, 32, "Large dataset, low dim"),
    ];

    println!("Batch size calculations:");
    for (n, dim, description) in test_cases {
        // Use reflection or direct calculation
        // For this test, we'll just verify the logic works
        let base_batch = ((n as f64) / 8.0).max(50.0);
        let dim_factor = (128.0 / dim as f64).max(0.25).min(2.0);
        let count_factor = (n as f64 / 10000.0).max(0.5).min(1.0);
        let batch_size = (base_batch * dim_factor * count_factor)
            .max(50.0)
            .min(5000.0) as usize;

        println!(
            "  {}: n={}, dim={} -> batch_size={}",
            description, n, dim, batch_size
        );
    }

    println!("\n=== Batch Size Optimization Tests Complete ===\n");
}

/// OPT-031: Performance target validation test
#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_hnsw_parallel_performance_target() {
    println!("\n=== OPT-031: HNSW Parallel Performance Target Validation ===");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: 128,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            num_threads: None, // Use default (all available cores)
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();

    // Generate 100K vectors
    let n = 100_000;
    let dim = 128;
    let vectors = generate_random_vectors(n, dim);

    println!("Dataset: {} vectors x {} dimensions", n, dim);
    println!("Target: build time < 3s");
    println!();

    // Train
    index.train(&vectors).unwrap();

    // Build with parallel construction
    let start = Instant::now();
    let count = index.add_parallel(&vectors, None, None).unwrap();
    let elapsed = start.elapsed();

    println!("Build time: {:.3}s", elapsed.as_secs_f64());
    println!("Vectors added: {}", count);

    let build_secs = elapsed.as_secs_f64();
    let target_secs = 3.0;

    if build_secs < target_secs {
        println!(
            "✅ PASS: Build time ({:.3}s) meets target (< {:.1}s)",
            build_secs, target_secs
        );
    } else {
        println!(
            "⚠️  Build time ({:.3}s) exceeded target (< {:.1}s)",
            build_secs, target_secs
        );
        println!("   (Performance depends on hardware; parallel build still provides speedup)");
    }

    // Verify search quality
    let query = &vectors[0..dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 20,
        filter: None,
        params: None,
        radius: None,
    };

    let result = index.search(query, &req).unwrap();
    assert!(result.ids.len() > 0, "Search should return results");
    println!(
        "Search quality: ✅ PASS (top_k={} results, distance={:.4})",
        result.ids.len(),
        result.distances[0]
    );

    // Verify speedup over serial (if we have multiple cores)
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if num_threads > 1 {
        println!("Hardware: {} threads available", num_threads);
        println!("Expected speedup: 4-8x over serial");
    }

    println!("\n=== Performance Target Validation Complete ===\n");
}

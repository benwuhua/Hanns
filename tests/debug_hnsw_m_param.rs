#![cfg(feature = "long-tests")]
//! Debug HNSW M parameter issue
//!
//! Tests different M values (8, 16, 32, 64) with fixed ef_construction and ef_search
//! to isolate the effect of M on recall rate.
//!
//! # Usage
//! ```bash
//! cargo test --release --test debug_hnsw_m_param -- --nocapture
//! ```

use knowhere_rs::api::{IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::IndexType;
use knowhere_rs::MetricType;
use rand::Rng;
use std::collections::HashSet;
use std::time::Instant;

/// Generate random dataset with Gaussian distribution
fn generate_gaussian_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        let u1 = rng.gen::<f32>();
        let u2 = rng.gen::<f32>();
        let gaussian = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        data.push(gaussian);
    }

    data
}

/// Compute ground truth using brute-force
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

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Calculate recall@k
fn calculate_recall(results: &[Vec<i64>], ground_truth: &[Vec<i32>], k: usize) -> f64 {
    let mut total_recall = 0.0;

    for (i, result) in results.iter().enumerate() {
        let gt_set: HashSet<i32> = ground_truth[i].iter().copied().collect();
        let matches: usize = result
            .iter()
            .take(k)
            .filter(|&id| gt_set.contains(&(*id as i32)))
            .count();
        total_recall += matches as f64 / k.min(ground_truth[i].len()) as f64;
    }

    total_recall / results.len() as f64
}

/// Test result for a specific M value
struct MTestResult {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    build_time_ms: f64,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    max_level: usize,
    avg_level: f64,
}

/// Benchmark HNSW with specific M value
fn benchmark_hnsw_m(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
    num_queries: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> MTestResult {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()), // Optimal level multiplier
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Collect statistics about the graph
    let max_level = index.max_level();
    let avg_level = index.average_node_level();

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: ef_search,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);

    let recall_at_1 = calculate_recall(&all_results, ground_truth, 1);
    let recall_at_10 = calculate_recall(&all_results, ground_truth, 10);
    let recall_at_100 = calculate_recall(&all_results, ground_truth, 100);

    MTestResult {
        m,
        ef_construction,
        ef_search,
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
        max_level,
        avg_level,
    }
}

#[test]
fn test_debug_m_parameter() {
    println!("\n{}", "=".repeat(100));
    println!("🔍 BUG-001: Debug HNSW M Parameter Issue");
    println!("{}", "=".repeat(100));

    const NUM_VECTORS: usize = 10_000;
    const NUM_QUERIES: usize = 20;
    const DIM: usize = 128;
    const K: usize = 100;

    // Fixed parameters for fair comparison
    const EF_CONSTRUCTION: usize = 200;
    const EF_SEARCH: usize = 256;

    // Test different M values
    let m_values = vec![8, 16, 32, 64];

    println!("\n📋 Test Configuration:");
    println!("  - Dataset: {} vectors x {} dimensions", NUM_VECTORS, DIM);
    println!("  - Queries: {}", NUM_QUERIES);
    println!("  - Fixed ef_construction: {}", EF_CONSTRUCTION);
    println!("  - Fixed ef_search: {}", EF_SEARCH);
    println!("  - M values to test: {:?}", m_values);

    // Generate dataset
    println!("\n📊 Generating Gaussian dataset...");
    let base = generate_gaussian_dataset(NUM_VECTORS, DIM);
    let query = generate_gaussian_dataset(NUM_QUERIES, DIM);

    // Compute ground truth
    println!("🎯 Computing ground truth (brute-force)...");
    let gt_start = Instant::now();
    let ground_truth = compute_ground_truth(&base, &query, NUM_QUERIES, DIM, K);
    let gt_time = gt_start.elapsed().as_secs_f64();
    println!("  Ground truth computed in {:.2}s", gt_time);

    // Test each M value
    println!("\n🔬 Testing different M values...");
    let mut results = Vec::new();

    for &m in &m_values {
        println!("\n  Testing M = {}...", m);
        let result = benchmark_hnsw_m(
            &base,
            &query,
            &ground_truth,
            NUM_QUERIES,
            DIM,
            m,
            EF_CONSTRUCTION,
            EF_SEARCH,
        );
        println!(
            "    Build: {:.2}ms, Search: {:.2}ms, QPS: {:.0}",
            result.build_time_ms, result.search_time_ms, result.qps
        );
        println!(
            "    R@1: {:.3}, R@10: {:.3}, R@100: {:.3}",
            result.recall_at_1, result.recall_at_10, result.recall_at_100
        );
        println!(
            "    Max level: {}, Avg level: {:.2}",
            result.max_level, result.avg_level
        );
        results.push(result);
    }

    // Print summary table
    println!("\n{}", "=".repeat(120));
    println!("📊 M Parameter Sensitivity Results");
    println!("{}", "=".repeat(120));
    println!(
        "| {:>4} | {:>12} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} | {:>10} |",
        "M",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "MaxLvl",
        "AvgLevel",
        "Config"
    );
    println!("{}", "=".repeat(120));

    for r in &results {
        let config_str = format!("M{}-efC{}-efS{}", r.m, r.ef_construction, r.ef_search);
        println!(
            "| {:>4} | {:>12.2} | {:>10.2} | {:>8.0} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8} | {:>10.2} | {:>10} |",
            r.m, r.build_time_ms, r.search_time_ms, r.qps,
            r.recall_at_1, r.recall_at_10, r.recall_at_100,
            r.max_level, r.avg_level, config_str
        );
    }
    println!("{}", "=".repeat(120));

    // Analysis
    println!("\n📈 Analysis:");

    // Check if recall decreases with higher M
    let mut recall_decreasing = false;
    let mut prev_recall = results[0].recall_at_10;

    for i in 1..results.len() {
        if results[i].recall_at_10 < prev_recall - 0.01 {
            // Allow 1% tolerance
            recall_decreasing = true;
            println!(
                "  ⚠️  WARNING: R@10 decreases from M={} ({:.3}) to M={} ({:.3})",
                results[i - 1].m,
                prev_recall,
                results[i].m,
                results[i].recall_at_10
            );
        }
        prev_recall = results[i].recall_at_10;
    }

    if !recall_decreasing {
        println!("  ✅ R@10 generally increases or stays stable with higher M (expected behavior)");
    }

    // Check level distribution
    println!("\n  Level distribution analysis:");
    for r in &results {
        println!(
            "    M={}: max_level={}, avg_level={:.2}",
            r.m, r.max_level, r.avg_level
        );
    }

    // Find best M
    let best_m_result = results
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
        .unwrap();
    println!(
        "\n  🏆 Best M for R@10: {} (R@10={:.3})",
        best_m_result.m, best_m_result.recall_at_10
    );

    println!("\n{}", "=".repeat(100));
    println!("✅ Debug test complete!");
    println!("{}", "=".repeat(100));
}

#[test]
fn test_debug_m_with_varying_ef() {
    println!("\n{}", "=".repeat(100));
    println!("🔍 BUG-001: M Parameter with Varying ef_search");
    println!("{}", "=".repeat(100));

    const NUM_VECTORS: usize = 5_000;
    const NUM_QUERIES: usize = 10;
    const DIM: usize = 128;
    const K: usize = 50;

    const EF_CONSTRUCTION: usize = 200;

    // Test combinations of M and ef_search
    let m_values = vec![8, 16, 32, 64];
    let ef_search_values = vec![32, 64, 128, 256];

    println!("\n📋 Test Configuration:");
    println!("  - Dataset: {} vectors x {} dimensions", NUM_VECTORS, DIM);
    println!("  - Queries: {}", NUM_QUERIES);
    println!("  - Fixed ef_construction: {}", EF_CONSTRUCTION);
    println!("  - M values: {:?}", m_values);
    println!("  - ef_search values: {:?}", ef_search_values);

    // Generate dataset
    let base = generate_gaussian_dataset(NUM_VECTORS, DIM);
    let query = generate_gaussian_dataset(NUM_QUERIES, DIM);

    // Compute ground truth
    let ground_truth = compute_ground_truth(&base, &query, NUM_QUERIES, DIM, K);

    // Test all combinations
    let mut results: Vec<(usize, usize, f64)> = Vec::new(); // (m, ef_search, recall_at_10)

    for &m in &m_values {
        for &ef in &ef_search_values {
            let result = benchmark_hnsw_m(
                &base,
                &query,
                &ground_truth,
                NUM_QUERIES,
                DIM,
                m,
                EF_CONSTRUCTION,
                ef,
            );
            println!(
                "  M={}, ef_search={} -> R@10={:.3}",
                m, ef, result.recall_at_10
            );
            results.push((m, ef, result.recall_at_10));
        }
    }

    // Print as table
    println!("\n📊 Recall@10 Matrix:");
    print!("{:>8}", "M\\ef");
    for &ef in &ef_search_values {
        print!("{:>10}", ef);
    }
    println!();
    print!("{:-<8}", "");
    for _ in &ef_search_values {
        print!("{:-<10}", "");
    }
    println!();

    for &m in &m_values {
        print!("{:>8}", m);
        for &ef in &ef_search_values {
            let recall = results
                .iter()
                .find(|(mm, ee, _)| *mm == m && *ee == ef)
                .unwrap()
                .2;
            print!("{:>10.3}", recall);
        }
        println!();
    }

    println!("\n{}", "=".repeat(100));
}

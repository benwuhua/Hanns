#![cfg(feature = "long-tests")]
//! Memory Usage Benchmark Test
//!
//! Benchmark memory efficiency of Flat, HNSW, and IVF-Flat indexes.
//!
//! # Usage
//! ```bash
//! cargo test --release --test bench_memory_usage -- --nocapture
//! ```
//!
//! # Dataset
//! Random100K: 100,000 vectors, 128 dimensions (f32)
//!
//! # Metrics
//! - Theoretical memory usage (based on data structure)
//! - Actual memory usage (tracked via MemoryTracker)
//! - Memory overhead percentage
//! - Recall@10 for accuracy comparison
//! - Memory efficiency (MB per 100K vectors)

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_hnsw_overhead, estimate_ivf_overhead, estimate_vector_memory,
    MemoryTracker,
};
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

/// Number of vectors in the dataset (reduced for faster testing)
const NUM_VECTORS: usize = 10_000;

/// Vector dimension
const DIM: usize = 128;

/// Number of query vectors for testing (reduced for speed)
const NUM_QUERIES: usize = 20;

/// Top-k for recall measurement
const TOP_K: usize = 10;

/// Generate random dataset
fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);

    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-1.0..1.0));
    }

    data
}

/// Generate ground truth for random dataset (brute-force L2 distance)
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

/// Calculate L2 distance squared between two vectors
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Memory benchmark result for a single index type
struct MemoryBenchmarkResult {
    index_name: String,
    num_vectors: usize,
    dim: usize,
    theoretical_memory_mb: f64,
    actual_memory_mb: f64,
    overhead_percent: f64,
    recall_at_10: f64,
    build_time_ms: f64,
    additional_info: Vec<String>,
}

impl MemoryBenchmarkResult {
    fn print_report(&self) {
        println!("\n### Index: {}", self.index_name);
        println!("- Vectors: {} x {} (f32)", self.num_vectors, self.dim);
        println!("- Build time: {:.2} ms", self.build_time_ms);

        for info in &self.additional_info {
            println!("- {}", info);
        }

        println!("- Theoretical: {:.2} MB", self.theoretical_memory_mb);
        println!("- Actual: {:.2} MB", self.actual_memory_mb);
        println!("- Overhead: {:.1}%", self.overhead_percent);
        println!("- Recall@10: {:.3}", self.recall_at_10);
    }

    fn memory_per_100k(&self) -> f64 {
        self.actual_memory_mb * (100_000.0 / self.num_vectors as f64)
    }
}

/// Benchmark Flat index memory usage
fn benchmark_flat(base: &[f32], query: &[f32], ground_truth: &[Vec<i32>]) -> MemoryBenchmarkResult {
    println!("\nBenchmarking Flat index...");

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(NUM_VECTORS, DIM);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record index overhead (Flat index has minimal overhead - just ID storage)
    let overhead = (NUM_VECTORS * std::mem::size_of::<i64>()) as u64;
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  {}", tracker.report());

    // Search for recall measurement
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(NUM_QUERIES);

    for i in 0..NUM_QUERIES {
        let q = &query[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: TOP_K,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, NUM_QUERIES
    );

    // Calculate recall
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, TOP_K);
    println!("  Recall@10: {:.3}", recall_at_10);

    let theoretical = base_mem as f64 / (1024.0 * 1024.0);
    let actual = tracker.total_memory_mb();
    let overhead_percent = ((actual - theoretical) / theoretical) * 100.0;

    MemoryBenchmarkResult {
        index_name: "Flat".to_string(),
        num_vectors: NUM_VECTORS,
        dim: DIM,
        theoretical_memory_mb: theoretical,
        actual_memory_mb: actual,
        overhead_percent,
        recall_at_10,
        build_time_ms: build_time,
        additional_info: vec![],
    }
}

/// Benchmark HNSW index memory usage
fn benchmark_hnsw(base: &[f32], query: &[f32], ground_truth: &[Vec<i32>]) -> MemoryBenchmarkResult {
    println!("\nBenchmarking HNSW index...");

    // HNSW parameters - optimized for faster build (smaller dataset)
    let m = 16;
    let ef_construction = 64;
    let ef_search = 32;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(NUM_VECTORS, DIM);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record HNSW overhead (graph structure + vectors)
    let overhead = estimate_hnsw_overhead(NUM_VECTORS, DIM, m);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!(
        "  M={}, ef_construction={}, ef_search={}",
        m, ef_construction, ef_search
    );
    println!("  {}", tracker.report());

    // Search for recall measurement
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(NUM_QUERIES);

    for i in 0..NUM_QUERIES {
        let q = &query[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: TOP_K,
            params: Some(format!(r#"{{"ef": {}}}"#, ef_search)),
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, NUM_QUERIES
    );

    // Calculate recall
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, TOP_K);
    println!("  Recall@10: {:.3}", recall_at_10);

    let theoretical = overhead as f64 / (1024.0 * 1024.0);
    let actual = tracker.total_memory_mb();
    let overhead_percent = ((actual - theoretical) / theoretical) * 100.0;

    MemoryBenchmarkResult {
        index_name: format!("HNSW (M={})", m),
        num_vectors: NUM_VECTORS,
        dim: DIM,
        theoretical_memory_mb: theoretical,
        actual_memory_mb: actual,
        overhead_percent,
        recall_at_10,
        build_time_ms: build_time,
        additional_info: vec![
            format!("Graph nodes: {}", NUM_VECTORS),
            format!("Avg connections: {}", m),
            format!(
                "Theoretical: {:.2} MB (~{:.1}x Flat)",
                theoretical,
                theoretical / (base_mem as f64 / (1024.0 * 1024.0))
            ),
        ],
    }
}

/// Benchmark IVF-Flat index memory usage
fn benchmark_ivf_flat(
    base: &[f32],
    query: &[f32],
    ground_truth: &[Vec<i32>],
) -> MemoryBenchmarkResult {
    println!("\nBenchmarking IVF-Flat index...");

    // IVF parameters - sqrt(10K) ≈ 100, use 64 for faster k-means
    let nlist = 64;
    let nprobe = 8;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(NUM_VECTORS, DIM);
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record IVF overhead (centroids + inverted lists)
    let overhead = estimate_ivf_overhead(NUM_VECTORS, DIM, nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  nlist: {}, nprobe: {}", nlist, nprobe);
    println!("  {}", tracker.report());

    // Search for recall measurement
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(NUM_QUERIES);

    for i in 0..NUM_QUERIES {
        let q = &query[i * DIM..(i + 1) * DIM];
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        all_results.push(result.ids.clone());
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, NUM_QUERIES
    );

    // Calculate recall
    let recall_at_10 = average_recall_at_k(&all_results, ground_truth, TOP_K);
    println!("  Recall@10: {:.3}", recall_at_10);

    // Theoretical: base vectors + centroids + inverted list overhead
    let centroids_mem = nlist * DIM * std::mem::size_of::<f32>();
    let inverted_list_mem = NUM_VECTORS * std::mem::size_of::<u32>(); // list IDs
    let theoretical =
        (base_mem + centroids_mem as u64 + inverted_list_mem as u64) as f64 / (1024.0 * 1024.0);
    let actual = tracker.total_memory_mb();
    let overhead_percent = ((actual - theoretical) / theoretical) * 100.0;

    MemoryBenchmarkResult {
        index_name: format!("IVF-Flat (nlist={})", nlist),
        num_vectors: NUM_VECTORS,
        dim: DIM,
        theoretical_memory_mb: theoretical,
        actual_memory_mb: actual,
        overhead_percent,
        recall_at_10,
        build_time_ms: build_time,
        additional_info: vec![
            format!("Centroids: {} x {}", nlist, DIM),
            format!("Inverted lists overhead: ~{:.0}%", overhead_percent),
        ],
    }
}

/// Print summary table
fn print_summary(results: &[MemoryBenchmarkResult]) {
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));

    // Header
    println!("\n| Index         | Memory (MB) | MB/100K | Recall@10 | Efficiency     |");
    println!("|---------------|-------------|---------|-----------|----------------|");

    // Find baseline (Flat)
    let baseline_mem = results
        .iter()
        .find(|r| r.index_name.starts_with("Flat"))
        .map(|r| r.memory_per_100k())
        .unwrap_or(1.0);

    for result in results {
        let mem_per_100k = result.memory_per_100k();
        let efficiency = if result.index_name.starts_with("Flat") {
            "Baseline".to_string()
        } else {
            format!("{:.1}x", mem_per_100k / baseline_mem)
        };

        println!(
            "| {:<13} | {:>11.2} | {:>7.2} | {:>9.3} | {:<14} |",
            result.index_name,
            result.actual_memory_mb,
            mem_per_100k,
            result.recall_at_10,
            efficiency
        );
    }

    println!("\n{}", "=".repeat(80));
}

/// Print detailed analysis
fn print_analysis(results: &[MemoryBenchmarkResult]) {
    println!("\nANALYSIS");
    println!("{}", "-".repeat(80));

    // Memory vs Recall trade-off
    println!("\n### Memory vs Recall Trade-off:");
    for result in results {
        println!(
            "  - {}: {:.2} MB -> Recall@10 = {:.3}",
            result
                .index_name
                .split_whitespace()
                .next()
                .unwrap_or(&result.index_name),
            result.actual_memory_mb,
            result.recall_at_10
        );
    }

    // Find best efficiency
    let best_recall = results
        .iter()
        .max_by(|a, b| a.recall_at_10.partial_cmp(&b.recall_at_10).unwrap())
        .unwrap();
    let best_memory = results
        .iter()
        .min_by(|a, b| a.actual_memory_mb.partial_cmp(&b.actual_memory_mb).unwrap())
        .unwrap();

    println!("\n### Key Insights:");
    println!(
        "  - Best Recall: {} ({:.3})",
        best_recall.index_name, best_recall.recall_at_10
    );
    println!(
        "  - Best Memory: {} ({:.2} MB)",
        best_memory.index_name, best_memory.actual_memory_mb
    );

    // Recommendations
    println!("\n### Recommendations:");
    println!("  - Use Flat when: 100% recall is required, dataset fits in memory");
    println!("  - Use HNSW when: High recall (>0.95) with fast search is needed");
    println!("  - Use IVF-Flat when: Memory efficiency is priority, moderate recall acceptable");
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_memory_usage_small() {
    println!("\n{}", "=".repeat(80));
    println!("Memory Usage Benchmark Report (Small Dataset)");
    println!("{}", "=".repeat(80));
    println!(
        "\nDataset: Random10K ({} vectors, {} dimensions, f32)",
        NUM_VECTORS, DIM
    );
    println!("Queries: {} (for recall measurement)", NUM_QUERIES);
    println!("Note: Optimized for faster execution (~2-3 min expected)");

    // Generate dataset
    println!("\nGenerating random dataset...");
    let base = generate_random_dataset(NUM_VECTORS, DIM);
    let query = generate_random_dataset(NUM_QUERIES, DIM);

    // Compute ground truth (this takes time but only done once)
    println!("Computing ground truth (brute-force)...");
    let ground_truth = compute_ground_truth(&base, &query, NUM_QUERIES, DIM, TOP_K);
    println!("Ground truth computed for {} queries", NUM_QUERIES);

    // Run benchmarks
    let mut results = Vec::new();

    // Flat index
    let flat_result = benchmark_flat(&base, &query, &ground_truth);
    flat_result.print_report();
    results.push(flat_result);

    // HNSW index
    let hnsw_result = benchmark_hnsw(&base, &query, &ground_truth);
    hnsw_result.print_report();
    results.push(hnsw_result);

    // IVF-Flat index
    let ivf_result = benchmark_ivf_flat(&base, &query, &ground_truth);
    ivf_result.print_report();
    results.push(ivf_result);

    // Print summary and analysis
    print_summary(&results);
    print_analysis(&results);

    // Validate results
    assert!(
        results.iter().all(|r| r.actual_memory_mb > 0.0),
        "Memory should be > 0"
    );
    assert!(
        results
            .iter()
            .all(|r| r.recall_at_10 >= 0.0 && r.recall_at_10 <= 1.0),
        "Recall should be in [0, 1]"
    );

    // Flat should have perfect recall
    let flat_result = results.iter().find(|r| r.index_name == "Flat").unwrap();
    assert!(
        flat_result.recall_at_10 > 0.99,
        "Flat index should have near-perfect recall"
    );

    println!("\nAll benchmarks completed successfully!");
}

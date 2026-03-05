//! SIFT1M Benchmark Test
//!
//! Benchmark Flat, HNSW, and IVF-Flat indexes on SIFT1M dataset.
//!
//! # Usage
//! ```bash
//! cargo test --test bench_sift1m -- --nocapture
//! ```
//!
//! # Dataset
//! Download SIFT1M from: http://corpus-texmex.irisa.fr/
//! - base.fvecs: 1M base vectors (128D)
//! - query.fvecs: 10K query vectors (128D)
//! - groundtruth.ivecs: 10K x 100 ground truth neighbors

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::{
    average_recall_at_k, estimate_hnsw_overhead, estimate_ivf_overhead, estimate_vector_memory,
    BenchmarkResult, DistanceValidationReport, MemoryTracker,
};
use knowhere_rs::dataset::{load_sift1m_complete, SiftDataset};
use knowhere_rs::faiss::{
    HnswIndex, IvfFlatIndex, IvfPqIndex, IvfSq8Index, MemIndex as FlatIndex, ScaNNConfig,
    ScaNNIndex,
};
use knowhere_rs::MetricType;
use knowhere_rs::{IvfRaBitqConfig, IvfRaBitqIndex};
use std::env;
use std::time::Instant;

/// Get SIFT1M dataset path from environment or use default
fn get_sift_path() -> String {
    env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string())
}

/// Load or return None if dataset not found
fn try_load_sift1m() -> Option<SiftDataset> {
    let path = get_sift_path();
    match load_sift1m_complete(&path) {
        Ok(dataset) => {
            println!("Loaded SIFT1M dataset from {}", path);
            println!("  Base vectors: {}", dataset.num_base());
            println!("  Query vectors: {}", dataset.num_query());
            println!("  Dimension: {}", dataset.dim());
            Some(dataset)
        }
        Err(e) => {
            eprintln!("Failed to load SIFT1M dataset from {}: {}", path, e);
            eprintln!("Set SIFT1M_PATH environment variable or place dataset in ./data/sift/");
            None
        }
    }
}

/// Benchmark Flat index
fn benchmark_flat(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking Flat index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        params: IndexParams::default(),
    };

    // Build index
    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record index overhead (Flat index has minimal overhead)
    let overhead = (dataset.num_base() * 4) as u64; // Just ID storage
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 40,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall - only use ground_truth for num_queries queries
    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark HNSW index
fn benchmark_hnsw(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking HNSW index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    // BENCH-036: Use optimized parameters from OPT-021/OPT-029
    // M=32, ef_construction=400, ef_search=400 → R@10: 95%
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(32),
            ef_construction: Some(400),
            ef_search: Some(400),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record HNSW overhead
    let overhead = estimate_hnsw_overhead(dataset.num_base(), dataset.dim(), 16);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        // BENCH-036: Use ef_search=400 for optimal recall
        let req = SearchRequest {
            top_k: 100,
            nprobe: 400,
            params: Some(r#"{"ef": 400}"#.to_string()),
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall
    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "HNSW".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark IVF-Flat index
fn benchmark_ivf_flat(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking IVF-Flat index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 10;

    // Initialize memory tracker
    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    // Build index
    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    // Record IVF overhead
    let overhead = estimate_ivf_overhead(dataset.num_base(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  nlist: {}, nprobe: {}", nlist, nprobe);
    println!("{}", tracker.report());

    // Search
    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    // Distance validation
    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,     // metric_l2
        0.0,      // low_bound
        f32::MAX, // high_bound
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    // Calculate recall
    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "IVF-Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark IVF-PQ index
fn benchmark_ivf_pq(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking IVF-PQ index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 32;
    let pq_m = 16; // 128 / 16 = 8 bytes per subvector
    let nbits = 8; // 256 centroids per subspace

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            m: Some(pq_m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = IvfPqIndex::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = estimate_ivf_overhead(dataset.num_base(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!(
        "  nlist: {}, nprobe: {}, pq_m: {}, nbits: {}",
        nlist, nprobe, pq_m, nbits
    );
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,
        0.0,
        f32::MAX,
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "IVF-PQ".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark IVF-SQ8 index
fn benchmark_ivf_sq8(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking IVF-SQ8 index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 32;

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = IvfSq8Index::new(&config).unwrap();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = estimate_ivf_overhead(dataset.num_base(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  nlist: {}, nprobe: {}", nlist, nprobe);
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,
        0.0,
        f32::MAX,
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "IVF-SQ8".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark RaBitQ index
fn benchmark_rabitq(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking RaBitQ index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let nlist = ((dataset.num_base() as f64).sqrt() as i32).max(1) as usize;
    let nprobe = 32;

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = IvfRaBitqConfig::new(dataset.dim(), nlist);
    let mut index = IvfRaBitqIndex::new(config);

    let build_start = Instant::now();
    index.train(base_vectors).unwrap();
    index.add(base_vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let overhead = estimate_ivf_overhead(dataset.num_base(), dataset.dim(), nlist);
    tracker.record_index_overhead(overhead);

    println!("  Build time: {:.2} ms", build_time);
    println!("  nlist: {}, nprobe: {}", nlist, nprobe);
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            nprobe,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(&result.distances);
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,
        0.0,
        f32::MAX,
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "RaBitQ".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

/// Benchmark ScaNN index
fn benchmark_scann(dataset: &SiftDataset, num_queries: usize) -> BenchmarkResult {
    println!("\nBenchmarking ScaNN index...");

    let base_vectors = dataset.base.vectors();
    let query_vectors = dataset.query.vectors();
    let ground_truth = &dataset.ground_truth;

    let num_partitions = 16; // 128 / 16 = 8 bytes per subvector
    let num_centroids = 256;
    let reorder_k = 100;

    let tracker = MemoryTracker::new();
    let base_mem = estimate_vector_memory(dataset.num_base(), dataset.dim());
    tracker.record_base_memory(base_mem);

    let config = ScaNNConfig::new(num_partitions, num_centroids, reorder_k);
    let mut index = ScaNNIndex::new(dataset.dim(), config).unwrap();

    let build_start = Instant::now();
    index.train(base_vectors, Some(query_vectors));
    let added = index.add(base_vectors, None);
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  Build time: {:.2} ms (added {} vectors)",
        build_time, added
    );
    println!(
        "  num_partitions: {}, num_centroids: {}, reorder_k: {}",
        num_partitions, num_centroids, reorder_k
    );
    println!("{}", tracker.report());

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    let mut all_distances: Vec<f32> = Vec::with_capacity(num_queries * 100);

    for i in 0..num_queries {
        let query = &query_vectors[i * dataset.dim()..(i + 1) * dataset.dim()];
        let result = index.search(query, 100); // ScaNN search API: (query, k)
        all_results.push(result.iter().map(|(id, _)| *id).collect());
        all_distances.extend(result.iter().map(|(_, dist)| *dist));
    }

    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
    let qps = num_queries as f64 / (search_time / 1000.0);
    println!(
        "  Search time: {:.2} ms ({} queries)",
        search_time, num_queries
    );
    println!("  QPS: {:.0}", qps);

    let report = DistanceValidationReport::validate_knn(
        &all_distances,
        num_queries,
        100,
        true,
        0.0,
        f32::MAX,
    );
    report.print();
    assert!(report.all_passed(), "Distance validation failed");

    let gt_subset: Vec<_> = ground_truth.iter().take(num_queries).cloned().collect();
    let recall_at_1 = average_recall_at_k(&all_results, &gt_subset, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &gt_subset, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &gt_subset, 100);
    println!("  Recall@1: {:.3}", recall_at_1);
    println!("  Recall@10: {:.3}", recall_at_10);
    println!("  Recall@100: {:.3}", recall_at_100);

    BenchmarkResult {
        index_name: "ScaNN".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        num_queries,
        qps,
        recall_at_1,
        recall_at_10,
        recall_at_100,
    }
}

#[test]
fn test_sift1m_benchmark() {
    // Try to load dataset
    let dataset = match try_load_sift1m() {
        Some(ds) => ds,
        None => {
            println!("\nSkipping benchmark - SIFT1M dataset not found");
            println!("To run benchmark:");
            println!("1. Download SIFT1M from http://corpus-texmex.irisa.fr/");
            println!("2. Extract to ./data/sift/ or set SIFT1M_PATH env var");
            println!("3. Run: cargo test --test bench_sift1m -- --nocapture");
            return;
        }
    };

    // Use subset of queries for faster testing (or all 10K for full benchmark)
    let num_queries = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    println!("\nRunning SIFT1M benchmark with {} queries", num_queries);

    // Run benchmarks for all index types
    let mut results = Vec::new();

    // Flat (baseline)
    results.push(benchmark_flat(&dataset, num_queries));

    // HNSW (high recall)
    results.push(benchmark_hnsw(&dataset, num_queries));

    // IVF-Flat (cluster-based)
    results.push(benchmark_ivf_flat(&dataset, num_queries));

    // IVF-PQ (quantization, high compression)
    results.push(benchmark_ivf_pq(&dataset, num_queries));

    // IVF-SQ8 (scalar quantization, 8-bit)
    results.push(benchmark_ivf_sq8(&dataset, num_queries));

    // RaBitQ (binary quantization)
    results.push(benchmark_rabitq(&dataset, num_queries));

    // ScaNN (anisotropic quantization)
    results.push(benchmark_scann(&dataset, num_queries));

    // Print summary table
    BenchmarkResult::print_table(&results);
    BenchmarkResult::print_markdown_table(&results, "SIFT1M");

    // Save JSON if requested
    if let Ok(json_path) = env::var("JSON_OUTPUT") {
        match BenchmarkResult::save_json(&results, "SIFT1M", &json_path) {
            Ok(_) => println!("✓ JSON results saved to: {}", json_path),
            Err(e) => eprintln!("Failed to save JSON: {}", e),
        }
    }
}

#[test]
fn test_sift1m_quick() {
    // Quick test with small dataset subset
    let dataset = match try_load_sift1m() {
        Some(ds) => ds,
        None => {
            println!("Skipping quick test - SIFT1M dataset not found");
            return;
        }
    };

    let num_queries = 100;
    println!("\nRunning quick SIFT1M test with {} queries", num_queries);

    // Quick test: Flat, HNSW, IVF-Flat, IVF-PQ (most representative)
    let mut results = Vec::new();
    results.push(benchmark_flat(&dataset, num_queries));
    results.push(benchmark_hnsw(&dataset, num_queries));
    results.push(benchmark_ivf_flat(&dataset, num_queries));
    results.push(benchmark_ivf_pq(&dataset, num_queries));

    BenchmarkResult::print_table(&results);

    // Save JSON if requested
    if let Ok(json_path) = env::var("JSON_OUTPUT") {
        match BenchmarkResult::save_json(&results, "SIFT1M", &json_path) {
            Ok(_) => println!("✓ JSON results saved to: {}", json_path),
            Err(e) => eprintln!("Failed to save JSON: {}", e),
        }
    }
}

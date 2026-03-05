//! knowhere-rs 性能测试
//! 对比不同索引类型的性能
//!
//! 包含距离验证功能，确保搜索结果质量

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::faiss::{HnswIndex, IvfFlatIndex, MemIndex as FlatIndex};
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

/// Compute ground truth for random dataset (brute-force)
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

struct PerfResult {
    name: String,
    build_time_ms: f64,
    search_time_ms: f64,
    qps: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    distance_monotonic: bool,
    min_distance: f32,
    max_distance: f32,
    avg_distance: f32,
}

fn test_flat_index(n: usize, dim: usize) -> PerfResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Compute ground truth for recall calculation (k=100 for comprehensive recall)
    let ground_truth = compute_ground_truth(&vectors, &queries, 100, dim, 100);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::default(),
    };

    let build_start = Instant::now();
    let mut index = FlatIndex::new(&config).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(100);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        // Check if distances are sorted (monotonic increasing for L2)
        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall at different levels
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // Distance statistics
    let min_distance = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_distance = all_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_distance = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

    PerfResult {
        name: "Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        distance_monotonic: distances_sorted,
        min_distance,
        max_distance,
        avg_distance,
    }
}

fn test_hnsw_index(n: usize, dim: usize) -> PerfResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Compute ground truth for recall calculation (k=100 for comprehensive recall)
    let ground_truth = compute_ground_truth(&vectors, &queries, 100, dim, 100);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(32),                // OPT-029: M=16→32 for better graph connectivity
            ef_construction: Some(400), // OPT-029: Higher ef_construction for better graph quality
            ef_search: Some(400),       // OPT-029: Higher ef_search for better recall (128→400)
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(100);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        // Check if distances are sorted
        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall at different levels
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // Distance statistics
    let min_distance = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_distance = all_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_distance = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

    PerfResult {
        name: "HNSW".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        distance_monotonic: distances_sorted,
        min_distance,
        max_distance,
        avg_distance,
    }
}

fn test_ivf_flat_index(n: usize, dim: usize) -> PerfResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Compute ground truth for recall calculation (k=100 for comprehensive recall)
    let ground_truth = compute_ground_truth(&vectors, &queries, 100, dim, 100);

    let nlist = ((n as f64).sqrt() as i32).max(1);
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            nlist: Some(nlist as usize),
            nprobe: Some(10),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(100);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 10,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        // Check if distances are sorted
        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall at different levels
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // Distance statistics
    let min_distance = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_distance = all_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_distance = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

    PerfResult {
        name: "IVF-Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        distance_monotonic: distances_sorted,
        min_distance,
        max_distance,
        avg_distance,
    }
}

fn test_ivf_flat_index_fast(n: usize, dim: usize) -> PerfResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Compute ground truth for recall calculation (k=100 for comprehensive recall)
    let ground_truth = compute_ground_truth(&vectors, &queries, 100, dim, 100);

    let nlist = ((n as f64).sqrt() as i32).max(1);
    // 使用快速构建配置
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams::ivf_flat_fast(nlist as usize, 10),
    };

    let build_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(100);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 100,
            nprobe: 10,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        // Check if distances are sorted
        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall at different levels
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // Distance statistics
    let min_distance = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_distance = all_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_distance = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

    PerfResult {
        name: "IVF-Flat".to_string(),
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        distance_monotonic: distances_sorted,
        min_distance,
        max_distance,
        avg_distance,
    }
}

#[test]
fn test_performance_comparison_small() {
    println!("\n=== 小规模测试 (10K 向量，128 维) ===\n");

    let results = vec![
        test_flat_index(10000, 128),
        test_hnsw_index(10000, 128),
        test_ivf_flat_index(10000, 128),
    ];

    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Index",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "Dist OK",
        "Min",
        "Max",
        "Avg"
    );
    println!("{:-<110}", "");
    for r in &results {
        println!("{:<12} {:>10.2} {:>10.2} {:>8.0} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.2} {:>10.2} {:>10.2}", 
            r.name, r.build_time_ms, r.search_time_ms, r.qps, r.recall_at_1, r.recall_at_10, r.recall_at_100,
            if r.distance_monotonic { "✅" } else { "❌" }, r.min_distance, r.max_distance, r.avg_distance);
    }
    println!();
}

#[test]
fn test_performance_comparison_100k() {
    println!("\n=== 中等规模测试 (100K 向量，128 维) ===\n");

    let results = vec![
        test_flat_index(100000, 128),
        test_hnsw_index(100000, 128),
        test_ivf_flat_index(100000, 128),
    ];

    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Index",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "Dist OK",
        "Min",
        "Max",
        "Avg"
    );
    println!("{:-<110}", "");
    for r in &results {
        println!("{:<12} {:>10.2} {:>10.2} {:>8.0} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.2} {:>10.2} {:>10.2}", 
            r.name, r.build_time_ms, r.search_time_ms, r.qps, r.recall_at_1, r.recall_at_10, r.recall_at_100,
            if r.distance_monotonic { "✅" } else { "❌" }, r.min_distance, r.max_distance, r.avg_distance);
    }
    println!();
}

#[test]
fn test_performance_comparison_1m() {
    println!("\n=== 大规模测试 (1M 向量，128 维) ===\n");

    let results = vec![
        test_flat_index(1000000, 128),
        test_hnsw_index(1000000, 128),
        test_ivf_flat_index(1000000, 128),
    ];

    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Index",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "Dist OK",
        "Min",
        "Max",
        "Avg"
    );
    println!("{:-<110}", "");
    for r in &results {
        println!("{:<12} {:>10.2} {:>10.2} {:>8.0} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.2} {:>10.2} {:>10.2}", 
            r.name, r.build_time_ms, r.search_time_ms, r.qps, r.recall_at_1, r.recall_at_10, r.recall_at_100,
            if r.distance_monotonic { "✅" } else { "❌" }, r.min_distance, r.max_distance, r.avg_distance);
    }
    println!();
}

#[test]
fn test_ivf_flat_build_optimization() {
    println!("\n=== OPT-013: IVF-Flat 构建优化对比测试 (5K 向量，128 维) ===\n");

    let n = 5000;
    let dim = 128;

    // 测试标准配置
    let standard = test_ivf_flat_index(n, dim);

    // 测试快速配置
    let fast = test_ivf_flat_index_fast(n, dim);

    println!(
        "{:<12} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Index",
        "Build(ms)",
        "Search(ms)",
        "QPS",
        "R@1",
        "R@10",
        "R@100",
        "Dist OK",
        "Min",
        "Max",
        "Avg"
    );
    println!("{:-<110}", "");
    println!("{:<12} {:>10.2} {:>10.2} {:>8.0} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.2} {:>10.2} {:>10.2}", 
        "Standard", standard.build_time_ms, standard.search_time_ms, standard.qps, standard.recall_at_1, standard.recall_at_10, standard.recall_at_100,
        if standard.distance_monotonic { "✅" } else { "❌" }, standard.min_distance, standard.max_distance, standard.avg_distance);
    println!("{:<12} {:>10.2} {:>10.2} {:>8.0} {:>8.3} {:>8.3} {:>8.3} {:>10} {:>10.2} {:>10.2} {:>10.2}", 
        "Fast", fast.build_time_ms, fast.search_time_ms, fast.qps, fast.recall_at_1, fast.recall_at_10, fast.recall_at_100,
        if fast.distance_monotonic { "✅" } else { "❌" }, fast.min_distance, fast.max_distance, fast.avg_distance);

    // 计算优化效果
    let build_speedup = standard.build_time_ms / fast.build_time_ms;
    let build_reduction = (1.0 - fast.build_time_ms / standard.build_time_ms) * 100.0;

    println!();
    println!("📊 优化效果:");
    println!(
        "   构建时间：{:.2}ms → {:.2}ms (减少 {:.1}%, 加速 {:.2}x)",
        standard.build_time_ms, fast.build_time_ms, build_reduction, build_speedup
    );
    println!(
        "   召回率 R@1: {:.3} → {:.3}",
        standard.recall_at_1, fast.recall_at_1
    );
    println!(
        "   召回率 R@10: {:.3} → {:.3}",
        standard.recall_at_10, fast.recall_at_10
    );
    println!(
        "   召回率 R@100: {:.3} → {:.3}",
        standard.recall_at_100, fast.recall_at_100
    );

    // 验收标准检查
    println!();
    println!("✅ 验收标准检查:");
    let build_ok = fast.build_time_ms < 2000.0;
    let recall_ok =
        fast.recall_at_1 >= 0.95 && fast.recall_at_10 >= 0.99 && fast.recall_at_100 >= 0.99;
    let dist_ok = fast.distance_monotonic;

    println!(
        "   构建时间 <2s: {} ({:.2}ms)",
        if build_ok { "✅ PASS" } else { "❌ FAIL" },
        fast.build_time_ms
    );
    println!(
        "   召回率 >=95%: {} (R@1={:.3}, R@10={:.3}, R@100={:.3})",
        if recall_ok { "✅ PASS" } else { "❌ FAIL" },
        fast.recall_at_1,
        fast.recall_at_10,
        fast.recall_at_100
    );
    println!(
        "   距离单调性：{} ({})",
        if dist_ok { "✅ PASS" } else { "❌ FAIL" },
        if dist_ok { "通过" } else { "失败" }
    );

    assert!(build_ok, "构建时间应 <2s，实际 {:.2}ms", fast.build_time_ms);
    assert!(recall_ok, "召回率应保持高水平");
    assert!(dist_ok, "距离应保持单调性");

    println!();
    println!("🎉 OPT-013 优化验证通过!");
}

/// OPT-030: Test adaptive ef strategy
fn test_hnsw_adaptive_ef(
    n: usize,
    dim: usize,
    base_ef: usize,
    adaptive_k: f64,
    top_k: usize,
) -> PerfResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Compute ground truth for recall calculation
    let ground_truth = compute_ground_truth(&vectors, &queries, 100, dim, 100);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(base_ef),
            hnsw_adaptive_k: Some(adaptive_k),
            ..Default::default()
        },
    };

    let build_start = Instant::now();
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(100);
    let mut all_distances: Vec<f32> = Vec::new();
    let mut distances_sorted = true;
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            ..Default::default()
        };
        let result = index.search(query, &req).unwrap();
        all_results.push(result.ids.clone());
        all_distances.extend(result.distances.clone());

        // Check if distances are sorted
        for j in 1..result.distances.len() {
            if result.distances[j] < result.distances[j - 1] {
                distances_sorted = false;
            }
        }
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate recall at different levels
    let recall_at_1 = average_recall_at_k(&all_results, &ground_truth, 1);
    let recall_at_10 = average_recall_at_k(&all_results, &ground_truth, 10);
    let recall_at_100 = average_recall_at_k(&all_results, &ground_truth, 100);

    // Distance statistics
    let min_distance = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_distance = all_distances
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_distance = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

    PerfResult {
        name: format!("HNSW(ef={},k={:.1})", base_ef, adaptive_k),
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
        recall_at_1,
        recall_at_10,
        recall_at_100,
        distance_monotonic: distances_sorted,
        min_distance,
        max_distance,
        avg_distance,
    }
}

#[test]
fn test_opt030_adaptive_ef() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║         OPT-030: 自适应 ef 策略性能测试                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let n = 50_000;
    let dim = 128;
    let base_ef = 128;
    let top_k = 50;

    println!("\n测试配置:");
    println!("  数据集：{} 向量，{} 维", n, dim);
    println!("  base_ef: {}, top_k: {}", base_ef, top_k);
    println!("\n{:-<110}", "");
    println!("| Adaptive-k | Actual EF | Build(ms) | Search(ms) | QPS   | R@1   | R@10  | R@100 |");
    println!("{:-<110}", "");

    let adaptive_k_values = vec![1.0, 1.5, 2.0, 3.0];
    let mut results = Vec::new();

    for &adaptive_k in &adaptive_k_values {
        let actual_ef = base_ef.max((adaptive_k * top_k as f64) as usize);
        let result = test_hnsw_adaptive_ef(n, dim, base_ef, adaptive_k, top_k);
        results.push((adaptive_k, actual_ef, result));
    }

    for (adaptive_k, actual_ef, r) in &results {
        println!(
            "| {:>10.2} | {:>9} | {:>9.2} | {:>10.2} | {:>5.0} | {:>5.3} | {:>5.3} | {:>5.3} |",
            adaptive_k,
            actual_ef,
            r.build_time_ms,
            r.search_time_ms,
            r.qps,
            r.recall_at_1,
            r.recall_at_10,
            r.recall_at_100
        );
    }
    println!("{:-<110}", "");

    // 对比分析
    let baseline = &results[0];
    println!("\n📊 性能对比 (baseline: adaptive_k=1.0):");
    for (i, (adaptive_k, _actual_ef, r)) in results.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let dr10 = r.recall_at_10 - baseline.2.recall_at_10;
        let dqps = ((r.qps - baseline.2.qps) / baseline.2.qps) * 100.0;
        println!(
            "  adaptive_k={:.1}: ΔR@10={:+.3}, ΔQPS={:+.1}%",
            adaptive_k, dr10, dqps
        );
    }

    println!("\n✅ OPT-030 自适应 ef 策略测试完成!");
}

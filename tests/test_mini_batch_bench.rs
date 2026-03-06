//! Mini-Batch K-Means Benchmark
//! 对比标准 K-Means 和 Mini-Batch K-Means 的训练性能

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::faiss::IvfFlatIndex;
use knowhere_rs::MetricType;
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

struct TrainResult {
    name: String,
    train_time_ms: f64,
    build_time_ms: f64,
    search_time_ms: f64,
    qps: f64,
}

/// 测试标准 K-Means 训练的 IVF-Flat
fn test_ivf_flat_standard(n: usize, dim: usize, nlist: usize) -> TrainResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf(nlist, 10),
    };

    let train_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    let build_start = Instant::now();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 10,
            ..Default::default()
        };
        let _ = index.search(query, &req);
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    TrainResult {
        name: "IVF-Flat (Std KM)".to_string(),
        train_time_ms: train_time,
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
    }
}

/// 测试 Mini-Batch K-Means 训练的 IVF-Flat
fn test_ivf_flat_mini_batch(n: usize, dim: usize, nlist: usize, batch_size: usize) -> TrainResult {
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_mini_batch(nlist, 10, batch_size, 100, 1e-4),
    };

    let train_start = Instant::now();
    let mut index = IvfFlatIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    let train_time = train_start.elapsed().as_secs_f64() * 1000.0;

    let build_start = Instant::now();
    index.add(&vectors, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

    let search_start = Instant::now();
    for i in 0..100 {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 10,
            ..Default::default()
        };
        let _ = index.search(query, &req);
    }
    let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

    TrainResult {
        name: "IVF-Flat (MB KM)".to_string(),
        train_time_ms: train_time,
        build_time_ms: build_time,
        search_time_ms: search_time,
        qps: 100.0 / (search_time / 1000.0),
    }
}

#[test]
fn test_mini_batch_kmeans_10k() {
    println!("\n=== Mini-Batch K-Means Benchmark (10K 向量，128 维) ===\n");

    let n = 10_000;
    let dim = 128;
    let nlist = 100;
    let batch_size = 5000;

    let result_std = test_ivf_flat_standard(n, dim, nlist);
    let result_mb = test_ivf_flat_mini_batch(n, dim, nlist, batch_size);

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10}",
        "Index", "Train(ms)", "Build(ms)", "Search(ms)", "QPS"
    );
    println!("{:-<70}", "");
    println!(
        "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>10.0}",
        result_std.name,
        result_std.train_time_ms,
        result_std.build_time_ms,
        result_std.search_time_ms,
        result_std.qps
    );
    println!(
        "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>10.0}",
        result_mb.name,
        result_mb.train_time_ms,
        result_mb.build_time_ms,
        result_mb.search_time_ms,
        result_mb.qps
    );

    let train_speedup = result_std.train_time_ms / result_mb.train_time_ms;
    println!("\n训练加速比：{:.2}x", train_speedup);
    println!();
}

#[test]
fn test_mini_batch_kmeans_100k() {
    println!("\n=== Mini-Batch K-Means Benchmark (100K 向量，128 维) ===\n");

    let n = 100_000;
    let dim = 128;
    let nlist = 316; // sqrt(100K)
    let batch_size = 10000;

    let result_std = test_ivf_flat_standard(n, dim, nlist);
    let result_mb = test_ivf_flat_mini_batch(n, dim, nlist, batch_size);

    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10}",
        "Index", "Train(ms)", "Build(ms)", "Search(ms)", "QPS"
    );
    println!("{:-<70}", "");
    println!(
        "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>10.0}",
        result_std.name,
        result_std.train_time_ms,
        result_std.build_time_ms,
        result_std.search_time_ms,
        result_std.qps
    );
    println!(
        "{:<20} {:>12.2} {:>12.2} {:>12.2} {:>10.0}",
        result_mb.name,
        result_mb.train_time_ms,
        result_mb.build_time_ms,
        result_mb.search_time_ms,
        result_mb.qps
    );

    let train_speedup = result_std.train_time_ms / result_mb.train_time_ms;
    println!("\n训练加速比：{:.2}x", train_speedup);
    println!();
}

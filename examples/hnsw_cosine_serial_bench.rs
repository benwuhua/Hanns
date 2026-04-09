//! HNSW cosine serial search latency benchmark.
//! Usage: cargo run --example hnsw_cosine_serial_bench --release
//!
//! Measures p50/p95/p99 single-query latency for cosine HNSW,
//! matching HannsDB's serial search use-case.

use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::HnswIndex;
use std::time::Instant;

fn run_bench(n: usize, dim: usize, m: usize, ef_search: usize) {
    let mut rng_state: u64 = 0xdeadbeef_u64.wrapping_add(dim as u64);
    let mut next = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as i64 % 10000) as f32) / 10000.0
    };

    let data: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let queries: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();

    let params = IndexParams {
        nlist: None,
        nprobe: Some(ef_search),
        m: Some(m),
        ef_construction: Some(200),
        ..Default::default()
    };
    let cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        data_type: DataType::Float,
        dim,
        params,
    };
    let mut idx = HnswIndex::new(&cfg).expect("new");
    let t = Instant::now();
    idx.train(&data).expect("train");
    idx.add(&data, None).expect("add");
    let build_s = t.elapsed().as_secs_f64();

    let req = SearchRequest {
        top_k: 10,
        nprobe: ef_search,
        ..Default::default()
    };
    for q in &queries[..50] {
        let _ = idx.search(q, &req);
    }

    let mut latencies_us: Vec<u64> = Vec::with_capacity(queries.len());
    for q in &queries {
        let t0 = Instant::now();
        let _ = idx.search(q, &req);
        latencies_us.push(t0.elapsed().as_micros() as u64);
    }
    latencies_us.sort_unstable();
    let n_q = latencies_us.len();
    let p50 = latencies_us[n_q * 50 / 100];
    let p95 = latencies_us[n_q * 95 / 100];
    let p99 = latencies_us[n_q * 99 / 100];
    let mean_us = latencies_us.iter().sum::<u64>() / n_q as u64;
    let qps = 1_000_000.0 / mean_us as f64;
    let n_k = n / 1_000;

    println!(
        "{}K/{}/cosine/M={}/ef={}: build={:.1}s  p50={}µs p95={}µs p99={}µs mean={}µs QPS={:.0}",
        n_k, dim, m, ef_search, build_s, p50, p95, p99, mean_us, qps
    );
}

fn main() {
    println!("=== HNSW Cosine Serial Search Latency Benchmark ===");
    for &m in &[8usize, 16usize] {
        run_bench(10_000, 768, m, 32);
        run_bench(50_000, 768, m, 32);
        run_bench(10_000, 1536, m, 32);
        run_bench(50_000, 1536, m, 32);
    }
}

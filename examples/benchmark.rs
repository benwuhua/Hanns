//! Benchmark for KnowHere RS

use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use knowhere_rs::faiss::{DiskAnnIndex, HnswIndex, IvfPqIndex, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NUM_VECTORS: usize = 1_000;
const DIM: usize = 128;
const TOP_K: usize = 10;

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn benchmark_flat_index() {
    println!("\n=== Flat Index Benchmark ===");

    let config = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut index = MemIndex::new(&config).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!(
        "  Throughput: {:.2} vectors/sec",
        NUM_VECTORS as f64 / add_time.as_secs_f64()
    );

    // Benchmark search
    let query = generate_vectors(100, DIM);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 1,
        filter: None,
        params: None,
        radius: None,
    };

    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!(
        "  QPS: {:.2} queries/sec",
        100.0 / search_time.as_secs_f64()
    );
}

fn benchmark_hnsw_index() {
    println!("\n=== HNSW Index Benchmark ===");

    let params = IndexParams::hnsw(200, 50, 0.5);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = HnswIndex::new(&config).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Train
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train: {:?}", train_time);

    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!(
        "  Throughput: {:.2} vectors/sec",
        NUM_VECTORS as f64 / add_time.as_secs_f64()
    );

    // Benchmark search
    let query = generate_vectors(100, DIM);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 50,
        filter: None,
        params: None,
        radius: None,
    };

    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!(
        "  QPS: {:.2} queries/sec",
        100.0 / search_time.as_secs_f64()
    );
}

fn benchmark_ivfpq_index() {
    println!("\n=== IVF-PQ Index Benchmark ===");

    let params = IndexParams::ivf(100, 10);
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = IvfPqIndex::new(&config).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Train
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train (k-means): {:?}", train_time);

    // Benchmark add
    let start = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_time = start.elapsed();
    println!("Add {} vectors: {:?}", NUM_VECTORS, add_time);
    println!(
        "  Throughput: {:.2} vectors/sec",
        NUM_VECTORS as f64 / add_time.as_secs_f64()
    );

    // Benchmark search
    let query = generate_vectors(100, DIM);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };

    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!(
        "  QPS: {:.2} queries/sec",
        100.0 / search_time.as_secs_f64()
    );
}

fn benchmark_diskann_index() {
    println!("\n=== DiskANN Index Benchmark ===");

    let config = IndexConfig {
        index_type: IndexType::DiskAnn,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params: IndexParams::default(),
    };

    let mut index = DiskAnnIndex::new(&config).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Train (build graph)
    let start = Instant::now();
    index.train(&vectors).unwrap();
    let train_time = start.elapsed();
    println!("Train (graph build): {:?}", train_time);

    // Benchmark search
    let query = generate_vectors(100, DIM);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 50,
        filter: None,
        params: None,
        radius: None,
    };

    let start = Instant::now();
    for q in query.chunks(DIM) {
        let _ = index.search(q, &req).unwrap();
    }
    let search_time = start.elapsed();
    println!("Search 100 queries: {:?}", search_time);
    println!(
        "  QPS: {:.2} queries/sec",
        100.0 / search_time.as_secs_f64()
    );
}

fn brute_force_top_k(vectors: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<usize>> {
    let n = vectors.len() / dim;
    let nq = queries.len() / dim;
    let mut gt = Vec::with_capacity(nq);
    for q_idx in 0..nq {
        let q = &queries[q_idx * dim..(q_idx + 1) * dim];
        let mut scored = Vec::with_capacity(n);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let dist = q
                .iter()
                .zip(v.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f32>();
            scored.push((i, dist));
        }
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        gt.push(scored.into_iter().take(k).map(|(i, _)| i).collect());
    }
    gt
}

fn recall_at_k(result_ids: &[i64], gt_indices: &[Vec<usize>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (q_idx, gt) in gt_indices.iter().enumerate() {
        let start = q_idx * k;
        let end = (start + k).min(result_ids.len());
        let got = &result_ids[start..end];
        for &idx in gt.iter().take(k) {
            total += 1;
            if got.iter().any(|&id| id == idx as i64) {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn benchmark_diskann() {
    const NUM_VECTORS: usize = 10_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 1_000;
    const NUM_RECALL_QUERIES: usize = 100;

    println!("\n=== DiskANN Benchmark ===");

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let queries_qps = generate_vectors(NUM_QPS_QUERIES, DIM);
    let queries_recall = generate_vectors(NUM_RECALL_QUERIES, DIM);

    let config = IndexConfig {
        index_type: IndexType::DiskAnn,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params: IndexParams {
            max_degree: Some(48),
            search_list_size: Some(128),
            construction_l: Some(128),
            num_threads: Some(rayon::current_num_threads()),
            ..IndexParams::default()
        },
    };
    let mut index = DiskAnnIndex::new(&config).unwrap();

    let start = Instant::now();
    index.train(&vectors).unwrap();
    let build_time = start.elapsed().as_secs_f64();
    println!("Build 10000 vectors (dim=128): {:.2}s", build_time);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 128,
        filter: None,
        params: None,
        radius: None,
    };
    let start = Instant::now();
    let qps_result = index.search(&queries_qps, &req).unwrap();
    let _ = qps_result.ids.len();
    let search_s = start.elapsed().as_secs_f64();
    let qps = NUM_QPS_QUERIES as f64 / search_s.max(f64::EPSILON);
    println!("Search QPS (L=128, R=48): {:.0} queries/sec", qps);

    let recall_result = index.search(&queries_recall, &req).unwrap();
    let gt = brute_force_top_k(&vectors, &queries_recall, DIM, TOP_K);
    let recall = recall_at_k(&recall_result.ids, &gt, TOP_K);
    println!("Recall@10 (100 queries): {:.3}", recall);
}

fn benchmark_pqflash() {
    const NUM_VECTORS: usize = 10_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 1_000;
    const NUM_RECALL_QUERIES: usize = 100;

    println!("\n=== PQFlashIndex Benchmark ===");

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let queries_qps = generate_vectors(NUM_QPS_QUERIES, DIM);
    let queries_recall = generate_vectors(NUM_RECALL_QUERIES, DIM);

    let config_nopq = AisaqConfig {
        disk_pq_dims: 0,
        ..AisaqConfig::default()
    };
    let config = AisaqConfig {
        disk_pq_dims: 32,
        rerank_expand_pct: 1000,
        search_list_size: 200,
        cache_all_on_load: true,
        run_refine_pass: true,
        ..AisaqConfig::default()
    };

    let run = |label: &str, cfg: AisaqConfig| {
        let mut index = PQFlashIndex::new(cfg, MetricType::L2, DIM).unwrap();

        let start = Instant::now();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();
        let build_time = start.elapsed().as_secs_f64();
        println!("[{label}] Build 10000 vectors (dim=128): {:.2}s", build_time);

        let start = Instant::now();
        let qps_result = index.search_batch(&queries_qps, TOP_K).unwrap();
        let _ = qps_result.ids.len();
        let search_s = start.elapsed().as_secs_f64();
        let qps = NUM_QPS_QUERIES as f64 / search_s.max(f64::EPSILON);
        println!("[{label}] Search QPS (L=128, R=48): {:.0} queries/sec", qps);

        let gt = brute_force_top_k(&vectors, &queries_recall, DIM, TOP_K);
        let recall_result = index.search_batch(&queries_recall, TOP_K).unwrap();
        let recall = recall_at_k(&recall_result.ids, &gt, TOP_K);
        println!("[{label}] Recall@10 (100 queries): {:.3}", recall);
    };

    run("NoPQ", config_nopq);
    run("PQ32", config);

    println!("\n[Disk path] Building and saving PQFlash NoPQ...");
    let disk_build_config = AisaqConfig {
        disk_pq_dims: 0,
        cache_all_on_load: false,
        pq_read_page_cache_size: 512 * 1024,
        ..AisaqConfig::default()
    };
    let mut build_idx = PQFlashIndex::new(disk_build_config, MetricType::L2, DIM).unwrap();
    build_idx.train(&vectors).unwrap();
    build_idx.add(&vectors).unwrap();
    let tmp_dir = std::env::temp_dir().join(format!(
        "pqflash_disk_bench_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    std::fs::create_dir_all(&tmp_dir).ok();
    let _ = build_idx.save(&tmp_dir).unwrap();

    let disk_idx = PQFlashIndex::load(&tmp_dir).unwrap();
    println!("[Disk NoPQ] Warming up...");
    for _ in 0..10 {
        let _ = disk_idx.search_batch(&queries_qps[..DIM * 10], TOP_K);
    }
    let start = Instant::now();
    let _ = disk_idx.search_batch(&queries_qps, TOP_K).unwrap();
    let disk_qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!("[Disk NoPQ] Cold→Hot QPS: {:.0} queries/sec", disk_qps);
    std::fs::remove_dir_all(&tmp_dir).ok();

    // --- Diagnostic: PQ32 recall sweep ---
    let gt = brute_force_top_k(&vectors, &queries_recall, DIM, TOP_K);
    let sweep_configs: &[(usize, usize, &str)] = &[
        (200, 300, "PQ32 L=200 rerank=3x"),
        (200, 1000, "PQ32 L=200 rerank=10x"),
        (500, 1000, "PQ32 L=500 rerank=10x"),
    ];
    for &(sl, rr, label) in sweep_configs {
        let cfg = AisaqConfig {
            disk_pq_dims: 32,
            rerank_expand_pct: rr,
            search_list_size: sl,
            cache_all_on_load: true,
            rearrange: true,
            run_refine_pass: true,
            ..AisaqConfig::default()
        };
        let mut idx = PQFlashIndex::new(cfg, MetricType::L2, DIM).unwrap();
        idx.train(&vectors).unwrap();
        idx.add(&vectors).unwrap();
        let recall_result = idx.search_batch(&queries_recall, TOP_K).unwrap();
        let recall = recall_at_k(&recall_result.ids, &gt, TOP_K);
        println!("[sweep {label}] Recall@10: {:.3}", recall);
    }
}

fn main() {
    println!("KnowHere RS Benchmark");
    println!("=====================");
    println!("Vectors: {}", NUM_VECTORS);
    println!("Dimension: {}", DIM);
    println!("Top-K: {}", TOP_K);

    benchmark_flat_index();
    benchmark_hnsw_index();
    benchmark_ivfpq_index();
    benchmark_diskann_index();
    benchmark_diskann();
    benchmark_pqflash();

    println!("\n✅ Benchmark complete!");
}

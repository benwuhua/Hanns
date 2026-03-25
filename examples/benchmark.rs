//! Benchmark for KnowHere RS

use std::time::Instant;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use knowhere_rs::faiss::{DiskAnnIndex, HnswIndex, IvfPqIndex, IvfSq8Index, MemIndex};
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

    let mut params = IndexParams::default();
    params.nlist = Some(32);
    params.nprobe = Some(8);
    params.m = Some(8);
    params.nbits_per_idx = Some(8);
    let config = IndexConfig {
        index_type: IndexType::IvfPq,
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
        nprobe: 8,
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

fn benchmark_ivfpq_100k() {
    const NUM_VECTORS: usize = 100_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 1_000;

    println!("\n=== IVF-PQ 100K Benchmark ===");

    let mut params = IndexParams::default();
    params.nlist = Some(256);
    params.nprobe = Some(64);
    params.m = Some(8);
    params.nbits_per_idx = Some(8);

    let config = IndexConfig {
        index_type: IndexType::IvfPq,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = IvfPqIndex::new(&config).unwrap();
    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let queries = generate_vectors(NUM_QPS_QUERIES, DIM);

    let t0 = Instant::now();
    index.train(&vectors).unwrap();
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_s = t1.elapsed().as_secs_f64();
    let add_tput = NUM_VECTORS as f64 / add_s.max(f64::EPSILON);

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 64,
        filter: None,
        params: None,
        radius: None,
    };

    let t2 = Instant::now();
    let _ = index.search(&queries, &req).unwrap();
    let qps = NUM_QPS_QUERIES as f64 / t2.elapsed().as_secs_f64().max(f64::EPSILON);

    println!("Train time: {:.2}s", train_s);
    println!("Add time: {:.2}s (throughput {:.0} vec/s)", add_s, add_tput);
    println!("Search QPS: {:.0}", qps);
}

fn overlap_at_k(a: &[i64], b: &[i64]) -> usize {
    let mut hits = 0usize;
    for &id in b.iter().take(TOP_K) {
        if a.iter().take(TOP_K).any(|&x| x == id) {
            hits += 1;
        }
    }
    hits
}

fn benchmark_ivf_sq8() {
    const BASE_SIZE: usize = 100_000;
    const DIM: usize = 128;
    const NLIST: usize = 256;
    const RECALL_QUERIES: usize = 200;
    const QPS_QUERIES: usize = 1_000;
    const NPROBES: [usize; 4] = [32, 64, 128, 256];

    println!("\n=== IVF-SQ8 100K Benchmark ===");
    println!(
        "base={} dim={} nlist={} top_k={} seed=42",
        BASE_SIZE, DIM, NLIST, TOP_K
    );

    let mut rng = StdRng::seed_from_u64(42);
    let base: Vec<f32> = (0..BASE_SIZE * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let recall_queries: Vec<f32> = (0..RECALL_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let qps_queries: Vec<f32> = (0..QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg).unwrap();
    gt_index.add(&base, None).unwrap();

    let mut params = IndexParams::default();
    params.nlist = Some(NLIST);
    params.nprobe = Some(32);
    let cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };
    let mut index = IvfSq8Index::new(&cfg).unwrap();

    let t0 = Instant::now();
    index.train(&base).unwrap();
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(&base, None).unwrap();
    let add_s = t1.elapsed().as_secs_f64();
    println!(
        "Train {:.2}s, Add {:.2}s ({:.0} vec/s)",
        train_s,
        add_s,
        BASE_SIZE as f64 / add_s.max(f64::EPSILON)
    );

    let gt_req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };
    let gt_top10: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            gt_index.search(q, &gt_req).unwrap().ids
        })
        .collect();

    println!("single-query sweep:");
    println!("nprobe | recall@10 | qps");
    println!("-------|-----------|------");

    for &nprobe in &NPROBES {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let qps_start = Instant::now();
        for q in qps_queries.chunks(DIM) {
            let _ = index.search(q, &req).unwrap();
        }
        let qps = (QPS_QUERIES as f64 / qps_start.elapsed().as_secs_f64().max(f64::EPSILON)).round()
            as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req).unwrap();
            recall_sum += overlap_at_k(gt_ids, &res.ids) as f64 / TOP_K as f64;
        }
        let recall = recall_sum / RECALL_QUERIES as f64;

        println!("{:>6} | {:>9.3} | {}", nprobe, recall, qps);
    }

    #[cfg(feature = "parallel")]
    {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: 32,
            ..Default::default()
        };

        let t = Instant::now();
        let _ = index.search_parallel(&qps_queries, &req, 0).unwrap();
        let batch_qps =
            (QPS_QUERIES as f64 / t.elapsed().as_secs_f64().max(f64::EPSILON)).round() as u64;

        let batch_recall_res = index.search_parallel(&recall_queries, &req, 0).unwrap();
        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let start = i * TOP_K;
            let end = start + TOP_K;
            recall_sum += overlap_at_k(gt_ids, &batch_recall_res.ids[start..end]) as f64
                / TOP_K as f64;
        }
        let batch_recall = recall_sum / RECALL_QUERIES as f64;

        println!(
            "batch parallel (nprobe=32): recall@10={:.3}, qps={}",
            batch_recall, batch_qps
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("batch parallel (nprobe=32): skipped (feature \"parallel\" not enabled)");
    }
}

fn benchmark_ivf_sq8_1m() {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 128;
    const NLIST: usize = 1024;
    const RECALL_QUERIES: usize = 200;
    const QPS_QUERIES: usize = 1_000;
    const NPROBES: [usize; 5] = [32, 64, 128, 256, 512];

    println!("\n=== IVF-SQ8 1M Benchmark ===");
    println!(
        "base={} dim={} nlist={} top_k={} seed=42",
        NUM_VECTORS, DIM, NLIST, TOP_K
    );

    let mut rng = StdRng::seed_from_u64(42);
    let base: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let recall_queries: Vec<f32> = (0..RECALL_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let qps_queries: Vec<f32> = (0..QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg).unwrap();
    gt_index.add(&base, None).unwrap();

    let mut params = IndexParams::default();
    params.nlist = Some(NLIST);
    params.nprobe = Some(32);
    let cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params,
    };
    let mut index = IvfSq8Index::new(&cfg).unwrap();

    let t0 = Instant::now();
    index.train(&base).unwrap();
    let train_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    index.add(&base, None).unwrap();
    let add_s = t1.elapsed().as_secs_f64();

    println!(
        "Build 1M vectors: {:.1}s (train {:.1}s + add {:.1}s)",
        train_s + add_s,
        train_s,
        add_s
    );

    let gt_req = SearchRequest {
        top_k: TOP_K,
        ..Default::default()
    };
    let gt_top10: Vec<Vec<i64>> = (0..RECALL_QUERIES)
        .map(|i| {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            gt_index.search(q, &gt_req).unwrap().ids
        })
        .collect();

    println!("single-query sweep:");
    println!("nprobe | recall@10 | qps");
    println!("-------|-----------|------");
    for &nprobe in &NPROBES {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe,
            ..Default::default()
        };

        let qps_start = Instant::now();
        for q in qps_queries.chunks(DIM) {
            let _ = index.search(q, &req).unwrap();
        }
        let qps = (QPS_QUERIES as f64 / qps_start.elapsed().as_secs_f64().max(f64::EPSILON)).round()
            as u64;

        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let q = &recall_queries[i * DIM..(i + 1) * DIM];
            let res = index.search(q, &req).unwrap();
            recall_sum += overlap_at_k(gt_ids, &res.ids) as f64 / TOP_K as f64;
        }
        let recall = recall_sum / RECALL_QUERIES as f64;

        println!("{:>6} | {:>9.3} | {}", nprobe, recall, qps);
    }

    #[cfg(feature = "parallel")]
    {
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: 32,
            ..Default::default()
        };

        let t = Instant::now();
        let _ = index.search_parallel(&qps_queries, &req, 0).unwrap();
        let batch_qps =
            (QPS_QUERIES as f64 / t.elapsed().as_secs_f64().max(f64::EPSILON)).round() as u64;

        let batch_recall_res = index.search_parallel(&recall_queries, &req, 0).unwrap();
        let mut recall_sum = 0.0;
        for (i, gt_ids) in gt_top10.iter().enumerate() {
            let start = i * TOP_K;
            let end = start + TOP_K;
            recall_sum += overlap_at_k(gt_ids, &batch_recall_res.ids[start..end]) as f64
                / TOP_K as f64;
        }
        let batch_recall = recall_sum / RECALL_QUERIES as f64;

        println!(
            "batch parallel (nprobe=32): recall@10={:.3}, qps={}",
            batch_recall, batch_qps
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("batch parallel (nprobe=32): skipped (feature \"parallel\" not enabled)");
    }
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

    // Page cache path (warm) — 256-shard Mutex, no node pre-loading
    let disk_idx = PQFlashIndex::load(&tmp_dir).unwrap();
    println!("[Disk NoPQ] Warming up...");
    for _ in 0..10 {
        let _ = disk_idx.search_batch(&queries_qps[..DIM * 10], TOP_K);
    }
    let start = Instant::now();
    let _ = disk_idx.search_batch(&queries_qps, TOP_K).unwrap();
    let disk_qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!("[Disk NoPQ] PageCache warm QPS: {:.0} queries/sec", disk_qps);

    let disk_mmap_idx = PQFlashIndex::load_with_mmap(&tmp_dir).unwrap();
    println!("[Disk NoPQ] Warming up direct mmap...");
    for _ in 0..10 {
        let _ = disk_mmap_idx.search_batch(&queries_qps[..DIM * 10], TOP_K);
    }
    let start = Instant::now();
    let _ = disk_mmap_idx.search_batch(&queries_qps, TOP_K).unwrap();
    let disk_mmap_qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!("[Disk NoPQ] Direct mmap QPS: {:.0} queries/sec", disk_mmap_qps);

    // lock-free HashMap path via enable_node_cache — Arc::clone per node, no Mutex contention
    let mut disk_cached_idx = PQFlashIndex::load(&tmp_dir).unwrap();
    disk_cached_idx.enable_node_cache().unwrap();
    let start = Instant::now();
    let _ = disk_cached_idx.search_batch(&queries_qps, TOP_K).unwrap();
    let disk_cached_qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!("[Disk NoPQ] Cached (lock-free) QPS: {:.0} queries/sec", disk_cached_qps);

    let native_prefix = tmp_dir.join("native_roundtrip");
    build_idx
        .export_native_disk_index(native_prefix.to_str().unwrap())
        .unwrap();
    let native_import_idx = PQFlashIndex::import_native_disk_index(native_prefix.to_str().unwrap())
        .unwrap();
    let start = Instant::now();
    let _ = native_import_idx.search_batch(&queries_qps, TOP_K).unwrap();
    let native_import_qps =
        NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!(
        "[Disk NoPQ] Native disk.index import QPS: {:.0} queries/sec",
        native_import_qps
    );
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

fn benchmark_diskann_100k() {
    const NUM_VECTORS: usize = 100_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 1_000;

    println!("\n=== DiskANN 100K Benchmark ===");
    let mut rng = StdRng::seed_from_u64(123);
    let vectors: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NUM_QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

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
    println!("Build 100K vectors: {:.1}s", start.elapsed().as_secs_f64());

    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 128,
        filter: None,
        params: None,
        radius: None,
    };
    let start = Instant::now();
    let _ = index.search(&queries, &req).unwrap();
    let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
    println!("Search QPS: {:.0}", qps);
}

fn benchmark_pqflash_100k() {
    const NUM_VECTORS: usize = 100_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 1_000;

    println!("\n=== PQFlash 100K Benchmark ===");
    let mut rng = StdRng::seed_from_u64(123);
    let vectors: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NUM_QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

    // NoPQ 100K
    {
        let config = AisaqConfig {
            disk_pq_dims: 0,
            search_list_size: 128,
            cache_all_on_load: true,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).unwrap();
        let start = Instant::now();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();
        let build_s = start.elapsed().as_secs_f64();
        println!("[NoPQ] Build 100K vectors: {:.1}s", build_s);
        let start = Instant::now();
        let _ = index.search_batch(&queries, TOP_K).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        println!("[NoPQ] Search QPS: {:.0}", qps);
    }

    // PQ32 100K — with phase timing
    {
        let config = AisaqConfig {
            disk_pq_dims: 32,
            rerank_expand_pct: 300,
            search_list_size: 200,
            cache_all_on_load: true,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).unwrap();
        let t0 = Instant::now();
        index.train(&vectors).unwrap();
        let train_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        index.add(&vectors).unwrap();
        let add_s = t1.elapsed().as_secs_f64();
        println!("[PQ32] Build 100K vectors: {:.1}s  (train {:.1}s + add {:.1}s)",
            train_s + add_s, train_s, add_s);
        let start = Instant::now();
        let _ = index.search_batch(&queries, TOP_K).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        println!("[PQ32] Search QPS: {:.0}", qps);
    }
}

fn benchmark_pqflash_1m() {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 128;
    const TOP_K: usize = 10;
    const NUM_QPS_QUERIES: usize = 500;  // 1M时查询集小一点

    println!("\n=== PQFlash 1M Benchmark ===");
    let mut rng = StdRng::seed_from_u64(456);
    let vectors: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NUM_QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();

    // NoPQ 1M
    {
        let config = AisaqConfig {
            disk_pq_dims: 0,
            search_list_size: 128,
            cache_all_on_load: true,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).unwrap();
        let t0 = Instant::now();
        index.train(&vectors).unwrap();
        let train_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        index.add(&vectors).unwrap();
        let add_s = t1.elapsed().as_secs_f64();
        println!("[NoPQ] Build 1M vectors: {:.1}s  (train {:.1}s + add {:.1}s)",
            train_s + add_s, train_s, add_s);
        let start = Instant::now();
        let _ = index.search_batch(&queries, TOP_K).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        println!("[NoPQ] Search QPS: {:.0}", qps);
    }

    // PQ32 1M
    {
        let config = AisaqConfig {
            disk_pq_dims: 32,
            rerank_expand_pct: 300,
            search_list_size: 200,
            cache_all_on_load: true,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).unwrap();
        let t0 = Instant::now();
        index.train(&vectors).unwrap();
        let train_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        index.add(&vectors).unwrap();
        let add_s = t1.elapsed().as_secs_f64();
        println!("[PQ32] Build 1M vectors: {:.1}s  (train {:.1}s + add {:.1}s)",
            train_s + add_s, train_s, add_s);
        let start = Instant::now();
        let _ = index.search_batch(&queries, TOP_K).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        println!("[PQ32] Search QPS: {:.0}", qps);
    }

    // NoPQ + SQ8 prefilter 1M
    {
        let config = AisaqConfig {
            disk_pq_dims: 0,
            search_list_size: 128,
            cache_all_on_load: true,
            use_sq8_prefilter: true,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).unwrap();
        let t0 = Instant::now();
        index.train(&vectors).unwrap();
        let train_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        index.add(&vectors).unwrap();
        let add_s = t1.elapsed().as_secs_f64();
        println!(
            "[NoPQ+SQ8] Build 1M vectors: {:.1}s  (train {:.1}s + add {:.1}s)",
            train_s + add_s,
            train_s,
            add_s
        );
        let start = Instant::now();
        let _ = index.search_batch(&queries, TOP_K).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        println!("[NoPQ+SQ8] Search QPS: {:.0}", qps);
    }
}

fn benchmark_hnsw_1m() {
    const NUM_VECTORS: usize = 1_000_000;
    const DIM: usize = 128;
    const NUM_QPS_QUERIES: usize = 500;
    const NUM_RECALL_QUERIES: usize = 500;
    const TOP_K: usize = 10;

    println!("\n=== HNSW 1M Benchmark ===");

    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..NUM_VECTORS * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..NUM_QPS_QUERIES * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let recall_queries = &queries[..NUM_RECALL_QUERIES * DIM];

    let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, DIM);
    let mut gt_index = MemIndex::new(&flat_cfg).unwrap();
    gt_index.add(&vectors, None).unwrap();
    let gt_req = SearchRequest {
        top_k: TOP_K,
        nprobe: 1,
        filter: None,
        params: None,
        radius: None,
    };
    let gt = gt_index.search(recall_queries, &gt_req).unwrap();
    let gt_indices: Vec<Vec<usize>> = gt
        .ids
        .chunks(TOP_K)
        .map(|row| row.iter().map(|&id| id as usize).collect())
        .collect();

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        dim: DIM,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ..IndexParams::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    let t_train = Instant::now();
    index.train(&vectors).unwrap();
    let train_s = t_train.elapsed().as_secs_f64();
    let t_add = Instant::now();
    index.add(&vectors, None).unwrap();
    let add_s = t_add.elapsed().as_secs_f64();
    println!(
        "Build 1M vectors: {:.1}s  (train {:.1}s + add {:.1}s)",
        train_s + add_s,
        train_s,
        add_s
    );

    for ef in [50usize, 100, 200, 400, 800] {
        index.set_ef_search(ef);
        let req = SearchRequest {
            top_k: TOP_K,
            nprobe: 0,
            filter: None,
            params: None,
            radius: None,
        };
        let start = Instant::now();
        let result = index.search(&queries, &req).unwrap();
        let qps = NUM_QPS_QUERIES as f64 / start.elapsed().as_secs_f64().max(f64::EPSILON);
        let recall = recall_at_k(&result.ids[..NUM_RECALL_QUERIES * TOP_K], &gt_indices, TOP_K);
        println!("M=16 ef={ef}: recall@10={recall:.4} QPS={qps:.0} (batch)");
    }
}

fn benchmark_aisaq_refine_passes() {
    println!("\n=== AISAQ num_refine_passes sweep (100K, NoPQ) ===");

    let n: usize = 200_000;
    let dim: usize = 128;
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<f32> = (0..n * dim).map(|_| rng.r#gen::<f32>()).collect();
    let queries: Vec<f32> = (0..1000 * dim).map(|_| rng.r#gen::<f32>()).collect();

    // Brute-force ground truth (top-10)
    let top_k = 10usize;
    let gt: Vec<Vec<usize>> = (0..1000)
        .map(|qi| {
            let qv = &queries[qi * dim..(qi + 1) * dim];
            let mut dists: Vec<(f32, usize)> = (0..n)
                .map(|di| {
                    let dv = &data[di * dim..(di + 1) * dim];
                    let d: f32 = qv
                        .iter()
                        .zip(dv.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum();
                    (d, di)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.iter().take(top_k).map(|&(_, id)| id).collect()
        })
        .collect();

    for passes in [1usize, 2, 3] {
        let cfg = AisaqConfig {
            max_degree: 32,
            build_search_list_size: 64,
            num_refine_passes: passes,
            search_list_size: 64,
            cache_all_on_load: true,
            ..AisaqConfig::default()
        };

        let t_build = Instant::now();
        let mut index = PQFlashIndex::new(cfg.clone(), MetricType::L2, dim).unwrap();
        index.train(&data).unwrap();
        index.add(&data).unwrap();
        let build_s = t_build.elapsed().as_secs_f64();

        let t_search = Instant::now();
        let batch = index.search_batch(&queries, top_k).unwrap();
        let qps = 1000.0 / t_search.elapsed().as_secs_f64().max(f64::EPSILON);

        let mut hits = 0usize;
        for (qi, gt_row) in gt.iter().enumerate().take(1000) {
            let result_ids = &batch.ids[qi * top_k..(qi + 1) * top_k];
            for &gt_id in gt_row {
                if result_ids.iter().any(|&rid| rid as usize == gt_id) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (1000 * top_k) as f64;

        println!(
            "passes={}: build={:.1}s recall@10={:.4} QPS={:.0}",
            passes, build_s, recall, qps
        );
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
    benchmark_ivfpq_100k();
    benchmark_ivf_sq8();
    benchmark_ivf_sq8_1m();
    benchmark_hnsw_1m();
    benchmark_diskann_index();
    benchmark_diskann();
    benchmark_pqflash();
    benchmark_diskann_100k();
    benchmark_pqflash_100k();
    benchmark_pqflash_1m();
    benchmark_aisaq_refine_passes();

    println!("\n✅ Benchmark complete!");
}

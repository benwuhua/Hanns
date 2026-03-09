use crate::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use crate::benchmark::{average_recall_at_k, confidence_from_recall, write_report, BenchmarkReport, BenchmarkReportRow};
use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use crate::faiss::sparse_inverted::SparseVector;
use crate::faiss::{
    HnswIndex, IvfFlatIndex, IvfRaBitqConfig, IvfRaBitqIndex, MemIndex as FlatIndex, ScaNNConfig,
    ScaNNIndex, SparseMetricType, SparseWandIndex,
};
use rand::Rng;
use std::error::Error;
use std::time::Instant;

pub const DIM: usize = 64;
pub const BASE_SIZE: usize = 5000;
pub const QUERY_SIZE: usize = 100;
pub const TOP_K: usize = 10;
pub const RECALL_GATE: f64 = 0.80;
pub const DEFAULT_OUTPUT_PATH: &str = "benchmark_results/recall_gated_baseline.json";

fn generate_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut gt = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &queries[i * dim..(i + 1) * dim];
        let mut pairs: Vec<(usize, f32)> = (0..num_base)
            .map(|j| {
                let b = &base[j * dim..(j + 1) * dim];
                (j, l2_distance_squared(q, b))
            })
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("distance compare"));
        gt.push(pairs.into_iter().take(k).map(|(idx, _)| idx as i32).collect());
    }

    gt
}

fn explanation_for(index: &str, confidence: &str) -> String {
    match (index, confidence) {
        (_, "trusted") => String::new(),
        ("HNSW", "unreliable") | ("HNSW", "recheck required") => {
            "建议将 ef_search 从 128 提升到 >=256，并在固定随机种子下复跑 3 次对比 recall 漂移。".to_string()
        }
        ("IVF-Flat", "unreliable") | ("IVF-Flat", "recheck required") => {
            "建议增大 nprobe（16 -> 32/64）并检查 nlist 与样本规模匹配关系，复跑同一数据集验证 recall。".to_string()
        }
        ("DiskANN", "unreliable") | ("DiskANN", "recheck required") => {
            "建议提高 search_list_size/beamwidth（128/8 -> 256/16）并复测；当前参数更偏吞吐，召回不足。".to_string()
        }
        ("ScaNN", "unreliable") | ("ScaNN", "recheck required") => {
            "建议提升 reorder_k 与 num_partitions，并引入代表性 query_sample 重新训练量化器后复测。".to_string()
        }
        ("RaBitQ", "unreliable") | ("RaBitQ", "recheck required") => {
            "RaBitQ 为有损量化，建议提高 nprobe 并开启 refine（fp32/dataview）后复测 recall@10。".to_string()
        }
        ("SparseWand", "unreliable") | ("SparseWand", "recheck required") => {
            "建议检查稀疏度阈值和查询非零分布；可对照 TAAT 精确路径核验 WAND 早停带来的召回损失。".to_string()
        }
        _ => "低可信结果需要复跑并记录参数与数据分布。".to_string(),
    }
}

fn make_row(index: &str, qps: f64, recall_at_10: f64, ground_truth_source: &str) -> BenchmarkReportRow {
    let confidence = confidence_from_recall(recall_at_10, RECALL_GATE).to_string();
    BenchmarkReportRow {
        index: index.to_string(),
        qps,
        recall_at_10,
        ground_truth_source: ground_truth_source.to_string(),
        confidence_explanation: explanation_for(index, &confidence),
        confidence,
    }
}

fn bench_hnsw(base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> BenchmarkReportRow {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(128),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create hnsw");
    index.train(base).expect("train hnsw");
    index.add(base, None).expect("add hnsw");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    ..Default::default()
                },
            )
            .expect("search hnsw");
        all_results.push(result.ids);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, gt, TOP_K);

    make_row("HNSW", qps, recall_at_10, "flat_exact_l2_bruteforce")
}

fn bench_ivf(base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> BenchmarkReportRow {
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            nlist: Some(128),
            nprobe: Some(16),
            ..Default::default()
        },
    };

    let mut index = IvfFlatIndex::new(&config).expect("create ivf");
    index.train(base).expect("train ivf");
    index.add(base, None).expect("add ivf");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe: 16,
                    ..Default::default()
                },
            )
            .expect("search ivf");
        all_results.push(result.ids);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, gt, TOP_K);

    make_row("IVF-Flat", qps, recall_at_10, "flat_exact_l2_bruteforce")
}

fn bench_diskann(base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> BenchmarkReportRow {
    let config = AisaqConfig {
        max_degree: 48,
        search_list_size: 128,
        beamwidth: 8,
        num_entry_points: 1,
        ..AisaqConfig::default()
    };

    let mut index = PQFlashIndex::new(config, MetricType::L2, DIM).expect("create diskann");
    index.add(base).expect("add diskann");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index.search(q, TOP_K).expect("search diskann");
        all_results.push(result.ids);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, gt, TOP_K);

    make_row("DiskANN", qps, recall_at_10, "flat_exact_l2_bruteforce")
}

fn bench_scann(base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> BenchmarkReportRow {
    let mut index =
        ScaNNIndex::new(DIM, ScaNNConfig::new(64, 16, 128)).expect("create scann");
    index.train(base, Some(queries));
    index.add(base, None);

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index.search(q, TOP_K);
        all_results.push(result.into_iter().map(|(id, _)| id).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, gt, TOP_K);

    make_row("ScaNN", qps, recall_at_10, "flat_exact_l2_bruteforce")
}

fn bench_rabitq(base: &[f32], queries: &[f32], gt: &[Vec<i32>]) -> BenchmarkReportRow {
    let mut index = IvfRaBitqIndex::new(IvfRaBitqConfig::new(DIM, 128).with_nprobe(16));
    index.train(base).expect("train rabitq");
    index.add(base, None).expect("add rabitq");

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for i in 0..QUERY_SIZE {
        let q = &queries[i * DIM..(i + 1) * DIM];
        let result = index
            .search(
                q,
                &SearchRequest {
                    top_k: TOP_K,
                    nprobe: 16,
                    ..Default::default()
                },
            )
            .expect("search rabitq");
        all_results.push(result.ids);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, gt, TOP_K);

    make_row("RaBitQ", qps, recall_at_10, "flat_exact_l2_bruteforce")
}

fn generate_sparse_vectors(n: usize, dim: usize) -> Vec<SparseVector> {
    let mut rng = rand::thread_rng();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut dense = vec![0.0f32; dim];
        for v in &mut dense {
            let x = rng.r#gen::<f32>();
            *v = if x > 0.78 { x } else { 0.0 };
        }
        out.push(SparseVector::from_dense(&dense, 1e-8));
    }
    out
}

fn sparse_dot(a: &SparseVector, b: &SparseVector) -> f32 {
    a.dot(b)
}

fn compute_sparse_ground_truth(base: &[SparseVector], queries: &[SparseVector], k: usize) -> Vec<Vec<i32>> {
    let mut gt = Vec::with_capacity(queries.len());
    for q in queries {
        let mut pairs: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .map(|(idx, b)| (idx, sparse_dot(q, b)))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("sparse score compare"));
        gt.push(pairs.into_iter().take(k).map(|(idx, _)| idx as i32).collect());
    }
    gt
}

fn bench_sparse_wand() -> BenchmarkReportRow {
    let base = generate_sparse_vectors(BASE_SIZE, DIM);
    let queries = generate_sparse_vectors(QUERY_SIZE, DIM);
    let gt = compute_sparse_ground_truth(&base, &queries, TOP_K);

    let mut index = SparseWandIndex::new(SparseMetricType::Ip);
    for (doc_id, vector) in base.iter().enumerate() {
        index
            .add(vector, doc_id as i64)
            .expect("add sparse vector to sparse wand");
    }

    let mut all_results = Vec::with_capacity(QUERY_SIZE);
    let start = Instant::now();
    for q in &queries {
        let result = index.search(q, TOP_K, None);
        all_results.push(result.into_iter().map(|(id, _)| id).collect());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = QUERY_SIZE as f64 / elapsed;
    let recall_at_10 = average_recall_at_k(&all_results, &gt, TOP_K);

    make_row("SparseWand", qps, recall_at_10, "sparse_exact_ip_bruteforce")
}

pub fn build_recall_gated_baseline_report() -> BenchmarkReport {
    let base = generate_vectors(BASE_SIZE, DIM);
    let queries = generate_vectors(QUERY_SIZE, DIM);

    let config = IndexConfig {
        index_type: IndexType::Flat,
        dim: DIM,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams::default(),
    };
    let mut flat = FlatIndex::new(&config).expect("create flat");
    flat.add(&base, None).expect("add flat");

    let gt = compute_ground_truth(&base, &queries, DIM, TOP_K);

    let rows = vec![
        bench_hnsw(&base, &queries, &gt),
        bench_ivf(&base, &queries, &gt),
        bench_diskann(&base, &queries, &gt),
        bench_scann(&base, &queries, &gt),
        bench_rabitq(&base, &queries, &gt),
        bench_sparse_wand(),
    ];

    BenchmarkReport {
        benchmark: "BENCH-P1-001-recall-gated-baseline".to_string(),
        dataset: "synthetic-random-l2".to_string(),
        base_size: BASE_SIZE,
        query_size: QUERY_SIZE,
        dim: DIM,
        recall_gate: RECALL_GATE,
        rows,
    }
}

pub fn generate_recall_gated_baseline_report(path: &str) -> Result<BenchmarkReport, Box<dyn Error>> {
    let report = build_recall_gated_baseline_report();
    write_report(path, &report)?;
    Ok(report)
}

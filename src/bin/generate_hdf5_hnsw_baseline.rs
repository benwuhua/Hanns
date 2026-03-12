#[cfg(feature = "hdf5")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, SearchRequest};
#[cfg(feature = "hdf5")]
use knowhere_rs::benchmark::{average_recall_at_k, confidence_from_recall};
#[cfg(feature = "hdf5")]
use knowhere_rs::dataset::load_hdf5_dataset;
#[cfg(feature = "hdf5")]
use knowhere_rs::faiss::HnswIndex;
#[cfg(feature = "hdf5")]
use knowhere_rs::MetricType;
#[cfg(feature = "hdf5")]
use serde::Serialize;
#[cfg(feature = "hdf5")]
use std::env;
#[cfg(feature = "hdf5")]
use std::fs;
#[cfg(feature = "hdf5")]
use std::path::Path;
#[cfg(feature = "hdf5")]
use std::time::Instant;

#[cfg(feature = "hdf5")]
const DEFAULT_RECALL_GATE: f64 = 0.80;
#[cfg(feature = "hdf5")]
const DEFAULT_TOP_K: usize = 100;
#[cfg(feature = "hdf5")]
const DEFAULT_RECALL_AT: usize = 10;
#[cfg(feature = "hdf5")]
const DEFAULT_EF_SEARCH: usize = 138;
#[cfg(feature = "hdf5")]
const DEFAULT_M: usize = 16;
#[cfg(feature = "hdf5")]
const DEFAULT_EF_CONSTRUCTION: usize = 100;
#[cfg(feature = "hdf5")]
const DEFAULT_QUERY_LIMIT: usize = 1000;
#[cfg(feature = "hdf5")]
const DEFAULT_BASE_LIMIT: usize = 100000;
#[cfg(feature = "hdf5")]
const DEFAULT_OUTPUT_PATH: &str = "benchmark_results/rs_hnsw_sift128.local.json";

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsBaselineRow {
    dataset: String,
    index: String,
    params: String,
    thread_num: u32,
    qps: f64,
    recall_at_10: f64,
    ground_truth_source: String,
    confidence: String,
    confidence_explanation: String,
    runtime_seconds: f64,
    source: String,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsBaselineReport {
    benchmark: String,
    dataset: String,
    methodology: String,
    rows: Vec<RsBaselineRow>,
}

#[cfg(feature = "hdf5")]
fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

#[cfg(feature = "hdf5")]
fn parse_usize_arg(args: &[String], name: &str, default: usize) -> usize {
    arg_value(args, name)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "hdf5")]
fn parse_f64_arg(args: &[String], name: &str, default: f64) -> f64 {
    arg_value(args, name)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

#[cfg(feature = "hdf5")]
fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} --input <path> [--output <path>] [--base-limit <n>] [--query-limit <n>] [--top-k <n>] [--recall-at <n>] [--m <n>] [--ef-construction <n>] [--ef-search <n>] [--recall-gate <f>] [--with-repair]"
    );
    eprintln!("  --top-k      number of results to retrieve per query (default: {DEFAULT_TOP_K})");
    eprintln!("  --recall-at  k for recall@k measurement, independent of top-k (default: {DEFAULT_RECALL_AT})");
    eprintln!(
        "               Set --recall-at 10 to match native benchmark methodology (recall@10)"
    );
}

#[cfg(feature = "hdf5")]
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(feature = "hdf5")]
fn compute_ground_truth(base: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut gt = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let mut pairs: Vec<(usize, f32)> = (0..num_base)
            .map(|j| {
                let base_vec = &base[j * dim..(j + 1) * dim];
                (j, l2_distance_squared(query, base_vec))
            })
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("distance compare"));
        gt.push(
            pairs
                .into_iter()
                .take(k)
                .map(|(idx, _)| idx as i32)
                .collect(),
        );
    }

    gt
}

#[cfg(feature = "hdf5")]
fn run() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage(&args[0]);
        return;
    }

    let input_path = match arg_value(&args, "--input") {
        Some(path) => path,
        None => {
            print_usage(&args[0]);
            std::process::exit(2);
        }
    };
    let output_path =
        arg_value(&args, "--output").unwrap_or_else(|| DEFAULT_OUTPUT_PATH.to_string());
    let base_limit = parse_usize_arg(&args, "--base-limit", DEFAULT_BASE_LIMIT);
    let query_limit = parse_usize_arg(&args, "--query-limit", DEFAULT_QUERY_LIMIT);
    let top_k = parse_usize_arg(&args, "--top-k", DEFAULT_TOP_K);
    // recall_at controls the k for recall@k measurement, independent of search top_k.
    // Default 10 matches native benchmark methodology (recall@10 = fraction of true 10-NN found).
    let recall_at = parse_usize_arg(&args, "--recall-at", DEFAULT_RECALL_AT);
    let m = parse_usize_arg(&args, "--m", DEFAULT_M);
    let ef_construction = parse_usize_arg(&args, "--ef-construction", DEFAULT_EF_CONSTRUCTION);
    let ef_search = parse_usize_arg(&args, "--ef-search", DEFAULT_EF_SEARCH);
    let recall_gate = parse_f64_arg(&args, "--recall-gate", DEFAULT_RECALL_GATE);
    let with_repair = args.iter().any(|arg| arg == "--with-repair");

    let dataset = match load_hdf5_dataset(&input_path) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("failed to load HDF5 dataset {input_path}: {err}");
            std::process::exit(1);
        }
    };

    let base_count = base_limit.min(dataset.num_train());
    let query_count = query_limit.min(dataset.num_test());
    let dim = dataset.dim();
    let base_vectors = &dataset.train.vectors()[..base_count * dim];
    let test_vectors = dataset.test.vectors();
    let query_vectors = &test_vectors[..query_count * dim];
    let uses_full_dataset_ground_truth = base_count == dataset.num_train();
    // Ground truth is prepared at recall_at granularity (not top_k).
    // This matches native benchmark methodology: recall@10 checks whether the true
    // 10-NN are present among the returned top_k results.
    let (ground_truth, ground_truth_source) = if uses_full_dataset_ground_truth {
        (
            dataset.neighbors[..query_count]
                .iter()
                .map(|row| row.iter().copied().take(recall_at).collect::<Vec<i32>>())
                .collect::<Vec<Vec<i32>>>(),
            "ann_benchmarks_hdf5_neighbors".to_string(),
        )
    } else {
        (
            compute_ground_truth(base_vectors, query_vectors, dim, recall_at),
            format!("flat_exact_l2_bruteforce(base_limit={base_count},recall_at={recall_at})"),
        )
    };

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create hnsw");
    if with_repair {
        index
            .build_with_repair(base_vectors, None)
            .expect("build hnsw with repair");
    } else {
        index.train(base_vectors).expect("train hnsw");
        index.add(base_vectors, None).expect("add hnsw");
    }

    let mut all_results: Vec<Vec<i64>> = Vec::with_capacity(query_count);
    let start = Instant::now();
    for i in 0..query_count {
        let query = &query_vectors[i * dim..(i + 1) * dim];
        let request = SearchRequest {
            top_k,
            nprobe: ef_search,
            ..Default::default()
        };
        let result = index.search(query, &request).expect("search hnsw");
        all_results.push(result.ids);
    }
    let runtime_seconds = start.elapsed().as_secs_f64();
    let qps = query_count as f64 / runtime_seconds;
    // Measure recall at recall_at (default 10), independent of search top_k.
    // Matches native: "are the true recall_at-NN among the returned top_k results?"
    let recall_value = average_recall_at_k(&all_results, &ground_truth, recall_at);
    let confidence = confidence_from_recall(recall_value, recall_gate).to_string();
    let confidence_explanation = if confidence == "trusted" {
        "".to_string()
    } else {
        format!(
            "Rust HNSW on ann-benchmarks HDF5 fixture; recall@{recall_at}={:.4} below gate {:.2}.",
            recall_value, recall_gate
        )
    };

    let dataset_name = Path::new(&input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-dataset")
        .to_string();

    let report = RsBaselineReport {
        benchmark: "BASELINE-P3-001-rs-hnsw-hdf5".to_string(),
        dataset: dataset_name.clone(),
        methodology: "rust_hdf5_ann_benchmark".to_string(),
        rows: vec![RsBaselineRow {
            dataset: dataset_name,
            index: "HNSW(Rust)".to_string(),
            params: format!(
                "M={} | efConstruction={}, ef={}, top_k={}, recall_at={}",
                m, ef_construction, ef_search, top_k, recall_at
            ),
            thread_num: rayon::current_num_threads() as u32,
            qps,
            recall_at_10: recall_value,
            ground_truth_source,
            confidence,
            confidence_explanation,
            runtime_seconds,
            source: "knowhere_rs_hnsw_hdf5".to_string(),
        }],
    };

    let output = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    fs::write(&output_path, output).expect("write output");
    println!("wrote {}", output_path);
}

fn main() {
    #[cfg(feature = "hdf5")]
    run();

    #[cfg(not(feature = "hdf5"))]
    eprintln!("Error: this binary requires the 'hdf5' feature. Rebuild with --features hdf5");
}

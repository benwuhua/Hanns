#[cfg(feature = "hdf5")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, SearchRequest};
#[cfg(feature = "hdf5")]
use knowhere_rs::benchmark::{average_recall_at_k, confidence_from_recall};
#[cfg(feature = "hdf5")]
use knowhere_rs::bitset::BitsetView;
#[cfg(feature = "hdf5")]
use knowhere_rs::dataset::load_hdf5_dataset;
#[cfg(feature = "hdf5")]
use knowhere_rs::faiss::hnsw::HnswCandidateSearchProfileReport;
#[cfg(feature = "hdf5")]
use knowhere_rs::faiss::HnswIndex;
#[cfg(feature = "hdf5")]
use knowhere_rs::MetricType;
#[cfg(feature = "hdf5")]
use rayon::prelude::*;
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
const DEFAULT_HNSW_ADAPTIVE_K: f64 = 2.0;
#[cfg(feature = "hdf5")]
const DEFAULT_QUERY_BATCH_SIZE: usize = 32;
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
    requested_ef_search: usize,
    effective_ef_search: usize,
    adaptive_k: f64,
    requested_vector_datatype: String,
    vector_datatype: String,
    query_dispatch_model: String,
    query_batch_size: usize,
    qps: f64,
    qps_runs: Vec<f64>,
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
#[derive(Debug, Serialize)]
struct RsSearchCostDiagnosisRow {
    ef_search: usize,
    qps: f64,
    qps_runs: Vec<f64>,
    recall_at_10: f64,
    average_visited_nodes: f64,
    average_frontier_pushes: f64,
    average_frontier_pops: f64,
    average_distance_calls: f64,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsSearchCostDiagnosisReport {
    benchmark: String,
    dataset: String,
    methodology: String,
    build_random_seed: Option<u64>,
    qps_repeat_count: usize,
    selected_recall_gate: f64,
    ef_sweep: Vec<RsSearchCostDiagnosisRow>,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsBitsetSearchCostDiagnosisReport {
    benchmark: String,
    dataset: String,
    methodology: String,
    build_random_seed: Option<u64>,
    qps_repeat_count: usize,
    bitset_stride: usize,
    filtered_fraction: f64,
    selected_recall_gate: f64,
    ef_sweep: Vec<RsSearchCostDiagnosisRow>,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsCandidateSearchProfileArtifact {
    benchmark: String,
    dataset: String,
    methodology: String,
    build_random_seed: Option<u64>,
    query_sample_size: usize,
    ef_search: usize,
    top_k: usize,
    requested_vector_datatype: String,
    vector_datatype: String,
    query_dispatch_model: String,
    query_batch_size: usize,
    production_layer0_l2_search_mode: String,
    profiled_layer0_l2_search_mode: String,
    production_layer0_layout_mode: String,
    profiled_layer0_layout_mode: String,
    production_layer0_avoids_profile_timing: bool,
    sampled_search_cost_summary: RsSampledSearchCostSummary,
    profile: HnswCandidateSearchProfileReport,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Serialize)]
struct RsSampledSearchCostSummary {
    query_sample_size: usize,
    average_distance_calls: f64,
    p50_distance_calls: usize,
    p95_distance_calls: usize,
    p99_distance_calls: usize,
    average_visited_nodes: f64,
    p95_visited_nodes: usize,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryDispatchMode {
    Serial,
    Parallel,
}

#[cfg(feature = "hdf5")]
impl QueryDispatchMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "serial" => Ok(Self::Serial),
            "parallel" => Ok(Self::Parallel),
            _ => Err(format!(
                "unsupported query dispatch mode `{value}`; expected `serial` or `parallel`"
            )),
        }
    }
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDispatchConfig {
    mode: QueryDispatchMode,
    requested_batch_size: usize,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDispatchMetadata {
    model: &'static str,
    batch_size: usize,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VectorDatatypeMetadata {
    requested_label: &'static str,
    effective_label: &'static str,
}

#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedVectorDatatype {
    Float32,
    BFloat16,
}

#[cfg(feature = "hdf5")]
impl RequestedVectorDatatype {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "float32" | "float" => Ok(Self::Float32),
            "bfloat16" | "bf16" => Ok(Self::BFloat16),
            _ => Err(format!(
                "unsupported vector datatype `{value}`; expected `float32` or `bfloat16`"
            )),
        }
    }

    fn index_data_type(self) -> DataType {
        match self {
            Self::Float32 => DataType::Float,
            Self::BFloat16 => DataType::BFloat16,
        }
    }

    fn requested_label(self) -> &'static str {
        match self {
            Self::Float32 => "Float32",
            Self::BFloat16 => "BFloat16",
        }
    }

    fn metadata_for_current_hnsw_lane(self) -> VectorDatatypeMetadata {
        // HNSW now supports a real BFloat16 storage + L2 distance lane.
        VectorDatatypeMetadata {
            requested_label: self.requested_label(),
            effective_label: self.requested_label(),
        }
    }
}

#[cfg(feature = "hdf5")]
impl QueryDispatchConfig {
    fn serial() -> Self {
        Self {
            mode: QueryDispatchMode::Serial,
            requested_batch_size: 1,
        }
    }

    fn parallel(batch_size: usize) -> Self {
        Self {
            mode: QueryDispatchMode::Parallel,
            requested_batch_size: batch_size.max(2),
        }
    }

    fn metadata_for_query_count(&self, query_count: usize) -> QueryDispatchMetadata {
        match self.mode {
            QueryDispatchMode::Serial => QueryDispatchMetadata {
                model: "serial_per_query_index_search",
                batch_size: 1,
            },
            QueryDispatchMode::Parallel => QueryDispatchMetadata {
                model: "rayon_query_batch_parallel_search",
                batch_size: self.effective_batch_size(query_count),
            },
        }
    }

    fn effective_batch_size(&self, query_count: usize) -> usize {
        match self.mode {
            QueryDispatchMode::Serial => 1,
            QueryDispatchMode::Parallel => {
                if query_count <= 1 {
                    1
                } else {
                    self.requested_batch_size.min(query_count).max(2)
                }
            }
        }
    }
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
fn parse_u64_arg(args: &[String], name: &str) -> Option<u64> {
    arg_value(args, name).and_then(|v| v.parse::<u64>().ok())
}

#[cfg(feature = "hdf5")]
fn parse_usize_list_arg(args: &[String], name: &str) -> Vec<usize> {
    arg_value(args, name)
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(feature = "hdf5")]
fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} --input <path> [--output <path>] [--base-limit <n>] [--query-limit <n>] [--top-k <n>] [--recall-at <n>] [--m <n>] [--ef-construction <n>] [--ef-search <n>] [--hnsw-adaptive-k <f>] [--vector-datatype <float32|bfloat16>] [--query-dispatch-mode <serial|parallel>] [--query-batch-size <n>] [--recall-gate <f>] [--with-repair]"
    );
    eprintln!("  --top-k      number of results to retrieve per query (default: {DEFAULT_TOP_K})");
    eprintln!("  --recall-at  k for recall@k measurement, independent of top-k (default: {DEFAULT_RECALL_AT})");
    eprintln!(
        "               Set --recall-at 10 to match native benchmark methodology (recall@10)"
    );
    eprintln!("  --random-seed <u64>        optional deterministic HNSW build seed");
    eprintln!(
        "  --hnsw-adaptive-k <f>     HNSW adaptive ef multiplier (0 disables the adaptive floor)"
    );
    eprintln!(
        "  --vector-datatype <float32|bfloat16>  requested datatype for the HNSW fair-lane audit"
    );
    eprintln!(
        "  --query-dispatch-mode <serial|parallel>  query execution mode for the benchmark lane"
    );
    eprintln!(
        "  --query-batch-size <n>    query chunk size used when --query-dispatch-mode parallel"
    );
    eprintln!("  --repeat <n>               repeat each query batch and report median qps");
    eprintln!("  --diagnosis-output <path>  optional JSON output for search-cost diagnosis");
    eprintln!(
        "  --candidate-profile-output <path>  optional JSON output for candidate-search profile"
    );
    eprintln!(
        "  --candidate-profile-query-limit <n>  max queries sampled for candidate profile (default: 128)"
    );
    eprintln!(
        "  --bitset-diagnosis-output <path>  optional JSON output for bitset search-cost diagnosis"
    );
    eprintln!("  --bitset-step <n>          mask every n-th base vector in bitset diagnosis mode");
    eprintln!("  --ef-sweep <csv>           comma-separated ef values for diagnosis mode");
}

#[cfg(feature = "hdf5")]
fn build_hnsw_config(
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    hnsw_adaptive_k: f64,
    requested_vector_datatype: RequestedVectorDatatype,
    random_seed: Option<u64>,
) -> IndexConfig {
    IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim,
        data_type: requested_vector_datatype.index_data_type(),
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            hnsw_adaptive_k: Some(hnsw_adaptive_k),
            random_seed,
            ..Default::default()
        },
    }
}

#[cfg(feature = "hdf5")]
fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(feature = "hdf5")]
fn percentile_usize(values: &[usize], percentile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let clamped = percentile.clamp(0.0, 100.0);
    let rank = ((clamped / 100.0) * (sorted.len().saturating_sub(1) as f64)).round() as usize;
    sorted[rank]
}

#[cfg(feature = "hdf5")]
fn run_query_batch(
    index: &HnswIndex,
    query_vectors: &[f32],
    query_count: usize,
    dim: usize,
    top_k: usize,
    ef_search: usize,
    dispatch: &QueryDispatchConfig,
) -> (Vec<Vec<i64>>, f64) {
    let start = Instant::now();
    let all_results: Vec<Vec<i64>> = match dispatch.mode {
        QueryDispatchMode::Serial => (0..query_count)
            .map(|i| {
                let query = &query_vectors[i * dim..(i + 1) * dim];
                let request = SearchRequest {
                    top_k,
                    nprobe: ef_search,
                    ..Default::default()
                };
                index.search(query, &request).expect("search hnsw").ids
            })
            .collect(),
        QueryDispatchMode::Parallel => {
            let batch_len = dispatch.effective_batch_size(query_count) * dim;
            let chunked_results: Vec<Vec<Vec<i64>>> = query_vectors
                .par_chunks(batch_len)
                .map(|query_chunk| {
                    query_chunk
                        .chunks_exact(dim)
                        .map(|query| {
                            let request = SearchRequest {
                                top_k,
                                nprobe: ef_search,
                                ..Default::default()
                            };
                            index.search(query, &request).expect("search hnsw").ids
                        })
                        .collect::<Vec<Vec<i64>>>()
                })
                .collect();

            chunked_results.into_iter().flatten().collect()
        }
    };

    (all_results, start.elapsed().as_secs_f64())
}

#[cfg(feature = "hdf5")]
fn run_query_batch_repeated(
    index: &HnswIndex,
    query_vectors: &[f32],
    query_count: usize,
    dim: usize,
    top_k: usize,
    ef_search: usize,
    repeat: usize,
    dispatch: &QueryDispatchConfig,
) -> (Vec<Vec<i64>>, Vec<f64>) {
    let mut qps_runs = Vec::with_capacity(repeat);
    let mut first_results = Vec::new();

    for run_idx in 0..repeat {
        let (results, runtime_seconds) = run_query_batch(
            index,
            query_vectors,
            query_count,
            dim,
            top_k,
            ef_search,
            dispatch,
        );
        if run_idx == 0 {
            first_results = results;
        }
        qps_runs.push(query_count as f64 / runtime_seconds);
    }

    (first_results, qps_runs)
}

#[cfg(feature = "hdf5")]
fn run_query_batch_with_bitset(
    index: &HnswIndex,
    query_vectors: &[f32],
    query_count: usize,
    dim: usize,
    top_k: usize,
    ef_search: usize,
    bitset: &BitsetView,
    dispatch: &QueryDispatchConfig,
) -> (Vec<Vec<i64>>, f64) {
    let start = Instant::now();
    let all_results: Vec<Vec<i64>> = match dispatch.mode {
        QueryDispatchMode::Serial => (0..query_count)
            .map(|i| {
                let query = &query_vectors[i * dim..(i + 1) * dim];
                let request = SearchRequest {
                    top_k,
                    nprobe: ef_search,
                    ..Default::default()
                };
                index
                    .search_with_bitset(query, &request, bitset)
                    .expect("bitset search hnsw")
                    .ids
            })
            .collect(),
        QueryDispatchMode::Parallel => {
            let batch_len = dispatch.effective_batch_size(query_count) * dim;
            let chunked_results: Vec<Vec<Vec<i64>>> = query_vectors
                .par_chunks(batch_len)
                .map(|query_chunk| {
                    query_chunk
                        .chunks_exact(dim)
                        .map(|query| {
                            let request = SearchRequest {
                                top_k,
                                nprobe: ef_search,
                                ..Default::default()
                            };
                            index
                                .search_with_bitset(query, &request, bitset)
                                .expect("bitset search hnsw")
                                .ids
                        })
                        .collect::<Vec<Vec<i64>>>()
                })
                .collect();

            chunked_results.into_iter().flatten().collect()
        }
    };

    (all_results, start.elapsed().as_secs_f64())
}

#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
fn run_query_batch_with_bitset_repeated(
    index: &HnswIndex,
    query_vectors: &[f32],
    query_count: usize,
    dim: usize,
    top_k: usize,
    ef_search: usize,
    bitset: &BitsetView,
    repeat: usize,
    dispatch: &QueryDispatchConfig,
) -> (Vec<Vec<i64>>, Vec<f64>) {
    let mut qps_runs = Vec::with_capacity(repeat);
    let mut first_results = Vec::new();

    for run_idx in 0..repeat {
        let (results, runtime_seconds) = run_query_batch_with_bitset(
            index,
            query_vectors,
            query_count,
            dim,
            top_k,
            ef_search,
            bitset,
            dispatch,
        );
        if run_idx == 0 {
            first_results = results;
        }
        qps_runs.push(query_count as f64 / runtime_seconds);
    }

    (first_results, qps_runs)
}

#[cfg(feature = "hdf5")]
fn build_stride_bitset(base_count: usize, bitset_step: usize) -> BitsetView {
    let mut bitset = BitsetView::new(base_count);
    for idx in (0..base_count).step_by(bitset_step) {
        bitset.set(idx, true);
    }
    bitset
}

#[cfg(feature = "hdf5")]
fn filtered_fraction(bitset: &BitsetView, base_count: usize) -> f64 {
    let filtered = (0..base_count)
        .filter(|&idx| idx < bitset.len() && bitset.get(idx))
        .count();
    filtered as f64 / base_count.max(1) as f64
}

#[cfg(feature = "hdf5")]
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(feature = "hdf5")]
fn compute_ground_truth(
    base: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
    bitset: Option<&BitsetView>,
) -> Vec<Vec<i32>> {
    let num_base = base.len() / dim;
    let num_queries = queries.len() / dim;
    let mut gt = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let query = &queries[i * dim..(i + 1) * dim];
        let mut pairs: Vec<(usize, f32)> = (0..num_base)
            .filter(|&j| !bitset.is_some_and(|mask| j < mask.len() && mask.get(j)))
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
    let hnsw_adaptive_k = parse_f64_arg(&args, "--hnsw-adaptive-k", DEFAULT_HNSW_ADAPTIVE_K);
    let requested_vector_datatype = arg_value(&args, "--vector-datatype")
        .map(|value| {
            RequestedVectorDatatype::parse(&value).unwrap_or_else(|message| {
                eprintln!("{message}");
                std::process::exit(2);
            })
        })
        .unwrap_or(RequestedVectorDatatype::Float32);
    let query_dispatch_mode = arg_value(&args, "--query-dispatch-mode")
        .map(|value| {
            QueryDispatchMode::parse(&value).unwrap_or_else(|message| {
                eprintln!("{message}");
                std::process::exit(2);
            })
        })
        .unwrap_or(QueryDispatchMode::Serial);
    let query_batch_size = parse_usize_arg(&args, "--query-batch-size", DEFAULT_QUERY_BATCH_SIZE);
    let recall_gate = parse_f64_arg(&args, "--recall-gate", DEFAULT_RECALL_GATE);
    let random_seed = parse_u64_arg(&args, "--random-seed");
    let repeat = parse_usize_arg(&args, "--repeat", 1).max(1);
    let diagnosis_output = arg_value(&args, "--diagnosis-output");
    let candidate_profile_output = arg_value(&args, "--candidate-profile-output");
    let candidate_profile_query_limit =
        parse_usize_arg(&args, "--candidate-profile-query-limit", 128).max(1);
    let bitset_diagnosis_output = arg_value(&args, "--bitset-diagnosis-output");
    let bitset_step = parse_usize_arg(&args, "--bitset-step", 0);
    let ef_sweep = parse_usize_list_arg(&args, "--ef-sweep");
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
            compute_ground_truth(base_vectors, query_vectors, dim, recall_at, None),
            format!("flat_exact_l2_bruteforce(base_limit={base_count},recall_at={recall_at})"),
        )
    };

    let config = build_hnsw_config(
        dim,
        m,
        ef_construction,
        ef_search,
        hnsw_adaptive_k,
        requested_vector_datatype,
        random_seed,
    );
    let adaptive_k = config.params.hnsw_adaptive_k();
    let effective_ef_search = config
        .params
        .effective_hnsw_ef_search(ef_search, ef_search, top_k);
    let vector_datatype_metadata = requested_vector_datatype.metadata_for_current_hnsw_lane();
    let query_dispatch = match query_dispatch_mode {
        QueryDispatchMode::Serial => QueryDispatchConfig::serial(),
        QueryDispatchMode::Parallel => QueryDispatchConfig::parallel(query_batch_size),
    };
    let query_dispatch_metadata = query_dispatch.metadata_for_query_count(query_count);

    let mut index = HnswIndex::new(&config).expect("create hnsw");
    if with_repair {
        index
            .build_with_repair(base_vectors, None)
            .expect("build hnsw with repair");
    } else {
        index.train(base_vectors).expect("train hnsw");
        index.add(base_vectors, None).expect("add hnsw");
    }

    let (all_results, qps_runs) = run_query_batch_repeated(
        &index,
        query_vectors,
        query_count,
        dim,
        top_k,
        ef_search,
        repeat,
        &query_dispatch,
    );
    let qps = median_f64(&qps_runs);
    let runtime_seconds = query_count as f64 / qps;
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
            dataset: dataset_name.clone(),
            index: "HNSW(Rust)".to_string(),
            params: format!(
                "M={} | efConstruction={}, ef={}, top_k={}, recall_at={}{} | repeat={repeat}",
                m,
                ef_construction,
                ef_search,
                top_k,
                recall_at,
                random_seed
                    .map(|seed| format!(" | seed={seed}"))
                    .unwrap_or_default()
            ),
            thread_num: rayon::current_num_threads() as u32,
            requested_ef_search: ef_search,
            effective_ef_search,
            adaptive_k,
            requested_vector_datatype: vector_datatype_metadata.requested_label.to_string(),
            vector_datatype: vector_datatype_metadata.effective_label.to_string(),
            query_dispatch_model: query_dispatch_metadata.model.to_string(),
            query_batch_size: query_dispatch_metadata.batch_size,
            qps,
            qps_runs,
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

    if let Some(candidate_profile_output) = candidate_profile_output {
        let sampled_query_count = query_count.min(candidate_profile_query_limit);
        let sampled_queries = &query_vectors[..sampled_query_count * dim];
        let mut sampled_distance_calls = Vec::with_capacity(sampled_query_count);
        let mut sampled_visited_nodes = Vec::with_capacity(sampled_query_count);
        for i in 0..sampled_query_count {
            let query = &sampled_queries[i * dim..(i + 1) * dim];
            let diagnosis = index.search_cost_diagnosis(query, ef_search, top_k);
            sampled_distance_calls.push(diagnosis.distance_calls);
            sampled_visited_nodes.push(diagnosis.visited_nodes);
        }
        let sampled_search_cost_summary = RsSampledSearchCostSummary {
            query_sample_size: sampled_query_count,
            average_distance_calls: sampled_distance_calls.iter().sum::<usize>() as f64
                / sampled_query_count.max(1) as f64,
            p50_distance_calls: percentile_usize(&sampled_distance_calls, 50.0),
            p95_distance_calls: percentile_usize(&sampled_distance_calls, 95.0),
            p99_distance_calls: percentile_usize(&sampled_distance_calls, 99.0),
            average_visited_nodes: sampled_visited_nodes.iter().sum::<usize>() as f64
                / sampled_query_count.max(1) as f64,
            p95_visited_nodes: percentile_usize(&sampled_visited_nodes, 95.0),
        };
        let profile = index
            .candidate_search_profile_report(sampled_queries, ef_search, top_k)
            .expect("generate candidate-search profile");
        let profile_artifact = RsCandidateSearchProfileArtifact {
            benchmark: "HNSW-FAIR-LANE-candidate-search-profile".to_string(),
            dataset: dataset_name.clone(),
            methodology: "rust_hdf5_ann_candidate_search_profile".to_string(),
            build_random_seed: random_seed,
            query_sample_size: sampled_query_count,
            ef_search,
            top_k,
            requested_vector_datatype: vector_datatype_metadata.requested_label.to_string(),
            vector_datatype: vector_datatype_metadata.effective_label.to_string(),
            query_dispatch_model: query_dispatch_metadata.model.to_string(),
            query_batch_size: query_dispatch_metadata.batch_size,
            production_layer0_l2_search_mode: index
                .layer0_l2_search_mode_for_audit(false)
                .to_string(),
            profiled_layer0_l2_search_mode: index.layer0_l2_search_mode_for_audit(true).to_string(),
            production_layer0_layout_mode: index
                .production_layer0_layout_mode_for_audit()
                .to_string(),
            profiled_layer0_layout_mode: index.profiled_layer0_layout_mode_for_audit().to_string(),
            production_layer0_avoids_profile_timing: index
                .production_layer0_avoids_profile_timing_for_audit(),
            sampled_search_cost_summary,
            profile,
        };
        let profile_output_json = serde_json::to_string_pretty(&profile_artifact)
            .expect("serialize candidate-search profile artifact");
        if let Some(parent) = Path::new(&candidate_profile_output).parent() {
            fs::create_dir_all(parent).expect("create candidate profile output dir");
        }
        fs::write(&candidate_profile_output, profile_output_json)
            .expect("write candidate profile output");
        println!("wrote {}", candidate_profile_output);
    }

    if let Some(diagnosis_output) = diagnosis_output {
        let diagnosis_efs = if ef_sweep.is_empty() {
            vec![ef_search]
        } else {
            ef_sweep.clone()
        };

        let mut ef_rows = Vec::with_capacity(diagnosis_efs.len());
        for diagnosis_ef in diagnosis_efs {
            let (sweep_results, sweep_qps_runs) = run_query_batch_repeated(
                &index,
                query_vectors,
                query_count,
                dim,
                top_k,
                diagnosis_ef,
                repeat,
                &query_dispatch,
            );
            let sweep_qps = median_f64(&sweep_qps_runs);
            let sweep_recall = average_recall_at_k(&sweep_results, &ground_truth, recall_at);

            let mut visited_sum = 0usize;
            let mut frontier_push_sum = 0usize;
            let mut frontier_pop_sum = 0usize;
            let mut distance_call_sum = 0usize;

            for i in 0..query_count {
                let query = &query_vectors[i * dim..(i + 1) * dim];
                let diagnosis = index.search_cost_diagnosis(query, diagnosis_ef, top_k);
                visited_sum += diagnosis.visited_nodes;
                frontier_push_sum += diagnosis.frontier_pushes;
                frontier_pop_sum += diagnosis.frontier_pops;
                distance_call_sum += diagnosis.distance_calls;
            }

            let denom = query_count.max(1) as f64;
            ef_rows.push(RsSearchCostDiagnosisRow {
                ef_search: diagnosis_ef,
                qps: sweep_qps,
                qps_runs: sweep_qps_runs,
                recall_at_10: sweep_recall,
                average_visited_nodes: visited_sum as f64 / denom,
                average_frontier_pushes: frontier_push_sum as f64 / denom,
                average_frontier_pops: frontier_pop_sum as f64 / denom,
                average_distance_calls: distance_call_sum as f64 / denom,
            });
        }

        let diagnosis_report = RsSearchCostDiagnosisReport {
            benchmark: "HNSW-CORE-REWRITE-STAGE1-search-cost-diagnosis".to_string(),
            dataset: dataset_name.clone(),
            methodology: "rust_hdf5_ann_benchmark_search_cost".to_string(),
            build_random_seed: random_seed,
            qps_repeat_count: repeat,
            selected_recall_gate: recall_gate,
            ef_sweep: ef_rows,
        };

        let diagnosis_output_json =
            serde_json::to_string_pretty(&diagnosis_report).expect("serialize diagnosis report");
        if let Some(parent) = Path::new(&diagnosis_output).parent() {
            fs::create_dir_all(parent).expect("create diagnosis output dir");
        }
        fs::write(&diagnosis_output, diagnosis_output_json).expect("write diagnosis output");
        println!("wrote {}", diagnosis_output);
    }

    if let Some(bitset_diagnosis_output) = bitset_diagnosis_output {
        assert!(
            bitset_step >= 2,
            "--bitset-step must be at least 2 when --bitset-diagnosis-output is provided"
        );

        let diagnosis_efs = if ef_sweep.is_empty() {
            vec![ef_search]
        } else {
            ef_sweep.clone()
        };

        let bitset = build_stride_bitset(base_count, bitset_step);
        let filtered_fraction = filtered_fraction(&bitset, base_count);
        let bitset_ground_truth =
            compute_ground_truth(base_vectors, query_vectors, dim, recall_at, Some(&bitset));

        let mut ef_rows = Vec::with_capacity(diagnosis_efs.len());
        for diagnosis_ef in diagnosis_efs {
            let (sweep_results, sweep_qps_runs) = run_query_batch_with_bitset_repeated(
                &index,
                query_vectors,
                query_count,
                dim,
                top_k,
                diagnosis_ef,
                &bitset,
                repeat,
                &query_dispatch,
            );
            let sweep_qps = median_f64(&sweep_qps_runs);
            let sweep_recall = average_recall_at_k(&sweep_results, &bitset_ground_truth, recall_at);

            let mut visited_sum = 0usize;
            let mut frontier_push_sum = 0usize;
            let mut frontier_pop_sum = 0usize;
            let mut distance_call_sum = 0usize;

            for i in 0..query_count {
                let query = &query_vectors[i * dim..(i + 1) * dim];
                let diagnosis =
                    index.search_cost_diagnosis_with_bitset(query, diagnosis_ef, top_k, &bitset);
                visited_sum += diagnosis.visited_nodes;
                frontier_push_sum += diagnosis.frontier_pushes;
                frontier_pop_sum += diagnosis.frontier_pops;
                distance_call_sum += diagnosis.distance_calls;
            }

            let denom = query_count.max(1) as f64;
            ef_rows.push(RsSearchCostDiagnosisRow {
                ef_search: diagnosis_ef,
                qps: sweep_qps,
                qps_runs: sweep_qps_runs,
                recall_at_10: sweep_recall,
                average_visited_nodes: visited_sum as f64 / denom,
                average_frontier_pushes: frontier_push_sum as f64 / denom,
                average_frontier_pops: frontier_pop_sum as f64 / denom,
                average_distance_calls: distance_call_sum as f64 / denom,
            });
        }

        let diagnosis_report = RsBitsetSearchCostDiagnosisReport {
            benchmark: "HNSW-SEARCH-FIRST-bitset-search-cost-diagnosis".to_string(),
            dataset: dataset_name.clone(),
            methodology: "rust_hdf5_ann_benchmark_bitset_search_cost".to_string(),
            build_random_seed: random_seed,
            qps_repeat_count: repeat,
            bitset_stride: bitset_step,
            filtered_fraction,
            selected_recall_gate: recall_gate,
            ef_sweep: ef_rows,
        };

        let diagnosis_output_json = serde_json::to_string_pretty(&diagnosis_report)
            .expect("serialize bitset diagnosis report");
        if let Some(parent) = Path::new(&bitset_diagnosis_output).parent() {
            fs::create_dir_all(parent).expect("create bitset diagnosis output dir");
        }
        fs::write(&bitset_diagnosis_output, diagnosis_output_json)
            .expect("write bitset diagnosis output");
        println!("wrote {}", bitset_diagnosis_output);
    }
}

fn main() {
    #[cfg(feature = "hdf5")]
    run();

    #[cfg(not(feature = "hdf5"))]
    eprintln!("Error: this binary requires the 'hdf5' feature. Rebuild with --features hdf5");
}

#[cfg(all(test, feature = "hdf5"))]
mod tests {
    use super::*;

    #[test]
    fn parallel_query_dispatch_matches_serial_results_and_reports_batch_metadata() {
        let dim = 2;
        let base = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let queries = vec![0.1, 0.0, 0.0, 0.9, 1.1, 1.0, 0.0, 1.9];
        let query_count = queries.len() / dim;
        let top_k = 2;
        let ef_search = 8;

        let config = build_hnsw_config(
            dim,
            8,
            40,
            ef_search,
            0.0,
            RequestedVectorDatatype::Float32,
            Some(42),
        );
        let mut index = HnswIndex::new(&config).expect("create hnsw");
        index.train(&base).expect("train hnsw");
        index.add(&base, None).expect("add hnsw");

        let serial_dispatch = QueryDispatchConfig::serial();
        let parallel_dispatch = QueryDispatchConfig::parallel(2);

        let (serial_results, _) = run_query_batch(
            &index,
            &queries,
            query_count,
            dim,
            top_k,
            ef_search,
            &serial_dispatch,
        );
        let (parallel_results, _) = run_query_batch(
            &index,
            &queries,
            query_count,
            dim,
            top_k,
            ef_search,
            &parallel_dispatch,
        );

        assert_eq!(parallel_results, serial_results);

        let metadata = parallel_dispatch.metadata_for_query_count(query_count);
        assert_eq!(metadata.model, "rayon_query_batch_parallel_search");
        assert_eq!(metadata.batch_size, 2);
    }

    #[test]
    fn parallel_query_dispatch_reports_single_query_batch_size_as_one() {
        let dispatch = QueryDispatchConfig::parallel(32);
        let metadata = dispatch.metadata_for_query_count(1);

        assert_eq!(metadata.model, "rayon_query_batch_parallel_search");
        assert_eq!(metadata.batch_size, 1);
    }

    #[test]
    fn requested_bfloat16_reports_bfloat16_effective_lane_when_supported() {
        let requested = RequestedVectorDatatype::parse("bfloat16").expect("parse datatype");
        let metadata = requested.metadata_for_current_hnsw_lane();

        assert_eq!(requested.index_data_type(), DataType::BFloat16);
        assert_eq!(metadata.requested_label, "BFloat16");
        assert_eq!(metadata.effective_label, "BFloat16");
    }

    #[test]
    fn build_hnsw_config_carries_random_seed_and_adaptive_k() {
        let config = build_hnsw_config(
            128,
            16,
            100,
            138,
            0.0,
            RequestedVectorDatatype::Float32,
            Some(42),
        );

        assert_eq!(config.params.random_seed, Some(42));
        assert_eq!(config.params.hnsw_adaptive_k, Some(0.0));
        assert_eq!(config.data_type, DataType::Float);
    }

    #[test]
    fn median_f64_sorts_before_picking_middle_value() {
        let values = vec![9.0, 3.0, 5.0];

        assert_eq!(median_f64(&values), 5.0);
    }

    #[test]
    fn percentile_usize_returns_expected_percentiles() {
        let values = vec![10, 50, 20, 40, 30];

        assert_eq!(percentile_usize(&values, 50.0), 30);
        assert_eq!(percentile_usize(&values, 95.0), 50);
        assert_eq!(percentile_usize(&values, 0.0), 10);
    }
}

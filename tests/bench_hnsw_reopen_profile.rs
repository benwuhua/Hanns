#[cfg(feature = "long-tests")]
use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, SearchRequest};
#[cfg(feature = "long-tests")]
use hanns::faiss::{hnsw::HnswParallelBuildProfileReport, HnswIndex};
#[cfg(feature = "long-tests")]
use hanns::MetricType;
#[cfg(feature = "long-tests")]
use serde_json::json;
#[cfg(feature = "long-tests")]
use std::fs;
#[cfg(feature = "long-tests")]
use std::path::Path;
#[cfg(feature = "long-tests")]
use std::time::Instant;

#[cfg(feature = "long-tests")]
const HNSW_REOPEN_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_baseline.json";
#[cfg(feature = "long-tests")]
const HNSW_REOPEN_PROFILE_PATH: &str = "benchmark_results/hnsw_reopen_profile_round1.json";

#[cfg(feature = "long-tests")]
fn profile_dimensions() -> (usize, usize, usize) {
    (12_000, 128, 128)
}

#[cfg(feature = "long-tests")]
fn generate_profile_base_vectors(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(num_vectors * dim);
    for i in 0..num_vectors {
        let cluster = (i % 64) as f32 * 0.01;
        let band = ((i / 64) % 32) as f32 * 0.001;
        for d in 0..dim {
            let raw = ((i * 131 + d * 17 + (i % 29) * (d % 7)) % 1024) as f32 / 1024.0;
            vectors.push(raw + cluster + band + (d % 11) as f32 * 0.0001);
        }
    }
    vectors
}

#[cfg(feature = "long-tests")]
fn generate_profile_query_vectors(base: &[f32], num_queries: usize, dim: usize) -> Vec<f32> {
    let num_base = base.len() / dim;
    let mut queries = Vec::with_capacity(num_queries * dim);
    for i in 0..num_queries {
        let src = (i * 97) % num_base;
        for d in 0..dim {
            let base_value = base[src * dim + d];
            let offset = (((i + d) % 5) as f32 - 2.0) * 0.0002;
            queries.push(base_value + offset);
        }
    }
    queries
}

#[cfg(feature = "long-tests")]
fn run_sample_search(index: &HnswIndex, queries: &[f32], dim: usize, ef_search: usize) -> f64 {
    let search_start = Instant::now();
    for i in 0..(queries.len() / dim) {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: ef_search,
            params: Some(format!(r#"{{"ef": {ef_search}}}"#)),
            ..Default::default()
        };
        let result = index
            .search(query, &req)
            .expect("profile search must succeed");
        assert!(
            !result.ids.is_empty(),
            "profile search should return at least one neighbor"
        );
    }
    search_start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "long-tests")]
fn build_profile_artifact(
    report: HnswParallelBuildProfileReport,
    build_wall_clock_ms: f64,
    sample_search_ms: f64,
) -> serde_json::Value {
    let (num_base, num_queries, dim) = profile_dimensions();
    let sample_search_qps = if sample_search_ms > 0.0 {
        num_queries as f64 / (sample_search_ms / 1000.0)
    } else {
        num_queries as f64
    };

    json!({
        "task_id": "HNSW-REOPEN-PROFILE-ROUND1",
        "family": "HNSW",
        "benchmark_lane": "hnsw_reopen_build_path_profile_round1",
        "authority_scope": "remote_x86_only",
        "baseline_source": HNSW_REOPEN_BASELINE_PATH,
        "dataset_source": "synthetic_sift_like_128d",
        "dataset": {
            "base_vectors": num_base,
            "query_vectors": num_queries,
            "dim": dim
        },
        "config": {
            "m": 16,
            "ef_construction": 200,
            "ef_search": 64,
            "top_k": 10
        },
        "timing_buckets": report.timing_buckets,
        "call_counts": report.call_counts,
        "candidate_search_internal": report.candidate_search_internal,
        "graph_quality_call_counts": report.graph_quality_call_counts,
        "parallel_insert_entry_descent_mode": report.parallel_insert_entry_descent_mode,
        "upper_layer_overflow_shrink_mode": report.upper_layer_overflow_shrink_mode,
        "hotspot_ranking": report.hotspot_ranking,
        "recommended_first_rework_target": report.recommended_first_rework_target,
        "total_profiled_ms": report.total_profiled_ms,
        "build_wall_clock_ms": build_wall_clock_ms,
        "vectors_added": report.vectors_added,
        "repair_operations": report.repair_operations,
        "sample_search": {
            "query_count": num_queries,
            "search_wall_clock_ms": sample_search_ms,
            "qps": sample_search_qps
        }
    })
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "reopen profile artifact generation; excluded from default regression"]
fn test_generate_hnsw_reopen_profile_round1() {
    let artifact_path = Path::new(HNSW_REOPEN_PROFILE_PATH);
    if let Some(parent) = artifact_path.parent() {
        assert!(
            parent.exists(),
            "benchmark_results directory must exist before generating the reopen profile artifact"
        );
    }

    let (num_base, num_queries, dim) = profile_dimensions();
    let base = generate_profile_base_vectors(num_base, dim);
    let queries = generate_profile_query_vectors(&base, num_queries, dim);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(64),
            num_threads: Some(4),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW reopen profiler index");
    let build_start = Instant::now();
    let report = index
        .parallel_build_profile_report(&base, None)
        .expect("build HNSW reopen profile report");
    let build_wall_clock_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let sample_search_ms = run_sample_search(&index, &queries, dim, 64);
    let artifact = build_profile_artifact(report, build_wall_clock_ms, sample_search_ms);

    fs::write(
        artifact_path,
        serde_json::to_string_pretty(&artifact).expect("serialize HNSW reopen profile artifact"),
    )
    .expect("write HNSW reopen profile artifact");

    assert!(
        artifact["hotspot_ranking"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "generated artifact must include a non-empty hotspot ranking"
    );
}

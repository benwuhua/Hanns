#[cfg(feature = "long-tests")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::HnswIndex;
#[cfg(feature = "long-tests")]
use knowhere_rs::MetricType;
#[cfg(feature = "long-tests")]
use serde_json::json;
#[cfg(feature = "long-tests")]
use std::fs;
#[cfg(feature = "long-tests")]
use std::path::Path;
#[cfg(feature = "long-tests")]
use std::time::Instant;

#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND3_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round3_baseline.json";
#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND3_PROFILE_PATH: &str =
    "benchmark_results/hnsw_reopen_distance_compute_profile_round3.json";

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
fn recommended_distance_compute_target(
    report: &knowhere_rs::faiss::hnsw::HnswCandidateSearchProfileReport,
) -> &'static str {
    let breakdown = &report.distance_compute_breakdown;
    let mut ranking = [
        (
            "upper_layer_query_distance",
            breakdown.upper_layer_query_distance_ms,
        ),
        ("layer0_query_distance", breakdown.layer0_query_distance_ms),
        ("node_node_distance", breakdown.node_node_distance_ms),
    ];
    ranking.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranking[0].0
}

#[cfg(feature = "long-tests")]
fn build_round3_profile_artifact(
    report: &knowhere_rs::faiss::hnsw::HnswCandidateSearchProfileReport,
    search_wall_clock_ms: f64,
) -> serde_json::Value {
    let (num_base, num_queries, dim) = profile_dimensions();
    let search_qps = if search_wall_clock_ms > 0.0 {
        num_queries as f64 / (search_wall_clock_ms / 1000.0)
    } else {
        num_queries as f64
    };

    json!({
        "task_id": "HNSW-REOPEN-DISTANCE-COMPUTE-PROFILE-ROUND3",
        "family": "HNSW",
        "benchmark_lane": "hnsw_reopen_distance_compute_profile_round3",
        "authority_scope": "remote_x86_only",
        "round3_baseline_source": HNSW_REOPEN_ROUND3_BASELINE_PATH,
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
        "candidate_search_breakdown": &report.candidate_search_breakdown,
        "call_counts": &report.call_counts,
        "distance_compute_breakdown": &report.distance_compute_breakdown,
        "distance_compute_call_counts": &report.distance_compute_call_counts,
        "hotspot_ranking": &report.hotspot_ranking,
        "recommended_next_target": recommended_distance_compute_target(report),
        "profiled_distance_hotspot_share": if report.candidate_search_breakdown.distance_compute_ms > 0.0 {
            report.distance_compute_breakdown.layer0_query_distance_ms
                .max(report.distance_compute_breakdown.upper_layer_query_distance_ms)
                / report.candidate_search_breakdown.distance_compute_ms
        } else {
            0.0
        },
        "total_profiled_ms": report.total_profiled_ms,
        "sample_search": {
            "query_count": report.query_count,
            "search_wall_clock_ms": search_wall_clock_ms,
            "qps": search_qps
        }
    })
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "round 3 distance-compute profile artifact generation; excluded from default regression"]
fn test_generate_hnsw_reopen_round3_distance_compute_profile() {
    let artifact_path = Path::new(HNSW_REOPEN_ROUND3_PROFILE_PATH);
    if let Some(parent) = artifact_path.parent() {
        assert!(
            parent.exists(),
            "benchmark_results directory must exist before generating the round 3 distance-compute profile"
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
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW round 3 profiler index");
    index
        .train(&base)
        .expect("train HNSW round 3 profiler index");
    index
        .add(&base, None)
        .expect("add base vectors for HNSW round 3 profiler index");

    let search_start = Instant::now();
    let report = index
        .candidate_search_profile_report(&queries, 64, 10)
        .expect("profile HNSW round 3 distance compute");
    let search_wall_clock_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let artifact = build_round3_profile_artifact(&report, search_wall_clock_ms);

    fs::write(
        artifact_path,
        serde_json::to_string_pretty(&artifact)
            .expect("serialize round 3 distance-compute profile artifact"),
    )
    .expect("write round 3 distance-compute profile artifact");

    assert!(
        artifact["hotspot_ranking"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "generated artifact must include a non-empty hotspot ranking"
    );
    assert!(
        artifact["distance_compute_call_counts"]["layer0_query_distance_calls"]
            .as_u64()
            .is_some(),
        "generated artifact must include layer-0 distance call counts"
    );
}

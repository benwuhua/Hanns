#[cfg(feature = "long-tests")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::hnsw::HnswCandidateSearchProfileReport;
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
const HNSW_REOPEN_ROUND4_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round4_baseline.json";
#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND4_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json";

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
fn build_round4_audit_artifact(
    report: &HnswCandidateSearchProfileReport,
    search_wall_clock_ms: f64,
) -> serde_json::Value {
    let (num_base, num_queries, dim) = profile_dimensions();
    let search_qps = if search_wall_clock_ms > 0.0 {
        num_queries as f64 / (search_wall_clock_ms / 1000.0)
    } else {
        num_queries as f64
    };
    let parity_gap_summary = if report.batch_distance_mode.rust_batch_enabled {
        "Round 4 core rework has replaced the old dual-BinaryHeap layer-0 search core with an ordered-pool path and enabled batch-4 query distance evaluation. The next step is no longer another search-core audit; it is the real same-schema authority rerun that decides whether this new layer-0 shape actually moves the trusted Rust HNSW row."
    } else {
        "Native knowhere layer-0 search uses NeighborSetDoublePopList with batch-4 query distance evaluation, while the current Rust path still uses dual BinaryHeap containers and scalar pointer-based query distance calls. Round 4 core rework should therefore focus on ordered-pool parity and batched layer-0 query distance."
    };
    let recommended_next_target = if report.batch_distance_mode.rust_batch_enabled {
        "round4_same_schema_authority_rerun"
    } else {
        "layer0_searcher_core_parity_rework"
    };

    json!({
        "task_id": "HNSW-REOPEN-LAYER0-SEARCHER-AUDIT-ROUND4",
        "family": "HNSW",
        "benchmark_lane": "hnsw_reopen_layer0_searcher_audit_round4",
        "authority_scope": "remote_x86_only",
        "round4_baseline_source": HNSW_REOPEN_ROUND4_BASELINE_PATH,
        "native_reference_files": [
            "thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HnswSearcher.h",
            "thirdparty/faiss/faiss/cppcontrib/knowhere/impl/Neighbor.h",
            "src/index/hnsw/impl/IndexHNSWWrapper.cc"
        ],
        "rust_reference_files": [
            "src/faiss/hnsw.rs"
        ],
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
        "search_core_shape": &report.search_core_shape,
        "batch_distance_mode": &report.batch_distance_mode,
        "batch_distance_call_counts": &report.batch_distance_call_counts,
        "candidate_search_breakdown": &report.candidate_search_breakdown,
        "call_counts": &report.call_counts,
        "distance_compute_breakdown": &report.distance_compute_breakdown,
        "distance_compute_call_counts": &report.distance_compute_call_counts,
        "hotspot_ranking": &report.hotspot_ranking,
        "recommended_next_target": recommended_next_target,
        "parity_gap_summary": parity_gap_summary,
        "sample_search": {
            "query_count": report.query_count,
            "search_wall_clock_ms": search_wall_clock_ms,
            "qps": search_qps
        },
        "total_profiled_ms": report.total_profiled_ms
    })
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "round 4 layer-0 searcher audit artifact generation; excluded from default regression"]
fn test_generate_hnsw_reopen_round4_layer0_searcher_audit() {
    let artifact_path = Path::new(HNSW_REOPEN_ROUND4_AUDIT_PATH);
    if let Some(parent) = artifact_path.parent() {
        assert!(
            parent.exists(),
            "benchmark_results directory must exist before generating the round 4 audit artifact"
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

    let mut index = HnswIndex::new(&config).expect("create HNSW round 4 audit index");
    index.train(&base).expect("train HNSW round 4 audit index");
    index
        .add(&base, None)
        .expect("add base vectors for HNSW round 4 audit index");

    let search_start = Instant::now();
    let report = index
        .candidate_search_profile_report(&queries, 64, 10)
        .expect("profile HNSW round 4 layer-0 searcher audit");
    let search_wall_clock_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let artifact = build_round4_audit_artifact(&report, search_wall_clock_ms);

    fs::write(
        artifact_path,
        serde_json::to_string_pretty(&artifact)
            .expect("serialize round 4 layer-0 searcher audit artifact"),
    )
    .expect("write round 4 layer-0 searcher audit artifact");

    assert!(
        artifact["native_reference_files"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()),
        "generated artifact must include native reference files"
    );
    assert_eq!(
        artifact["search_core_shape"]["rust_layer0_candidate_container"],
        "ordered_pool"
    );
    assert!(
        artifact["batch_distance_call_counts"]["layer0_batch4_calls"]
            .as_u64()
            .is_some_and(|calls| calls > 0),
        "generated artifact must show non-zero batch-4 calls after the round-4 core rework"
    );
}

#[cfg(feature = "long-tests")]
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType};
#[cfg(feature = "long-tests")]
use knowhere_rs::faiss::{hnsw::HnswParallelBuildProfileReport, HnswIndex};
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
const HNSW_REOPEN_ROUND8_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round8_baseline.json";
#[cfg(feature = "long-tests")]
const HNSW_REOPEN_ROUND8_PARALLEL_BUILD_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_parallel_build_audit_round8.json";

#[cfg(feature = "long-tests")]
fn profile_dimensions() -> (usize, usize) {
    (12_000, 128)
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
fn build_round8_parallel_build_audit_artifact(
    report: &HnswParallelBuildProfileReport,
    build_wall_clock_ms: f64,
) -> serde_json::Value {
    let (num_base, dim) = profile_dimensions();

    json!({
        "task_id": "HNSW-REOPEN-PARALLEL-BUILD-AUDIT-ROUND8",
        "family": "HNSW",
        "benchmark_lane": "hnsw_reopen_parallel_build_audit_round8",
        "authority_scope": "remote_x86_only",
        "round8_baseline_source": HNSW_REOPEN_ROUND8_BASELINE_PATH,
        "native_reference_files": [
            "thirdparty/hnswlib/hnswlib/hnswalg.h"
        ],
        "rust_reference_files": [
            "src/faiss/hnsw.rs"
        ],
        "dataset_source": "synthetic_sift_like_128d",
        "dataset": {
            "base_vectors": num_base,
            "dim": dim
        },
        "config": {
            "m": 8,
            "ef_construction": 200,
            "ef_search": 64,
            "num_threads": 4
        },
        "timing_buckets": &report.timing_buckets,
        "call_counts": &report.call_counts,
        "hotspot_ranking": &report.hotspot_ranking,
        "recommended_first_rework_target": &report.recommended_first_rework_target,
        "total_profiled_ms": report.total_profiled_ms,
        "build_wall_clock_ms": build_wall_clock_ms,
        "vectors_added": report.vectors_added,
        "repair_operations": report.repair_operations,
        "parallel_insert_entry_descent_mode": &report.parallel_insert_entry_descent_mode,
        "upper_layer_overflow_shrink_mode": &report.upper_layer_overflow_shrink_mode,
        "build_profile_fields": &report.graph_quality_call_counts,
        "build_graph_quality_notes": "Round 8 audit confirms the current parallel build still skips upper-layer greedy descent before searching node_level, and upper-layer overflow still uses truncate_to_best instead of the native diversity heuristic."
    })
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "round 8 parallel-build audit artifact generation; excluded from default regression"]
fn test_generate_hnsw_reopen_round8_parallel_build_audit() {
    let artifact_path = Path::new(HNSW_REOPEN_ROUND8_PARALLEL_BUILD_AUDIT_PATH);
    if let Some(parent) = artifact_path.parent() {
        assert!(
            parent.exists(),
            "benchmark_results directory must exist before generating the round 8 parallel-build audit artifact"
        );
    }

    let (num_base, dim) = profile_dimensions();
    let base = generate_profile_base_vectors(num_base, dim);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(8),
            ef_construction: Some(200),
            ef_search: Some(64),
            num_threads: Some(4),
            ..Default::default()
        },
    };

    let mut index =
        HnswIndex::new(&config).expect("create HNSW round 8 parallel-build audit index");
    let build_start = Instant::now();
    let report = index
        .parallel_build_profile_report(&base, None)
        .expect("build HNSW round 8 parallel-build audit profile report");
    let build_wall_clock_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let artifact = build_round8_parallel_build_audit_artifact(&report, build_wall_clock_ms);

    fs::write(
        artifact_path,
        serde_json::to_string_pretty(&artifact)
            .expect("serialize round 8 parallel-build audit artifact"),
    )
    .expect("write round 8 parallel-build audit artifact");

    assert_eq!(
        artifact["parallel_insert_entry_descent_mode"],
        "direct_entry_at_node_level"
    );
    assert_eq!(
        artifact["upper_layer_overflow_shrink_mode"],
        "truncate_to_best"
    );
    assert!(
        artifact["build_profile_fields"]["omitted_upper_layer_descent_levels"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "generated artifact must record omitted upper-layer descent levels"
    );
    assert!(
        artifact["build_profile_fields"]["upper_layer_connection_update_calls"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "generated artifact must record upper-layer connection updates"
    );
}

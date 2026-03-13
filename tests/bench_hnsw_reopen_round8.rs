use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND8_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round8_baseline.json";
const HNSW_REOPEN_ROUND8_PARALLEL_BUILD_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_parallel_build_audit_round8.json";

fn load_round8_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND8_BASELINE_PATH)
        .expect("HNSW reopen round 8 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 8 baseline artifact must be valid JSON")
}

fn load_round8_parallel_build_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND8_PARALLEL_BUILD_AUDIT_PATH)
        .expect("HNSW reopen round 8 parallel-build audit artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 8 parallel-build audit artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round8_requires_baseline_artifact() {
    let baseline = load_round8_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND8-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round5_stability_source"],
        "benchmark_results/hnsw_reopen_round5_stability_gate.json"
    );
    assert_eq!(
        baseline["round6_prefetch_audit_source"],
        "benchmark_results/hnsw_reopen_layer0_prefetch_audit_round6.json"
    );
    assert_eq!(
        baseline["round7_flat_graph_audit_source"],
        "benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json"
    );
    assert_eq!(
        baseline["round8_target"],
        "parallel_build_graph_quality_parity"
    );
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("functional-but-not-leading"),
        "round 8 baseline summary should disclose the unchanged family verdict"
    );
}

#[test]
fn hnsw_reopen_round8_requires_parallel_build_audit_artifact() {
    let audit = load_round8_parallel_build_audit();

    assert_eq!(audit["task_id"], "HNSW-REOPEN-PARALLEL-BUILD-AUDIT-ROUND8");
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round8_baseline_source"],
        "benchmark_results/hnsw_reopen_round8_baseline.json"
    );
    assert_eq!(
        audit["parallel_insert_entry_descent_mode"],
        "direct_entry_at_node_level"
    );
    assert_eq!(
        audit["upper_layer_overflow_shrink_mode"],
        "truncate_to_best"
    );
    assert_eq!(
        audit["native_reference_files"][0],
        "thirdparty/hnswlib/hnswlib/hnswalg.h"
    );
    assert_eq!(audit["rust_reference_files"][0], "src/faiss/hnsw.rs");
    assert!(
        audit["build_profile_fields"]["omitted_upper_layer_descent_levels"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 8 audit must report omitted upper-layer descent levels"
    );
    assert!(
        audit["build_profile_fields"]["upper_layer_connection_update_calls"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 8 audit must exercise upper-layer connection updates"
    );
    assert!(
        audit["build_graph_quality_notes"]
            .as_str()
            .expect("build_graph_quality_notes must be a string")
            .contains("greedy descent"),
        "round 8 audit notes must describe the build-parity gap"
    );
}

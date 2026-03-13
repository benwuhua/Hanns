use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND8_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round8_baseline.json";
const HNSW_REOPEN_ROUND8_PARALLEL_BUILD_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_parallel_build_audit_round8.json";
const HNSW_REOPEN_ROUND8_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round8_authority_summary.json";

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

fn load_round8_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND8_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 8 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 8 authority summary artifact must be valid JSON")
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
        "greedy_from_max_level"
    );
    assert_eq!(
        audit["upper_layer_overflow_shrink_mode"],
        "heuristic_shrink"
    );
    assert_eq!(
        audit["native_reference_files"][0],
        "thirdparty/hnswlib/hnswlib/hnswalg.h"
    );
    assert_eq!(audit["rust_reference_files"][0], "src/faiss/hnsw.rs");
    assert_eq!(
        audit["build_profile_fields"]["omitted_upper_layer_descent_levels"], 0,
        "round 8 audit should report zero omitted upper-layer descent levels after the rework"
    );
    assert!(
        audit["build_profile_fields"]["upper_layer_connection_update_calls"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 8 audit must exercise upper-layer connection updates"
    );
    assert_eq!(
        audit["build_profile_fields"]["upper_layer_truncate_to_best_events"], 0,
        "round 8 audit should stop recording upper-layer truncate-to-best events after the rework"
    );
    assert!(
        audit["build_profile_fields"]["upper_layer_heuristic_shrink_events"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 8 audit must record heuristic upper-layer shrink events after the rework"
    );
    assert!(
        audit["build_graph_quality_notes"]
            .as_str()
            .expect("build_graph_quality_notes must be a string")
            .contains("greedy descent"),
        "round 8 audit notes must describe the bulk-build greedy descent behavior"
    );
}

#[test]
fn hnsw_reopen_round8_requires_authority_summary_artifact() {
    let summary = load_round8_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND8-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(
        summary["round8_target"],
        "parallel_build_graph_quality_parity"
    );
    assert_eq!(
        summary["round8_baseline_source"],
        "benchmark_results/hnsw_reopen_round8_baseline.json"
    );
    assert_eq!(
        summary["round8_parallel_build_audit_source"],
        "benchmark_results/hnsw_reopen_parallel_build_audit_round8.json"
    );
    assert_eq!(
        summary["same_schema_sources"]["rust"],
        "benchmark_results/rs_hnsw_sift128.full_k100.json"
    );
    assert!(summary["verdict_refresh_allowed"].is_boolean());
    let next_action = summary["next_action"]
        .as_str()
        .expect("next_action must be a string");
    assert!(
        matches!(next_action, "continue" | "soft_stop" | "hard_stop"),
        "next_action must be continue, soft_stop, or hard_stop"
    );
    assert!(
        summary["same_schema_current"]["rust_qps"].is_number(),
        "authority summary must record the current Rust qps"
    );
    assert!(
        summary["same_schema_current"]["native_qps"].is_number(),
        "authority summary must record the current native qps"
    );
    assert!(
        summary["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("qps"),
        "summary must describe the same-schema qps outcome"
    );
}

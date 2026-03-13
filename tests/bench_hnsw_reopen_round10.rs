use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND10_BASELINE_PATH: &str =
    "benchmark_results/hnsw_reopen_round10_baseline.json";
const HNSW_REOPEN_LAYER0_SLAB_AUDIT_ROUND10_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json";
const HNSW_REOPEN_ROUND10_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round10_authority_summary.json";

fn load_round10_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND10_BASELINE_PATH)
        .expect("HNSW reopen round 10 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 10 baseline artifact must be valid JSON")
}

fn load_round10_layer0_slab_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_LAYER0_SLAB_AUDIT_ROUND10_PATH)
        .expect("HNSW reopen round 10 layer-0 slab audit artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 10 layer-0 slab audit artifact must be valid JSON")
}

fn load_round10_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND10_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 10 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 10 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round10_requires_baseline_artifact() {
    let baseline = load_round10_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND10-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round9_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round9_authority_summary.json"
    );
    assert_eq!(baseline["round10_target"], "layer0_slab_locality");
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("5.61x"),
        "round 10 baseline summary should disclose the current round-9 authority gap"
    );
}

#[test]
fn hnsw_reopen_round10_requires_layer0_slab_audit_artifact() {
    let audit = load_round10_layer0_slab_audit();

    assert_eq!(audit["task_id"], "HNSW-REOPEN-LAYER0-SLAB-AUDIT-ROUND10");
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round10_baseline_source"],
        "benchmark_results/hnsw_reopen_round10_baseline.json"
    );
    assert_eq!(audit["production_layer0_layout_mode"], "layer0_slab");
    assert_eq!(audit["profiled_layer0_layout_mode"], "flat_graph_profiled");
    assert_eq!(audit["layer0_slab_enabled"], true);
    assert!(audit["layer0_slab_stride_bytes"].is_number());
    assert!(audit["layer0_slab_vector_offset_bytes"].is_number());
}

#[test]
fn hnsw_reopen_round10_requires_authority_summary_artifact() {
    let summary = load_round10_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND10-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round10_target"], "layer0_slab_locality");
    assert_eq!(
        summary["round10_baseline_source"],
        "benchmark_results/hnsw_reopen_round10_baseline.json"
    );
    assert_eq!(
        summary["round10_layer0_slab_audit_source"],
        "benchmark_results/hnsw_reopen_layer0_slab_audit_round10.json"
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
    assert!(summary["same_schema_current"]["rust_qps"].is_number());
    assert!(summary["same_schema_current"]["native_qps"].is_number());
    assert!(
        summary["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("qps"),
        "summary must describe the same-schema qps outcome"
    );
}

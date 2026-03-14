use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND12_BASELINE_PATH: &str =
    "benchmark_results/hnsw_reopen_round12_baseline.json";
const HNSW_REOPEN_ROUND12_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round12_authority_summary.json";

fn load_round12_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND12_BASELINE_PATH)
        .expect("HNSW reopen round 12 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 12 baseline artifact must be valid JSON")
}

fn load_round12_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND12_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 12 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 12 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round12_requires_baseline_artifact() {
    let baseline = load_round12_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND12-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round11_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round11_authority_summary.json"
    );
    assert_eq!(
        baseline["screen_promote_context"]["source"],
        "benchmark_results/hnsw_bitset_search_cost_diagnosis.json"
    );
    assert_eq!(baseline["round12_target"], "shared_bitset_batch4");
    assert_eq!(
        baseline["screen_promote_context"]["screen_result"],
        "promote"
    );
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("Task 16"),
        "round 12 baseline summary should disclose the promoted Task 16 screen evidence"
    );
}

#[test]
fn hnsw_reopen_round12_requires_authority_summary_artifact() {
    let summary = load_round12_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND12-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round12_target"], "shared_bitset_batch4");
    assert_eq!(
        summary["round12_baseline_source"],
        "benchmark_results/hnsw_reopen_round12_baseline.json"
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
    let rust_qps = &summary["same_schema_current"]["rust_qps"];
    let rust_run_outcome = summary["same_schema_current"]["rust_run_outcome"]
        .as_str()
        .expect("rust_run_outcome must be a string");
    assert!(
        rust_qps.is_number()
            || (rust_qps.is_null()
                && matches!(
                    rust_run_outcome,
                    "completed" | "timed_out" | "aborted_after_catastrophic_regression"
                )),
        "rust_qps must be numeric for completed runs or null for timeout/abort outcomes"
    );
    assert!(summary["same_schema_current"]["native_qps"].is_number());
    assert!(
        summary["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("qps"),
        "summary must describe the same-schema qps outcome"
    );
}

use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND11_BASELINE_PATH: &str =
    "benchmark_results/hnsw_reopen_round11_baseline.json";
const HNSW_REOPEN_ROUND11_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round11_authority_summary.json";

fn load_round11_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND11_BASELINE_PATH)
        .expect("HNSW reopen round 11 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 11 baseline artifact must be valid JSON")
}

fn load_round11_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND11_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 11 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 11 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round11_requires_baseline_artifact() {
    let baseline = load_round11_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND11-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round10_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round10_authority_summary.json"
    );
    assert_eq!(baseline["round11_target"], "filtered_bruteforce_fallback");
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("4.86x"),
        "round 11 baseline summary should disclose the current round-10 authority gap"
    );
}

#[test]
#[ignore = "authority summary lands in the round11 authority rerun slice"]
fn hnsw_reopen_round11_requires_authority_summary_artifact() {
    let summary = load_round11_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND11-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round11_target"], "filtered_bruteforce_fallback");
    assert_eq!(
        summary["round11_baseline_source"],
        "benchmark_results/hnsw_reopen_round11_baseline.json"
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

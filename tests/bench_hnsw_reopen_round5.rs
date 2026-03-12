use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND5_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round5_baseline.json";
const HNSW_REOPEN_ROUND5_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round5_authority_summary.json";

fn load_round5_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND5_BASELINE_PATH)
        .expect("HNSW reopen round 5 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 5 baseline artifact must be valid JSON")
}

fn load_round5_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND5_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 5 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 5 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round5_requires_baseline_artifact() {
    let baseline = load_round5_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND5-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["round4_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round4_authority_summary.json"
    );
    assert_eq!(baseline["round5_target"], "distance_dispatch_cache");
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be string")
            .contains("15."),
        "round 5 baseline summary should disclose round 4 gap context"
    );
}

#[test]
fn hnsw_reopen_round5_requires_authority_summary_artifact() {
    let summary = load_round5_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND5-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round5_target"], "distance_dispatch_cache");
    assert_eq!(
        summary["round5_baseline_source"],
        "benchmark_results/hnsw_reopen_round5_baseline.json"
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
        summary["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("qps"),
        "summary must mention qps deltas"
    );
}

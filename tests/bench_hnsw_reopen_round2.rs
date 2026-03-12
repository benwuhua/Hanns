use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND2_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round2_baseline.json";
const HNSW_REOPEN_ROUND2_PROFILE_PATH: &str =
    "benchmark_results/hnsw_reopen_candidate_search_profile_round2.json";
const HNSW_REOPEN_ROUND2_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round2_authority_summary.json";

fn load_hnsw_reopen_round2_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND2_BASELINE_PATH)
        .expect("HNSW reopen round 2 baseline artifact must exist for the round 2 lane");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 2 baseline artifact must be valid JSON")
}

fn load_hnsw_reopen_round2_profile() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND2_PROFILE_PATH)
        .expect("HNSW reopen round 2 candidate-search profile artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 2 candidate-search profile artifact must be valid JSON")
}

fn load_hnsw_reopen_round2_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND2_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 2 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 2 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round2_requires_activation_baseline_artifact() {
    let baseline = load_hnsw_reopen_round2_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND2-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round1_baseline_source"],
        "benchmark_results/hnsw_reopen_baseline.json"
    );
    assert_eq!(
        baseline["round1_profile_source"],
        "benchmark_results/hnsw_reopen_profile_round1.json"
    );
    assert_eq!(
        baseline["round2_target"],
        "candidate_search_same_schema_qps"
    );
    assert_eq!(
        baseline["historical_classification"],
        "functional-but-not-leading"
    );
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("functional-but-not-leading"),
        "summary must disclose the unchanged historical HNSW verdict"
    );
}

#[test]
fn hnsw_reopen_round2_requires_candidate_search_profile_artifact() {
    let profile = load_hnsw_reopen_round2_profile();
    let buckets = profile["candidate_search_breakdown"]
        .as_object()
        .expect("candidate_search_breakdown must be an object");

    assert_eq!(
        profile["task_id"],
        "HNSW-REOPEN-CANDIDATE-SEARCH-PROFILE-ROUND2"
    );
    assert_eq!(profile["family"], "HNSW");
    assert_eq!(
        profile["benchmark_lane"],
        "hnsw_reopen_candidate_search_profile_round2"
    );
    assert_eq!(profile["authority_scope"], "remote_x86_only");
    assert_eq!(
        profile["round2_baseline_source"],
        "benchmark_results/hnsw_reopen_round2_baseline.json"
    );

    for key in [
        "entry_descent_ms",
        "frontier_ops_ms",
        "visited_ops_ms",
        "distance_compute_ms",
        "candidate_pruning_ms",
    ] {
        let value = buckets[key]
            .as_f64()
            .unwrap_or_else(|| panic!("candidate search bucket {key} must be numeric"));
        assert!(
            value >= 0.0,
            "candidate search bucket {key} must be non-negative"
        );
    }
}

#[test]
fn hnsw_reopen_round2_requires_authority_summary_artifact() {
    let summary = load_hnsw_reopen_round2_authority_summary();
    let delta = summary["delta_vs_reopen_baseline"]
        .as_object()
        .expect("delta_vs_reopen_baseline must be an object");
    let current = summary["same_schema_current"]
        .as_object()
        .expect("same_schema_current must be an object");

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND2-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round2_target"], "candidate_search_same_schema_qps");
    assert_eq!(
        summary["round2_baseline_source"],
        "benchmark_results/hnsw_reopen_round2_baseline.json"
    );
    assert_eq!(
        summary["round2_profile_source"],
        "benchmark_results/hnsw_reopen_candidate_search_profile_round2.json"
    );
    assert_eq!(
        summary["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert!(
        summary["verdict_refresh_allowed"].is_boolean(),
        "verdict_refresh_allowed must be boolean"
    );
    assert!(
        matches!(
            summary["next_action"].as_str(),
            Some("continue" | "soft_stop" | "hard_stop")
        ),
        "next_action must be one of continue/soft_stop/hard_stop"
    );

    for key in [
        "rust_recall_at_10",
        "rust_qps",
        "native_recall_at_10",
        "native_qps",
        "native_over_rust_qps_ratio",
    ] {
        assert!(
            current[key].as_f64().is_some(),
            "same_schema_current.{key} must be numeric"
        );
    }

    for key in [
        "rust_recall_at_10_delta",
        "rust_qps_pct",
        "native_qps_pct",
        "native_over_rust_qps_ratio_pct",
    ] {
        assert!(
            delta[key].as_f64().is_some(),
            "delta_vs_reopen_baseline.{key} must be numeric"
        );
    }

    assert!(
        summary["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("qps"),
        "summary must mention same-schema qps movement versus the reopen baseline"
    );
}

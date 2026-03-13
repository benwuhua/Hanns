use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND9_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round9_baseline.json";
const HNSW_REOPEN_SEARCH_FASTPATH_AUDIT_ROUND9_PATH: &str =
    "benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json";
const HNSW_REOPEN_ROUND9_AUTHORITY_SUMMARY_PATH: &str =
    "benchmark_results/hnsw_reopen_round9_authority_summary.json";

fn load_round9_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND9_BASELINE_PATH)
        .expect("HNSW reopen round 9 baseline artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 9 baseline artifact must be valid JSON")
}

fn load_round9_search_fastpath_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_SEARCH_FASTPATH_AUDIT_ROUND9_PATH)
        .expect("HNSW reopen round 9 search fast-path audit artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 9 search fast-path audit artifact must be valid JSON")
}

fn load_round9_authority_summary() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND9_AUTHORITY_SUMMARY_PATH)
        .expect("HNSW reopen round 9 authority summary artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 9 authority summary artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round9_requires_baseline_artifact() {
    let baseline = load_round9_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND9-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round8_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round8_authority_summary.json"
    );
    assert_eq!(baseline["round9_target"], "search_fastpath_cleanup");
    assert!(
        baseline["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("hard stop"),
        "round 9 baseline summary should disclose the round 8 hard-stop context"
    );
}

#[test]
fn hnsw_reopen_round9_requires_search_fastpath_audit_artifact() {
    let audit = load_round9_search_fastpath_audit();

    assert_eq!(audit["task_id"], "HNSW-REOPEN-SEARCH-FASTPATH-AUDIT-ROUND9");
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round9_baseline_source"],
        "benchmark_results/hnsw_reopen_round9_baseline.json"
    );
    assert_eq!(audit["production_layer0_fastpath_mode"], "fast_unprofiled");
    assert_eq!(audit["profiled_layer0_mode"], "profiled_optional");
    assert_eq!(audit["production_avoids_profile_timing"], true);
    assert_eq!(audit["batch4_dispatch_mode"], "cached_once_lock");
}

#[test]
fn hnsw_reopen_round9_requires_authority_summary_artifact() {
    let summary = load_round9_authority_summary();

    assert_eq!(summary["task_id"], "HNSW-REOPEN-ROUND9-AUTHORITY-SUMMARY");
    assert_eq!(summary["family"], "HNSW");
    assert_eq!(summary["authority_scope"], "remote_x86_only");
    assert_eq!(summary["round9_target"], "search_fastpath_cleanup");
    assert_eq!(
        summary["round9_baseline_source"],
        "benchmark_results/hnsw_reopen_round9_baseline.json"
    );
    assert_eq!(
        summary["round9_search_fastpath_audit_source"],
        "benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json"
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

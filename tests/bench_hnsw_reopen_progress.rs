use serde_json::Value;
use std::fs;

const HNSW_REOPEN_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_baseline.json";
const HNSW_REOPEN_PROFILE_PATH: &str = "benchmark_results/hnsw_reopen_profile_round1.json";

fn load_hnsw_reopen_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_BASELINE_PATH)
        .expect("HNSW reopen baseline artifact must exist for the progress lane");
    serde_json::from_str(&content).expect("HNSW reopen baseline artifact must be valid JSON")
}

fn load_hnsw_reopen_profile() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_PROFILE_PATH)
        .expect("HNSW reopen profile artifact must exist for the progress lane");
    serde_json::from_str(&content).expect("HNSW reopen profile artifact must be valid JSON")
}

fn assert_close(actual: &Value, expected: f64) {
    let actual = actual
        .as_f64()
        .expect("baseline metric values must be encoded as numbers");
    let delta = (actual - expected).abs();
    assert!(
        delta < 1e-9,
        "expected {expected} but found {actual} (delta={delta})"
    );
}

#[test]
fn hnsw_reopen_progress_lane_requires_baseline_artifact() {
    let baseline = load_hnsw_reopen_baseline();

    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["same_schema_source"],
        "benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json"
    );
    assert_eq!(
        baseline["baseline_stop_go_source"],
        "benchmark_results/baseline_p3_001_stop_go_verdict.json"
    );

    assert_eq!(
        baseline["historical_classification"],
        "functional-but-not-leading"
    );
    assert_eq!(baseline["reopen_status"], "active");
    assert_close(
        &baseline["baseline_metrics"]["rust_recall_at_10"],
        0.9914999999999993,
    );
    assert_close(&baseline["baseline_metrics"]["rust_qps"], 710.9623183892526);
    assert_close(&baseline["baseline_metrics"]["native_recall_at_10"], 0.95);
    assert_close(&baseline["baseline_metrics"]["native_qps"], 10524.841);
    assert_close(
        &baseline["baseline_metrics"]["native_over_rust_qps_ratio"],
        14.803655169580505,
    );
}

#[test]
fn hnsw_reopen_progress_lane_requires_profile_artifact() {
    let profile = load_hnsw_reopen_profile();
    let timing_buckets = profile["timing_buckets"]
        .as_object()
        .expect("timing_buckets must be an object");
    let call_counts = profile["call_counts"]
        .as_object()
        .expect("call_counts must be an object");
    let hotspot_ranking = profile["hotspot_ranking"]
        .as_array()
        .expect("hotspot_ranking must be an array");

    assert_eq!(profile["task_id"], "HNSW-REOPEN-PROFILE-ROUND1");
    assert_eq!(profile["family"], "HNSW");
    assert_eq!(
        profile["benchmark_lane"],
        "hnsw_reopen_build_path_profile_round1"
    );
    assert_eq!(profile["authority_scope"], "remote_x86_only");
    assert_eq!(
        profile["baseline_source"],
        "benchmark_results/hnsw_reopen_baseline.json"
    );

    for key in [
        "layer_descent_ms",
        "candidate_search_ms",
        "neighbor_selection_ms",
        "connection_update_ms",
        "repair_ms",
    ] {
        let value = timing_buckets[key]
            .as_f64()
            .unwrap_or_else(|| panic!("timing bucket {key} must be numeric"));
        assert!(value >= 0.0, "timing bucket {key} must be non-negative");
    }

    for key in [
        "layer_descent_calls",
        "candidate_search_calls",
        "neighbor_selection_calls",
        "connection_update_calls",
        "repair_calls",
    ] {
        assert!(
            call_counts[key].as_u64().is_some(),
            "call count {key} must be an unsigned integer"
        );
    }

    assert!(
        !hotspot_ranking.is_empty(),
        "hotspot_ranking must include at least one ranked hotspot"
    );
    assert!(
        profile["recommended_first_rework_target"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "recommended_first_rework_target must be a non-empty string"
    );
}

use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND3_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round3_baseline.json";
const HNSW_REOPEN_ROUND3_PROFILE_PATH: &str =
    "benchmark_results/hnsw_reopen_distance_compute_profile_round3.json";

fn load_hnsw_reopen_round3_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND3_BASELINE_PATH)
        .expect("HNSW reopen round 3 baseline artifact must exist for the round 3 lane");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 3 baseline artifact must be valid JSON")
}

fn load_hnsw_reopen_round3_profile() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND3_PROFILE_PATH)
        .expect("HNSW reopen round 3 distance-compute profile artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 3 distance-compute profile artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round3_requires_activation_baseline_artifact() {
    let baseline = load_hnsw_reopen_round3_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND3-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round2_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round2_authority_summary.json"
    );
    assert_eq!(baseline["round3_target"], "distance_compute_inner_loop");
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
fn hnsw_reopen_round3_requires_distance_compute_profile_artifact() {
    let profile = load_hnsw_reopen_round3_profile();
    let buckets = profile["distance_compute_breakdown"]
        .as_object()
        .expect("distance_compute_breakdown must be an object");
    let call_counts = profile["distance_compute_call_counts"]
        .as_object()
        .expect("distance_compute_call_counts must be an object");

    assert_eq!(
        profile["task_id"],
        "HNSW-REOPEN-DISTANCE-COMPUTE-PROFILE-ROUND3"
    );
    assert_eq!(profile["family"], "HNSW");
    assert_eq!(
        profile["benchmark_lane"],
        "hnsw_reopen_distance_compute_profile_round3"
    );
    assert_eq!(profile["authority_scope"], "remote_x86_only");
    assert_eq!(
        profile["round3_baseline_source"],
        "benchmark_results/hnsw_reopen_round3_baseline.json"
    );

    for key in [
        "upper_layer_query_distance_ms",
        "layer0_query_distance_ms",
        "node_node_distance_ms",
    ] {
        let value = buckets[key]
            .as_f64()
            .unwrap_or_else(|| panic!("distance compute bucket {key} must be numeric"));
        assert!(
            value >= 0.0,
            "distance compute bucket {key} must be non-negative"
        );
    }

    for key in [
        "upper_layer_query_distance_calls",
        "layer0_query_distance_calls",
        "node_node_distance_calls",
    ] {
        assert!(
            call_counts[key].as_u64().is_some(),
            "distance compute call count {key} must be an integer"
        );
    }
}

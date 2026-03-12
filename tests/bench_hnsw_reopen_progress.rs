use serde_json::Value;
use std::fs;

const HNSW_REOPEN_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_baseline.json";

fn load_hnsw_reopen_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_BASELINE_PATH)
        .expect("HNSW reopen baseline artifact must exist for the progress lane");
    serde_json::from_str(&content).expect("HNSW reopen baseline artifact must be valid JSON")
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

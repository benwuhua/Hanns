use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND2_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round2_baseline.json";

fn load_hnsw_reopen_round2_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND2_BASELINE_PATH)
        .expect("HNSW reopen round 2 baseline artifact must exist for the round 2 lane");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 2 baseline artifact must be valid JSON")
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
    assert_eq!(baseline["round2_target"], "candidate_search_same_schema_qps");
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

use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND5_STABILITY_PATH: &str =
    "benchmark_results/hnsw_reopen_round5_stability_gate.json";

fn load_round5_stability() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND5_STABILITY_PATH)
        .expect("HNSW round 5 stability artifact must exist");
    serde_json::from_str(&content).expect("HNSW round 5 stability artifact must be valid JSON")
}

#[test]
fn hnsw_round5_requires_stability_gate_artifact() {
    let stability = load_round5_stability();

    assert_eq!(stability["task_id"], "HNSW-REOPEN-ROUND5-STABILITY-GATE");
    assert_eq!(stability["family"], "HNSW");
    assert_eq!(stability["authority_scope"], "remote_x86_only");
    assert_eq!(
        stability["round5_summary_source"],
        "benchmark_results/hnsw_reopen_round5_authority_summary.json"
    );
    assert!(
        stability["authority_logs"]
            .as_array()
            .is_some_and(|logs| logs.len() >= 2),
        "stability gate must include at least two authority logs"
    );
    assert!(
        stability["rerun_comparison"]["rust_qps_diff_pct"].is_number(),
        "stability gate must include Rust qps delta percent"
    );
    assert!(
        stability["rerun_comparison"]["native_qps_diff_pct"].is_number(),
        "stability gate must include native qps delta percent"
    );
    let verdict = stability["stability_verdict"]
        .as_str()
        .expect("stability_verdict must be a string");
    assert!(
        matches!(verdict, "stable" | "unstable"),
        "stability_verdict must be stable or unstable"
    );
    let next_action = stability["next_action"]
        .as_str()
        .expect("next_action must be a string");
    assert!(
        matches!(next_action, "continue" | "soft_stop" | "hard_stop"),
        "next_action must be continue, soft_stop, or hard_stop"
    );
}

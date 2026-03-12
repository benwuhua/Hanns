use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND4_BASELINE_PATH: &str = "benchmark_results/hnsw_reopen_round4_baseline.json";
const HNSW_REOPEN_ROUND4_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json";

fn load_hnsw_reopen_round4_baseline() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND4_BASELINE_PATH)
        .expect("HNSW reopen round 4 baseline artifact must exist for the round 4 lane");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 4 baseline artifact must be valid JSON")
}

fn load_hnsw_reopen_round4_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND4_AUDIT_PATH)
        .expect("HNSW reopen round 4 audit artifact must exist before the audit lane can pass");
    serde_json::from_str(&content).expect("HNSW reopen round 4 audit artifact must be valid JSON")
}

#[test]
fn hnsw_reopen_round4_requires_activation_baseline_artifact() {
    let baseline = load_hnsw_reopen_round4_baseline();

    assert_eq!(baseline["task_id"], "HNSW-REOPEN-ROUND4-BASELINE");
    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(
        baseline["historical_verdict_source"],
        "benchmark_results/hnsw_p3_002_final_verdict.json"
    );
    assert_eq!(
        baseline["round3_authority_summary_source"],
        "benchmark_results/hnsw_reopen_round3_authority_summary.json"
    );
    assert_eq!(baseline["round4_target"], "layer0_searcher_parity");
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
fn hnsw_reopen_round4_requires_layer0_audit_artifact() {
    let audit = load_hnsw_reopen_round4_audit();

    assert_eq!(audit["task_id"], "HNSW-REOPEN-LAYER0-SEARCHER-AUDIT-ROUND4");
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round4_baseline_source"],
        "benchmark_results/hnsw_reopen_round4_baseline.json"
    );
    assert_eq!(
        audit["native_reference_files"][0],
        "thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HnswSearcher.h"
    );
    assert_eq!(audit["rust_reference_files"][0], "src/faiss/hnsw.rs");
    assert_eq!(
        audit["search_core_shape"]["rust_layer0_candidate_container"],
        "ordered_pool"
    );
    assert_eq!(
        audit["search_core_shape"]["native_layer0_candidate_container"],
        "NeighborSetDoublePopList"
    );
    assert_eq!(
        audit["batch_distance_mode"]["rust_layer0_query_distance"],
        "batch4_pointer_fast_path"
    );
    assert_eq!(audit["batch_distance_mode"]["rust_batch_enabled"], true);
    assert_eq!(audit["batch_distance_mode"]["rust_batch_width"], 4);
    assert_eq!(
        audit["search_core_shape"]["rust_scratch_reuse_scope"],
        "visited_epoch_and_layer0_pools"
    );
    assert!(
        audit["batch_distance_call_counts"]["layer0_batch4_calls"]
            .as_u64()
            .is_some_and(|calls| calls > 0),
        "round 4 audit artifact must record non-zero batch-4 calls after the core rework"
    );
}

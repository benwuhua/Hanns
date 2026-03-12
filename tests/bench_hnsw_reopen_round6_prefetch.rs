use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND6_PREFETCH_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_prefetch_audit_round6.json";

fn load_round6_prefetch_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND6_PREFETCH_AUDIT_PATH)
        .expect("HNSW reopen round 6 prefetch audit artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 6 prefetch audit artifact must be valid JSON")
}

#[test]
fn hnsw_round6_requires_prefetch_audit_artifact() {
    let audit = load_round6_prefetch_audit();

    assert_eq!(audit["task_id"], "HNSW-REOPEN-LAYER0-PREFETCH-AUDIT-ROUND6");
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round6_baseline_source"],
        "benchmark_results/hnsw_reopen_round5_stability_gate.json"
    );
    assert_eq!(
        audit["search_core_shape"]["rust_layer0_candidate_container"],
        "ordered_pool"
    );
    assert_eq!(
        audit["prefetch_mode"]["native_layer0_vector_prefetch"],
        "next_neighbor_vector_l1"
    );
    assert!(
        audit["prefetch_mode"]["rust_prefetch_enabled"].is_boolean(),
        "prefetch mode must explicitly disclose if Rust prefetch is enabled"
    );
    assert!(
        audit["prefetch_call_counts"]["layer0_vector_prefetches"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 6 prefetch audit must record non-zero layer0 vector prefetch calls"
    );
}

use serde_json::Value;
use std::fs;

const HNSW_REOPEN_ROUND7_FLAT_GRAPH_AUDIT_PATH: &str =
    "benchmark_results/hnsw_reopen_layer0_flat_graph_audit_round7.json";

fn load_round7_flat_graph_audit() -> Value {
    let content = fs::read_to_string(HNSW_REOPEN_ROUND7_FLAT_GRAPH_AUDIT_PATH)
        .expect("HNSW reopen round 7 flat-graph audit artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW reopen round 7 flat-graph audit artifact must be valid JSON")
}

#[test]
fn hnsw_round7_requires_flat_graph_audit_artifact() {
    let audit = load_round7_flat_graph_audit();

    assert_eq!(
        audit["task_id"],
        "HNSW-REOPEN-LAYER0-FLAT-GRAPH-AUDIT-ROUND7"
    );
    assert_eq!(audit["family"], "HNSW");
    assert_eq!(audit["authority_scope"], "remote_x86_only");
    assert_eq!(
        audit["round7_baseline_source"],
        "benchmark_results/hnsw_reopen_layer0_prefetch_audit_round6.json"
    );
    assert_eq!(
        audit["search_core_shape"]["rust_layer0_candidate_container"],
        "ordered_pool"
    );
    assert_eq!(
        audit["search_core_shape"]["rust_layer0_neighbor_layout"],
        "flat_u32_adjacency"
    );
    assert_eq!(
        audit["search_core_shape"]["rust_layer0_neighbor_id_type"],
        "u32"
    );
    assert!(
        audit["layer0_neighbor_access_call_counts"]["layer0_flat_graph_neighbor_reads"]
            .as_u64()
            .is_some_and(|count| count > 0),
        "round 7 flat-graph audit must record non-zero flat graph neighbor reads"
    );
}

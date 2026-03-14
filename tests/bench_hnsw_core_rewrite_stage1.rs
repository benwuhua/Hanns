use serde_json::Value;
use std::fs;

const HNSW_CORE_REWRITE_BASELINE_LOCK_PATH: &str =
    "benchmark_results/hnsw_core_rewrite_baseline_lock.json";
const HNSW_GRAPH_DIAGNOSIS_RUST_PATH: &str = "benchmark_results/hnsw_graph_diagnosis_rust.json";
const HNSW_SEARCH_COST_DIAGNOSIS_PATH: &str = "benchmark_results/hnsw_search_cost_diagnosis.json";
const HNSW_CORE_REWRITE_DECISION_GATE_PATH: &str =
    "benchmark_results/hnsw_core_rewrite_decision_gate.json";

fn load_baseline_lock() -> Value {
    let content = fs::read_to_string(HNSW_CORE_REWRITE_BASELINE_LOCK_PATH)
        .expect("HNSW core rewrite baseline lock artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW core rewrite baseline lock artifact must be valid JSON")
}

fn load_rust_graph_diagnosis() -> Value {
    let content = fs::read_to_string(HNSW_GRAPH_DIAGNOSIS_RUST_PATH)
        .expect("HNSW Rust graph diagnosis artifact must exist");
    serde_json::from_str(&content).expect("HNSW Rust graph diagnosis artifact must be valid JSON")
}

fn load_search_cost_diagnosis() -> Value {
    let content = fs::read_to_string(HNSW_SEARCH_COST_DIAGNOSIS_PATH)
        .expect("HNSW search-cost diagnosis artifact must exist");
    serde_json::from_str(&content).expect("HNSW search-cost diagnosis artifact must be valid JSON")
}

fn load_decision_gate() -> Value {
    let content = fs::read_to_string(HNSW_CORE_REWRITE_DECISION_GATE_PATH)
        .expect("HNSW core rewrite decision gate artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW core rewrite decision gate artifact must be valid JSON")
}

#[test]
fn hnsw_core_rewrite_stage1_requires_baseline_lock_artifact() {
    let baseline = load_baseline_lock();

    assert_eq!(baseline["family"], "HNSW");
    assert_eq!(baseline["authority_scope"], "remote_x86_only");
    assert_eq!(baseline["dataset"], "sift-128-euclidean.hdf5");
}

#[test]
fn hnsw_core_rewrite_stage1_requires_rust_graph_diagnosis_artifact() {
    let artifact = load_rust_graph_diagnosis();

    assert!(
        artifact["node_count"]
            .as_u64()
            .expect("node_count must be numeric")
            > 0,
        "Rust graph diagnosis artifact must describe a non-empty graph"
    );
    assert!(
        artifact["level_histogram"].is_object(),
        "Rust graph diagnosis artifact must expose a level histogram object"
    );
}

#[test]
fn hnsw_core_rewrite_stage1_requires_search_cost_diagnosis_artifact() {
    let artifact = load_search_cost_diagnosis();

    assert!(
        artifact["ef_sweep"].is_array(),
        "search-cost diagnosis artifact must expose an ef_sweep array"
    );
    assert!(
        artifact["selected_recall_gate"].is_number(),
        "search-cost diagnosis artifact must expose the selected recall gate"
    );
}

#[test]
fn hnsw_core_rewrite_stage1_requires_decision_gate_artifact() {
    let gate = load_decision_gate();

    let branch = gate["selected_branch"]
        .as_str()
        .expect("selected_branch must be a string");
    assert!(
        matches!(branch, "build_first" | "search_first" | "dual_rewrite"),
        "selected_branch must be build_first, search_first, or dual_rewrite"
    );
}

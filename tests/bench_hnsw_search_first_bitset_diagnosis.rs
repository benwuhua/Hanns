use serde_json::Value;
use std::fs;

const HNSW_BITSET_SEARCH_COST_DIAGNOSIS_PATH: &str =
    "benchmark_results/hnsw_bitset_search_cost_diagnosis.json";

fn load_bitset_search_cost_diagnosis() -> Value {
    let content = fs::read_to_string(HNSW_BITSET_SEARCH_COST_DIAGNOSIS_PATH)
        .expect("HNSW bitset search-cost diagnosis artifact must exist");
    serde_json::from_str(&content)
        .expect("HNSW bitset search-cost diagnosis artifact must be valid JSON")
}

#[test]
fn hnsw_search_first_requires_bitset_search_cost_diagnosis_artifact() {
    let artifact = load_bitset_search_cost_diagnosis();

    assert_eq!(
        artifact["benchmark"],
        "HNSW-SEARCH-FIRST-bitset-search-cost-diagnosis"
    );
    assert!(
        artifact["ef_sweep"].is_array(),
        "bitset search-cost diagnosis artifact must expose an ef_sweep array"
    );
    assert!(
        artifact["bitset_stride"].is_number(),
        "bitset search-cost diagnosis artifact must expose the deterministic stride"
    );
    assert!(
        artifact["filtered_fraction"].is_number(),
        "bitset search-cost diagnosis artifact must expose the masked fraction"
    );
    assert!(
        artifact["selected_recall_gate"].is_number(),
        "bitset search-cost diagnosis artifact must expose the selected recall gate"
    );
}

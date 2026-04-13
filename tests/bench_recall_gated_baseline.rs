use serde_json::Value;
use std::fs;

const RECALL_GATED_BASELINE_PATH: &str = "benchmark_results/recall_gated_baseline.json";
const FINAL_CORE_PATH_CLASSIFICATION_PATH: &str =
    "benchmark_results/final_core_path_classification.json";

fn load_recall_gated_baseline() -> Value {
    let content = fs::read_to_string(RECALL_GATED_BASELINE_PATH)
        .expect("recall-gated baseline artifact must exist");
    serde_json::from_str(&content).expect("recall-gated baseline artifact must be valid JSON")
}

fn load_final_core_path_classification() -> Value {
    let content = fs::read_to_string(FINAL_CORE_PATH_CLASSIFICATION_PATH)
        .expect("final core path classification artifact must exist");
    serde_json::from_str(&content)
        .expect("final core path classification artifact must be valid JSON")
}

fn classification_row<'a>(artifact: &'a Value, family: &str) -> &'a Value {
    artifact["families"]
        .as_array()
        .expect("families must be an array")
        .iter()
        .find(|row| row["family"] == family)
        .unwrap_or_else(|| panic!("final core path artifact must include {family}"))
}

#[test]
fn recall_gated_baseline_keeps_ivfpq_below_gate_until_new_evidence_exists() {
    let report = load_recall_gated_baseline();
    let recall_gate = report["recall_gate"]
        .as_f64()
        .expect("recall gate must be a number");
    let rows = report["rows"].as_array().expect("rows must be an array");
    let ivfpq_row = rows
        .iter()
        .find(|row| row["index"] == "IVF-PQ")
        .expect("baseline artifact must include an IVF-PQ row");
    let recall = ivfpq_row["recall_at_10"]
        .as_f64()
        .expect("IVF-PQ recall_at_10 must be numeric");
    let confidence = ivfpq_row["confidence"]
        .as_str()
        .expect("IVF-PQ confidence must be a string");

    assert!(
        recall < recall_gate,
        "IVF-PQ should remain below the recall gate until a new authority artifact says otherwise"
    );
    assert_ne!(
        confidence, "trusted",
        "IVF-PQ should not be marked trusted while the baseline artifact remains sub-gate"
    );
}

#[test]
fn recall_gated_baseline_keeps_diskann_below_gate_until_new_evidence_exists() {
    let report = load_recall_gated_baseline();
    let recall_gate = report["recall_gate"]
        .as_f64()
        .expect("recall gate must be a number");
    let rows = report["rows"].as_array().expect("rows must be an array");
    let diskann_row = rows
        .iter()
        .find(|row| row["index"] == "DiskANN")
        .expect("baseline artifact must include a DiskANN row");
    let recall = diskann_row["recall_at_10"]
        .as_f64()
        .expect("DiskANN recall_at_10 must be numeric");
    let confidence = diskann_row["confidence"]
        .as_str()
        .expect("DiskANN confidence must be a string");

    assert!(
        recall < recall_gate,
        "DiskANN should remain below the recall gate until a new authority artifact says otherwise"
    );
    assert_ne!(
        confidence, "trusted",
        "DiskANN should not be marked trusted while the baseline artifact remains sub-gate"
    );
}

#[test]
fn recall_gated_baseline_matches_final_core_path_classification() {
    let report = load_recall_gated_baseline();
    let classification = load_final_core_path_classification();
    let recall_gate = report["recall_gate"]
        .as_f64()
        .expect("recall gate must be a number");
    let rows = report["rows"].as_array().expect("rows must be an array");

    let hnsw = classification_row(&classification, "HNSW");
    let ivfpq = classification_row(&classification, "IVF-PQ");
    let diskann = classification_row(&classification, "DiskANN");

    let hnsw_row = rows
        .iter()
        .find(|row| row["index"] == "HNSW")
        .expect("baseline artifact must include an HNSW row");
    let ivfpq_row = rows
        .iter()
        .find(|row| row["index"] == "IVF-PQ")
        .expect("baseline artifact must include an IVF-PQ row");
    let diskann_row = rows
        .iter()
        .find(|row| row["index"] == "DiskANN")
        .expect("baseline artifact must include a DiskANN row");

    assert_eq!(hnsw["classification"], "leading");
    assert_eq!(hnsw["leadership_claim_allowed"], true);
    assert!(
        hnsw_row["recall_at_10"]
            .as_f64()
            .expect("HNSW recall_at_10 must be numeric")
            >= recall_gate
    );
    assert_eq!(hnsw_row["confidence"], "trusted");

    assert_eq!(ivfpq["classification"], "no-go");
    assert_eq!(ivfpq["leadership_claim_allowed"], false);
    assert!(
        ivfpq_row["recall_at_10"]
            .as_f64()
            .expect("IVF-PQ recall_at_10 must be numeric")
            < recall_gate
            || ivfpq_row["confidence"] != "trusted"
    );

    assert_eq!(diskann["classification"], "constrained");
    assert_eq!(diskann["leadership_claim_allowed"], false);
    assert!(
        diskann_row["recall_at_10"]
            .as_f64()
            .expect("DiskANN recall_at_10 must be numeric")
            < recall_gate
            || diskann_row["confidence"] != "trusted"
    );
}

#[cfg(feature = "long-tests")]
use hanns::benchmark::{generate_recall_gated_baseline_report, DEFAULT_OUTPUT_PATH, RECALL_GATE};

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "benchmark baseline generation; excluded from default regression"]
fn test_generate_recall_gated_baseline() {
    let report = generate_recall_gated_baseline_report(DEFAULT_OUTPUT_PATH)
        .expect("generate recall-gated baseline report");

    assert_eq!(report.benchmark, "BENCH-P1-001-recall-gated-baseline");
    assert_eq!(report.recall_gate, RECALL_GATE);
    assert!(!report.rows.is_empty(), "report rows must not be empty");
}

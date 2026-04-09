use serde_json::Value;
use std::fs;

const CROSS_DATASET_ARTIFACT_PATH: &str = "benchmark_results/cross_dataset_sampling.json";
const FINAL_CORE_PATH_CLASSIFICATION_PATH: &str =
    "benchmark_results/final_core_path_classification.json";

fn load_cross_dataset_artifact() -> Value {
    let content =
        fs::read_to_string(CROSS_DATASET_ARTIFACT_PATH).expect("cross-dataset artifact must exist");
    serde_json::from_str(&content).expect("cross-dataset artifact must be valid JSON")
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
fn cross_dataset_sampling_keeps_ivfpq_out_of_trusted_pass_state() {
    let artifact = load_cross_dataset_artifact();
    let recall_gate = artifact["recall_gate"]
        .as_f64()
        .expect("recall gate must be numeric");
    let rows = artifact["rows"].as_array().expect("rows must be an array");
    let ivfpq_rows: Vec<&Value> = rows.iter().filter(|row| row["index"] == "IVF-PQ").collect();

    assert!(
        !ivfpq_rows.is_empty(),
        "cross-dataset artifact must include IVF-PQ rows"
    );
    assert!(
        ivfpq_rows.iter().all(|row| {
            let recall = row["recall_at_10"]
                .as_f64()
                .expect("IVF-PQ recall_at_10 must be numeric");
            let confidence = row["confidence"]
                .as_str()
                .expect("IVF-PQ confidence must be a string");
            recall < recall_gate || confidence != "trusted"
        }),
        "every IVF-PQ sampled row must stay below the recall gate or remain non-trusted until new authority evidence lands"
    );
}

#[test]
fn cross_dataset_sampling_includes_diskann_rows_as_explicit_no_go_evidence() {
    let artifact = load_cross_dataset_artifact();
    let recall_gate = artifact["recall_gate"]
        .as_f64()
        .expect("recall gate must be numeric");
    let rows = artifact["rows"].as_array().expect("rows must be an array");
    let diskann_rows: Vec<&Value> = rows
        .iter()
        .filter(|row| row["index"] == "DiskANN")
        .collect();

    assert_eq!(
        diskann_rows.len(),
        3,
        "cross-dataset artifact must include one DiskANN row per sampled dataset"
    );
    assert!(
        diskann_rows.iter().all(|row| {
            let recall = row["recall_at_10"]
                .as_f64()
                .expect("DiskANN recall_at_10 must be numeric");
            let confidence = row["confidence"]
                .as_str()
                .expect("DiskANN confidence must be a string");
            recall < recall_gate || confidence != "trusted"
        }),
        "every DiskANN sampled row must stay below the recall gate or remain non-trusted until native-comparable evidence exists"
    );
}

#[test]
fn cross_dataset_sampling_matches_final_core_path_classification() {
    let artifact = load_cross_dataset_artifact();
    let classification = load_final_core_path_classification();
    let recall_gate = artifact["recall_gate"]
        .as_f64()
        .expect("recall gate must be numeric");
    let rows = artifact["rows"].as_array().expect("rows must be an array");

    let hnsw = classification_row(&classification, "HNSW");
    let ivfpq = classification_row(&classification, "IVF-PQ");
    let diskann = classification_row(&classification, "DiskANN");

    let hnsw_rows: Vec<&Value> = rows.iter().filter(|row| row["index"] == "HNSW").collect();
    let ivfpq_rows: Vec<&Value> = rows.iter().filter(|row| row["index"] == "IVF-PQ").collect();
    let diskann_rows: Vec<&Value> = rows
        .iter()
        .filter(|row| row["index"] == "DiskANN")
        .collect();

    assert_eq!(hnsw["classification"], "leading");
    assert_eq!(hnsw["leadership_claim_allowed"], true);
    assert_eq!(
        hnsw_rows.len(),
        3,
        "cross-dataset artifact must include one HNSW row per sampled dataset"
    );
    assert!(
        hnsw_rows.iter().all(|row| {
            row["recall_at_10"]
                .as_f64()
                .expect("HNSW recall_at_10 must be numeric")
                >= recall_gate
                && row["confidence"] == "trusted"
        }),
        "HNSW sampled rows must stay trusted and above the recall gate while the family remains functional-but-not-leading"
    );

    assert_eq!(ivfpq["classification"], "no-go");
    assert_eq!(ivfpq["leadership_claim_allowed"], false);
    assert_eq!(
        ivfpq_rows.len(),
        3,
        "cross-dataset artifact must include one IVF-PQ row per sampled dataset"
    );
    assert!(
        ivfpq_rows.iter().all(|row| {
            let recall = row["recall_at_10"]
                .as_f64()
                .expect("IVF-PQ recall_at_10 must be numeric");
            let confidence = row["confidence"]
                .as_str()
                .expect("IVF-PQ confidence must be a string");
            recall < recall_gate || confidence != "trusted"
        }),
        "IVF-PQ sampled rows must stay sub-gate or non-trusted while the family remains no-go"
    );

    assert_eq!(diskann["classification"], "constrained");
    assert_eq!(diskann["leadership_claim_allowed"], false);
    assert_eq!(
        diskann_rows.len(),
        3,
        "cross-dataset artifact must include one DiskANN row per sampled dataset"
    );
    assert!(
        diskann_rows.iter().all(|row| {
            let recall = row["recall_at_10"]
                .as_f64()
                .expect("DiskANN recall_at_10 must be numeric");
            let confidence = row["confidence"]
                .as_str()
                .expect("DiskANN confidence must be a string");
            recall < recall_gate || confidence != "trusted"
        }),
        "DiskANN sampled rows must stay sub-gate or non-trusted while the family remains constrained"
    );
}

#[cfg(feature = "long-tests")]
use hanns::benchmark::{generate_cross_dataset_artifact, CROSS_DATASET_OUTPUT_PATH};

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "cross-dataset artifact generation; excluded from default regression"]
fn test_generate_cross_dataset_sampling() {
    let artifact = generate_cross_dataset_artifact(CROSS_DATASET_OUTPUT_PATH)
        .expect("generate cross dataset artifact");

    assert_eq!(artifact.benchmark, "BENCH-P2-003-cross-dataset-sampling");
    assert!(
        artifact.rows.len() >= 12,
        "artifact rows must cover >= 3 datasets x 4 indexes"
    );
}

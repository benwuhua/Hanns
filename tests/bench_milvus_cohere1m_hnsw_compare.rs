use serde_json::Value;
use std::fs;

const MILVUS_COHERE1M_COMPARE_PATH: &str =
    "benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json";

fn load_compare_artifact() -> Value {
    let content = fs::read_to_string(MILVUS_COHERE1M_COMPARE_PATH)
        .expect("Milvus Cohere 1M compare artifact must exist");
    serde_json::from_str(&content).expect("Milvus Cohere 1M compare artifact must be valid JSON")
}

fn find_row<'a>(rows: &'a [Value], backend: &str) -> &'a Value {
    rows.iter()
        .find(|row| row["backend"] == backend)
        .unwrap_or_else(|| panic!("missing row for backend {backend}"))
}

#[test]
fn milvus_cohere1m_compare_artifact_locks_fairness_contract() {
    let artifact = load_compare_artifact();
    let params = artifact["params"]
        .as_object()
        .expect("params must be a JSON object");
    let rows = artifact["rows"]
        .as_array()
        .expect("rows must be encoded as an array");

    assert_eq!(
        artifact["benchmark"],
        "milvus-vectordbbench-cohere1m-hnsw-compare"
    );
    assert_eq!(artifact["authority_scope"], "remote_x86_only");
    assert_eq!(artifact["host"], "hannsdb-x86");

    assert_eq!(params["case_type"], "Performance768D1M");
    assert_eq!(params["db"], "Milvus");
    assert_eq!(params["index"], "HNSW");
    assert_eq!(params["metric_type"], "COSINE");
    assert_eq!(params["top_k"], 100);
    assert_eq!(params["m"], 16);
    assert_eq!(params["ef_construction"], 128);
    assert_eq!(params["ef_search"], 128);

    assert_eq!(
        rows.len(),
        2,
        "comparison artifact must contain exactly two rows"
    );

    let native = find_row(rows, "milvus-native-knowhere");
    let rs = find_row(rows, "milvus-hanns");

    assert!(native["qps"].as_f64().unwrap() > 0.0);
    assert!(native["recall"].as_f64().unwrap() > 0.0);
    assert!(native["load_duration"].as_f64().unwrap() > 0.0);
    assert!(native["source_result"]
        .as_str()
        .is_some_and(|path| path.contains("native")));

    assert!(rs["qps"].as_f64().unwrap() > 0.0);
    assert!(rs["recall"].as_f64().unwrap() > 0.0);
    assert!(rs["load_duration"].as_f64().unwrap() > 0.0);
    assert!(rs["source_result"]
        .as_str()
        .is_some_and(|path| path.contains("hanns")));
}

#[test]
fn milvus_cohere1m_compare_artifact_locks_current_regression_shape() {
    let artifact = load_compare_artifact();
    let rows = artifact["rows"]
        .as_array()
        .expect("rows must be encoded as an array");
    let ratios = artifact["ratios"]
        .as_object()
        .expect("ratios must be encoded as an object");

    let native = find_row(rows, "milvus-native-knowhere");
    let rs = find_row(rows, "milvus-hanns");

    assert!(
        rs["insert_duration"].as_f64().unwrap() < native["insert_duration"].as_f64().unwrap(),
        "hanns lane should currently insert faster than native in this benchmark"
    );
    assert!(
        rs["load_duration"].as_f64().unwrap() > native["load_duration"].as_f64().unwrap(),
        "hanns lane should currently load slower than native in this benchmark"
    );
    assert!(
        rs["qps"].as_f64().unwrap() < native["qps"].as_f64().unwrap(),
        "hanns lane should currently deliver lower QPS than native in this benchmark"
    );
    assert!(
        rs["recall"].as_f64().unwrap() > native["recall"].as_f64().unwrap(),
        "hanns lane should currently deliver higher recall than native in this benchmark"
    );

    assert!(ratios["rs_over_native_qps"].as_f64().unwrap() < 1.0);
    assert!(ratios["rs_over_native_load_duration"].as_f64().unwrap() > 1.0);
    assert!(ratios["rs_over_native_insert_duration"].as_f64().unwrap() < 1.0);
}

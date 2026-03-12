#![cfg(feature = "long-tests")]

use knowhere_rs::benchmark::{
    generate_recall_gated_baseline_report, DEFAULT_OUTPUT_PATH, RECALL_GATE,
};

#[test]
#[ignore = "benchmark baseline generation; excluded from default regression"]
fn test_generate_recall_gated_baseline() {
    let report = generate_recall_gated_baseline_report(DEFAULT_OUTPUT_PATH)
        .expect("generate recall-gated baseline report");

    assert_eq!(report.benchmark, "BENCH-P1-001-recall-gated-baseline");
    assert_eq!(report.recall_gate, RECALL_GATE);
    assert!(!report.rows.is_empty(), "report rows must not be empty");
}

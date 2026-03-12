use serde_json::Value;
use std::fs;

const FINAL_PRODUCTION_ACCEPTANCE_PATH: &str = "benchmark_results/final_production_acceptance.json";

fn load_final_production_acceptance() -> Value {
    let content = fs::read_to_string(FINAL_PRODUCTION_ACCEPTANCE_PATH)
        .expect("final production acceptance artifact must exist");
    serde_json::from_str(&content).expect("final production acceptance artifact must be valid JSON")
}

fn production_gate<'a>(artifact: &'a Value, gate: &str) -> &'a Value {
    artifact["production_gates"]
        .as_array()
        .expect("production_gates must be an array")
        .iter()
        .find(|entry| entry["gate"] == gate)
        .unwrap_or_else(|| panic!("production gate {gate} must exist"))
}

#[test]
fn final_production_acceptance_archives_the_current_not_accepted_verdict() {
    let verdict = load_final_production_acceptance();
    let lint_build = production_gate(&verdict, "remote_fmt_clippy_build");
    let ffi_contract = production_gate(&verdict, "ffi_observability_persistence");
    let operator_docs = production_gate(&verdict, "remote_operator_docs");

    assert_eq!(verdict["task_id"], "FINAL-PRODUCTION-ACCEPTANCE");
    assert_eq!(verdict["authority_scope"], "remote_x86_only");
    assert_eq!(verdict["production_accepted"], false);
    assert_eq!(verdict["acceptance_status"], "not_accepted");
    assert_eq!(
        verdict["final_core_path_classification_source"],
        "benchmark_results/final_core_path_classification.json"
    );
    assert_eq!(
        verdict["final_performance_leadership_proof_source"],
        "benchmark_results/final_performance_leadership_proof.json"
    );

    assert_eq!(lint_build["status"], "closed");
    assert_eq!(lint_build["source_feature"], "prod-all-targets-clippy-fmt");

    assert_eq!(ffi_contract["status"], "closed");
    assert_eq!(
        ffi_contract["source_feature"],
        "prod-ffi-observability-persistence-gate"
    );

    assert_eq!(operator_docs["status"], "closed");
    assert_eq!(
        operator_docs["source_feature"],
        "prod-readme-remote-workflow-docs"
    );

    assert_eq!(
        verdict["acceptance_requirements"]["production_engineering_closed"],
        true
    );
    assert_eq!(
        verdict["acceptance_requirements"]["core_path_replaceability_proven"],
        false
    );
    assert_eq!(
        verdict["acceptance_requirements"]["credible_leadership_result_over_native"],
        false
    );
    assert!(
        verdict["summary"]
            .as_str()
            .expect("summary must be a string")
            .contains("not accepted"),
        "summary must explicitly state that the project is not accepted on current evidence"
    );
}

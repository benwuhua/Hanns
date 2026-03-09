use serde::Serialize;
use std::fs;
use std::path::Path;

pub const CONFIDENCE_TRUSTED: &str = "trusted";
pub const CONFIDENCE_UNRELIABLE: &str = "unreliable";
pub const CONFIDENCE_RECHECK_REQUIRED: &str = "recheck required";

pub fn confidence_from_recall(recall_at_10: f64, recall_gate: f64) -> &'static str {
    if recall_at_10 >= recall_gate {
        CONFIDENCE_TRUSTED
    } else if recall_at_10 >= 0.5 {
        CONFIDENCE_UNRELIABLE
    } else {
        CONFIDENCE_RECHECK_REQUIRED
    }
}

fn is_valid_confidence(value: &str) -> bool {
    matches!(
        value,
        CONFIDENCE_TRUSTED | CONFIDENCE_UNRELIABLE | CONFIDENCE_RECHECK_REQUIRED
    )
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReportRow {
    pub index: String,
    pub qps: f64,
    pub recall_at_10: f64,
    pub ground_truth_source: String,
    pub confidence: String,
    pub confidence_explanation: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub benchmark: String,
    pub dataset: String,
    pub base_size: usize,
    pub query_size: usize,
    pub dim: usize,
    pub recall_gate: f64,
    pub rows: Vec<BenchmarkReportRow>,
}

pub fn validate_required_fields(report: &BenchmarkReport) -> Result<(), String> {
    if report.rows.is_empty() {
        return Err("rows must not be empty".to_string());
    }

    for row in &report.rows {
        if row.ground_truth_source.trim().is_empty() {
            return Err(format!("{}: missing ground_truth_source", row.index));
        }
        if !row.recall_at_10.is_finite() || !(0.0..=1.0).contains(&row.recall_at_10) {
            return Err(format!("{}: recall_at_10 must be in [0,1]", row.index));
        }
        if !row.qps.is_finite() || row.qps <= 0.0 {
            return Err(format!("{}: qps must be > 0", row.index));
        }
        if !is_valid_confidence(row.confidence.trim()) {
            return Err(format!(
                "{}: confidence must be one of: trusted | unreliable | recheck required",
                row.index
            ));
        }
        if row.confidence != CONFIDENCE_TRUSTED && row.confidence_explanation.trim().is_empty() {
            return Err(format!(
                "{}: confidence_explanation is required when confidence is not trusted",
                row.index
            ));
        }
    }

    Ok(())
}

pub fn write_report(path: &str, report: &BenchmarkReport) -> Result<(), String> {
    validate_required_fields(report)?;

    let output = serde_json::to_string_pretty(report)
        .map_err(|e| format!("serialize benchmark report failed: {e}"))?;

    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent dir failed: {e}"))?;
    }

    fs::write(path, output).map_err(|e| format!("write benchmark report failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> BenchmarkReport {
        BenchmarkReport {
            benchmark: "bench".to_string(),
            dataset: "dataset".to_string(),
            base_size: 1,
            query_size: 1,
            dim: 1,
            recall_gate: 0.8,
            rows: vec![BenchmarkReportRow {
                index: "HNSW".to_string(),
                qps: 1000.0,
                recall_at_10: 0.9,
                ground_truth_source: "flat_exact_l2_bruteforce".to_string(),
                confidence: "trusted".to_string(),
                confidence_explanation: "".to_string(),
            }],
        }
    }

    #[test]
    fn test_validate_required_fields_passes() {
        let report = sample_report();
        validate_required_fields(&report).expect("validation should pass");
    }

    #[test]
    fn test_validate_required_fields_rejects_missing_ground_truth_source() {
        let mut report = sample_report();
        report.rows[0].ground_truth_source.clear();
        let err = validate_required_fields(&report).expect_err("validation should fail");
        assert!(err.contains("ground_truth_source"));
    }

    #[test]
    fn test_confidence_from_recall_uses_three_levels() {
        assert_eq!(confidence_from_recall(0.80, 0.80), CONFIDENCE_TRUSTED);
        assert_eq!(confidence_from_recall(0.79, 0.80), CONFIDENCE_UNRELIABLE);
        assert_eq!(confidence_from_recall(0.49, 0.80), CONFIDENCE_RECHECK_REQUIRED);
    }

    #[test]
    fn test_validate_required_fields_rejects_invalid_confidence_value() {
        let mut report = sample_report();
        report.rows[0].confidence = "untrusted".to_string();
        let err = validate_required_fields(&report).expect_err("validation should fail");
        assert!(err.contains("confidence must be one of"));
    }

    #[test]
    fn test_validate_required_fields_requires_explanation_for_low_confidence() {
        let mut report = sample_report();
        report.rows[0].confidence = CONFIDENCE_UNRELIABLE.to_string();
        report.rows[0].confidence_explanation.clear();
        let err = validate_required_fields(&report).expect_err("validation should fail");
        assert!(err.contains("confidence_explanation"));
    }
}

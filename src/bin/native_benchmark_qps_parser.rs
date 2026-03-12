use serde::Serialize;
use std::env;
use std::fs;

#[derive(Debug, Clone, PartialEq, Serialize)]
struct NativeBenchmarkRow {
    dataset: String,
    index: String,
    params: String,
    thread_num: u32,
    recall_at_10: f64,
    qps: f64,
    runtime_seconds: f64,
    source: String,
}

fn parse_header(line: &str) -> Option<(String, String, String, f64)> {
    let parts: Vec<&str> = line.split("|").map(|part| part.trim()).collect();
    if parts.len() < 4 {
        return None;
    }

    let dataset = parts[0].rsplit(']').next()?.trim().to_string();
    let index = parts[1].to_string();

    let mut params_parts: Vec<String> = parts[2..]
        .iter()
        .map(|part| (*part).trim().to_string())
        .collect();
    let last = params_parts.pop()?;
    let mut last_split = last.split("R@=");
    let last_params = last_split.next()?.trim().trim_end_matches(',').trim();
    let recall_raw = last_split.next()?.trim();
    if !last_params.is_empty() {
        params_parts.push(last_params.to_string());
    }
    let params = params_parts.join(" | ");
    let recall_at_10 = recall_raw.parse().ok()?;

    Some((dataset, index, params, recall_at_10))
}

fn parse_thread_line(line: &str) -> Option<(u32, f64, f64)> {
    let thread_num = line
        .split("thread_num =")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .parse()
        .ok()?;
    let runtime_seconds = line
        .split("elapse =")
        .nth(1)?
        .split('s')
        .next()?
        .trim()
        .parse()
        .ok()?;
    let qps = line.split("VPS =").nth(1)?.trim().parse().ok()?;

    Some((thread_num, runtime_seconds, qps))
}

fn parse_native_benchmark_log_rows(text: &str) -> Result<Vec<NativeBenchmarkRow>, String> {
    let mut current_header: Option<(String, String, String, f64)> = None;
    let mut candidates: Vec<NativeBenchmarkRow> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(header) = parse_header(line) {
            current_header = Some(header);
            continue;
        }

        if let Some((thread_num, runtime_seconds, qps)) = parse_thread_line(line) {
            if let Some((dataset, index, params, recall_at_10)) = &current_header {
                candidates.push(NativeBenchmarkRow {
                    dataset: dataset.clone(),
                    index: index.clone(),
                    params: params.clone(),
                    thread_num,
                    recall_at_10: *recall_at_10,
                    qps,
                    runtime_seconds,
                    source: "knowhere_cpp_benchmark_float_qps".to_string(),
                });
            }
        }
    }

    if candidates.is_empty() {
        return Err("no parseable benchmark rows found".to_string());
    }

    Ok(candidates)
}

fn select_native_benchmark_row(
    candidates: Vec<NativeBenchmarkRow>,
    requested_thread_num: Option<u32>,
    min_recall_at_10: Option<f64>,
) -> Result<NativeBenchmarkRow, String> {
    let filtered: Vec<NativeBenchmarkRow> = candidates
        .into_iter()
        .filter(|row| {
            min_recall_at_10
                .map(|min| row.recall_at_10 >= min)
                .unwrap_or(true)
        })
        .collect();

    if filtered.is_empty() {
        return Err(match min_recall_at_10 {
            Some(min) => format!("no benchmark rows satisfy min_recall_at_10 >= {min}"),
            None => "no benchmark rows after filtering".to_string(),
        });
    }

    if let Some(thread_num) = requested_thread_num {
        return filtered
            .into_iter()
            .filter(|row| row.thread_num == thread_num)
            .max_by(|a, b| {
                a.qps
                    .partial_cmp(&b.qps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| format!("thread_num {} not found in benchmark output", thread_num));
    }

    filtered
        .into_iter()
        .max_by(|a, b| {
            a.thread_num.cmp(&b.thread_num).then_with(|| {
                a.qps
                    .partial_cmp(&b.qps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .ok_or_else(|| "no benchmark rows after parsing".to_string())
}

fn print_usage(program: &str) {
    eprintln!(
        "Usage: {program} --input <path> [--thread-num <n>] [--min-recall-at-10 <f>] [--all]\n\
Outputs one JSON row with recall_at_10/qps/runtime_seconds mapped from native benchmark_float_qps logs.\n\
Use --all to emit every parseable row instead of selecting one."
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let mut input_path: Option<String> = None;
    let mut thread_num: Option<u32> = None;
    let mut min_recall_at_10: Option<f64> = None;
    let mut emit_all = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input_path = args.next(),
            "--thread-num" => thread_num = args.next().and_then(|value| value.parse::<u32>().ok()),
            "--min-recall-at-10" => {
                min_recall_at_10 = args.next().and_then(|value| value.parse::<f64>().ok())
            }
            "--all" => emit_all = true,
            "-h" | "--help" => {
                print_usage(
                    &env::args()
                        .next()
                        .unwrap_or_else(|| "native_benchmark_qps_parser".to_string()),
                );
                return;
            }
            _ => {
                eprintln!("unknown argument: {arg}");
                print_usage(
                    &env::args()
                        .next()
                        .unwrap_or_else(|| "native_benchmark_qps_parser".to_string()),
                );
                std::process::exit(2);
            }
        }
    }

    let Some(input_path) = input_path else {
        print_usage(
            &env::args()
                .next()
                .unwrap_or_else(|| "native_benchmark_qps_parser".to_string()),
        );
        std::process::exit(2);
    };

    let text = match fs::read_to_string(&input_path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("failed to read {input_path}: {err}");
            std::process::exit(1);
        }
    };

    let candidates = match parse_native_benchmark_log_rows(&text) {
        Ok(rows) => rows,
        Err(err) => {
            eprintln!("parse failed: {err}");
            std::process::exit(1);
        }
    };

    if emit_all {
        println!(
            "{}",
            serde_json::to_string_pretty(&candidates).expect("serialize rows")
        );
        return;
    }

    match select_native_benchmark_row(candidates, thread_num, min_recall_at_10) {
        Ok(row) => println!(
            "{}",
            serde_json::to_string_pretty(&row).expect("serialize row")
        ),
        Err(err) => {
            eprintln!("parse failed: {err}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[0.245 s] clustered_l2_4k | HNSW(fp32) | M=16 | efConstruction=200, ef=128, k=10, R@=0.9300
================================================================================
  thread_num =  1, elapse =  0.200s, VPS = 500.000
  thread_num =  8, elapse =  0.050s, VPS = 2000.000
================================================================================
[0.500 s] Test 'clustered_l2_4k/HNSW' done
"#;

    #[test]
    fn parses_highest_thread_row_by_default() {
        let rows = parse_native_benchmark_log_rows(SAMPLE).expect("parse sample");
        let row = select_native_benchmark_row(rows, None, None).expect("select sample row");
        assert_eq!(row.dataset, "clustered_l2_4k");
        assert_eq!(row.index, "HNSW(fp32)");
        assert_eq!(row.params, "M=16 | efConstruction=200, ef=128, k=10");
        assert_eq!(row.thread_num, 8);
        assert!((row.recall_at_10 - 0.93).abs() < 1e-9);
        assert!((row.qps - 2000.0).abs() < 1e-9);
        assert!((row.runtime_seconds - 0.05).abs() < 1e-9);
    }

    #[test]
    fn parses_requested_thread_row() {
        let rows = parse_native_benchmark_log_rows(SAMPLE).expect("parse sample");
        let row = select_native_benchmark_row(rows, Some(1), None).expect("parse sample thread 1");
        assert_eq!(row.thread_num, 1);
        assert!((row.qps - 500.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_missing_thread() {
        let rows = parse_native_benchmark_log_rows(SAMPLE).expect("parse sample");
        let err = select_native_benchmark_row(rows, Some(16), None)
            .expect_err("missing thread should fail");
        assert!(err.contains("thread_num 16 not found"));
    }

    #[test]
    fn supports_min_recall_filter() {
        let sample = r#"
[0.245 s] clustered_l2_4k | HNSW(fp32) | M=16 | efConstruction=200, ef=64, k=10, R@=0.1000
================================================================================
  thread_num =  8, elapse =  0.010s, VPS = 9999.000
================================================================================
[0.500 s] clustered_l2_4k | HNSW(fp32) | M=16 | efConstruction=200, ef=128, k=10, R@=0.9300
================================================================================
  thread_num =  8, elapse =  0.050s, VPS = 2000.000
================================================================================
"#;
        let rows = parse_native_benchmark_log_rows(sample).expect("parse sample");
        let row = select_native_benchmark_row(rows, None, Some(0.8)).expect("filter by recall");
        assert!((row.recall_at_10 - 0.93).abs() < 1e-9);
        assert!((row.qps - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn requested_thread_prefers_highest_qps_among_matching_rows() {
        let sample = r#"
[1.000 s] sift-128-euclidean | HNSW(FP16) | M=16 | efConstruction=100, ef=138, k=100, R@=0.9500
================================================================================
  thread_num =  8, elapse =  0.700s, VPS = 14000.000
================================================================================
[2.000 s] sift-128-euclidean | HNSW(BF16) | M=16 | efConstruction=100, ef=139, k=100, R@=0.9505
================================================================================
  thread_num =  8, elapse =  0.660s, VPS = 15144.811
================================================================================
"#;
        let rows = parse_native_benchmark_log_rows(sample).expect("parse sample");
        let row =
            select_native_benchmark_row(rows, Some(8), Some(0.95)).expect("select best thread row");
        assert_eq!(row.thread_num, 8);
        assert_eq!(row.index, "HNSW(BF16)");
        assert!((row.qps - 15144.811).abs() < 1e-9);
        assert!((row.recall_at_10 - 0.9505).abs() < 1e-9);
    }
}

//! Recall Calculation for Benchmark
//!
//! Compare search results against ground truth to calculate recall@k.

use std::collections::HashSet;

/// Calculate recall@k for a single query
///
/// # Arguments
/// * `result` - Search result IDs (top-k neighbors)
/// * `ground_truth` - Ground truth neighbor IDs (should have at least k elements)
/// * `k` - Number of results to consider
///
/// # Returns
/// * Recall value in [0.0, 1.0]
///
/// # Implementation Notes
/// This follows the same logic as C++ knowhere's GetKNNRecall:
/// - Only compares the top-k results against top-k ground truth
/// - Recall = matched_count / k
pub fn recall_at_k(result: &[i64], ground_truth: &[i32], k: usize) -> f64 {
    if ground_truth.is_empty() || result.is_empty() {
        return 0.0;
    }

    // BUG-002 FIX: Only consider top-k ground truth, matching C++ knowhere behavior
    let gt_k = k.min(ground_truth.len());
    let gt_set: HashSet<i32> = ground_truth.iter().take(gt_k).copied().collect();

    // Only consider top-k results
    let result_k = k.min(result.len());
    let matched = result
        .iter()
        .take(result_k)
        .filter(|&id| gt_set.contains(&(*id as i32)))
        .count();

    // Recall = matched / k (denominator is fixed to k, not min(k, gt_len))
    matched as f64 / result_k as f64
}

/// Calculate average recall@k across all queries
///
/// # Arguments
/// * `results` - Search results for all queries (num_queries x top_k)
/// * `ground_truth` - Ground truth for all queries (num_queries x gt_k)
/// * `k` - Number of results to consider
///
/// # Returns
/// * Average recall value in [0.0, 1.0]
pub fn average_recall_at_k(results: &[Vec<i64>], ground_truth: &[Vec<i32>], k: usize) -> f64 {
    if results.is_empty() || ground_truth.is_empty() {
        return 0.0;
    }

    assert_eq!(
        results.len(),
        ground_truth.len(),
        "Results and ground truth must have same number of queries"
    );

    let total_recall: f64 = results
        .iter()
        .zip(ground_truth.iter())
        .map(|(result, gt)| recall_at_k(result, gt, k))
        .sum();

    total_recall / results.len() as f64
}

/// Calculate recall@k with progress reporting
///
/// # Arguments
/// * `results` - Search results for all queries
/// * `ground_truth` - Ground truth for all queries
/// * `k` - Number of results to consider
/// * `report_interval` - Report progress every N queries
pub fn recall_with_progress(
    results: &[Vec<i64>],
    ground_truth: &[Vec<i32>],
    k: usize,
    report_interval: usize,
) -> f64 {
    if results.is_empty() || ground_truth.is_empty() {
        return 0.0;
    }

    assert_eq!(
        results.len(),
        ground_truth.len(),
        "Results and ground truth must have same number of queries"
    );

    let mut total_recall = 0.0f64;
    let num_queries = results.len();

    for (i, (result, gt)) in results.iter().zip(ground_truth.iter()).enumerate() {
        total_recall += recall_at_k(result, gt, k);

        if (i + 1) % report_interval == 0 {
            let current_avg = total_recall / (i + 1) as f64;
            eprintln!(
                "Progress: {}/{} queries, current recall@{} = {:.4}",
                i + 1,
                num_queries,
                k,
                current_avg
            );
        }
    }

    total_recall / num_queries as f64
}

/// Benchmark result with recall metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkResult {
    pub index_name: String,
    pub build_time_ms: f64,
    pub search_time_ms: f64,
    pub num_queries: usize,
    pub qps: f64,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
}

impl BenchmarkResult {
    /// Print benchmark results in table format
    pub fn print_table(results: &[BenchmarkResult]) {
        println!("\n=== Benchmark Results ===\n");
        println!(
            "{:<12} {:>12} {:>12} {:>10} {:>12} {:>12} {:>12}",
            "Index", "Build(ms)", "Search(ms)", "QPS", "R@1", "R@10", "R@100"
        );
        println!("{:-<90}", "");

        for r in results {
            println!(
                "{:<12} {:>12.2} {:>12.2} {:>10.0} {:>12.3} {:>12.3} {:>12.3}",
                r.index_name,
                r.build_time_ms,
                r.search_time_ms,
                r.qps,
                r.recall_at_1,
                r.recall_at_10,
                r.recall_at_100,
            );
        }
        println!();
    }

    /// Print benchmark results in Markdown table format
    pub fn print_markdown_table(results: &[BenchmarkResult], dataset: &str) {
        println!("\n## {} Benchmark Results\n", dataset);
        println!("| Index | Build Time (ms) | Search Time (ms) | QPS | Recall@1 | Recall@10 | Recall@100 |");
        println!("|-------|-----------------|------------------|-----|----------|-----------|------------|");

        for r in results {
            println!(
                "| {} | {:.2} | {:.2} | {:.0} | {:.3} | {:.3} | {:.3} |",
                r.index_name,
                r.build_time_ms,
                r.search_time_ms,
                r.qps,
                r.recall_at_1,
                r.recall_at_10,
                r.recall_at_100,
            );
        }
        println!();
    }

    /// Export benchmark results to JSON string
    pub fn to_json(results: &[BenchmarkResult], dataset: &str) -> String {
        let json_obj = serde_json::json!({
            "dataset": dataset,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "results": results
        });
        serde_json::to_string_pretty(&json_obj).unwrap_or_default()
    }

    /// Export benchmark results to JSON file
    pub fn save_json(
        results: &[BenchmarkResult],
        dataset: &str,
        output_path: &str,
    ) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let json = Self::to_json(results, dataset);
        let mut file = File::create(output_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_perfect() {
        let result = vec![1, 2, 3, 4, 5];
        let ground_truth = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let recall = recall_at_k(&result, &ground_truth, 5);
        assert!((recall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_partial() {
        let result = vec![1, 2, 99, 100, 101];
        let ground_truth = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let recall = recall_at_k(&result, &ground_truth, 5);
        assert!((recall - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_recall_zero() {
        let result = vec![100, 200, 300, 400, 500];
        let ground_truth = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let recall = recall_at_k(&result, &ground_truth, 5);
        assert!((recall - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_bug002_large_k() {
        // BUG-002: Test case for large top_k scenario
        // When ground_truth has more elements than k, only top-k should be considered
        let result = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ground_truth = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        // Recall@5: result has [1,2,3,4,5,6,7,8,9,10], gt top-5 is [1,2,3,4,5]
        // Matched: 5 (IDs 1,2,3,4,5), result_k = 5, recall = 5/5 = 1.0
        let recall = recall_at_k(&result, &ground_truth, 5);
        assert!((recall - 1.0).abs() < 1e-10);

        // Recall@10: result has [1,2,3,4,5,6,7,8,9,10], gt top-10 is [1,2,3,4,5,6,7,8,9,10]
        // Matched: 10, result_k = 10, recall = 10/10 = 1.0
        let recall = recall_at_k(&result, &ground_truth, 10);
        assert!((recall - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_ground_truth_shorter() {
        // When ground_truth has fewer elements than k
        let result = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ground_truth = vec![1, 2, 3, 4, 5];

        // Recall@10 with only 5 ground_truth elements
        let recall = recall_at_k(&result, &ground_truth, 10);
        assert!((recall - 0.5).abs() < 1e-10); // 5 matched out of 10
    }

    #[test]
    fn test_average_recall() {
        let results = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let ground_truth = vec![vec![1, 2, 3, 4, 5], vec![4, 5, 6, 7, 8]];

        let avg_recall = average_recall_at_k(&results, &ground_truth, 3);
        assert!((avg_recall - 1.0).abs() < 1e-10);
    }
}

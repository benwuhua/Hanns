//! Distance Validation for Benchmark
//!
//! Validate search result distances are within expected bounds.
//! Similar to C++ knowhere's CheckDistanceInScope() function.

/// Validate distances for KNN search results
///
/// # Arguments
/// * `distances` - Flat array of distances (num_queries * top_k)
/// * `num_queries` - Number of queries
/// * `top_k` - Number of results per query
/// * `low_bound` - Lower bound for distances (exclusive)
/// * `high_bound` - Upper bound for distances (exclusive)
///
/// # Returns
/// * `true` if all valid distances are within bounds
/// * `false` if any distance is out of bounds
pub fn check_distance_in_scope_knn(
    distances: &[f32],
    num_queries: usize,
    top_k: usize,
    low_bound: f32,
    high_bound: f32,
) -> bool {
    for i in 0..num_queries {
        for j in 0..top_k {
            let idx = i * top_k + j;
            let d = distances[idx];
            // Check if distance is within bounds (allow -1 for invalid results)
            if d != -1.0 && !(low_bound < d && d < high_bound) {
                return false;
            }
        }
    }
    true
}

/// Validate distances for range search results
///
/// # Arguments
/// * `distances` - Flat array of distances
/// * `lims` - Limit array (num_queries + 1), lims[i] to lims[i+1] gives range for query i
/// * `low_bound` - Lower bound for distances (exclusive)
/// * `high_bound` - Upper bound for distances (exclusive)
///
/// # Returns
/// * `true` if all valid distances are within bounds
/// * `false` if any distance is out of bounds
pub fn check_distance_in_scope_range(
    distances: &[f32],
    lims: &[usize],
    low_bound: f32,
    high_bound: f32,
) -> bool {
    let num_queries = lims.len() - 1;
    for i in 0..num_queries {
        for &d in distances.iter().take(lims[i + 1]).skip(lims[i]) {
            if d != -1.0 && !(low_bound < d && d < high_bound) {
                return false;
            }
        }
    }
    true
}

/// Validate L2 distances are non-negative
///
/// # Arguments
/// * `distances` - Distance array to validate
///
/// # Returns
/// * `true` if all distances >= 0
/// * `false` if any distance < 0
pub fn validate_l2_distances(distances: &[f32]) -> bool {
    distances.iter().all(|&d| d >= 0.0)
}

/// Validate IP (inner product) distances are in expected range
///
/// For normalized vectors, inner product should be in [-1, 1].
/// For unnormalized vectors, this checks against provided bounds.
///
/// # Arguments
/// * `distances` - Distance array to validate
/// * `min_expected` - Minimum expected value (default: -10.0)
/// * `max_expected` - Maximum expected value (default: 10.0)
///
/// # Returns
/// * `true` if all distances are in expected range
/// * `false` if any distance is out of range
pub fn validate_ip_distances(distances: &[f32], min_expected: f32, max_expected: f32) -> bool {
    distances
        .iter()
        .all(|&d| d >= min_expected && d <= max_expected)
}

/// Calculate distance statistics
///
/// # Returns
/// * (min, max, mean, std_dev) tuple
pub fn distance_statistics(distances: &[f32]) -> (f32, f32, f32, f32) {
    if distances.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let min = distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;

    let variance =
        distances.iter().map(|&d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
    let std_dev = variance.sqrt();

    (min, max, mean, std_dev)
}

/// Validate distance monotonicity (for sorted results)
///
/// In KNN search, results should be sorted by distance.
/// For L2/IP, distances should be non-decreasing.
///
/// # Arguments
/// * `distances` - Distance array (num_queries * top_k)
/// * `num_queries` - Number of queries
/// * `top_k` - Number of results per query
/// * `less_is_better` - true for L2 (smaller distance = better), false for IP (larger = better)
///
/// # Returns
/// * `true` if distances are properly sorted
/// * `false` if sorting is violated
pub fn validate_distance_monotonicity(
    distances: &[f32],
    num_queries: usize,
    top_k: usize,
    less_is_better: bool,
) -> bool {
    for i in 0..num_queries {
        for j in 1..top_k {
            let idx_prev = i * top_k + j - 1;
            let idx_curr = i * top_k + j;

            if less_is_better {
                // L2: distances should be non-decreasing
                if distances[idx_curr] < distances[idx_prev] {
                    return false;
                }
            } else {
                // IP: distances should be non-increasing (larger = better)
                if distances[idx_curr] > distances[idx_prev] {
                    return false;
                }
            }
        }
    }
    true
}

/// Comprehensive distance validation report
#[derive(Debug, Clone)]
pub struct DistanceValidationReport {
    pub knscope_passed: bool,
    pub l2_non_negative: bool,
    pub monotonicity_passed: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub mean_distance: f32,
    pub std_dev: f32,
}

impl DistanceValidationReport {
    /// Run all validations and return report
    pub fn validate_knn(
        distances: &[f32],
        num_queries: usize,
        top_k: usize,
        metric_l2: bool,
        low_bound: f32,
        high_bound: f32,
    ) -> Self {
        let knscope =
            check_distance_in_scope_knn(distances, num_queries, top_k, low_bound, high_bound);
        let l2_valid = if metric_l2 {
            validate_l2_distances(distances)
        } else {
            true
        };
        let monotonic = validate_distance_monotonicity(distances, num_queries, top_k, metric_l2);
        let (min, max, mean, std_dev) = distance_statistics(distances);

        Self {
            knscope_passed: knscope,
            l2_non_negative: l2_valid,
            monotonicity_passed: monotonic,
            min_distance: min,
            max_distance: max,
            mean_distance: mean,
            std_dev,
        }
    }

    /// Print validation report
    pub fn print(&self) {
        println!("\n=== Distance Validation Report ===");
        println!(
            "  Distance in scope: {}",
            if self.knscope_passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!(
            "  L2 non-negative: {}",
            if self.l2_non_negative {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!(
            "  Monotonicity: {}",
            if self.monotonicity_passed {
                "✅ PASS"
            } else {
                "❌ FAIL"
            }
        );
        println!("  Statistics:");
        println!("    Min: {:.6}", self.min_distance);
        println!("    Max: {:.6}", self.max_distance);
        println!("    Mean: {:.6}", self.mean_distance);
        println!("    StdDev: {:.6}", self.std_dev);
        println!();
    }

    /// Check if all validations passed
    pub fn all_passed(&self) -> bool {
        self.knscope_passed && self.l2_non_negative && self.monotonicity_passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_distance_in_scope_knn_pass() {
        let distances = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert!(check_distance_in_scope_knn(&distances, 2, 3, 0.0, 1.0));
    }

    #[test]
    fn test_check_distance_in_scope_knn_fail() {
        let distances = vec![0.1, 0.2, 1.5, 0.4, 0.5, 0.6];
        assert!(!check_distance_in_scope_knn(&distances, 2, 3, 0.0, 1.0));
    }

    #[test]
    fn test_validate_l2_distances() {
        let distances = vec![0.0, 1.5, 2.3, 10.0];
        assert!(validate_l2_distances(&distances));

        let negative = vec![0.0, -0.1, 2.3];
        assert!(!validate_l2_distances(&negative));
    }

    #[test]
    fn test_validate_ip_distances() {
        let distances = vec![-0.8, -0.5, 0.0, 0.5, 0.8];
        assert!(validate_ip_distances(&distances, -1.0, 1.0));

        let out_of_range = vec![-0.8, 1.5, 0.0];
        assert!(!validate_ip_distances(&out_of_range, -1.0, 1.0));
    }

    #[test]
    fn test_distance_statistics() {
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std_dev) = distance_statistics(&distances);
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
        assert!((mean - 3.0).abs() < 1e-6);
        assert!(std_dev > 1.4);
    }

    #[test]
    fn test_monotonicity_l2_pass() {
        let distances = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert!(validate_distance_monotonicity(&distances, 2, 3, true));
    }

    #[test]
    fn test_monotonicity_l2_fail() {
        let distances = vec![0.1, 0.3, 0.2, 0.4, 0.5, 0.6];
        assert!(!validate_distance_monotonicity(&distances, 2, 3, true));
    }

    #[test]
    fn test_monotonicity_ip_pass() {
        let distances = vec![0.9, 0.7, 0.5, 0.3, 0.1, -0.1];
        assert!(validate_distance_monotonicity(&distances, 2, 3, false));
    }

    #[test]
    fn test_validation_report() {
        let distances = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let report = DistanceValidationReport::validate_knn(&distances, 2, 3, true, 0.0, 1.0);
        assert!(report.all_passed());
        report.print();
    }
}

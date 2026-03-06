/// Debug test for HNSW search results count
///
/// This test checks if search_layer returns enough candidates.
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::index::Index;
use rand::Rng;

fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[test]
fn test_hnsw_search_count() {
    let dim = 128;
    let num_base = 10000;

    println!("\n=== HNSW Search Count Debug ===");

    // Generate random data
    let mut rng = rand::thread_rng();
    let base: Vec<f32> = (0..num_base * dim).map(|_| rng.gen::<f32>()).collect();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Build HNSW index
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
            data_type: crate::api::DataType::Float,
        params: IndexParams::hnsw(16, 200, 0.5),
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&base).unwrap();
    index.add(&base, None).unwrap();

    // Search
    let req = SearchRequest {
        top_k: 100,
        ..Default::default()
    };
    let result = index.search(&query, &req).unwrap();

    println!("Requested top_k: 100");
    println!("Returned IDs: {}", result.ids.len());
    println!("Returned distances: {}", result.distances.len());

    // Count how many are valid (not -1)
    let valid_count = result.ids.iter().filter(|&&id| id != -1).count();
    println!("Valid results: {}", valid_count);

    // Check distance values
    println!("\nDistance statistics:");
    let valid_dists: Vec<f32> = result.distances.iter().copied().collect();
    if !valid_dists.is_empty() {
        let min_dist = valid_dists.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_dist = valid_dists
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = valid_dists.iter().sum();
        let mean = sum / valid_dists.len() as f32;
        println!("  Min: {:.6}", min_dist);
        println!("  Max: {:.6}", max_dist);
        println!("  Mean: {:.6}", mean);

        // Check if distances are sorted
        let mut is_sorted = true;
        for i in 1..valid_dists.len().min(50) {
            if valid_dists[i] < valid_dists[i - 1] {
                is_sorted = false;
                break;
            }
        }
        println!(
            "  Sorted (first 50): {}",
            if is_sorted { "YES" } else { "NO" }
        );
    }

    // Compute ground truth for comparison
    let mut gt_distances: Vec<(usize, f32)> = base
        .chunks(dim)
        .enumerate()
        .map(|(idx, vec)| (idx, l2_distance_sq(&query, vec)))
        .collect();
    gt_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\nGround Truth (first 10):");
    for (i, (idx, dist)) in gt_distances.iter().take(10).enumerate() {
        println!("  {:>3}: idx={:>6}, dist={:.6}", i, idx, dist.sqrt());
    }

    println!("\nHNSW Results (first 10):");
    for i in 0..10.min(result.ids.len()) {
        println!(
            "  {:>3}: id={:>6}, dist={:.6}",
            i, result.ids[i], result.distances[i]
        );
    }

    // Check overlap
    let gt_ids: std::collections::HashSet<i64> = gt_distances
        .iter()
        .take(100)
        .map(|(idx, _)| *idx as i64)
        .collect();
    let matched: usize = result
        .ids
        .iter()
        .take(100)
        .filter(|&&id| gt_ids.contains(&id))
        .count();
    println!(
        "\nRecall@100: {}/100 = {:.3}",
        matched,
        matched as f64 / 100.0
    );
}

#![cfg(feature = "long-tests")]
//! Debug HNSW recall issue
//!
//! Print detailed comparison between ground truth and HNSW search results

mod common;
use hanns::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::HnswIndex;
use rand::Rng;
use std::collections::HashSet;
use std::time::Instant;

fn generate_random_dataset(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(num_vectors * dim);
    for _ in 0..(num_vectors * dim) {
        data.push(rng.gen_range(-100.0..100.0));
    }
    data
}

fn compute_ground_truth(
    base: &[f32],
    query: &[f32],
    num_queries: usize,
    dim: usize,
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    let num_base = base.len() / dim;
    let mut ground_truth = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query[i * dim..(i + 1) * dim];
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(num_base);

        for j in 0..num_base {
            let b = &base[j * dim..(j + 1) * dim];
            let dist = l2_distance(q, b); // Use L2 distance (not squared)
            distances.push((j, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbors: Vec<(usize, f32)> = distances.into_iter().take(k).collect();
        ground_truth.push(neighbors);
    }

    ground_truth
}

/// L2 distance (not squared) - matches HNSW return format
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    common::l2_distance_squared(a, b).sqrt()
}

#[test]
fn debug_hnsw_recall_detailed() {
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    Debug HNSW Recall Issue                                                        ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    // Small dataset for debugging
    let num_base = 1000;
    let num_queries = 5;
    let dim = 128;
    let m = 16;
    let ef_construction = 200;
    let ef_search = 64; // Test with the reported ef_search value
    let top_k = 10;

    println!("Configuration:");
    println!("  Base vectors: {}", num_base);
    println!("  Query vectors: {}", num_queries);
    println!("  Dimensions: {}", dim);
    println!("  HNSW M: {}", m);
    println!("  EF Construction: {}", ef_construction);
    println!("  EF Search: {}", ef_search);
    println!("  Top-K: {}", top_k);

    // Generate dataset
    println!("\nGenerating dataset...");
    let base_data = generate_random_dataset(num_base, dim);
    let query_data = generate_random_dataset(num_queries, dim);

    // Build HNSW
    println!("\nBuilding HNSW index...");
    let build_start = Instant::now();
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(1.0 / (m as f32).ln()),
            ..Default::default()
        },
    };
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&base_data).unwrap();
    index.add(&base_data, None).unwrap();
    let build_time = build_start.elapsed().as_secs_f64() * 1000.0;
    println!("HNSW built in {:.2}ms", build_time);

    // Compute ground truth
    println!("\nComputing ground truth (k={})...", top_k);
    let ground_truth =
        common::compute_ground_truth(&base_data, &query_data, num_queries, dim, top_k);

    // Search with HNSW
    println!("\nSearching with HNSW...");
    let mut hnsw_results: Vec<Vec<(i64, f32)>> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let q = &query_data[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k,
            nprobe: ef_search,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();

        let mut query_results = Vec::with_capacity(top_k);
        for j in 0..top_k {
            let id = result.ids[j];
            let dist = result.distances[j];
            query_results.push((id, dist));
        }
        hnsw_results.push(query_results);
    }

    // Detailed comparison
    println!("\n═══════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("Detailed Comparison (first 3 queries):");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════\n");

    for q_idx in 0..num_queries.min(3) {
        println!("Query {}:", q_idx);
        println!("{:-<120}", "");

        let gt = &ground_truth[q_idx];
        let hnsw = &hnsw_results[q_idx];

        // Print ground truth
        println!("  Ground Truth (top {}):", top_k);
        for (rank, &(id, dist)) in gt.iter().enumerate().take(5) {
            println!("    #{:<3} ID: {:<6} Dist: {:.4}", rank + 1, id, dist);
        }
        if gt.len() > 5 {
            println!("    ... ({} more)", gt.len() - 5);
        }

        // Print HNSW results
        println!("  HNSW Results (top {}):", top_k);
        for (rank, &(id, dist)) in hnsw.iter().enumerate() {
            println!("    #{:<3} ID: {:<6} Dist: {:.4}", rank + 1, id, dist);
        }

        // Check matches
        let gt_ids: HashSet<usize> = gt.iter().map(|&(id, _)| id).collect();
        let matched: Vec<_> = hnsw
            .iter()
            .filter(|&(id, _)| gt_ids.contains(&(*id as usize)))
            .collect();

        println!("  Matched: {} / {}", matched.len(), top_k);

        // Check distance consistency
        println!("  Distance Check:");
        for (_rank, &(id, dist)) in hnsw.iter().enumerate().take(3) {
            let idx = id as usize;
            let q = &query_data[q_idx * dim..(q_idx + 1) * dim];
            let b = &base_data[idx * dim..(idx + 1) * dim];
            let manual_dist = l2_distance(q, b); // Use L2 distance (not squared)
            let dist_diff = (dist - manual_dist).abs();
            println!(
                "    ID {}: HNSW dist={:.4}, Manual dist={:.4}, Diff={:.6}",
                id, dist, manual_dist, dist_diff
            );
        }

        println!();
    }

    // Calculate recall
    let mut total_matched = 0;
    let mut total_possible = 0;

    for q_idx in 0..num_queries {
        let gt_ids: HashSet<usize> = ground_truth[q_idx].iter().map(|&(id, _)| id).collect();
        let hnsw_ids: HashSet<i64> = hnsw_results[q_idx].iter().map(|&(id, _)| id).collect();

        let matched = hnsw_ids
            .iter()
            .filter(|&id| gt_ids.contains(&(*id as usize)))
            .count();
        total_matched += matched;
        total_possible += top_k;
    }

    let recall = total_matched as f64 / total_possible as f64;
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "Overall Recall@{}: {} / {} = {:.4}",
        top_k, total_matched, total_possible, recall
    );
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════\n");

    // Assert reasonable recall
    assert!(
        recall > 0.5,
        "Recall@{} should be > 0.5, but got {:.4}",
        top_k,
        recall
    );
}

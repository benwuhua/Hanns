#![cfg(feature = "long-tests")]
//! Quick HNSW test on SIFT1M to verify recall on real clustered data
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, SearchRequest};
use knowhere_rs::benchmark::average_recall_at_k;
use knowhere_rs::dataset::load_sift1m_complete;
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use std::env;

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_sift1m_hnsw_quick() {
    let path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());

    let dataset = match load_sift1m_complete(&path) {
        Ok(ds) => ds,
        Err(e) => {
            println!("Skipping - SIFT1M not found: {}", e);
            return;
        }
    };

    println!(
        "Loaded SIFT1M: {} base, {} query, {} dim",
        dataset.num_base(),
        dataset.num_query(),
        dataset.dim()
    );

    // Use smaller subset for quick test
    let num_queries = 100;
    let base = dataset.base.vectors();
    let query = dataset.query.vectors();
    let gt = &dataset.ground_truth;

    // Build HNSW
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim: dataset.dim(),
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(32),
            ef_construction: Some(400),
            ef_search: Some(400),
            ..Default::default()
        },
    };

    println!("\nBuilding HNSW (M=32, ef_c=400)...");
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(base).unwrap();
    index.add(base, None).unwrap();

    let (max_layer, _, avg_neighbors) = index.get_graph_stats();
    println!(
        "Graph: max_layer={}, avg_neighbors_l0={:.1}",
        max_layer, avg_neighbors
    );

    // Search with ef=400
    println!("\nSearching with ef=400...");
    let mut results: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let q = &query[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        results.push(result.ids);
    }

    let gt_subset: Vec<_> = gt.iter().take(num_queries).cloned().collect();
    let r1 = average_recall_at_k(&results, &gt_subset, 1);
    let r10 = average_recall_at_k(&results, &gt_subset, 10);
    let r100 = average_recall_at_k(&results, &gt_subset, 100);

    println!(
        "Results (ef=400): R@1={:.3}, R@10={:.3}, R@100={:.3}",
        r1, r10, r100
    );

    // Also test with ef=800
    index.set_ef_search(800);
    let mut results2: Vec<Vec<i64>> = Vec::with_capacity(num_queries);
    for i in 0..num_queries {
        let q = &query[i * dataset.dim()..(i + 1) * dataset.dim()];
        let req = SearchRequest {
            top_k: 100,
            ..Default::default()
        };
        let result = index.search(q, &req).unwrap();
        results2.push(result.ids);
    }

    let r1_2 = average_recall_at_k(&results2, &gt_subset, 1);
    let r10_2 = average_recall_at_k(&results2, &gt_subset, 10);
    let r100_2 = average_recall_at_k(&results2, &gt_subset, 100);

    println!(
        "Results (ef=800): R@1={:.3}, R@10={:.3}, R@100={:.3}",
        r1_2, r10_2, r100_2
    );

    // Assertions for quality
    assert!(r10 > 0.90, "R@10 should be >90% on SIFT1M, got {:.3}", r10);
    println!("\n✅ HNSW recall on SIFT1M is good!");
}

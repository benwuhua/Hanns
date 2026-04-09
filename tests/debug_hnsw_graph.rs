#![cfg(feature = "long-tests")]
mod common;
use knowhere_rs::api::{IndexConfig, IndexParams, IndexType};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::MetricType;
use rand::Rng;


#[test]
fn debug_hnsw_graph_stats() {
    let n = 10000;
    let dim = 128;
    let vectors = common::generate_vectors(n, dim);

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams {
            m: Some(32),
            ef_construction: Some(400),
            ef_search: Some(400),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    let (max_layer, layer_dist, avg_neighbors_l0) = index.get_graph_stats();

    println!("HNSW Graph Statistics:");
    println!("  Max layer: {}", max_layer);
    println!("  Avg neighbors at layer 0: {:.2}", avg_neighbors_l0);
    println!("  Layer distribution:");
    for (layer, count) in layer_dist.iter().enumerate() {
        if *count > 0 {
            println!(
                "    Layer {}: {} nodes ({:.1}%)",
                layer,
                count,
                *count as f64 / n as f64 * 100.0
            );
        }
    }

    // Check connectivity
    println!(
        "\n  Expected M: 32, Actual avg neighbors: {:.2}",
        avg_neighbors_l0
    );
    if avg_neighbors_l0 < 20.0 {
        println!("  ⚠️  WARNING: Low connectivity - graph may have quality issues");
    }
}

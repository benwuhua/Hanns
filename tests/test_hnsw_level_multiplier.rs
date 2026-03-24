use knowhere_rs::api::{DataType, IndexConfig, IndexParams};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::{IndexType, MetricType};

fn generate_vectors(num_vectors: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(num_vectors * dim);
    for i in 0..num_vectors {
        for j in 0..dim {
            vectors.push((i * dim + j) as f32 / 1000.0);
        }
    }
    vectors
}

fn build_index(m: usize, ml: Option<f32>, num_vectors: usize) -> HnswIndex {
    let dim = 4;
    let vectors = generate_vectors(num_vectors, dim);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(m),
            ef_construction: Some(32),
            ef_search: Some(32),
            ml,
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    index
}

/// Test that high-M builds use the runtime-M level multiplier rather than a fixed reference M.
///
/// For M = 64, ml = 1/ln(64) ≈ 0.2404:
/// - P(level >= 1) = 1/64 ≈ 0.015625
/// - P(level >= 2) = 1/64^2 ≈ 0.000244
///
/// With 50000 vectors:
/// - Expected average level is about 1 / (M - 1) ≈ 0.0159
/// - Expected at level >= 2: ~12 nodes, so max_level >= 2 is still realistic
#[test]
fn test_hnsw_default_level_distribution_stays_stable_at_high_m() {
    // Use 50K vectors for statistically reliable level distribution
    let index = build_index(64, None, 50_000);

    // Primary check: average level should reflect the runtime M=64 distribution,
    // not the old reference-M=16 behavior.
    let avg_level = index.average_node_level();
    assert!(
        avg_level > 0.01 && avg_level < 0.03,
        "expected runtime-M level distribution for high M, got avg_level={}",
        avg_level
    );

    // Secondary check: should have a multi-layer graph
    // With 50K vectors and ml≈0.24, we still expect a handful of nodes at level >= 2.
    let max_level = index.max_level();
    assert!(
        max_level >= 2,
        "expected multi-layer graph, got max_level={}",
        max_level
    );
}

#[test]
fn test_hnsw_explicit_ml_override_changes_level_distribution() {
    // Use fewer vectors for this test since explicit ml=1.2 creates more layers
    let index = build_index(64, Some(1.2), 10_000);
    let avg_level = index.average_node_level();
    assert!(
        avg_level > 0.25,
        "expected explicit ml override to raise average level, got avg_level={}",
        avg_level
    );
}

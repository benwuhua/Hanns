use hanns::api::{DataType, IndexConfig, IndexParams, SearchRequest};
use hanns::{HnswIndex, IndexType, MetricType};

const DIM: usize = 32;

fn make_vectors(n: usize) -> Vec<f32> {
    (0..n * DIM).map(|k| (k as f32 * 0.001).sin()).collect()
}

fn build_test_index(n: usize, metric: MetricType) -> HnswIndex {
    let vectors = make_vectors(n);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: metric,
        dim: DIM,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(64),
            ef_search: Some(64),
            ..Default::default()
        },
    };
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    index
}

fn search_top_k(index: &HnswIndex, queries: &[f32], k: usize) -> Vec<Vec<i64>> {
    let n_queries = queries.len() / DIM;
    let req = SearchRequest {
        top_k: k,
        nprobe: 64,
        filter: None,
        params: None,
        radius: None,
    };
    let result = index.search(queries, &req).unwrap();
    (0..n_queries)
        .map(|q| result.ids[q * k..(q + 1) * k].to_vec())
        .collect()
}

/// Round-trip serialize → deserialize must return identical search results (L2).
#[test]
fn test_serialize_roundtrip_l2_search() {
    let index = build_test_index(500, MetricType::L2);
    let queries = make_vectors(10);
    let k = 5;

    let before = search_top_k(&index, &queries, k);

    let bytes = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes).unwrap();

    let after = search_top_k(&restored, &queries, k);

    assert_eq!(
        before, after,
        "search results changed after L2 serialize/deserialize roundtrip"
    );
}

/// Round-trip for Cosine must also return identical results (normalization not double-applied).
#[test]
fn test_serialize_roundtrip_cosine_search() {
    let index = build_test_index(500, MetricType::Cosine);
    let queries = make_vectors(10);
    let k = 5;

    let before = search_top_k(&index, &queries, k);

    let bytes = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes).unwrap();

    let after = search_top_k(&restored, &queries, k);

    assert_eq!(
        before, after,
        "search results changed after Cosine serialize/deserialize roundtrip"
    );
}

/// Serialized bytes must be stable: re-serializing a deserialized index produces
/// the same byte sequence (no double-normalization or data mutation).
#[test]
fn test_serialize_bytes_stable() {
    let index = build_test_index(200, MetricType::L2);
    let bytes1 = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes1).unwrap();
    let bytes2 = restored.serialize_to_bytes().unwrap();
    assert_eq!(
        bytes1, bytes2,
        "re-serializing a deserialized index produced different bytes"
    );
}

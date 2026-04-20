use hanns::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, HnswRuntime};
use hanns::kernel::{AnnRuntime, IndexFamily};

#[test]
fn hnsw_runtime_matches_hnsw_search() {
    let dim = 4;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(8);
    cfg.params.ef_construction = Some(32);

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let ids = vec![10, 11, 12, 13];
    let query = [0.0, 0.0, 0.0, 0.0];
    let req = SearchRequest {
        top_k: 2,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };

    let mut index = HnswIndex::new(&cfg).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, Some(&ids)).unwrap();

    let expected = index.search(&query, &req).unwrap();
    let runtime = HnswRuntime::new(index);
    let mut got_ids = [-1_i64; 2];
    let mut got_dists = [f32::INFINITY; 2];
    let n = runtime
        .search_into(&query, &req, &mut got_ids, &mut got_dists)
        .unwrap();

    assert_eq!(runtime.family(), IndexFamily::Hnsw);
    assert_eq!(n, expected.ids.len());
    assert_eq!(&got_ids[..n], expected.ids.as_slice());
}

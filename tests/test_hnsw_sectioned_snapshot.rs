use hanns::api::{IndexConfig, IndexType, MetricType};
use hanns::faiss::HnswIndex;

fn build_small_hnsw() -> HnswIndex {
    let dim = 4;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(4);
    cfg.params.ef_construction = Some(16);
    cfg.params.ef_search = Some(16);
    cfg.params.random_seed = Some(42);

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let ids = vec![10, 11, 12, 13];

    let mut index = HnswIndex::new(&cfg).expect("hnsw index should build");
    index.train(&vectors).expect("hnsw index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("hnsw vectors should add");
    index
}

#[test]
fn hnsw_exports_sectioned_snapshot_shape() {
    let index = build_small_hnsw();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 4);
    assert_eq!(export.count, 4);
    assert_eq!(export.ids.len(), 4);
    assert_eq!(export.levels.len(), 4);
    assert_eq!(export.vectors.len(), 16);
    assert_eq!(export.neighbor_offsets.first().copied(), Some(0));
    assert!(export.neighbor_offsets.len() > export.levels.len());
    assert_eq!(export.neighbor_ids.len(), export.neighbor_dists.len());
}

use hanns::api::{IndexConfig, IndexType, MetricType, SqMode};
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
    assert_eq!(
        export.neighbor_offsets.last().copied(),
        Some(export.neighbor_ids.len() as u64)
    );
    assert_eq!(
        export.neighbor_offsets.len(),
        export
            .levels
            .iter()
            .map(|level| *level as usize + 1)
            .sum::<usize>()
            + 1
    );
    assert!(export
        .neighbor_offsets
        .windows(2)
        .all(|window| window[0] <= window[1]));
    assert_eq!(export.neighbor_ids.len(), export.neighbor_dists.len());
}

#[test]
fn hnsw_exports_deleted_ids_sorted_for_determinism() {
    let mut index = build_small_hnsw();
    index.mark_deleted(13);
    index.mark_deleted(11);

    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.deleted_ids, vec![11, 13]);
}

#[test]
fn hnsw_exports_sq_metadata_with_codes() {
    let dim = 8;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(4);
    cfg.params.ef_construction = Some(16);
    cfg.params.ef_search = Some(16);
    cfg.params.num_threads = Some(1);
    cfg.params.random_seed = Some(42);
    cfg.params.sq_mode = Some(SqMode::SQ8);

    let mut vectors = Vec::with_capacity(16 * dim);
    for i in 0..16usize {
        for d in 0..dim {
            vectors.push((i as f32 * 0.01) + (d as f32 * 0.001));
        }
    }

    let mut index = HnswIndex::new(&cfg).expect("hnsw sq index should build");
    index.train(&vectors).expect("hnsw sq index should train");
    index
        .add(&vectors, None)
        .expect("hnsw sq vectors should add");

    let export = index.export_sectioned_snapshot().expect("export");

    assert_ne!(export.sq_mode, SqMode::None);
    let sq_meta = export.sq_meta.expect("sq metadata should export");
    assert!(!export.sq_codes.is_empty());
    assert_eq!(sq_meta.dim, dim);
}

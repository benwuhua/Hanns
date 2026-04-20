use hanns::api::{IndexConfig, IndexType, MetricType, SearchRequest, SqMode};
use hanns::faiss::{HnswIndex, HnswSectionedSnapshot, HnswSnapshotLoader};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, AnnSnapshotLoader, IndexArtifactReader, LoadMode, MemoryArtifactStore,
};

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

#[test]
fn hnsw_sectioned_snapshot_writes_expected_sections() {
    let index = build_small_hnsw();
    let export = index.export_sectioned_snapshot().expect("export");
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();

    snapshot.write_snapshot(&mut store).expect("write");
    let manifest = store.manifest().expect("manifest");

    assert_eq!(manifest.family, IndexFamily::Hnsw);
    assert_eq!(manifest.variant, "hnsw_sections_v1");
    assert!(store.section_len("hnsw.meta.json").unwrap() > 0);
    assert_eq!(store.section_len("hnsw.ids.i64").unwrap(), 4 * 8);
    assert_eq!(store.section_len("hnsw.vectors.f32").unwrap(), 16 * 4);

    assert_eq!(
        store.section_len("hnsw.neighbors.offsets.u64").unwrap(),
        export.neighbor_offsets.len() as u64 * 8
    );
    assert_eq!(
        store.section_len("hnsw.neighbors.ids.i64").unwrap(),
        export.neighbor_ids.len() as u64 * 8
    );
    assert_eq!(
        store.section_len("hnsw.neighbors.dists.f32").unwrap(),
        export.neighbor_dists.len() as u64 * 4
    );

    for section in [
        "hnsw.meta.json",
        "hnsw.ids.i64",
        "hnsw.vectors.f32",
        "hnsw.levels.u32",
        "hnsw.neighbors.offsets.u64",
        "hnsw.neighbors.ids.i64",
        "hnsw.neighbors.dists.f32",
        "hnsw.deleted.ids.i64",
    ] {
        let descriptor = manifest
            .sections
            .iter()
            .find(|candidate| candidate.name == section)
            .unwrap_or_else(|| panic!("missing manifest descriptor for {section}"));
        assert_eq!(descriptor.len, store.section_len(section).unwrap());
    }
}

#[test]
fn hnsw_sectioned_snapshot_roundtrips_search_results() {
    let index = build_small_hnsw();
    let req = SearchRequest {
        top_k: 3,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };
    let query = [0.0, 0.0, 0.0, 0.0];
    let expected = index.search(&query, &req).expect("search");

    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loader = HnswSnapshotLoader;
    let runtime = loader
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("load");
    let mut ids = vec![-1; req.top_k];
    let mut dists = vec![f32::INFINITY; req.top_k];
    let n = runtime
        .search_into(&query, &req, &mut ids, &mut dists)
        .expect("runtime search");

    assert_eq!(&ids[..n], expected.ids.as_slice());
}

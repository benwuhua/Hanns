use hanns::api::{IndexConfig, IndexType, KnowhereError, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, HnswSnapshot, HnswSnapshotLoader, HNSW_SNAPSHOT_SECTION};
use hanns::index::Index;
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, MemoryArtifactStore,
};
use tempfile::tempdir;

fn build_small_hnsw() -> HnswIndex {
    let dim = 4;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(8);
    cfg.params.ef_construction = Some(32);
    cfg.params.ef_search = Some(32);
    cfg.params.random_seed = Some(42);

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let ids = vec![10, 11, 12, 13, 14];

    let mut index = HnswIndex::new(&cfg).expect("hnsw index should build");
    index.train(&vectors).expect("hnsw index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("hnsw vectors should add");
    index
}

fn assert_runtime_matches_index(index: &HnswIndex, runtime: &dyn hanns::kernel::AnnRuntime) {
    let query = [0.0, 0.0, 0.0, 0.0];
    let req = SearchRequest {
        top_k: 3,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };
    let expected = index
        .search(&query, &req)
        .expect("original search should run");
    let mut ids = [-1_i64; 3];
    let mut dists = [f32::INFINITY; 3];

    let n = runtime
        .search_into(&query, &req, &mut ids, &mut dists)
        .expect("snapshot runtime search should run");

    assert_eq!(runtime.family(), IndexFamily::Hnsw);
    assert_eq!(runtime.dim(), index.dim());
    assert_eq!(runtime.len(), index.ntotal());
    assert_eq!(n, expected.ids.len());
    assert_eq!(&ids[..n], expected.ids.as_slice());
    assert_eq!(&dists[..n], expected.distances.as_slice());
}

#[test]
fn hnsw_snapshot_roundtrips_through_memory_store() {
    let index = build_small_hnsw();
    let snapshot = HnswSnapshot::from_index(&index).expect("snapshot should build");
    let mut store = MemoryArtifactStore::default();

    snapshot
        .write_snapshot(&mut store)
        .expect("snapshot should write");

    let manifest = store.manifest().expect("manifest should exist");
    assert_eq!(manifest.family, IndexFamily::Hnsw);
    assert_eq!(manifest.variant, "hnsw_blob_v1");
    assert_eq!(manifest.dim, index.dim());
    assert_eq!(manifest.count, index.ntotal());
    assert_eq!(manifest.sections.len(), 1);
    assert_eq!(manifest.sections[0].name, HNSW_SNAPSHOT_SECTION);
    assert_eq!(
        manifest.sections[0].len,
        store
            .section_len(HNSW_SNAPSHOT_SECTION)
            .expect("hnsw bytes should exist")
    );

    let runtime = HnswSnapshotLoader
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_runtime_matches_index(&index, runtime.as_ref());
}

#[test]
fn hnsw_snapshot_roundtrips_through_file_store() {
    let index = build_small_hnsw();
    let snapshot = HnswSnapshot::from_index(&index).expect("snapshot should build");
    let dir = tempdir().expect("tempdir should build");
    let mut writer = FileArtifactStore::new(dir.path()).expect("file store should open");

    snapshot
        .write_snapshot(&mut writer)
        .expect("snapshot should write");

    let reader = FileArtifactStore::new(dir.path()).expect("file store should reopen");
    let runtime = HnswSnapshotLoader
        .load_snapshot(&reader, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_runtime_matches_index(&index, runtime.as_ref());
}

#[test]
fn hnsw_snapshot_loader_errors_when_hnsw_bytes_section_is_missing() {
    let mut store = MemoryArtifactStore::default();
    store
        .finish_manifest(&IndexManifest {
            version: 1,
            family: IndexFamily::Hnsw,
            variant: "hnsw_blob_v1".to_string(),
            dim: 4,
            metric: "unknown".to_string(),
            count: 0,
            sections: Vec::new(),
        })
        .expect("manifest should write");

    let error = match HnswSnapshotLoader.load_snapshot(&store, LoadMode::OwnedMemory) {
        Ok(_) => panic!("missing hnsw bytes should error"),
        Err(error) => error,
    };

    assert!(
        matches!(error, KnowhereError::Codec(_)),
        "expected Codec error, got {error:?}"
    );
    assert!(error.to_string().contains(HNSW_SNAPSHOT_SECTION));
}

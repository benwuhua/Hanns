use hanns::api::{
    DataType, IndexConfig, IndexParams, IndexType, KnowhereError, MetricType, SearchRequest,
};
use hanns::faiss::{
    load_ivf_flat_index_from_artifact, IvfFlatIndex, IvfFlatSectionedSnapshot,
    IVF_FLAT_CENTROIDS_SECTION, IVF_FLAT_IDS_SECTION, IVF_FLAT_LIST_IDS_SECTION,
    IVF_FLAT_LIST_OFFSETS_SECTION, IVF_FLAT_LIST_SIZES_SECTION, IVF_FLAT_LIST_VECTORS_SECTION,
    IVF_FLAT_META_SECTION, IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT, IVF_FLAT_VECTORS_SECTION,
};
use hanns::kernel::IndexFamily;
use hanns::storage::{AnnSnapshot, IndexArtifactReader, MemoryArtifactStore};

fn build_small_ivf_flat() -> (IvfFlatIndex, usize) {
    let dim = 4;
    let nlist = 3;
    let cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim,
        data_type: DataType::Float,
        params: IndexParams::ivf(nlist, 2),
    };
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        0.1, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, //
        1.1, 1.0, 1.0, 1.0, //
        2.0, 2.0, 2.0, 2.0, //
        2.1, 2.0, 2.0, 2.0,
    ];
    let ids = vec![10, 11, 12, 13, 14, 15];

    let mut index = IvfFlatIndex::new(&cfg).expect("ivf-flat index should build");
    index.train(&vectors).expect("ivf-flat index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("ivf-flat vectors should add");
    (index, nlist)
}

#[test]
fn ivf_flat_rejects_untrained_sectioned_export() {
    let cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: DataType::Float,
        params: IndexParams::ivf(3, 2),
    };
    let index = IvfFlatIndex::new(&cfg).expect("ivf-flat index should build");

    let error = index
        .export_sectioned_snapshot()
        .expect_err("untrained IVF-Flat sectioned export should fail");

    assert!(
        matches!(
            error,
            KnowhereError::InvalidArg(_) | KnowhereError::Codec(_)
        ),
        "expected clear validation error, got {error:?}"
    );
    assert!(
        error.to_string().contains("trained"),
        "error should mention trained state, got {error}"
    );
}

#[test]
fn ivf_flat_exports_sectioned_snapshot_shape() {
    let (index, nlist) = build_small_ivf_flat();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 4);
    assert_eq!(export.count, 6);
    assert_eq!(export.nlist, nlist);
    assert_eq!(export.centroids.len(), nlist * export.dim);
    assert_eq!(export.list_offsets.len(), nlist);
    assert_eq!(export.list_sizes.len(), nlist);

    let list_total = export.list_sizes.iter().sum::<u64>();
    assert_eq!(list_total, export.list_ids.len() as u64);
    assert_eq!(
        export.list_vectors.len(),
        export.list_ids.len() * export.dim
    );
    assert_eq!(export.ids.len(), export.count);
    assert_eq!(export.vectors.len(), export.count * export.dim);
}

#[test]
fn ivf_flat_sectioned_snapshot_writes_expected_sections() {
    let (index, _) = build_small_ivf_flat();
    let export = index.export_sectioned_snapshot().expect("export");
    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();

    snapshot.write_snapshot(&mut store).expect("write");
    let manifest = store.manifest().expect("manifest");

    assert_eq!(manifest.family, IndexFamily::Ivf);
    assert_eq!(manifest.variant, IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT);
    assert!(store.section_len(IVF_FLAT_META_SECTION).unwrap() > 0);

    for (section, expected_len) in [
        (
            IVF_FLAT_CENTROIDS_SECTION,
            export.centroids.len() as u64 * 4,
        ),
        (
            IVF_FLAT_LIST_OFFSETS_SECTION,
            export.list_offsets.len() as u64 * 8,
        ),
        (
            IVF_FLAT_LIST_SIZES_SECTION,
            export.list_sizes.len() as u64 * 8,
        ),
        (IVF_FLAT_LIST_IDS_SECTION, export.list_ids.len() as u64 * 8),
        (
            IVF_FLAT_LIST_VECTORS_SECTION,
            export.list_vectors.len() as u64 * 4,
        ),
        (IVF_FLAT_IDS_SECTION, export.ids.len() as u64 * 8),
        (IVF_FLAT_VECTORS_SECTION, export.vectors.len() as u64 * 4),
    ] {
        assert_eq!(store.section_len(section).unwrap(), expected_len);
    }

    for section in [
        IVF_FLAT_META_SECTION,
        IVF_FLAT_CENTROIDS_SECTION,
        IVF_FLAT_LIST_OFFSETS_SECTION,
        IVF_FLAT_LIST_SIZES_SECTION,
        IVF_FLAT_LIST_IDS_SECTION,
        IVF_FLAT_LIST_VECTORS_SECTION,
        IVF_FLAT_IDS_SECTION,
        IVF_FLAT_VECTORS_SECTION,
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
fn ivf_flat_sectioned_snapshot_roundtrips_search_results() {
    let (index, _) = build_small_ivf_flat();
    let query = [1.05, 1.0, 1.0, 1.0];
    let req = SearchRequest {
        top_k: 4,
        nprobe: 3,
        ..Default::default()
    };
    let original = index.search(&query, &req).expect("original search");

    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_ivf_flat_index_from_artifact(&store).expect("load");
    let roundtrip = loaded.search(&query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

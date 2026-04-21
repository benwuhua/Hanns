use std::collections::BTreeMap;

use hanns::api::{
    DataType, IndexConfig, IndexParams, IndexType, KnowhereError, MetricType, SearchRequest,
};
use hanns::bitset::BitsetView;
use hanns::dataset::Dataset;
use hanns::faiss::{
    load_ivf_sq8_index_from_artifact, load_ivf_sq8_sectioned_snapshot,
    save_ivf_sq8_sectioned_snapshot, IvfSq8Index, IvfSq8SectionedSnapshot,
    IVF_SQ8_CENTROIDS_SECTION, IVF_SQ8_IDS_SECTION, IVF_SQ8_LIST_CODES_SECTION,
    IVF_SQ8_LIST_IDS_SECTION, IVF_SQ8_LIST_OFFSETS_SECTION, IVF_SQ8_LIST_ROWS_SECTION,
    IVF_SQ8_LIST_SIZES_SECTION, IVF_SQ8_META_SECTION, IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT,
    IVF_SQ8_VECTORS_SECTION,
};
use hanns::index::Index as _;
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, IndexArtifactReader, IndexArtifactWriter, IndexManifest, MemoryArtifactStore,
};
use tempfile::tempdir;

fn build_small_ivf_sq8() -> (IvfSq8Index, Vec<f32>, Vec<i64>, usize) {
    let dim = 4;
    let nlist = 3;
    let cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
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
    let ids = vec![100, 101, 102, 103, 104, 105];

    let mut index = IvfSq8Index::new(&cfg).expect("ivf-sq8 index should build");
    index.train(&vectors).expect("ivf-sq8 index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("ivf-sq8 vectors should add");
    (index, vectors, ids, nlist)
}

fn sectioned_store_with(
    index: &IvfSq8Index,
    edit: impl FnOnce(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = IvfSq8SectionedSnapshot::from_index(index).expect("snapshot");
    let mut source = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut source).expect("write");

    let mut manifest = source.manifest().expect("manifest").clone();
    let mut sections = BTreeMap::new();
    for descriptor in &manifest.sections {
        sections.insert(
            descriptor.name.clone(),
            source
                .read_section(&descriptor.name)
                .expect("section should read")
                .into_owned(),
        );
    }

    edit(&mut manifest, &mut sections);

    let mut store = MemoryArtifactStore::default();
    for (name, bytes) in sections {
        store.write_section(&name, &bytes).expect("section write");
    }
    store.finish_manifest(&manifest).expect("manifest write");
    store
}

fn refresh_manifest_lengths(manifest: &mut IndexManifest, sections: &BTreeMap<String, Vec<u8>>) {
    for descriptor in &mut manifest.sections {
        descriptor.len = sections
            .get(&descriptor.name)
            .unwrap_or_else(|| panic!("section {} should exist", descriptor.name))
            .len() as u64;
    }
}

fn load_error(store: &MemoryArtifactStore) -> KnowhereError {
    match load_ivf_sq8_index_from_artifact(store) {
        Ok(_) => panic!("malformed sectioned IVF-SQ8 snapshot should error"),
        Err(error) => error,
    }
}

fn assert_codec_contains(error: KnowhereError, expected: &str) {
    assert!(
        matches!(error, KnowhereError::Codec(_)),
        "expected Codec error, got {error:?}"
    );
    assert!(
        error.to_string().contains(expected),
        "error should contain {expected:?}, got {error}"
    );
}

fn decode_u64s(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn encode_u64s(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn ivf_sq8_rejects_untrained_sectioned_export() {
    let cfg = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: DataType::Float,
        params: IndexParams::ivf(3, 2),
    };
    let index = IvfSq8Index::new(&cfg).expect("ivf-sq8 index should build");

    let error = index
        .export_sectioned_snapshot()
        .expect_err("untrained IVF-SQ8 sectioned export should fail");

    assert!(
        error.to_string().contains("untrained"),
        "error should mention trained state, got {error}"
    );
}

#[test]
fn ivf_sq8_exports_sectioned_snapshot_shape() {
    let (index, _, _, nlist) = build_small_ivf_sq8();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 4);
    assert_eq!(export.count, 6);
    assert_eq!(export.nlist, nlist);
    assert_eq!(export.quantizer_bit, 8);
    assert!(export.quantizer_scale > 0.0);
    assert_eq!(export.centroids.len(), nlist * export.dim);
    assert_eq!(export.list_offsets.len(), nlist);
    assert_eq!(export.list_sizes.len(), nlist);

    let list_total = export.list_sizes.iter().sum::<u64>();
    assert_eq!(list_total, export.list_ids.len() as u64);
    assert_eq!(export.list_rows.len(), export.list_ids.len());
    assert_eq!(export.list_codes.len(), export.list_ids.len() * export.dim);
    assert_eq!(export.ids.len(), export.count);
    assert_eq!(export.vectors.len(), export.count * export.dim);
}

#[test]
fn ivf_sq8_sectioned_snapshot_writes_expected_sections() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let export = index.export_sectioned_snapshot().expect("export");
    let snapshot = IvfSq8SectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();

    snapshot.write_snapshot(&mut store).expect("write");
    let manifest = store.manifest().expect("manifest");

    assert_eq!(manifest.family, IndexFamily::Ivf);
    assert_eq!(manifest.variant, IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT);
    assert!(store.section_len(IVF_SQ8_META_SECTION).unwrap() > 0);

    for (section, expected_len) in [
        (IVF_SQ8_CENTROIDS_SECTION, export.centroids.len() as u64 * 4),
        (
            IVF_SQ8_LIST_OFFSETS_SECTION,
            export.list_offsets.len() as u64 * 8,
        ),
        (
            IVF_SQ8_LIST_SIZES_SECTION,
            export.list_sizes.len() as u64 * 8,
        ),
        (IVF_SQ8_LIST_IDS_SECTION, export.list_ids.len() as u64 * 8),
        (IVF_SQ8_LIST_ROWS_SECTION, export.list_rows.len() as u64 * 8),
        (IVF_SQ8_LIST_CODES_SECTION, export.list_codes.len() as u64),
        (IVF_SQ8_IDS_SECTION, export.ids.len() as u64 * 8),
        (IVF_SQ8_VECTORS_SECTION, export.vectors.len() as u64 * 4),
    ] {
        assert_eq!(store.section_len(section).unwrap(), expected_len);
    }

    for section in [
        IVF_SQ8_META_SECTION,
        IVF_SQ8_CENTROIDS_SECTION,
        IVF_SQ8_LIST_OFFSETS_SECTION,
        IVF_SQ8_LIST_SIZES_SECTION,
        IVF_SQ8_LIST_IDS_SECTION,
        IVF_SQ8_LIST_ROWS_SECTION,
        IVF_SQ8_LIST_CODES_SECTION,
        IVF_SQ8_IDS_SECTION,
        IVF_SQ8_VECTORS_SECTION,
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
fn ivf_sq8_sectioned_snapshot_roundtrips_search_results() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let query = [1.05, 1.0, 1.0, 1.0];
    let req = SearchRequest {
        top_k: 4,
        nprobe: 3,
        ..Default::default()
    };
    let original = index.search(&query, &req).expect("original search");

    let snapshot = IvfSq8SectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_ivf_sq8_index_from_artifact(&store).expect("load");
    let roundtrip = loaded.search(&query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn ivf_sq8_sectioned_file_helpers_roundtrip() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let query = [2.05, 2.0, 2.0, 2.0];
    let req = SearchRequest {
        top_k: 4,
        nprobe: 3,
        ..Default::default()
    };
    let original = index.search(&query, &req).expect("original search");
    let dir = tempdir().expect("tempdir should build");

    save_ivf_sq8_sectioned_snapshot(&index, dir.path()).expect("save sectioned snapshot");
    let loaded = load_ivf_sq8_sectioned_snapshot(dir.path()).expect("load sectioned snapshot");
    let roundtrip = loaded.search(&query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn ivf_sq8_sectioned_restore_preserves_bitset_internal_rows() {
    let (index, vectors, ids, _) = build_small_ivf_sq8();
    let query = Dataset::from_f32_slice(&vectors[0..4], 4);
    let mut bitset = BitsetView::new(ids.len());
    bitset.set(0, true);

    let snapshot = IvfSq8SectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");
    let loaded = load_ivf_sq8_index_from_artifact(&store).expect("load");

    let filtered = loaded
        .search_with_bitset(&query, 3, &bitset)
        .expect("bitset search");

    assert!(
        !filtered.ids.contains(&ids[0]),
        "internal row 0 should filter external ID {} after sectioned restore, got {:?}",
        ids[0],
        filtered.ids
    );
}

#[test]
fn ivf_sq8_sectioned_snapshot_rejects_bad_centroid_length() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let centroids = sections
            .get_mut(IVF_SQ8_CENTROIDS_SECTION)
            .expect("centroids section");
        centroids.truncate(centroids.len() - std::mem::size_of::<f32>());
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "centroids len 11 != nlist * dim 12");
}

#[test]
fn ivf_sq8_sectioned_snapshot_rejects_bad_list_row() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut rows = decode_u64s(
            sections
                .get(IVF_SQ8_LIST_ROWS_SECTION)
                .expect("list rows section"),
        );
        rows[0] = 999;
        sections.insert(IVF_SQ8_LIST_ROWS_SECTION.to_string(), encode_u64s(&rows));
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "row 999 >= count 6");
}

#[test]
fn ivf_sq8_sectioned_snapshot_rejects_noncanonical_list_offset() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut list_offsets = decode_u64s(
            sections
                .get(IVF_SQ8_LIST_OFFSETS_SECTION)
                .expect("list offsets section"),
        );
        list_offsets[0] = 1;
        sections.insert(
            IVF_SQ8_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&list_offsets),
        );
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(
        load_error(&store),
        "list 0 offset 1 != expected canonical offset 0",
    );
}

#[test]
fn ivf_sq8_sectioned_snapshot_rejects_manifest_descriptor_length_mismatch() {
    let (index, _, _, _) = build_small_ivf_sq8();
    let store = sectioned_store_with(&index, |manifest, _sections| {
        let descriptor = manifest
            .sections
            .iter_mut()
            .find(|descriptor| descriptor.name == IVF_SQ8_IDS_SECTION)
            .expect("ids descriptor");
        descriptor.len += 1;
    });

    assert_codec_contains(load_error(&store), "descriptor length");
}

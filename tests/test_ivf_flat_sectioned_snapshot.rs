use std::collections::BTreeMap;

use hanns::api::{
    DataType, IndexConfig, IndexParams, IndexType, KnowhereError, MetricType, SearchRequest,
};
use hanns::faiss::{
    load_ivf_flat_index_from_artifact, load_ivf_flat_sectioned_snapshot,
    save_ivf_flat_sectioned_snapshot, IvfFlatIndex, IvfFlatSectionedSnapshot,
    IVF_FLAT_CENTROIDS_SECTION, IVF_FLAT_IDS_SECTION, IVF_FLAT_LIST_IDS_SECTION,
    IVF_FLAT_LIST_OFFSETS_SECTION, IVF_FLAT_LIST_SIZES_SECTION, IVF_FLAT_LIST_VECTORS_SECTION,
    IVF_FLAT_META_SECTION, IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT, IVF_FLAT_VECTORS_SECTION,
};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, IndexArtifactReader, IndexArtifactWriter, IndexManifest, MemoryArtifactStore,
};
use tempfile::tempdir;

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

fn sectioned_store_with(
    index: &IvfFlatIndex,
    edit: impl FnOnce(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = IvfFlatSectionedSnapshot::from_index(index).expect("snapshot");
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
    match load_ivf_flat_index_from_artifact(store) {
        Ok(_) => panic!("malformed sectioned IVF-Flat snapshot should error"),
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

#[test]
fn ivf_flat_sectioned_file_helpers_roundtrip() {
    let (index, _) = build_small_ivf_flat();
    let query = [1.05, 1.0, 1.0, 1.0];
    let req = SearchRequest {
        top_k: 4,
        nprobe: 3,
        ..Default::default()
    };
    let original = index.search(&query, &req).expect("original search");
    let dir = tempdir().expect("tempdir should build");

    save_ivf_flat_sectioned_snapshot(&index, dir.path()).expect("save sectioned snapshot");
    let loaded = load_ivf_flat_sectioned_snapshot(dir.path()).expect("load sectioned snapshot");
    let roundtrip = loaded.search(&query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn ivf_flat_sectioned_restore_preserves_legacy_next_id_for_auto_ids() {
    let (index, _) = build_small_ivf_flat();
    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let mut loaded = load_ivf_flat_index_from_artifact(&store).expect("load");
    let auto_vector = vec![3.0, 3.0, 3.0, 3.0];
    loaded.add(&auto_vector, None).expect("auto add");

    assert_eq!(loaded.get_vectors(&[6]), vec![Some(auto_vector)]);
    assert_eq!(loaded.get_vectors(&[16]), vec![None]);
}

#[test]
fn ivf_flat_sectioned_snapshot_rejects_malformed_list_count() {
    let (index, _) = build_small_ivf_flat();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut list_sizes = decode_u64s(
            sections
                .get(IVF_FLAT_LIST_SIZES_SECTION)
                .expect("list sizes section"),
        );
        let non_empty_list = list_sizes
            .iter()
            .position(|&size| size > 0)
            .expect("at least one populated list");
        list_sizes[non_empty_list] -= 1;

        let mut list_offsets = Vec::with_capacity(list_sizes.len());
        let mut running = 0u64;
        for &size in &list_sizes {
            list_offsets.push(running);
            running += size;
        }

        sections.insert(
            IVF_FLAT_LIST_SIZES_SECTION.to_string(),
            encode_u64s(&list_sizes),
        );
        sections.insert(
            IVF_FLAT_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&list_offsets),
        );

        let mut list_ids = sections
            .get(IVF_FLAT_LIST_IDS_SECTION)
            .expect("list ids section")
            .clone();
        list_ids.truncate(list_ids.len() - std::mem::size_of::<i64>());
        sections.insert(IVF_FLAT_LIST_IDS_SECTION.to_string(), list_ids);

        let mut list_vectors = sections
            .get(IVF_FLAT_LIST_VECTORS_SECTION)
            .expect("list vectors section")
            .clone();
        list_vectors.truncate(list_vectors.len() - 4 * std::mem::size_of::<f32>());
        sections.insert(IVF_FLAT_LIST_VECTORS_SECTION.to_string(), list_vectors);

        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "list_ids len 5 != count 6");
}

#[test]
fn ivf_flat_sectioned_snapshot_rejects_bad_centroid_length() {
    let (index, _) = build_small_ivf_flat();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let centroids = sections
            .get_mut(IVF_FLAT_CENTROIDS_SECTION)
            .expect("centroids section");
        centroids.truncate(centroids.len() - std::mem::size_of::<f32>());
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "centroids len 11 != nlist * dim 12");
}

#[test]
fn ivf_flat_sectioned_snapshot_rejects_noncanonical_list_offset() {
    let (index, _) = build_small_ivf_flat();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut list_offsets = decode_u64s(
            sections
                .get(IVF_FLAT_LIST_OFFSETS_SECTION)
                .expect("list offsets section"),
        );
        list_offsets[0] = 1;
        sections.insert(
            IVF_FLAT_LIST_OFFSETS_SECTION.to_string(),
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
fn ivf_flat_sectioned_snapshot_rejects_manifest_descriptor_length_mismatch() {
    let (index, _) = build_small_ivf_flat();
    let store = sectioned_store_with(&index, |manifest, _sections| {
        let descriptor = manifest
            .sections
            .iter_mut()
            .find(|descriptor| descriptor.name == IVF_FLAT_IDS_SECTION)
            .expect("ids descriptor");
        descriptor.len += 1;
    });

    assert_codec_contains(load_error(&store), "descriptor length");
}

#[test]
fn ivf_flat_sectioned_snapshot_rejects_unsupported_manifest_variant() {
    let (index, _) = build_small_ivf_flat();
    let store = sectioned_store_with(&index, |manifest, _sections| {
        manifest.variant = "ivf_flat_sections_v2".to_string();
    });

    assert_codec_contains(load_error(&store), "expected variant ivf_flat_sections_v1");
}

#[test]
fn ivf_flat_sectioned_snapshot_roundtrips_get_vectors() {
    let (index, _) = build_small_ivf_flat();
    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_ivf_flat_index_from_artifact(&store).expect("load");
    let ids = [10, 12, 15, 999];

    assert_eq!(loaded.get_vectors(&ids), index.get_vectors(&ids));
}

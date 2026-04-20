use std::collections::BTreeMap;

use hanns::api::{IndexConfig, IndexType, KnowhereError, MetricType, SearchRequest, SqMode};
use hanns::faiss::hnsw_snapshot::{
    HNSW_LEVELS_SECTION, HNSW_META_SECTION, HNSW_NEIGHBOR_DISTS_SECTION, HNSW_NEIGHBOR_IDS_SECTION,
    HNSW_NEIGHBOR_OFFSETS_SECTION,
};
use hanns::faiss::{HnswIndex, HnswSectionedSnapshot, HnswSnapshotLoader};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, MemoryArtifactStore,
};
use serde_json::Value;
use tempfile::tempdir;

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

fn build_sequential_hnsw() -> HnswIndex {
    let dim = 4;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(4);
    cfg.params.ef_construction = Some(16);
    cfg.params.ef_search = Some(16);
    cfg.params.random_seed = Some(7);

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];

    let mut index = HnswIndex::new(&cfg).expect("hnsw index should build");
    index.train(&vectors).expect("hnsw index should train");
    index
        .add(&vectors, None)
        .expect("hnsw sequential vectors should add");
    index
}

fn sectioned_store_with(
    index: &HnswIndex,
    mut edit: impl FnMut(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = HnswSectionedSnapshot::from_index(index).expect("snapshot");
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
    for descriptor in &mut manifest.sections {
        descriptor.len = sections
            .get(&descriptor.name)
            .unwrap_or_else(|| panic!("section {} should exist", descriptor.name))
            .len() as u64;
    }

    let mut store = MemoryArtifactStore::default();
    for (name, bytes) in sections {
        store.write_section(&name, &bytes).expect("section write");
    }
    store.finish_manifest(&manifest).expect("manifest write");
    store
}

fn edit_meta(sections: &mut BTreeMap<String, Vec<u8>>, edit: impl FnOnce(&mut Value)) {
    let meta = sections
        .get_mut(HNSW_META_SECTION)
        .expect("metadata section should exist");
    let mut value: Value = serde_json::from_slice(meta).expect("metadata should decode");
    edit(&mut value);
    *meta = serde_json::to_vec_pretty(&value).expect("metadata should encode");
}

fn load_error(store: &MemoryArtifactStore) -> KnowhereError {
    match HnswSnapshotLoader.load_snapshot(store, LoadMode::OwnedMemory) {
        Ok(_) => panic!("malformed sectioned HNSW snapshot should error"),
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

fn assert_search_matches(index: &HnswIndex, runtime: &dyn hanns::kernel::AnnRuntime) {
    let req = SearchRequest {
        top_k: 3,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };
    let query = [0.0, 0.0, 0.0, 0.0];
    let expected = index.search(&query, &req).expect("search");
    let mut ids = vec![-1; req.top_k];
    let mut dists = vec![f32::INFINITY; req.top_k];

    let n = runtime
        .search_into(&query, &req, &mut ids, &mut dists)
        .expect("runtime search");

    assert_eq!(&ids[..n], expected.ids.as_slice());
    assert_eq!(n, expected.distances.len());
    for (actual, expected) in dists[..n].iter().zip(expected.distances.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "distance mismatch: actual {actual}, expected {expected}"
        );
    }
}

fn encode_i64s(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_u64s(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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
fn hnsw_sectioned_snapshot_roundtrips_distances_and_deleted_filter() {
    let mut index = build_small_hnsw();
    index.mark_deleted(11);
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loader = HnswSnapshotLoader;
    let runtime = loader
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("load");
    assert_search_matches(&index, runtime.as_ref());

    let req = SearchRequest {
        top_k: 4,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };
    let query = [1.0, 0.0, 0.0, 0.0];
    let mut ids = vec![-1; req.top_k];
    let mut dists = vec![f32::INFINITY; req.top_k];
    let n = runtime
        .search_into(&query, &req, &mut ids, &mut dists)
        .expect("runtime search");

    assert!(
        !ids[..n].contains(&11),
        "deleted id should remain filtered after sectioned load: {:?}",
        &ids[..n]
    );
}

#[test]
fn hnsw_sectioned_snapshot_roundtrips_through_file_store() {
    let index = build_sequential_hnsw();
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let dir = tempdir().expect("tempdir should build");
    let mut writer = FileArtifactStore::new(dir.path()).expect("file store should open");

    snapshot.write_snapshot(&mut writer).expect("write");

    let reader = FileArtifactStore::new(dir.path()).expect("file store should reopen");
    let runtime = HnswSnapshotLoader
        .load_snapshot(&reader, LoadMode::OwnedMemory)
        .expect("load");

    assert_search_matches(&index, runtime.as_ref());
}

#[test]
fn hnsw_sectioned_snapshot_rejects_invalid_upper_layer_neighbor() {
    let index = build_small_hnsw();
    let store = sectioned_store_with(&index, |_, sections| {
        sections.insert(HNSW_LEVELS_SECTION.to_string(), encode_u32s(&[1, 0, 0, 0]));
        sections.insert(
            HNSW_NEIGHBOR_OFFSETS_SECTION.to_string(),
            encode_u64s(&[0, 0, 1, 1, 1, 1]),
        );
        sections.insert(HNSW_NEIGHBOR_IDS_SECTION.to_string(), encode_i64s(&[11]));
        sections.insert(HNSW_NEIGHBOR_DISTS_SECTION.to_string(), encode_f32s(&[1.0]));
        edit_meta(sections, |meta| {
            meta["max_level"] = Value::from(1);
            meta["entry_point"] = Value::from(10);
        });
    });

    assert_codec_contains(
        load_error(&store),
        "neighbor id 11 has max layer 0 below layer 1",
    );
}

#[test]
fn hnsw_sectioned_snapshot_rejects_bad_max_level_and_entry_point_level() {
    let index = build_small_hnsw();
    let bad_max_level = sectioned_store_with(&index, |_, sections| {
        sections.insert(HNSW_LEVELS_SECTION.to_string(), encode_u32s(&[1, 0, 0, 0]));
        sections.insert(
            HNSW_NEIGHBOR_OFFSETS_SECTION.to_string(),
            encode_u64s(&[0, 0, 0, 0, 0, 0]),
        );
        sections.insert(HNSW_NEIGHBOR_IDS_SECTION.to_string(), Vec::new());
        sections.insert(HNSW_NEIGHBOR_DISTS_SECTION.to_string(), Vec::new());
        edit_meta(sections, |meta| {
            meta["max_level"] = Value::from(0);
            meta["entry_point"] = Value::from(10);
        });
    });
    assert_codec_contains(load_error(&bad_max_level), "max_level 0 != max(levels) 1");

    let bad_entry_point_level = sectioned_store_with(&index, |_, sections| {
        sections.insert(HNSW_LEVELS_SECTION.to_string(), encode_u32s(&[1, 0, 0, 0]));
        sections.insert(
            HNSW_NEIGHBOR_OFFSETS_SECTION.to_string(),
            encode_u64s(&[0, 0, 0, 0, 0, 0]),
        );
        sections.insert(HNSW_NEIGHBOR_IDS_SECTION.to_string(), Vec::new());
        sections.insert(HNSW_NEIGHBOR_DISTS_SECTION.to_string(), Vec::new());
        edit_meta(sections, |meta| {
            meta["max_level"] = Value::from(1);
            meta["entry_point"] = Value::from(11);
        });
    });
    assert_codec_contains(
        load_error(&bad_entry_point_level),
        "entry point id 11 has level 0 != max_level 1",
    );
}

#[test]
fn hnsw_sectioned_snapshot_rejects_degree_overflow() {
    let index = build_small_hnsw();
    let store = sectioned_store_with(&index, |_, sections| {
        sections.insert(HNSW_LEVELS_SECTION.to_string(), encode_u32s(&[0, 0, 0, 0]));
        sections.insert(
            HNSW_NEIGHBOR_OFFSETS_SECTION.to_string(),
            encode_u64s(&[0, 9, 9, 9, 9]),
        );
        sections.insert(
            HNSW_NEIGHBOR_IDS_SECTION.to_string(),
            encode_i64s(&[10, 11, 12, 13, 10, 11, 12, 13, 10]),
        );
        sections.insert(
            HNSW_NEIGHBOR_DISTS_SECTION.to_string(),
            encode_f32s(&[1.0; 9]),
        );
        edit_meta(sections, |meta| {
            meta["max_level"] = Value::from(0);
            meta["entry_point"] = Value::from(10);
        });
    });

    assert_codec_contains(
        load_error(&store),
        "layer 0 degree 9 exceeds configured max 8",
    );
}

#[test]
fn hnsw_sectioned_snapshot_rejects_sq_import_until_supported() {
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

    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    assert_codec_contains(
        load_error(&store),
        "sectioned HNSW SQ snapshot import is not supported yet",
    );
}

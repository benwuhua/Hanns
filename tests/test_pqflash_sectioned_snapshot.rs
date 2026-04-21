use std::collections::BTreeMap;

use hanns::api::{KnowhereError, MetricType, SearchRequest};
use hanns::faiss::{
    default_ann_snapshot_registry, load_pqflash_index_from_artifact,
    load_pqflash_sectioned_snapshot, save_pqflash_sectioned_snapshot, AisaqConfig, PQFlashIndex,
    PqFlashSectionedSnapshot, PQFLASH_NEIGHBOR_COUNTS_SECTION, PQFLASH_NEIGHBOR_IDS_SECTION,
    PQFLASH_SECTIONS_SNAPSHOT_VARIANT,
};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, IndexArtifactReader, IndexArtifactWriter, IndexManifest, LoadMode,
    MemoryArtifactStore,
};
use tempfile::tempdir;

fn build_small_pqflash() -> (PQFlashIndex, Vec<f32>, Vec<i64>) {
    let dim = 4;
    let n = 48;
    let config = AisaqConfig {
        max_degree: 8,
        search_list_size: 32,
        beamwidth: 4,
        disk_pq_dims: 0,
        num_entry_points: 2,
        random_seed: 7,
        run_refine_pass: true,
        num_refine_passes: 1,
        ..Default::default()
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let cluster = (i % 4) as f32 * 5.0;
            vectors.push(cluster + ((i * 17 + j * 11) % 29) as f32 / 29.0);
        }
    }
    let ids: Vec<i64> = (0..n as i64).map(|id| id * 13 + 3).collect();

    let mut index = PQFlashIndex::new(config, MetricType::L2, dim).expect("index");
    index
        .add_with_ids(&vectors, Some(&ids))
        .expect("vectors should add");
    (index, vectors, ids)
}

fn build_small_pqflash_with_pq() -> (PQFlashIndex, Vec<f32>) {
    let dim = 4;
    let n = 80;
    let config = AisaqConfig {
        max_degree: 8,
        search_list_size: 32,
        beamwidth: 4,
        disk_pq_dims: 2,
        num_entry_points: 2,
        random_seed: 11,
        run_refine_pass: true,
        num_refine_passes: 1,
        ..Default::default()
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let cluster = (i % 5) as f32 * 4.0;
            vectors.push(cluster + ((i * 23 + j * 7) % 37) as f32 / 37.0);
        }
    }

    let mut index = PQFlashIndex::new(config, MetricType::L2, dim).expect("index");
    index.add(&vectors).expect("vectors should add");
    (index, vectors)
}

fn build_small_pqflash_with_hvq_sq8() -> (PQFlashIndex, Vec<f32>) {
    let dim = 4;
    let n = 80;
    let config = AisaqConfig {
        max_degree: 8,
        search_list_size: 32,
        beamwidth: 4,
        disk_pq_dims: 0,
        num_entry_points: 2,
        random_seed: 19,
        run_refine_pass: true,
        num_refine_passes: 1,
        use_hvq: true,
        hvq_nbits: 4,
        use_sq8_prefilter: true,
        ..Default::default()
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let cluster = (i % 5) as f32 * 4.0;
            vectors.push(cluster + ((i * 31 + j * 13) % 41) as f32 / 41.0);
        }
    }

    let mut index = PQFlashIndex::new(config, MetricType::L2, dim).expect("index");
    index.add(&vectors).expect("vectors should add");
    (index, vectors)
}

fn sectioned_store_with(
    index: &PQFlashIndex,
    edit: impl FnOnce(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = PqFlashSectionedSnapshot::from_index(index).expect("snapshot");
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

fn decode_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn load_error(store: &MemoryArtifactStore) -> KnowhereError {
    match load_pqflash_index_from_artifact(store) {
        Ok(_) => panic!("malformed PQFlash sectioned snapshot should error"),
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

#[test]
fn pqflash_sectioned_snapshot_roundtrips_search_results() {
    let (index, vectors, _) = build_small_pqflash();
    let query = &vectors[0..4];
    let original = index.search(query, 8).expect("original search");

    let snapshot = PqFlashSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");
    assert_eq!(
        store.manifest().expect("manifest").variant,
        PQFLASH_SECTIONS_SNAPSHOT_VARIANT
    );

    let loaded = load_pqflash_index_from_artifact(&store).expect("load");
    let roundtrip = loaded.search(query, 8).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn pqflash_sectioned_file_helpers_roundtrip() {
    let (index, vectors, _) = build_small_pqflash();
    let query = &vectors[4..8];
    let original = index.search(query, 8).expect("original search");
    let dir = tempdir().expect("tempdir");

    save_pqflash_sectioned_snapshot(&index, dir.path()).expect("save");
    let loaded = load_pqflash_sectioned_snapshot(dir.path()).expect("load");
    let roundtrip = loaded.search(query, 8).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn pqflash_sectioned_snapshot_preserves_pq_payload() {
    let (index, vectors) = build_small_pqflash_with_pq();
    let export = index.export_sectioned_snapshot().expect("export");
    assert!(export.pq_code_size > 0);
    assert!(export.pq.is_some());
    assert_eq!(
        export.node_pq_codes.len(),
        export.count * export.pq_code_size
    );

    let query = &vectors[0..4];
    let original = index.search(query, 8).expect("original search");
    let snapshot = PqFlashSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_pqflash_index_from_artifact(&store).expect("load");
    let loaded_export = loaded.export_sectioned_snapshot().expect("loaded export");
    assert_eq!(loaded_export.pq_code_size, export.pq_code_size);
    assert_eq!(loaded_export.node_pq_codes, export.node_pq_codes);
    assert_eq!(
        loaded_export.pq.as_ref().unwrap().centroids,
        export.pq.as_ref().unwrap().centroids
    );

    let roundtrip = loaded.search(query, 8).expect("loaded search");
    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn pqflash_sectioned_snapshot_preserves_hvq_and_sq8_payloads() {
    let (index, vectors) = build_small_pqflash_with_hvq_sq8();
    let export = index.export_sectioned_snapshot().expect("export");
    assert!(export.hvq.is_some());
    assert!(export.sq8.is_some());

    let query = &vectors[0..4];
    let original = index.search(query, 8).expect("original search");
    let snapshot = PqFlashSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_pqflash_index_from_artifact(&store).expect("load");
    let loaded_export = loaded.export_sectioned_snapshot().expect("loaded export");
    assert_eq!(
        loaded_export.hvq.as_ref().unwrap().codes,
        export.hvq.as_ref().unwrap().codes
    );
    assert_eq!(
        loaded_export.hvq.as_ref().unwrap().rotation_matrix,
        export.hvq.as_ref().unwrap().rotation_matrix
    );
    assert_eq!(
        loaded_export.sq8.as_ref().unwrap().codes,
        export.sq8.as_ref().unwrap().codes
    );
    assert_eq!(
        loaded_export.sq8.as_ref().unwrap().scale,
        export.sq8.as_ref().unwrap().scale
    );

    let roundtrip = loaded.search(query, 8).expect("loaded search");
    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn pqflash_registry_loads_diskann_runtime() {
    let (index, vectors, _) = build_small_pqflash();
    let snapshot = PqFlashSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let registry = default_ann_snapshot_registry().expect("registry");
    assert!(registry
        .registered_variants()
        .contains(&PQFLASH_SECTIONS_SNAPSHOT_VARIANT));
    let runtime = registry
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime load");

    assert_eq!(runtime.family(), IndexFamily::DiskAnn);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), index.len());

    let mut ids = vec![-1; 4];
    let mut dists = vec![f32::MAX; 4];
    let count = runtime
        .search_into(
            &vectors[0..4],
            &SearchRequest {
                top_k: 4,
                ..Default::default()
            },
            &mut ids,
            &mut dists,
        )
        .expect("runtime search");
    assert_eq!(count, 4);
    assert!(ids[0] >= 0);
}

#[test]
fn pqflash_sectioned_snapshot_rejects_bad_neighbor_count() {
    let (index, _, _) = build_small_pqflash();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut counts = decode_u32s(
            sections
                .get(PQFLASH_NEIGHBOR_COUNTS_SECTION)
                .expect("neighbor counts"),
        );
        counts[0] = u32::MAX;
        sections.insert(
            PQFLASH_NEIGHBOR_COUNTS_SECTION.to_string(),
            encode_u32s(&counts),
        );
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "degree");
}

#[test]
fn pqflash_sectioned_snapshot_rejects_bad_neighbor_id() {
    let (index, _, _) = build_small_pqflash();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut counts = decode_u32s(
            sections
                .get(PQFLASH_NEIGHBOR_COUNTS_SECTION)
                .expect("neighbor counts"),
        );
        counts[0] = counts[0].max(1);
        let mut neighbors = decode_u32s(
            sections
                .get(PQFLASH_NEIGHBOR_IDS_SECTION)
                .expect("neighbor ids"),
        );
        neighbors[0] = u32::MAX;
        sections.insert(
            PQFLASH_NEIGHBOR_COUNTS_SECTION.to_string(),
            encode_u32s(&counts),
        );
        sections.insert(
            PQFLASH_NEIGHBOR_IDS_SECTION.to_string(),
            encode_u32s(&neighbors),
        );
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "neighbor");
}

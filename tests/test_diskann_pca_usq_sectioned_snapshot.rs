use std::collections::BTreeMap;

use hanns::api::{KnowhereError, MetricType};
use hanns::faiss::{
    load_diskann_pca_usq_index_from_artifact, load_diskann_pca_usq_sectioned_snapshot,
    save_diskann_pca_usq_sectioned_snapshot, AisaqConfig, DiskAnnPcaUsqConfig, DiskAnnPcaUsqIndex,
    DiskAnnPcaUsqSectionedSnapshot, DISKANN_PCA_USQ_PACKED_BITS_SECTION,
    DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT,
};
use hanns::storage::{
    AnnSnapshot, IndexArtifactReader, IndexArtifactWriter, IndexManifest, MemoryArtifactStore,
};
use tempfile::tempdir;

fn build_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let cluster = (i % 4) as f32 * 3.0;
            vectors.push(cluster + ((i * 19 + j * 11) % 31) as f32 / 31.0);
        }
    }
    vectors
}

fn build_diskann_pca_usq() -> (DiskAnnPcaUsqIndex, Vec<f32>) {
    let dim = 16;
    let n = 64;
    let config = DiskAnnPcaUsqConfig {
        base: AisaqConfig {
            max_degree: 8,
            search_list_size: 32,
            beamwidth: 4,
            num_entry_points: 2,
            random_seed: 23,
            num_refine_passes: 1,
            ..Default::default()
        },
        pca_dim: 8,
        bits_per_dim: 4,
        rotation_seed: 5,
        rerank_k: 16,
    };
    let vectors = build_vectors(n, dim);
    let mut index = DiskAnnPcaUsqIndex::new(dim, MetricType::L2, config).expect("new index");
    index.build(&vectors, n, dim).expect("build index");
    (index, vectors)
}

fn sectioned_store_with(
    index: &DiskAnnPcaUsqIndex,
    edit: impl FnOnce(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = DiskAnnPcaUsqSectionedSnapshot::from_index(index).expect("snapshot");
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
    match load_diskann_pca_usq_index_from_artifact(store) {
        Ok(_) => panic!("malformed DiskAnnPcaUsq sectioned snapshot should error"),
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
fn diskann_pca_usq_sectioned_snapshot_roundtrips_search_results() {
    let (index, vectors) = build_diskann_pca_usq();
    let query = &vectors[0..16];
    let original = index.search(query, 8).expect("original search");

    let snapshot = DiskAnnPcaUsqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");
    assert_eq!(
        store.manifest().expect("manifest").variant,
        DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT
    );

    let loaded = load_diskann_pca_usq_index_from_artifact(&store).expect("load");
    let roundtrip = loaded.search(query, 8).expect("loaded search");
    assert_eq!(roundtrip, original);
}

#[test]
fn diskann_pca_usq_sectioned_file_helpers_roundtrip() {
    let (index, vectors) = build_diskann_pca_usq();
    let query = &vectors[16..32];
    let original = index.search(query, 8).expect("original search");
    let dir = tempdir().expect("tempdir");

    save_diskann_pca_usq_sectioned_snapshot(&index, dir.path()).expect("save");
    let loaded = load_diskann_pca_usq_sectioned_snapshot(dir.path()).expect("load");
    let roundtrip = loaded.search(query, 8).expect("loaded search");
    assert_eq!(roundtrip, original);
}

#[test]
fn diskann_pca_usq_sectioned_snapshot_preserves_pca_and_usq_payloads() {
    let (index, _) = build_diskann_pca_usq();
    let export = index.export_sectioned_snapshot().expect("export");
    assert!(export.pca.is_some());
    assert_eq!(export.d_proj, 8);
    assert_eq!(export.quantizer_centroid.len(), 8);
    assert_eq!(export.packed_bits.len(), export.n * export.code_bytes);
    assert_eq!(export.sign_bits.len(), export.n * export.sign_bytes);

    let snapshot = DiskAnnPcaUsqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");
    let loaded = load_diskann_pca_usq_index_from_artifact(&store).expect("load");
    let loaded_export = loaded.export_sectioned_snapshot().expect("loaded export");

    assert_eq!(loaded_export.quantizer_centroid, export.quantizer_centroid);
    assert_eq!(loaded_export.packed_bits, export.packed_bits);
    assert_eq!(
        loaded_export.pca.as_ref().unwrap().components,
        export.pca.as_ref().unwrap().components
    );
}

#[test]
fn diskann_pca_usq_sectioned_snapshot_rejects_bad_packed_length() {
    let (index, _) = build_diskann_pca_usq();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let packed = sections
            .get_mut(DISKANN_PCA_USQ_PACKED_BITS_SECTION)
            .expect("packed section");
        packed.pop();
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "packed_bits len");
}

use std::collections::BTreeMap;

use hanns::api::{KnowhereError, MetricType, SearchRequest};
use hanns::faiss::{
    load_ivf_usq_index_from_artifact, load_ivf_usq_sectioned_snapshot,
    save_ivf_usq_sectioned_snapshot, IvfUsqConfig, IvfUsqIndex, IvfUsqSectionedSnapshot,
    IVF_USQ_CENTROIDS_SECTION, IVF_USQ_LIST_IDS_SECTION, IVF_USQ_LIST_OFFSETS_SECTION,
    IVF_USQ_LIST_SIZES_SECTION, IVF_USQ_META_SECTION, IVF_USQ_NORMS_SECTION,
    IVF_USQ_NORMS_SQ_SECTION, IVF_USQ_PACKED_BITS_SECTION, IVF_USQ_QUANTIZER_CENTROID_SECTION,
    IVF_USQ_QUANT_QUALITIES_SECTION, IVF_USQ_SECTIONS_SNAPSHOT_VARIANT, IVF_USQ_SIGN_BITS_SECTION,
    IVF_USQ_VMAXS_SECTION,
};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, IndexArtifactReader, IndexArtifactWriter, IndexManifest, MemoryArtifactStore,
};
use tempfile::tempdir;

fn build_small_ivf_usq() -> (IvfUsqIndex, Vec<f32>, Vec<i64>, usize) {
    let dim = 8;
    let n = 48;
    let nlist = 4;
    let config = IvfUsqConfig::new(dim, nlist, 4)
        .with_nprobe(4)
        .with_metric(MetricType::L2)
        .with_rotation_seed(17)
        .with_rerank_k(32);
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let cluster = (i % nlist) as f32 * 7.0;
            vectors.push(cluster + ((i * 13 + j * 5) % 31) as f32 / 31.0);
        }
    }
    let ids: Vec<i64> = (0..n as i64).map(|id| id * 3 + 5).collect();

    let mut index = IvfUsqIndex::new(config);
    index.train(&vectors).expect("ivf-usq index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("ivf-usq vectors should add");
    (index, vectors, ids, nlist)
}

fn sectioned_store_with(
    index: &IvfUsqIndex,
    edit: impl FnOnce(&mut IndexManifest, &mut BTreeMap<String, Vec<u8>>),
) -> MemoryArtifactStore {
    let snapshot = IvfUsqSectionedSnapshot::from_index(index).expect("snapshot");
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
    match load_ivf_usq_index_from_artifact(store) {
        Ok(_) => panic!("malformed sectioned IVF-USQ snapshot should error"),
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
fn ivf_usq_rejects_untrained_sectioned_export() {
    let index = IvfUsqIndex::new(IvfUsqConfig::new(8, 4, 4));

    let error = index
        .export_sectioned_snapshot()
        .expect_err("untrained IVF-USQ sectioned export should fail");

    assert!(
        error.to_string().contains("untrained"),
        "error should mention trained state, got {error}"
    );
}

#[test]
fn ivf_usq_exports_sectioned_snapshot_shape() {
    let (index, _, _, nlist) = build_small_ivf_usq();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 8);
    assert_eq!(export.ntotal, 48);
    assert_eq!(export.nlist, nlist);
    assert_eq!(export.bits_per_dim, 4);
    assert_eq!(export.padded_dim, 64);
    assert_eq!(export.code_bytes, 32);
    assert_eq!(export.sign_bytes, 8);
    assert_eq!(export.centroids.len(), nlist * export.dim);
    assert_eq!(export.quantizer_centroid.len(), export.dim);
    assert_eq!(export.list_offsets.len(), nlist);
    assert_eq!(export.list_sizes.len(), nlist);

    let list_total = export.list_sizes.iter().sum::<u64>();
    assert_eq!(list_total, export.list_ids.len() as u64);
    assert_eq!(export.list_ids.len(), export.ntotal);
    assert_eq!(export.packed_bits.len(), export.ntotal * export.code_bytes);
    assert_eq!(export.sign_bits.len(), export.ntotal * export.sign_bytes);
    assert_eq!(export.norms.len(), export.ntotal);
    assert_eq!(export.norms_sq.len(), export.ntotal);
    assert_eq!(export.vmaxs.len(), export.ntotal);
    assert_eq!(export.quant_qualities.len(), export.ntotal);
}

#[test]
fn ivf_usq_sectioned_snapshot_writes_expected_sections() {
    let (index, _, _, _) = build_small_ivf_usq();
    let export = index.export_sectioned_snapshot().expect("export");
    let snapshot = IvfUsqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();

    snapshot.write_snapshot(&mut store).expect("write");
    let manifest = store.manifest().expect("manifest");

    assert_eq!(manifest.family, IndexFamily::Ivf);
    assert_eq!(manifest.variant, IVF_USQ_SECTIONS_SNAPSHOT_VARIANT);
    assert!(store.section_len(IVF_USQ_META_SECTION).unwrap() > 0);

    for (section, expected_len) in [
        (IVF_USQ_CENTROIDS_SECTION, export.centroids.len() as u64 * 4),
        (
            IVF_USQ_QUANTIZER_CENTROID_SECTION,
            export.quantizer_centroid.len() as u64 * 4,
        ),
        (
            IVF_USQ_LIST_OFFSETS_SECTION,
            export.list_offsets.len() as u64 * 8,
        ),
        (
            IVF_USQ_LIST_SIZES_SECTION,
            export.list_sizes.len() as u64 * 8,
        ),
        (IVF_USQ_LIST_IDS_SECTION, export.list_ids.len() as u64 * 8),
        (IVF_USQ_PACKED_BITS_SECTION, export.packed_bits.len() as u64),
        (IVF_USQ_SIGN_BITS_SECTION, export.sign_bits.len() as u64),
        (IVF_USQ_NORMS_SECTION, export.norms.len() as u64 * 4),
        (IVF_USQ_NORMS_SQ_SECTION, export.norms_sq.len() as u64 * 4),
        (IVF_USQ_VMAXS_SECTION, export.vmaxs.len() as u64 * 4),
        (
            IVF_USQ_QUANT_QUALITIES_SECTION,
            export.quant_qualities.len() as u64 * 4,
        ),
    ] {
        assert_eq!(store.section_len(section).unwrap(), expected_len);
    }

    for section in [
        IVF_USQ_META_SECTION,
        IVF_USQ_CENTROIDS_SECTION,
        IVF_USQ_QUANTIZER_CENTROID_SECTION,
        IVF_USQ_LIST_OFFSETS_SECTION,
        IVF_USQ_LIST_SIZES_SECTION,
        IVF_USQ_LIST_IDS_SECTION,
        IVF_USQ_PACKED_BITS_SECTION,
        IVF_USQ_SIGN_BITS_SECTION,
        IVF_USQ_NORMS_SECTION,
        IVF_USQ_NORMS_SQ_SECTION,
        IVF_USQ_VMAXS_SECTION,
        IVF_USQ_QUANT_QUALITIES_SECTION,
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
fn ivf_usq_sectioned_snapshot_roundtrips_search_results() {
    let (index, vectors, _, _) = build_small_ivf_usq();
    let query = &vectors[0..8];
    let req = SearchRequest {
        top_k: 6,
        nprobe: 4,
        ..Default::default()
    };
    let original = index.search(query, &req).expect("original search");

    let snapshot = IvfUsqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loaded = load_ivf_usq_index_from_artifact(&store).expect("load");
    let roundtrip = loaded.search(query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn ivf_usq_sectioned_file_helpers_roundtrip() {
    let (index, vectors, _, _) = build_small_ivf_usq();
    let query = &vectors[8..16];
    let req = SearchRequest {
        top_k: 6,
        nprobe: 4,
        ..Default::default()
    };
    let original = index.search(query, &req).expect("original search");
    let dir = tempdir().expect("tempdir should build");

    save_ivf_usq_sectioned_snapshot(&index, dir.path()).expect("save sectioned snapshot");
    let loaded = load_ivf_usq_sectioned_snapshot(dir.path()).expect("load sectioned snapshot");
    let roundtrip = loaded.search(query, &req).expect("loaded search");

    assert_eq!(roundtrip.ids, original.ids);
    assert_eq!(roundtrip.distances, original.distances);
}

#[test]
fn ivf_usq_sectioned_snapshot_rejects_bad_centroid_length() {
    let (index, _, _, _) = build_small_ivf_usq();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let centroids = sections
            .get_mut(IVF_USQ_CENTROIDS_SECTION)
            .expect("centroids section");
        centroids.truncate(centroids.len() - std::mem::size_of::<f32>());
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "centroids len");
}

#[test]
fn ivf_usq_sectioned_snapshot_rejects_bad_packed_length() {
    let (index, _, _, _) = build_small_ivf_usq();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let packed = sections
            .get_mut(IVF_USQ_PACKED_BITS_SECTION)
            .expect("packed bits section");
        packed.pop();
        refresh_manifest_lengths(manifest, sections);
    });

    assert_codec_contains(load_error(&store), "packed_bits len");
}

#[test]
fn ivf_usq_sectioned_snapshot_rejects_noncanonical_list_offset() {
    let (index, _, _, _) = build_small_ivf_usq();
    let store = sectioned_store_with(&index, |manifest, sections| {
        let mut list_offsets = decode_u64s(
            sections
                .get(IVF_USQ_LIST_OFFSETS_SECTION)
                .expect("list offsets section"),
        );
        list_offsets[0] = 1;
        sections.insert(
            IVF_USQ_LIST_OFFSETS_SECTION.to_string(),
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
fn ivf_usq_sectioned_snapshot_rejects_manifest_descriptor_length_mismatch() {
    let (index, _, _, _) = build_small_ivf_usq();
    let store = sectioned_store_with(&index, |manifest, _sections| {
        let descriptor = manifest
            .sections
            .iter_mut()
            .find(|descriptor| descriptor.name == IVF_USQ_LIST_IDS_SECTION)
            .expect("ids descriptor");
        descriptor.len += 1;
    });

    assert_codec_contains(load_error(&store), "descriptor length");
}

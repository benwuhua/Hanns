use std::cell::Cell;
use std::collections::BTreeMap;

use hanns::api::{IndexConfig, IndexType, KnowhereError, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, HnswSectionedSnapshot, HnswSnapshotLoader};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, AnnSnapshotLoader, CallbackArtifactReader, IndexArtifactReader, IndexManifest,
    LoadMode, ManifestFeatures, MemoryArtifactStore, SectionDescriptor,
};

fn manifest() -> IndexManifest {
    IndexManifest {
        version: 1,
        family: IndexFamily::Hnsw,
        variant: "hnsw_sections_v1".to_string(),
        dim: 4,
        metric: "l2".to_string(),
        count: 2,
        supported_load_modes: vec![LoadMode::OwnedMemory],
        features: ManifestFeatures {
            raw_vectors: true,
            graph_payload: true,
            quantized_payload: false,
            compressed_vectors: false,
        },
        sections: vec![
            SectionDescriptor {
                name: "hnsw.meta.json".to_string(),
                len: 2,
                checksum: None,
            },
            SectionDescriptor {
                name: "hnsw.vectors.f32".to_string(),
                len: 4,
                checksum: None,
            },
        ],
    }
}

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
fn callback_artifact_reader_reads_sections_from_external_callbacks() {
    let sections = BTreeMap::from([
        ("hnsw.meta.json".to_string(), vec![1, 2]),
        ("hnsw.vectors.f32".to_string(), vec![3, 4, 5, 6]),
    ]);
    let range_calls = Cell::new(0usize);
    let reader = CallbackArtifactReader::new(
        manifest(),
        |name: &str| -> hanns::Result<u64> {
            sections
                .get(name)
                .map(|bytes| bytes.len() as u64)
                .ok_or_else(|| KnowhereError::Codec(format!("missing test section: {name}")))
        },
        |name: &str, offset, len| -> hanns::Result<Vec<u8>> {
            range_calls.set(range_calls.get() + 1);
            let section = sections
                .get(name)
                .ok_or_else(|| KnowhereError::Codec(format!("missing test section: {name}")))?;
            let offset = offset as usize;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| KnowhereError::Codec("test range overflow".to_string()))?;
            section
                .get(offset..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| KnowhereError::Codec(format!("test range out of bounds: {name}")))
        },
    );

    assert_eq!(reader.manifest().unwrap().family, IndexFamily::Hnsw);
    assert_eq!(reader.section_len("hnsw.vectors.f32").unwrap(), 4);
    assert_eq!(
        reader.read_range("hnsw.vectors.f32", 1, 2).unwrap(),
        vec![4, 5]
    );
    assert_eq!(&*reader.read_section("hnsw.meta.json").unwrap(), &[1, 2]);
    assert_eq!(range_calls.get(), 2);
}

#[test]
fn callback_artifact_reader_propagates_len_errors_before_full_section_read() {
    let range_called = Cell::new(false);
    let reader = CallbackArtifactReader::new(
        manifest(),
        |name: &str| -> hanns::Result<u64> {
            Err(KnowhereError::Codec(format!(
                "external len unavailable: {name}"
            )))
        },
        |_: &str, _, _| -> hanns::Result<Vec<u8>> {
            range_called.set(true);
            Ok(Vec::new())
        },
    );

    let error = reader
        .read_section("hnsw.vectors.f32")
        .expect_err("length callback error should be returned");
    assert!(
        error.to_string().contains("external len unavailable"),
        "unexpected error: {error}"
    );
    assert!(!range_called.get(), "range callback should not be called");
}

#[test]
fn callback_artifact_reader_loads_hnsw_sectioned_snapshot() {
    let index = build_small_hnsw();
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot should build");
    let mut source = MemoryArtifactStore::default();
    snapshot
        .write_snapshot(&mut source)
        .expect("snapshot should write");

    let manifest = source.manifest().expect("manifest should exist").clone();
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

    let reader = CallbackArtifactReader::new(
        manifest,
        |name: &str| -> hanns::Result<u64> {
            sections
                .get(name)
                .map(|bytes| bytes.len() as u64)
                .ok_or_else(|| KnowhereError::Codec(format!("missing test section: {name}")))
        },
        |name: &str, offset, len| -> hanns::Result<Vec<u8>> {
            let section = sections
                .get(name)
                .ok_or_else(|| KnowhereError::Codec(format!("missing test section: {name}")))?;
            let offset = offset as usize;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| KnowhereError::Codec("test range overflow".to_string()))?;
            section
                .get(offset..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| KnowhereError::Codec(format!("test range out of bounds: {name}")))
        },
    );

    let runtime = HnswSnapshotLoader
        .load_snapshot(&reader, LoadMode::OwnedMemory)
        .expect("callback-backed HNSW snapshot should load");

    let query = [0.0, 0.0, 0.0, 0.0];
    let req = SearchRequest {
        top_k: 3,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };
    let expected = index.search(&query, &req).expect("original search");
    let mut ids = vec![-1; req.top_k];
    let mut dists = vec![f32::INFINITY; req.top_k];

    let n = runtime
        .search_into(&query, &req, &mut ids, &mut dists)
        .expect("runtime search");

    assert_eq!(&ids[..n], expected.ids.as_slice());
    assert_eq!(n, expected.distances.len());
}

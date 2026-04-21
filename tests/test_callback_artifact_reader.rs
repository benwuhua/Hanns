use std::cell::Cell;
use std::collections::BTreeMap;

use hanns::api::KnowhereError;
use hanns::kernel::IndexFamily;
use hanns::storage::{
    CallbackArtifactReader, IndexArtifactReader, IndexManifest, LoadMode, ManifestFeatures,
    SectionDescriptor,
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

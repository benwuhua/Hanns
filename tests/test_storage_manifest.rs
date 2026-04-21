use hanns::kernel::IndexFamily;
use hanns::storage::{IndexManifest, LoadMode, SectionDescriptor};

#[test]
fn manifest_roundtrips_as_json() {
    let manifest = IndexManifest {
        version: 1,
        family: IndexFamily::Hnsw,
        variant: "hnsw".to_string(),
        dim: 128,
        metric: "l2".to_string(),
        count: 10,
        supported_load_modes: vec![LoadMode::OwnedMemory],
        sections: vec![SectionDescriptor {
            name: "ids".to_string(),
            len: 80,
            checksum: None,
        }],
    };

    let json = serde_json::to_string(&manifest).unwrap();
    let loaded: IndexManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.family, IndexFamily::Hnsw);
    assert_eq!(loaded.sections[0].name, "ids");
    assert!(loaded.supports_load_mode(LoadMode::OwnedMemory));
}

#[test]
fn manifest_defaults_supported_load_modes_for_legacy_json() {
    let json = r#"{
        "version":1,
        "family":"ivf",
        "variant":"legacy",
        "dim":4,
        "metric":"l2",
        "count":0,
        "sections":[]
    }"#;

    let loaded: IndexManifest = serde_json::from_str(json).unwrap();
    assert_eq!(loaded.supported_load_modes, vec![LoadMode::OwnedMemory]);
}

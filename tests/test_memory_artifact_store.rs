use hanns::kernel::IndexFamily;
use hanns::storage::{
    IndexArtifactReader, IndexArtifactWriter, IndexManifest, LoadMode, MemoryArtifactStore,
};

fn manifest() -> IndexManifest {
    IndexManifest {
        version: 1,
        family: IndexFamily::Flat,
        variant: "flat".to_string(),
        dim: 4,
        metric: "l2".to_string(),
        count: 1,
        supported_load_modes: vec![LoadMode::OwnedMemory],
        sections: Vec::new(),
    }
}

#[test]
fn memory_artifact_store_reads_written_sections() {
    let mut store = MemoryArtifactStore::default();
    store.write_section("ids", &[1, 2, 3, 4]).unwrap();
    store.finish_manifest(&manifest()).unwrap();

    assert_eq!(store.section_len("ids").unwrap(), 4);
    assert_eq!(&*store.read_section("ids").unwrap(), &[1, 2, 3, 4]);
    assert_eq!(store.read_range("ids", 1, 2).unwrap(), vec![2, 3]);
    assert_eq!(store.manifest().unwrap().family, IndexFamily::Flat);
}

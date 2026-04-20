use hanns::api::SearchRequest;
use hanns::kernel::{AnnRuntime, IndexFamily};
use hanns::storage::{
    AnnSnapshot, AnnSnapshotLoader, IndexArtifactReader, IndexArtifactWriter, IndexManifest,
    LoadMode, MemoryArtifactStore, SectionDescriptor,
};

struct DummySnapshot;

impl AnnSnapshot for DummySnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> hanns::api::Result<()> {
        writer.write_section("payload", &[1, 2, 3, 4])?;
        writer.finish_manifest(&IndexManifest {
            version: 1,
            family: IndexFamily::Flat,
            variant: "dummy".to_string(),
            dim: 4,
            metric: "l2".to_string(),
            count: 1,
            sections: vec![SectionDescriptor {
                name: "payload".to_string(),
                len: 4,
                checksum: None,
            }],
        })
    }
}

struct DummyLoader;

impl AnnSnapshotLoader for DummyLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> hanns::api::Result<Box<dyn AnnRuntime>> {
        assert_eq!(mode, LoadMode::OwnedMemory);
        assert_eq!(reader.manifest()?.variant, "dummy");
        assert_eq!(&*reader.read_section("payload")?, &[1, 2, 3, 4]);
        Ok(Box::new(DummyRuntime))
    }
}

struct DummyRuntime;

impl AnnRuntime for DummyRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Flat
    }

    fn dim(&self) -> usize {
        4
    }

    fn len(&self) -> usize {
        1
    }

    fn search_into(
        &self,
        _query: &[f32],
        _req: &SearchRequest,
        _ids: &mut [i64],
        _dists: &mut [f32],
    ) -> hanns::api::Result<usize> {
        Ok(0)
    }
}

#[test]
fn ann_snapshot_writes_to_artifact_store() {
    let snapshot: Box<dyn AnnSnapshot> = Box::new(DummySnapshot);
    let mut store = MemoryArtifactStore::default();

    snapshot
        .write_snapshot(&mut store)
        .expect("snapshot should write");

    assert_eq!(
        store.manifest().expect("manifest should exist").variant,
        "dummy"
    );
    assert_eq!(
        &*store.read_section("payload").expect("payload should exist"),
        &[1, 2, 3, 4]
    );
}

#[test]
fn ann_snapshot_loader_opens_runtime_from_artifact_store() {
    let snapshot = DummySnapshot;
    let mut store = MemoryArtifactStore::default();
    snapshot
        .write_snapshot(&mut store)
        .expect("snapshot should write");

    let loader: Box<dyn AnnSnapshotLoader> = Box::new(DummyLoader);
    let runtime = loader
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_eq!(runtime.family(), IndexFamily::Flat);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 1);
}

#[test]
fn load_mode_planned_variants_are_public() {
    let modes = [
        LoadMode::OwnedMemory,
        LoadMode::Mmap,
        LoadMode::PageCache,
        LoadMode::Lazy,
    ];

    assert_eq!(modes.len(), 4);
    assert!(modes.contains(&LoadMode::OwnedMemory));
}

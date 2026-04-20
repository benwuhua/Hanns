use hanns::kernel::IndexFamily;
use hanns::storage::{
    FileArtifactStore, IndexArtifactReader, IndexArtifactWriter, IndexManifest, SectionDescriptor,
};
use tempfile::tempdir;

fn manifest() -> IndexManifest {
    IndexManifest {
        version: 1,
        family: IndexFamily::Flat,
        variant: "flat".to_string(),
        dim: 4,
        metric: "l2".to_string(),
        count: 2,
        sections: vec![
            SectionDescriptor {
                name: "vectors".to_string(),
                len: 8,
                checksum: None,
            },
            SectionDescriptor {
                name: "ids".to_string(),
                len: 4,
                checksum: None,
            },
        ],
    }
}

#[test]
fn file_artifact_store_persists_manifest_and_sections() {
    let dir = tempdir().expect("tempdir should build");
    let mut writer = FileArtifactStore::new(dir.path()).expect("store should open");

    writer
        .write_section("vectors", &[10, 20, 30, 40, 50, 60, 70, 80])
        .expect("vectors section should write");
    writer
        .write_section("ids", &[1, 0, 2, 0])
        .expect("ids section should write");
    writer
        .finish_manifest(&manifest())
        .expect("manifest should write");

    let reader = FileArtifactStore::new(dir.path()).expect("store should reopen");

    assert_eq!(
        reader.manifest().expect("manifest should load"),
        &manifest()
    );
    assert_eq!(
        &*reader.read_section("vectors").expect("vectors should read"),
        &[10, 20, 30, 40, 50, 60, 70, 80]
    );
    assert_eq!(
        &*reader.read_section("ids").expect("ids should read"),
        &[1, 0, 2, 0]
    );
}

#[test]
fn file_artifact_store_reads_requested_range() {
    let dir = tempdir().expect("tempdir should build");
    let mut writer = FileArtifactStore::new(dir.path()).expect("store should open");

    writer
        .write_section("vectors", &[10, 20, 30, 40, 50, 60, 70, 80])
        .expect("vectors section should write");
    writer
        .finish_manifest(&manifest())
        .expect("manifest should write");

    let reader = FileArtifactStore::new(dir.path()).expect("store should reopen");

    assert_eq!(
        reader
            .read_range("vectors", 2, 4)
            .expect("range should read"),
        vec![30, 40, 50, 60]
    );
}

#[test]
fn file_artifact_store_errors_for_missing_section() {
    let dir = tempdir().expect("tempdir should build");
    let mut writer = FileArtifactStore::new(dir.path()).expect("store should open");
    writer
        .finish_manifest(&manifest())
        .expect("manifest should write");

    let reader = FileArtifactStore::new(dir.path()).expect("store should reopen");
    let error = reader
        .read_section("missing")
        .expect_err("missing section should error");

    assert!(error.to_string().contains("missing section"));
}

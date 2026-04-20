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

fn manifest_with_section_name(name: &str) -> IndexManifest {
    IndexManifest {
        version: 1,
        family: IndexFamily::Flat,
        variant: "flat".to_string(),
        dim: 4,
        metric: "l2".to_string(),
        count: 1,
        sections: vec![SectionDescriptor {
            name: name.to_string(),
            len: 4,
            checksum: None,
        }],
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

#[test]
fn file_artifact_store_rejects_invalid_manifest_section_names() {
    let invalid_names = ["nested/section", "nested\\section", "section..name"];

    for name in invalid_names {
        let dir = tempdir().expect("tempdir should build");
        let mut writer = FileArtifactStore::new(dir.path()).expect("store should open");
        let error = writer
            .finish_manifest(&manifest_with_section_name(name))
            .expect_err("invalid manifest section name should error");

        assert!(
            error.to_string().contains("invalid section name"),
            "unexpected error for {name}: {error}"
        );
    }
}

#[test]
fn file_artifact_store_reopened_artifacts_are_read_only_for_writes() {
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

    let mut reopened = FileArtifactStore::new(dir.path()).expect("store should reopen");
    let write_error = reopened
        .write_section("vectors", &[99, 99, 99])
        .expect_err("existing artifact should reject section writes");
    assert!(
        write_error.to_string().contains("artifact already exists"),
        "unexpected write error: {write_error}"
    );

    let manifest_error = reopened
        .finish_manifest(&manifest())
        .expect_err("existing artifact should reject manifest writes");
    assert!(
        manifest_error
            .to_string()
            .contains("artifact already exists"),
        "unexpected manifest error: {manifest_error}"
    );

    let reader = FileArtifactStore::new(dir.path()).expect("store should reopen");
    assert_eq!(
        &*reader.read_section("vectors").expect("vectors should read"),
        &[10, 20, 30, 40, 50, 60, 70, 80]
    );
}

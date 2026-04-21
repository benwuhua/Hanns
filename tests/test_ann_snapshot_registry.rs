use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{
    default_ann_snapshot_registry, AisaqConfig, DiskAnnPcaUsqConfig, DiskAnnPcaUsqIndex,
    DiskAnnPcaUsqSectionedSnapshot, DiskAnnSqConfig, DiskAnnSqIndex, DiskAnnSqSectionedSnapshot,
    HnswIndex, HnswSectionedSnapshot, IvfFlatIndex, IvfFlatSectionedSnapshot,
};
use hanns::kernel::IndexFamily;
use hanns::storage::{
    AnnSnapshot, IndexArtifactWriter, IndexManifest, LoadMode, MemoryArtifactStore,
};

fn build_ivf_flat() -> IvfFlatIndex {
    let config = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: DataType::Float,
        params: IndexParams::ivf(2, 2),
    };
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        0.1, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, //
        1.1, 1.0, 1.0, 1.0,
    ];
    let ids = vec![10, 11, 12, 13];

    let mut index = IvfFlatIndex::new(&config).expect("ivf-flat index should build");
    index.train(&vectors).expect("ivf-flat should train");
    index
        .add(&vectors, Some(&ids))
        .expect("ivf-flat should add");
    index
}

fn build_hnsw() -> HnswIndex {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: DataType::Float,
        params: IndexParams::hnsw(8, 32, 1.0),
    };
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        0.1, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, //
        1.1, 1.0, 1.0, 1.0,
    ];
    let ids = vec![20, 21, 22, 23];

    let mut index = HnswIndex::new(&config).expect("hnsw index should build");
    index.train(&vectors).expect("hnsw should train");
    index
        .add(&vectors, Some(&ids))
        .expect("hnsw vectors should add");
    index
}

fn build_diskann_sq() -> (DiskAnnSqIndex, Vec<f32>) {
    let dim = 4;
    let n = 32;
    let config = DiskAnnSqConfig {
        base: AisaqConfig {
            max_degree: 8,
            search_list_size: 32,
            beamwidth: 4,
            num_entry_points: 2,
            num_refine_passes: 1,
            ..Default::default()
        },
        use_pca: false,
        pca_dim: 4,
        rerank_k: 16,
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            vectors.push(((i * 13 + j * 7) % 31) as f32 / 31.0);
        }
    }
    let mut index = DiskAnnSqIndex::new(dim, MetricType::L2, config).expect("diskann sq");
    index.build(&vectors, n, dim).expect("diskann sq build");
    (index, vectors)
}

fn build_diskann_pca_usq() -> (DiskAnnPcaUsqIndex, Vec<f32>) {
    let dim = 8;
    let n = 32;
    let config = DiskAnnPcaUsqConfig {
        base: AisaqConfig {
            max_degree: 8,
            search_list_size: 32,
            beamwidth: 4,
            num_entry_points: 2,
            num_refine_passes: 1,
            ..Default::default()
        },
        pca_dim: 4,
        bits_per_dim: 4,
        rotation_seed: 7,
        rerank_k: 16,
    };
    let mut vectors = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            vectors.push(((i * 17 + j * 11) % 37) as f32 / 37.0);
        }
    }
    let mut index = DiskAnnPcaUsqIndex::new(dim, MetricType::L2, config).expect("diskann pca usq");
    index
        .build(&vectors, n, dim)
        .expect("diskann pca usq build");
    (index, vectors)
}

#[test]
fn default_registry_loads_ivf_flat_runtime_by_manifest_variant() {
    let index = build_ivf_flat();
    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let registry = default_ann_snapshot_registry().expect("registry");
    assert!(registry
        .registered_variants()
        .contains(&"ivf_flat_sections_v1"));

    let runtime = registry
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");
    assert_eq!(runtime.family(), IndexFamily::Ivf);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 4);

    let mut ids = vec![-1; 2];
    let mut dists = vec![f32::MAX; 2];
    let count = runtime
        .search_into(
            &[0.05, 0.0, 0.0, 0.0],
            &SearchRequest {
                top_k: 2,
                nprobe: 2,
                ..Default::default()
            },
            &mut ids,
            &mut dists,
        )
        .expect("runtime search");

    assert_eq!(count, 2);
    assert_eq!(ids[0], 10);
}

#[test]
fn default_registry_rejects_non_owned_load_modes_until_supported() {
    let index = build_ivf_flat();
    let snapshot = IvfFlatSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");
    let registry = default_ann_snapshot_registry().expect("registry");

    for mode in [LoadMode::Mmap, LoadMode::PageCache, LoadMode::Lazy] {
        let error = match registry.load_snapshot(&store, mode) {
            Ok(_) => panic!("mode {mode:?} should be rejected until implemented"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("LoadMode::OwnedMemory"),
            "unexpected error for mode {mode:?}: {error}"
        );
    }
}

#[test]
fn default_registry_loads_hnsw_sectioned_runtime_by_manifest_variant() {
    let index = build_hnsw();
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let registry = default_ann_snapshot_registry().expect("registry");
    let runtime = registry
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_eq!(runtime.family(), IndexFamily::Hnsw);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 4);
}

#[test]
fn default_registry_loads_diskann_sq_runtime_by_manifest_variant() {
    let (index, vectors) = build_diskann_sq();
    let snapshot = DiskAnnSqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let registry = default_ann_snapshot_registry().expect("registry");
    let runtime = registry
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_eq!(runtime.family(), IndexFamily::DiskAnn);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 32);

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
fn default_registry_loads_diskann_pca_usq_runtime_by_manifest_variant() {
    let (index, vectors) = build_diskann_pca_usq();
    let snapshot = DiskAnnPcaUsqSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let registry = default_ann_snapshot_registry().expect("registry");
    let runtime = registry
        .load_snapshot(&store, LoadMode::OwnedMemory)
        .expect("runtime should load");

    assert_eq!(runtime.family(), IndexFamily::DiskAnn);
    assert_eq!(runtime.dim(), 8);
    assert_eq!(runtime.len(), 32);

    let mut ids = vec![-1; 4];
    let mut dists = vec![f32::MAX; 4];
    let count = runtime
        .search_into(
            &vectors[0..8],
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
fn registry_rejects_unregistered_variant() {
    let mut store = MemoryArtifactStore::default();
    store.write_section("payload", &[1]).expect("section write");
    store
        .finish_manifest(&IndexManifest {
            version: 1,
            family: IndexFamily::Ivf,
            variant: "unknown_sections_v1".to_string(),
            dim: 4,
            metric: "l2".to_string(),
            count: 0,
            sections: vec![],
        })
        .expect("manifest write");

    let registry = default_ann_snapshot_registry().expect("registry");
    let error = match registry.load_snapshot(&store, LoadMode::OwnedMemory) {
        Ok(_) => panic!("unknown variant should fail"),
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("no snapshot loader registered"),
        "unexpected error: {error}"
    );
}

#[test]
fn registry_rejects_duplicate_variant_registration() {
    let mut registry = hanns::storage::AnnSnapshotRegistry::new();
    registry
        .register_loader("dummy", Box::new(hanns::faiss::IvfFlatSnapshotLoader))
        .expect("first registration");
    let error = registry
        .register_loader("dummy", Box::new(hanns::faiss::IvfFlatSnapshotLoader))
        .expect_err("duplicate registration should fail");

    assert!(
        error.to_string().contains("already registered"),
        "unexpected error: {error}"
    );
}

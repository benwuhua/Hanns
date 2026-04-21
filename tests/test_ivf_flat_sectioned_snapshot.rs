use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, KnowhereError, MetricType};
use hanns::faiss::IvfFlatIndex;

fn build_small_ivf_flat() -> (IvfFlatIndex, usize) {
    let dim = 4;
    let nlist = 3;
    let cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim,
        data_type: DataType::Float,
        params: IndexParams::ivf(nlist, 2),
    };
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, //
        0.1, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, //
        1.1, 1.0, 1.0, 1.0, //
        2.0, 2.0, 2.0, 2.0, //
        2.1, 2.0, 2.0, 2.0,
    ];
    let ids = vec![10, 11, 12, 13, 14, 15];

    let mut index = IvfFlatIndex::new(&cfg).expect("ivf-flat index should build");
    index.train(&vectors).expect("ivf-flat index should train");
    index
        .add(&vectors, Some(&ids))
        .expect("ivf-flat vectors should add");
    (index, nlist)
}

#[test]
fn ivf_flat_rejects_untrained_sectioned_export() {
    let cfg = IndexConfig {
        index_type: IndexType::IvfFlat,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: DataType::Float,
        params: IndexParams::ivf(3, 2),
    };
    let index = IvfFlatIndex::new(&cfg).expect("ivf-flat index should build");

    let error = index
        .export_sectioned_snapshot()
        .expect_err("untrained IVF-Flat sectioned export should fail");

    assert!(
        matches!(
            error,
            KnowhereError::InvalidArg(_) | KnowhereError::Codec(_)
        ),
        "expected clear validation error, got {error:?}"
    );
    assert!(
        error.to_string().contains("trained"),
        "error should mention trained state, got {error}"
    );
}

#[test]
fn ivf_flat_exports_sectioned_snapshot_shape() {
    let (index, nlist) = build_small_ivf_flat();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 4);
    assert_eq!(export.count, 6);
    assert_eq!(export.nlist, nlist);
    assert_eq!(export.centroids.len(), nlist * export.dim);
    assert_eq!(export.list_offsets.len(), nlist);
    assert_eq!(export.list_sizes.len(), nlist);

    let list_total = export.list_sizes.iter().sum::<u64>();
    assert_eq!(list_total, export.list_ids.len() as u64);
    assert_eq!(
        export.list_vectors.len(),
        export.list_ids.len() * export.dim
    );
    assert_eq!(export.ids.len(), export.count);
    assert_eq!(export.vectors.len(), export.count * export.dim);
}

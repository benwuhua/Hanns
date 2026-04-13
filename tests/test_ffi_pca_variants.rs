use hanns::ffi::*;

fn make_data(n: usize, dim: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        for j in 0..dim {
            data[i * dim + j] = ((i * 17 + j * 13) % 101) as f32 / 100.0;
        }
    }
    data
}

unsafe fn result_ids(result: *mut CSearchResult) -> Vec<i64> {
    let r = &*result;
    std::slice::from_raw_parts(r.ids, r.num_results).to_vec()
}

#[test]
fn test_ffi_hnsw_pca_sq_train_add_search_smoke() {
    let dim = 16usize;
    let n = 32usize;
    let data = make_data(n, dim);
    let ids: Vec<i64> = (100..100 + n as i64).collect();

    let index = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::HnswPcaSq,
        metric_type: CMetricType::L2,
        dim,
        pca_dim: 8,
        ef_construction: 64,
        ef_search: 32,
        ..Default::default()
    });
    assert!(!index.is_null());

    assert_eq!(
        knowhere_train_index(index, data.as_ptr(), n, dim),
        CError::Success as i32
    );
    assert_eq!(
        knowhere_add_index(index, data.as_ptr(), ids.as_ptr(), n, dim),
        CError::Success as i32
    );

    let result = knowhere_search(index, data.as_ptr(), 1, 5, dim);
    assert!(!result.is_null());
    let got = unsafe { result_ids(result) };
    assert_eq!(got.len(), 5);
    assert!(
        got.contains(&100),
        "self id missing from HNSW-PCA-SQ results: {got:?}"
    );

    knowhere_free_result(result);
    knowhere_free_index(index);
}

#[test]
fn test_ffi_hnsw_pca_usq_train_add_search_smoke() {
    let dim = 16usize;
    let n = 32usize;
    let data = make_data(n, dim);
    let ids: Vec<i64> = (200..200 + n as i64).collect();

    let index = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::HnswPcaUsq,
        metric_type: CMetricType::L2,
        dim,
        pca_dim: 8,
        ef_construction: 64,
        ef_search: 32,
        ..Default::default()
    });
    assert!(!index.is_null());

    assert_eq!(
        knowhere_train_index(index, data.as_ptr(), n, dim),
        CError::Success as i32
    );
    assert_eq!(
        knowhere_add_index(index, data.as_ptr(), ids.as_ptr(), n, dim),
        CError::Success as i32
    );

    let result = knowhere_search(index, data.as_ptr(), 1, 5, dim);
    assert!(!result.is_null());
    let got = unsafe { result_ids(result) };
    assert_eq!(got.len(), 5);
    assert!(
        got.contains(&200),
        "self id missing from HNSW-PCA-USQ results: {got:?}"
    );

    knowhere_free_result(result);
    knowhere_free_index(index);
}

#[test]
fn test_ffi_diskann_pca_usq_build_search_smoke() {
    let dim = 16usize;
    let n = 32usize;
    let data = make_data(n, dim);

    let index = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::DiskAnnPcaUsq,
        metric_type: CMetricType::L2,
        dim,
        pca_dim: 8,
        ef_construction: 48,
        ef_search: 32,
        beamwidth: 8,
        ..Default::default()
    });
    assert!(!index.is_null());

    // DiskANN-PCA-USQ does training during build/add; train is a no-op success gate.
    assert_eq!(
        knowhere_train_index(index, data.as_ptr(), n, dim),
        CError::Success as i32
    );
    assert_eq!(
        knowhere_add_index(index, data.as_ptr(), std::ptr::null(), n, dim),
        CError::Success as i32
    );

    let result = knowhere_search(index, data.as_ptr(), 1, 5, dim);
    assert!(!result.is_null());
    let got = unsafe { result_ids(result) };
    assert_eq!(got.len(), 5);
    assert!(
        got.contains(&0),
        "self id missing from DiskANN-PCA-USQ results: {got:?}"
    );

    knowhere_free_result(result);
    knowhere_free_index(index);
}

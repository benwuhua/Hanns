//! FFI ABI metadata contract test.
//!
//! Extracted from `src/ffi.rs` to avoid recompiling 4600+ lines of production
//! code when only test assertions change. Run standalone:
//!
//!     cargo test --test test_ffi_metadata_contract

use knowhere_rs::ffi::*;

#[test]
fn test_ffi_abi_metadata_contract() {
    let flat = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::Flat,
        metric_type: CMetricType::L2,
        dim: 16,
        ..Default::default()
    });
    assert!(!flat.is_null());

    assert_eq!(knowhere_is_additional_scalar_supported(flat, false), 0);
    assert_eq!(knowhere_is_additional_scalar_supported(flat, true), 0);

    let flat_meta_ptr = knowhere_get_index_meta(flat);
    assert!(!flat_meta_ptr.is_null());

    let flat_meta_str = unsafe { std::ffi::CStr::from_ptr(flat_meta_ptr) }
        .to_str()
        .unwrap();
    let flat_meta_json: serde_json::Value = serde_json::from_str(flat_meta_str).unwrap();
    assert_eq!(flat_meta_json["index_type"], "Flat");
    assert_eq!(flat_meta_json["dim"], 16);
    assert_eq!(flat_meta_json["count"], 0);
    assert_eq!(flat_meta_json["is_trained"], false);
    assert_eq!(flat_meta_json["has_raw_data"], true);
    assert_eq!(flat_meta_json["additional_scalar_supported"], false);
    assert_eq!(
        flat_meta_json["additional_scalar"]["support_mode"],
        "unsupported"
    );
    assert_eq!(flat_meta_json["additional_scalar"]["mv_only_query"], true);
    assert_eq!(
        flat_meta_json["additional_scalar"]["unsupported_reason"],
        "additional-scalar filtering is unsupported for this index type in the current Rust FFI"
    );
    assert_eq!(
        flat_meta_json["capabilities"]["get_vector_by_ids"],
        "supported"
    );
    assert_eq!(
        flat_meta_json["capabilities"]["ann_iterator"],
        "unsupported"
    );
    assert_eq!(flat_meta_json["capabilities"]["persistence"], "supported");
    assert_eq!(flat_meta_json["semantics"]["family"], "generic");
    assert_eq!(
        flat_meta_json["semantics"]["raw_data_gate"],
        "raw_vectors_retained"
    );
    assert_eq!(
        flat_meta_json["semantics"]["persistence_mode"],
        "file_save_load+memory_serialize"
    );
    assert_eq!(
        flat_meta_json["semantics"]["persistence"]["file_save_load"],
        "supported"
    );
    assert_eq!(
        flat_meta_json["semantics"]["persistence"]["memory_serialize"],
        "supported"
    );
    assert_eq!(
        flat_meta_json["semantics"]["persistence"]["deserialize_from_file"],
        "supported"
    );
    assert_eq!(
        flat_meta_json["observability"]["schema_version"],
        "runtime_observability.v1"
    );
    assert_eq!(
        flat_meta_json["observability"]["build_event"],
        "knowhere.index.build"
    );
    assert_eq!(
        flat_meta_json["observability"]["search_event"],
        "knowhere.index.search"
    );
    assert_eq!(
        flat_meta_json["observability"]["load_event"],
        "knowhere.index.load"
    );
    assert_eq!(
        flat_meta_json["trace_propagation"]["ffi_entrypoint"],
        "index_meta.trace_context_json"
    );
    assert_eq!(
        flat_meta_json["trace_propagation"]["gate_runner_entrypoint"],
        "OPENCLAW_TRACE_CONTEXT_JSON"
    );
    assert_eq!(
        flat_meta_json["trace_propagation"]["context_encoding"],
        "w3c-traceparent-json"
    );
    assert_eq!(
        flat_meta_json["resource_contract"]["schema_version"],
        "resource_contract.v1"
    );
    assert_eq!(
        flat_meta_json["resource_contract"]["memory_bytes"],
        "estimated_runtime_memory_bytes"
    );
    assert_eq!(
        flat_meta_json["resource_contract"]["disk_bytes"],
        "estimated_file_bytes"
    );
    assert_eq!(flat_meta_json["resource_contract"]["mmap_supported"], true);
    assert_eq!(
        flat_meta_json["resource_contract"]["unsupported_reason"],
        ""
    );

    knowhere_free_cstring(flat_meta_ptr);
    knowhere_free_index(flat);

    let hnsw = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::Hnsw,
        metric_type: CMetricType::L2,
        dim: 16,
        ..Default::default()
    });
    assert!(!hnsw.is_null());
    assert_eq!(knowhere_is_additional_scalar_supported(hnsw, false), 0);
    assert_eq!(knowhere_is_additional_scalar_supported(hnsw, true), 0);
    let hnsw_meta_ptr = knowhere_get_index_meta(hnsw);
    assert!(!hnsw_meta_ptr.is_null());
    let hnsw_meta_str = unsafe { std::ffi::CStr::from_ptr(hnsw_meta_ptr) }
        .to_str()
        .unwrap();
    let hnsw_meta_json: serde_json::Value = serde_json::from_str(hnsw_meta_str).unwrap();
    assert_eq!(hnsw_meta_json["index_type"], "HNSW");
    assert_eq!(hnsw_meta_json["semantics"]["family"], "hnsw");
    assert_eq!(
        hnsw_meta_json["semantics"]["raw_data_gate"],
        "raw_vectors_retained"
    );
    assert_eq!(
        hnsw_meta_json["semantics"]["persistence_mode"],
        "file_save_load+memory_serialize"
    );
    assert_eq!(
        hnsw_meta_json["semantics"]["persistence"]["file_save_load"],
        "supported"
    );
    assert_eq!(
        hnsw_meta_json["semantics"]["persistence"]["memory_serialize"],
        "supported"
    );
    assert_eq!(
        hnsw_meta_json["semantics"]["persistence"]["deserialize_from_file"],
        "supported"
    );
    assert_eq!(hnsw_meta_json["capabilities"]["ann_iterator"], "supported");
    assert_eq!(hnsw_meta_json["capabilities"]["persistence"], "supported");
    assert_eq!(
        hnsw_meta_json["capabilities"]["get_vector_by_ids"],
        "supported"
    );
    assert_eq!(
        hnsw_meta_json["additional_scalar"]["unsupported_reason"],
        "HNSW does not expose additional-scalar filtering through the current Rust FFI"
    );
    assert_eq!(hnsw_meta_json["resource_contract"]["mmap_supported"], true);
    assert_eq!(
        hnsw_meta_json["resource_contract"]["disk_bytes"],
        "estimated_file_bytes"
    );
    knowhere_free_cstring(hnsw_meta_ptr);
    knowhere_free_index(hnsw);

    let ivf = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::IvfSq8,
        metric_type: CMetricType::L2,
        dim: 16,
        ..Default::default()
    });
    assert!(!ivf.is_null());
    assert_eq!(knowhere_is_additional_scalar_supported(ivf, true), 0);
    let ivf_meta_ptr = knowhere_get_index_meta(ivf);
    assert!(!ivf_meta_ptr.is_null());
    let ivf_meta_str = unsafe { std::ffi::CStr::from_ptr(ivf_meta_ptr) }
        .to_str()
        .unwrap();
    let ivf_meta_json: serde_json::Value = serde_json::from_str(ivf_meta_str).unwrap();
    assert_eq!(ivf_meta_json["index_type"], "IVF_SQ8");
    assert_eq!(ivf_meta_json["semantics"]["family"], "ivf");
    assert_eq!(
        ivf_meta_json["semantics"]["raw_data_gate"],
        "quantized_or_codebook_only"
    );
    assert_eq!(
        ivf_meta_json["semantics"]["persistence_mode"],
        "file_save_load+memory_serialize"
    );
    assert_eq!(
        ivf_meta_json["semantics"]["persistence"]["file_save_load"],
        "supported"
    );
    assert_eq!(
        ivf_meta_json["semantics"]["persistence"]["memory_serialize"],
        "supported"
    );
    assert_eq!(
        ivf_meta_json["semantics"]["persistence"]["deserialize_from_file"],
        "supported"
    );
    assert_eq!(
        ivf_meta_json["capabilities"]["get_vector_by_ids"],
        "unsupported"
    );
    assert_eq!(
        ivf_meta_json["additional_scalar"]["unsupported_reason"],
        "IVF variants do not expose additional-scalar filtering through the current Rust FFI"
    );
    assert_eq!(
        ivf_meta_json["resource_contract"]["memory_bytes"],
        "estimated_runtime_memory_bytes_or_codebook_only"
    );
    assert_eq!(
        ivf_meta_json["resource_contract"]["disk_bytes"],
        "estimated_file_bytes"
    );
    assert_eq!(ivf_meta_json["resource_contract"]["mmap_supported"], true);
    knowhere_free_cstring(ivf_meta_ptr);
    knowhere_free_index(ivf);

    let ivfpq = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::IvfPq,
        metric_type: CMetricType::L2,
        dim: 16,
        num_clusters: 8,
        nprobe: 2,
        ..Default::default()
    });
    assert!(!ivfpq.is_null());
    assert_eq!(knowhere_is_additional_scalar_supported(ivfpq, true), 0);
    let ivfpq_meta_ptr = knowhere_get_index_meta(ivfpq);
    assert!(!ivfpq_meta_ptr.is_null());
    let ivfpq_meta_str = unsafe { std::ffi::CStr::from_ptr(ivfpq_meta_ptr) }
        .to_str()
        .unwrap();
    let ivfpq_meta_json: serde_json::Value = serde_json::from_str(ivfpq_meta_str).unwrap();
    assert_eq!(ivfpq_meta_json["index_type"], "IVF_PQ");
    assert_eq!(ivfpq_meta_json["semantics"]["family"], "ivf");
    assert_eq!(
        ivfpq_meta_json["semantics"]["raw_data_gate"],
        "quantized_or_codebook_only"
    );
    assert_eq!(
        ivfpq_meta_json["semantics"]["persistence_mode"],
        "file_save_load+memory_serialize"
    );
    assert_eq!(
        ivfpq_meta_json["semantics"]["persistence"]["file_save_load"],
        "supported"
    );
    assert_eq!(
        ivfpq_meta_json["semantics"]["persistence"]["memory_serialize"],
        "supported"
    );
    assert_eq!(
        ivfpq_meta_json["semantics"]["persistence"]["deserialize_from_file"],
        "supported"
    );
    assert_eq!(
        ivfpq_meta_json["capabilities"]["get_vector_by_ids"],
        "unsupported"
    );
    assert_eq!(
        ivfpq_meta_json["capabilities"]["ann_iterator"],
        "unsupported"
    );
    assert_eq!(ivfpq_meta_json["capabilities"]["persistence"], "supported");
    assert_eq!(
        ivfpq_meta_json["additional_scalar"]["unsupported_reason"],
        "IVF variants do not expose additional-scalar filtering through the current Rust FFI"
    );
    assert_eq!(ivfpq_meta_json["resource_contract"]["mmap_supported"], true);
    knowhere_free_cstring(ivfpq_meta_ptr);
    knowhere_free_index(ivfpq);

    #[cfg(feature = "scann")]
    {
        let scann = knowhere_create_index(CIndexConfig {
            index_type: CIndexType::Scann,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        });
        assert!(!scann.is_null());
        assert_eq!(knowhere_is_additional_scalar_supported(scann, true), 0);
        let scann_meta_ptr = knowhere_get_index_meta(scann);
        assert!(!scann_meta_ptr.is_null());
        let scann_meta_str = unsafe { std::ffi::CStr::from_ptr(scann_meta_ptr) }
            .to_str()
            .unwrap();
        let scann_meta_json: serde_json::Value = serde_json::from_str(scann_meta_str).unwrap();
        assert_eq!(scann_meta_json["index_type"], "ScaNN");
        assert_eq!(scann_meta_json["semantics"]["family"], "scann");
        assert_eq!(
            scann_meta_json["semantics"]["persistence_mode"],
            "file_save_load"
        );
        assert_eq!(
            scann_meta_json["semantics"]["persistence"]["file_save_load"],
            "supported"
        );
        assert_eq!(
            scann_meta_json["semantics"]["persistence"]["memory_serialize"],
            "unsupported"
        );
        assert_eq!(
            scann_meta_json["semantics"]["persistence"]["deserialize_from_file"],
            "supported"
        );
        assert_eq!(scann_meta_json["capabilities"]["ann_iterator"], "supported");
        assert_eq!(scann_meta_json["capabilities"]["persistence"], "supported");
        assert_eq!(
            scann_meta_json["additional_scalar"]["unsupported_reason"],
            "ScaNN does not expose additional-scalar filtering through the current Rust FFI"
        );
        assert_eq!(
            scann_meta_json["trace_propagation"]["propagation_mode"],
            "optional_passthrough"
        );
        assert_eq!(scann_meta_json["resource_contract"]["mmap_supported"], true);
        knowhere_free_cstring(scann_meta_ptr);
        knowhere_free_index(scann);
    }

    let sparse = knowhere_create_index(CIndexConfig {
        index_type: CIndexType::SparseWand,
        metric_type: CMetricType::Ip,
        dim: 16,
        data_type: 104,
        ..Default::default()
    });
    assert!(!sparse.is_null());

    assert_eq!(knowhere_is_additional_scalar_supported(sparse, false), 0);
    assert_eq!(knowhere_is_additional_scalar_supported(sparse, true), 1);

    let sparse_meta_ptr = knowhere_get_index_meta(sparse);
    assert!(!sparse_meta_ptr.is_null());
    let sparse_meta_str = unsafe { std::ffi::CStr::from_ptr(sparse_meta_ptr) }
        .to_str()
        .unwrap();
    let sparse_meta_json: serde_json::Value = serde_json::from_str(sparse_meta_str).unwrap();
    assert_eq!(sparse_meta_json["index_type"], "SparseWand");
    assert_eq!(sparse_meta_json["additional_scalar_supported"], true);
    assert_eq!(
        sparse_meta_json["additional_scalar"]["support_mode"],
        "partial"
    );
    assert_eq!(
        sparse_meta_json["additional_scalar"]["unsupported_reason"],
        "only sparse indexes expose MV-only additional-scalar filtering via the current Rust FFI"
    );
    assert_eq!(
        sparse_meta_json["capabilities"]["ann_iterator"],
        "supported"
    );
    assert_eq!(sparse_meta_json["capabilities"]["persistence"], "supported");
    assert_eq!(sparse_meta_json["semantics"]["family"], "sparse");
    assert_eq!(
        sparse_meta_json["semantics"]["persistence_mode"],
        "file_save_load+memory_serialize"
    );
    assert_eq!(
        sparse_meta_json["semantics"]["persistence"]["file_save_load"],
        "supported"
    );
    assert_eq!(
        sparse_meta_json["semantics"]["persistence"]["memory_serialize"],
        "supported"
    );
    assert_eq!(
        sparse_meta_json["semantics"]["persistence"]["deserialize_from_file"],
        "supported"
    );
    assert_eq!(
        sparse_meta_json["observability"]["required_fields"][0],
        "index_type"
    );
    assert_eq!(
        sparse_meta_json["trace_propagation"]["context_encoding"],
        "w3c-traceparent-json"
    );
    assert_eq!(
        sparse_meta_json["resource_contract"]["memory_bytes"],
        "estimated_runtime_memory_bytes"
    );
    assert_eq!(
        sparse_meta_json["resource_contract"]["disk_bytes"],
        "estimated_file_bytes"
    );
    assert_eq!(
        sparse_meta_json["resource_contract"]["mmap_supported"],
        true
    );

    knowhere_free_cstring(sparse_meta_ptr);
    knowhere_free_index(sparse);

    assert_eq!(
        knowhere_is_additional_scalar_supported(std::ptr::null(), true),
        0
    );
    assert!(knowhere_get_index_meta(std::ptr::null()).is_null());
}

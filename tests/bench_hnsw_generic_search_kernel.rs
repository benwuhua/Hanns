use hanns::api::IndexParams;
use hanns::bitset::BitsetView;
use hanns::faiss::HnswIndex;
use hanns::{IndexConfig, IndexType, MetricType, SearchRequest};

fn build_fixture() -> HnswIndex {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: hanns::api::DataType::Float,
        params: IndexParams {
            ef_search: Some(16),
            ..Default::default()
        },
    };

    let mut index = HnswIndex::new(&config).expect("create HNSW test fixture");
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0,
        0.0, 0.0,
    ];
    let ids = vec![0, 1, 2, 3, 4];

    index.train(&vectors).expect("train HNSW fixture");
    index
        .add(&vectors, Some(&ids))
        .expect("add HNSW fixture vectors");
    index
}

#[test]
fn generic_kernel_report_declares_shared_idx_traversal() {
    let index = build_fixture();

    let report = index.generic_search_kernel_report();

    assert_eq!(
        report.unfiltered_layer_search_core,
        "shared_idx_binary_heap"
    );
    assert_eq!(report.bitset_layer_search_core, "shared_idx_binary_heap");
    assert_eq!(report.frontier_reuse_scope, "scratch_binary_heap");
    assert_eq!(report.result_reuse_scope, "scratch_binary_heap");
    assert_eq!(report.shared_l2_distance_dispatch, "idx_ptr_kernel");
    assert_eq!(
        report.shared_bitset_distance_mode,
        "idx_ptr_batch4_when_grouped"
    );
    assert_eq!(
        report.shared_layer0_neighbor_layout,
        "flat_u32_adjacency_when_enabled"
    );
    assert_eq!(report.shared_result_threshold_mode, "scratch_cached_worst");
    assert_eq!(
        report.visited_reuse_scope,
        "visited_epoch_and_generic_heaps"
    );
}

#[test]
fn generic_kernel_report_preserves_bitset_fixture_results() {
    let index = build_fixture();
    let query = vec![0.5, 0.0, 0.0, 0.0];
    let req = SearchRequest {
        top_k: 5,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };
    let mut bitset = BitsetView::new(index.ntotal());
    bitset.set(0, true);
    bitset.set(2, true);

    let result = index
        .search_with_bitset(&query, &req, &bitset)
        .expect("bitset search should succeed on the shared-kernel fixture");

    assert_eq!(result.ids, vec![1, 3, 4]);
}

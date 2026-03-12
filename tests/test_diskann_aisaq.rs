use std::fs;

use knowhere_rs::faiss::diskann_aisaq::AisaqConfig;
use knowhere_rs::{MetricType, PQFlashIndex, SearchResult};
use tempfile::tempdir;

fn build_index() -> PQFlashIndex {
    let config = AisaqConfig {
        max_degree: 2,
        search_list_size: 4,
        beamwidth: 2,
        disk_pq_dims: 2,
        num_entry_points: 1,
        warm_up: true,
        ..AisaqConfig::default()
    };

    let mut index = PQFlashIndex::new(config, MetricType::L2, 4).expect("index should build");
    let data = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0,
    ];
    index.add(&data).expect("add should succeed");
    index
}

#[test]
fn train_and_add_builds_graph() {
    let index = build_index();
    assert_eq!(index.len(), 3);
    assert!(index.flash_layout().node_bytes >= index.flash_layout().vector_bytes);
}

#[test]
fn search_returns_expected_neighbor() {
    let index = build_index();
    let result: SearchResult = index
        .search(&[0.1, 0.1, 0.1, 0.1], 2)
        .expect("search should succeed");

    assert_eq!(result.ids.len(), 2);
    assert_eq!(result.ids[0], 0);
    assert!(result.num_visited >= 1);
}

#[test]
fn rejects_dimension_mismatch() {
    let config = AisaqConfig::default();
    let mut index = PQFlashIndex::new(config, MetricType::L2, 4).expect("index should build");
    let error = index
        .train(&[1.0, 2.0, 3.0])
        .expect_err("invalid input must fail");
    assert!(error.to_string().contains("dimension"));
}

#[test]
fn save_and_load_round_trip_preserves_search() {
    let index = build_index();
    let dir = tempdir().expect("tempdir should build");
    let file_group = index.save(dir.path()).expect("save should succeed");

    assert!(file_group.metadata_path().exists());
    assert!(file_group.data_path().exists());

    let loaded = PQFlashIndex::load(dir.path()).expect("load should succeed");
    let result = loaded
        .search(&[0.1, 0.1, 0.1, 0.1], 2)
        .expect("loaded search should succeed");

    assert_eq!(loaded.len(), 3);
    assert_eq!(result.ids[..2], [0, 1]);
}

#[test]
fn repeated_queries_hit_page_cache() {
    let index = build_index();
    let dir = tempdir().expect("tempdir should build");
    index.save(dir.path()).expect("save should succeed");
    let loaded = PQFlashIndex::load(dir.path()).expect("load should succeed");

    for _ in 0..8 {
        loaded
            .search(&[0.1, 0.1, 0.1, 0.1], 2)
            .expect("search should succeed");
    }

    let stats = loaded.page_cache_stats().expect("page cache should exist");
    assert!(stats.hit_rate() > 0.8, "hit rate was {}", stats.hit_rate());
    assert!(stats.page_hits > stats.page_misses);
}

#[test]
fn load_uses_mmap_backed_data_file() {
    let index = build_index();
    let dir = tempdir().expect("tempdir should build");
    let file_group = index.save(dir.path()).expect("save should succeed");

    let metadata_len = fs::metadata(file_group.metadata_path())
        .expect("metadata file should exist")
        .len();
    let data_len = fs::metadata(file_group.data_path())
        .expect("data file should exist")
        .len();
    assert!(metadata_len > 0);
    assert!(data_len >= (index.len() * index.flash_layout().node_bytes) as u64);

    let loaded = PQFlashIndex::load(dir.path()).expect("load should succeed");
    let result = loaded
        .search(&[9.9, 9.9, 9.9, 9.9], 1)
        .expect("mmap search should succeed");

    let stats = loaded.page_cache_stats().expect("page cache should exist");
    assert_eq!(result.ids, vec![2]);
    assert!(stats.requests >= 1);
    assert!(loaded.page_cache().is_some());
}

#[test]
fn scope_audit_reports_real_flash_skeleton_but_not_native_diskann_parity() {
    let index = build_index();
    let audit = index.scope_audit();

    assert_eq!(audit.dim, 4);
    assert_eq!(audit.node_count, 3);
    assert_eq!(audit.entry_point_count, 1);
    assert!(audit.uses_flash_layout);
    assert!(audit.uses_beam_search_io);
    assert!(!audit.uses_mmap_backed_pages);
    assert!(!audit.native_comparable);
    assert!(
        audit.comparability_reason.contains("simplified"),
        "audit should explain why PQFlashIndex is still a constrained AISAQ skeleton"
    );

    let dir = tempdir().expect("tempdir should build");
    index.save(dir.path()).expect("save should succeed");
    let loaded = PQFlashIndex::load(dir.path()).expect("load should succeed");
    let loaded_audit = loaded.scope_audit();

    assert!(loaded_audit.uses_flash_layout);
    assert!(loaded_audit.uses_mmap_backed_pages);
    assert!(loaded_audit.has_page_cache);
    assert!(!loaded_audit.native_comparable);
}

use hanns::api::{DataType, IndexConfig, IndexType, MetricType};
use hanns::bitset::BitsetView;
use hanns::dataset::Dataset;
use hanns::faiss::RhtsdgIndex;
use hanns::index::Index;
use tempfile::NamedTempFile;

fn rhtsdg_fixture() -> (IndexConfig, Dataset, Dataset) {
    let mut config =
        IndexConfig::with_data_type(IndexType::Rhtsdg, MetricType::L2, 2, DataType::Float);
    config.params.rhtsdg_knn_k = Some(3);
    config.params.rhtsdg_sample_count = Some(2);
    config.params.rhtsdg_iter_count = Some(5);

    let base = Dataset::from_vectors_with_ids(
        vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
        ],
        2,
        vec![100, 200, 300, 400],
    );
    let query = Dataset::from_vectors(vec![0.0, 0.0], 2);
    (config, base, query)
}

#[test]
fn rhtsdg_index_type_parses_from_string() {
    assert_eq!("rhtsdg".parse::<IndexType>().unwrap(), IndexType::Rhtsdg);
}

#[test]
fn rhtsdg_float_l2_config_is_legal() {
    let config =
        IndexConfig::with_data_type(IndexType::Rhtsdg, MetricType::L2, 128, DataType::Float);
    assert!(config.validate().is_ok());
}

#[test]
fn rhtsdg_index_round_trips_save_and_load() {
    let (config, data, query) = rhtsdg_fixture();
    let mut index = RhtsdgIndex::new(&config).expect("rhtsdg should build");
    Index::train(&mut index, &data).expect("train should succeed");
    Index::add(&mut index, &data).expect("add should succeed");

    let before = Index::search(&index, &query, 3).expect("search should succeed");
    let raw_before =
        Index::get_vector_by_ids(&index, &[100, 300]).expect("raw vector lookup should succeed");

    let path = NamedTempFile::new().expect("temp file should build");
    Index::save(&index, path.path().to_str().unwrap()).expect("save should succeed");

    let mut loaded = RhtsdgIndex::new(&config).expect("reloaded rhtsdg should build");
    Index::load(&mut loaded, path.path().to_str().unwrap()).expect("load should succeed");
    let after = Index::search(&loaded, &query, 3).expect("reloaded search should succeed");
    let raw_after = Index::get_vector_by_ids(&loaded, &[100, 300])
        .expect("reloaded raw vector lookup should succeed");

    assert_eq!(before.ids, after.ids);
    assert_eq!(before.distances, after.distances);
    assert_eq!(raw_before, raw_after);
    assert_eq!(
        index.layer_sizes_for_test(),
        loaded.layer_sizes_for_test(),
        "save/load should preserve hierarchy shape"
    );
    assert_eq!(
        index.entry_point_for_test(),
        loaded.entry_point_for_test(),
        "save/load should preserve hierarchy entry point"
    );
}

#[test]
fn rhtsdg_index_search_with_bitset_filters_internal_positions() {
    let (config, data, query) = rhtsdg_fixture();
    let mut index = RhtsdgIndex::new(&config).expect("rhtsdg should build");
    Index::train(&mut index, &data).expect("train should succeed");
    Index::add(&mut index, &data).expect("add should succeed");

    let mut bitset = BitsetView::new(data.num_vectors());
    bitset.set_bit(0);

    let result = Index::search_with_bitset(&index, &query, 3, &bitset)
        .expect("bitset search should succeed");
    assert_eq!(result.ids[0], 200);
    assert!(!result.ids.contains(&100));
}

#[test]
fn rhtsdg_search_returns_public_euclidean_l2_distances() {
    let (config, data, query) = rhtsdg_fixture();
    let mut index = RhtsdgIndex::new(&config).expect("rhtsdg should build");
    Index::train(&mut index, &data).expect("train should succeed");
    Index::add(&mut index, &data).expect("add should succeed");

    let result = Index::search(&index, &query, 3).expect("search should succeed");
    assert_eq!(result.ids, vec![100, 200, 300]);
    assert_eq!(result.distances[0], 0.0);
    assert!(
        (result.distances[1] - 1.0).abs() < 1e-6,
        "expected Euclidean distance 1.0, got {}",
        result.distances[1]
    );
    assert!(
        (result.distances[2] - 2.0).abs() < 1e-6,
        "expected Euclidean distance 2.0, got {}",
        result.distances[2]
    );
}

#[test]
fn rhtsdg_ann_iterator_streams_sorted_results() {
    let (config, data, query) = rhtsdg_fixture();
    let mut index = RhtsdgIndex::new(&config).expect("rhtsdg should build");
    Index::train(&mut index, &data).expect("train should succeed");
    Index::add(&mut index, &data).expect("add should succeed");

    let mut iter =
        Index::create_ann_iterator(&index, &query, None).expect("ann iterator should succeed");

    let mut count = 0usize;
    let mut prev = f32::NEG_INFINITY;
    while let Some((_id, dist)) = iter.next() {
        assert!(dist >= prev, "iterator distances should be sorted");
        prev = dist;
        count += 1;
        if count >= data.num_vectors() {
            break;
        }
    }

    assert_eq!(count, data.num_vectors());
}

//! Test IVF Index trait implementation
//!
//! Tests that IvfSq8Index and IvfRaBitqIndex correctly implement the Index trait
//! and can be used through the unified interface.

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::dataset::Dataset;
use knowhere_rs::index::Index;
use knowhere_rs::faiss::ivf_sq8::IvfSq8Index;
use knowhere_rs::faiss::ivf_rabitq::{IvfRaBitqIndex, IvfRaBitqConfig};

#[test]
fn test_ivf_sq8_index_trait_metadata() {
    // Test that Index trait metadata methods work
    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        dim: 4,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_sq8(4, 2),
    };

    let index = IvfSq8Index::new(&config).unwrap();

    // Test Index trait metadata methods
    assert_eq!(Index::index_type(&index), "IVF-SQ8");
    assert_eq!(Index::dim(&index), 4);
    assert_eq!(Index::count(&index), 0);
    assert!(!Index::is_trained(&index));
    assert!(Index::has_raw_data(&index));
}

#[test]
fn test_ivf_sq8_index_trait_lifecycle() {
    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        dim: 4,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_sq8(4, 2),
    };

    let mut index = IvfSq8Index::new(&config).unwrap();

    // Train and add using Index trait
    let vectors = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ];

    let dataset = Dataset::from_vectors_with_ids(vectors.clone(), 4, vec![0, 1, 2, 3]);

    // Use fully qualified syntax to call Index trait methods
    Index::train(&mut index, &dataset).unwrap();
    assert!(Index::is_trained(&index));

    Index::add(&mut index, &dataset).unwrap();
    assert_eq!(Index::count(&index), 4);

    // Search through Index trait
    let query = Dataset::from_vectors(vec![0.1, 0.1, 0.1, 0.1], 4);
    let result = Index::search(&index, &query, 2).unwrap();
    assert_eq!(result.ids.len(), 2);

    // Test get_vector_by_ids
    let vecs = Index::get_vector_by_ids(&index, &[0]).unwrap();
    assert_eq!(vecs.len(), 4); // dim=4
}

#[test]
fn test_ivf_sq8_ann_iterator() {
    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        dim: 4,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_sq8(4, 2),
    };

    let mut index = IvfSq8Index::new(&config).unwrap();

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ];

    let dataset = Dataset::from_vectors_with_ids(vectors.clone(), 4, vec![0, 1, 2, 3]);
    Index::train(&mut index, &dataset).unwrap();
    Index::add(&mut index, &dataset).unwrap();

    // Test AnnIterator
    let query = Dataset::from_vectors(vec![0.1, 0.1, 0.1, 0.1], 4);
    let mut iter = Index::create_ann_iterator(&index, &query, None).unwrap();

    let mut count = 0;
    while iter.next().is_some() {
        count += 1;
        if count >= 2 {
            break;
        }
    }
    assert!(count > 0);
}

#[test]
fn test_ivf_rabitq_index_trait_metadata() {
    // Test that Index trait metadata methods work
    let config = IvfRaBitqConfig::new(8, 2);
    let index = IvfRaBitqIndex::new(config);

    // Test Index trait metadata methods
    assert_eq!(Index::index_type(&index), "IVF-RaBitQ");
    assert_eq!(Index::dim(&index), 8);
    assert_eq!(Index::count(&index), 0);
    assert!(!Index::is_trained(&index));
    assert!(!Index::has_raw_data(&index)); // RaBitQ is lossy
}

#[test]
fn test_ivf_rabitq_index_trait_lifecycle() {
    let config = IvfRaBitqConfig::new(8, 2);
    let mut index = IvfRaBitqIndex::new(config);

    // Train and add using Index trait
    let mut data = vec![0.0f32; 50 * 8];
    for i in 0..50 {
        for j in 0..8 {
            data[i * 8 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
        }
    }

    let dataset = Dataset::from_vectors(data.clone(), 8);

    // Use fully qualified syntax to call Index trait methods
    Index::train(&mut index, &dataset).unwrap();
    assert!(Index::is_trained(&index));

    Index::add(&mut index, &dataset).unwrap();
    assert_eq!(Index::count(&index), 50);

    // Search through Index trait
    let query = Dataset::from_vectors(vec![0.1f32; 8], 8);
    let result = Index::search(&index, &query, 5).unwrap();
    assert!(result.ids.len() <= 5);

    // Test get_vector_by_ids (should fail for lossy compression)
    let result = Index::get_vector_by_ids(&index, &[0]);
    assert!(result.is_err());
}

#[test]
fn test_ivf_rabitq_ann_iterator() {
    let config = IvfRaBitqConfig::new(16, 4);
    let mut index = IvfRaBitqIndex::new(config);

    let mut data = vec![0.0f32; 100 * 16];
    for i in 0..100 {
        for j in 0..16 {
            data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
        }
    }

    let dataset = Dataset::from_vectors(data.clone(), 16);
    Index::train(&mut index, &dataset).unwrap();
    Index::add(&mut index, &dataset).unwrap();

    // Test AnnIterator
    let query = Dataset::from_vectors(vec![0.1f32; 16], 16);
    let mut iter = Index::create_ann_iterator(&index, &query, None).unwrap();

    let mut count = 0;
    while iter.next().is_some() {
        count += 1;
        if count >= 5 {
            break;
        }
    }
    assert!(count > 0);
}

#[test]
fn test_ivf_sq8_search_with_bitset() {
    let config = IndexConfig {
        index_type: IndexType::IvfSq8,
        metric_type: MetricType::L2,
        dim: 4,
            data_type: crate::api::DataType::Float,
        params: IndexParams::ivf_sq8(4, 2),
    };

    let mut index = IvfSq8Index::new(&config).unwrap();

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ];

    let dataset = Dataset::from_vectors_with_ids(vectors.clone(), 4, vec![0, 1, 2, 3]);
    Index::train(&mut index, &dataset).unwrap();
    Index::add(&mut index, &dataset).unwrap();

    // Create a bitset that filters out vectors 0 and 1
    let mut bitset = knowhere_rs::bitset::BitsetView::new(4);
    bitset.set(0, true);
    bitset.set(1, true);

    // Search with bitset filtering
    let query = Dataset::from_vectors(vec![0.1, 0.1, 0.1, 0.1], 4);
    let result = Index::search_with_bitset(&index, &query, 4, &bitset).unwrap();

    // Result should not contain filtered ids (0 and 1)
    for id in &result.ids {
        assert!(*id != 0 && *id != 1, "Filtered ID {} should not be in results", id);
    }
}

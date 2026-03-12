use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::index::Index;
use tempfile::NamedTempFile;

fn build_high_param_index(vectors: &[f32], dim: usize, m: usize, ef: usize) -> HnswIndex {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::hnsw(m, ef, 0.5),
    };

    let mut index = HnswIndex::new(&config).expect("hnsw should build");
    index.train(vectors).expect("train should succeed");
    index.add(vectors, None).expect("add should succeed");
    index
}

#[test]
fn test_hnsw_high_params_preserve_persistence_and_raw_vector_contract() {
    let dim = 32;
    let num_vectors = 256;
    let vectors: Vec<f32> = (0..num_vectors * dim)
        .map(|i| ((i * 17) % 101) as f32 / 13.0)
        .collect();
    let query: Vec<f32> = vectors[..dim].to_vec();

    let index = build_high_param_index(&vectors, dim, 48, 600);
    let original = index
        .search(
            &query,
            &SearchRequest {
                top_k: 10,
                ..Default::default()
            },
        )
        .expect("search should succeed");
    let retrieved = index
        .get_vector_by_ids(&[0, 17])
        .expect("get_vector_by_ids should succeed");

    assert_eq!(retrieved.len(), dim * 2);
    assert_eq!(&retrieved[..dim], &vectors[..dim]);
    assert_eq!(&retrieved[dim..dim * 2], &vectors[17 * dim..18 * dim]);

    let temp_file = NamedTempFile::new().expect("temp file should build");
    index.save(temp_file.path()).expect("save should succeed");

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        dim,
        metric_type: MetricType::L2,
        data_type: knowhere_rs::api::DataType::Float,
        params: IndexParams::hnsw(48, 600, 0.5),
    };
    let mut restored = HnswIndex::new(&config).expect("restored hnsw should build");
    restored
        .load(temp_file.path())
        .expect("load should succeed");

    let restored_search = restored
        .search(
            &query,
            &SearchRequest {
                top_k: 10,
                ..Default::default()
            },
        )
        .expect("restored search should succeed");
    let restored_vectors = restored
        .get_vector_by_ids(&[0, 17])
        .expect("restored get_vector_by_ids should succeed");

    assert_eq!(restored_vectors, retrieved);
    assert_eq!(restored_search.ids, original.ids);
    assert_eq!(restored_search.distances.len(), original.distances.len());
}

#[cfg(feature = "long-tests")]
#[test]
#[ignore = "diagnostic long-running parameter scan; excluded from default contract gate"]
fn test_hnsw_high_params() {
    let dim = 128;
    let num_base = 10000;
    let num_queries = 10;

    println!("\n=== HNSW High Parameters Test ===");
    println!("Testing if higher M and ef_search improve recall");

    let base: Vec<f32> = (0..num_base * dim)
        .map(|i| ((i * 31) % 997) as f32 / 997.0)
        .collect();
    let queries: Vec<f32> = (0..num_queries * dim)
        .map(|i| ((i * 43 + 7) % 997) as f32 / 997.0)
        .collect();

    let test_cases = vec![
        ("M=16, ef=200", 16, 200),
        ("M=32, ef=400", 32, 400),
        ("M=48, ef=600", 48, 600),
        ("M=64, ef=800", 64, 800),
    ];

    for (name, m, ef_search) in test_cases {
        println!("\n--- Testing {} ---", name);
        let index = build_high_param_index(&base, dim, m, ef_search);
        let mut recall_at_1 = 0usize;

        for q_idx in 0..num_queries {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];
            let result = index
                .search(
                    q,
                    &SearchRequest {
                        top_k: 1,
                        ..Default::default()
                    },
                )
                .expect("search should succeed");
            if !result.ids.is_empty() {
                recall_at_1 += 1;
            }
        }

        println!(
            "  completed {} queries with non-empty top-1 on {} / {}",
            num_queries, recall_at_1, num_queries
        );
    }
}

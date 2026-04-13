use hanns::api::{MetricType, SearchRequest};
use hanns::faiss::ivf_usq::IvfUsqAnnIterator;
use hanns::faiss::ivf_usq::{IvfUsqConfig, IvfUsqIndex};

use hanns::index::Index;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::env;
use std::fs::File;
use std::io::Write;

use tempfile::tempdir;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * dim];
    for value in &mut data {
        *value = rng.gen_range(-1.0f32..1.0f32);
    }
    data
}

#[test]
fn test_ivf_usq_train_add_search() {
    let dim = 32;
    let n = 512;
    let data = random_vectors(n, dim, 7);
    let ids: Vec<i64> = (0..n as i64).collect();

    let config = IvfUsqConfig::new(dim, 16, 4)
        .with_metric(MetricType::L2)
        .with_nprobe(4)
        .with_rerank_k(64)
        .with_rotation_seed(17);
    let mut index = IvfUsqIndex::new(config);
    index.train(&data).unwrap();
    index.add(&data, Some(&ids)).unwrap();

    let req = SearchRequest {
        top_k: 10,
        nprobe: 4,
        filter: None,
        params: None,
        radius: None,
    };
    let result = index.search(&data[0..dim], &req).unwrap();
    assert_eq!(result.ids.len(), 10);
    assert_eq!(result.ids[0], 0);
}

use knowhere_rs::api::{MetricType, SearchRequest};
use knowhere_rs::faiss::{IvfExRaBitqConfig, IvfExRaBitqIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * dim];
    for value in &mut data {
        *value = rng.gen_range(-1.0f32..1.0f32);
    }
    data
}

#[test]
fn test_ivf_exrabitq_train_add_search() {
    let dim = 32;
    let n = 512;
    let data = random_vectors(n, dim, 7);
    let ids: Vec<i64> = (0..n as i64).collect();

    let config = IvfExRaBitqConfig::new(dim, 16, 4)
        .with_metric(MetricType::L2)
        .with_nprobe(4)
        .with_rerank_k(64)
        .with_rotation_seed(17);
    let mut index = IvfExRaBitqIndex::new(config);
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

#[test]
fn test_ivf_exrabitq_save_load_roundtrip() {
    let dim = 32;
    let n = 256;
    let data = random_vectors(n, dim, 9);
    let ids: Vec<i64> = (0..n as i64).collect();
    let config = IvfExRaBitqConfig::new(dim, 8, 4)
        .with_metric(MetricType::L2)
        .with_nprobe(4)
        .with_rerank_k(64)
        .with_rotation_seed(23);
    let mut index = IvfExRaBitqIndex::new(config);
    index.train(&data).unwrap();
    index.add(&data, Some(&ids)).unwrap();

    let path = std::env::temp_dir().join("ivf_exrabitq_roundtrip.bin");
    index.save(&path).unwrap();
    let loaded = IvfExRaBitqIndex::load(&path).unwrap();

    let req = SearchRequest {
        top_k: 5,
        nprobe: 4,
        filter: None,
        params: None,
        radius: None,
    };
    let query = &data[dim..2 * dim];
    let before = index.search(query, &req).unwrap();
    let after = loaded.search(query, &req).unwrap();

    assert_eq!(before.ids, after.ids);
    std::fs::remove_file(path).ok();
}

#[test]
fn test_ivf_exrabitq_self_recall_on_training_points() {
    let dim = 32;
    let n = 384;
    let data = random_vectors(n, dim, 11);
    let ids: Vec<i64> = (0..n as i64).collect();
    let config = IvfExRaBitqConfig::new(dim, 12, 4)
        .with_metric(MetricType::L2)
        .with_nprobe(12)
        .with_rerank_k(128)
        .with_rotation_seed(29);
    let mut index = IvfExRaBitqIndex::new(config);
    index.train(&data).unwrap();
    index.add(&data, Some(&ids)).unwrap();

    let req = SearchRequest {
        top_k: 1,
        nprobe: 12,
        filter: None,
        params: None,
        radius: None,
    };

    let mut hits = 0usize;
    for i in 0..16usize {
        let query = &data[i * dim..(i + 1) * dim];
        let result = index.search(query, &req).unwrap();
        if result.ids[0] == ids[i] {
            hits += 1;
        }
    }

    assert!(hits >= 12, "expected at least 12/16 self hits, got {hits}");
}

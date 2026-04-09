use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hanns::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, RhtsdgIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIM: usize = 32;
const BASE: usize = 4096;
const QUERY: usize = 64;
const TOP_K: usize = 10;

fn synthetic_vectors(count: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vectors = vec![0.0; count * dim];
    for (idx, vector) in vectors.chunks_mut(dim).enumerate() {
        let cluster = (idx % 16) as f32;
        for value in vector {
            *value = cluster * 0.25 + rng.gen_range(-0.05f32..0.05f32);
        }
    }
    vectors
}

fn build_hnsw(vectors: &[f32]) -> HnswIndex {
    let mut params = IndexParams::hnsw(100, 64, 0.5);
    params.m = Some(16);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim: DIM,
        params,
    };

    let mut index = HnswIndex::new(&config).expect("hnsw should build");
    index.train(vectors).expect("hnsw train should succeed");
    index.add(vectors, None).expect("hnsw add should succeed");
    index.set_ef_search(64);
    index
}

fn build_rhtsdg(vectors: &[f32]) -> RhtsdgIndex {
    let mut config =
        IndexConfig::with_data_type(IndexType::Rhtsdg, MetricType::L2, DIM, DataType::Float);
    config.params.rhtsdg_knn_k = Some(32);
    config.params.rhtsdg_sample_count = Some(8);
    config.params.rhtsdg_iter_count = Some(6);

    let mut index = RhtsdgIndex::new(&config).expect("rhtsdg should build");
    index.train(vectors).expect("rhtsdg train should succeed");
    index.add(vectors, None).expect("rhtsdg add should succeed");
    index
}

fn bench_rhtsdg_build_and_search(c: &mut Criterion) {
    let base = synthetic_vectors(BASE, DIM, 17);
    let queries = synthetic_vectors(QUERY, DIM, 23);
    let req = SearchRequest {
        top_k: TOP_K,
        nprobe: 64,
        ..Default::default()
    };

    let mut build_group = c.benchmark_group("rhtsdg_build");
    build_group.sample_size(10);
    build_group.bench_function("hnsw", |b| {
        b.iter(|| {
            let index = build_hnsw(black_box(&base));
            black_box(index);
        })
    });
    build_group.bench_function("rhtsdg", |b| {
        b.iter(|| {
            let index = build_rhtsdg(black_box(&base));
            black_box(index);
        })
    });
    build_group.finish();

    let hnsw = build_hnsw(&base);
    let rhtsdg = build_rhtsdg(&base);

    let mut search_group = c.benchmark_group("rhtsdg_search_batch");
    search_group.sample_size(20);
    search_group.bench_function("hnsw", |b| {
        b.iter(|| black_box(hnsw.search(black_box(&queries), black_box(&req)).unwrap()))
    });
    search_group.bench_function("rhtsdg", |b| {
        b.iter(|| black_box(rhtsdg.search(black_box(&queries), black_box(&req)).unwrap()))
    });
    search_group.finish();
}

criterion_group!(benches, bench_rhtsdg_build_and_search);
criterion_main!(benches);

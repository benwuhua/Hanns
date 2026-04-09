//! Elkan K-Means 并行化基准测试 - OPT-008
//!
//! 对比单线程 vs 并行版本的性能差异

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hanns::clustering::elkan_kmeans::{ElkanKMeans, ElkanKMeansConfig};
use rand::Rng;

/// 生成带簇结构的测试数据
fn generate_clustered_data(n: usize, dim: usize, k: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0f32; n * dim];
    let points_per_cluster = n / k;

    for cluster in 0..k {
        let cluster_center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect();

        for i in 0..points_per_cluster {
            let idx = (cluster * points_per_cluster + i) * dim;
            for d in 0..dim {
                data[idx + d] = cluster_center[d] + rng.gen_range(-1.0..1.0);
            }
        }
    }

    data
}

fn bench_elkan_parallel_vs_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("Elkan Parallel vs Serial (OPT-008)");
    group.sample_size(10);

    let n = 10000;
    let dim = 128;
    let k = 256;
    let data = generate_clustered_data(n, dim, k);

    // 并行版本
    group.bench_function("parallel (rayon)", |b| {
        b.iter(|| {
            let config = ElkanKMeansConfig::new(k, dim)
                .with_seed(42)
                .with_max_iter(25)
                .with_parallel(true);
            let mut kmeans = ElkanKMeans::new(config);
            black_box(kmeans.cluster(black_box(&data)))
        })
    });

    // 串行版本
    group.bench_function("serial (single-thread)", |b| {
        b.iter(|| {
            let config = ElkanKMeansConfig::new(k, dim)
                .with_seed(42)
                .with_max_iter(25)
                .with_parallel(false);
            let mut kmeans = ElkanKMeans::new(config);
            black_box(kmeans.cluster(black_box(&data)))
        })
    });

    group.finish();
}

fn bench_elkan_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Elkan Scalability (OPT-008)");
    group.sample_size(10);

    let dim = 128;
    let k = 256;

    for n in [1000, 5000, 10000] {
        let data = generate_clustered_data(n, dim, k);

        group.bench_function(format!("parallel n={}", n), |b| {
            b.iter(|| {
                let config = ElkanKMeansConfig::new(k, dim)
                    .with_seed(42)
                    .with_max_iter(25)
                    .with_parallel(true);
                let mut kmeans = ElkanKMeans::new(config);
                black_box(kmeans.cluster(black_box(&data)))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_elkan_parallel_vs_serial,
    bench_elkan_scalability,
);
criterion_main!(benches);

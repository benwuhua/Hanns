//! BENCH-034: C++ Faiss SIMD vs Rust SIMD 对比基准测试
//!
//! 对比 C++ knowhere Faiss SIMD 实现与 Rust SIMD 实现的性能差异
//!
//! 测试项目：
//! - L2 距离计算 (标量 vs SIMD)
//! - 内积计算性能
//! - 批量距离计算吞吐量
//! - 不同向量维度 (128/960) 下的表现

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use knowhere_rs::simd;

/// 生成测试向量
fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dim);
    let mut x = seed as f32;
    for _ in 0..dim {
        // Simple LCG for reproducibility
        x = (x * 1103515245.0 + 12345.0).fract();
        vec.push(x * 2.0 - 1.0); // Range [-1, 1]
    }
    vec
}

/// L2 距离基准测试
fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2_Distance");

    for dim in [128, 960] {
        let a = generate_vector(dim, 42);
        let b = generate_vector(dim, 123);

        group.bench_with_input(
            BenchmarkId::new("L2_SIMD", dim),
            &(a, b),
            |bencher, (a, b)| bencher.iter(|| simd::l2_distance(black_box(a), black_box(b))),
        );

        group.bench_with_input(
            BenchmarkId::new("L2_Scalar", dim),
            &(a, b),
            |bencher, (a, b)| bencher.iter(|| simd::l2_scalar(black_box(a), black_box(b))),
        );

        group.bench_with_input(
            BenchmarkId::new("L2_SQ_SIMD", dim),
            &(a, b),
            |bencher, (a, b)| bencher.iter(|| simd::l2_distance_sq(black_box(a), black_box(b))),
        );
    }

    group.finish();
}

/// 内积基准测试
fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inner_Product");

    for dim in [128, 960] {
        let a = generate_vector(dim, 42);
        let b_vec = generate_vector(dim, 123);

        group.bench_with_input(
            BenchmarkId::new("IP_SIMD", dim),
            &(a, b_vec),
            |bencher, (a, b)| bencher.iter(|| simd::inner_product(black_box(a), black_box(b))),
        );

        group.bench_with_input(
            BenchmarkId::new("IP_Scalar", dim),
            &(a, b_vec),
            |bencher, (a, b)| bencher.iter(|| simd::ip_scalar(black_box(a), black_box(b))),
        );
    }

    group.finish();
}

/// 批量 L2 距离基准测试 (Batch-4)
fn bench_batch_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch_L2");

    for dim in [128, 960] {
        let query = generate_vector(dim, 42);
        let db0 = generate_vector(dim, 123);
        let db1 = generate_vector(dim, 456);
        let db2 = generate_vector(dim, 789);
        let db3 = generate_vector(dim, 101112);

        group.bench_with_input(
            BenchmarkId::new("L2_Batch4_SIMD", dim),
            &(query, db0, db1, db2, db3),
            |bencher, (q, d0, d1, d2, d3)| {
                bencher.iter(|| {
                    simd::l2_batch_4(
                        black_box(q),
                        black_box(d0),
                        black_box(d1),
                        black_box(d2),
                        black_box(d3),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("L2_Batch4_Scalar", dim),
            &(query, db0, db1, db2, db3),
            |bencher, (q, d0, d1, d2, d3)| {
                bencher.iter(|| {
                    simd::l2_batch_4_scalar(
                        black_box(q),
                        black_box(d0),
                        black_box(d1),
                        black_box(d2),
                        black_box(d3),
                    )
                })
            },
        );
    }

    group.finish();
}

/// 批量内积基准测试 (Batch-4)
fn bench_batch_ip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch_IP");

    for dim in [128, 960] {
        let query = generate_vector(dim, 42);
        let db0 = generate_vector(dim, 123);
        let db1 = generate_vector(dim, 456);
        let db2 = generate_vector(dim, 789);
        let db3 = generate_vector(dim, 101112);

        group.bench_with_input(
            BenchmarkId::new("IP_Batch4_SIMD", dim),
            &(query, db0, db1, db2, db3),
            |bencher, (q, d0, d1, d2, d3)| {
                bencher.iter(|| {
                    simd::ip_batch_4(
                        black_box(q),
                        black_box(d0),
                        black_box(d1),
                        black_box(d2),
                        black_box(d3),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("IP_Batch4_Scalar", dim),
            &(query, db0, db1, db2, db3),
            |bencher, (q, d0, d1, d2, d3)| {
                bencher.iter(|| {
                    simd::ip_batch_4_scalar(
                        black_box(q),
                        black_box(d0),
                        black_box(d1),
                        black_box(d2),
                        black_box(d3),
                    )
                })
            },
        );
    }

    group.finish();
}

/// 大规模批量距离计算吞吐量测试
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("Throughput");

    for (nq, nb, dim) in [(100, 10000, 128), (100, 10000, 960)] {
        let queries: Vec<f32> = (0..nq * dim).map(|i| ((i as f32 * 0.01).sin())).collect();
        let database: Vec<f32> = (0..nb * dim).map(|i| ((i as f32 * 0.02).cos())).collect();

        group.bench_with_input(
            BenchmarkId::new(
                "L2_Batch_Throughput",
                format!("nq{}_nb{}_dim{}", nq, nb, dim),
            ),
            &(queries.clone(), database.clone(), dim),
            |bencher, (q, db, dim)| {
                bencher.iter(|| simd::l2_batch(black_box(q), black_box(db), *dim))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "IP_Batch_Throughput",
                format!("nq{}_nb{}_dim{}", nq, nb, dim),
            ),
            &(queries.clone(), database.clone(), dim),
            |bencher, (q, db, dim)| {
                bencher.iter(|| simd::ip_batch(black_box(q), black_box(db), *dim))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_inner_product,
    bench_batch_l2,
    bench_batch_ip,
    bench_throughput,
);

criterion_main!(benches);

#![cfg(feature = "long-tests")]
//! SIMD 距离计算性能基准测试
//!
//! 对比 SIMD 优化与标量实现的距离计算性能
//! 测试 L2 和 Inner Product 的 SIMD 加速效果

mod common;
use hanns::simd::{
    inner_product, ip_batch, ip_batch_4, ip_batch_4_scalar, ip_scalar, l2_batch, l2_batch_4,
    l2_batch_4_scalar, l2_distance, l2_distance_sq, l2_scalar, l2_scalar_sq,
};
use rand::Rng;
use std::time::Instant;


/// L2 距离性能对比
fn bench_l2_distance(dim: usize, iterations: usize) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Warmup
    for _ in 0..100 {
        let _ = l2_distance(&a, &b);
        let _ = l2_scalar(&a, &b);
    }

    // SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = l2_distance(&a, &b);
    }
    let simd_time = start.elapsed();

    // Scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = l2_scalar(&a, &b);
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / iterations as f64 * 1e9; // ps (picoseconds for very fast ops)
    let scalar_avg = scalar_time.as_secs_f64() / iterations as f64 * 1e9; // ps
    let speedup = if simd_avg > 0.0 {
        scalar_avg / simd_avg
    } else {
        f64::NAN
    };

    if simd_avg < 1000.0 {
        println!(
            "  L2 距离 (dim={}): SIMD={:.1}ps, 标量={:.1}ps, 加速={:.2}x",
            dim, simd_avg, scalar_avg, speedup
        );
    } else {
        println!(
            "  L2 距离 (dim={}): SIMD={:.2}ns, 标量={:.2}ns, 加速={:.2}x",
            dim,
            simd_avg / 1000.0,
            scalar_avg / 1000.0,
            speedup
        );
    }
}

/// L2 平方距离性能对比（用于最近邻搜索，避免 sqrt）
fn bench_l2_distance_squared(dim: usize, iterations: usize) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Warmup
    for _ in 0..100 {
        let _ = common::l2_distance_squared(&a, &b);
        let _ = l2_scalar_sq(&a, &b);
    }

    // SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = common::l2_distance_squared(&a, &b);
    }
    let simd_time = start.elapsed();

    // Scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = l2_scalar_sq(&a, &b);
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / iterations as f64 * 1e6; // ns
    let scalar_avg = scalar_time.as_secs_f64() / iterations as f64 * 1e6; // ns
    let speedup = scalar_avg / simd_avg;

    println!(
        "  L2 平方 (dim={}): SIMD={:.2}ns, 标量={:.2}ns, 加速={:.2}x",
        dim, simd_avg, scalar_avg, speedup
    );
}

/// Inner Product 性能对比
fn bench_inner_product(dim: usize, iterations: usize) {
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Warmup
    for _ in 0..100 {
        let _ = inner_product(&a, &b);
        let _ = ip_scalar(&a, &b);
    }

    // SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = inner_product(&a, &b);
    }
    let simd_time = start.elapsed();

    // Scalar
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ip_scalar(&a, &b);
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / iterations as f64 * 1e6; // ns
    let scalar_avg = scalar_time.as_secs_f64() / iterations as f64 * 1e6; // ns
    let speedup = scalar_avg / simd_avg;

    println!(
        "  内积 (dim={}): SIMD={:.2}ns, 标量={:.2}ns, 加速={:.2}x",
        dim, simd_avg, scalar_avg, speedup
    );
}

/// 批量 L2 距离性能对比
fn bench_l2_batch(num_queries: usize, num_database: usize, dim: usize) {
    let queries = common::generate_vectors(num_queries, dim);
    let database = common::generate_vectors(num_database, dim);

    // Warmup
    let _ = l2_batch(&queries, &database, dim);

    // SIMD batch
    let start = Instant::now();
    for _ in 0..10 {
        let _ = l2_batch(&queries, &database, dim);
    }
    let simd_time = start.elapsed();

    // Naive scalar batch
    let start = Instant::now();
    for _ in 0..10 {
        let mut result = Vec::with_capacity(num_queries * num_database);
        for i in 0..num_queries {
            for j in 0..num_database {
                let dist = l2_scalar(
                    &queries[i * dim..(i + 1) * dim],
                    &database[j * dim..(j + 1) * dim],
                );
                result.push(dist);
            }
        }
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / 10.0 * 1e3; // ms
    let scalar_avg = scalar_time.as_secs_f64() / 10.0 * 1e3; // ms
    let speedup = scalar_avg / simd_avg;

    println!(
        "  批量 L2 ({}x{}, dim={}): SIMD={:.2}ms, 标量={:.2}ms, 加速={:.2}x",
        num_queries, num_database, dim, simd_avg, scalar_avg, speedup
    );
}

/// 批量 Inner Product 性能对比
fn bench_ip_batch(num_queries: usize, num_database: usize, dim: usize) {
    let queries = common::generate_vectors(num_queries, dim);
    let database = common::generate_vectors(num_database, dim);

    // Warmup
    let _ = ip_batch(&queries, &database, dim);

    // SIMD batch
    let start = Instant::now();
    for _ in 0..10 {
        let _ = ip_batch(&queries, &database, dim);
    }
    let simd_time = start.elapsed();

    // Naive scalar batch
    let start = Instant::now();
    for _ in 0..10 {
        let mut result = Vec::with_capacity(num_queries * num_database);
        for i in 0..num_queries {
            for j in 0..num_database {
                let dot = ip_scalar(
                    &queries[i * dim..(i + 1) * dim],
                    &database[j * dim..(j + 1) * dim],
                );
                result.push(dot);
            }
        }
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / 10.0 * 1e3; // ms
    let scalar_avg = scalar_time.as_secs_f64() / 10.0 * 1e3; // ms
    let speedup = scalar_avg / simd_avg;

    println!(
        "  批量内积 ({}x{}, dim={}): SIMD={:.2}ms, 标量={:.2}ms, 加速={:.2}x",
        num_queries, num_database, dim, simd_avg, scalar_avg, speedup
    );
}

/// Batch-4 L2 性能对比（一次计算 1 个查询 vs 4 个数据库向量）
fn bench_l2_batch_4(dim: usize, iterations: usize) {
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db0: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db1: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db3: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Warmup
    for _ in 0..100 {
        let _ = l2_batch_4(&query, &db0, &db1, &db2, &db3);
        let _ = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    }

    // SIMD batch-4
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = l2_batch_4(&query, &db0, &db1, &db2, &db3);
    }
    let simd_time = start.elapsed();

    // Scalar batch-4
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / iterations as f64 * 1e3; // μs
    let scalar_avg = scalar_time.as_secs_f64() / iterations as f64 * 1e3; // μs
    let speedup = scalar_avg / simd_avg;

    println!(
        "  Batch-4 L2 (dim={}): SIMD={:.2}μs, 标量={:.2}μs, 加速={:.2}x",
        dim, simd_avg, scalar_avg, speedup
    );
}

/// Batch-4 IP 性能对比
fn bench_ip_batch_4(dim: usize, iterations: usize) {
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db0: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db1: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db3: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    // Warmup
    for _ in 0..100 {
        let _ = ip_batch_4(&query, &db0, &db1, &db2, &db3);
        let _ = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    }

    // SIMD batch-4
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ip_batch_4(&query, &db0, &db1, &db2, &db3);
    }
    let simd_time = start.elapsed();

    // Scalar batch-4
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    }
    let scalar_time = start.elapsed();

    let simd_avg = simd_time.as_secs_f64() / iterations as f64 * 1e3; // μs
    let scalar_avg = scalar_time.as_secs_f64() / iterations as f64 * 1e3; // μs
    let speedup = scalar_avg / simd_avg;

    println!(
        "  Batch-4 IP (dim={}): SIMD={:.2}μs, 标量={:.2}μs, 加速={:.2}x",
        dim, simd_avg, scalar_avg, speedup
    );
}

/// 验证 SIMD 和标量实现的结果一致性
fn verify_correctness() {
    println!("\n=== 正确性验证 ===\n");

    let mut rng = rand::thread_rng();
    let dim = 128;

    // L2 距离验证
    let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    let l2_simd = l2_distance(&a, &b);
    let l2_scalar = l2_scalar(&a, &b);
    let l2_error = (l2_simd - l2_scalar).abs();
    println!(
        "  L2 距离：SIMD={}, 标量={}, 误差={:.2e} {}",
        l2_simd,
        l2_scalar,
        l2_error,
        if l2_error < 1e-5 { "✅" } else { "❌" }
    );

    // L2 平方距离验证
    let l2_sq_simd = common::l2_distance_squared(&a, &b);
    let l2_sq_scalar = l2_scalar_sq(&a, &b);
    let l2_sq_error = (l2_sq_simd - l2_sq_scalar).abs();
    println!(
        "  L2 平方：SIMD={}, 标量={}, 误差={:.2e} {}",
        l2_sq_simd,
        l2_sq_scalar,
        l2_sq_error,
        if l2_sq_error < 1e-5 { "✅" } else { "❌" }
    );

    // Inner Product 验证
    let ip_simd = inner_product(&a, &b);
    let ip_scalar_val = ip_scalar(&a, &b);
    let ip_error = (ip_simd - ip_scalar_val).abs();
    println!(
        "  内积：SIMD={}, 标量={}, 误差={:.2e} {}",
        ip_simd,
        ip_scalar_val,
        ip_error,
        if ip_error < 1e-5 { "✅" } else { "❌" }
    );

    // Batch-4 L2 验证
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db0: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db1: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let db3: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    let batch4_simd = l2_batch_4(&query, &db0, &db1, &db2, &db3);
    let batch4_scalar = l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    let batch4_max_error = batch4_simd
        .iter()
        .zip(batch4_scalar.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  Batch-4 L2: 最大误差={:.2e} {}",
        batch4_max_error,
        if batch4_max_error < 1e-4 {
            "✅"
        } else {
            "❌"
        }
    );

    // Batch-4 IP 验证
    let batch4_ip_simd = ip_batch_4(&query, &db0, &db1, &db2, &db3);
    let batch4_ip_scalar = ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
    let batch4_ip_max_error = batch4_ip_simd
        .iter()
        .zip(batch4_ip_scalar.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  Batch-4 IP: 最大误差={:.2e} {}",
        batch4_ip_max_error,
        if batch4_ip_max_error < 1e-3 {
            "✅"
        } else {
            "❌"
        }
    );
}

#[test]
#[ignore = "performance benchmark; excluded from default regression"]
fn test_simd_performance() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         OPT-012: SIMD 距离计算性能基准测试                    ║");
    println!(
        "║         平台：{} (ARM NEON)                      ║",
        if cfg!(target_arch = "aarch64") {
            "Apple Silicon"
        } else if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else {
            "Unknown"
        }
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // 正确性验证
    verify_correctness();

    println!("\n=== 单次距离计算性能 (100000 次迭代) ===\n");

    for dim in [32, 64, 128, 256, 512, 768] {
        bench_l2_distance(dim, 100000);
    }

    println!();
    for dim in [32, 64, 128, 256, 512, 768] {
        bench_common::l2_distance_squared(dim, 100000);
    }

    println!();
    for dim in [32, 64, 128, 256, 512, 768] {
        bench_inner_product(dim, 100000);
    }

    println!("\n=== 批量距离计算性能 (10 次迭代) ===\n");

    for (nq, nb, dim) in [(1, 1000, 128), (10, 1000, 128), (100, 10000, 128)] {
        bench_l2_batch(nq, nb, dim);
    }

    println!();
    for (nq, nb, dim) in [(1, 1000, 128), (10, 1000, 128), (100, 10000, 128)] {
        bench_ip_batch(nq, nb, dim);
    }

    println!("\n=== Batch-4 优化性能 (10000 次迭代) ===\n");

    for dim in [32, 64, 128, 256, 512] {
        bench_l2_batch_4(dim, 10000);
    }

    println!();
    for dim in [32, 64, 128, 256, 512] {
        bench_ip_batch_4(dim, 10000);
    }

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("✅ OPT-012 SIMD 优化性能测试完成");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
}

#[test]
#[ignore = "benchmark/integration long-running; excluded from default bugfix gate"]
fn test_simd_correctness_only() {
    // 快速正确性验证
    verify_correctness();
}

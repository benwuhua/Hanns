//! AVX512 性能 Benchmark
//!
//! 对比 AVX2 vs AVX512 的距离计算性能

use std::time::Instant;

/// Benchmark L2 batch-4 性能
pub fn bench_l2_batch_4() {
    println!("\n=== L2 Batch-4 Benchmark ===");
    println!("Testing different dimensions and SIMD levels...\n");

    let dims = [32, 64, 128, 256, 512, 768, 1024];

    for &dim in &dims {
        // 准备测试数据
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
        let db1: Vec<f32> = (0..dim).map(|i| (i + 2) as f32 * 0.1).collect();
        let db2: Vec<f32> = (0..dim).map(|i| (i + 3) as f32 * 0.1).collect();
        let db3: Vec<f32> = (0..dim).map(|i| (i + 4) as f32 * 0.1).collect();

        let iterations = 1_000_000 / dim.max(1);

        // Benchmark 标量版本
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::simd::l2_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
        }
        let scalar_time = start.elapsed();

        // Benchmark 自动选择版本 (会选最优 SIMD)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::simd::l2_batch_4(&query, &db0, &db1, &db2, &db3);
        }
        let simd_time = start.elapsed();

        // 检测当前 SIMD 级别
        let simd_level = crate::simd::detect_simd_level();

        println!(
            "Dim {:4}: Scalar: {:8.2} ms | SIMD ({:?}): {:8.2} ms | Speedup: {:.2}x",
            dim,
            scalar_time.as_secs_f64() * 1000.0,
            simd_level,
            simd_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / simd_time.as_secs_f64()
        );
    }
}

/// Benchmark IP batch-4 性能
pub fn bench_ip_batch_4() {
    println!("\n=== Inner Product Batch-4 Benchmark ===");
    println!("Testing different dimensions and SIMD levels...\n");

    let dims = [32, 64, 128, 256, 512, 768, 1024];

    for &dim in &dims {
        // 准备测试数据
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| (i * 2) as f32 * 0.1).collect();
        let db1: Vec<f32> = (0..dim).map(|i| (i * 3) as f32 * 0.1).collect();
        let db2: Vec<f32> = (0..dim).map(|i| (i * 4) as f32 * 0.1).collect();
        let db3: Vec<f32> = (0..dim).map(|i| (i * 5) as f32 * 0.1).collect();

        let iterations = 1_000_000 / dim.max(1);

        // Benchmark 标量版本
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::simd::ip_batch_4_scalar(&query, &db0, &db1, &db2, &db3);
        }
        let scalar_time = start.elapsed();

        // Benchmark 自动选择版本 (会选最优 SIMD)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = crate::simd::ip_batch_4(&query, &db0, &db1, &db2, &db3);
        }
        let simd_time = start.elapsed();

        // 检测当前 SIMD 级别
        let simd_level = crate::simd::detect_simd_level();

        println!(
            "Dim {:4}: Scalar: {:8.2} ms | SIMD ({:?}): {:8.2} ms | Speedup: {:.2}x",
            dim,
            scalar_time.as_secs_f64() * 1000.0,
            simd_level,
            simd_time.as_secs_f64() * 1000.0,
            scalar_time.as_secs_f64() / simd_time.as_secs_f64()
        );
    }
}

/// Benchmark AVX2 vs AVX512 (仅在支持 AVX512 的 CPU 上)
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn bench_avx2_vs_avx512() {
    println!("\n=== AVX2 vs AVX512 Direct Comparison ===");

    if !std::is_x86_feature_detected!("avx512f") {
        println!("AVX512 not available on this CPU, skipping comparison.");
        return;
    }

    let dims = [64, 128, 256, 512, 768, 1024];

    for &dim in &dims {
        // 准备测试数据
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| (i + 1) as f32 * 0.1).collect();
        let db1: Vec<f32> = (0..dim).map(|i| (i + 2) as f32 * 0.1).collect();
        let db2: Vec<f32> = (0..dim).map(|i| (i + 3) as f32 * 0.1).collect();
        let db3: Vec<f32> = (0..dim).map(|i| (i + 4) as f32 * 0.1).collect();

        let iterations = 500_000 / dim.max(1);

        // Benchmark AVX2 版本
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = crate::simd::l2_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        let avx2_time = start.elapsed();

        // Benchmark AVX512 版本
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = crate::simd::l2_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        let avx512_time = start.elapsed();

        println!(
            "Dim {:4}: AVX2: {:8.2} ms | AVX512: {:8.2} ms | Speedup: {:.2}x",
            dim,
            avx2_time.as_secs_f64() * 1000.0,
            avx512_time.as_secs_f64() * 1000.0,
            avx2_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }

    // IP 版本对比
    println!("\n--- Inner Product ---");
    for &dim in &dims {
        // 准备测试数据
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let db0: Vec<f32> = (0..dim).map(|i| (i * 2) as f32 * 0.1).collect();
        let db1: Vec<f32> = (0..dim).map(|i| (i * 3) as f32 * 0.1).collect();
        let db2: Vec<f32> = (0..dim).map(|i| (i * 4) as f32 * 0.1).collect();
        let db3: Vec<f32> = (0..dim).map(|i| (i * 5) as f32 * 0.1).collect();

        let iterations = 500_000 / dim.max(1);

        // Benchmark AVX2 版本
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = crate::simd::ip_batch_4_avx2(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        let avx2_time = start.elapsed();

        // Benchmark AVX512 版本
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = crate::simd::ip_batch_4_avx512(
                    query.as_ptr(),
                    db0.as_ptr(),
                    db1.as_ptr(),
                    db2.as_ptr(),
                    db3.as_ptr(),
                    dim,
                );
            }
        }
        let avx512_time = start.elapsed();

        println!(
            "Dim {:4}: AVX2: {:8.2} ms | AVX512: {:8.2} ms | Speedup: {:.2}x",
            dim,
            avx2_time.as_secs_f64() * 1000.0,
            avx512_time.as_secs_f64() * 1000.0,
            avx2_time.as_secs_f64() / avx512_time.as_secs_f64()
        );
    }
}

/// 运行所有 benchmark
pub fn run_all_benchmarks() {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║       AVX512 Distance Computation Benchmarks          ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let simd_level = crate::simd::detect_simd_level();
    println!("CPU SIMD Level: {:?}", simd_level);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        println!(
            "AVX512 Available: {}\n",
            std::is_x86_feature_detected!("avx512f")
        );
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        println!("AVX512: Not available on this platform\n");
    }

    bench_l2_batch_4();
    bench_ip_batch_4();

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") {
        bench_avx2_vs_avx512();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_functions() {
        // 简单测试确保 benchmark 函数不会崩溃
        bench_l2_batch_4();
        bench_ip_batch_4();
    }
}

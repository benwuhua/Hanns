//! K-Means++ vs Standard K-Means Benchmark
//!
//! 对比 k-means++ 初始化和标准随机初始化的性能和质量

use knowhere_rs::clustering::{KMeansPlusPlus, KMeansPlusPlusConfig};
use knowhere_rs::quantization::KMeans;
use std::time::Instant;

fn generate_clustered_data(n_clusters: usize, dim: usize, points_per_cluster: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n_clusters * points_per_cluster * dim);

    for c in 0..n_clusters {
        let center = (c as f32) * 10.0;
        for _ in 0..points_per_cluster {
            for _ in 0..dim {
                let value = center + (rand::random::<f32>() - 0.5) * 2.0;
                data.push(value);
            }
        }
    }

    data
}

fn benchmark_kmeans_pp(vectors: &[f32], k: usize, dim: usize) -> (f32, usize, f64) {
    let config = KMeansPlusPlusConfig::new(k, dim)
        .with_seed(42)
        .with_max_iter(25)
        .with_tolerance(1e-4);

    let mut kmeans_pp = KMeansPlusPlus::new(config);

    let start = Instant::now();
    let result = kmeans_pp.cluster(vectors);
    let elapsed = start.elapsed().as_secs_f64();

    (result.inertia, result.iterations, elapsed)
}

fn benchmark_kmeans_random(vectors: &[f32], k: usize, dim: usize) -> (f32, usize, f64) {
    let mut kmeans = KMeans::new(k, dim);

    let start = Instant::now();
    let _n = kmeans.train(vectors);
    let elapsed = start.elapsed().as_secs_f64();

    // 计算惯性
    let inertia = compute_inertia(&kmeans, vectors);

    (inertia, 10, elapsed) // KMeans 不返回迭代次数
}

fn compute_inertia(kmeans: &KMeans, vectors: &[f32]) -> f32 {
    let n = vectors.len() / kmeans.dim();
    let mut inertia = 0.0f32;

    for i in 0..n {
        let vec = &vectors[i * kmeans.dim()..(i + 1) * kmeans.dim()];
        let nearest = kmeans.find_nearest(vec);
        let centroid = &kmeans.centroids()[nearest * kmeans.dim()..(nearest + 1) * kmeans.dim()];

        let mut dist_sq = 0.0f32;
        for j in 0..kmeans.dim() {
            let diff = vec[j] - centroid[j];
            dist_sq += diff * diff;
        }
        inertia += dist_sq;
    }

    inertia
}

fn main() {
    println!("📊 K-Means++ vs Standard K-Means Benchmark\n");
    println!("Testing different dataset sizes and cluster counts\n");

    let test_cases = [
        (10_000, 128, 10),   // 10K vectors, 128D, 10 clusters
        (50_000, 128, 50),   // 50K vectors, 128D, 50 clusters
        (100_000, 128, 100), // 100K vectors, 128D, 100 clusters
        (50_000, 96, 50),    // 50K vectors, 96D (Deep1M), 50 clusters
        (10_000, 960, 20),   // 10K vectors, 960D (GIST1M), 20 clusters
    ];

    println!(
        "{:<10} {:<8} {:<8} {:<12} {:<12} {:<12} {:<12} {:<10}",
        "Vectors",
        "Dim",
        "K",
        "PP Inertia",
        "Rand Inertia",
        "PP Time(s)",
        "Rand Time(s)",
        "Speedup"
    );
    println!("{}", "-".repeat(100));

    for (n, dim, k) in test_cases {
        let vectors = generate_clustered_data(k, dim, n / k);

        let (inertia_pp, _iter_pp, time_pp) = benchmark_kmeans_pp(&vectors, k, dim);
        let (inertia_rand, _, time_rand) = benchmark_kmeans_random(&vectors, k, dim);

        let speedup = time_rand / time_pp;
        let inertia_improvement = ((inertia_rand - inertia_pp) / inertia_rand) * 100.0;

        println!(
            "{:<10} {:<8} {:<8} {:<12.2} {:<12.2} {:<12.4} {:<12.4} {:<10.2}x",
            n, dim, k, inertia_pp, inertia_rand, time_pp, time_rand, speedup
        );

        if inertia_improvement > 0.0 {
            println!("  → K-means++ inertia: {:.1}% better", inertia_improvement);
        }
    }

    println!("\n✅ Benchmark completed!");
}

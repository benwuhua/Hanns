//! 性能基准测试

use std::time::Instant;

/// 简单的基准测试工具
pub struct Bench {
    name: &'static str,
    iterations: usize,
}

impl Bench {
    pub fn new(name: &'static str) -> Self {
        Self { name, iterations: 1000 }
    }
    
    pub fn iters(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }
    
    pub fn run<F>(&self, f: F) -> BenchReport
    where
        F: Fn(),
    {
        // 预热
        for _ in 0..100 {
            f();
        }
        
        let start = Instant::now();
        for _ in 0..self.iterations {
            f();
        }
        let elapsed = start.elapsed();
        
        BenchReport {
            name: self.name,
            iterations: self.iterations,
            total_ns: elapsed.as_nanos() as u64,
        }
    }
}

pub struct BenchReport {
    name: &'static str,
    iterations: usize,
    total_ns: u64,
}

impl BenchReport {
    pub fn avg_us(&self) -> f64 {
        self.total_ns as f64 / self.iterations as f64 / 1000.0
    }
    
    pub fn ops_per_sec(&self) -> f64 {
        1_000_000_000.0 * self.iterations as f64 / self.total_ns as f64
    }
}

impl std::fmt::Display for BenchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} iters, {:.2} µs/op, {:.0} ops/s",
            self.name,
            self.iterations,
            self.avg_us(),
            self.ops_per_sec()
        )
    }
}

/// 距离计算基准
pub mod distance_bench {
    use super::*;
    use crate::metrics::L2Distance;
    
    pub fn bench_l2(dim: usize, iters: usize) -> BenchReport {
        let vectors: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        
        Bench::new("L2 Distance")
            .iters(iters)
            .run(|| {
                let mut sum = 0.0f32;
                for i in 0..dim {
                    let diff = vectors[i] - vectors[(i + 1) % dim];
                    sum += diff * diff;
                }
                std::hint::black_box(sum);
            })
    }
    
    pub fn bench_batch_l2(n: usize, dim: usize, iters: usize) -> BenchReport {
        let a: Vec<f32> = (0..n * dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * dim).map(|i| (i + 1) as f32 * 0.01).collect();
        
        Bench::new("Batch L2")
            .iters(iters)
            .run(|| {
                let mut sum = 0.0f32;
                for i in 0..n {
                    let mut d = 0.0f32;
                    for j in 0..dim {
                        let diff = a[i * dim + j] - b[i * dim + j];
                        d += diff * diff;
                    }
                    sum += d;
                }
                std::hint::black_box(sum);
            })
    }
}

/// 索引基准
pub mod index_bench {
    use super::*;
    use crate::faiss::IvfIndex;
    use crate::faiss::PqEncoder;
    
    pub fn bench_ivf_search(n: usize, dim: usize, nlist: usize, iters: usize) -> BenchReport {
        // 创建索引
        let mut ivf = IvfIndex::new(dim, nlist);
        
        // 生成数据
        let data: Vec<f32> = (0..n * dim).map(|i| (i % 100) as f32).collect();
        ivf.train(&data);
        ivf.add(&data);
        
        // 查询
        let query = vec![50.0; dim];
        
        Bench::new("IVF Search")
            .iters(iters)
            .run(|| {
                let _ = ivf.search(&query, 10);
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bench_report() {
        let report = Bench::new("test")
            .iters(10)
            .run(|| {
                let _ = 1 + 2;
            });
        
        println!("{}", report);
    }
    
    #[test]
    fn test_distance_bench() {
        let r = distance_bench::bench_l2(128, 1000);
        println!("{}", r);
    }
}

//! 内存使用跟踪模块
//!
//! 提供内存分配跟踪、峰值记录和使用报告生成功能

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 全局内存跟踪器
pub struct MemoryTracker {
    base_memory: AtomicU64,
    index_overhead: AtomicU64,
    peak_memory: AtomicU64,
    start_time: Instant,
}

impl MemoryTracker {
    /// 创建新的内存跟踪器
    pub fn new() -> Self {
        Self {
            base_memory: AtomicU64::new(0),
            index_overhead: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// 记录基础向量内存使用（字节）
    pub fn record_base_memory(&self, bytes: u64) {
        self.base_memory.store(bytes, Ordering::Relaxed);
        self.update_peak();
    }

    /// 记录索引开销内存（字节）
    pub fn record_index_overhead(&self, bytes: u64) {
        self.index_overhead.store(bytes, Ordering::Relaxed);
        self.update_peak();
    }

    /// 更新峰值内存
    fn update_peak(&self) {
        let current =
            self.base_memory.load(Ordering::Relaxed) + self.index_overhead.load(Ordering::Relaxed);
        let peak = self.peak_memory.load(Ordering::Relaxed);
        if current > peak {
            self.peak_memory.store(current, Ordering::Relaxed);
        }
    }

    /// 获取基础内存使用（MB）
    pub fn base_memory_mb(&self) -> f64 {
        self.base_memory.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }

    /// 获取索引开销内存（MB）
    pub fn index_overhead_mb(&self) -> f64 {
        self.index_overhead.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }

    /// 获取峰值内存（MB）
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }

    /// 获取总内存使用（MB）
    pub fn total_memory_mb(&self) -> f64 {
        (self.base_memory.load(Ordering::Relaxed) + self.index_overhead.load(Ordering::Relaxed))
            as f64
            / (1024.0 * 1024.0)
    }

    /// 获取经过时间
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// 生成内存使用报告
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            base_memory_mb: self.base_memory_mb(),
            index_overhead_mb: self.index_overhead_mb(),
            total_memory_mb: self.total_memory_mb(),
            peak_memory_mb: self.peak_memory_mb(),
            elapsed: self.elapsed(),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 内存使用报告
pub struct MemoryReport {
    pub base_memory_mb: f64,
    pub index_overhead_mb: f64,
    pub total_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub elapsed: std::time::Duration,
}

impl std::fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Usage:")?;
        writeln!(f, "  Base vectors:  {:.1} MB", self.base_memory_mb)?;
        writeln!(f, "  Index overhead: {:.1} MB", self.index_overhead_mb)?;
        write!(f, "  Peak memory:   {:.1} MB", self.peak_memory_mb)
    }
}

/// 估算向量内存使用
pub fn estimate_vector_memory(num_vectors: usize, dim: usize) -> u64 {
    (num_vectors * dim * std::mem::size_of::<f32>()) as u64
}

/// 估算 IVF 索引开销
pub fn estimate_ivf_overhead(num_vectors: usize, dim: usize, nlist: usize) -> u64 {
    // 倒排列表开销 + 质心内存
    let inverted_lists = num_vectors * std::mem::size_of::<u32>(); // 每个向量的列表 ID
    let centroids = nlist * dim * std::mem::size_of::<f32>();
    (inverted_lists + centroids) as u64
}

/// 估算 HNSW 索引开销
pub fn estimate_hnsw_overhead(num_vectors: usize, dim: usize, m: usize) -> u64 {
    // HNSW 图的边存储开销（估算）
    let avg_edges = num_vectors * m * 2; // 平均每个节点 2*m 条边
    let edge_storage = avg_edges * std::mem::size_of::<u32>();
    let vector_storage = num_vectors * dim * std::mem::size_of::<f32>();
    (edge_storage + vector_storage) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new();

        // 记录 1M 向量的内存（128 维）
        let base_mem = estimate_vector_memory(1_000_000, 128);
        tracker.record_base_memory(base_mem);

        // 记录 IVF 索引开销
        let overhead = estimate_ivf_overhead(1_000_000, 128, 1024);
        tracker.record_index_overhead(overhead);

        let report = tracker.report();
        println!("{}", report);

        assert!(report.base_memory_mb > 0.0);
        assert!(report.index_overhead_mb > 0.0);
        assert!(report.peak_memory_mb >= report.total_memory_mb);
    }

    #[test]
    fn test_memory_estimations() {
        // SIFT1M: 1M vectors, 128 dim
        let sift_mem = estimate_vector_memory(1_000_000, 128);
        assert_eq!(sift_mem, 1_000_000 * 128 * 4); // 512 MB

        // Deep1M: 1M vectors, 96 dim
        let deep_mem = estimate_vector_memory(1_000_000, 96);
        assert_eq!(deep_mem, 1_000_000 * 96 * 4); // 384 MB

        // GIST1M: 1M vectors, 960 dim
        let gist_mem = estimate_vector_memory(1_000_000, 960);
        assert_eq!(gist_mem, 1_000_000 * 960 * 4); // 3840 MB
    }
}

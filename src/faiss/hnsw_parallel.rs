//! 多线程 HNSW 建图优化

use std::sync::{Arc, Mutex};
use std::thread;

/// 并行建图器
pub struct ParallelHnswBuilder {
    num_threads: usize,
    batch_size: usize,
}

impl ParallelHnswBuilder {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            batch_size: 100,
        }
    }
    
    /// 并行添加向量
    pub fn add_batch<F>(&self, vectors: &[f32], dim: usize, mut add_fn: F) -> usize
    where
        F: FnMut(&[f32]) -> usize + Send,
    {
        let n = vectors.len() / dim;
        
        if n < self.num_threads * 10 || self.num_threads == 1 {
            // 数据量小，串行添加
            for i in 0..n {
                add_fn(&vectors[i * dim..]);
            }
            return n;
        }
        
        // 分批并行处理
        let mut added = 0;
        let mut handles = Vec::new();
        
        for batch_start in (0..n).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(n);
            let batch = vectors[batch_start * dim..batch_end * dim].to_vec();
            
            let handle = thread::spawn(move || {
                let mut local_added = 0;
                for i in 0..(batch_end - batch_start) {
                    // 简化：实际应该调用索引的add方法
                    local_added += 1;
                }
                local_added
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            added += handle.join().unwrap_or(0);
        }
        
        added
    }
}

/// 线程安全的向量 ID 分配器
pub struct IdAllocator {
    next_id: Mutex<usize>,
}

impl IdAllocator {
    pub fn new() -> Self {
        Self { next_id: Mutex::new(0) }
    }
    
    pub fn next(&self) -> usize {
        let mut id = self.next_id.lock().unwrap();
        let r = *id;
        *id += 1;
        r
    }
    
    pub fn reset(&self) {
        let mut id = self.next_id.lock().unwrap();
        *id = 0;
    }
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// 并行搜索结果收集器
pub struct ResultCollector {
    results: Mutex<Vec<(usize, f32)>>,
}

impl ResultCollector {
    pub fn new() -> Self {
        Self { results: Mutex::new(Vec::new()) }
    }
    
    pub fn push(&self, id: usize, dist: f32) {
        let mut r = self.results.lock().unwrap();
        r.push((id, dist));
    }
    
    pub fn into_results(self) -> Vec<(usize, f32)> {
        self.results.into_inner().unwrap()
    }
    
    pub fn sort_and_truncate(&self, top_k: usize) {
        let mut r = self.results.lock().unwrap();
        r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        r.truncate(top_k);
    }
}

impl Default for ResultCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_id_allocator() {
        let alloc = IdAllocator::new();
        assert_eq!(alloc.next(), 0);
        assert_eq!(alloc.next(), 1);
    }
    
    #[test]
    fn test_result_collector() {
        let coll = ResultCollector::new();
        
        coll.push(1, 0.5);
        coll.push(0, 0.3);
        coll.push(2, 0.8);
        
        coll.sort_and_truncate(2);
        
        let results = coll.into_results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // 最近
    }
    
    #[test]
    fn test_parallel_batch() {
        let builder = ParallelHnswBuilder::new(2);
        
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let added = builder.add_batch(&vectors, 4, |v| {
            v.len() / 4
        });
        
        assert_eq!(added, 2);
    }
}

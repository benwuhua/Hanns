//! 并发搜索执行器

/// 并发搜索器（简化版）
pub struct ConcurrentSearcher {
    pub num_threads: usize,
}

impl ConcurrentSearcher {
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads: num_threads.max(1) }
    }
    
    /// 获取线程数
    pub fn threads(&self) -> usize {
        self.num_threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_concurrent_searcher_new() {
        let searcher = ConcurrentSearcher::new(4);
        assert_eq!(searcher.num_threads, 4);
    }
    
    #[test]
    fn test_concurrent_searcher_min() {
        let searcher = ConcurrentSearcher::new(0);
        assert_eq!(searcher.num_threads, 1);
    }
}

//! 统计与监控模块

use std::time::{Duration, Instant};

/// 运行时统计
#[derive(Default)]
pub struct Stats {
    pub search_count: u64,
    pub add_count: u64,
    pub total_search_ns: u64,
    pub total_add_ns: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_search(&mut self, duration: Duration) {
        self.search_count += 1;
        self.total_search_ns += duration.as_nanos() as u64;
    }
    
    pub fn record_add(&mut self, duration: Duration) {
        self.add_count += 1;
        self.total_add_ns += duration.as_nanos() as u64;
    }
    
    pub fn avg_search_us(&self) -> f64 {
        if self.search_count == 0 { 0.0 } else {
            self.total_search_ns as f64 / self.search_count as f64 / 1000.0
        }
    }
    
    pub fn avg_add_us(&self) -> f64 {
        if self.add_count == 0 { 0.0 } else {
            self.total_add_ns as f64 / self.add_count as f64 / 1000.0
        }
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else {
            self.cache_hits as f64 / total as f64
        }
    }
    
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stats: {} searches ({:.2} us avg), {} adds ({:.2} us avg), cache hit rate: {:.1}%",
            self.search_count,
            self.avg_search_us(),
            self.add_count,
            self.avg_add_us(),
            self.cache_hit_rate() * 100.0
        )
    }
}

/// 性能计时器
pub struct Timer {
    start: Instant,
    name: &'static str,
}

impl Timer {
    pub fn new(name: &'static str) -> Self {
        Self { start: Instant::now(), name }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!("{} took {:.2} ms", self.name, self.elapsed_ms());
    }
}

/// 简单计数器
pub struct Counter {
    count: std::sync::atomic::AtomicU64,
}

impl Counter {
    pub fn new() -> Self {
        Self { count: std::sync::atomic::AtomicU64::new(0) }
    }
    
    pub fn inc(&self) {
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get(&self) -> u64 {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_stats() {
        let mut stats = Stats::new();
        
        stats.record_search(Duration::from_micros(100));
        stats.record_search(Duration::from_micros(200));
        stats.record_add(Duration::from_micros(50));
        
        assert_eq!(stats.search_count, 2);
        assert!((stats.avg_search_us() - 150.0).abs() < 0.1);
    }
    
    #[test]
    fn test_timer() {
        let _timer = Timer::new("test");
        thread::sleep(Duration::from_millis(1));
    }
    
    #[test]
    fn test_counter() {
        let counter = Counter::new();
        
        counter.inc();
        counter.inc();
        
        assert_eq!(counter.get(), 2);
    }
}

//! 原子操作工具

/// 原子计数器
pub struct AtomicCounter {
    count: std::sync::atomic::AtomicU64,
}

impl AtomicCounter {
    pub fn new() -> Self {
        Self { count: std::sync::atomic::AtomicU64::new(0) }
    }
    
    pub fn inc(&self) {
        self.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn add(&self, n: u64) {
        self.count.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get(&self) -> u64 {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn reset(&self) {
        self.count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for AtomicCounter {
    fn default() -> Self { Self::new() }
}

/// 原子布尔值
pub struct AtomicBool {
    flag: std::sync::atomic::AtomicBool,
}

impl AtomicBool {
    pub fn new(val: bool) -> Self {
        Self { flag: std::sync::atomic::AtomicBool::new(val) }
    }
    
    pub fn set(&self, val: bool) {
        self.flag.store(val, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn compare_and_set(&self, expected: bool, new: bool) -> bool {
        self.flag.compare_exchange(expected, new, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_counter() {
        let c = AtomicCounter::new();
        c.inc();
        c.inc();
        assert_eq!(c.get(), 2);
    }
    
    #[test]
    fn test_atomic_bool() {
        let b = AtomicBool::new(false);
        assert!(!b.get());
        b.set(true);
        assert!(b.get());
    }
}

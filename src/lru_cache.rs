//! LRU 缓存实现（简化版）
//! 
//! 使用 HashMap + Vec 实现高效的缓存操作

use std::collections::HashMap;

/// 高性能 LRU 缓存
pub struct LruCache<K, V> {
    data: Vec<(K, V)>,
    map: HashMap<K, usize>,
    capacity: usize,
    access_order: Vec<usize>,  // 访问顺序索引
}

impl<K: Eq + std::hash::Hash + Clone, V> LruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::new(),
            map: HashMap::new(),
            capacity: capacity.max(1),
            access_order: Vec::new(),
        }
    }
    
    /// 获取值
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            // 提升到访问顺序末尾
            if let Some(pos) = self.access_order.iter().position(|&x| x == idx) {
                self.access_order.remove(pos);
                self.access_order.push(idx);
            }
            self.data.get(idx).map(|(_, v)| v)
        } else {
            None
        }
    }
    
    /// 放入新值
    pub fn put(&mut self, key: K, value: V) {
        // 如果 key 已存在，更新值
        if let Some(&idx) = self.map.get(&key) {
            self.data[idx] = (key, value);
            // 提升到访问顺序末尾
            if let Some(pos) = self.access_order.iter().position(|&x| x == idx) {
                self.access_order.remove(pos);
                self.access_order.push(idx);
            }
            return;
        }
        
        // 如果容量已满，移除最久未使用的
        if self.data.len() >= self.capacity {
            if let Some(&idx_to_remove) = self.access_order.first() {
                if let Some((k, _)) = self.data.get(idx_to_remove) {
                    self.map.remove(&k);
                }
                // 用新值替换最旧的位置
                let new_idx = idx_to_remove;
                self.data[new_idx] = (key.clone(), value);
                self.access_order.remove(0);
                self.access_order.push(new_idx);
                self.map.insert(key, new_idx);
                return;
            }
        }
        
        // 添加新值
        let idx = self.data.len();
        self.data.push((key, value));
        self.access_order.push(idx);
        if let Some((ref k, _)) = self.data.last() {
            self.map.insert(k.clone(), idx);
        }
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.data.clear();
        self.map.clear();
        self.access_order.clear();
    }
}

/// 简化版 LRU（用于不需要 key 查找的场景）
pub struct SimpleLruCache<V> {
    data: Vec<(u64, V)>,
    capacity: usize,
    next_key: u64,
}

impl<V> SimpleLruCache<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::new(),
            capacity: capacity.max(1),
            next_key: 0,
        }
    }
    
    pub fn get(&mut self, key: u64) -> Option<&V> {
        if let Some(pos) = self.data.iter().position(|(k, _)| *k == key) {
            let (_, v) = self.data.remove(pos);
            self.data.push((key, v));
            self.data.last().map(|(_, v)| v)
        } else {
            None
        }
    }
    
    pub fn put(&mut self, value: V) -> u64 {
        if self.data.len() >= self.capacity {
            self.data.remove(0);
        }
        
        let key = self.next_key;
        self.next_key += 1;
        self.data.push((key, value));
        key
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// 缓存预热策略
/// 
/// 用于在搜索前将热点数据加载到内存缓存，减少冷启动延迟
pub struct CachePrewarm {
    /// 预热数据大小阈值 (bytes)
    pub warm_threshold_bytes: usize,
    /// 预热数据条数阈值
    pub warm_threshold_count: usize,
}

impl Default for CachePrewarm {
    fn default() -> Self {
        Self {
            warm_threshold_bytes: 64 * 1024 * 1024,  // 64MB
            warm_threshold_count: 1000,
        }
    }
}

impl CachePrewarm {
    /// 判断是否需要预热
    pub fn needs_warm(&self, data_size_bytes: usize, count: usize) -> bool {
        data_size_bytes >= self.warm_threshold_bytes || count >= self.warm_threshold_count
    }
    
    /// 预热回调 - 实际预热由调用方实现
    pub fn warmup<F>(&self, data: &[f32], dim: usize, mut warm_fn: F)
    where
        F: FnMut(&[f32], usize),
    {
        let count = data.len() / dim;
        if self.needs_warm(data.len() * std::mem::size_of::<f32>(), count) {
            // 将数据切分成多个批次进行预热
            let batch_size = (count / 4).max(1);
            for i in (0..count).step_by(batch_size) {
                let end = (i + batch_size).min(count);
                let batch = &data[i * dim..end * dim];
                warm_fn(batch, end - i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_new() {
        let cache: LruCache<i32, i32> = LruCache::new(3);
        assert!(cache.is_empty());
    }
    
    #[test]
    fn test_lru_put_get() {
        let mut cache = LruCache::new(3);
        
        cache.put(1, 10);
        cache.put(2, 20);
        
        assert_eq!(cache.get(&1), Some(&10));
        assert_eq!(cache.get(&2), Some(&20));
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::new(3);
        
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        cache.put(4, 40);
        
        assert_eq!(cache.get(&1), None);  // Evicted
        assert_eq!(cache.get(&4), Some(&40));
    }
    
    #[test]
    fn test_lru_update() {
        let mut cache = LruCache::new(3);
        
        cache.put(1, 10);
        cache.put(1, 100);  // Update existing key
        
        assert_eq!(cache.get(&1), Some(&100));
        assert_eq!(cache.len(), 1);
    }
    
    #[test]
    fn test_simple_lru() {
        let mut cache: SimpleLruCache<i32> = SimpleLruCache::new(3);
        
        let k1 = cache.put(10);
        let k2 = cache.put(20);
        
        assert_eq!(cache.get(k1), Some(&10));
        assert_eq!(cache.get(k2), Some(&20));
    }
    
    #[test]
    fn test_cache_prewarm_defaults() {
        let warm = CachePrewarm::default();
        assert!(!warm.needs_warm(1000, 10));  // Small data
        assert!(warm.needs_warm(100 * 1024 * 1024, 100));  // Large data
    }
    
    #[test]
    fn test_cache_prewarm_warmup() {
        // Use custom threshold to test warmup
        let mut warm = CachePrewarm::default();
        warm.warm_threshold_bytes = 1000;  // Low threshold for test
        
        let data: Vec<f32> = (0..4000).map(|i| i as f32).collect();  // 100 vectors * 128 dim
        
        let mut call_count = 0;
        warm.warmup(&data, 128, |_batch, count| {
            call_count += 1;
            // 验证批次大小合理
            assert!(count > 0);
        });
        
        // 应该触发预热回调
        assert!(call_count > 0);
    }
}

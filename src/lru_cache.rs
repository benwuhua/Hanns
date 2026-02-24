//! LRU 缓存实现

/// LRU 缓存
pub struct LruCache<V> {
    data: Vec<(u64, V)>,
    capacity: usize,
    next_key: u64,
}

impl<V> LruCache<V> {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lru_new() {
        let cache: LruCache<i32> = LruCache::new(3);
        assert!(cache.is_empty());
    }
    
    #[test]
    fn test_lru_put_get() {
        let mut cache = LruCache::new(3);
        
        let k1 = cache.put(10);
        let k2 = cache.put(20);
        
        assert_eq!(cache.get(k1), Some(&10));
        assert_eq!(cache.get(k2), Some(&20));
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = LruCache::new(3);
        
        let k1 = cache.put(10);
        cache.put(20);
        cache.put(30);
        cache.put(40);
        
        assert_eq!(cache.get(k1), None);
    }
}

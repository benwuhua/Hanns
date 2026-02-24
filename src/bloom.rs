//! Bloom Filter - 快速存在性判断

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Bloom Filter 实现
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
    inserted: usize,
}

impl BloomFilter {
    /// 创建 Bloom Filter
    /// - num_bits: 位数组大小
    /// - num_hashes: 哈希函数数量
    pub fn new(num_bits: usize, num_hashes: usize) -> Self {
        let num_u64s = (num_bits + 63) / 64;
        Self {
            bits: vec![0; num_u64s],
            num_bits,
            num_hashes,
            inserted: 0,
        }
    }
    
    /// 根据预期插入数量计算最优参数
    pub fn with_expected_items(expected: usize) -> Self {
        // 最优哈希数: k = (m/n) * ln(2)
        // 最优位数: m = -n * ln(p) / (ln(2)^2), p = 0.01
        let num_bits = ((expected as f64) * 6.64) as usize; // 1% 误判率
        let num_hashes = ((num_bits as f64 / expected as f64) * 0.693) as usize;
        
        Self::new(num_bits.max(64), num_hashes.max(1))
    }
    
    /// 插入元素
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let idx = self.hash(item, i);
            self.bits[idx / 64] |= 1 << (idx % 64);
        }
        self.inserted += 1;
    }
    
    /// 检查元素是否存在（可能误判）
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash(item, i);
            if self.bits[idx / 64] & (1 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }
    
    /// 误判率估计
    pub fn false_positive_rate(&self) -> f64 {
        let filled = self.bits.iter().fold(0u64, |acc, &x| acc + x.count_ones() as u64);
        let p = filled as f64 / (self.bits.len() * 64) as f64;
        p.powi(self.num_hashes as i32)
    }
    
    fn hash<T: Hash>(&self, item: &T, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        ((hasher.finish() as usize).wrapping_mul(seed).wrapping_add(seed)) % self.num_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bloom_new() {
        let bf = BloomFilter::new(1000, 3);
        assert_eq!(bf.num_bits, 1000);
        assert_eq!(bf.num_hashes, 3);
    }
    
    #[test]
    fn test_bloom_with_expected() {
        let bf = BloomFilter::with_expected_items(1000);
        assert!(bf.num_bits > 6000);
    }
    
    #[test]
    fn test_bloom_insert_contains() {
        let mut bf = BloomFilter::new(1000, 3);
        
        bf.insert(&42);
        bf.insert(&"hello");
        
        assert!(bf.contains(&42));
        assert!(bf.contains(&"hello"));
        assert!(!bf.contains(&999));
    }
    
    #[test]
    fn test_false_positive() {
        let mut bf = BloomFilter::new(1000, 3);
        
        for i in 0..100 {
            bf.insert(&i);
        }
        
        // 检查误判率
        let rate = bf.false_positive_rate();
        println!("False positive rate: {}", rate);
    }
}

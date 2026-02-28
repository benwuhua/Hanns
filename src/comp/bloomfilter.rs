//! Bloom Filter Implementation
//! 
//! A space-efficient probabilistic data structure for membership testing.
//! Based on C++ knowhere implementation: knowhere/comp/bloomfilter.h

use std::io::{Read, Write};
use std::hash::Hasher;

/// Bloom Filter for fast membership testing
/// 
/// False positives are possible, but false negatives are not.
#[derive(Debug, Clone)]
pub struct BloomFilter<T> {
    /// Expected number of elements
    n: usize,
    /// Desired false positive probability
    p: f64,
    /// Number of bits in the filter
    m: usize,
    /// Number of hash functions
    k: usize,
    /// Bit array (stored as bytes)
    bits: Vec<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> BloomFilter<T> {
    /// Create a new Bloom Filter
    /// 
    /// # Arguments
    /// * `expected_elements` - Expected number of elements to insert
    /// * `false_positive_prob` - Desired false positive probability (e.g., 0.01 for 1%)
    pub fn new(expected_elements: usize, false_positive_prob: f64) -> Self {
        let n = expected_elements;
        let p = false_positive_prob;
        
        // Calculate optimal m (number of bits) and k (number of hash functions)
        // m = -(n * ln(p)) / (ln(2)^2)
        // k = (m/n) * ln(2)
        let m = ((-(n as f64) * p.ln()) / (2.0_f64.ln().powi(2))).ceil() as usize;
        let k = (((m as f64 / n as f64) * 2.0_f64.ln()).ceil() as usize).max(1);
        let m = m.max(1);
        
        // Allocate bits (rounded up to nearest byte)
        let bytes = (m + 7) / 8;
        let bits = vec![0u8; bytes];
        
        Self {
            n,
            p,
            m,
            k,
            bits,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the number of bits in the filter
    pub fn num_bits(&self) -> usize {
        self.m
    }

    /// Get the number of hash functions
    pub fn num_hash_functions(&self) -> usize {
        self.k
    }

    /// Get the expected number of elements
    pub fn expected_elements(&self) -> usize {
        self.n
    }

    /// Get the false positive probability
    pub fn false_positive_probability(&self) -> f64 {
        self.p
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.bits.len()
    }

    /// Hash function using FNV-1a algorithm with seed
    fn hash(&self, data: &[u8], seed: usize) -> usize {
        let mut hasher = FnvHasher::with_seed(seed);
        hasher.write(data);
        let hash = hasher.finish() as usize;
        hash % self.m
    }

    /// Add an element to the filter
    pub fn add(&mut self, element: &T) 
    where
        T: serde::Serialize,
    {
        let data = bincode::serialize(element).unwrap_or_default();
        for i in 0..self.k {
            let pos = self.hash(&data, i);
            self.bits[pos / 8] |= 1 << (pos % 8);
        }
    }

    /// Check if an element might be in the filter
    /// 
    /// Returns true if the element might be in the set (possible false positive),
    /// false if the element is definitely not in the set.
    pub fn contains(&self, element: &T) -> bool 
    where
        T: serde::Serialize,
    {
        let data = bincode::serialize(element).unwrap_or_default();
        for i in 0..self.k {
            let pos = self.hash(&data, i);
            if self.bits[pos / 8] & (1 << (pos % 8)) == 0 {
                return false;
            }
        }
        true
    }

    /// Add an element from raw bytes (for FFI compatibility)
    pub fn add_bytes(&mut self, data: &[u8]) {
        for i in 0..self.k {
            let pos = self.hash(data, i);
            self.bits[pos / 8] |= 1 << (pos % 8);
        }
    }

    /// Check if raw bytes might be in the filter
    pub fn contains_bytes(&self, data: &[u8]) -> bool {
        for i in 0..self.k {
            let pos = self.hash(data, i);
            if self.bits[pos / 8] & (1 << (pos % 8)) == 0 {
                return false;
            }
        }
        true
    }

    /// Save the Bloom Filter to a writer
    pub fn save<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.m.to_le_bytes())?;
        writer.write_all(&self.k.to_le_bytes())?;
        writer.write_all(&self.n.to_le_bytes())?;
        writer.write_all(&self.p.to_le_bytes())?;
        writer.write_all(&self.bits)?;
        Ok(())
    }

    /// Load the Bloom Filter from a reader
    pub fn load<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        
        reader.read_exact(&mut buf)?;
        let m = usize::from_le_bytes(buf);
        
        reader.read_exact(&mut buf)?;
        let k = usize::from_le_bytes(buf);
        
        reader.read_exact(&mut buf)?;
        let n = usize::from_le_bytes(buf);
        
        reader.read_exact(&mut buf)?;
        let p = f64::from_le_bytes(buf);
        
        let bytes = (m + 7) / 8;
        let mut bits = vec![0u8; bytes];
        reader.read_exact(&mut bits)?;
        
        Ok(Self {
            n,
            p,
            m,
            k,
            bits,
            _marker: std::marker::PhantomData,
        })
    }

    /// Clear all bits in the filter
    pub fn clear(&mut self) {
        for byte in &mut self.bits {
            *byte = 0;
        }
    }

    /// Get the estimated current number of elements
    pub fn estimated_cardinality(&self) -> usize {
        let mut ones = 0usize;
        for byte in &self.bits {
            ones += byte.count_ones() as usize;
        }
        
        let m = self.m as f64;
        let k = self.k as f64;
        let x = ones as f64;
        
        // n = -(m/k) * ln(1 - x/m)
        if x >= m {
            self.n
        } else {
            (-(m / k) * (1.0 - x / m).ln()) as usize
        }
    }
}

/// FNV-1a hash function with seed support
struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    fn with_seed(seed: usize) -> Self {
        // FNV offset basis with seed mixing
        let offset_basis = 0xcbf29ce484222325u64;
        Self {
            state: offset_basis ^ (seed as u64),
        }
    }
}

impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        const FNV_PRIME: u64 = 0x100000001b3;
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut bf = BloomFilter::<u64>::new(1000, 0.01);
        
        // Add some elements
        for i in 0..100u64 {
            bf.add(&i);
        }
        
        // Check that all added elements are found
        for i in 0..100u64 {
            assert!(bf.contains(&i), "Element {} should be found", i);
        }
        
        // Check that non-added elements are mostly not found (some false positives expected)
        let mut false_positives = 0;
        for i in 1000..1100u64 {
            if bf.contains(&i) {
                false_positives += 1;
            }
        }
        
        // False positive rate should be around 1%
        let fp_rate = false_positives as f64 / 100.0;
        assert!(fp_rate < 0.05, "False positive rate {} too high (expected ~1%)", fp_rate);
    }

    #[test]
    fn test_bloom_filter_bytes() {
        let mut bf = BloomFilter::<Vec<u8>>::new(1000, 0.01);
        
        let data1 = vec![1u8, 2, 3, 4];
        let data2 = vec![5u8, 6, 7, 8];
        
        bf.add(&data1);
        
        assert!(bf.contains(&data1));
        assert!(!bf.contains(&data2));
    }

    #[test]
    fn test_bloom_filter_save_load() {
        let mut bf = BloomFilter::<u64>::new(1000, 0.01);
        
        for i in 0..100u64 {
            bf.add(&i);
        }
        
        // Save to buffer
        let mut buffer = Vec::new();
        bf.save(&mut buffer).unwrap();
        
        // Load from buffer
        let mut cursor = std::io::Cursor::new(buffer);
        let bf2 = BloomFilter::<u64>::load(&mut cursor).unwrap();
        
        // Verify loaded filter works
        for i in 0..100u64 {
            assert!(bf2.contains(&i), "Loaded filter should contain {}", i);
        }
    }

    #[test]
    fn test_bloom_filter_parameters() {
        let bf = BloomFilter::<u64>::new(10000, 0.001);
        
        assert_eq!(bf.expected_elements(), 10000);
        assert!((bf.false_positive_probability() - 0.001).abs() < 0.0001);
        assert!(bf.memory_usage() > 0);
        assert!(bf.num_bits() > 0);
        assert!(bf.num_hash_functions() > 0);
    }

    #[test]
    fn test_bloom_filter_clear() {
        let mut bf = BloomFilter::<u64>::new(1000, 0.01);
        
        for i in 0..100u64 {
            bf.add(&i);
        }
        
        bf.clear();
        
        // After clearing, nothing should be found
        for i in 0..100u64 {
            assert!(!bf.contains(&i), "Cleared filter should not contain {}", i);
        }
    }
}

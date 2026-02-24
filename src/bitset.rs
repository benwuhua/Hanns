//! BitsetView - 高性能位图实现
//! 
//! 用于 Milvus 的软删除机制，支持高效的位运算。

use std::ops::{BitAnd, BitOr, BitXor};

/// 位图迭代器
pub struct BitsetIter<'a> {
    bitset: &'a BitsetView,
    pos: usize,
}

impl<'a> Iterator for BitsetIter<'a> {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.bitset.len() {
            let pos = self.pos;
            self.pos += 1;
            if self.bitset.get(pos) {
                return Some(pos);
            }
        }
        None
    }
}

/// 高性能位图视图
/// 
/// # 设计目标
/// - O(1) 访问时间
/// - 内存高效（按需分配）
/// - 与 C++ BitsetView 对齐
#[derive(Clone)]
pub struct BitsetView {
    /// 内部存储：每个元素是一个 u64（64位）
    data: Vec<u64>,
    /// 位图长度（位数，非字节）
    len: usize,
}

impl BitsetView {
    /// 创建一个新的空位图
    #[inline]
    pub fn new(len: usize) -> Self {
        let words = (len + 63) / 64;
        Self {
            data: vec![0u64; words],
            len,
        }
    }
    
    /// 从现有数据创建位图
    #[inline]
    pub fn from_vec(data: Vec<u64>, len: usize) -> Self {
        Self { data, len }
    }
    
    /// 获取位图长度
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// 获取指定位置的位值
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        self.data[word_idx] & (1u64 << bit_idx) != 0
    }
    
    /// 设置指定位置的位值
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.len {
            return;
        }
        
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        let mask = 1u64 << bit_idx;
        
        if value {
            self.data[word_idx] |= mask;
        } else {
            self.data[word_idx] &= !mask;
        }
    }
    
    /// 设置指定位置的位为 1
    #[inline]
    pub fn set_bit(&mut self, index: usize) {
        self.set(index, true);
    }
    
    /// 清除指定位置的位
    #[inline]
    pub fn clear_bit(&mut self, index: usize) {
        self.set(index, false);
    }
    
    /// 翻转指定位置的位
    #[inline]
    pub fn flip(&mut self, index: usize) {
        if index >= self.len {
            return;
        }
        
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        let mask = 1u64 << bit_idx;
        
        self.data[word_idx] ^= mask;
    }
    
    /// 检查是否所有位都为 1
    #[inline]
    pub fn all(&self) -> bool {
        let full_words = self.len >> 6;
        let remaining = self.len & 63;
        
        for i in 0..full_words {
            if self.data[i] != u64::MAX {
                return false;
            }
        }
        
        if remaining > 0 {
            let mask = (1u64 << remaining) - 1;
            if self.data[full_words] & mask != mask {
                return false;
            }
        }
        
        true
    }
    
    /// 检查是否所有位都为 0
    #[inline]
    pub fn none(&self) -> bool {
        for word in &self.data {
            if *word != 0 {
                return false;
            }
        }
        true
    }
    
    /// 检查是否有任何位为 1
    #[inline]
    pub fn any(&self) -> bool {
        !self.none()
    }
    
    /// 统计为 1 的位数
    #[inline]
    pub fn count(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }
    
    /// 获取所有为 1 的位的索引迭代器
    #[inline]
    pub fn iter(&self) -> BitsetIter {
        BitsetIter {
            bitset: self,
            pos: 0,
        }
    }
    
    /// 查找第一个为 1 的位
    #[inline]
    pub fn find_first(&self) -> Option<usize> {
        for (i, &word) in self.data.iter().enumerate() {
            if word != 0 {
                return Some(i * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }
    
    /// 查找下一个为 1 的位（从指定位置开始）
    #[inline]
    pub fn find_next(&self, from: usize) -> Option<usize> {
        if from >= self.len {
            return None;
        }
        
        let start_word = from >> 6;
        let start_bit = from & 63;
        
        // 检查当前字
        let word = self.data[start_word] & (!0u64 << start_bit);
        if word != 0 {
            return Some(start_word * 64 + word.trailing_zeros() as usize);
        }
        
        // 检查后续字
        for i in (start_word + 1)..self.data.len() {
            if self.data[i] != 0 {
                return Some(i * 64 + self.data[i].trailing_zeros() as usize);
            }
        }
        
        None
    }
    
    /// 清除所有位
    #[inline]
    pub fn clear(&mut self) {
        for word in &mut self.data {
            *word = 0;
        }
    }
    
    /// 设置所有位为 1
    #[inline]
    pub fn set_all(&mut self) {
        let full_words = self.len >> 6;
        let remaining = self.len & 63;
        
        for i in 0..full_words {
            self.data[i] = u64::MAX;
        }
        
        if remaining > 0 {
            self.data[full_words] = (1u64 << remaining) - 1;
        }
    }
    
    /// 获取底层数据（用于 FFI）
    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }
    
    /// 获取可变底层数据（用于 FFI）
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.data
    }
}

impl Default for BitsetView {
    fn default() -> Self {
        Self::new(0)
    }
}

impl std::fmt::Debug for BitsetView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitsetView(len={}, ", self.len)?;
        
        // 显示前 64 位作为示例
        let preview: String = (0..self.len.min(64))
            .map(|i| if self.get(i) { '1' } else { '0' })
            .collect();
        
        write!(f, "preview={})", preview)
    }
}

/// 位运算：AND
impl BitAnd for &BitsetView {
    type Output = BitsetView;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        let len = self.len.max(rhs.len());
        let mut result = BitsetView::new(len);
        
        for i in 0..result.data.len() {
            let s = self.data.get(i).copied().unwrap_or(0);
            let r = rhs.data.get(i).copied().unwrap_or(0);
            result.data[i] = s & r;
        }
        
        result
    }
}

/// 位运算：OR
impl BitOr for &BitsetView {
    type Output = BitsetView;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        let len = self.len.max(rhs.len());
        let mut result = BitsetView::new(len);
        
        for i in 0..result.data.len() {
            let s = self.data.get(i).copied().unwrap_or(0);
            let r = rhs.data.get(i).copied().unwrap_or(0);
            result.data[i] = s | r;
        }
        
        result
    }
}

/// 位运算：XOR
impl BitXor for &BitsetView {
    type Output = BitsetView;
    
    fn bitxor(self, rhs: Self) -> Self::Output {
        let len = self.len.max(rhs.len());
        let mut result = BitsetView::new(len);
        
        for i in 0..result.data.len() {
            let s = self.data.get(i).copied().unwrap_or(0);
            let r = rhs.data.get(i).copied().unwrap_or(0);
            result.data[i] = s ^ r;
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic() {
        let mut bits = BitsetView::new(100);
        
        assert_eq!(bits.len(), 100);
        assert!(!bits.is_empty());
        
        // 默认应该全是 0
        assert!(bits.none());
        assert!(!bits.all());
        
        // 设置几位
        bits.set(0, true);
        bits.set(10, true);
        bits.set(63, true);
        bits.set(64, true);
        
        assert!(bits.get(0));
        assert!(bits.get(10));
        assert!(bits.get(63));
        assert!(bits.get(64));
        assert!(!bits.get(1));
        assert!(!bits.get(65));
        
        // 统计
        assert_eq!(bits.count(), 4);
    }
    
    #[test]
    fn test_iter() {
        let mut bits = BitsetView::new(200);
        bits.set(5, true);
        bits.set(10, true);
        bits.set(15, true);
        
        let collected: Vec<usize> = bits.iter().collect();
        assert_eq!(collected, vec![5, 10, 15]);
    }
    
    #[test]
    fn test_bitwise() {
        let mut a = BitsetView::new(64);
        let mut b = BitsetView::new(64);
        
        a.set(0, true);
        a.set(1, true);
        
        b.set(1, true);
        b.set(2, true);
        
        let and = &a & &b;
        assert!(!and.get(0));
        assert!(and.get(1));
        assert!(!and.get(2));
        
        let or = &a | &b;
        assert!(or.get(0));
        assert!(or.get(1));
        assert!(or.get(2));
        
        let xor = &a ^ &b;
        assert!(xor.get(0));
        assert!(!xor.get(1));
        assert!(xor.get(2));
    }
    
    #[test]
    fn test_clear_set_all() {
        let mut bits = BitsetView::new(100);
        
        bits.set(50, true);
        assert!(bits.get(50));
        
        bits.clear();
        assert!(bits.none());
        
        bits.set_all();
        assert!(bits.all());
    }
}

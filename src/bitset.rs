//! BitsetView - 高性能位图实现
//! 
//! 用于 Milvus 的软删除机制，支持高效的位运算。
//! 与 C++ knowhere 的 BitsetView 对齐，支持 out_ids 用于压缩 bitset。

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
/// - 支持 out_ids 用于压缩 bitset（ID 映射）
#[derive(Clone)]
pub struct BitsetView {
    /// 内部存储：每个元素是一个 u64（64 位）
    data: Vec<u64>,
    /// 位图长度（位数，非字节）
    len: usize,
    /// 被过滤的位数（为 1 的位数）
    num_filtered_out_bits: usize,
    /// ID 偏移量（用于多 chunk 场景）
    id_offset: usize,
    /// 可选的 ID 映射（用于压缩 bitset）
    /// 如果 Some，则 out_ids[i] 表示第 i 个内部 ID 对应的外部 ID
    out_ids: Option<Vec<u32>>,
    /// 内部 ID 数量（当使用 out_ids 时）
    num_internal_ids: usize,
    /// 被过滤的内部 ID 数量（当使用 out_ids 时）
    num_filtered_out_ids: usize,
}

impl BitsetView {
    /// 创建一个新的空位图
    #[inline]
    pub fn new(len: usize) -> Self {
        let words = (len + 63) / 64;
        Self {
            data: vec![0u64; words],
            len,
            num_filtered_out_bits: 0,
            id_offset: 0,
            out_ids: None,
            num_internal_ids: 0,
            num_filtered_out_ids: 0,
        }
    }
    
    /// 从现有数据创建位图
    #[inline]
    pub fn from_vec(data: Vec<u64>, len: usize) -> Self {
        Self {
            data,
            len,
            num_filtered_out_bits: 0,
            id_offset: 0,
            out_ids: None,
            num_internal_ids: 0,
            num_filtered_out_ids: 0,
        }
    }
    
    /// 获取位图长度
    /// 如果有 out_ids，返回内部 ID 数量；否则返回位数
    #[inline]
    pub fn len(&self) -> usize {
        if self.out_ids.is_some() {
            self.num_internal_ids
        } else {
            self.len
        }
    }
    
    /// 获取原始位图长度（位数）
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.len
    }
    
    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 获取指定位置的位值
    /// 如果有 out_ids，会先进行 ID 映射
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        // 应用 ID 偏移
        let out_id = if let Some(ref out_ids) = self.out_ids {
            if index >= out_ids.len() {
                return false;
            }
            out_ids[index] as usize
        } else {
            index + self.id_offset
        };
        
        // 检查是否超出范围
        if out_id >= self.len {
            return true; // 超出范围的被视为已过滤
        }
        
        let word_idx = out_id >> 6;
        let bit_idx = out_id & 63;
        self.data[word_idx] & (1u64 << bit_idx) != 0
    }
    
    /// 测试指定索引是否被过滤（与 C++ test() 对齐）
    /// 如果 test 返回 true，则该索引应该在搜索时被跳过
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        self.get(index)
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
        
        let was_set = self.data[word_idx] & mask != 0;
        
        if value {
            self.data[word_idx] |= mask;
            if !was_set {
                self.num_filtered_out_bits += 1;
            }
        } else {
            self.data[word_idx] &= !mask;
            if was_set {
                self.num_filtered_out_bits -= 1;
            }
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
        
        let was_set = self.data[word_idx] & mask != 0;
        self.data[word_idx] ^= mask;
        
        if was_set {
            self.num_filtered_out_bits -= 1;
        } else {
            self.num_filtered_out_bits += 1;
        }
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
    
    /// 统计为 1 的位数（被过滤的数量）
    /// 如果有 out_ids，返回被过滤的内部 ID 数量
    #[inline]
    pub fn count(&self) -> usize {
        if self.out_ids.is_some() {
            self.num_filtered_out_ids
        } else {
            self.num_filtered_out_bits
        }
    }
    
    /// 获取被过滤的位数（不使用 out_ids 时的原始计数）
    #[inline]
    pub fn num_filtered_out_bits(&self) -> usize {
        self.num_filtered_out_bits
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
        self.num_filtered_out_bits = 0;
        self.num_filtered_out_ids = 0;
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
        
        self.num_filtered_out_bits = self.len;
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
    
    // ========== out_ids 相关方法（与 C++ knowhere 对齐） ==========
    
    /// 检查是否有 out_ids（ID 映射）
    #[inline]
    pub fn has_out_ids(&self) -> bool {
        self.out_ids.is_some()
    }
    
    /// 设置 out_ids（ID 映射）
    /// 
    /// # Arguments
    /// * `out_ids` - ID 映射数组，out_ids[i] 表示第 i 个内部 ID 对应的外部 ID
    /// * `num_filtered_out_ids` - 可选的被过滤的内部 ID 数量，如果为 None 则自动计算
    #[inline]
    pub fn set_out_ids(&mut self, out_ids: Vec<u32>, num_filtered_out_ids: Option<usize>) {
        let num_internal_ids = out_ids.len();
        
        // 如果没有提供 num_filtered_out_ids，则自动计算
        let filtered_count = num_filtered_out_ids.unwrap_or_else(|| {
            // 计算被过滤的内部 ID 数量
            // 注意：这里不能直接调用 self.test()，因为 out_ids 还没设置
            // 需要手动进行 ID 映射并检查位
            let mut count = 0;
            for i in 0..num_internal_ids {
                let external_id = out_ids[i] as usize;
                // 检查外部 ID 是否被过滤（位为 1）
                if external_id >= self.len {
                    // 超出范围被视为已过滤
                    count += 1;
                } else {
                    let word_idx = external_id >> 6;
                    let bit_idx = external_id & 63;
                    if self.data[word_idx] & (1u64 << bit_idx) != 0 {
                        count += 1;
                    }
                }
            }
            count
        });
        
        self.out_ids = Some(out_ids);
        self.num_internal_ids = num_internal_ids;
        self.num_filtered_out_ids = filtered_count;
    }
    
    /// 获取 out_ids 数据
    #[inline]
    pub fn out_ids_data(&self) -> Option<&[u32]> {
        self.out_ids.as_deref()
    }
    
    /// 获取 out_ids 的可变引用
    #[inline]
    pub fn out_ids_data_mut(&mut self) -> Option<&mut [u32]> {
        self.out_ids.as_deref_mut()
    }
    
    /// 获取内部 ID 数量
    #[inline]
    pub fn num_internal_ids(&self) -> usize {
        self.num_internal_ids
    }
    
    /// 设置 ID 偏移量
    #[inline]
    pub fn set_id_offset(&mut self, offset: usize) {
        self.id_offset = offset;
    }
    
    /// 获取 ID 偏移量
    #[inline]
    pub fn id_offset(&self) -> usize {
        self.id_offset
    }
    
    /// 获取过滤比例
    #[inline]
    pub fn filter_ratio(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.count() as f32 / self.len() as f32
        }
    }
    
    /// 获取第一个有效的索引（未被过滤的）
    pub fn get_first_valid_index(&self) -> usize {
        for i in 0..self.len() {
            if !self.test(i) {
                return i;
            }
        }
        self.len()
    }
}

impl Default for BitsetView {
    fn default() -> Self {
        Self::new(0)
    }
}

impl std::fmt::Debug for BitsetView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitsetView(len={}, ", self.len())?;
        
        // 显示前 64 位作为示例
        let preview: String = (0..self.len().min(64))
            .map(|i| if self.get(i) { '1' } else { '0' })
            .collect();
        
        if self.out_ids.is_some() {
            write!(f, "with_out_ids, preview={})", preview)
        } else {
            write!(f, "preview={})", preview)
        }
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
    
    #[test]
    fn test_out_ids_basic() {
        let mut bits = BitsetView::new(100);
        
        // 设置一些位
        bits.set(5, true);
        bits.set(10, true);
        bits.set(50, true);
        
        // 初始没有 out_ids
        assert!(!bits.has_out_ids());
        assert_eq!(bits.len(), 100);
        assert_eq!(bits.count(), 3);
        
        // 设置 out_ids：压缩映射
        // 内部 ID 0,1,2 映射到外部 ID 5,10,50
        let out_ids = vec![5u32, 10, 50];
        bits.set_out_ids(out_ids, None);
        
        // 现在有 out_ids
        assert!(bits.has_out_ids());
        assert_eq!(bits.len(), 3); // 内部 ID 数量
        assert_eq!(bits.num_bits(), 100); // 原始位数
        
        // 测试映射
        assert!(bits.get(0)); // 内部 0 -> 外部 5 (已设置)
        assert!(bits.get(1)); // 内部 1 -> 外部 10 (已设置)
        assert!(bits.get(2)); // 内部 2 -> 外部 50 (已设置)
        
        // count 应该返回被过滤的内部 ID 数量
        assert_eq!(bits.count(), 3);
    }
    
    #[test]
    fn test_out_ids_with_explicit_count() {
        let mut bits = BitsetView::new(100);
        
        // 设置 out_ids，并显式指定被过滤的数量
        let out_ids = vec![0u32, 1, 2, 3, 4];
        bits.set_out_ids(out_ids, Some(2));
        
        assert!(bits.has_out_ids());
        assert_eq!(bits.len(), 5);
        assert_eq!(bits.count(), 2); // 显式指定的值
    }
    
    #[test]
    fn test_out_ids_data() {
        let mut bits = BitsetView::new(100);
        
        let out_ids = vec![10u32, 20, 30];
        bits.set_out_ids(out_ids.clone(), None);
        
        assert_eq!(bits.out_ids_data(), Some(&[10u32, 20, 30][..]));
        assert_eq!(bits.num_internal_ids(), 3);
    }
    
    #[test]
    fn test_id_offset() {
        let mut bits = BitsetView::new(100);
        bits.set(50, true);
        
        assert_eq!(bits.id_offset(), 0);
        
        bits.set_id_offset(10);
        assert_eq!(bits.id_offset(), 10);
        
        // 带偏移的访问
        // index 40 + offset 10 = 50，应该返回 true
        assert!(bits.get(40));
    }
    
    #[test]
    fn test_filter_ratio() {
        let mut bits = BitsetView::new(100);
        assert_eq!(bits.filter_ratio(), 0.0);
        
        bits.set_all();
        assert_eq!(bits.filter_ratio(), 1.0);
        
        bits.clear();
        bits.set(0, true);
        bits.set(1, true);
        assert!((bits.filter_ratio() - 0.02).abs() < 0.001);
    }
    
    #[test]
    fn test_get_first_valid_index() {
        let mut bits = BitsetView::new(100);
        assert_eq!(bits.get_first_valid_index(), 0);
        
        bits.set(0, true);
        bits.set(1, true);
        bits.set(2, true);
        assert_eq!(bits.get_first_valid_index(), 3);
        
        bits.set_all();
        assert_eq!(bits.get_first_valid_index(), 100);
    }
}

//! Sparse WAND Index - 稀疏 WAND (Weak AND) 索引
//! 
//! 基于 SparseInvertedIndex 的薄封装，使用 WAND 算法作为默认搜索策略
//! WAND (Document-At-A-Time Weak AND) 是一种高效的稀疏向量搜索算法

use crate::faiss::sparse_inverted::{
    SparseInvertedIndex, 
    SparseInvertedSearcher, 
    SparseMetricType, 
    InvertedIndexAlgo,
    SparseVector,
};

/// 稀疏 WAND 索引
pub struct SparseWandIndex {
    inner: SparseInvertedIndex,
}

impl SparseWandIndex {
    /// 创建新的 WAND 索引
    pub fn new(metric_type: SparseMetricType) -> Self {
        Self {
            inner: SparseInvertedIndex::new(metric_type),
        }
    }
    
    /// 添加单个向量
    pub fn add(&mut self, vector: &SparseVector, doc_id: i64) -> Result<(), String> {
        self.inner.add(vector, doc_id)
    }
    
    /// 使用 WAND 算法搜索
    pub fn search(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        let searcher = SparseInvertedSearcher::new(&self.inner, InvertedIndexAlgo::DaatWand);
        searcher.search(query, k, bitset)
    }
    
    /// 按 ID 获取向量
    pub fn get_vector_by_id(&self, doc_id: i64) -> Option<SparseVector> {
        self.inner.get_vector_by_id(doc_id)
    }
    
    /// 获取行数
    pub fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }
    
    /// 获取内存大小 (字节)
    pub fn size(&self) -> usize {
        self.inner.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_wand_new() {
        let index = SparseWandIndex::new(SparseMetricType::Ip);
        assert_eq!(index.n_rows(), 0);
    }
    
    #[test]
    fn test_sparse_wand_add_search() {
        let mut index = SparseWandIndex::new(SparseMetricType::Ip);
        
        let v1 = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]);
        
        index.add(&v1, 0).unwrap();
        index.add(&v2, 1).unwrap();
        
        let query = SparseVector::from_pairs(&[(0, 1.0)]);
        let results = index.search(&query, 2, None);
        
        assert!(!results.is_empty());
    }
}

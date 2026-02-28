//! Sparse WAND Index CC (Concurrent Version) - 并发稀疏 WAND 索引
//! 
//! 使用 WAND 算法的并发稀疏向量索引
//! 使用 RwLock 提供线程安全的并发访问

use std::sync::RwLock;
use crate::faiss::sparse_inverted::{
    SparseInvertedIndex, 
    SparseInvertedSearcher, 
    SparseMetricType, 
    InvertedIndexAlgo,
    SparseVector,
};

/// WAND 索引配置 (并发版本)
#[derive(Debug, Clone)]
pub struct SparseWandConfigCC {
    pub metric_type: SparseMetricType,
    pub ssize: usize,
}

impl Default for SparseWandConfigCC {
    fn default() -> Self {
        Self {
            metric_type: SparseMetricType::Ip,
            ssize: 1000,
        }
    }
}

impl SparseWandConfigCC {
    pub fn new(metric_type: SparseMetricType) -> Self {
        Self {
            metric_type,
            ..Default::default()
        }
    }
    
    pub fn with_ssize(mut self, ssize: usize) -> Self {
        self.ssize = ssize;
        self
    }
}

/// 并发稀疏 WAND 索引
pub struct SparseWandIndexCC {
    inner: RwLock<SparseInvertedIndex>,
}

impl SparseWandIndexCC {
    /// 创建新的并发 WAND 索引
    pub fn new(metric_type: SparseMetricType, ssize: usize) -> Self {
        let _ = ssize;  // ssize is kept for API compatibility but not used internally
        Self {
            inner: RwLock::new(SparseInvertedIndex::new(metric_type)),
        }
    }
    
    /// 添加单个向量 (线程安全)
    pub fn add(&self, vector: &SparseVector, doc_id: i64) -> Result<(), String> {
        let mut inner = self.inner.write().map_err(|e| e.to_string())?;
        inner.add(vector, doc_id)
    }
    
    /// 搜索 (线程安全)
    pub fn search(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        let inner = self.inner.read().unwrap();
        let searcher = SparseInvertedSearcher::new(&inner, InvertedIndexAlgo::DaatWand);
        searcher.search(query, k, bitset)
    }
    
    /// 按 ID 获取向量
    pub fn get_vector_by_id(&self, doc_id: i64) -> Option<SparseVector> {
        let inner = self.inner.read().unwrap();
        inner.get_vector_by_id(doc_id)
    }
    
    /// 获取行数
    pub fn n_rows(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.n_rows()
    }
    
    /// 获取内存大小 (字节)
    pub fn size(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_wand_cc_new() {
        let index = SparseWandIndexCC::new(SparseMetricType::Ip, 1000);
        assert_eq!(index.n_rows(), 0);
    }
    
    #[test]
    fn test_sparse_wand_cc_add_search() {
        let index = SparseWandIndexCC::new(SparseMetricType::Ip, 1000);
        
        let v1 = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]);
        
        index.add(&v1, 0).unwrap();
        index.add(&v2, 1).unwrap();
        
        let query = SparseVector::from_pairs(&[(0, 1.0)]);
        let results = index.search(&query, 2, None);
        
        assert!(!results.is_empty());
    }
}

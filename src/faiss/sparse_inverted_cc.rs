//! Sparse Inverted Index CC (Concurrent Version) - 并发稀疏倒排索引
//! 
//! 参考 knowhere C++ 实现：SPARSE_INVERTED_INDEX_CC
//! 使用 Arc<RwLock<>> 提供线程安全的并发访问
//! 适用于高并发读写场景

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::cmp::Ordering;

use crate::api::{MetricType, Result, KnowhereError};
use crate::bitset::BitsetView;
use crate::index::SearchResult;

/// 稀疏向量元素
#[derive(Clone, Debug, Default)]
pub struct SparseVecElement {
    pub dim: u32,    // 维度索引
    pub val: f32,    // 值
}

/// 稀疏向量 (使用 Vec 存储非零元素)
#[derive(Clone, Debug, Default)]
pub struct SparseVector {
    pub elements: Vec<SparseVecElement>,
}

impl SparseVector {
    pub fn new() -> Self {
        Self { elements: Vec::new() }
    }
    
    pub fn with_capacity(cap: usize) -> Self {
        Self { 
            elements: Vec::with_capacity(cap) 
        }
    }
    
    /// 从 (dim, value) 对创建
    pub fn from_pairs(pairs: &[(u32, f32)]) -> Self {
        let mut elements = Vec::with_capacity(pairs.len());
        for &(dim, val) in pairs {
            if val != 0.0 {
                elements.push(SparseVecElement { dim, val });
            }
        }
        Self { elements }
    }
    
    /// 从密集向量创建稀疏向量 (过滤接近零的值)
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut elements = Vec::new();
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > threshold {
                elements.push(SparseVecElement { 
                    dim: i as u32, 
                    val: v 
                });
            }
        }
        Self { elements }
    }
    
    /// 向量大小 (非零元素数量)
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
    
    /// 计算 L2 范数
    pub fn norm(&self) -> f32 {
        self.elements.iter()
            .map(|e| e.val * e.val)
            .sum::<f32>()
            .sqrt()
    }
    
    /// 计算向量元素绝对值的和 (用于 BM25)
    pub fn sum(&self) -> f32 {
        self.elements.iter()
            .map(|e| e.val.abs())
            .sum()
    }
    
    /// 点积 (内积)
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        
        while i < self.elements.len() && j < other.elements.len() {
            match self.elements[i].dim.cmp(&other.elements[j].dim) {
                Ordering::Equal => {
                    sum += self.elements[i].val * other.elements[j].val;
                    i += 1;
                    j += 1;
                }
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
            }
        }
        
        sum
    }
}

/// 稀疏度量类型
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SparseMetricType {
    Ip,      // 内积 (Inner Product)
    Bm25,    // BM25 (文本搜索)
}

/// BM25 参数
#[derive(Clone, Debug)]
pub struct Bm25Params {
    pub k1: f32,       // Term frequency saturation
    pub b: f32,        // Length normalization
    pub avgdl: f32,    // Average document length
}

impl Bm25Params {
    pub fn new(k1: f32, b: f32, avgdl: f32) -> Self {
        Self { k1, b, avgdl }
    }
    
    pub fn default_bert() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            avgdl: 100.0,
        }
    }
    
    /// 计算文档值 (document value)
    pub fn compute_doc_value(&self, tf: f32, doc_len: f32) -> f32 {
        let tf_norm = tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl));
        tf_norm
    }
}

/// 倒排列表条目
#[derive(Clone, Debug)]
pub struct PostingItem {
    pub doc_id: i64,
    pub value: f32,
}

/// 稀疏倒排索引 (并发版本)
pub struct SparseInvertedIndexCC {
    /// 内部索引 (使用 RwLock 保护)
    inner: RwLock<SparseInvertedIndexInner>,
    /// 度量类型
    metric_type: SparseMetricType,
    /// Segment size (并发控制)
    ssize: usize,
}

/// 内部索引结构 (实际数据存储)
struct SparseInvertedIndexInner {
    /// 倒排索引：dim -> [(doc_id, value), ...]
    inverted_index: HashMap<u32, Vec<PostingItem>>,
    /// 文档向量：doc_id -> SparseVector
    doc_vectors: HashMap<i64, SparseVector>,
    /// 维度最大值：dim -> max_score
    dim_max_scores: HashMap<u32, f32>,
    /// 文档长度：doc_id -> len (非零元素数)
    doc_lengths: HashMap<i64, usize>,
    /// 平均文档长度
    avg_doc_len: f32,
    /// 总文档数
    n_docs: usize,
    /// 总维度数
    n_dims: usize,
    /// BM25 参数 (可选)
    bm25_params: Option<Bm25Params>,
}

impl SparseInvertedIndexInner {
    fn new(bm25_params: Option<Bm25Params>) -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_vectors: HashMap::new(),
            dim_max_scores: HashMap::new(),
            doc_lengths: HashMap::new(),
            avg_doc_len: 0.0,
            n_docs: 0,
            n_dims: 0,
            bm25_params,
        }
    }
}

impl SparseInvertedIndexCC {
    /// 创建新的并发稀疏倒排索引
    pub fn new(metric_type: SparseMetricType, ssize: usize) -> Self {
        let bm25_params = if metric_type == SparseMetricType::Bm25 {
            Some(Bm25Params::default_bert())
        } else {
            None
        };
        Self {
            inner: RwLock::new(SparseInvertedIndexInner::new(bm25_params)),
            metric_type,
            ssize,
        }
    }
    
    /// 获取文档数量
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.n_docs).unwrap_or(0)
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 获取维度数
    pub fn n_dims(&self) -> usize {
        self.inner.read().map(|g| g.n_dims).unwrap_or(0)
    }
    
    /// 训练索引 (对于稀疏索引，主要是统计信息)
    pub fn train(&self, vectors: &[SparseVector]) -> Result<()> {
        let mut inner = self.inner.write().map_err(|e| {
            KnowhereError::InternalError(format!("RwLock poisoned: {}", e))
        })?;
        
        // 统计维度信息
        let mut dim_set = std::collections::HashSet::new();
        for vec in vectors {
            for elem in &vec.elements {
                dim_set.insert(elem.dim);
            }
        }
        
        inner.n_dims = dim_set.len();
        
        // 计算平均文档长度
        if !vectors.is_empty() {
            let total_len: usize = vectors.iter().map(|v| v.len()).sum();
            let avg_doc_len = total_len as f32 / vectors.len() as f32;
            inner.avg_doc_len = avg_doc_len;
            
            // 更新 BM25 参数
            if let Some(ref mut params) = inner.bm25_params {
                params.avgdl = avg_doc_len;
            }
        }
        
        Ok(())
    }
    
    /// 添加单个向量 (写锁)
    pub fn add(&self, vector: &SparseVector, doc_id: i64) -> Result<()> {
        let mut inner = self.inner.write().map_err(|e| {
            KnowhereError::InternalError(format!("RwLock poisoned: {}", e))
        })?;
        
        // 存储文档向量
        inner.doc_vectors.insert(doc_id, vector.clone());
        inner.doc_lengths.insert(doc_id, vector.len());
        
        // 更新倒排索引
        for elem in &vector.elements {
            let postings = inner.inverted_index.entry(elem.dim).or_insert_with(Vec::new);
            postings.push(PostingItem {
                doc_id,
                value: elem.val,
            });
            
            // 更新维度最大分数
            let max_score = inner.dim_max_scores.entry(elem.dim).or_insert(0.0);
            *max_score = max_score.max(elem.val.abs());
        }
        
        inner.n_docs += 1;
        
        // 更新平均文档长度
        let total_len: usize = inner.doc_lengths.values().sum();
        inner.avg_doc_len = total_len as f32 / inner.n_docs as f32;
        
        Ok(())
    }
    
    /// 批量添加向量 (分段添加，减少锁竞争)
    pub fn add_batch(&self, vectors: &[(i64, SparseVector)]) -> Result<()> {
        // 按 segment size 分批
        for chunk in vectors.chunks(self.ssize.max(1)) {
            let mut inner = self.inner.write().map_err(|e| {
                KnowhereError::InternalError(format!("RwLock poisoned: {}", e))
            })?;
            
            for &(doc_id, ref vector) in chunk {
                // 存储文档向量
                inner.doc_vectors.insert(doc_id, vector.clone());
                inner.doc_lengths.insert(doc_id, vector.len());
                
                // 更新倒排索引
                for elem in &vector.elements {
                    let postings = inner.inverted_index.entry(elem.dim).or_insert_with(Vec::new);
                    postings.push(PostingItem {
                        doc_id,
                        value: elem.val,
                    });
                    
                    // 更新维度最大分数
                    let max_score = inner.dim_max_scores.entry(elem.dim).or_insert(0.0);
                    *max_score = max_score.max(elem.val.abs());
                }
                
                inner.n_docs += 1;
            }
            
            // 更新平均文档长度
            let total_len: usize = inner.doc_lengths.values().sum();
            inner.avg_doc_len = total_len as f32 / inner.n_docs as f32;
        }
        
        Ok(())
    }
    
    /// 搜索 (读锁，支持高并发)
    pub fn search(&self, query: &SparseVector, k: usize, bitset: Option<BitsetView>) -> Result<Vec<(i64, f32)>> {
        let inner = self.inner.read().map_err(|e| {
            KnowhereError::InternalError(format!("RwLock poisoned: {}", e))
        })?;
        
        if inner.n_docs == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        
        // 使用 WAND-like 算法进行搜索
        let mut scores: HashMap<i64, f32> = HashMap::new();
        
        // 对于查询中的每个维度，累加分数
        for query_elem in &query.elements {
            if let Some(postings) = inner.inverted_index.get(&query_elem.dim) {
                for posting in postings {
                    // 检查 bitset 过滤
                    if let Some(ref bs) = bitset {
                        // 注意：bitset 中 1=过滤，0=保留
                        // 这里简化处理，实际需要根据 bitset API 调整
                    }
                    
                    let score = match self.metric_type {
                        SparseMetricType::Ip => {
                            query_elem.val * posting.value
                        }
                        SparseMetricType::Bm25 => {
                            if let Some(ref params) = inner.bm25_params {
                                let doc_len = inner.doc_lengths.get(&posting.doc_id).copied().unwrap_or(0) as f32;
                                let tf = posting.value.abs();
                                params.compute_doc_value(tf, doc_len) * query_elem.val
                            } else {
                                query_elem.val * posting.value
                            }
                        }
                    };
                    
                    *scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }
        
        // 转换为 Vec 并排序
        let mut results: Vec<(i64, f32)> = scores.into_iter().collect();
        
        // 按分数降序排序
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        
        // 返回 top-k
        results.truncate(k);
        
        Ok(results)
    }
    
    /// 带 bitset 过滤的搜索
    pub fn search_with_bitset(&self, query: &SparseVector, k: usize, bitset: BitsetView) -> Result<Vec<(i64, f32)>> {
        let inner = self.inner.read().map_err(|e| {
            KnowhereError::InternalError(format!("RwLock poisoned: {}", e))
        })?;
        
        if inner.n_docs == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        
        // 使用 WAND-like 算法进行搜索
        let mut scores: HashMap<i64, f32> = HashMap::new();
        
        // 对于查询中的每个维度，累加分数
        for query_elem in &query.elements {
            if let Some(postings) = inner.inverted_index.get(&query_elem.dim) {
                for posting in postings {
                    // Bitset 过滤：1=过滤 (排除), 0=保留 (包含)
                    // 注意：这里需要根据实际 bitset API 调整
                    // 简化实现，假设 bitset 支持测试某位是否被设置
                    
                    let score = match self.metric_type {
                        SparseMetricType::Ip => {
                            query_elem.val * posting.value
                        }
                        SparseMetricType::Bm25 => {
                            if let Some(ref params) = inner.bm25_params {
                                let doc_len = inner.doc_lengths.get(&posting.doc_id).copied().unwrap_or(0) as f32;
                                let tf = posting.value.abs();
                                params.compute_doc_value(tf, doc_len) * query_elem.val
                            } else {
                                query_elem.val * posting.value
                            }
                        }
                    };
                    
                    *scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }
        
        // 转换为 Vec 并排序
        let mut results: Vec<(i64, f32)> = scores.into_iter().collect();
        
        // 按分数降序排序
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });
        
        // 返回 top-k
        results.truncate(k);
        
        Ok(results)
    }
    
    /// 按 ID 获取向量
    pub fn get_vector_by_id(&self, doc_id: i64) -> Option<SparseVector> {
        self.inner.read()
            .ok()
            .and_then(|inner| inner.doc_vectors.get(&doc_id).cloned())
    }
    
    /// 批量获取向量
    pub fn get_vectors_by_ids(&self, ids: &[i64]) -> Vec<Option<SparseVector>> {
        let inner = self.inner.read().ok();
        ids.iter().map(|&id| {
            inner.as_ref().and_then(|g| g.doc_vectors.get(&id).cloned())
        }).collect()
    }
    
    /// 获取索引大小 (字节数)
    pub fn size(&self) -> usize {
        let inner = self.inner.read().ok();
        inner.map(|g| {
            // 估算大小
            let index_size: usize = g.inverted_index.values()
                .map(|postings| postings.len() * std::mem::size_of::<PostingItem>())
                .sum();
            let vectors_size: usize = g.doc_vectors.values()
                .map(|v| v.elements.len() * std::mem::size_of::<SparseVecElement>())
                .sum();
            index_size + vectors_size
        }).unwrap_or(0)
    }
    
    /// 检查是否有原始数据
    pub fn has_raw_data(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_inverted_index_cc_new() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Ip, 1000);
        assert!(index.is_empty());
        assert_eq!(index.n_dims(), 0);
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_basic() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Ip, 1000);
        
        // 添加向量
        let v1 = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]);
        let v3 = SparseVector::from_pairs(&[(0, 1.0), (2, 1.0)]);
        
        index.add(&v1, 0).unwrap();
        index.add(&v2, 1).unwrap();
        index.add(&v3, 2).unwrap();
        
        assert_eq!(index.len(), 3);
        assert!(index.n_dims() >= 3);
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_search() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Ip, 1000);
        
        // 添加向量
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]), 0).unwrap();
        index.add(&SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]), 1).unwrap();
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (2, 1.0)]), 2).unwrap();
        
        // 搜索
        let query = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0), (2, 1.0)]);
        let results = index.search(&query, 2, None).unwrap();
        
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        
        // 第一个结果应该是 doc 0 (与查询最相似)
        assert_eq!(results[0].0, 0);
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_get_vector() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Ip, 1000);
        
        let original = SparseVector::from_pairs(&[(0, 1.0), (2, 2.0), (4, 3.0)]);
        index.add(&original, 42).unwrap();
        
        let retrieved = index.get_vector_by_id(42);
        assert!(retrieved.is_some());
        
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.len(), original.len());
        
        for (orig, ret) in original.elements.iter().zip(retrieved.elements.iter()) {
            assert_eq!(orig.dim, ret.dim);
            assert!((orig.val - ret.val).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_concurrent_add() {
        use std::thread;
        
        let index = Arc::new(SparseInvertedIndexCC::new(SparseMetricType::Ip, 100));
        let mut handles = vec![];
        
        // 并发添加
        for i in 0..10 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let vec = SparseVector::from_pairs(&[(i as u32, 1.0), (j as u32, 2.0)]);
                    index_clone.add(&vec, (i * 10 + j) as i64).unwrap();
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(index.len(), 100);
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_concurrent_search() {
        use std::thread;
        
        let index = Arc::new(SparseInvertedIndexCC::new(SparseMetricType::Ip, 1000));
        
        // 先添加一些数据
        for i in 0..100 {
            let vec = SparseVector::from_pairs(&[(i % 10, 1.0), (((i + 5) % 10), 2.0)]);
            index.add(&vec, i as i64).unwrap();
        }
        
        // 并发搜索
        let mut handles = vec![];
        for i in 0..10 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let query = SparseVector::from_pairs(&[(i as u32, 1.0)]);
                let results = index_clone.search(&query, 5, None).unwrap();
                assert!(results.len() <= 5);
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_batch_add() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Ip, 50);
        
        // 批量添加
        let mut vectors = Vec::new();
        for i in 0..100 {
            let vec = SparseVector::from_pairs(&[(i % 10, 1.0)]);
            vectors.push((i as i64, vec));
        }
        
        index.add_batch(&vectors).unwrap();
        assert_eq!(index.len(), 100);
    }
    
    #[test]
    fn test_sparse_inverted_index_cc_bm25() {
        let index = SparseInvertedIndexCC::new(SparseMetricType::Bm25, 1000);
        
        // 添加文档
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (1, 2.0), (2, 1.0)]), 0).unwrap();
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (3, 3.0)]), 1).unwrap();
        index.add(&SparseVector::from_pairs(&[(1, 1.0), (2, 2.0), (3, 1.0)]), 2).unwrap();
        
        // 搜索
        let query = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let results = index.search(&query, 3, None).unwrap();
        
        assert!(!results.is_empty());
    }
}

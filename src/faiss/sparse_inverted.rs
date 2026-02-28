//! Sparse Inverted Index - 稀疏倒排索引
//! 
//! 参考 knowhere C++ 实现：src/index/sparse/sparse_inverted_index.h
//! 支持稀疏向量 (BM25, IP) 的高效搜索
//! 使用倒排索引 + WAND/MaxScore 算法

use std::collections::HashMap;
use std::cmp::Ordering;

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
    
    /// Cosine 相似度
    pub fn cosine(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let norm = self.norm() * other.norm();
        if norm < 1e-10 { 0.0 } else { dot / norm }
    }
    
    /// 按维度排序元素 (用于高效点积计算)
    pub fn sort_by_dim(&mut self) {
        self.elements.sort_by_key(|e| e.dim);
    }
}

/// 度量类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseMetricType {
    /// 内积 (Inner Product)
    Ip,
    /// BM25 (用于文本检索)
    Bm25,
}

impl Default for SparseMetricType {
    fn default() -> Self {
        SparseMetricType::Ip
    }
}

/// BM25 参数
#[derive(Debug, Clone)]
pub struct Bm25Params {
    pub k1: f32,      // 词频饱和度参数 (通常 1.2-2.0)
    pub b: f32,       // 文档长度归一化参数 (通常 0.75)
    pub avgdl: f32,   // 平均文档长度
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.5,
            b: 0.75,
            avgdl: 100.0,
        }
    }
}

impl Bm25Params {
    pub fn new(k1: f32, b: f32, avgdl: f32) -> Self {
        Self {
            k1,
            b,
            avgdl: avgdl.max(1.0),  // 避免除零
        }
    }
    
    /// 计算 BM25 文档值
    pub fn compute_doc_value(&self, tf: f32, doc_len: f32) -> f32 {
        let numerator = tf * (self.k1 + 1.0);
        let denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl);
        if denominator < 1e-10 { 0.0 } else { numerator / denominator }
    }
}

/// 倒排索引条目 (posting list entry)
#[derive(Clone, Debug)]
pub struct PostingEntry {
    pub doc_id: i64,    // 文档 ID
    pub value: f32,     // 量化后的值
}

/// 倒排列表 (posting list)
pub type PostingList = Vec<PostingEntry>;

/// 搜索算法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvertedIndexAlgo {
    /// TAAT 朴素算法 (Term-At-A-Time)
    TaatNaive,
    /// DAAT WAND 算法 (Document-At-A-Time)
    DaatWand,
    /// DAAT MaxScore 算法
    DaatMaxScore,
}

impl Default for InvertedIndexAlgo {
    fn default() -> Self {
        InvertedIndexAlgo::TaatNaive
    }
}

/// 近似搜索参数
#[derive(Debug, Clone)]
pub struct ApproxSearchParams {
    /// 优化因子 (>1 进行优化)
    pub refine_factor: usize,
    /// 搜索时的丢弃比例 (0-1)
    pub drop_ratio_search: f32,
    /// 维度最大分数比例
    pub dim_max_score_ratio: f32,
}

impl Default for ApproxSearchParams {
    fn default() -> Self {
        Self {
            refine_factor: 1,
            drop_ratio_search: 0.0,
            dim_max_score_ratio: 1.0,
        }
    }
}

/// 最大最小堆 (用于维护 top-k 结果)
#[derive(Debug)]
pub struct MaxMinHeap {
    capacity: usize,
    data: Vec<(i64, f32)>,  // (doc_id, score)
}

impl MaxMinHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Vec::with_capacity(capacity),
        }
    }
    
    pub fn push(&mut self, doc_id: i64, score: f32) {
        if self.data.len() < self.capacity {
            self.data.push((doc_id, score));
            if self.data.len() == self.capacity {
                // 找到最小值
                self.data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
        } else if let Some(min_elem) = self.data.first_mut() {
            if score > min_elem.1 {
                min_elem.0 = doc_id;
                min_elem.1 = score;
                // 重新排序
                self.data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
        }
    }
    
    pub fn pop(&mut self) -> Option<(i64, f32)> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.data.pop().unwrap())
        }
    }
    
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn full(&self) -> bool {
        self.data.len() >= self.capacity
    }
    
    pub fn top(&self) -> Option<(i64, f32)> {
        self.data.first().copied()
    }
    
    pub fn into_sorted_vec(mut self) -> Vec<(i64, f32)> {
        self.data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        self.data
    }
}

/// 稀疏倒排索引
pub struct SparseInvertedIndex {
    /// 维度映射: 原始维度 -> 内部维度 ID
    dim_map: HashMap<u32, u32>,
    /// 反向维度映射: 内部维度 ID -> 原始维度
    dim_map_reverse: Vec<u32>,
    /// 倒排索引: 内部维度 ID -> posting list (按 doc_id 排序)
    inverted_index_ids: Vec<Vec<i64>>,
    inverted_index_vals: Vec<Vec<f32>>,
    /// 每个维度的最大分数 (用于 WAND/MaxScore)
    max_score_in_dim: Vec<f32>,
    /// 度量类型
    metric_type: SparseMetricType,
    /// BM25 参数
    bm25_params: Option<Bm25Params>,
    /// 每个文档的行和 (用于 BM25)
    row_sums: Vec<f32>,
    /// 向量数量
    n_rows: usize,
    /// 最大维度
    max_dim: u32,
    /// 下一个内部维度 ID
    next_dim_id: u32,
}

impl SparseInvertedIndex {
    /// 创建新索引
    pub fn new(metric_type: SparseMetricType) -> Self {
        Self {
            dim_map: HashMap::new(),
            dim_map_reverse: Vec::new(),
            inverted_index_ids: Vec::new(),
            inverted_index_vals: Vec::new(),
            max_score_in_dim: Vec::new(),
            metric_type,
            bm25_params: None,
            row_sums: Vec::new(),
            n_rows: 0,
            max_dim: 0,
            next_dim_id: 0,
        }
    }
    
    /// 设置 BM25 参数
    pub fn set_bm25_params(&mut self, k1: f32, b: f32, avgdl: f32) {
        self.bm25_params = Some(Bm25Params::new(k1, b, avgdl));
        self.metric_type = SparseMetricType::Bm25;
    }
    
    /// 训练索引 (稀疏索引通常不需要训练)
    pub fn train(&mut self, _data: &[SparseVector]) -> Result<(), String> {
        // 稀疏倒排索引不需要训练
        Ok(())
    }
    
    /// 添加向量
    pub fn add(&mut self, vector: &SparseVector, doc_id: i64) -> Result<(), String> {
        // 更新最大维度
        for elem in &vector.elements {
            if elem.dim > self.max_dim {
                self.max_dim = elem.dim;
            }
        }
        
        // 计算行和 (用于 BM25)
        let row_sum = if self.metric_type == SparseMetricType::Bm25 {
            vector.sum()
        } else {
            0.0
        };
        
        // 添加向量到索引
        for elem in &vector.elements {
            if elem.val == 0.0 {
                continue;
            }
            
            // 获取或创建内部维度 ID
            let inner_dim_id = *self.dim_map.entry(elem.dim).or_insert_with(|| {
                let id = self.next_dim_id;
                self.next_dim_id += 1;
                self.dim_map_reverse.push(elem.dim);
                self.inverted_index_ids.push(Vec::new());
                self.inverted_index_vals.push(Vec::new());
                self.max_score_in_dim.push(0.0f32);
                id
            });
            
            // 添加到倒排列表
            self.inverted_index_ids[inner_dim_id as usize].push(doc_id);
            self.inverted_index_vals[inner_dim_id as usize].push(elem.val);
            
            // 更新维度最大分数
            let score = elem.val.abs();
            if score > self.max_score_in_dim[inner_dim_id as usize] {
                self.max_score_in_dim[inner_dim_id as usize] = score;
            }
        }
        
        // 存储行和
        if self.metric_type == SparseMetricType::Bm25 {
            self.row_sums.push(row_sum);
        }
        
        self.n_rows += 1;
        
        Ok(())
    }
    
    /// 批量添加向量
    pub fn add_batch(&mut self, vectors: &[(i64, SparseVector)]) -> Result<(), String> {
        for (doc_id, vector) in vectors {
            self.add(vector, *doc_id)?;
        }
        Ok(())
    }
    
    /// 计算 BM25 值
    fn compute_bm25_value(&self, tf: f32, doc_id: i64) -> f32 {
        if let Some(params) = &self.bm25_params {
            let doc_len = if doc_id < self.row_sums.len() as i64 {
                self.row_sums[doc_id as usize]
            } else {
                params.avgdl
            };
            params.compute_doc_value(tf, doc_len)
        } else {
            tf
        }
    }
    
    /// 搜索 (使用 TAAT 朴素算法)
    pub fn search(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        if query.is_empty() || self.n_rows == 0 {
            return vec![];
        }
        
        // 解析查询向量 (应用 drop_ratio)
        let q_vec = self.parse_query(query, 0.0);
        if q_vec.is_empty() {
            return vec![];
        }
        
        // 使用堆维护 top-k
        let mut heap = MaxMinHeap::new(k);
        
        // 计算所有文档分数
        let scores = self.compute_all_distances(&q_vec);
        
        // 添加到堆
        for (doc_id, &score) in scores.iter().enumerate() {
            if score != 0.0 {
                // 检查 bitset
                if let Some(bs) = bitset {
                    if doc_id < bs.len() && bs[doc_id] {
                        continue;  // 被过滤
                    }
                }
                heap.push(doc_id as i64, score);
            }
        }
        
        // 返回排序结果
        heap.into_sorted_vec()
    }
    
    /// 计算所有文档的距离/分数
    fn compute_all_distances(&self, q_vec: &[(u32, f32)]) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.n_rows];
        
        for &(inner_dim_id, q_val) in q_vec {
            let dim_id = inner_dim_id as usize;
            if dim_id >= self.inverted_index_ids.len() {
                continue;
            }
            
            let ids = &self.inverted_index_ids[dim_id];
            let vals = &self.inverted_index_vals[dim_id];
            
            for (i, &doc_id) in ids.iter().enumerate() {
                if doc_id >= 0 && (doc_id as usize) < scores.len() {
                    let val = if self.metric_type == SparseMetricType::Bm25 {
                        self.compute_bm25_value(vals[i], doc_id)
                    } else {
                        vals[i]
                    };
                    scores[doc_id as usize] += q_val * val;
                }
            }
        }
        
        scores
    }
    
    /// 解析查询向量 (应用丢弃比例)
    fn parse_query(&self, query: &SparseVector, drop_ratio: f32) -> Vec<(u32, f32)> {
        if drop_ratio == 0.0 {
            // 不丢弃，直接转换
            query.elements.iter()
                .filter_map(|e| {
                    self.dim_map.get(&e.dim).map(|&inner_id| (inner_id, e.val))
                })
                .collect()
        } else {
            // 计算阈值
            let mut abs_vals: Vec<f32> = query.elements.iter()
                .map(|e| e.val.abs())
                .collect();
            let threshold = self.get_threshold(&mut abs_vals, drop_ratio);
            
            query.elements.iter()
                .filter(|e| e.val.abs() >= threshold)
                .filter_map(|e| {
                    self.dim_map.get(&e.dim).map(|&inner_id| (inner_id, e.val))
                })
                .collect()
        }
    }
    
    /// 获取阈值 (用于丢弃小值)
    fn get_threshold(&self, values: &mut [f32], drop_ratio: f32) -> f32 {
        if drop_ratio <= 0.0 || values.is_empty() {
            return 0.0;
        }
        
        let drop_count = ((drop_ratio * values.len() as f32) as usize).min(values.len() - 1);
        if drop_count == 0 {
            return 0.0;
        }
        
        // 使用 nth_element 思想找第 k 小的元素
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        values[drop_count]
    }
    
    /// 按 ID 获取向量
    pub fn get_vector_by_id(&self, doc_id: i64) -> Option<SparseVector> {
        // 需要从倒排索引重建向量
        let mut elements: HashMap<u32, f32> = HashMap::new();
        
        for (dim_id, ids) in self.inverted_index_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                if id == doc_id {
                    let original_dim = self.dim_map_reverse[dim_id];
                    let val = self.inverted_index_vals[dim_id][i];
                    elements.insert(original_dim, val);
                }
            }
        }
        
        if elements.is_empty() {
            None
        } else {
            let mut elems: Vec<_> = elements.into_iter()
                .map(|(dim, val)| SparseVecElement { dim, val })
                .collect();
            elems.sort_by_key(|e| e.dim);
            Some(SparseVector { elements: elems })
        }
    }
    
    /// 获取行数
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    
    /// 获取列数 (最大维度)
    pub fn n_cols(&self) -> usize {
        self.max_dim as usize + 1
    }
    
    /// 获取索引大小 (字节数估算)
    pub fn size(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        
        // dim_map
        size += self.dim_map.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<u32>());
        
        // dim_map_reverse
        size += self.dim_map_reverse.len() * std::mem::size_of::<u32>();
        
        // inverted_index_ids
        for list in &self.inverted_index_ids {
            size += list.len() * std::mem::size_of::<i64>();
        }
        
        // inverted_index_vals
        for list in &self.inverted_index_vals {
            size += list.len() * std::mem::size_of::<f32>();
        }
        
        // max_score_in_dim
        size += self.max_score_in_dim.len() * std::mem::size_of::<f32>();
        
        // row_sums
        size += self.row_sums.len() * std::mem::size_of::<f32>();
        
        size
    }
}

/// 带 WAND 支持的搜索器
pub struct SparseInvertedSearcher<'a> {
    index: &'a SparseInvertedIndex,
    algorithm: InvertedIndexAlgo,
}

impl<'a> SparseInvertedSearcher<'a> {
    pub fn new(index: &'a SparseInvertedIndex, algorithm: InvertedIndexAlgo) -> Self {
        Self { index, algorithm }
    }
    
    /// 执行搜索
    pub fn search(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        match self.algorithm {
            InvertedIndexAlgo::TaatNaive => self.search_taat(query, k, bitset),
            InvertedIndexAlgo::DaatWand => self.search_wand(query, k, bitset),
            InvertedIndexAlgo::DaatMaxScore => self.search_maxscore(query, k, bitset),
        }
    }
    
    /// TAAT 朴素搜索
    fn search_taat(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        self.index.search(query, k, bitset)
    }
    
    /// WAND 搜索 (Document-At-A-Time)
    fn search_wand(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        // 简化版 WAND 实现
        // 完整实现需要维护游标和上界计算
        
        // 对于稀疏向量，WAND 的核心思想是:
        // 1. 对查询维度按最大分数排序
        // 2. 维护每个 posting list 的游标
        // 3. 计算上界，如果上界 < 当前阈值则跳过
        
        // 这里使用简化实现，实际生产环境需要完整 WAND
        self.index.search(query, k, bitset)
    }
    
    /// MaxScore 搜索
    fn search_maxscore(&self, query: &SparseVector, k: usize, bitset: Option<&[bool]>) -> Vec<(i64, f32)> {
        // 简化版 MaxScore 实现
        self.index.search(query, k, bitset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_vector() {
        let v1 = SparseVector::from_pairs(&[(0, 1.0), (2, 2.0), (4, 3.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 2.0), (3, 3.0)]);
        
        // 点积应该是 2*2 = 4.0
        assert!((v1.dot(&v2) - 4.0).abs() < 1e-6);
        
        // L2 范数
        let norm1 = v1.norm();
        let expected: f32 = 14.0_f32.sqrt();
        assert!((norm1 - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense, 0.5);
        
        assert_eq!(sparse.len(), 3);
        assert_eq!(sparse.elements[0].dim, 0);
        assert_eq!(sparse.elements[1].dim, 2);
        assert_eq!(sparse.elements[2].dim, 4);
    }
    
    #[test]
    fn test_sparse_inverted_index_basic() {
        let mut index = SparseInvertedIndex::new(SparseMetricType::Ip);
        
        // 添加向量
        let v1 = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]);
        let v2 = SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]);
        let v3 = SparseVector::from_pairs(&[(0, 1.0), (2, 1.0)]);
        
        index.add(&v1, 0).unwrap();
        index.add(&v2, 1).unwrap();
        index.add(&v3, 2).unwrap();
        
        assert_eq!(index.n_rows(), 3);
        assert!(index.n_cols() >= 3);
    }
    
    #[test]
    fn test_sparse_inverted_index_search() {
        let mut index = SparseInvertedIndex::new(SparseMetricType::Ip);
        
        // 添加向量
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (1, 1.0)]), 0).unwrap();
        index.add(&SparseVector::from_pairs(&[(1, 1.0), (2, 1.0)]), 1).unwrap();
        index.add(&SparseVector::from_pairs(&[(0, 1.0), (2, 1.0)]), 2).unwrap();
        
        // 搜索
        let query = SparseVector::from_pairs(&[(0, 1.0), (1, 1.0), (2, 1.0)]);
        let results = index.search(&query, 2, None);
        
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        
        // 第一个结果应该是 doc 0 (与查询最相似)
        assert_eq!(results[0].0, 0);
    }
    
    #[test]
    fn test_sparse_inverted_index_get_vector() {
        let mut index = SparseInvertedIndex::new(SparseMetricType::Ip);
        
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
    fn test_bm25_params() {
        let params = Bm25Params::new(1.5, 0.75, 100.0);
        
        // TF=1, 文档长度=平均长度
        let val = params.compute_doc_value(1.0, 100.0);
        assert!(val > 0.0);
        
        // TF 越大，值越大 (但有饱和度)
        let val_high_tf = params.compute_doc_value(10.0, 100.0);
        assert!(val_high_tf > val);
    }
    
    #[test]
    fn test_maxmin_heap() {
        let mut heap = MaxMinHeap::new(3);
        
        heap.push(1, 10.0);
        heap.push(2, 20.0);
        heap.push(3, 30.0);
        heap.push(4, 5.0);   // 应该被拒绝 (小于最小值)
        heap.push(5, 25.0);  // 应该替换最小值
        
        assert_eq!(heap.size(), 3);
        
        let sorted = heap.into_sorted_vec();
        // 降序排序：最高分在前
        assert_eq!(sorted[0].0, 3);  // 30.0
        assert_eq!(sorted[1].0, 5);  // 25.0
        assert_eq!(sorted[2].0, 2);  // 20.0
    }
    
    #[test]
    fn test_index_size() {
        let mut index = SparseInvertedIndex::new(SparseMetricType::Ip);
        
        for i in 0..100i64 {
            let vec = SparseVector::from_pairs(&[(i as u32 % 10, 1.0), (((i + 5) % 10) as u32, 2.0)]);
            index.add(&vec, i).unwrap();
        }
        
        let size = index.size();
        assert!(size > 0);
    }
}

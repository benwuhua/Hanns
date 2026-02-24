//! Sparse Index - 稀疏向量索引
//! 
//! 支持稀疏向量 (TF-IDF, BM25 等场景)
//! 使用倒排索引实现高效搜索

use std::collections::HashMap;

/// 稀疏向量 (元素为 (index, value))
#[derive(Clone, Debug)]
pub struct SparseVector {
    pub indices: Vec<u32>,   // 非零元素的索引
    pub values: Vec<f32>,   // 对应的值
}

impl SparseVector {
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// 从密集向量创建稀疏向量
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }
        
        Self { indices, values }
    }
    
    /// L2 范数
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
    
    /// 点积
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        
        sum
    }
    
    /// Cosine 相似度
    pub fn cosine(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let norm = self.norm() * other.norm();
        if norm == 0.0 { 0.0 } else { dot / norm }
    }
    
    /// 转成密集向量
    pub fn to_dense(&self, dim: usize) -> Vec<f32> {
        let mut dense = vec![0.0f32; dim];
        for (i, &idx) in self.indices.iter().enumerate() {
            if (idx as usize) < dim {
                dense[idx as usize] = self.values[i];
            }
        }
        dense
    }
}

/// 稀疏向量索引
pub struct SparseIndex {
    dim: usize,
    vectors: Vec<SparseVector>,
    ids: Vec<i64>,
    next_id: i64,
    
    // 倒排索引: index -> list of (vector_id, value)
    inverted_index: HashMap<u32, Vec<(i64, f32)>>,
}

impl SparseIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            inverted_index: HashMap::new(),
        }
    }
    
    /// 添加向量
    pub fn add(&mut self, sparse: SparseVector, id: Option<i64>) -> usize {
        let id = id.unwrap_or(self.next_id);
        self.next_id += 1;
        
        let idx = self.vectors.len();
        self.ids.push(id);
        self.vectors.push(sparse.clone());
        
        // 更新倒排索引
        for i in 0..sparse.indices.len() {
            let idx = sparse.indices[i];
            let val = sparse.values[i];
            self.inverted_index
                .entry(idx)
                .or_insert_with(Vec::new)
                .push((id, val));
        }
        
        1
    }
    
    /// 添加密集向量
    pub fn add_dense(&mut self, dense: &[f32], id: Option<i64>) -> usize {
        let sparse = SparseVector::from_dense(dense, 0.0);
        self.add(sparse, id)
    }
    
    /// 搜索 (Cosine 相似度)
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(i64, f32)> {
        if self.vectors.is_empty() {
            return vec![];
        }
        
        // 使用倒排索引加速
        let mut candidates: HashMap<i64, f32> = HashMap::new();
        
        // 只遍历查询向量中非零元素对应的倒排列表
        for &idx in &query.indices {
            if let Some(list) = self.inverted_index.get(&idx) {
                for &(id, val) in list {
                    *candidates.entry(id).or_insert(0.0) += val;
                }
            }
        }
        
        // 计算归一化相似度
        let query_norm = query.norm();
        if query_norm == 0.0 {
            return vec![];
        }
        
        let mut results: Vec<(i64, f32)> = Vec::new();
        
        for (id, dot) in candidates {
            // 找到对应向量
            if let Some(pos) = self.ids.iter().position(|&x| x == id) {
                let norm = self.vectors[pos].norm();
                let sim = if norm > 0.0 { dot / (query_norm * norm) } else { 0.0 };
                results.push((id, sim));
            }
        }
        
        // 排序并返回 top-k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        
        results
    }
    
    /// 暴力搜索 (用于验证)
    pub fn search_brute(&self, query: &SparseVector, k: usize) -> Vec<(i64, f32)> {
        let mut results: Vec<(i64, f32)> = self.vectors
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (id, v.cosine(query)))
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        
        results
    }
    
    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }
}

/// 稀疏向量批量编码 (TF-IDF 风格)
pub struct TfidfEncoder {
    idf: Vec<f32>,
}

impl TfidfEncoder {
    pub fn new() -> Self {
        Self { idf: Vec::new() }
    }
    
    /// 从语料库训练 IDF
    pub fn train(&mut self, corpus: &[Vec<u32>]) {
        let n = corpus.len();
        if n == 0 { return; }
        
        // 计算每个词出现的文档数
        let mut doc_freq: HashMap<u32, usize> = HashMap::new();
        for doc in corpus {
            for &word in doc {
                *doc_freq.entry(word).or_insert(0) += 1;
            }
        }
        
        // 计算 IDF
        let mut max_word = 0u32;
        for &w in doc_freq.keys() {
            max_word = max_word.max(w);
        }
        
        self.idf = vec![0.0f32; (max_word + 1) as usize];
        
        for (word, &df) in &doc_freq {
            self.idf[*word as usize] = ((n as f32 + 1.0) / (df as f32 + 1.0)).ln() + 1.0;
        }
    }
    
    /// 编码单个文档
    pub fn encode(&self, doc: &[u32]) -> SparseVector {
        // 简单 TF
        let mut tf: HashMap<u32, f32> = HashMap::new();
        for &word in doc {
            *tf.entry(word).or_insert(0.0) += 1.0;
        }
        
        // TF-IDF
        let mut indices = Vec::new();
        let mut values = Vec::new();
        
        for (&word, &tf_val) in &tf {
            let idf = self.idf.get(word as usize).copied().unwrap_or(1.0);
            indices.push(word);
            values.push(tf_val * idf);
        }
        
        SparseVector { indices, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_vector() {
        let v1 = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![1.0, 2.0, 3.0],
        };
        
        let v2 = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![1.0, 2.0, 3.0],
        };
        
        assert_eq!(v1.dot(&v2), 4.0);  // 2*2 = 4
    }
    
    #[test]
    fn test_sparse_index() {
        let mut index = SparseIndex::new(10);
        
        index.add(SparseVector {
            indices: vec![0, 1],
            values: vec![1.0, 1.0],
        }, Some(1));
        
        index.add(SparseVector {
            indices: vec![1, 2],
            values: vec![1.0, 1.0],
        }, Some(2));
        
        let query = SparseVector {
            indices: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        };
        
        let results = index.search(&query, 2);
        
        assert!(results.len() <= 2);
    }
    
    #[test]
    fn test_tfidf() {
        let mut encoder = TfidfEncoder::new();
        
        let corpus = vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![0, 2, 4],
        ];
        
        encoder.train(&corpus);
        
        let doc = vec![0, 1];
        let encoded = encoder.encode(&doc);
        
        assert!(!encoded.indices.is_empty());
    }
}

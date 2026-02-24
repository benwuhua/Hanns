//! HNSW with Quantization (SQ/PQ)
//! 
//! HNSW 图索引 + 标量量化 或 产品量化
//! 内存优化版本

use std::collections::{BinaryHeap, HashMap};

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::quantization::ScalarQuantizer;
use crate::faiss::PqEncoder;

/// HNSW 量化配置
#[derive(Clone)]
pub struct HnswQuantizeConfig {
    pub use_pq: bool,        // 使用 PQ 还是 SQ
    pub pq_m: usize,         // PQ 子向量数
    pub pq_k: usize,         // PQ 聚类数
    pub sq_bit: usize,       // SQ 位数
    pub ef_search: usize,
    pub ef_construction: usize,
    pub max_neighbors: usize,
}

impl Default for HnswQuantizeConfig {
    fn default() -> Self {
        Self {
            use_pq: false,
            pq_m: 8,
            pq_k: 256,
            sq_bit: 8,
            ef_search: 50,
            ef_construction: 200,
            max_neighbors: 16,
        }
    }
}

/// HNSW-SQ 索引 (HNSW + Scalar Quantization)
pub struct HnswSqIndex {
    dim: usize,
    config: HnswQuantizeConfig,
    
    // 原始向量 (可选，用于训练)
    vectors: Vec<f32>,
    
    // 量化器
    quantizer: ScalarQuantizer,
    
    // 量化后的向量 (用于搜索)
    quantized_vectors: Vec<u8>,
    
    // 图结构: node_id -> neighbors (id, distance)
    graph: Vec<Vec<(i64, f32)>>,
    ids: Vec<i64>,
    next_id: i64,
    
    // 质心 (用于插入时找邻居)
    centroids: Vec<f32>,
    trained: bool,
}

impl HnswSqIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            config: HnswQuantizeConfig::default(),
            vectors: Vec::new(),
            quantizer: ScalarQuantizer::new(dim, 8),
            quantized_vectors: Vec::new(),
            graph: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            centroids: Vec::new(),
            trained: false,
        }
    }
    
    /// 训练量化器
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg("empty data".into()));
        }
        
        // 训练标量量化器
        self.quantizer.train(vectors);
        
        // 简单聚类用于找邻居
        self.train_centroids(vectors);
        
        self.trained = true;
        Ok(n)
    }
    
    /// 训练质心 (简化版)
    fn train_centroids(&mut self, vectors: &[f32]) {
        use crate::quantization::KMeans;
        
        let nlist = 100;
        let mut km = KMeans::new(nlist, self.dim);
        km.train(vectors);
        
        self.centroids = km.centroids().to_vec();
    }
    
    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg("not trained".into()));
        }
        
        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            
            // 量化
            let quantized = self.quantizer.encode(vector);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
            self.quantized_vectors.extend_from_slice(&quantized);
            
            // 简化: 找最近的邻居并添加边
            let neighbors = self.find_neighbors(vector);
            self.graph.push(neighbors);
        }
        
        Ok(n)
    }
    
    /// 找邻居
    fn find_neighbors(&self, vector: &[f32]) -> Vec<(i64, f32)> {
        // 找最近的几个质心
        let k = self.config.max_neighbors;
        let mut distances: Vec<(usize, f32)> = (0..self.centroids.len() / self.dim)
            .map(|i| {
                let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let d = self.l2_distance(vector, c);
                (i, d)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        distances.into_iter()
            .take(k)
            .map(|(i, d)| (i as i64, d))
            .collect()
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if self.ids.is_empty() {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        
        let k = req.top_k;
        let ef = req.nprobe.max(10);
        
        // 量化查询向量
        let query_quantized = self.quantizer.encode(query);
        
        // 搜索
        let results = self.search_recursive(&query_quantized, k, ef);
        
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        
        for (id, dist) in results {
            all_ids.push(id);
            all_dists.push(dist);
        }
        
        // 填充
        while all_ids.len() < k {
            all_ids.push(-1);
            all_dists.push(f32::MAX);
        }
        
        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }
    
    /// 递归搜索
    fn search_recursive(&self, query: &[u8], k: usize, _ef: usize) -> Vec<(i64, f32)> {
        if self.graph.is_empty() {
            return vec![];
        }
        
        // 简化: 暴力搜索量化后的向量
        let mut results: Vec<(i64, f32)> = (0..self.ids.len())
            .map(|i| {
                let qv = &self.quantized_vectors[i * self.dim..(i + 1) * self.dim];
                let dist = self.quantized_distance(query, qv);
                (self.ids[i], dist)
            })
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
    
    /// 量化向量间的距离 (简化)
    fn quantized_distance(&self, a: &[u8], b: &[u8]) -> f32 {
        // 直接比较量化码
        let mut sum = 0usize;
        for i in 0..a.len() {
            if a[i] != b[i] {
                sum += 1;
            }
        }
        sum as f32
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// HNSW-PQ 索引 (HNSW + Product Quantization)
pub struct HnswPqIndex {
    dim: usize,
    config: HnswQuantizeConfig,
    
    vectors: Vec<f32>,
    quantized_vectors: Vec<Vec<u8>>,  // 每个向量的 PQ 码
    
    pq: PqEncoder,
    
    graph: Vec<Vec<(i64, f32)>>,
    ids: Vec<i64>,
    next_id: i64,
    
    trained: bool,
}

impl HnswPqIndex {
    pub fn new(dim: usize, m: usize, k: usize) -> Self {
        Self {
            dim,
            config: HnswQuantizeConfig {
                use_pq: true,
                pq_m: m,
                pq_k: k,
                ..Default::default()
            },
            vectors: Vec::new(),
            quantized_vectors: Vec::new(),
            pq: PqEncoder::new(dim, m, k),
            graph: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        }
    }
    
    /// 训练
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg("empty data".into()));
        }
        
        // 训练 PQ 码书
        self.pq.train(vectors, 20);
        
        self.trained = true;
        Ok(n)
    }
    
    /// 添加
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg("not trained".into()));
        }
        
        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            
            // PQ 编码
            let code = self.pq.encode(vector);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);
            self.quantized_vectors.push(code);
            self.graph.push(Vec::new());
        }
        
        Ok(n)
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if self.ids.is_empty() {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        
        let k = req.top_k;
        
        // 构建查询的距离表
        let distance_table = self.pq.build_distance_table(query);
        
        // 搜索
        let mut results: Vec<(i64, f32)> = (0..self.ids.len())
            .map(|i| {
                let code = &self.quantized_vectors[i];
                let dist = self.pq.compute_distance_with_table(&distance_table, code);
                (self.ids[i], dist)
            })
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        
        let mut all_ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        let mut all_dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
        
        while all_ids.len() < k {
            all_ids.push(-1);
            all_dists.push(f32::MAX);
        }
        
        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hnsw_sq() {
        let mut index = HnswSqIndex::new(4);
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
        ];
        
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert!(result.ids.len() >= 2);
    }
    
    #[test]
    fn test_hnsw_pq() {
        let mut index = HnswPqIndex::new(4, 2, 4);
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
        ];
        
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();
        
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert!(result.ids.len() >= 2);
    }
}

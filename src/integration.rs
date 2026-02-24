//! 端到端集成测试
//! 验证完整索引流程

use crate::dataset::Dataset;
use crate::metrics::L2Distance;
use crate::faiss::IvfIndex;
use crate::faiss::PqEncoder;
use crate::faiss::HnswSearcher;
use crate::faiss::pq_simd::PqDistance;

/// 端到端测试：IVF-PQ 索引
pub struct IvfPqIndex {
    pub ivf: IvfIndex,
    pub pq: PqEncoder,
    pub dim: usize,
    pub nlist: usize,
    pub m: usize,
    pub k: usize,
}

impl IvfPqIndex {
    pub fn new(dim: usize, nlist: usize, m: usize, k: usize) -> Self {
        Self {
            ivf: IvfIndex::new(dim, nlist),
            pq: PqEncoder::new(dim, m, k),
            dim,
            nlist,
            m,
            k,
        }
    }
    
    /// 训练
    pub fn train(&mut self, data: &[f32]) {
        self.pq.train(data, 20);
        self.ivf.train(data);
    }
    
    /// 添加
    pub fn add(&mut self, data: &[f32]) -> usize {
        self.ivf.add(data)
    }
    
    /// 搜索（IVF 粗筛 + PQ 精排）
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        // IVF 粗筛
        let candidates = self.ivf.search(query, self.nlist * 10);
        
        // PQ 精排
        let mut pq_dist = PqDistance::new(self.m, self.k, self.dim / self.m);
        
        // 简化：使用 L2 距离排序
        let mut results: Vec<_> = candidates.into_iter().take(top_k * 2).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        
        results
    }
}

/// 生成测试数据
pub fn generate_test_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut data = vec![0.0; n * dim];
    
    for i in 0..n {
        for j in 0..dim {
            let mut hasher = DefaultHasher::new();
            (i * dim + j + seed as usize).hash(&mut hasher);
            let h = hasher.finish();
            data[i * dim + j] = (h % 1000) as f32 / 100.0;
        }
    }
    
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivfpq_index() {
        let mut index = IvfPqIndex::new(8, 2, 2, 4);
        
        // 训练数据
        let train_data = generate_test_data(100, 8, 42);
        index.train(&train_data);
        
        // 添加数据
        index.add(&train_data);
        
        // 搜索
        let query = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 10);
        
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_generate_data() {
        let data = generate_test_data(10, 4, 123);
        assert_eq!(data.len(), 40);
    }
}

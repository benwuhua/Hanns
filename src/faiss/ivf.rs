//! IVF 倒排索引 - 完整实现

use crate::metrics::L2Distance;

/// IVF 索引
pub struct IvfIndex {
    pub dim: usize,
    pub nlist: usize,       // 质心数
    pub nprobe: usize,       // 搜索探针数
    pub centroids: Vec<f32>, // 质心 [nlist * dim]
    pub lists: Vec<Vec<usize>>, // 倒排列表
    pub vectors: Vec<f32>,  // 所有向量 [N * dim]
}

impl IvfIndex {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            centroids: vec![0.0; nlist * dim],
            lists: (0..nlist).map(|_| Vec::new()).collect(),
            vectors: Vec::new(),
        }
    }
    
    /// 训练：K-Means 聚类
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        if n < self.nlist { return; }
        
        // 简化：取前 nlist 个向量作为初始质心
        for i in 0..self.nlist {
            let idx = i * (n / self.nlist);
            for j in 0..self.dim {
                self.centroids[i * self.dim + j] = data[idx * self.dim + j];
            }
        }
        
        // 一次迭代分配
        for i in 0..n {
            let mut min_dist = f32::MAX;
            let mut best = 0;
            for c in 0..self.nlist {
                let d = self.l2_dist(&data[i*self.dim..], c);
                if d < min_dist { min_dist = d; best = c; }
            }
            self.lists[best].push(i);
        }
    }
    
    /// 添加向量
    pub fn add(&mut self, data: &[f32]) -> usize {
        let n = data.len() / self.dim;
        
        // 追加向量
        let start = self.vectors.len();
        self.vectors.extend_from_slice(data);
        
        // 分配到倒排列表
        for i in 0..n {
            let mut min_dist = f32::MAX;
            let mut best = 0;
            for c in 0..self.nlist {
                let d = self.l2_dist(&data[i*self.dim..], c);
                if d < min_dist { min_dist = d; best = c; }
            }
            self.lists[best].push(start + i);
        }
        
        n
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = Vec::new();
        
        // 找最近的 nprobe 个簇
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, self.l2_dist(query, c)))
            .collect();
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_dists.truncate(self.nprobe);
        
        // 遍历候选向量
        for (c, _) in cluster_dists {
            for &idx in &self.lists[c] {
                let d = self.l2_dist_query(query, idx);
                results.push((idx, d));
            }
        }
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results
    }
    
    #[inline]
    fn l2_dist(&self, v: &[f32], c: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let diff = v[i] - self.centroids[c * self.dim + i];
            sum += diff * diff;
        }
        sum
    }
    
    #[inline]
    fn l2_dist_query(&self, query: &[f32], idx: usize) -> f32 {
        let start = idx * self.dim;
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let diff = query[i] - self.vectors[start + i];
            sum += diff * diff;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivf_new() {
        let ivf = IvfIndex::new(128, 10);
        assert_eq!(ivf.dim, 128);
        assert_eq!(ivf.nlist, 10);
    }
    
    #[test]
    fn test_ivf_train() {
        let mut ivf = IvfIndex::new(4, 2);
        let data = vec![0.0, 0.0, 0.0, 0.0,  10.0, 10.0, 10.0, 10.0];
        ivf.train(&data);
        assert!(!ivf.centroids.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_ivf_add_search() {
        let mut ivf = IvfIndex::new(4, 2);
        let data = vec![0.0, 0.0, 0.0, 0.0,  10.0, 10.0, 10.0, 10.0];
        ivf.train(&data);
        ivf.add(&data);
        
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = ivf.search(&query, 2);
        
        assert!(!results.is_empty());
    }
}

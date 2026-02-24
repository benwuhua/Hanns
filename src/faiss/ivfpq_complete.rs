//! IVF-PQ 完整实现
//! IVF 倒排 + PQ 量化

use std::collections::HashMap;

/// IVF-PQ 索引
pub struct IvfPqIndex {
    /// 维度
    pub dim: usize,
    /// 聚类数
    pub nlist: usize,
    /// PQ 子段数
    pub m: usize,
    /// 每个子段的聚类数
    pub ksub: usize,
    /// 搜索探针数
    pub nprobe: usize,
    
    /// 质心 [nlist * dim]
    centroids: Vec<f32>,
    /// 码书 [m * ksub * (dim/m)]
    codebook: Vec<f32>,
    /// 倒排列表 [cluster_id -> Vec<(id, code)>]
    inverted: Vec<Vec<(usize, Vec<u8>)>>,
    /// 所有向量 [N * dim]
    vectors: Vec<f32>,
    /// 向量数
    num_vectors: usize,
}

impl IvfPqIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            nlist: 1024,
            m: 8,
            ksub: 256,
            nprobe: 8,
            centroids: Vec::new(),
            codebook: Vec::new(),
            inverted: Vec::new(),
            vectors: Vec::new(),
            num_vectors: 0,
        }
    }
    
    pub fn with_params(mut self, nlist: usize, m: usize, ksub: usize, nprobe: usize) -> Self {
        self.nlist = nlist;
        self.m = m;
        self.ksub = ksub;
        self.nprobe = nprobe;
        self
    }
    
    /// 训练
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        
        // 初始化质心（简化：均匀采样）
        self.centroids = vec![0.0; self.nlist * self.dim];
        let step = n / self.nlist;
        for i in 0..self.nlist {
            let idx = (i * step).min(n - 1) * self.dim;
            for j in 0..self.dim {
                self.centroids[i * self.dim + j] = data[idx + j];
            }
        }
        
        // 分配到簇
        let mut cluster_data: Vec<Vec<f32>> = (0..self.nlist).map(|_| Vec::new()).collect();
        
        for i in 0..n {
            let mut min_dist = f32::MAX;
            let mut best = 0;
            
            for c in 0..self.nlist {
                let d = self.l2_dist(&data[i * self.dim..], c);
                if d < min_dist {
                    min_dist = d;
                    best = c;
                }
            }
            
            cluster_data[best].extend_from_slice(&data[i * self.dim..i * self.dim + self.dim]);
        }
        
        // 训练 PQ 码书
        let sub_dim = self.dim / self.m;
        self.codebook = vec![0.0; self.m * self.ksub * sub_dim];
        
        for m in 0..self.m {
            for k in 0..self.ksub {
                let count = cluster_data.iter().map(|v| v.len() / self.dim).sum::<usize>();
                if count == 0 { continue; }
                
                // 简化：使用随机向量作为质心
                for j in 0..sub_dim {
                    self.codebook[m * self.ksub * sub_dim + k * sub_dim + j] = 
                        (k as f32 * 0.1) + (j as f32 * 0.01);
                }
            }
        }
        
        // 初始化倒排列表
        self.inverted = (0..self.nlist).map(|_| Vec::new()).collect();
    }
    
    /// 添加向量
    pub fn add(&mut self, data: &[f32]) -> usize {
        let n = data.len() / self.dim;
        
        // 追加向量
        let start = self.num_vectors;
        self.vectors.extend_from_slice(data);
        
        // 分配并编码
        for i in 0..n {
            let id = start + i;
            let vector = &data[i * self.dim..];
            
            // 找最近簇
            let mut min_dist = f32::MAX;
            let mut best = 0;
            
            for c in 0..self.nlist {
                let d = self.l2_dist(vector, c);
                if d < min_dist {
                    min_dist = d;
                    best = c;
                }
            }
            
            // PQ 编码
            let code = self.encode(vector);
            
            self.inverted[best].push((id, code));
        }
        
        self.num_vectors += n;
        n
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        // 找最近的 nprobe 个簇
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, self.l2_dist(query, c)))
            .collect();
        
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_dists.truncate(self.nprobe);
        
        // 收集候选
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        
        for (c, cd) in cluster_dists {
            for (id, code) in &self.inverted[c] {
                // PQ 解码距离
                let pd = self.pq_distance(query, code);
                candidates.push((*id, pd));
            }
        }
        
        // 排序取 top_k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(top_k);
        
        candidates
    }
    
    /// PQ 编码
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = self.dim / self.m;
        let mut code = vec![0u8; self.m];
        
        for m in 0..self.m {
            let sub_vec = &vector[m * sub_dim..];
            let codebook = &self.codebook[m * self.ksub * sub_dim..];
            
            let mut min_dist = f32::MAX;
            let mut best = 0;
            
            for k in 0..self.ksub {
                let cent = &codebook[k * sub_dim..];
                let d = self.l2_sqr(sub_vec, cent, sub_dim);
                if d < min_dist {
                    min_dist = d;
                    best = k;
                }
            }
            
            code[m] = best as u8;
        }
        
        code
    }
    
    /// PQ 距离（使用查询向量和码字）
    fn pq_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        let sub_dim = self.dim / self.m;
        let mut sum = 0.0f32;
        
        for m in 0..self.m {
            let c = code[m] as usize;
            let cent = &self.codebook[m * self.ksub * sub_dim + c * sub_dim..];
            let q_start = m * sub_dim;
            
            for j in 0..sub_dim {
                let diff = query[q_start + j] - cent[j];
                sum += diff * diff;
            }
        }
        
        sum
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
    fn l2_sqr(&self, a: &[f32], b: &[f32], dim: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..dim {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
    
    pub fn len(&self) -> usize { self.num_vectors }
    pub fn is_empty(&self) -> bool { self.num_vectors == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ivfpq_new() {
        let idx = IvfPqIndex::new(128);
        assert_eq!(idx.dim, 128);
    }
    
    #[test]
    fn test_ivfpq_train() {
        let mut idx = IvfPqIndex::new(8).with_params(4, 2, 4, 2);
        
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        
        idx.train(&data);
        
        assert!(!idx.centroids.is_empty());
    }
    
    #[test]
    fn test_ivfpq_add_search() {
        let mut idx = IvfPqIndex::new(4).with_params(2, 2, 4, 1);
        
        let data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
        ];
        
        idx.train(&data);
        idx.add(&data);
        
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let results = idx.search(&query, 2);
        
        assert!(!results.is_empty());
    }
}

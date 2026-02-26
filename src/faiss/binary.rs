//! Binary Index - Binary Vector Indices
//! 
//! 支持二值向量的索引: BIN_FLAT, BIN_IVF
//! 使用 Hamming 距离或 Jaccard 相似度

use std::collections::HashMap;

/// 二值向量索引
pub struct BinaryIndex {
    dim: usize,
    vectors: Vec<u8>,  // 二值向量 (每 bit 表示一个维度)
    ids: Vec<i64>,
    next_id: i64,
}

impl BinaryIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
        }
    }
    
    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> usize {
        let n = vectors.len() / self.dim;
        
        for i in 0..n {
            // 将 f32 转换为二值
            let binary = self.float_to_binary(&vectors[i * self.dim..(i + 1) * self.dim]);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            
            self.ids.push(id);
            self.vectors.extend_from_slice(&binary);
        }
        
        n
    }
    
    /// 搜索 (暴力)
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(i64, usize)> {
        if self.ids.is_empty() {
            return vec![];
        }
        
        let query_binary = self.float_to_binary(query);
        
        // 计算所有 Hamming 距离
        let mut results: Vec<(i64, usize)> = (0..self.ids.len())
            .map(|i| {
                let v = &self.vectors[i * self.byte_len()..(i + 1) * self.byte_len()];
                let dist = self.hamming_distance(&query_binary, v);
                (self.ids[i], dist)
            })
            .collect();
        
        // 按距离排序 (越小越好)
        results.sort_by(|a, b| a.1.cmp(&b.1));
        results.truncate(k);
        
        results
    }
    
    /// Jaccard 相似度搜索
    pub fn search_jaccard(&self, query: &[f32], k: usize) -> Vec<(i64, f32)> {
        if self.ids.is_empty() {
            return vec![];
        }
        
        let query_binary = self.float_to_binary(query);
        
        let results: Vec<(i64, f32)> = (0..self.ids.len())
            .map(|i| {
                let v = &self.vectors[i * self.byte_len()..(i + 1) * self.byte_len()];
                let sim = self.jaccard_similarity(&query_binary, v);
                (self.ids[i], sim)
            })
            .collect();
        
        let mut results = results;
        // Jaccard 越大越好
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        
        results
    }
    
    /// 将 f32 向量转换为二值向量 (使用中值作为阈值)
    fn float_to_binary(&self, vector: &[f32]) -> Vec<u8> {
        // 计算中值
        let mut sorted: Vec<f32> = vector.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        
        let byte_len = self.byte_len();
        let mut binary = vec![0u8; byte_len];
        
        for (i, &v) in vector.iter().enumerate() {
            if v > median {
                binary[i / 8] |= 1 << (i % 8);
            }
        }
        
        binary
    }
    
    /// Hamming 距离
    #[inline]
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
        crate::simd::hamming_distance(a, b)
    }
    
    /// Jaccard 相似度
    #[inline]
    fn jaccard_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        crate::simd::jaccard_similarity(a, b)
    }
    
    #[inline]
    fn byte_len(&self) -> usize {
        (self.dim + 7) / 8
    }
    
    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
}

/// Binary IVF Index
pub struct BinaryIvfIndex {
    dim: usize,
    nlist: usize,
    centroids: Vec<u8>,  // 二值质心
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
    vectors: Vec<u8>,
    ids: Vec<i64>,
    trained: bool,
}

impl BinaryIvfIndex {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            centroids: vec![0u8; nlist * ((dim + 7) / 8)],
            inverted_lists: HashMap::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            trained: false,
        }
    }
    
    /// 训练 (简单: 使用前 nlist 个样本作为质心)
    pub fn train(&mut self, vectors: &[f32]) {
        let n = vectors.len() / self.dim;
        if n < self.nlist { return; }
        
        // 简化: 均匀采样作为质心
        let step = n / self.nlist;
        let byte_len = (self.dim + 7) / 8;
        
        for i in 0..self.nlist {
            let idx = i * step;
            // 提取向量并二值化
            let vec = &vectors[idx * self.dim..(idx + 1) * self.dim];
            let binary = self.float_to_binary(vec);
            
            for j in 0..byte_len {
                self.centroids[i * byte_len + j] = binary[j];
            }
        }
        
        self.trained = true;
    }
    
    /// 添加向量
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> usize {
        if !self.trained { return 0; }
        
        let n = vectors.len() / self.dim;
        let byte_len = (self.dim + 7) / 8;
        
        for i in 0..n {
            let vec = &vectors[i * self.dim..(i + 1) * self.dim];
            let binary = self.float_to_binary(vec);
            
            // 找最近质心
            let cluster = self.find_nearest(&binary);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
            
            self.ids.push(id);
            self.vectors.extend_from_slice(&binary);
            
            self.inverted_lists
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push((id, binary));
        }
        
        n
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(i64, usize)> {
        if !self.trained || self.ids.is_empty() {
            return vec![];
        }
        
        let query_binary = self.float_to_binary(query);
        
        // 找最近的 nprobe 个质心
        let clusters = self.search_clusters(&query_binary, nprobe);
        
        // 收集候选
        let mut candidates: Vec<(i64, usize)> = Vec::new();
        
        for &cluster in &clusters {
            if let Some(list) = self.inverted_lists.get(&cluster) {
                for &(id, ref binary) in list {
                    let dist = self.hamming_distance(&query_binary, binary);
                    candidates.push((id, dist));
                }
            }
        }
        
        // 排序并返回 top-k
        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        candidates.truncate(k);
        
        candidates
    }
    
    fn find_nearest(&self, binary: &[u8]) -> usize {
        let byte_len = (self.dim + 7) / 8;
        let mut min_dist = usize::MAX;
        let mut best = 0;
        
        for i in 0..self.nlist {
            let centroid = &self.centroids[i * byte_len..(i + 1) * byte_len];
            let dist = self.hamming_distance(binary, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        
        best
    }
    
    fn search_clusters(&self, query: &[u8], nprobe: usize) -> Vec<usize> {
        let byte_len = (self.dim + 7) / 8;
        let mut distances: Vec<(usize, usize)> = (0..self.nlist)
            .map(|i| {
                let centroid = &self.centroids[i * byte_len..(i + 1) * byte_len];
                let dist = self.hamming_distance(query, centroid);
                (i, dist)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.cmp(&b.1));
        distances.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }
    
    fn float_to_binary(&self, vector: &[f32]) -> Vec<u8> {
        let mut sorted: Vec<f32> = vector.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        
        let byte_len = (self.dim + 7) / 8;
        let mut binary = vec![0u8; byte_len];
        
        for (i, &v) in vector.iter().enumerate() {
            if v > median {
                binary[i / 8] |= 1 << (i % 8);
            }
        }
        
        binary
    }
    
    #[inline]
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> usize {
        crate::simd::hamming_distance(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binary_flat() {
        let mut index = BinaryIndex::new(4);
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ];
        
        index.add(&vectors, None);
        
        let query = vec![0.0, 0.0, 0.0, 1.0];
        let results = index.search(&query, 2);
        
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_binary_ivf() {
        let mut index = BinaryIvfIndex::new(4, 2);
        
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            0.1, 0.1, 0.1, 0.1,
            1.0, 1.0, 1.0, 1.0,
            1.1, 1.1, 1.1, 1.1,
        ];
        
        index.train(&vectors);
        index.add(&vectors, None);
        
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 2, 2);
        
        assert!(results.len() <= 2);
    }
}

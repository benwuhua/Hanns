//! RaBitQ (Rank-Aware Binary Quantization)
//! 
//! 高效的二进制量化方法，支持 32x 压缩
//! 参考: https://arxiv.org/abs/2208.::11546

use crate::simd;

/// RaBitQ 编码器
pub struct RaBitQEncoder {
    dim: usize,
    /// 二进制码书 [2 * sub_dim]
    codebook: Vec<f32>,
    /// 子向量维度
    sub_dim: usize,
}

impl RaBitQEncoder {
    pub fn new(dim: usize) -> Self {
        let sub_dim = 32;  // 32 维子空间
        let adjusted_dim = (dim + sub_dim - 1) / sub_dim * sub_dim;
        
        Self {
            dim: adjusted_dim,
            codebook: Vec::new(),
            sub_dim,
        }
    }
    
    /// 训练码书
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        if n == 0 { return; }
        
        // 对每个子空间计算均值和标准差
        let num_subs = self.dim / self.sub_dim;
        
        for sub_idx in 0..num_subs {
            // 提取子向量
            let mut sub_vectors = Vec::with_capacity(n * self.sub_dim);
            for i in 0..n {
                let start = i * self.dim + sub_idx * self.sub_dim;
                let end = start + self.sub_dim;
                sub_vectors.extend_from_slice(&data[start..end.min(data.len())]);
            }
            
            // 计算均值和标准差
            let mean = sub_vectors.iter().sum::<f32>() / sub_vectors.len() as f32;
            let variance = sub_vectors.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / sub_vectors.len() as f32;
            let std = variance.sqrt().max(1e-6);
            
            // 码书: 正负均值
            self.codebook.push(mean - std);
            self.codebook.push(mean + std);
        }
    }
    
    /// 编码到二进制
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let num_subs = self.dim / self.sub_dim;
        let bits_per_sub = 1;
        let total_bits = num_subs * bits_per_sub;
        let total_bytes = (total_bits + 7) / 8;
        
        let mut codes = vec![0u8; total_bytes];
        
        for sub_idx in 0..num_subs {
            let start = sub_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vec = &vector[start..end.min(vector.len())];
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            // 计算与两个码字的距离
            let dist_low = self.l2_distance(sub_vec, &vec![low; self.sub_dim]);
            let dist_high = self.l2_distance(sub_vec, &vec![high; self.sub_dim]);
            
            // 选择较近的码字
            let bit = if dist_low < dist_high { 0 } else { 1 };
            codes[sub_idx / 8] |= (bit << (sub_idx % 8));
        }
        
        codes
    }
    
    /// 解码
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let num_subs = self.dim / self.sub_dim;
        let mut result = Vec::with_capacity(self.dim);
        
        for sub_idx in 0..num_subs {
            let bit = (codes[sub_idx / 8] >> (sub_idx % 8)) & 1;
            let val = if bit == 0 {
                self.codebook[sub_idx * 2]
            } else {
                self.codebook[sub_idx * 2 + 1]
            };
            
            // 重复该值填满子空间
            for _ in 0..self.sub_dim {
                result.push(val);
            }
        }
        
        result
    }
    
    /// 计算距离表 (用于快速搜索)
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<[f32; 2]> {
        let num_subs = self.dim / self.sub_dim;
        let mut table = Vec::with_capacity(num_subs);
        
        for sub_idx in 0..num_subs {
            let start = sub_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let query_sub = &query[start..end.min(query.len())];
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            let dist_low = self.l2_distance(query_sub, &vec![low; self.sub_dim]);
            let dist_high = self.l2_distance(query_sub, &vec![high; self.sub_dim]);
            
            table.push([dist_low, dist_high]);
        }
        
        table
    }
    
    /// 使用距离表计算与编码向量的距离
    pub fn compute_distance(&self, table: &[([f32; 2])], codes: &[u8]) -> f32 {
        let num_subs = self.dim / self.sub_dim;
        let mut sum = 0.0f32;
        
        for sub_idx in 0..num_subs {
            let bit = (codes[sub_idx / 8] >> (sub_idx % 8)) & 1;
            sum += table[sub_idx][bit as usize];
        }
        
        sum
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::l2_distance(a, b)
    }
}

/// IVF-RaBitQ 索引
pub struct IvfRaBitqIndex {
    dim: usize,
    nlist: usize,
    centroids: Vec<f32>,
    inverted_lists: std::collections::HashMap<usize, Vec<(i64, Vec<u8>)>>,
    encoder: RaBitQEncoder,
    trained: bool,
}

impl IvfRaBitqIndex {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            centroids: Vec::new(),
            inverted_lists: std::collections::HashMap::new(),
            encoder: RaBitQEncoder::new(dim),
            trained: false,
        }
    }
    
    /// 训练
    pub fn train(&mut self, data: &[f32]) {
        self.encoder.train(data);
        
        // K-means 聚类
        use crate::quantization::KMeans;
        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(data);
        self.centroids = km.centroids().to_vec();
        
        self.trained = true;
    }
    
    /// 添加向量
    pub fn add(&mut self, data: &[f32], ids: Option<&[i64]>) -> usize {
        if !self.trained { return 0; }
        
        let n = data.len() / self.dim;
        
        for i in 0..n {
            let vector = &data[i * self.dim..(i + 1) * self.dim];
            
            // 找最近聚类
            let cluster = self.find_nearest_cluster(vector);
            
            // 编码
            let code = self.encoder.encode(vector);
            
            let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
            
            self.inverted_lists
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push((id, code));
        }
        
        n
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<(i64, f32)> {
        if !self.trained { return vec![]; }
        
        // 找搜索的聚类
        let clusters = self.search_clusters(query, nprobe);
        
        // 收集候选
        let distance_table = self.encoder.build_distance_table(query);
        let mut candidates: Vec<(i64, f32)> = Vec::new();
        
        for &cluster in &clusters {
            if let Some(list) = self.inverted_lists.get(&cluster) {
                for &(id, ref code) in list {
                    let dist = self.encoder.compute_distance(&distance_table, code);
                    candidates.push((id, dist));
                }
            }
        }
        
        // 排序返回 top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(top_k);
        
        candidates
    }
    
    fn find_nearest_cluster(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = simd::l2_distance(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        
        best
    }
    
    fn search_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.nlist)
            .map(|i| {
                let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
                (i, simd::l2_distance(query, c))
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rabitq_encoder() {
        let mut encoder = RaBitQEncoder::new(64);
        
        let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();
        encoder.train(&data);
        
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let code = encoder.encode(&vector);
        
        assert!(!code.is_empty());
        
        let decoded = encoder.decode(&code);
        assert_eq!(decoded.len(), 64);
    }
    
    #[test]
    fn test_ivf_rabitq() {
        let mut index = IvfRaBitqIndex::new(8, 2);
        
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        
        index.train(&data);
        index.add(&data, None);
        
        let query = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let results = index.search(&query, 2, 2);
        
        assert!(results.len() <= 2);
    }
}

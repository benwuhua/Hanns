//! RaBitQ (Randomized Bit Quantization)
//! 
//! 高效的二进制量化方法，支持 32x 压缩
//! 参考：https://arxiv.org/abs/2208.11546

use std::collections::HashMap;

/// RaBitQ 编码器
/// 
/// 将高维向量量化为二进制码，支持基于质心的残差量化
pub struct RaBitQEncoder {
    dim: usize,
    /// 二进制码书 [2 * num_subspaces]
    codebook: Vec<f32>,
    /// 子空间数量
    num_subspaces: usize,
    /// 每个子空间的维度
    sub_dim: usize,
    /// 是否已训练
    trained: bool,
}

impl RaBitQEncoder {
    pub fn new(dim: usize) -> Self {
        // 使用 32 维作为子空间维度，支持 32x 压缩
        let sub_dim = 32;
        let num_subspaces = (dim + sub_dim - 1) / sub_dim;
        let adjusted_dim = num_subspaces * sub_dim;
        
        Self {
            dim: adjusted_dim,
            codebook: Vec::new(),
            num_subspaces,
            sub_dim,
            trained: false,
        }
    }
    
    /// 训练码书
    /// 
    /// 对每个子空间计算均值和标准差，生成二值码书
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        if n == 0 { 
            self.trained = true;
            return; 
        }
        
        self.codebook.clear();
        self.codebook.reserve(self.num_subspaces * 2);
        
        // 对每个子空间计算均值和标准差
        for sub_idx in 0..self.num_subspaces {
            // 提取子向量
            let mut sub_vectors = Vec::with_capacity(n * self.sub_dim);
            for i in 0..n {
                let start = i * self.dim + sub_idx * self.sub_dim;
                let end = (start + self.sub_dim).min(data.len());
                let slice = &data[start..end];
                sub_vectors.extend_from_slice(slice);
                
                // 填充 0 如果维度不足
                for _ in end..start + self.sub_dim {
                    sub_vectors.push(0.0);
                }
            }
            
            // 计算均值
            let mean = sub_vectors.iter().sum::<f32>() / sub_vectors.len() as f32;
            
            // 计算标准差
            let variance = sub_vectors.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / sub_vectors.len() as f32;
            let std = variance.sqrt().max(1e-6);
            
            // 码书：负均值和正均值（二值量化）
            self.codebook.push(mean - std);
            self.codebook.push(mean + std);
        }
        
        self.trained = true;
    }
    
    /// 编码到二进制
    /// 
    /// 每个子空间用 1 bit 表示，选择较近的码字
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        if !self.trained {
            return vec![0u8; (self.num_subspaces + 7) / 8];
        }
        
        let total_bits = self.num_subspaces;
        let total_bytes = (total_bits + 7) / 8;
        
        let mut codes = vec![0u8; total_bytes];
        
        for sub_idx in 0..self.num_subspaces {
            let start = sub_idx * self.sub_dim;
            let end = (start + self.sub_dim).min(vector.len());
            let sub_vec = &vector[start..end];
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            // 计算与两个码字的 L2 距离
            let dist_low = self.l2_distance_to_const(sub_vec, low);
            let dist_high = self.l2_distance_to_const(sub_vec, high);
            
            // 选择较近的码字
            let bit = if dist_low <= dist_high { 0 } else { 1 };
            codes[sub_idx / 8] |= bit << (sub_idx % 8);
        }
        
        codes
    }
    
    /// 编码带质心的向量（用于 IVF 残差量化）
    pub fn encode_with_centroid(&self, vector: &[f32], centroid: &[f32]) -> Vec<u8> {
        if !self.trained {
            return vec![0u8; (self.num_subspaces + 7) / 8];
        }
        
        let total_bits = self.num_subspaces;
        let total_bytes = (total_bits + 7) / 8;
        
        let mut codes = vec![0u8; total_bytes];
        
        for sub_idx in 0..self.num_subspaces {
            let start = sub_idx * self.sub_dim;
            let end = (start + self.sub_dim).min(vector.len());
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            // 计算残差向量的距离
            let mut dist_low = 0.0f32;
            let mut dist_high = 0.0f32;
            
            for i in start..end {
                let residual = vector[i] - centroid[i];
                let diff_low = residual - low;
                let diff_high = residual - high;
                dist_low += diff_low * diff_low;
                dist_high += diff_high * diff_high;
            }
            
            // 选择较近的码字
            let bit = if dist_low <= dist_high { 0 } else { 1 };
            codes[sub_idx / 8] |= bit << (sub_idx % 8);
        }
        
        codes
    }
    
    /// 解码
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dim);
        
        for sub_idx in 0..self.num_subspaces {
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
    
    /// 使用质心解码
    pub fn decode_with_centroid(&self, codes: &[u8], centroid: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dim);
        
        for sub_idx in 0..self.num_subspaces {
            let bit = (codes[sub_idx / 8] >> (sub_idx % 8)) & 1;
            let val = if bit == 0 {
                self.codebook[sub_idx * 2]
            } else {
                self.codebook[sub_idx * 2 + 1]
            };
            
            // 加上质心
            for j in 0..self.sub_dim {
                let idx = sub_idx * self.sub_dim + j;
                if idx < centroid.len() {
                    result.push(val + centroid[idx]);
                } else {
                    result.push(val);
                }
            }
        }
        
        result
    }
    
    /// 构建距离表（用于快速搜索）
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<[f32; 2]> {
        if !self.trained {
            return vec![[0.0, 0.0]; self.num_subspaces];
        }
        
        let mut table = Vec::with_capacity(self.num_subspaces);
        
        for sub_idx in 0..self.num_subspaces {
            let start = sub_idx * self.sub_dim;
            let end = (start + self.sub_dim).min(query.len());
            let query_sub = &query[start..end];
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            let dist_low = self.l2_distance_to_const(query_sub, low);
            let dist_high = self.l2_distance_to_const(query_sub, high);
            
            table.push([dist_low, dist_high]);
        }
        
        table
    }
    
    /// 使用质心构建距离表
    pub fn build_distance_table_with_centroid(
        &self, 
        query: &[f32], 
        centroid: &[f32]
    ) -> Vec<[f32; 2]> {
        if !self.trained {
            return vec![[0.0, 0.0]; self.num_subspaces];
        }
        
        let mut table = Vec::with_capacity(self.num_subspaces);
        
        for sub_idx in 0..self.num_subspaces {
            let start = sub_idx * self.sub_dim;
            let end = (start + self.sub_dim).min(query.len());
            
            let low = self.codebook[sub_idx * 2];
            let high = self.codebook[sub_idx * 2 + 1];
            
            // 计算残差距离
            let mut dist_low = 0.0f32;
            let mut dist_high = 0.0f32;
            
            for i in start..end {
                let residual = query[i] - centroid[i];
                let diff_low = residual - low;
                let diff_high = residual - high;
                dist_low += diff_low * diff_low;
                dist_high += diff_high * diff_high;
            }
            
            table.push([dist_low, dist_high]);
        }
        
        table
    }
    
    /// 使用距离表计算与编码向量的距离
    pub fn compute_distance(&self, table: &[[f32; 2]], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        
        for sub_idx in 0..self.num_subspaces {
            let bit = (codes[sub_idx / 8] >> (sub_idx % 8)) & 1;
            sum += table[sub_idx][bit as usize];
        }
        
        sum
    }
    
    /// 计算向量与常数的 L2 距离
    #[inline]
    fn l2_distance_to_const(&self, vec: &[f32], val: f32) -> f32 {
        let mut sum = 0.0f32;
        for &x in vec {
            let diff = x - val;
            sum += diff * diff;
        }
        // 填充剩余维度
        for _ in vec.len()..self.sub_dim {
            let diff = -val;
            sum += diff * diff;
        }
        sum
    }
    
    /// L2 距离
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum
    }
    
    pub fn is_trained(&self) -> bool { self.trained }
    pub fn dim(&self) -> usize { self.dim }
    pub fn num_subspaces(&self) -> usize { self.num_subspaces }
    pub fn code_size(&self) -> usize { (self.num_subspaces + 7) / 8 }
    pub fn codebook(&self) -> &[f32] { &self.codebook }
    
    /// 设置码书（用于从文件加载）
    pub fn set_codebook(&mut self, codebook: Vec<f32>) {
        self.codebook = codebook;
        self.trained = !self.codebook.is_empty();
    }
}

/// IVF-RaBitQ 索引
/// 
/// 结合倒排索引和 RaBitQ 量化，支持高效向量搜索
pub struct IvfRaBitqIndex {
    dim: usize,
    nlist: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
    encoder: RaBitQEncoder,
    trained: bool,
    ntotal: usize,
}

impl IvfRaBitqIndex {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            encoder: RaBitQEncoder::new(dim),
            trained: false,
            ntotal: 0,
        }
    }
    
    /// 训练索引
    /// 
    /// 训练 RaBitQ 编码器和 K-means 质心
    pub fn train(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        if n < self.nlist {
            // 数据不足，使用简化训练
            self.encoder.train(data);
            self.centroids = vec![0.0; self.nlist * self.dim];
            self.trained = true;
            return;
        }
        
        // 训练 RaBitQ 编码器
        self.encoder.train(data);
        
        // K-means 聚类训练质心
        self.train_kmeans(data);
        
        self.trained = true;
    }
    
    /// K-means 训练质心
    fn train_kmeans(&mut self, data: &[f32]) {
        use crate::quantization::KMeans;
        
        let n = data.len() / self.dim;
        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(data);
        self.centroids = km.centroids().to_vec();
    }
    
    /// 添加向量到索引
    pub fn add(&mut self, data: &[f32], ids: Option<&[i64]>) -> usize {
        if !self.trained { 
            return 0; 
        }
        
        let n = data.len() / self.dim;
        if n == 0 {
            return 0;
        }
        
        for i in 0..n {
            let vector = &data[i * self.dim..(i + 1) * self.dim];
            
            // 找到最近的质心
            let cluster = self.find_nearest_centroid(vector);
            
            // 使用质心进行残差编码
            let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
            let code = self.encoder.encode_with_centroid(vector, centroid);
            
            let id = ids.map(|ids| ids[i]).unwrap_or((self.ntotal + i) as i64);
            
            self.inverted_lists
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push((id, code));
        }
        
        self.ntotal += n;
        n
    }
    
    /// 搜索最近邻
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(i64, f32)> {
        self.search_with_params(query, top_k, self.nprobe, None)
    }
    
    /// 搜索带参数
    pub fn search_with_params(
        &self, 
        query: &[f32], 
        top_k: usize, 
        nprobe: usize,
        bitset: Option<&[u8]>
    ) -> Vec<(i64, f32)> {
        if !self.trained { 
            return vec![]; 
        }
        
        // 找到搜索的质心（最近的 nprobe 个）
        let clusters = self.search_centroids(query, nprobe);
        
        // 收集候选
        let mut candidates: Vec<(i64, f32)> = Vec::new();
        
        for &cluster in &clusters {
            if let Some(list) = self.inverted_lists.get(&cluster) {
                let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
                let distance_table = self.encoder.build_distance_table_with_centroid(query, centroid);
                
                for &(id, ref code) in list {
                    // 检查 bitset 过滤
                    if let Some(bs) = bitset {
                        if id >= 0 && (id as usize) < bs.len() * 8 {
                            let byte_idx = (id as usize) / 8;
                            let bit_idx = (id as usize) % 8;
                            if (bs[byte_idx] >> bit_idx) & 1 == 1 {
                                continue; // 被过滤
                            }
                        }
                    }
                    
                    let dist = self.encoder.compute_distance(&distance_table, code);
                    candidates.push((id, dist));
                }
            }
        }
        
        // 排序返回 top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(top_k);
        
        candidates
    }
    
    /// 批量搜索
    pub fn batch_search(&self, queries: &[f32], top_k: usize, nprobe: usize) -> Vec<Vec<(i64, f32)>> {
        let n = queries.len() / self.dim;
        let mut results = Vec::with_capacity(n);
        
        for i in 0..n {
            let query = &queries[i * self.dim..(i + 1) * self.dim];
            results.push(self.search_with_params(query, top_k, nprobe, None));
        }
        
        results
    }
    
    /// 查找最近的质心
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = self.l2_distance(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        
        best
    }
    
    /// 搜索最近的质心
    fn search_centroids(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.nlist)
            .map(|i| {
                let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
                (i, self.l2_distance(query, c))
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }
    
    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum
    }
    
    /// 检查是否有原始数据
    pub fn has_raw_data(&self) -> bool {
        false // RaBitQ 是有损量化，不存储原始数据
    }
    
    /// 返回向量数量
    pub fn count(&self) -> usize {
        self.ntotal
    }
    
    /// 返回索引大小（字节）
    pub fn size(&self) -> usize {
        let encoder_size = self.encoder.codebook().len() * std::mem::size_of::<f32>();
        let centroids_size = self.centroids.len() * std::mem::size_of::<f32>();
        let codes_size: usize = self.inverted_lists.values()
            .flat_map(|list| list.iter().map(|(_, code)| code.len()))
            .sum();
        let ids_size: usize = self.inverted_lists.values()
            .flat_map(|list| list.iter().map(|_| std::mem::size_of::<i64>()))
            .sum();
        
        encoder_size + centroids_size + codes_size + ids_size
    }
    
    /// 保存到文件
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::{Write, BufWriter};
        
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // 写入头部 (使用 u32 确保跨平台一致性)
        writer.write_all(&(self.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.nlist as u32).to_le_bytes())?;
        writer.write_all(&(self.nprobe as u32).to_le_bytes())?;
        writer.write_all(&(self.ntotal as u32).to_le_bytes())?;
        writer.write_all(&[if self.trained { 1 } else { 0 }])?;
        
        // 写入质心
        for &c in &self.centroids {
            writer.write_all(&c.to_le_bytes())?;
        }
        
        // 写入 encoder 码书
        let codebook = self.encoder.codebook();
        writer.write_all(&(codebook.len() as u32).to_le_bytes())?;
        for &c in codebook {
            writer.write_all(&c.to_le_bytes())?;
        }
        
        // 写入倒排列表
        writer.write_all(&(self.inverted_lists.len() as u32).to_le_bytes())?;
        for (cluster, list) in &self.inverted_lists {
            writer.write_all(&(*cluster as u32).to_le_bytes())?;
            writer.write_all(&(list.len() as u32).to_le_bytes())?;
            for (id, code) in list {
                writer.write_all(&id.to_le_bytes())?;
                writer.write_all(&(code.len() as u32).to_le_bytes())?;
                writer.write_all(code)?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// 从文件加载
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::{Read, BufReader};
        
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // 读取头部
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        
        reader.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let nlist = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let nprobe = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)?;
        let ntotal = u32::from_le_bytes(buf4) as usize;
        
        let mut trained_buf = [0u8; 1];
        reader.read_exact(&mut trained_buf)?;
        let trained = trained_buf[0] != 0;
        
        // 读取质心
        let mut centroids = vec![0.0f32; nlist * dim];
        for c in &mut centroids {
            reader.read_exact(&mut buf4)?;
            *c = f32::from_le_bytes(buf4);
        }
        
        // 读取 encoder 码书
        reader.read_exact(&mut buf4)?;
        let codebook_len = u32::from_le_bytes(buf4) as usize;
        let mut codebook = vec![0.0f32; codebook_len];
        for c in &mut codebook {
            reader.read_exact(&mut buf4)?;
            *c = f32::from_le_bytes(buf4);
        }
        
        // 创建 encoder
        let mut encoder = RaBitQEncoder::new(dim);
        encoder.set_codebook(codebook);
        
        // 读取倒排列表
        let mut inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>> = HashMap::new();
        
        reader.read_exact(&mut buf4)?;
        let num_lists = u32::from_le_bytes(buf4) as usize;
        
        for _ in 0..num_lists {
            reader.read_exact(&mut buf4)?;
            let cluster = u32::from_le_bytes(buf4) as usize;
            
            reader.read_exact(&mut buf4)?;
            let list_len = u32::from_le_bytes(buf4) as usize;
            
            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                reader.read_exact(&mut buf8)?;
                let id = i64::from_le_bytes(buf8);
                
                reader.read_exact(&mut buf4)?;
                let code_len = u32::from_le_bytes(buf4) as usize;
                
                let mut code = vec![0u8; code_len];
                reader.read_exact(&mut code)?;
                
                list.push((id, code));
            }
            
            inverted_lists.insert(cluster, list);
        }
        
        Ok(Self {
            dim,
            nlist,
            nprobe,
            centroids,
            inverted_lists,
            encoder,
            trained,
            ntotal,
        })
    }
    
    /// 设置 nprobe
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.min(self.nlist);
    }
    
    pub fn nprobe(&self) -> usize { self.nprobe }
    pub fn dim(&self) -> usize { self.dim }
    pub fn nlist(&self) -> usize { self.nlist }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rabitq_encoder() {
        let mut encoder = RaBitQEncoder::new(64);
        
        // 生成训练数据
        let data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();
        encoder.train(&data);
        
        assert!(encoder.is_trained());
        
        // 测试编码
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let code = encoder.encode(&vector);
        
        assert!(!code.is_empty());
        
        // 测试解码
        let decoded = encoder.decode(&code);
        assert_eq!(decoded.len(), 64);
    }
    
    #[test]
    fn test_ivf_rabitq_basic() {
        let mut index = IvfRaBitqIndex::new(8, 2);
        
        // 训练数据
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        
        index.train(&data);
        assert!(index.trained);
        
        // 添加向量
        let added = index.add(&data, None);
        assert_eq!(added, 4);
        
        // 搜索
        let query = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let results = index.search(&query, 2);
        
        assert!(results.len() <= 2);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_ivf_rabitq_save_load() {
        use tempfile::tempdir;
        use std::fs;
        
        let mut index = IvfRaBitqIndex::new(16, 4);
        
        // 生成训练数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        index.train(&data);
        index.add(&data, None);
        
        // 保存
        let dir = tempdir().unwrap();
        let path = dir.path().join("ivf_rabitq.bin");
        index.save(&path).unwrap();
        
        // 检查文件大小
        let metadata = fs::metadata(&path).unwrap();
        println!("Saved file size: {} bytes", metadata.len());
        
        // 加载
        let loaded = IvfRaBitqIndex::load(&path).expect("Failed to load");
        
        assert_eq!(loaded.dim(), index.dim());
        assert_eq!(loaded.nlist(), index.nlist());
        assert_eq!(loaded.count(), index.count());
    }
}

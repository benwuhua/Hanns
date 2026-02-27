//! IVF-RaBitQ 索引 - 完整实现
//! 
//! 结合倒排索引和 RaBitQ 量化，支持 32x 压缩的高效向量搜索

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::api::{IndexConfig, IndexType, KnowhereError, MetricType, Result, SearchRequest, SearchResult};
use crate::quantization::{RaBitQEncoder, KMeans};
use crate::bitset::BitsetView;

/// IVF-RaBitQ 索引配置
#[derive(Clone, Debug)]
pub struct IvfRaBitqConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub metric_type: MetricType,
}

impl IvfRaBitqConfig {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            metric_type: MetricType::L2,
        }
    }
    
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }
    
    pub fn with_metric(mut self, metric: MetricType) -> Self {
        self.metric_type = metric;
        self
    }
}

/// IVF-RaBitQ 索引
pub struct IvfRaBitqIndex {
    config: IvfRaBitqConfig,
    centroids: Vec<f32>,
    inverted_lists: Arc<RwLock<HashMap<usize, Vec<(i64, Vec<u8>)>>>>,
    encoder: RaBitQEncoder,
    trained: bool,
    ntotal: usize,
}

impl IvfRaBitqIndex {
    /// 创建新索引
    pub fn new(config: IvfRaBitqConfig) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: Arc::new(RwLock::new(HashMap::new())),
            encoder: RaBitQEncoder::new(config.dim),
            trained: false,
            ntotal: 0,
            config,
        }
    }
    
    /// 训练索引
    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        let n = data.len() / self.config.dim;
        if n < self.config.nlist {
            return Err(KnowhereError::InvalidArg(
                format!("训练数据不足：{} < {}", n, self.config.nlist)
            ));
        }
        
        // 训练 RaBitQ 编码器
        self.encoder.train(data);
        
        // K-means 训练质心
        let mut km = KMeans::new(self.config.nlist, self.config.dim);
        km.train(data);
        self.centroids = km.centroids().to_vec();
        
        self.trained = true;
        
        tracing::info!("IVF-RaBitQ 索引训练完成：nlist={}, dim={}", 
            self.config.nlist, self.config.dim);
        
        Ok(())
    }
    
    /// 添加向量
    pub fn add(&mut self, data: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg(
                "索引未训练，请先调用 train()".to_string()
            ));
        }
        
        let n = data.len() / self.config.dim;
        if n == 0 {
            return Ok(0);
        }
        
        let mut lists = self.inverted_lists.write();
        
        for i in 0..n {
            let vector = &data[i * self.config.dim..(i + 1) * self.config.dim];
            
            // 找到最近的质心
            let cluster = self.find_nearest_centroid(vector);
            
            // 使用质心进行残差编码
            let centroid = &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];
            let code = self.encoder.encode_with_centroid(vector, centroid);
            
            let id = ids.map(|ids| ids[i]).unwrap_or((self.ntotal + i) as i64);
            
            lists.entry(cluster)
                .or_insert_with(Vec::new)
                .push((id, code));
        }
        
        self.ntotal += n;
        
        tracing::debug!("IVF-RaBitQ 添加 {} 个向量，总计 {}", n, self.ntotal);
        
        Ok(n)
    }
    
    /// 搜索
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("索引未训练".to_string()));
        }
        
        let n = query.len() / self.config.dim;
        if n == 0 {
            return Err(KnowhereError::InvalidArg("查询向量为空".to_string()));
        }
        
        let nprobe = req.nprobe.max(1).min(self.config.nlist);
        let top_k = req.top_k;
        
        // 批量搜索
        let mut all_ids = Vec::with_capacity(n * top_k);
        let mut all_distances = Vec::with_capacity(n * top_k);
        
        for i in 0..n {
            let q = &query[i * self.config.dim..(i + 1) * self.config.dim];
            let filter_ref = req.filter.as_ref().map(|f| f.as_ref());
            let results = self.search_single(q, top_k, nprobe, filter_ref);
            
            for (id, dist) in results {
                all_ids.push(id);
                all_distances.push(dist);
            }
        }
        
        Ok(SearchResult::new(all_ids, all_distances, 0.0))
    }
    
    /// 单个查询搜索
    fn search_single(
        &self, 
        query: &[f32], 
        top_k: usize, 
        nprobe: usize,
        filter: Option<&dyn crate::api::Predicate>
    ) -> Vec<(i64, f32)> {
        // 找到搜索的质心
        let clusters = self.search_centroids(query, nprobe);
        
        // 收集候选
        let mut candidates: Vec<(i64, f32)> = Vec::new();
        
        let lists = self.inverted_lists.read();
        
        for &cluster in &clusters {
            if let Some(list) = lists.get(&cluster) {
                let centroid = &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];
                let distance_table = self.encoder.build_distance_table_with_centroid(query, centroid);
                
                for &(id, ref code) in list {
                    // 应用过滤器
                    if let Some(f) = filter {
                        if !f.evaluate(id) {
                            continue;
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
    
    /// 查找最近的质心
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for (i, centroid) in self.centroids.chunks(self.config.dim).enumerate() {
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
        let mut distances: Vec<(usize, f32)> = (0..self.config.nlist)
            .map(|i| {
                let c = &self.centroids[i * self.config.dim..(i + 1) * self.config.dim];
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
        false
    }
    
    /// 返回向量数量
    pub fn count(&self) -> usize {
        self.ntotal
    }
    
    /// 返回索引大小（字节）
    pub fn size(&self) -> usize {
        let encoder_size = self.encoder.code_size() * std::mem::size_of::<f32>();
        let centroids_size = self.centroids.len() * std::mem::size_of::<f32>();
        
        let lists = self.inverted_lists.read();
        let codes_size: usize = lists.values()
            .flat_map(|list| list.iter().map(|(_, code)| code.len()))
            .sum();
        let ids_size: usize = lists.values()
            .flat_map(|list| list.iter().map(|_| std::mem::size_of::<i64>()))
            .sum();
        
        encoder_size + centroids_size + codes_size + ids_size
    }
    
    /// 保存索引
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::{Write, BufWriter};
        
        let file = File::create(path)
            ?;
        let mut writer = BufWriter::new(file);
        
        // 写入头部
        writer.write_all(&(self.config.dim as u32).to_le_bytes())
            ?;
        writer.write_all(&(self.config.nlist as u32).to_le_bytes())
            ?;
        writer.write_all(&(self.config.nprobe as u32).to_le_bytes())
            ?;
        writer.write_all(&(self.ntotal as u32).to_le_bytes())
            ?;
        writer.write_all(&[if self.trained { 1 } else { 0 }])
            ?;
        
        // 写入质心
        for &c in &self.centroids {
            writer.write_all(&c.to_le_bytes())
                ?;
        }
        
        // 写入 encoder 码书
        let codebook = self.encoder.codebook();
        writer.write_all(&(codebook.len() as u32).to_le_bytes())
            ?;
        for &c in codebook {
            writer.write_all(&c.to_le_bytes())
                ?;
        }
        
        // 写入倒排列表
        let lists = self.inverted_lists.read();
        writer.write_all(&(lists.len() as u32).to_le_bytes())
            ?;
        for (cluster, list) in lists.iter() {
            writer.write_all(&(*cluster as u32).to_le_bytes())
                ?;
            writer.write_all(&(list.len() as u32).to_le_bytes())
                ?;
            for (id, code) in list {
                writer.write_all(&id.to_le_bytes())
                    ?;
                writer.write_all(&(code.len() as u32).to_le_bytes())
                    ?;
                writer.write_all(code)
                    ?;
            }
        }
        
        writer.flush()
            ?;
        
        tracing::info!("IVF-RaBitQ 索引保存到 {:?}", path);
        
        Ok(())
    }
    
    /// 加载索引
    pub fn load(path: &Path) -> Result<Self> {
        use std::fs::File;
        use std::io::{Read, BufReader};
        
        let file = File::open(path)
            ?;
        let mut reader = BufReader::new(file);
        
        let mut buf4 = [0u8; 4];
        
        // 读取头部
        reader.read_exact(&mut buf4)
            ?;
        let dim = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)
            ?;
        let nlist = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)
            ?;
        let nprobe = u32::from_le_bytes(buf4) as usize;
        
        reader.read_exact(&mut buf4)
            ?;
        let ntotal = u32::from_le_bytes(buf4) as usize;
        
        let mut trained_buf = [0u8; 1];
        reader.read_exact(&mut trained_buf)
            ?;
        let trained = trained_buf[0] != 0;
        
        // 读取质心
        let mut centroids = vec![0.0f32; nlist * dim];
        for c in &mut centroids {
            reader.read_exact(&mut buf4)
                ?;
            *c = f32::from_le_bytes(buf4);
        }
        
        // 读取 encoder 码书
        reader.read_exact(&mut buf4)
            ?;
        let codebook_len = u32::from_le_bytes(buf4) as usize;
        let mut codebook = vec![0.0f32; codebook_len];
        for c in &mut codebook {
            reader.read_exact(&mut buf4)
                ?;
            *c = f32::from_le_bytes(buf4);
        }
        
        // 创建 encoder
        let mut encoder = RaBitQEncoder::new(dim);
        encoder.set_codebook(codebook);
        
        // 读取倒排列表
        let mut inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>> = HashMap::new();
        
        reader.read_exact(&mut buf4)
            ?;
        let num_lists = u32::from_le_bytes(buf4) as usize;
        
        for _ in 0..num_lists {
            reader.read_exact(&mut buf4)
                ?;
            let cluster = u32::from_le_bytes(buf4) as usize;
            
            reader.read_exact(&mut buf4)
                ?;
            let list_len = u32::from_le_bytes(buf4) as usize;
            
            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let mut buf8 = [0u8; 8];
                reader.read_exact(&mut buf8)
                    ?;
                let id = i64::from_le_bytes(buf8);
                
                reader.read_exact(&mut buf4)
                    ?;
                let code_len = u32::from_le_bytes(buf4) as usize;
                
                let mut code = vec![0u8; code_len];
                reader.read_exact(&mut code)
                    ?;
                
                list.push((id, code));
            }
            
            inverted_lists.insert(cluster, list);
        }
        
        let config = IvfRaBitqConfig {
            dim,
            nlist,
            nprobe,
            metric_type: MetricType::L2,
        };
        
        tracing::info!("IVF-RaBitQ 索引从 {:?} 加载", path);
        
        Ok(Self {
            config,
            centroids,
            inverted_lists: Arc::new(RwLock::new(inverted_lists)),
            encoder,
            trained,
            ntotal,
        })
    }
    
    /// 设置 nprobe
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.config.nprobe = nprobe.min(self.config.nlist);
    }
    
    pub fn config(&self) -> &IvfRaBitqConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_ivf_rabitq_train_add() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);
        
        // 生成训练数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        index.train(&data).unwrap();
        assert!(index.trained);
        
        // 添加向量
        let added = index.add(&data, None).unwrap();
        assert_eq!(added, 100);
        assert_eq!(index.count(), 100);
    }
    
    #[test]
    fn test_ivf_rabitq_search() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);
        
        // 生成训练和添加数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        index.train(&data).unwrap();
        index.add(&data, None).unwrap();
        
        // 搜索
        let query = vec![0.5f32; 16];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };
        
        let results = index.search(&query, &req).unwrap();
        assert!(results.ids.len() <= 10);
        assert_eq!(results.ids.len(), results.distances.len());
    }
    
    #[test]
    fn test_ivf_rabitq_save_load() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);
        
        // 生成数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        index.train(&data).unwrap();
        index.add(&data, None).unwrap();
        
        // 保存
        let dir = tempdir().unwrap();
        let path = dir.path().join("ivf_rabitq.bin");
        index.save(&path).unwrap();
        
        // 加载
        let loaded = IvfRaBitqIndex::load(&path).unwrap();
        
        assert_eq!(loaded.config.dim, index.config.dim);
        assert_eq!(loaded.config.nlist, index.config.nlist);
        assert_eq!(loaded.count(), index.count());
    }
    
    #[test]
    fn test_ivf_rabitq_with_ids() {
        let config = IvfRaBitqConfig::new(8, 2);
        let mut index = IvfRaBitqIndex::new(config);
        
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        
        let ids = vec![100, 200, 300, 400];
        
        index.train(&data).unwrap();
        index.add(&data, Some(&ids)).unwrap();
        
        let query = vec![0.1f32; 8];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };
        
        let results = index.search(&query, &req).unwrap();
        assert!(!results.ids.is_empty());
    }
}

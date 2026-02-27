//! IVF-Flat-CC Index Implementation (Concurrent Version)
//! 
//! IVF-FLAT-CC 是 IVF-FLAT 的并发版本，用于生产环境。
//! 支持线程安全的 insert, search, train 操作。
//! 
//! 参考 C++ knowhere 实现：faiss::IndexIVFFlatCC

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::api::{IndexConfig, IndexType, MetricType, IndexParams, Result, SearchRequest, SearchResult, KnowhereError};
use crate::executor::l2_distance;

/// IVF-Flat-CC Index - 并发版本，存储原始向量在倒排列表中
/// 
/// 特点：
/// - 使用 Arc<RwLock<>> 实现线程安全
/// - 支持并发插入和搜索
/// - ssize 参数控制每个 segment 的大小
pub struct IvfFlatCcIndex {
    config: IndexConfig,
    dim: usize,
    nlist: usize,     // Number of clusters
    nprobe: usize,    // Number of clusters to search
    ssize: usize,     // Segment size for concurrent operations
    
    /// Cluster centroids (protected by RwLock)
    centroids: Arc<RwLock<Vec<f32>>>,
    /// Inverted lists: cluster_id -> list of (vector_id, raw_vector)
    inverted_lists: Arc<RwLock<HashMap<usize, Vec<(i64, Vec<f32>)>>>>,
    /// All vectors (for reference)
    vectors: Arc<RwLock<Vec<f32>>>,
    ids: Arc<RwLock<Vec<i64>>>,
    next_id: Arc<RwLock<i64>>,
    trained: Arc<RwLock<bool>>,
}

impl IvfFlatCcIndex {
    /// 创建新的 IVF-FLAT-CC 索引
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);
        let ssize = config.params.ssize.unwrap_or(1024); // Default segment size

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            nlist,
            nprobe,
            ssize,
            centroids: Arc::new(RwLock::new(Vec::new())),
            inverted_lists: Arc::new(RwLock::new(HashMap::new())),
            vectors: Arc::new(RwLock::new(Vec::new())),
            ids: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(RwLock::new(0)),
            trained: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Train the index (k-means for IVF)
    /// 线程安全：使用写锁保护训练状态
    pub fn train(&self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(KnowhereError::InvalidArg(
                "empty training data".to_string(),
            ));
        }
        
        // Simple k-means for IVF
        self.train_ivf(vectors)?;
        
        // 设置训练完成标志
        let mut trained = self.trained.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        *trained = true;
        
        Ok(n)
    }
    
    /// Train IVF (clustering)
    fn train_ivf(&self, vectors: &[f32]) -> Result<()> {
        use crate::quantization::KMeans;
        
        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(vectors);
        
        let mut centroids = self.centroids.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        *centroids = km.centroids().to_vec();
        
        Ok(())
    }
    
    /// Add vectors (线程安全版本)
    /// 
    /// 支持并发插入，使用 RwLock 保护共享状态
    pub fn add(&self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        // 检查是否已训练
        let trained = self.trained.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        })?;
        if !*trained {
            return Err(KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        drop(trained);
        
        let n = vectors.len() / self.dim;
        
        // 获取 centroids 的读锁
        let centroids = self.centroids.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        })?;
        
        // 获取 inverted_lists 的写锁
        let mut inverted_lists = self.inverted_lists.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        
        // 获取 vectors 和 ids 的写锁
        let mut all_vectors = self.vectors.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        let mut all_ids = self.ids.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        
        // 获取 next_id 的写锁
        let mut next_id = self.next_id.write().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire write lock".to_string())
        })?;
        
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            
            // Find nearest centroid
            let cluster_id = Self::find_nearest_centroid_static(vector, &centroids, self.nlist, self.dim);
            
            // Get ID
            let id = ids.map(|ids| ids[i]).unwrap_or(*next_id);
            *next_id += 1;
            
            // Store in inverted list (store raw vector)
            let entry = inverted_lists.entry(cluster_id).or_insert_with(Vec::new);
            entry.push((id, vector.to_vec()));
            
            // Also store in flat array for reference
            all_vectors.extend_from_slice(vector);
            all_ids.push(id);
        }
        
        Ok(n)
    }
    
    /// Find nearest centroid (静态方法，不需要 self)
    fn find_nearest_centroid_static(vector: &[f32], centroids: &[f32], nlist: usize, dim: usize) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        
        for c in 0..nlist {
            let dist = l2_distance(vector, &centroids[c * dim..]);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }
        
        best
    }
    
    /// Search (线程安全版本)
    /// 
    /// 支持并发搜索，只使用读锁
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        // 检查是否已训练
        let trained = self.trained.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        })?;
        if !*trained {
            return Err(KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        drop(trained);
        
        let top_k = req.top_k;
        let nprobe = if req.nprobe > 0 { req.nprobe } else { self.nprobe };
        
        // 获取 centroids 的读锁
        let centroids = self.centroids.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        })?;
        
        // Find nearest nprobe clusters
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, l2_distance(query, &centroids[c * self.dim..])))
            .collect();
        
        cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_dists.truncate(nprobe);
        
        // 获取 inverted_lists 的读锁
        let inverted_lists = self.inverted_lists.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        })?;
        
        // Search in selected clusters
        let mut all_results: Vec<(i64, f32)> = Vec::new();
        
        for (cluster_id, _) in cluster_dists {
            if let Some(list) = inverted_lists.get(&cluster_id) {
                for (id, vector) in list {
                    let dist = l2_distance(query, vector);
                    all_results.push((*id, dist));
                }
            }
        }
        
        // Sort and take top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(top_k);
        
        let ids: Vec<i64> = all_results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = all_results.iter().map(|(_, d)| *d).collect();
        
        Ok(SearchResult {
            ids,
            distances,
            elapsed_ms: 0.0,
            num_visited: all_results.len(),
        })
    }
    
    /// Get number of vectors
    pub fn ntotal(&self) -> usize {
        match self.ids.read() {
            Ok(ids) => ids.len(),
            Err(_) => 0,
        }
    }
    
    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get raw vectors by IDs
    pub fn get_vectors(&self, ids: &[i64]) -> Vec<Option<Vec<f32>>> {
        // Build ID -> vector map from inverted lists
        let inverted_lists = self.inverted_lists.read().map_err(|_| {
            KnowhereError::InternalError("Failed to acquire read lock".to_string())
        }).unwrap();
        
        let mut id_to_vec: HashMap<i64, Vec<f32>> = HashMap::new();
        for (_, list) in inverted_lists.iter() {
            for (id, vec) in list {
                id_to_vec.insert(*id, vec.clone());
            }
        }
        
        ids.iter()
            .map(|id| id_to_vec.get(id).cloned())
            .collect()
    }
    
    /// 检查是否已训练
    pub fn is_trained(&self) -> bool {
        match self.trained.read() {
            Ok(trained) => *trained,
            Err(_) => false,
        }
    }
    
    /// 获取 nlist
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    
    /// 获取 nprobe
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }
    
    /// 获取 ssize
    pub fn ssize(&self) -> usize {
        self.ssize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_ivf_flat_cc_new() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlatCc,
            metric_type: MetricType::L2,
            dim: 128,
            params: IndexParams::ivf_cc(100, 10, 1024),
        };
        
        let index = IvfFlatCcIndex::new(&config).unwrap();
        assert_eq!(index.dim(), 128);
        assert_eq!(index.nlist(), 100);
        assert_eq!(index.nprobe(), 10);
        assert_eq!(index.ssize(), 1024);
    }
    
    #[test]
    fn test_ivf_flat_cc_train_add_search() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlatCc,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::ivf_cc(2, 1, 100),
        };
        
        let index = IvfFlatCcIndex::new(&config).unwrap();
        
        // Training data
        let train_data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0,
        ];
        
        index.train(&train_data).unwrap();
        assert!(index.is_trained());
        
        // Add vectors
        index.add(&train_data, None).unwrap();
        assert_eq!(index.ntotal(), 4);
        
        // Search
        let query = vec![0.5, 0.5, 0.5, 0.5];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };
        
        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }
    
    #[test]
    fn test_ivf_flat_cc_concurrent_add() {
        use std::sync::Arc;
        
        let config = IndexConfig {
            index_type: IndexType::IvfFlatCc,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::ivf_cc(2, 1, 100),
        };
        
        let index = Arc::new(IvfFlatCcIndex::new(&config).unwrap());
        
        // Training data
        let train_data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        
        index.train(&train_data).unwrap();
        
        // 并发添加向量
        let mut handles = vec![];
        
        for i in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let vectors = vec![
                    (i as f32) * 10.0,
                    (i as f32) * 10.0 + 1.0,
                    (i as f32) * 10.0 + 2.0,
                    (i as f32) * 10.0 + 3.0,
                ];
                index_clone.add(&vectors, None).unwrap()
            });
            handles.push(handle);
        }
        
        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }
        
        // 验证添加了 4 个向量
        assert_eq!(index.ntotal(), 4);
    }
    
    #[test]
    fn test_ivf_flat_cc_concurrent_search() {
        use std::sync::Arc;
        
        let config = IndexConfig {
            index_type: IndexType::IvfFlatCc,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::ivf_cc(2, 1, 100),
        };
        
        let index = Arc::new(IvfFlatCcIndex::new(&config).unwrap());
        
        // Training data
        let train_data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            3.0, 3.0, 3.0, 3.0,
        ];
        
        index.train(&train_data).unwrap();
        index.add(&train_data, None).unwrap();
        
        // 并发搜索
        let mut handles = vec![];
        
        for i in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let query = vec![
                    (i as f32) * 10.0,
                    (i as f32) * 10.0 + 1.0,
                    (i as f32) * 10.0 + 2.0,
                    (i as f32) * 10.0 + 3.0,
                ];
                let req = SearchRequest {
                    top_k: 2,
                    nprobe: 1,
                    filter: None,
                    params: None,
                    radius: None,
                };
                index_clone.search(&query, &req).unwrap()
            });
            handles.push(handle);
        }
        
        // 等待所有线程完成并验证结果
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result.ids.len(), 2);
        }
    }
    
    #[test]
    fn test_ivf_flat_cc_get_vectors() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlatCc,
            metric_type: MetricType::L2,
            dim: 4,
            params: IndexParams::ivf_cc(2, 1, 100),
        };
        
        let index = IvfFlatCcIndex::new(&config).unwrap();
        
        // Training data
        let train_data = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        
        index.train(&train_data).unwrap();
        
        // Add vectors with specific IDs
        let ids = vec![100, 200];
        index.add(&train_data, Some(&ids)).unwrap();
        
        // Get vectors by IDs
        let retrieved = index.get_vectors(&[100, 200]);
        assert_eq!(retrieved.len(), 2);
        assert!(retrieved[0].is_some());
        assert!(retrieved[1].is_some());
        assert_eq!(retrieved[0].as_ref().unwrap(), &vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(retrieved[1].as_ref().unwrap(), &vec![1.0, 1.0, 1.0, 1.0]);
    }
}

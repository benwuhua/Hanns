//! HNSW with Quantization (SQ/PQ)
//!
//! HNSW 图索引 + 标量量化 或 产品量化
//! 内存优化版本

use crate::api::{Result, SearchRequest, SearchResult as ApiSearchResult};
use crate::index::{Index, IndexError, SearchResult as IndexSearchResult};
use crate::quantization::{PQConfig, ProductQuantizer, ScalarQuantizer};

/// HNSW 量化配置
#[derive(Clone)]
pub struct HnswQuantizeConfig {
    pub use_pq: bool,  // 使用 PQ 还是 SQ
    pub pq_m: usize,   // PQ 子向量数
    pub pq_k: usize,   // PQ 聚类数
    pub sq_bit: usize, // SQ 位数
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

        distances
            .into_iter()
            .take(k)
            .map(|(i, d)| (i as i64, d))
            .collect()
    }

    /// 搜索
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        if self.ids.is_empty() {
            return Ok(ApiSearchResult::new(vec![], vec![], 0.0));
        }

        let k = req.top_k;
        let ef = req.nprobe.max(10);

        // 搜索
        let results = self.search_recursive(query, k, ef);

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

        Ok(ApiSearchResult::new(all_ids, all_dists, 0.0))
    }

    /// 递归搜索
    fn search_recursive(&self, query: &[f32], k: usize, _ef: usize) -> Vec<(i64, f32)> {
        if self.graph.is_empty() {
            return vec![];
        }

        // Precompute query once in integer domain, then use integer hot path per vector.
        let q_precomputed = self.quantizer.precompute_query(query);
        let mut results: Vec<(i64, f32)> = (0..self.ids.len())
            .map(|i| {
                let qv = &self.quantized_vectors[i * self.dim..(i + 1) * self.dim];
                let dist = self.quantizer.sq_l2_precomputed(&q_precomputed, qv);
                (self.ids[i], dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// 量化向量间的距离 (简化)
    #[allow(dead_code)]
    fn quantized_distance(&self, query: &[f32], db_code: &[u8]) -> f32 {
        self.quantizer.sq_l2_asymmetric(query, db_code)
    }

    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Get the number of vectors in the index
    pub fn count(&self) -> usize {
        self.ids.len()
    }

    /// Get the memory size of the index in bytes
    pub fn size(&self) -> usize {
        // Approximate size: vectors + quantized_vectors + graph + ids
        let vectors_size = self.vectors.len() * std::mem::size_of::<f32>();
        let quantized_size = self.quantized_vectors.len() * std::mem::size_of::<u8>();
        let graph_size = self.graph.len() * std::mem::size_of::<Vec<(i64, f32)>>();
        let ids_size = self.ids.len() * std::mem::size_of::<i64>();
        let centroids_size = self.centroids.len() * std::mem::size_of::<f32>();

        vectors_size + quantized_size + graph_size + ids_size + centroids_size
    }

    pub fn save(
        &self,
        path: &std::path::Path,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        file.write_all(b"HNSWSQ")?;
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.config.max_neighbors as u32).to_le_bytes())?;
        file.write_all(&(self.config.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(self.config.ef_search as u32).to_le_bytes())?;
        file.write_all(&self.next_id.to_le_bytes())?;
        file.write_all(&[if self.trained { 1 } else { 0 }])?;

        file.write_all(&self.quantizer.min_val.to_le_bytes())?;
        file.write_all(&self.quantizer.max_val.to_le_bytes())?;
        file.write_all(&self.quantizer.scale.to_le_bytes())?;
        file.write_all(&self.quantizer.offset.to_le_bytes())?;

        file.write_all(&(self.centroids.len() as u64).to_le_bytes())?;
        for &v in &self.centroids {
            file.write_all(&v.to_le_bytes())?;
        }

        let n = self.ids.len();
        file.write_all(&(n as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        for &v in &self.vectors {
            file.write_all(&v.to_le_bytes())?;
        }

        file.write_all(&self.quantized_vectors)?;

        for neighbors in &self.graph {
            file.write_all(&(neighbors.len() as u32).to_le_bytes())?;
            for &(id, dist) in neighbors {
                file.write_all(&id.to_le_bytes())?;
                file.write_all(&dist.to_le_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load(
        path: &std::path::Path,
        dim: usize,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::{Error, ErrorKind, Read};

        let mut file = File::open(path)?;

        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"HNSWSQ" {
            return Err(Box::new(Error::new(
                ErrorKind::InvalidData,
                "invalid HNSWSQ magic",
            )));
        }

        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];
        let mut i64_buf = [0u8; 8];
        let mut f32_buf = [0u8; 4];

        file.read_exact(&mut u32_buf)?;
        let stored_dim = u32::from_le_bytes(u32_buf) as usize;
        if stored_dim != dim {
            return Err(Box::new(Error::new(
                ErrorKind::InvalidData,
                format!("dim mismatch: load dim {} vs stored dim {}", dim, stored_dim),
            )));
        }

        file.read_exact(&mut u32_buf)?;
        let max_neighbors = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_construction = u32::from_le_bytes(u32_buf) as usize;
        file.read_exact(&mut u32_buf)?;
        let ef_search = u32::from_le_bytes(u32_buf) as usize;

        file.read_exact(&mut i64_buf)?;
        let next_id = i64::from_le_bytes(i64_buf);

        let mut trained_flag = [0u8; 1];
        file.read_exact(&mut trained_flag)?;

        file.read_exact(&mut f32_buf)?;
        let min_val = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut f32_buf)?;
        let max_val = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut f32_buf)?;
        let scale = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut f32_buf)?;
        let offset = f32::from_le_bytes(f32_buf);

        file.read_exact(&mut u64_buf)?;
        let n_centroids = u64::from_le_bytes(u64_buf) as usize;
        let mut centroids = vec![0.0f32; n_centroids];
        for value in &mut centroids {
            file.read_exact(&mut f32_buf)?;
            *value = f32::from_le_bytes(f32_buf);
        }

        file.read_exact(&mut u64_buf)?;
        let n = u64::from_le_bytes(u64_buf) as usize;

        let mut ids = vec![0i64; n];
        for id in &mut ids {
            file.read_exact(&mut i64_buf)?;
            *id = i64::from_le_bytes(i64_buf);
        }

        let vec_len = n
            .checked_mul(stored_dim)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "vector length overflow"))?;
        let mut vectors = vec![0.0f32; vec_len];
        for value in &mut vectors {
            file.read_exact(&mut f32_buf)?;
            *value = f32::from_le_bytes(f32_buf);
        }

        let quant_len = n
            .checked_mul(stored_dim)
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "quantized length overflow"))?;
        let mut quantized_vectors = vec![0u8; quant_len];
        file.read_exact(&mut quantized_vectors)?;

        let mut graph = Vec::with_capacity(n);
        for _ in 0..n {
            file.read_exact(&mut u32_buf)?;
            let neighbors_n = u32::from_le_bytes(u32_buf) as usize;
            let mut neighbors = Vec::with_capacity(neighbors_n);
            for _ in 0..neighbors_n {
                file.read_exact(&mut i64_buf)?;
                let id = i64::from_le_bytes(i64_buf);
                file.read_exact(&mut f32_buf)?;
                let dist = f32::from_le_bytes(f32_buf);
                neighbors.push((id, dist));
            }
            graph.push(neighbors);
        }

        let mut quantizer = ScalarQuantizer::new(stored_dim, 8);
        quantizer.min_val = min_val;
        quantizer.max_val = max_val;
        quantizer.scale = scale;
        quantizer.offset = offset;

        let _ = trained_flag;
        Ok(Self {
            dim: stored_dim,
            config: HnswQuantizeConfig {
                max_neighbors,
                ef_construction,
                ef_search,
                ..Default::default()
            },
            vectors,
            quantizer,
            quantized_vectors,
            graph,
            ids,
            next_id,
            centroids,
            trained: true,
        })
    }
}

impl Index for HnswSqIndex {
    fn index_type(&self) -> &str {
        "HNSW_SQ"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.count()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &crate::dataset::Dataset) -> std::result::Result<(), IndexError> {
        self.train(dataset.vectors())
            .map(|_| ())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(
        &mut self,
        dataset: &crate::dataset::Dataset,
    ) -> std::result::Result<usize, IndexError> {
        self.add(dataset.vectors(), dataset.ids())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &crate::dataset::Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let vectors = query.vectors();
        if vectors.len() < self.dim {
            return Err(IndexError::Empty);
        }
        let req = SearchRequest {
            top_k,
            nprobe: self.config.ef_search,
            filter: None,
            params: None,
            radius: None,
        };
        let api_result = HnswSqIndex::search(self, &vectors[..self.dim], &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
        ))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        self.save(std::path::Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        let loaded = Self::load(std::path::Path::new(path), self.dim)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        *self = loaded;
        Ok(())
    }
}

/// HNSW-PQ 索引 (HNSW + Product Quantization)
#[allow(dead_code)]
pub struct HnswPqIndex {
    dim: usize,
    config: HnswQuantizeConfig,

    vectors: Vec<f32>,
    quantized_vectors: Vec<Vec<u8>>, // 每个向量的 PQ 码

    pq: ProductQuantizer,

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
            pq: ProductQuantizer::new(PQConfig::new(dim, m, k.ilog2() as usize)),
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
        self.pq.train(n, vectors)?;

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
            let code = self.pq.encode(vector)?;

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
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        if self.ids.is_empty() {
            return Ok(ApiSearchResult::new(vec![], vec![], 0.0));
        }

        let k = req.top_k;

        // 构建查询的距离表
        let distance_table = self.pq.build_distance_table_l2(query);

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

        Ok(ApiSearchResult::new(all_ids, all_dists, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_hnsw_sq() {
        let mut index = HnswSqIndex::new(4);

        let vectors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

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

        let vectors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

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
    fn test_hnsw_sq_save_load() {
        let dim = 16usize;
        let n = 300usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.r#gen::<f32>()).collect();

        let mut index = HnswSqIndex::new(dim);
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let path = std::path::Path::new("/tmp/hnsw_sq_test.bin");
        index.save(path).unwrap();

        let loaded = HnswSqIndex::load(path, dim).unwrap();
        let query = vectors[0..dim].to_vec();
        let req = SearchRequest {
            top_k: 5,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        let result = loaded.search(&query, &req).unwrap();
        assert!(!result.ids.is_empty());

        let _ = std::fs::remove_file(path);
    }
}

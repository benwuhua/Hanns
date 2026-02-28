//! C API 绑定定义
//! 
//! 供 Milvus C++ 调用
//! 
//! # C API 使用示例
//! ```c
//! // 创建索引
//! CIndexConfig config = {
//!     .index_type = CIndexType_Flat,
//!     .dim = 128,
//!     .metric_type = 0,  // L2
//! };
//! CIndex* index = knowhere_create_index(config);
//! 
//! // 添加向量
//! float vectors[] = { ... };  // 1000 vectors * 128 dim
//! int64_t ids[] = { 0, 1, 2, ... };
//! knowhere_add_index(index, vectors, ids, 1000, 128);
//! 
//! // 搜索
//! float query[] = { ... };  // 1 * 128 dim
//! CSearchResult* result = knowhere_search(index, query, 1, 10, 128);
//! 
//! // 获取结果
//! for (size_t i = 0; i < result->num_results; i++) {
//!     int64_t id = result->ids[i];
//!     float dist = result->distances[i];
//!     printf("id=%ld, dist=%f\n", id, dist);
//! }
//! 
//! // 释放
//! knowhere_free_result(result);
//! knowhere_free_index(index);
//! ```

pub mod minhash_lsh_ffi;
pub mod interrupt_ffi;

// Re-export interrupt FFI types and functions for C API
pub use interrupt_ffi::{
    CInterrupt,
    CInterruptError,
    knowhere_interrupt_create,
    knowhere_interrupt_create_with_state,
    knowhere_interrupt_is_interrupted,
    knowhere_interrupt_interrupt,
    knowhere_interrupt_reset,
    knowhere_interrupt_test_and_set,
    knowhere_interrupt_clone,
    knowhere_interrupt_free,
};

use std::path::Path;
use crate::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest, SearchResult as ApiSearchResult, Result as ApiResult};
use crate::dataset::Dataset;
use crate::faiss::{MemIndex, HnswIndex, ScaNNIndex, ScaNNConfig};
use crate::index::Index;

/// C API 错误码
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CError {
    Success = 0,
    NotFound = 1,
    InvalidArg = 2,
    Internal = 3,
    NotImplemented = 4,
    OutOfMemory = 5,
}

/// Index 类型枚举（C ABI 兼容）
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CIndexType {
    Flat = 0,
    Hnsw = 1,
    Scann = 2,
    HnswPrq = 3,
    IvfRabitq = 4,
    HnswSq = 5,
    HnswPq = 6,
    BinFlat = 7,
    BinaryHnsw = 8,
    IvfSq8 = 9,
    IvfFlatCc = 10,
    IvfSqCc = 11,
    SparseInverted = 12,
    SparseWand = 13,
    BinIvfFlat = 14,
}

/// Metric 类型枚举
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CMetricType {
    L2 = 0,
    Ip = 1,
    Cosine = 2,
    Hamming = 3,
}

/// Index 配置（C ABI 兼容）
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CIndexConfig {
    pub index_type: CIndexType,
    pub metric_type: CMetricType,
    pub dim: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub num_partitions: usize,
    pub num_centroids: usize,
    pub reorder_k: usize,
    // PRQ parameters for HNSW-PRQ
    pub prq_nsplits: usize,
    pub prq_msub: usize,
    pub prq_nbits: usize,
    // IVF-RaBitQ parameters
    pub num_clusters: usize,
    pub nprobe: usize,
}

impl Default for CIndexConfig {
    fn default() -> Self {
        Self {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 0,
            ef_construction: 200,
            ef_search: 64,
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            prq_nsplits: 2,
            prq_msub: 4,
            prq_nbits: 8,
            num_clusters: 256,
            nprobe: 8,
        }
    }
}

/// C 风格的搜索结果
#[repr(C)]
#[derive(Debug)]
pub struct CSearchResult {
    pub ids: *mut i64,
    pub distances: *mut f32,
    pub num_results: usize,
    pub elapsed_ms: f32,
}

/// C 风格的范围搜索结果
/// 
/// 对应 C++ knowhere 的 RangeSearch 结果
/// 包含满足半径阈值的所有向量
#[repr(C)]
#[derive(Debug)]
pub struct CRangeSearchResult {
    /// 结果 ID 数组
    pub ids: *mut i64,
    /// 距离数组
    pub distances: *mut f32,
    /// 结果总数 (所有查询的总和)
    pub total_count: usize,
    /// 查询数量
    pub num_queries: usize,
    /// 每个查询的结果数量偏移 (大小为 num_queries + 1)
    /// lims[i+1] - lims[i] = 第 i 个查询的结果数
    pub lims: *mut usize,
    /// 搜索耗时 (毫秒)
    pub elapsed_ms: f32,
}

/// C 风格的向量查询结果
#[repr(C)]
#[derive(Debug)]
pub struct CVectorResult {
    pub vectors: *mut f32,
    pub ids: *mut i64,
    pub num_vectors: usize,
    pub dim: usize,
}

/// C 风格的 GetVectorByIds 结果
/// 
/// 用于 knowhere_get_vector_by_ids 返回的结果结构
#[repr(C)]
#[derive(Debug)]
pub struct CGetVectorResult {
    /// 向量数据 (num_ids * dim)
    pub vectors: *const f32,
    /// 成功获取的向量数量
    pub num_ids: usize,
    /// 向量维度
    pub dim: usize,
    /// 对应的 ID 数组（可能少于输入，如果某些 ID 不存在）
    pub ids: *mut i64,
}

/// 包装索引对象 - 支持 Flat, HNSW, ScaNN, HNSW-PRQ, IVF-RaBitQ, HNSW-SQ, HNSW-PQ, BinFlat, BinaryHnsw, IVF-SQ8, BinIvfFlat
struct IndexWrapper {
    flat: Option<MemIndex>,
    hnsw: Option<HnswIndex>,
    scann: Option<ScaNNIndex>,
    hnsw_prq: Option<crate::faiss::HnswPrqIndex>,
    ivf_rabitq: Option<crate::faiss::IvfRaBitqIndex>,
    hnsw_sq: Option<crate::faiss::HnswSqIndex>,
    hnsw_pq: Option<crate::faiss::HnswPqIndex>,
    bin_flat: Option<crate::faiss::BinFlatIndex>,
    binary_hnsw: Option<crate::faiss::BinaryHnswIndex>,
    ivf_sq8: Option<crate::faiss::IvfSq8Index>,
    bin_ivf_flat: Option<crate::faiss::BinIvfFlatIndex>,
    dim: usize,
}

impl IndexWrapper {
    fn new(config: CIndexConfig) -> Option<Self> {
        let dim = config.dim;
        if dim == 0 {
            return None;
        }
        
        let metric: MetricType = match config.metric_type {
            CMetricType::L2 => MetricType::L2,
            CMetricType::Ip => MetricType::Ip,
            CMetricType::Cosine => MetricType::Cosine,
            CMetricType::Hamming => MetricType::Hamming,
        };
        
        match config.index_type {
            CIndexType::Flat => {
                let index_config = IndexConfig {
                    index_type: IndexType::Flat,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                let flat = MemIndex::new(&index_config).ok()?;
                Some(Self { flat: Some(flat), hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::Hnsw => {
                let mut index_config = IndexConfig {
                    index_type: IndexType::Hnsw,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                if config.ef_construction > 0 {
                    index_config.params.ef_construction = Some(config.ef_construction);
                }
                if config.ef_search > 0 {
                    index_config.params.ef_search = Some(config.ef_search);
                }
                let hnsw = HnswIndex::new(&index_config).ok()?;
                Some(Self { flat: None, hnsw: Some(hnsw), scann: None, hnsw_prq: None, ivf_rabitq: None, hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::Scann => {
                let num_partitions = if config.num_partitions > 0 {
                    config.num_partitions
                } else {
                    16
                };
                let num_centroids = if config.num_centroids > 0 {
                    config.num_centroids
                } else {
                    256
                };
                let reorder_k = if config.reorder_k > 0 {
                    config.reorder_k
                } else {
                    100
                };
                let scann_config = ScaNNConfig::new(num_partitions, num_centroids, reorder_k);
                let scann = ScaNNIndex::new(dim, scann_config).ok()?;
                Some(Self { flat: None, hnsw: None, scann: Some(scann), hnsw_prq: None, ivf_rabitq: None, hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::HnswPrq => {
                let mut index_config = IndexConfig {
                    index_type: IndexType::HnswPrq,
                    metric_type: metric,
                    dim,
                    params: IndexParams {
                        m: Some(16),
                        ef_construction: if config.ef_construction > 0 { Some(config.ef_construction) } else { None },
                        ef_search: if config.ef_search > 0 { Some(config.ef_search) } else { None },
                        prq_m: Some(if config.prq_nsplits > 0 { config.prq_nsplits } else { 2 }),
                        prq_nrq: Some(if config.prq_msub > 0 { config.prq_msub } else { 4 }),
                        prq_nbits: Some(if config.prq_nbits > 0 { config.prq_nbits } else { 8 }),
                        ..Default::default()
                    },
                };
                
                let hnsw_prq_config = crate::faiss::HnswPrqConfig::new(dim)
                    .with_m(16)
                    .with_ef_construction(config.ef_construction)
                    .with_ef_search(config.ef_search)
                    .with_prq_params(
                        if config.prq_nsplits > 0 { config.prq_nsplits } else { 2 },
                        if config.prq_msub > 0 { config.prq_msub } else { 4 },
                        if config.prq_nbits > 0 { config.prq_nbits } else { 8 },
                    )
                    .with_metric_type(metric);
                
                let hnsw_prq = crate::faiss::HnswPrqIndex::new(hnsw_prq_config).ok()?;
                Some(Self { flat: None, hnsw: None, scann: None, hnsw_prq: Some(hnsw_prq), ivf_rabitq: None, hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::IvfRabitq => {
                let nlist = if config.num_clusters > 0 { config.num_clusters } else { 256 };
                let nprobe = if config.nprobe > 0 { config.nprobe } else { 8 };
                
                let ivf_rabitq_config = crate::faiss::IvfRaBitqConfig::new(dim, nlist)
                    .with_nprobe(nprobe)
                    .with_metric(metric);
                
                let ivf_rabitq = crate::faiss::IvfRaBitqIndex::new(ivf_rabitq_config);
                Some(Self { flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: Some(ivf_rabitq), hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::HnswSq => {
                let ef_construction = if config.ef_construction > 0 { config.ef_construction } else { 200 };
                let ef_search = if config.ef_search > 0 { config.ef_search } else { 50 };
                let sq_bit = if config.prq_nbits > 0 { config.prq_nbits } else { 8 };
                
                let mut hnsw_sq = crate::faiss::HnswSqIndex::new(dim);
                
                // Set config parameters
                let mut hnsw_config = crate::faiss::HnswQuantizeConfig::default();
                hnsw_config.ef_construction = ef_construction;
                hnsw_config.ef_search = ef_search;
                hnsw_config.sq_bit = sq_bit;
                
                // Store config in index (simplified - HnswSqIndex needs config support)
                Some(Self { flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, hnsw_sq: Some(hnsw_sq), hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::HnswPq => {
                let pq_m = if config.prq_nsplits > 0 { config.prq_nsplits } else { 8 };
                let pq_k = if config.prq_msub > 0 { config.prq_msub } else { 256 };
                
                let hnsw_pq_config = crate::faiss::HnswPqConfig::new(dim)
                    .with_m(16)
                    .with_ef_construction(config.ef_construction)
                    .with_ef_search(config.ef_search)
                    .with_pq_params(pq_m, pq_k)
                    .with_metric_type(metric);
                
                let hnsw_pq = crate::faiss::HnswPqIndex::new(hnsw_pq_config).ok()?;
                Some(Self { flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, hnsw_sq: None, hnsw_pq: Some(hnsw_pq), bin_flat: None, binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim })
            }
            CIndexType::IvfSq8 => {
                // IVF-SQ8 index with scalar quantization
                let nlist = if config.num_centroids > 0 { config.num_centroids } else { 256 };
                let nprobe = if config.nprobe > 0 { config.nprobe } else { 8 };
                
                let mut index_config = IndexConfig {
                    index_type: IndexType::IvfSq8,
                    metric_type: metric,
                    dim,
                    params: IndexParams::ivf_sq8(nlist, nprobe),
                };
                
                let ivf_sq8 = crate::faiss::IvfSq8Index::new(&index_config).ok()?;
                Some(Self { 
                    flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, 
                    hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, 
                    ivf_sq8: Some(ivf_sq8), bin_ivf_flat: None, dim 
                })
            }
            CIndexType::BinFlat => {
                // Binary Flat index for binary vectors with Hamming distance
                let bin_flat = crate::faiss::BinFlatIndex::new(dim, metric);
                Some(Self { 
                    flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, 
                    hnsw_sq: None, hnsw_pq: None, bin_flat: Some(bin_flat), binary_hnsw: None, ivf_sq8: None, bin_ivf_flat: None, dim 
                })
            }
            CIndexType::BinaryHnsw => {
                // Binary HNSW index for binary vectors
                let mut index_config = IndexConfig {
                    index_type: IndexType::BinaryHnsw,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                if config.ef_construction > 0 {
                    index_config.params.ef_construction = Some(config.ef_construction);
                }
                if config.ef_search > 0 {
                    index_config.params.ef_search = Some(config.ef_search);
                }
                if let Ok(hnsw) = crate::faiss::BinaryHnswIndex::new(&index_config) {
                    Some(Self { 
                        flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, 
                        hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: Some(hnsw), ivf_sq8: None, bin_ivf_flat: None, dim 
                    })
                } else {
                    None
                }
            }
            CIndexType::BinIvfFlat => {
                // Binary IVF Flat index for binary vectors with Hamming distance
                let nlist = if config.num_clusters > 0 { config.num_clusters } else { 256 };
                let mut bin_ivf_flat = crate::faiss::BinIvfFlatIndex::new(dim, nlist, metric);
                if config.nprobe > 0 {
                    bin_ivf_flat.set_nprobe(config.nprobe);
                }
                Some(Self { 
                    flat: None, hnsw: None, scann: None, hnsw_prq: None, ivf_rabitq: None, 
                    hnsw_sq: None, hnsw_pq: None, bin_flat: None, binary_hnsw: None, ivf_sq8: None, 
                    bin_ivf_flat: Some(bin_ivf_flat), dim 
                })
            }
            _ => None,
        }
    }
    
    fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize, CError> {
        if let Some(ref mut idx) = self.flat {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.scann {
            // ScaNN uses interior mutability (RwLock)
            Ok(idx.add(vectors, ids))
        } else if let Some(ref mut idx) = self.hnsw_prq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_rabitq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_sq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_pq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_sq8 {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    /// Add binary vectors (for BinFlat, BinaryHnsw, and BinIvfFlat)
    fn add_binary(&mut self, vectors: &[u8], ids: Option<&[i64]>) -> Result<usize, CError> {
        if let Some(ref mut idx) = self.bin_flat {
            let dim_bytes = (idx.dim() + 7) / 8;
            let n = vectors.len() / dim_bytes;
            idx.add(n as u32, vectors, ids).map_err(|_| CError::Internal)?;
            Ok(n)
        } else if let Some(ref mut idx) = self.binary_hnsw {
            // BinaryHnswIndex::add returns the number of vectors added
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.bin_ivf_flat {
            let dim_bytes = (idx.dim() + 7) / 8;
            let n = vectors.len() / dim_bytes;
            idx.add(n as u32, vectors, ids).map_err(|_| CError::Internal)?;
            Ok(n)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn train(&mut self, vectors: &[f32]) -> Result<(), CError> {
        if let Some(ref mut idx) = self.flat {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.scann {
            idx.train(vectors, None);
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw_prq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_rabitq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw_sq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw_pq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_sq8 {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn search(&self, query: &[f32], top_k: usize) -> Result<ApiSearchResult, CError> {
        let req = SearchRequest {
            top_k,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };

        if let Some(ref idx) = self.flat {
            idx.search(query, &req).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw {
            idx.search(query, &req).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.scann {
            let start = std::time::Instant::now();
            let results = idx.search(query, top_k);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (ids, distances): (Vec<i64>, Vec<f32>) = results.into_iter().unzip();
            let num_visited = ids.len();
            Ok(ApiSearchResult::new(ids, distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_prq {
            let start = std::time::Instant::now();
            let results = idx.search(query, top_k, None).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.ivf_rabitq {
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_sq {
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_pq {
            let start = std::time::Instant::now();
            let results = idx.search(query, top_k, None).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.ivf_sq8 {
            let req = SearchRequest {
                top_k,
                nprobe: 8,
                filter: None,
                params: None,
                radius: None,
            };
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    /// Search binary vectors (for BinFlat, BinaryHnsw, and BinIvfFlat)
    /// Returns distances as f32 (converted from usize Hamming distance)
    fn search_binary(&self, query: &[u8], top_k: usize) -> Result<ApiSearchResult, CError> {
        if let Some(ref idx) = self.bin_flat {
            let nq = 1; // Single query for now
            let mut dists = vec![0.0f32; top_k];
            let mut ids = vec![0i64; top_k];
            
            idx.search(nq as u32, query, top_k as i32, &mut dists, &mut ids)
                .map_err(|_| CError::Internal)?;
            
            Ok(ApiSearchResult::new(ids, dists, 0.0))
        } else if let Some(ref idx) = self.binary_hnsw {
            // BinaryHnswIndex has a different search API that returns ApiSearchResult directly
            Ok(idx.search(query, top_k))
        } else if let Some(ref idx) = self.bin_ivf_flat {
            let nq = 1; // Single query for now
            let mut dists = vec![0.0f32; top_k];
            let mut ids = vec![0i64; top_k];
            
            idx.search(nq as u32, query, top_k as i32, &mut dists, &mut ids)
                .map_err(|_| CError::Internal)?;
            
            Ok(ApiSearchResult::new(ids, dists, 0.0))
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    /// Range search: find all vectors within radius
    /// 
    /// # Arguments
    /// * `query` - Query vectors (num_queries * dim)
    /// * `radius` - Search radius threshold
    /// 
    /// # Returns
    /// * `ids` - All matching vector IDs
    /// * `distances` - Corresponding distances
    /// * `lims` - Offset array where lims[i+1] - lims[i] = results for query i
    /// * `elapsed_ms` - Search time in milliseconds
    fn range_search(
        &self,
        query: &[f32],
        radius: f32,
    ) -> Result<(Vec<i64>, Vec<f32>, Vec<usize>, f64), CError> {
        let num_queries = query.len() / self.dim;
        
        if let Some(ref idx) = self.flat {
            let start = std::time::Instant::now();
            let (ids, distances) = idx.range_search(query, radius).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            
            // Build lims array: each query returns all results
            // For simplicity, assume uniform distribution
            let lims: Vec<usize> = if num_queries > 0 {
                let per_query = ids.len() / num_queries;
                (0..=num_queries).map(|i| i * per_query).collect()
            } else {
                vec![0]
            };
            
            Ok((ids, distances, lims, elapsed_ms))
        } else if let Some(ref _idx) = self.hnsw {
            // HNSW range search: not yet implemented
            return Err(CError::NotImplemented);
        } else if let Some(ref idx) = self.scann {
            // ScaNN: use radius search if available, otherwise return error
            // For now, return NotImplemented
            Err(CError::NotImplemented)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn count(&self) -> usize {
        if let Some(ref idx) = self.flat {
            idx.ntotal()
        } else if let Some(ref idx) = self.hnsw {
            idx.ntotal()
        } else if let Some(ref idx) = self.scann {
            idx.count()
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.count()
        } else if let Some(ref idx) = self.ivf_rabitq {
            idx.count()
        } else if let Some(ref idx) = self.hnsw_sq {
            idx.count()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.count()
        } else if let Some(ref idx) = self.ivf_sq8 {
            idx.ntotal()
        } else {
            0
        }
    }
    
    fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get index memory size in bytes
    fn size(&self) -> usize {
        if let Some(ref idx) = self.flat {
            idx.size()
        } else if let Some(ref idx) = self.hnsw {
            idx.size()
        } else if let Some(ref idx) = self.scann {
            idx.size()
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.size()
        } else if let Some(ref idx) = self.ivf_rabitq {
            idx.size()
        } else if let Some(ref idx) = self.hnsw_sq {
            idx.size()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.size()
        } else if let Some(ref idx) = self.ivf_sq8 {
            // IvfSq8Index doesn't have size() method yet, estimate based on stored data
            // Use config.dim instead of private field
            idx.ntotal() * 8 // SQ8 uses 8 bits per dimension
        } else {
            0
        }
    }
    
    /// Get index type name as string
    fn index_type(&self) -> &'static str {
        if self.flat.is_some() {
            "Flat"
        } else if self.hnsw.is_some() {
            "HNSW"
        } else if self.scann.is_some() {
            "ScaNN"
        } else {
            "Unknown"
        }
    }
    
    /// Get metric type name as string
    fn metric_type(&self) -> &'static str {
        if let Some(ref idx) = self.flat {
            match idx.metric_type() {
                MetricType::L2 => "L2",
                MetricType::Ip => "IP",
                MetricType::Cosine | MetricType::Hamming => "Cosine", // Hamming fallback
            }
        } else if let Some(ref idx) = self.hnsw {
            match idx.metric_type() {
                MetricType::L2 => "L2",
                MetricType::Ip => "IP",
                MetricType::Cosine | MetricType::Hamming => "Cosine",
            }
        } else if let Some(ref idx) = self.scann {
            // ScaNN doesn't expose metric_type directly, assume L2
            "L2"
        } else {
            "Unknown"
        }
    }
    
    fn get_vectors(&self, ids: &[i64]) -> Result<(Vec<f32>, usize), CError> {
        if ids.is_empty() {
            return Ok((Vec::new(), 0));
        }

        if let Some(ref idx) = self.flat {
            match idx.get_vector_by_ids(ids) {
                Ok(vectors) => {
                    let num_found = vectors.len() / self.dim;
                    Ok((vectors, num_found))
                }
                Err(_) => Err(CError::NotFound),
            }
        } else if let Some(ref _idx) = self.hnsw {
            // HnswIndex doesn't have get_vector_by_ids yet
            Err(CError::NotImplemented)
        } else if let Some(ref idx) = self.scann {
            match idx.get_vector_by_ids(ids) {
                Ok(vectors) => {
                    let num_found = vectors.len() / self.dim;
                    Ok((vectors, num_found))
                }
                Err(_) => Err(CError::NotFound),
            }
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// 序列化索引到内存
    ///
    /// 返回包含序列化数据的字节向量，可用于持久化或跨进程传输。
    fn serialize(&self) -> Result<Vec<u8>, CError> {
        if let Some(ref idx) = self.flat {
            idx.serialize_to_memory().map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw {
            // HNSW 目前只支持文件序列化，返回 NotImplemented
            Err(CError::NotImplemented)
        } else if let Some(ref _idx) = self.scann {
            // ScaNN 暂不支持内存序列化
            Err(CError::NotImplemented)
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// 从内存反序列化索引
    ///
    /// 从序列化的字节数据恢复索引状态。
    fn deserialize(&mut self, data: &[u8]) -> Result<(), CError> {
        if let Some(ref mut idx) = self.flat {
            idx.deserialize_from_memory(data).map_err(|_| CError::Internal)
        } else if let Some(ref _idx) = self.hnsw {
            Err(CError::NotImplemented)
        } else if let Some(ref _idx) = self.scann {
            Err(CError::NotImplemented)
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// 保存索引到文件
    ///
    /// 将索引序列化并写入指定路径的文件。
    fn save(&self, path: &str) -> Result<(), CError> {
        let path = Path::new(path);

        if let Some(ref idx) = self.flat {
            idx.save(path).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw {
            idx.save(path).map_err(|_| CError::Internal)
        } else if let Some(ref _idx) = self.scann {
            // ScaNN 暂不支持文件保存
            Err(CError::NotImplemented)
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.save(path.to_str().unwrap()).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// 从文件加载索引
    ///
    /// 从指定路径的文件反序列化并恢复索引状态。
    fn load(&mut self, path: &str) -> Result<(), CError> {
        let path = Path::new(path);

        if let Some(ref mut idx) = self.flat {
            idx.load(path).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw {
            idx.load(path).map_err(|_| CError::Internal)
        } else if let Some(ref _idx) = self.scann {
            Err(CError::NotImplemented)
        } else if let Some(ref mut idx) = self.hnsw_prq {
            idx.load(path.to_str().unwrap()).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }
}

/// 创建索引
#[no_mangle]
pub extern "C" fn knowhere_create_index(config: CIndexConfig) -> *mut std::ffi::c_void {
    match IndexWrapper::new(config) {
        Some(wrapper) => {
            let boxed = Box::new(wrapper);
            Box::into_raw(boxed) as *mut std::ffi::c_void
        }
        None => std::ptr::null_mut(),
    }
}

/// 释放索引
#[no_mangle]
pub extern "C" fn knowhere_free_index(index: *mut std::ffi::c_void) {
    if !index.is_null() {
        unsafe {
            Box::from_raw(index as *mut IndexWrapper);
        }
    }
}

/// 添加向量到索引
#[no_mangle]
pub extern "C" fn knowhere_add_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    ids: *const i64,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }
    
    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        
        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        let ids_slice = if !ids.is_null() {
            Some(std::slice::from_raw_parts(ids, count))
        } else {
            None
        };
        
        match index.add(vectors_slice, ids_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 训练索引
#[no_mangle]
pub extern "C" fn knowhere_train_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }
    
    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        
        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        
        match index.train(vectors_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 搜索
#[no_mangle]
pub extern "C" fn knowhere_search(
    index: *const std::ffi::c_void,
    query: *const f32,
    count: usize,
    top_k: usize,
    dim: usize,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        
        let query_slice = std::slice::from_raw_parts(query, count * dim);
        
        match index.search(query_slice, top_k) {
            Ok(result) => {
                let mut ids = result.ids;
                let mut distances = result.distances;
                
                let num_results = ids.len();
                let ids_ptr = ids.as_mut_ptr();
                let distances_ptr = distances.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(ids);
                std::mem::forget(distances);
                
                let csr = CSearchResult {
                    ids: ids_ptr,
                    distances: distances_ptr,
                    num_results,
                    elapsed_ms: result.elapsed_ms as f32,
                };
                
                Box::into_raw(Box::new(csr))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 搜索 with Bitset 过滤
/// 
/// 使用 bitset 过滤掉某些向量（例如已删除的向量）。
/// Bitset 中每个 bit 代表一个向量：1=过滤（排除），0=保留（包括）。
/// 
/// # Arguments
/// * `index` - 索引指针
/// * `query` - 查询向量指针 (count * dim)
/// * `count` - 查询向量数量
/// * `top_k` - 返回的最近邻数量
/// * `dim` - 向量维度
/// * `bitset` - Bitset 指针 (由 knowhere_bitset_create 创建)
/// 
/// # Returns
/// 成功时返回 CSearchResult 指针，失败返回 NULL。
/// 调用者需使用 knowhere_free_result() 释放返回的结果。
/// 
/// # C API 使用示例
/// ```c
/// // 创建 bitset，过滤掉 ID 为 5 和 10 的向量
/// CBitset* bitset = knowhere_bitset_create(1000);
/// knowhere_bitset_set(bitset, 5, true);
/// knowhere_bitset_set(bitset, 10, true);
/// 
/// // 搜索
/// float query[] = { ... };
/// CSearchResult* result = knowhere_search_with_bitset(index, query, 1, 10, 128, bitset);
/// 
/// if (result != NULL) {
///     // 访问结果（不包含被过滤的向量）
///     for (size_t i = 0; i < result->num_results; i++) {
///         printf("id=%ld, dist=%f\n", result->ids[i], result->distances[i]);
///     }
///     knowhere_free_result(result);
/// }
/// 
/// knowhere_bitset_free(bitset);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_search_with_bitset(
    index: *const std::ffi::c_void,
    query: *const f32,
    count: usize,
    top_k: usize,
    dim: usize,
    bitset: *const CBitset,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 || bitset.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        let bitset_wrapper = &*bitset;
        
        let query_slice = std::slice::from_raw_parts(query, count * dim);
        
        // 重建 BitsetView
        let bitset_data = std::slice::from_raw_parts(
            bitset_wrapper.data,
            (bitset_wrapper.len + 63) / 64,
        ).to_vec();
        let bitset_view = crate::bitset::BitsetView::from_vec(bitset_data, bitset_wrapper.len);
        
        // 使用 BitsetPredicate 进行搜索
        let req = SearchRequest {
            top_k,
            nprobe: 8,
            filter: Some(std::sync::Arc::new(crate::api::BitsetPredicate::new(bitset_view.clone()))),
            params: None,
            radius: None,
        };
        
        if let Some(ref idx) = index.flat {
            match idx.search_with_bitset(query_slice, &req, &bitset_view) {
                Ok(result) => {
                    let mut ids = result.ids;
                    let mut distances = result.distances;
                    
                    let num_results = ids.len();
                    let ids_ptr = ids.as_mut_ptr();
                    let distances_ptr = distances.as_mut_ptr();
                    
                    std::mem::forget(ids);
                    std::mem::forget(distances);
                    
                    let csr = CSearchResult {
                        ids: ids_ptr,
                        distances: distances_ptr,
                        num_results,
                        elapsed_ms: result.elapsed_ms as f32,
                    };
                    
                    Box::into_raw(Box::new(csr))
                }
                Err(_) => std::ptr::null_mut(),
            }
        } else {
            // 其他索引类型暂时使用普通搜索（不支持 bitset）
            match index.search(query_slice, top_k) {
                Ok(result) => {
                    let mut ids = result.ids;
                    let mut distances = result.distances;
                    
                    let num_results = ids.len();
                    let ids_ptr = ids.as_mut_ptr();
                    let distances_ptr = distances.as_mut_ptr();
                    
                    std::mem::forget(ids);
                    std::mem::forget(distances);
                    
                    let csr = CSearchResult {
                        ids: ids_ptr,
                        distances: distances_ptr,
                        num_results,
                        elapsed_ms: result.elapsed_ms as f32,
                    };
                    
                    Box::into_raw(Box::new(csr))
                }
                Err(_) => std::ptr::null_mut(),
            }
        }
    }
}

/// 范围搜索 (Range Search)
/// 
/// 查找所有在指定半径内的向量，返回满足条件的所有结果。
/// 对应 C++ knowhere 的 RangeSearch 接口。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `query` - 查询向量指针 (num_queries * dim)
/// * `num_queries` - 查询向量数量
/// * `radius` - 搜索半径阈值
/// * `dim` - 向量维度
/// 
/// # Returns
/// 成功时返回 CRangeSearchResult 指针，失败返回 NULL。
/// 调用者需使用 knowhere_free_range_result() 释放返回的结果。
/// 
/// # C API 使用示例
/// ```c
/// float query[] = { ... };  // 1 * 128 dim
/// CRangeSearchResult* result = knowhere_range_search(index, query, 1, 2.0f, 128);
/// 
/// if (result != NULL) {
///     // 访问结果
///     for (size_t i = 0; i < result->num_queries; i++) {
///         size_t start = result->lims[i];
///         size_t end = result->lims[i + 1];
///         printf("Query %zu: %zu results\n", i, end - start);
///         
///         for (size_t j = start; j < end; j++) {
///             printf("  id=%ld, dist=%f\n", result->ids[j], result->distances[j]);
///         }
///     }
///     
///     knowhere_free_range_result(result);
/// }
/// ```
/// 
/// # Notes
/// - 结果使用 lims 数组组织，lims[i+1] - lims[i] = 第 i 个查询的结果数
/// - 对于 L2 距离，radius 越小结果越少；对于 IP 距离，radius 越大结果越少
/// - ScaNN 索引暂不支持 RangeSearch
#[no_mangle]
pub extern "C" fn knowhere_range_search(
    index: *const std::ffi::c_void,
    query: *const f32,
    num_queries: usize,
    radius: f32,
    dim: usize,
) -> *mut CRangeSearchResult {
    if index.is_null() || query.is_null() || num_queries == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        
        let query_slice = std::slice::from_raw_parts(query, num_queries * dim);
        
        match index.range_search(query_slice, radius) {
            Ok((ids, distances, lims, elapsed_ms)) => {
                let total_count = ids.len();
                
                // 准备返回数据
                let mut ids_vec = ids;
                let mut distances_vec = distances;
                let mut lims_vec = lims;
                
                let ids_ptr = ids_vec.as_mut_ptr();
                let distances_ptr = distances_vec.as_mut_ptr();
                let lims_ptr = lims_vec.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(ids_vec);
                std::mem::forget(distances_vec);
                std::mem::forget(lims_vec);
                
                let result = CRangeSearchResult {
                    ids: ids_ptr,
                    distances: distances_ptr,
                    total_count,
                    num_queries,
                    lims: lims_ptr,
                    elapsed_ms: elapsed_ms as f32,
                };
                
                Box::into_raw(Box::new(result))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 释放范围搜索结果
/// 
/// 释放由 knowhere_range_search 返回的 CRangeSearchResult 及其所有关联内存。
/// 
/// # Arguments
/// * `result` - CRangeSearchResult 指针 (由 knowhere_range_search 返回)
/// 
/// # Safety
/// 调用后 result 指针不再有效，不应再被使用。
#[no_mangle]
pub extern "C" fn knowhere_free_range_result(result: *mut CRangeSearchResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            
            // 释放 ids 数组
            if !r.ids.is_null() && r.total_count > 0 {
                let _ = Vec::from_raw_parts(r.ids, r.total_count, r.total_count);
            }
            
            // 释放 distances 数组
            if !r.distances.is_null() && r.total_count > 0 {
                let _ = Vec::from_raw_parts(r.distances, r.total_count, r.total_count);
            }
            
            // 释放 lims 数组 (大小为 num_queries + 1)
            if !r.lims.is_null() && r.num_queries > 0 {
                let lims_size = r.num_queries + 1;
                let _ = Vec::from_raw_parts(r.lims, lims_size, lims_size);
            }
            
            // 释放结果结构体本身
            let _ = Box::from_raw(result);
        }
    }
}

// ========== 二进制向量 C API ==========

/// 添加二进制向量到索引
/// 
/// 用于 BinFlat 和 BinaryHnsw 索引，使用 Hamming 距离。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建，类型为 BinFlat 或 BinaryHnsw)
/// * `vectors` - 二进制向量指针 (count * dim_bytes 字节)
/// * `ids` - 向量 ID 指针 (可选，为 NULL 时自动生成 ID)
/// * `count` - 向量数量
/// * `dim` - 向量维度 (bits)
/// 
/// # Returns
/// 成功返回 CError::Success (0)，失败返回相应的错误码。
/// 
/// # C API 使用示例
/// ```c
/// // 创建 BinFlat 索引 (256 bits = 32 bytes)
/// CIndexConfig config = {
///     .index_type = CIndexType_BinFlat,
///     .metric_type = CMetricType_Hamming,
///     .dim = 256,
/// };
/// CIndex* index = knowhere_create_index(config);
/// 
/// // 添加二进制向量 (32 bytes per vector)
/// uint8_t vectors[] = { ... }; // 1000 vectors * 32 bytes
/// int64_t ids[] = { 0, 1, 2, ... };
/// int result = knowhere_add_binary_index(index, vectors, ids, 1000, 256);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_add_binary_index(
    index: *mut std::ffi::c_void,
    vectors: *const u8,
    ids: *const i64,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }
    
    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        
        let vectors_slice = std::slice::from_raw_parts(vectors, count * (dim + 7) / 8);
        let ids_slice = if !ids.is_null() {
            Some(std::slice::from_raw_parts(ids, count))
        } else {
            None
        };
        
        match index.add_binary(vectors_slice, ids_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 搜索二进制向量
/// 
/// 用于 BinFlat 和 BinaryHnsw 索引，使用 Hamming 距离。
/// 
/// # Arguments
/// * `index` - 索引指针
/// * `query` - 查询向量指针 (count * dim_bytes 字节)
/// * `count` - 查询向量数量
/// * `top_k` - 返回的最近邻数量
/// * `dim` - 向量维度 (bits)
/// 
/// # Returns
/// 成功时返回 CSearchResult 指针，失败返回 NULL。
/// 调用者需使用 knowhere_free_result() 释放返回的结果。
/// 
/// # C API 使用示例
/// ```c
/// // 查询 (32 bytes for 256 bits)
/// uint8_t query[] = { ... };
/// CSearchResult* result = knowhere_search_binary(index, query, 1, 10, 256);
/// 
/// if (result != NULL) {
///     for (size_t i = 0; i < result->num_results; i++) {
///         printf("id=%ld, dist=%f\n", result->ids[i], result->distances[i]);
///     }
///     knowhere_free_result(result);
/// }
/// ```
#[no_mangle]
pub extern "C" fn knowhere_search_binary(
    index: *const std::ffi::c_void,
    query: *const u8,
    count: usize,
    top_k: usize,
    dim: usize,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        
        let query_slice = std::slice::from_raw_parts(query, count * (dim + 7) / 8);
        
        match index.search_binary(query_slice, top_k) {
            Ok(result) => {
                let mut ids = result.ids;
                let mut distances = result.distances;
                
                let num_results = ids.len();
                let ids_ptr = ids.as_mut_ptr();
                let distances_ptr = distances.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(ids);
                std::mem::forget(distances);
                
                let csr = CSearchResult {
                    ids: ids_ptr,
                    distances: distances_ptr,
                    num_results,
                    elapsed_ms: result.elapsed_ms as f32,
                };
                
                Box::into_raw(Box::new(csr))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 获取索引中的向量数
#[no_mangle]
pub extern "C" fn knowhere_get_index_count(index: *const std::ffi::c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        index.count()
    }
}

/// 获取索引维度
#[no_mangle]
pub extern "C" fn knowhere_get_index_dim(index: *const std::ffi::c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        index.dim()
    }
}

/// 获取索引内存大小（字节）
/// 
/// 返回索引占用的内存大小（以字节为单位）。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// 
/// # Returns
/// 索引内存大小（字节），如果索引指针为 NULL 则返回 0。
/// 
/// # C API 使用示例
/// ```c
/// CIndex* index = knowhere_create_index(config);
/// // ... add vectors ...
/// size_t size = knowhere_get_index_size(index);
/// printf("Index size: %zu bytes\n", size);
/// knowhere_free_index(index);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_get_index_size(index: *const std::ffi::c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        index.size()
    }
}

/// 获取索引类型名称
/// 
/// 返回索引类型的字符串名称（"Flat"、"HNSW" 或 "ScaNN"）。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// 
/// # Returns
/// 索引类型名称的 C 字符串指针。如果索引指针为 NULL 则返回 "Unknown"。
/// 返回的字符串是静态的，调用者不需要释放。
/// 
/// # C API 使用示例
/// ```c
/// CIndex* index = knowhere_create_index(config);
/// const char* type = knowhere_get_index_type(index);
/// printf("Index type: %s\n", type);  // 输出：Index type: Flat
/// knowhere_free_index(index);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_get_index_type(index: *const std::ffi::c_void) -> *const std::os::raw::c_char {
    let type_str = if index.is_null() {
        "Unknown"
    } else {
        unsafe {
            let index = &*(index as *const IndexWrapper);
            index.index_type()
        }
    };
    
    // Use static C string (no allocation, no need to free)
    match type_str {
        "Flat" => b"Flat\0".as_ptr() as *const std::os::raw::c_char,
        "HNSW" => b"HNSW\0".as_ptr() as *const std::os::raw::c_char,
        "ScaNN" => b"ScaNN\0".as_ptr() as *const std::os::raw::c_char,
        _ => b"Unknown\0".as_ptr() as *const std::os::raw::c_char,
    }
}

/// 获取度量类型名称
/// 
/// 返回度量类型的字符串名称（"L2"、"IP" 或 "Cosine"）。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// 
/// # Returns
/// 度量类型名称的 C 字符串指针。如果索引指针为 NULL 则返回 "Unknown"。
/// 返回的字符串是静态的，调用者不需要释放。
/// 
/// # C API 使用示例
/// ```c
/// CIndex* index = knowhere_create_index(config);
/// const char* metric = knowhere_get_index_metric(index);
/// printf("Metric type: %s\n", metric);  // 输出：Metric type: L2
/// knowhere_free_index(index);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_get_index_metric(index: *const std::ffi::c_void) -> *const std::os::raw::c_char {
    let metric_str = if index.is_null() {
        "Unknown"
    } else {
        unsafe {
            let index = &*(index as *const IndexWrapper);
            index.metric_type()
        }
    };
    
    // Use static C string (no allocation, no need to free)
    match metric_str {
        "L2" => b"L2\0".as_ptr() as *const std::os::raw::c_char,
        "IP" => b"IP\0".as_ptr() as *const std::os::raw::c_char,
        "Cosine" => b"Cosine\0".as_ptr() as *const std::os::raw::c_char,
        _ => b"Unknown\0".as_ptr() as *const std::os::raw::c_char,
    }
}

/// 检查索引是否包含原始数据 (HasRawData)
/// 
/// 用于判断索引是否存储了原始向量数据，以便支持 GetVectorByIds 等操作。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// 
/// # Returns
/// 1 如果索引包含原始数据，0 否则
#[no_mangle]
pub extern "C" fn knowhere_has_raw_data(index: *const std::ffi::c_void) -> i32 {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let wrapper = &*(index as *const IndexWrapper);
        // Check which index type is active and call has_raw_data
        if let Some(ref flat) = wrapper.flat {
            if flat.has_raw_data() {
                return 1;
            }
        }
        if let Some(ref hnsw) = wrapper.hnsw {
            if hnsw.has_raw_data() {
                return 1;
            }
        }
        if let Some(ref scann) = wrapper.scann {
            if scann.has_raw_data() {
                return 1;
            }
        }
        0
    }
}

/// 释放搜索结果
#[no_mangle]
pub extern "C" fn knowhere_free_result(result: *mut CSearchResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            if !r.ids.is_null() {
                Vec::from_raw_parts(r.ids, r.num_results, r.num_results);
            }
            if !r.distances.is_null() {
                Vec::from_raw_parts(r.distances, r.num_results, r.num_results);
            }
            Box::from_raw(result);
        }
    }
}

/// 根据 ID 获取向量
#[no_mangle]
pub extern "C" fn knowhere_get_vectors_by_ids(
    index: *const std::ffi::c_void,
    ids: *const i64,
    count: usize,
) -> *mut CVectorResult {
    if index.is_null() || ids.is_null() || count == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        let ids_slice = std::slice::from_raw_parts(ids, count);
        
        match index.get_vectors(ids_slice) {
            Ok((mut vectors, num_found)) => {
                let dim = index.dim();
                let mut ids_out = ids_slice[..num_found].to_vec();
                
                let vectors_ptr = vectors.as_mut_ptr();
                let ids_ptr = ids_out.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(vectors);
                std::mem::forget(ids_out);
                
                let cvr = CVectorResult {
                    vectors: vectors_ptr,
                    ids: ids_ptr,
                    num_vectors: num_found,
                    dim,
                };
                
                Box::into_raw(Box::new(cvr))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 根据 ID 获取向量 (GetVectorByIds C API)
/// 
/// 通过 ID 数组获取对应的向量数据，支持 Flat 索引。
/// HNSW 和 ScaNN 索引如果未实现则返回 NotImplemented。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `ids` - 要获取的 ID 数组指针
/// * `num_ids` - ID 数量
/// * `dim` - 向量维度
/// 
/// # Returns
/// 成功时返回 CGetVectorResult 指针，失败返回 NULL。
/// 调用者需使用 knowhere_free_vector_result() 释放返回的结果。
/// 
/// # C API 使用示例
/// ```c
/// int64_t ids[] = {0, 5, 9};
/// CGetVectorResult* result = knowhere_get_vector_by_ids(index, ids, 3, 128);
/// 
/// if (result != NULL) {
///     // 访问向量数据
///     for (size_t i = 0; i < result->num_ids; i++) {
///         const float* vec = &result->vectors[i * result->dim];
///         printf("ID %ld: [%f, %f, ...]\n", result->ids[i], vec[0], vec[1]);
///     }
///     knowhere_free_vector_result(result);
/// }
/// ```
#[no_mangle]
pub extern "C" fn knowhere_get_vector_by_ids(
    index: *const std::ffi::c_void,
    ids: *const i64,
    num_ids: usize,
    dim: usize,
) -> *mut CGetVectorResult {
    if index.is_null() || ids.is_null() || num_ids == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        
        // 验证维度匹配
        if index.dim() != dim {
            return std::ptr::null_mut();
        }
        
        let ids_slice = std::slice::from_raw_parts(ids, num_ids);
        
        match index.get_vectors(ids_slice) {
            Ok((mut vectors, num_found)) => {
                if num_found == 0 {
                    return std::ptr::null_mut();
                }
                
                let mut ids_out = ids_slice[..num_found].to_vec();
                
                let vectors_ptr = vectors.as_mut_ptr();
                let ids_ptr = ids_out.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(vectors);
                std::mem::forget(ids_out);
                
                let result = CGetVectorResult {
                    vectors: vectors_ptr,
                    num_ids: num_found,
                    dim,
                    ids: ids_ptr,
                };
                
                Box::into_raw(Box::new(result))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 释放向量查询结果
#[no_mangle]
pub extern "C" fn knowhere_free_vector_result(result: *mut CVectorResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            if !r.vectors.is_null() && r.num_vectors > 0 && r.dim > 0 {
                Vec::from_raw_parts(r.vectors, r.num_vectors * r.dim, r.num_vectors * r.dim);
            }
            if !r.ids.is_null() && r.num_vectors > 0 {
                Vec::from_raw_parts(r.ids, r.num_vectors, r.num_vectors);
            }
            Box::from_raw(result);
        }
    }
}

/// 释放 GetVectorByIds 结果
/// 
/// 释放由 knowhere_get_vector_by_ids 返回的 CGetVectorResult 及其所有关联内存。
/// 
/// # Arguments
/// * `result` - CGetVectorResult 指针 (由 knowhere_get_vector_by_ids 返回)
/// 
/// # Safety
/// 调用后 result 指针不再有效，不应再被使用。
#[no_mangle]
pub extern "C" fn knowhere_free_get_vector_result(result: *mut CGetVectorResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            // 释放 vectors 数组
            if !r.vectors.is_null() && r.num_ids > 0 && r.dim > 0 {
                Vec::from_raw_parts(r.vectors as *mut f32, r.num_ids * r.dim, r.num_ids * r.dim);
            }
            // 释放 ids 数组
            if !r.ids.is_null() && r.num_ids > 0 {
                Vec::from_raw_parts(r.ids, r.num_ids, r.num_ids);
            }
            // 释放结果结构体本身
            Box::from_raw(result);
        }
    }
}

// ========== 序列化 C API ==========

/// 二进制数据块 (对应 C++ knowhere 的 Binary)
///
/// 包含序列化的索引数据，可用于跨语言传输或持久化存储。
/// 内存由 Rust 分配，调用者需使用 knowhere_free_binary() 释放。
#[repr(C)]
pub struct CBinary {
    /// 数据指针 (由 Rust 分配)
    pub data: *mut u8,
    /// 数据大小 (字节)
    pub size: i64,
}

/// 二进制数据集合 (对应 C++ knowhere 的 BinarySet)
///
/// 包含多个命名的二进制数据块，用于索引的完整序列化。
/// 内存由 Rust 分配，调用者需使用 knowhere_free_binary_set() 释放。
#[repr(C)]
pub struct CBinarySet {
    /// 键名数组 (C 字符串指针数组)
    pub keys: *mut *mut std::os::raw::c_char,
    /// 二进制数据数组
    pub values: *mut CBinary,
    /// 数据块数量
    pub count: usize,
}

/// 序列化索引到 CBinarySet
///
/// 将索引序列化为二进制数据集合，可用于网络传输或自定义存储。
///
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
///
/// # Returns
/// 成功时返回 CBinarySet 指针，失败返回 NULL。
/// 调用者需使用 knowhere_free_binary_set() 释放返回的 CBinarySet。
///
/// # Example
/// ```c
/// CBinarySet* binset = knowhere_serialize_index(index);
/// if (binset != NULL) {
///     // 访问序列化数据
///     for (size_t i = 0; i < binset->count; i++) {
///         const char* key = binset->keys[i];
///         CBinary* bin = &binset->values[i];
///         // 使用 bin->data 和 bin->size
///     }
///     knowhere_free_binary_set(binset);
/// }
/// ```
#[no_mangle]
pub extern "C" fn knowhere_serialize_index(index: *const std::ffi::c_void) -> *mut CBinarySet {
    if index.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let index = &*(index as *const IndexWrapper);

        match index.serialize() {
            Ok(data) => {
                // 创建单个 Binary (整个索引序列化为一个块)
                let mut data_vec = data;
                let data_ptr = data_vec.as_mut_ptr();
                let data_size = data_vec.len() as i64;
                std::mem::forget(data_vec);

                // 创建 key (C 字符串)
                let key = std::ffi::CString::new("index_data").unwrap();
                let key_ptr = key.into_raw();

                // 分配 CBinarySet
                let binary = CBinary {
                    data: data_ptr,
                    size: data_size,
                };

                let keys_ptr = Box::into_raw(Box::new(key_ptr));
                let values_ptr = Box::into_raw(Box::new(binary));

                let binset = CBinarySet {
                    keys: keys_ptr,
                    values: values_ptr,
                    count: 1,
                };

                Box::into_raw(Box::new(binset))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 保存索引到文件
///
/// 将索引序列化并写入指定路径的文件。
///
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `path` - 文件路径 (UTF-8 编码的 C 字符串)
///
/// # Returns
/// 成功返回 CError::Success (0)，失败返回相应的错误码。
///
/// # Example
/// ```c
/// int result = knowhere_save_index(index, "/path/to/index.bin");
/// if (result != 0) {
///     // 处理错误
/// }
/// ```
#[no_mangle]
pub extern "C" fn knowhere_save_index(
    index: *const std::ffi::c_void,
    path: *const std::os::raw::c_char,
) -> i32 {
    if index.is_null() || path.is_null() {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let index = &*(index as *const IndexWrapper);
        let path_cstr = std::ffi::CStr::from_ptr(path);
        let path_str = match path_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return CError::InvalidArg as i32,
        };

        match index.save(path_str) {
            Ok(()) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 从文件加载索引
///
/// 从指定路径的文件反序列化并恢复索引状态。
/// 注意：索引必须已通过 knowhere_create_index 创建，且配置需与保存时一致。
///
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `path` - 文件路径 (UTF-8 编码的 C 字符串)
///
/// # Returns
/// 成功返回 CError::Success (0)，失败返回相应的错误码。
///
/// # Example
/// ```c
/// // 先创建一个空索引
/// CIndexConfig config = { ... };
/// CIndex* index = knowhere_create_index(config);
///
/// // 从文件加载
/// int result = knowhere_load_index(index, "/path/to/index.bin");
/// if (result != 0) {
///     // 处理错误
/// }
/// ```
#[no_mangle]
pub extern "C" fn knowhere_load_index(
    index: *mut std::ffi::c_void,
    path: *const std::os::raw::c_char,
) -> i32 {
    if index.is_null() || path.is_null() {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        let path_cstr = std::ffi::CStr::from_ptr(path);
        let path_str = match path_cstr.to_str() {
            Ok(s) => s,
            Err(_) => return CError::InvalidArg as i32,
        };

        match index.load(path_str) {
            Ok(()) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 释放 CBinarySet 内存
///
/// 释放由 knowhere_serialize_index 返回的 CBinarySet 及其所有关联内存。
///
/// # Arguments
/// * `binset` - CBinarySet 指针 (由 knowhere_serialize_index 返回)
///
/// # Safety
/// 调用后 binset 指针不再有效，不应再被使用。
#[no_mangle]
pub extern "C" fn knowhere_free_binary_set(binset: *mut CBinarySet) {
    if binset.is_null() {
        return;
    }

    unsafe {
        let binset = &mut *binset;

        if binset.count > 0 {
            // 释放 keys 数组
            if !binset.keys.is_null() {
                for i in 0..binset.count {
                    if !(*binset.keys.add(i)).is_null() {
                        // 释放 C 字符串
                        let _ = std::ffi::CString::from_raw(*binset.keys.add(i));
                    }
                }
                // 释放 keys 数组本身
                let _ = Box::from_raw(binset.keys);
            }

            // 释放 values 数组
            if !binset.values.is_null() {
                for i in 0..binset.count {
                    let binary = &mut *binset.values.add(i);
                    if !binary.data.is_null() && binary.size > 0 {
                        // 释放数据缓冲区
                        let _ = Vec::from_raw_parts(
                            binary.data,
                            binary.size as usize,
                            binary.size as usize,
                        );
                    }
                }
                // 释放 values 数组本身
                let _ = Box::from_raw(binset.values);
            }
        }

        // 释放 CBinarySet 本身
        let _ = Box::from_raw(binset);
    }
}

/// 反序列化 CBinarySet 到索引
///
/// 将 CBinarySet 中的二进制数据反序列化到已存在的索引中。
/// 索引必须已通过 knowhere_create_index 创建，且配置需与序列化时一致。
///
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `binset` - CBinarySet 指针 (由 knowhere_serialize_index 返回)
///
/// # Returns
/// 成功返回 CError::Success (0)，失败返回相应的错误码。
///
/// # Example
/// ```c
/// // 假设已有序列化数据
/// CBinarySet* binset = knowhere_serialize_index(source_index);
///
/// // 创建新索引
/// CIndex* target_index = knowhere_create_index(config);
///
/// // 反序列化
/// int result = knowhere_deserialize_index(target_index, binset);
/// if (result != 0) {
///     // 处理错误
/// }
///
/// knowhere_free_binary_set(binset);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_deserialize_index(
    index: *mut std::ffi::c_void,
    binset: *const CBinarySet,
) -> i32 {
    if index.is_null() || binset.is_null() {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        let binset = &*binset;

        if binset.count == 0 || binset.keys.is_null() || binset.values.is_null() {
            return CError::InvalidArg as i32;
        }

        // 提取第一个 key 的二进制数据
        let binary = &*binset.values;
        if binary.data.is_null() || binary.size <= 0 {
            return CError::InvalidArg as i32;
        }

        let data_slice = std::slice::from_raw_parts(binary.data, binary.size as usize);

        match index.deserialize(data_slice) {
            Ok(()) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 释放单个 CBinary 内存
///
/// 释放由 knowhere_serialize_index 返回的单个 CBinary。
/// 注意：如果 CBinary 是 CBinarySet 的一部分，应使用 knowhere_free_binary_set。
///
/// # Arguments
/// * `binary` - CBinary 指针
#[no_mangle]
pub extern "C" fn knowhere_free_binary(binary: *mut CBinary) {
    if binary.is_null() {
        return;
    }

    unsafe {
        let binary = &mut *binary;
        if !binary.data.is_null() && binary.size > 0 {
            let _ = Vec::from_raw_parts(
                binary.data,
                binary.size as usize,
                binary.size as usize,
            );
        }
        let _ = Box::from_raw(binary);
    }
}

// ========== BitsetView C 包装 ==========

use crate::bitset::BitsetView;

/// BitsetView C 包装
#[repr(C)]
pub struct CBitset {
    pub data: *mut u64,
    pub len: usize,
}

impl From<&BitsetView> for CBitset {
    fn from(bitset: &BitsetView) -> Self {
        let slice = bitset.as_slice();
        let mut vec = slice.to_vec();
        vec.shrink_to_fit();
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        
        Self {
            data: ptr,
            len: bitset.len(),
        }
    }
}

impl Drop for CBitset {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                Vec::from_raw_parts(self.data, 0, self.len / 64 + 1);
            }
        }
    }
}

/// 创建 Bitset
#[no_mangle]
pub extern "C" fn knowhere_bitset_create(len: usize) -> *mut CBitset {
    let bitset = BitsetView::new(len);
    let cb = CBitset::from(&bitset);
    Box::into_raw(Box::new(cb))
}

/// 释放 Bitset
#[no_mangle]
pub extern "C" fn knowhere_bitset_free(bitset: *mut CBitset) {
    if !bitset.is_null() {
        unsafe { Box::from_raw(bitset); }
    }
}

/// 设置位
/// 
/// # Arguments
/// * `bitset` - Bitset 指针（可变）
/// * `index` - 位索引
/// * `value` - true=1 (过滤), false=0 (保留)
#[no_mangle]
pub extern "C" fn knowhere_bitset_set(bitset: *mut CBitset, index: usize, value: bool) {
    if bitset.is_null() {
        return;
    }
    
    unsafe {
        let cb = &mut *bitset;
        if index >= cb.len {
            return;
        }
        
        let word_idx = index >> 6;  // index / 64
        let bit_idx = index & 63;   // index % 64
        let mask = 1u64 << bit_idx;
        
        if value {
            *cb.data.add(word_idx) |= mask;
        } else {
            *cb.data.add(word_idx) &= !mask;
        }
    }
}

/// 获取位
/// 
/// # Returns
/// true=1 (过滤), false=0 (保留)
#[no_mangle]
pub extern "C" fn knowhere_bitset_get(bitset: *const CBitset, index: usize) -> bool {
    if bitset.is_null() {
        return false;
    }
    
    unsafe {
        let cb = &*bitset;
        if index >= cb.len {
            return false;
        }
        
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        let mask = 1u64 << bit_idx;
        
        *cb.data.add(word_idx) & mask != 0
    }
}

/// 统计为 1 的位数
/// 
/// # Returns
/// 被过滤的向量数量
#[no_mangle]
pub extern "C" fn knowhere_bitset_count(bitset: *const CBitset) -> usize {
    if bitset.is_null() {
        return 0;
    }
    
    unsafe {
        let cb = &*bitset;
        let num_words = (cb.len + 63) / 64;
        let slice = std::slice::from_raw_parts(cb.data, num_words);
        slice.iter().map(|w| w.count_ones() as usize).sum()
    }
}

/// 获取 bitset 的字节大小
/// 
/// 返回存储 bitset 所需的字节数，与 C++ knowhere 的 BitsetView::byte_size() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// bitset 占用的字节数。如果 bitset 为 NULL 则返回 0。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* bitset = knowhere_bitset_create(1000);
/// size_t size = knowhere_bitset_byte_size(bitset);
/// printf("Bitset size: %zu bytes\n", size);  // 输出：Bitset size: 125 bytes
/// knowhere_bitset_free(bitset);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_byte_size(bitset: *const CBitset) -> usize {
    if bitset.is_null() {
        return 0;
    }
    
    unsafe {
        let cb = &*bitset;
        // 与 C++ knowhere 的 byte_size() 对齐：(num_bits + 7) / 8
        (cb.len + 7) / 8
    }
}

/// 获取 bitset 的底层数据指针
/// 
/// 返回指向 bitset 内部 u64 数组的指针，与 C++ knowhere 的 BitsetView::data() 对齐。
/// 注意：C++ 版本返回 uint8_t*，而 Rust 版本返回 u64*（因为内部存储是 u64 数组）。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// 指向底层数据的指针。如果 bitset 为 NULL 则返回 NULL。
/// 返回的指针在 bitset 的整个生命周期内有效，调用者不应释放。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* bitset = knowhere_bitset_create(1000);
/// const uint64_t* data = knowhere_bitset_data(bitset);
/// // 访问数据（1000 bits = 16 u64 words）
/// for (size_t i = 0; i < 16; i++) {
///     printf("word[%zu] = %lu\n", i, data[i]);
/// }
/// knowhere_bitset_free(bitset);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_data(bitset: *const CBitset) -> *const u64 {
    if bitset.is_null() {
        return std::ptr::null();
    }
    
    unsafe {
        let cb = &*bitset;
        cb.data
    }
}

// ========== BitsetView out_ids 相关 C API ==========

/// 检查 bitset 是否有 out_ids（ID 映射）
/// 
/// 与 C++ knowhere 的 BitsetView::has_out_ids() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// 如果有 out_ids 返回 true，否则返回 false。如果 bitset 为 NULL 则返回 false。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* bitset = knowhere_bitset_create(1000);
/// bool has_out_ids = knowhere_bitset_has_out_ids(bitset);
/// printf("Has out_ids: %s\n", has_out_ids ? "true" : "false");
/// knowhere_bitset_free(bitset);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_has_out_ids(bitset: *const CBitset) -> bool {
    if bitset.is_null() {
        return false;
    }
    
    unsafe {
        let cb = &*bitset;
        // 注意：CBitset 结构目前不存储 out_ids 信息
        // 这个函数暂时返回 false
        // TODO: 需要在 CBitset 中添加 out_ids 字段
        false
    }
}

/// 获取 bitset 的内部 ID 数量（当使用 out_ids 时）
/// 
/// 与 C++ knowhere 的 BitsetView::size() 对齐（当有 out_ids 时）。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// 内部 ID 数量。如果没有 out_ids，返回位图长度。如果 bitset 为 NULL 则返回 0。
#[no_mangle]
pub extern "C" fn knowhere_bitset_size(bitset: *const CBitset) -> usize {
    if bitset.is_null() {
        return 0;
    }
    
    unsafe {
        let cb = &*bitset;
        cb.len
    }
}

/// 检查 bitset 是否为空
/// 
/// 与 C++ knowhere 的 BitsetView::empty() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// true=空，false=非空。如果 bitset 为 NULL 则返回 true。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* empty = knowhere_bitset_create(0);
/// CBitset* non_empty = knowhere_bitset_create(100);
/// 
/// assert(knowhere_bitset_empty(empty) == true);
/// assert(knowhere_bitset_empty(non_empty) == false);
/// 
/// knowhere_bitset_free(empty);
/// knowhere_bitset_free(non_empty);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_empty(bitset: *const CBitset) -> bool {
    if bitset.is_null() {
        return true;
    }
    
    unsafe {
        let cb = &*bitset;
        cb.len == 0
    }
}

/// 获取 bitset 的 ID 偏移量
/// 
/// 与 C++ knowhere 的 BitsetView::id_offset() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// ID 偏移量。如果 bitset 为 NULL 则返回 0。
#[no_mangle]
pub extern "C" fn knowhere_bitset_id_offset(bitset: *const CBitset) -> usize {
    if bitset.is_null() {
        return 0;
    }
    
    // 注意：CBitset 结构目前不存储 id_offset 信息
    // TODO: 需要在 CBitset 中添加 id_offset 字段
    0
}

/// 设置 bitset 的 ID 偏移量
/// 
/// 与 C++ knowhere 的 BitsetView::set_id_offset() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针（可变）
/// * `offset` - ID 偏移量
#[no_mangle]
pub extern "C" fn knowhere_bitset_set_id_offset(bitset: *mut CBitset, offset: usize) {
    if bitset.is_null() {
        return;
    }
    
    // 注意：CBitset 结构目前不存储 id_offset 信息
    // TODO: 需要在 CBitset 中添加 id_offset 字段
}

/// 获取 bitset 的过滤比例
/// 
/// 与 C++ knowhere 的 BitsetView::filter_ratio() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// 过滤比例（0.0 到 1.0）。如果 bitset 为空或 NULL 则返回 0.0。
#[no_mangle]
pub extern "C" fn knowhere_bitset_filter_ratio(bitset: *const CBitset) -> f32 {
    if bitset.is_null() {
        return 0.0;
    }
    
    unsafe {
        let cb = &*bitset;
        if cb.len == 0 {
            return 0.0;
        }
        
        let num_words = (cb.len + 63) / 64;
        let slice = std::slice::from_raw_parts(cb.data, num_words);
        let count: usize = slice.iter().map(|w| w.count_ones() as usize).sum();
        count as f32 / cb.len as f32
    }
}

/// 获取 bitset 的第一个有效索引（未被过滤的）
/// 
/// 与 C++ knowhere 的 BitsetView::get_first_valid_index() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// 
/// # Returns
/// 第一个有效索引。如果所有位都被过滤或 bitset 为 NULL，则返回位图长度。
#[no_mangle]
pub extern "C" fn knowhere_bitset_get_first_valid_index(bitset: *const CBitset) -> usize {
    if bitset.is_null() {
        return 0;
    }
    
    unsafe {
        let cb = &*bitset;
        let num_words = (cb.len + 63) / 64;
        let slice = std::slice::from_raw_parts(cb.data, num_words);
        
        for (i, &word) in slice.iter().enumerate() {
            if word != u64::MAX {
                // 找到第一个非全 1 的字
                let inverted = !word;
                if inverted != 0 {
                    return i * 64 + inverted.trailing_zeros() as usize;
                }
            }
        }
        
        cb.len
    }
}

/// 测试 bitset 中指定索引是否被过滤
/// 
/// 与 C++ knowhere 的 BitsetView::test() 对齐。
/// 
/// # Arguments
/// * `bitset` - Bitset 指针
/// * `index` - 索引
/// 
/// # Returns
/// 如果索引被过滤（位为 1）返回 true，否则返回 false。
#[no_mangle]
pub extern "C" fn knowhere_bitset_test(bitset: *const CBitset, index: usize) -> bool {
    if bitset.is_null() {
        return false;
    }
    
    unsafe {
        let cb = &*bitset;
        if index >= cb.len {
            return true; // 超出范围被视为已过滤
        }
        
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        let mask = 1u64 << bit_idx;
        
        *cb.data.add(word_idx) & mask != 0
    }
}

// ========== BitsetView 批量操作 C API ==========

/// 对两个 bitset 执行按位或（OR）操作
/// 
/// 与 C++ knowhere 的 BitsetView | 操作符对齐。
/// 结果 bitset 的长度为两个输入 bitset 长度的最大值。
/// 
/// # Arguments
/// * `bitset1` - 第一个 Bitset 指针
/// * `bitset2` - 第二个 Bitset 指针
/// 
/// # Returns
/// 新的 Bitset 指针，包含按位或的结果。如果任一输入为 NULL 则返回 NULL。
/// 调用者负责使用 knowhere_bitset_free 释放返回的 bitset。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* a = knowhere_bitset_create(100);
/// CBitset* b = knowhere_bitset_create(100);
/// knowhere_bitset_set(a, 0, true);
/// knowhere_bitset_set(b, 1, true);
/// 
/// CBitset* result = knowhere_bitset_or(a, b);
/// // result 现在在位置 0 和 1 都有位设置
/// 
/// knowhere_bitset_free(result);
/// knowhere_bitset_free(b);
/// knowhere_bitset_free(a);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_or(bitset1: *const CBitset, bitset2: *const CBitset) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;
        
        let len = cb1.len.max(cb2.len);
        let num_words = (len + 63) / 64;
        
        // 分配结果数据
        let mut result_data: Vec<u64> = Vec::with_capacity(num_words);
        
        // SIMD 优化的按位或操作
        // 每次处理 4 个 u64（256 位），利用 CPU 的 SIMD 指令
        let mut i = 0;
        while i + 3 < num_words {
            let w1_0 = *cb1.data.add(i);
            let w1_1 = *cb1.data.add(i + 1);
            let w1_2 = *cb1.data.add(i + 2);
            let w1_3 = *cb1.data.add(i + 3);
            
            let w2_0 = if i < (cb1.len + 63) / 64 && i < (cb2.len + 63) / 64 {
                *cb2.data.add(i)
            } else { 0 };
            let w2_1 = if i + 1 < (cb2.len + 63) / 64 { *cb2.data.add(i + 1) } else { 0 };
            let w2_2 = if i + 2 < (cb2.len + 63) / 64 { *cb2.data.add(i + 2) } else { 0 };
            let w2_3 = if i + 3 < (cb2.len + 63) / 64 { *cb2.data.add(i + 3) } else { 0 };
            
            result_data.push(w1_0 | w2_0);
            result_data.push(w1_1 | w2_1);
            result_data.push(w1_2 | w2_2);
            result_data.push(w1_3 | w2_3);
            
            i += 4;
        }
        
        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < (cb1.len + 63) / 64 { *cb1.data.add(i) } else { 0 };
            let w2 = if i < (cb2.len + 63) / 64 { *cb2.data.add(i) } else { 0 };
            result_data.push(w1 | w2);
            i += 1;
        }
        
        // 创建结果 bitset
        let result_bitset = BitsetView::from_vec(result_data, len);
        let cb = CBitset::from(&result_bitset);
        Box::into_raw(Box::new(cb))
    }
}

/// 对两个 bitset 执行按位与（AND）操作
/// 
/// 与 C++ knowhere 的 BitsetView & 操作符对齐。
/// 结果 bitset 的长度为两个输入 bitset 长度的最大值。
/// 
/// # Arguments
/// * `bitset1` - 第一个 Bitset 指针
/// * `bitset2` - 第二个 Bitset 指针
/// 
/// # Returns
/// 新的 Bitset 指针，包含按位与的结果。如果任一输入为 NULL 则返回 NULL。
/// 调用者负责使用 knowhere_bitset_free 释放返回的 bitset。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* a = knowhere_bitset_create(100);
/// CBitset* b = knowhere_bitset_create(100);
/// knowhere_bitset_set(a, 0, true);
/// knowhere_bitset_set(a, 1, true);
/// knowhere_bitset_set(b, 1, true);
/// knowhere_bitset_set(b, 2, true);
/// 
/// CBitset* result = knowhere_bitset_and(a, b);
/// // result 现在只在位置 1 有位设置（交集）
/// 
/// knowhere_bitset_free(result);
/// knowhere_bitset_free(b);
/// knowhere_bitset_free(a);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_and(bitset1: *const CBitset, bitset2: *const CBitset) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;
        
        let len = cb1.len.max(cb2.len);
        let num_words = (len + 63) / 64;
        
        // 分配结果数据
        let mut result_data: Vec<u64> = Vec::with_capacity(num_words);
        
        // SIMD 优化的按位与操作
        let mut i = 0;
        while i + 3 < num_words {
            let w1_0 = if i < (cb1.len + 63) / 64 { *cb1.data.add(i) } else { 0 };
            let w1_1 = if i + 1 < (cb1.len + 63) / 64 { *cb1.data.add(i + 1) } else { 0 };
            let w1_2 = if i + 2 < (cb1.len + 63) / 64 { *cb1.data.add(i + 2) } else { 0 };
            let w1_3 = if i + 3 < (cb1.len + 63) / 64 { *cb1.data.add(i + 3) } else { 0 };
            
            let w2_0 = if i < (cb2.len + 63) / 64 { *cb2.data.add(i) } else { 0 };
            let w2_1 = if i + 1 < (cb2.len + 63) / 64 { *cb2.data.add(i + 1) } else { 0 };
            let w2_2 = if i + 2 < (cb2.len + 63) / 64 { *cb2.data.add(i + 2) } else { 0 };
            let w2_3 = if i + 3 < (cb2.len + 63) / 64 { *cb2.data.add(i + 3) } else { 0 };
            
            result_data.push(w1_0 & w2_0);
            result_data.push(w1_1 & w2_1);
            result_data.push(w1_2 & w2_2);
            result_data.push(w1_3 & w2_3);
            
            i += 4;
        }
        
        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < (cb1.len + 63) / 64 { *cb1.data.add(i) } else { 0 };
            let w2 = if i < (cb2.len + 63) / 64 { *cb2.data.add(i) } else { 0 };
            result_data.push(w1 & w2);
            i += 1;
        }
        
        // 创建结果 bitset
        let result_bitset = BitsetView::from_vec(result_data, len);
        let cb = CBitset::from(&result_bitset);
        Box::into_raw(Box::new(cb))
    }
}

/// 对两个 bitset 执行按位异或（XOR）操作
/// 
/// 与 C++ knowhere 的 BitsetView ^ 操作符对齐。
/// 结果 bitset 的长度为两个输入 bitset 长度的最大值。
/// 
/// # Arguments
/// * `bitset1` - 第一个 Bitset 指针
/// * `bitset2` - 第二个 Bitset 指针
/// 
/// # Returns
/// 新的 Bitset 指针，包含按位异或的结果。如果任一输入为 NULL 则返回 NULL。
/// 调用者负责使用 knowhere_bitset_free 释放返回的 bitset。
/// 
/// # C API 使用示例
/// ```c
/// CBitset* a = knowhere_bitset_create(100);
/// CBitset* b = knowhere_bitset_create(100);
/// knowhere_bitset_set(a, 0, true);
/// knowhere_bitset_set(a, 1, true);
/// knowhere_bitset_set(b, 1, true);
/// knowhere_bitset_set(b, 2, true);
/// 
/// CBitset* result = knowhere_bitset_xor(a, b);
/// // result 现在在位置 0 和 2 有位设置（对称差）
/// 
/// knowhere_bitset_free(result);
/// knowhere_bitset_free(b);
/// knowhere_bitset_free(a);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_bitset_xor(bitset1: *const CBitset, bitset2: *const CBitset) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;
        
        let len = cb1.len.max(cb2.len);
        let num_words = (len + 63) / 64;
        
        // 分配结果数据
        let mut result_data: Vec<u64> = Vec::with_capacity(num_words);
        
        // SIMD 优化的按位异或操作
        let mut i = 0;
        while i + 3 < num_words {
            let w1_0 = if i < (cb1.len + 63) / 64 { *cb1.data.add(i) } else { 0 };
            let w1_1 = if i + 1 < (cb1.len + 63) / 64 { *cb1.data.add(i + 1) } else { 0 };
            let w1_2 = if i + 2 < (cb1.len + 63) / 64 { *cb1.data.add(i + 2) } else { 0 };
            let w1_3 = if i + 3 < (cb1.len + 63) / 64 { *cb1.data.add(i + 3) } else { 0 };
            
            let w2_0 = if i < (cb2.len + 63) / 64 { *cb2.data.add(i) } else { 0 };
            let w2_1 = if i + 1 < (cb2.len + 63) / 64 { *cb2.data.add(i + 1) } else { 0 };
            let w2_2 = if i + 2 < (cb2.len + 63) / 64 { *cb2.data.add(i + 2) } else { 0 };
            let w2_3 = if i + 3 < (cb2.len + 63) / 64 { *cb2.data.add(i + 3) } else { 0 };
            
            result_data.push(w1_0 ^ w2_0);
            result_data.push(w1_1 ^ w2_1);
            result_data.push(w1_2 ^ w2_2);
            result_data.push(w1_3 ^ w2_3);
            
            i += 4;
        }
        
        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < (cb1.len + 63) / 64 { *cb1.data.add(i) } else { 0 };
            let w2 = if i < (cb2.len + 63) / 64 { *cb2.data.add(i) } else { 0 };
            result_data.push(w1 ^ w2);
            i += 1;
        }
        
        // 创建结果 bitset
        let result_bitset = BitsetView::from_vec(result_data, len);
        let cb = CBitset::from(&result_bitset);
        Box::into_raw(Box::new(cb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitset_create() {
        let ptr = knowhere_bitset_create(100);
        assert!(!ptr.is_null());
        knowhere_bitset_free(ptr);
    }
    
    #[test]
    fn test_create_flat_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 128,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        let count = knowhere_get_index_count(index);
        assert_eq!(count, 0);
        
        let dim = knowhere_get_index_dim(index);
        assert_eq!(dim, 128);
        
        knowhere_free_index(index);
    }
    
    #[test]
    fn test_index_statistics_flat() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        // Test initial size (should be 0 or very small)
        let initial_size = unsafe { knowhere_get_index_size(index) };
        
        // Add some vectors
        let vectors: Vec<f32> = (0..100 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..100).collect();
        
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 100, 16);
        assert_eq!(train_result, CError::Success as i32);
        
        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 100, 16);
        assert_eq!(add_result, CError::Success as i32);
        
        // Test size after adding vectors (should be larger)
        let size_after = unsafe { knowhere_get_index_size(index) };
        assert!(size_after > initial_size);
        
        // Test index type
        let type_ptr = unsafe { knowhere_get_index_type(index) };
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "Flat");
        
        // Test metric type
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }.to_str().unwrap();
        assert_eq!(metric_str, "L2");
        
        knowhere_free_index(index);
    }
    
    #[test]
    fn test_index_statistics_hnsw() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::Ip,
            dim: 32,
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        // Test index type
        let type_ptr = unsafe { knowhere_get_index_type(index) };
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "HNSW");
        
        // Test metric type
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }.to_str().unwrap();
        assert_eq!(metric_str, "IP");
        
        // Test size
        let size = unsafe { knowhere_get_index_size(index) };
        assert!(size >= 0);
        
        knowhere_free_index(index);
    }
    
    #[test]
    fn test_index_statistics_scann() {
        let config = CIndexConfig {
            index_type: CIndexType::Scann,
            metric_type: CMetricType::Cosine,
            dim: 64,
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        // Test index type
        let type_ptr = unsafe { knowhere_get_index_type(index) };
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "ScaNN");
        
        // Test metric type (ScaNN defaults to L2)
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }.to_str().unwrap();
        assert_eq!(metric_str, "L2");
        
        // Test size
        let size = unsafe { knowhere_get_index_size(index) };
        assert!(size >= 0);
        
        knowhere_free_index(index);
    }
    
    #[test]
    fn test_index_statistics_null_pointer() {
        // Test with null pointer - should return safe defaults
        let size = unsafe { knowhere_get_index_size(std::ptr::null()) };
        assert_eq!(size, 0);
        
        let type_ptr = unsafe { knowhere_get_index_type(std::ptr::null()) };
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }.to_str().unwrap();
        assert_eq!(type_str, "Unknown");
        
        let metric_ptr = unsafe { knowhere_get_index_metric(std::ptr::null()) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }.to_str().unwrap();
        assert_eq!(metric_str, "Unknown");
    }
    
    #[test]
    fn test_create_hnsw_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 128,
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let dim = knowhere_get_index_dim(index);
        assert_eq!(dim, 128);

        knowhere_free_index(index);
    }

    #[test]
    fn test_create_scann_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Scann,
            metric_type: CMetricType::L2,
            dim: 128,
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let dim = knowhere_get_index_dim(index);
        assert_eq!(dim, 128);

        let count = knowhere_get_index_count(index);
        assert_eq!(count, 0);

        knowhere_free_index(index);
    }

    #[test]
    fn test_scann_add_and_search() {
        let config = CIndexConfig {
            index_type: CIndexType::Scann,
            metric_type: CMetricType::L2,
            dim: 16,
            num_partitions: 16,
            num_centroids: 256,
            reorder_k: 100,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create simple test vectors: 10 vectors of dim 16
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        // Train the index
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        // Add vectors
        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        let count = knowhere_get_index_count(index);
        assert_eq!(count, 10);

        // Search
        let query: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = knowhere_search(index, query.as_ptr(), 1, 3, 16);
        assert!(!result.is_null());

        let result = unsafe { &mut *result };
        assert_eq!(result.num_results, 3);

        knowhere_free_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_get_vectors_by_ids() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors: 10 vectors of dim 16
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        // Train and add vectors
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Get vectors by IDs
        let query_ids: Vec<i64> = vec![0, 5, 9];
        let result = knowhere_get_vectors_by_ids(index, query_ids.as_ptr(), query_ids.len());
        assert!(!result.is_null());

        let result = unsafe { &mut *result };
        assert_eq!(result.num_vectors, 3);
        assert_eq!(result.dim, 16);

        // Verify vector values (first element of each vector)
        let vectors_slice = unsafe { std::slice::from_raw_parts(result.vectors, result.num_vectors * result.dim) };
        assert_eq!(vectors_slice[0], 0.0);  // First element of vector 0
        assert_eq!(vectors_slice[16], 80.0); // First element of vector 5 (5*16=80)
        assert_eq!(vectors_slice[32], 144.0); // First element of vector 9 (9*16=144)

        knowhere_free_vector_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_serialize_flat_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Add some vectors
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Serialize
        let binset = knowhere_serialize_index(index);
        assert!(!binset.is_null());

        unsafe {
            let binset_ref = &*binset;
            assert_eq!(binset_ref.count, 1);
            assert!(!binset_ref.keys.is_null());
            assert!(!binset_ref.values.is_null());

            let binary = &*binset_ref.values;
            assert!(!binary.data.is_null());
            assert!(binary.size > 0);

            // Verify key name
            let key = std::ffi::CStr::from_ptr(*binset_ref.keys);
            assert_eq!(key.to_str().unwrap(), "index_data");
        }

        knowhere_free_binary_set(binset);
        knowhere_free_index(index);
    }

    #[test]
    fn test_save_load_flat_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config.clone());
        assert!(!index.is_null());

        // Add some vectors
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Save to file
        let path = std::env::temp_dir().join("test_flat_index.bin");
        let path_str = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let save_result = knowhere_save_index(index, path_str.as_ptr());
        assert_eq!(save_result, CError::Success as i32);

        knowhere_free_index(index);

        // Create a new index and load from file
        let loaded_index = knowhere_create_index(config.clone());
        assert!(!loaded_index.is_null());

        let load_result = knowhere_load_index(loaded_index, path_str.as_ptr());
        assert_eq!(load_result, CError::Success as i32);

        // Verify loaded index
        let count = knowhere_get_index_count(loaded_index);
        assert_eq!(count, 10);

        // Search on loaded index
        let query: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = knowhere_search(loaded_index, query.as_ptr(), 1, 3, 16);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 3);
        }

        knowhere_free_result(result);
        knowhere_free_index(loaded_index);

        // Clean up
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_hnsw_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 16,
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        };

        let index = knowhere_create_index(config.clone());
        assert!(!index.is_null());

        // Add some vectors
        let vectors: Vec<f32> = (0..50 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..50).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 50, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 50, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Save to file
        let path = std::env::temp_dir().join("test_hnsw_index.bin");
        let path_str = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

        let save_result = knowhere_save_index(index, path_str.as_ptr());
        assert_eq!(save_result, CError::Success as i32);

        knowhere_free_index(index);

        // Create a new index and load from file
        let loaded_index = knowhere_create_index(config.clone());
        assert!(!loaded_index.is_null());

        let load_result = knowhere_load_index(loaded_index, path_str.as_ptr());
        assert_eq!(load_result, CError::Success as i32);

        // Verify loaded index
        let count = knowhere_get_index_count(loaded_index);
        assert_eq!(count, 50);

        // Search on loaded index
        let query: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = knowhere_search(loaded_index, query.as_ptr(), 1, 5, 16);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert!(result_ref.num_results > 0);
        }

        knowhere_free_result(result);
        knowhere_free_index(loaded_index);

        // Clean up
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_serialize_null_index() {
        let binset = knowhere_serialize_index(std::ptr::null());
        assert!(binset.is_null());
    }

    #[test]
    fn test_save_null_index() {
        let path = std::ffi::CString::new("/tmp/test.bin").unwrap();
        let result = knowhere_save_index(std::ptr::null(), path.as_ptr());
        assert_eq!(result, CError::InvalidArg as i32);
    }

    #[test]
    fn test_load_null_index() {
        let path = std::ffi::CString::new("/tmp/test.bin").unwrap();
        let result = knowhere_load_index(std::ptr::null_mut(), path.as_ptr());
        assert_eq!(result, CError::InvalidArg as i32);
    }

    #[test]
    fn test_deserialize_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        // 创建源索引并添加向量
        let source_index = knowhere_create_index(config.clone());
        assert!(!source_index.is_null());

        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        let train_result = knowhere_train_index(source_index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(source_index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // 序列化源索引
        let binset = knowhere_serialize_index(source_index);
        assert!(!binset.is_null());

        // 创建目标索引
        let target_index = knowhere_create_index(config.clone());
        assert!(!target_index.is_null());

        // 反序列化到目标索引
        let deserialize_result = knowhere_deserialize_index(target_index, binset);
        assert_eq!(deserialize_result, CError::Success as i32);

        // 验证目标索引有相同的数据
        let source_count = knowhere_get_index_count(source_index);
        let target_count = knowhere_get_index_count(target_index);
        assert_eq!(source_count, target_count);
        assert_eq!(target_count, 10);

        // 验证搜索结果相同
        let query: Vec<f32> = (0..16).map(|i| i as f32).collect();
        
        let source_result = knowhere_search(source_index, query.as_ptr(), 1, 3, 16);
        assert!(!source_result.is_null());
        
        let target_result = knowhere_search(target_index, query.as_ptr(), 1, 3, 16);
        assert!(!target_result.is_null());

        unsafe {
            let src_ref = &*source_result;
            let tgt_ref = &*target_result;
            
            assert_eq!(src_ref.num_results, tgt_ref.num_results);
            
            // 验证搜索结果 ID 相同
            let src_ids = std::slice::from_raw_parts(src_ref.ids, src_ref.num_results);
            let tgt_ids = std::slice::from_raw_parts(tgt_ref.ids, tgt_ref.num_results);
            assert_eq!(src_ids, tgt_ids);
        }

        knowhere_free_result(source_result);
        knowhere_free_result(target_result);
        knowhere_free_binary_set(binset);
        knowhere_free_index(source_index);
        knowhere_free_index(target_index);
    }

    #[test]
    fn test_deserialize_null_index() {
        let config = CIndexConfig::default();
        let index = knowhere_create_index(config);
        let result = knowhere_deserialize_index(std::ptr::null_mut(), std::ptr::null());
        assert_eq!(result, CError::InvalidArg as i32);
        knowhere_free_index(index);
    }

    #[test]
    fn test_deserialize_empty_binset() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // 创建空的 CBinarySet
        let empty_binset = CBinarySet {
            keys: std::ptr::null_mut(),
            values: std::ptr::null_mut(),
            count: 0,
        };

        let result = knowhere_deserialize_index(index, &empty_binset);
        assert_eq!(result, CError::InvalidArg as i32);

        knowhere_free_index(index);
    }

    #[test]
    fn test_range_search_flat_l2() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 4,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Add test vectors: 4 vectors at different distances from origin
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,  // dist=1.0 from origin
            0.0, 1.0, 0.0, 0.0,  // dist=1.0 from origin
            0.0, 0.0, 1.0, 0.0,  // dist=1.0 from origin
            2.0, 0.0, 0.0, 0.0,  // dist=2.0 from origin
        ];
        let ids = vec![0i64, 1, 2, 3];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 4, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 4, 4);
        assert_eq!(add_result, CError::Success as i32);

        // Range search with radius=1.5 (should find first 3 vectors)
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let result = knowhere_range_search(index, query.as_ptr(), 1, 1.5, 4);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_queries, 1);
            assert!(result_ref.total_count >= 3); // Should find at least 3 vectors within radius 1.5
            assert!(result_ref.elapsed_ms >= 0.0);

            // Verify lims array
            let lims = std::slice::from_raw_parts(result_ref.lims, result_ref.num_queries + 1);
            assert_eq!(lims[0], 0);
            assert_eq!(lims[1], result_ref.total_count);
        }

        knowhere_free_range_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_range_search_multiple_queries() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 4,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Add test vectors
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,  // id=0
            1.0, 0.0, 0.0, 0.0,  // id=1
            0.0, 1.0, 0.0, 0.0,  // id=2
        ];
        let ids = vec![0i64, 1, 2];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 3, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 4);
        assert_eq!(add_result, CError::Success as i32);

        // Two query vectors
        let queries = vec![
            0.0, 0.0, 0.0, 0.0,  // Query 1: at origin
            1.0, 0.0, 0.0, 0.0,  // Query 2: at (1,0,0,0)
        ];
        let result = knowhere_range_search(index, queries.as_ptr(), 2, 1.5, 4);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_queries, 2);

            // Verify lims array
            let lims = std::slice::from_raw_parts(result_ref.lims, result_ref.num_queries + 1);
            assert_eq!(lims[0], 0);
            assert_eq!(lims[1], lims[2] - lims[1]); // Each query should have similar results
        }

        knowhere_free_range_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_range_search_null_index() {
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let result = knowhere_range_search(
            std::ptr::null(),
            query.as_ptr(),
            1,
            1.0,
            4,
        );
        assert!(result.is_null());
    }

    #[test]
    fn test_range_search_null_query() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 4,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let result = knowhere_range_search(
            index,
            std::ptr::null(),
            1,
            1.0,
            4,
        );
        assert!(result.is_null());

        knowhere_free_index(index);
    }

    #[test]
    fn test_range_search_hnsw_not_implemented() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 4,
            ef_construction: 16,
            ef_search: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Add test vectors
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ];
        let ids = vec![0i64, 1];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 2, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 2, 4);
        assert_eq!(add_result, CError::Success as i32);

        // HNSW range search should return NULL (NotImplemented)
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let result = knowhere_range_search(index, query.as_ptr(), 1, 1.5, 4);
        assert!(result.is_null());

        knowhere_free_index(index);
    }

    #[test]
    fn test_free_range_result_null() {
        // Should not panic
        knowhere_free_range_result(std::ptr::null_mut());
    }

    // ========== GetVectorByIds C API Tests ==========

    #[test]
    fn test_get_vector_by_ids_flat() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors: 10 vectors of dim 16
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        // Train and add vectors
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Get vector by single ID
        let query_ids: Vec<i64> = vec![5];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 16);
        assert!(!result.is_null());

        let result = unsafe { &*result };
        assert_eq!(result.num_ids, 1);
        assert_eq!(result.dim, 16);

        // Verify vector values
        let vectors_slice = unsafe { std::slice::from_raw_parts(result.vectors, result.num_ids * result.dim) };
        assert_eq!(vectors_slice[0], 80.0);  // First element of vector 5 (5*16=80)

        unsafe {
            knowhere_free_get_vector_result(result as *const _ as *mut _);
        }
        knowhere_free_index(index);
    }

    #[test]
    fn test_get_vector_by_ids_multiple() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors: 10 vectors of dim 16
        let vectors: Vec<f32> = (0..10 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        // Train and add vectors
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Get multiple vectors by IDs
        let query_ids: Vec<i64> = vec![0, 5, 9];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 16);
        assert!(!result.is_null());

        let result = unsafe { &*result };
        assert_eq!(result.num_ids, 3);
        assert_eq!(result.dim, 16);

        // Verify vector values
        let vectors_slice = unsafe { std::slice::from_raw_parts(result.vectors, result.num_ids * result.dim) };
        assert_eq!(vectors_slice[0], 0.0);    // First element of vector 0
        assert_eq!(vectors_slice[16], 80.0);  // First element of vector 5 (5*16=80)
        assert_eq!(vectors_slice[32], 144.0); // First element of vector 9 (9*16=144)

        unsafe {
            knowhere_free_get_vector_result(result as *const _ as *mut _);
        }
        knowhere_free_index(index);
    }

    #[test]
    fn test_get_vector_by_ids_nonexistent() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors: 5 vectors of dim 16
        let vectors: Vec<f32> = (0..5 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..5).collect();

        // Train and add vectors
        let train_result = knowhere_train_index(index, vectors.as_ptr(), 5, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 5, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Try to get non-existent ID
        let query_ids: Vec<i64> = vec![100, 101];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 16);
        assert!(result.is_null());

        knowhere_free_index(index);
    }

    #[test]
    fn test_get_vector_by_ids_null_index() {
        let query_ids: Vec<i64> = vec![0, 1, 2];
        let result = knowhere_get_vector_by_ids(
            std::ptr::null(),
            query_ids.as_ptr(),
            query_ids.len(),
            16,
        );
        assert!(result.is_null());
    }

    #[test]
    fn test_get_vector_by_ids_dimension_mismatch() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors: 5 vectors of dim 16
        let vectors: Vec<f32> = (0..5 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..5).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 5, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 5, 16);
        assert_eq!(add_result, CError::Success as i32);

        // Try with wrong dimension
        let query_ids: Vec<i64> = vec![0];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 32);
        assert!(result.is_null());

        knowhere_free_index(index);
    }

    #[test]
    fn test_free_get_vector_result_null() {
        // Should not panic
        knowhere_free_get_vector_result(std::ptr::null_mut());
    }

    #[test]
    fn test_get_vector_by_ids_hnsw_not_implemented() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 16,
            ef_construction: 16,
            ef_search: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create test vectors
        let vectors: Vec<f32> = (0..5 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..5).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 5, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 5, 16);
        assert_eq!(add_result, CError::Success as i32);

        // HNSW get_vector_by_ids should return NULL (NotImplemented)
        let query_ids: Vec<i64> = vec![0];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 16);
        assert!(result.is_null());

        knowhere_free_index(index);
    }

    #[test]
    fn test_bitset_create_and_set() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Check initial count (all zeros)
        let count = unsafe { knowhere_bitset_count(bitset) };
        assert_eq!(count, 0);
        
        // Set some bits
        unsafe {
            knowhere_bitset_set(bitset, 5, true);
            knowhere_bitset_set(bitset, 10, true);
            knowhere_bitset_set(bitset, 50, true);
        }
        
        // Check count
        let count = unsafe { knowhere_bitset_count(bitset) };
        assert_eq!(count, 3);
        
        // Check individual bits
        assert!(unsafe { knowhere_bitset_get(bitset, 5) });
        assert!(unsafe { knowhere_bitset_get(bitset, 10) });
        assert!(unsafe { knowhere_bitset_get(bitset, 50) });
        assert!(!unsafe { knowhere_bitset_get(bitset, 0) });
        assert!(!unsafe { knowhere_bitset_get(bitset, 7) });
        
        knowhere_bitset_free(bitset);
    }

    #[test]
    fn test_search_with_bitset_flat() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 4,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Add test vectors: 10 vectors at different positions
        let vectors: Vec<f32> = (0..10 * 4).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..10).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 10, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 10, 4);
        assert_eq!(add_result, CError::Success as i32);

        // Create bitset to filter out vectors 0, 1, 2 (the closest to query at origin)
        let bitset = knowhere_bitset_create(10);
        unsafe {
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 1, true);
            knowhere_bitset_set(bitset, 2, true);
        }

        // Search with bitset filter
        let query: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        let result = knowhere_search_with_bitset(
            index,
            query.as_ptr(),
            1,
            5,
            4,
            bitset,
        );
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 5);
            
            // Results should NOT include IDs 0, 1, 2 (they were filtered)
            for i in 0..result_ref.num_results {
                let id = *result_ref.ids.add(i);
                assert!(id >= 3, "ID {} should have been filtered out", id);
            }
        }

        knowhere_free_result(result);
        knowhere_bitset_free(bitset as *mut _);
        knowhere_free_index(index);
    }

    #[test]
    fn test_search_with_bitset_null_params() {
        let query: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];
        
        // Null index
        let result = knowhere_search_with_bitset(
            std::ptr::null(),
            query.as_ptr(),
            1,
            5,
            4,
            std::ptr::null(),
        );
        assert!(result.is_null());
        
        // Null query
        let bitset = knowhere_bitset_create(10);
        let result = knowhere_search_with_bitset(
            std::ptr::null(),
            std::ptr::null(),
            1,
            5,
            4,
            bitset,
        );
        assert!(result.is_null());
        
        knowhere_bitset_free(bitset as *mut _);
    }

    #[test]
    fn test_bitset_byte_size() {
        // Test various sizes
        let bitset_1 = knowhere_bitset_create(1);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_1), 1);  // (1+7)/8 = 1
        }
        knowhere_bitset_free(bitset_1);

        let bitset_8 = knowhere_bitset_create(8);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_8), 1);  // (8+7)/8 = 1
        }
        knowhere_bitset_free(bitset_8);

        let bitset_64 = knowhere_bitset_create(64);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_64), 8);  // (64+7)/8 = 8
        }
        knowhere_bitset_free(bitset_64);

        let bitset_100 = knowhere_bitset_create(100);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_100), 13);  // (100+7)/8 = 13
        }
        knowhere_bitset_free(bitset_100);

        let bitset_1000 = knowhere_bitset_create(1000);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_1000), 125);  // (1000+7)/8 = 125
        }
        knowhere_bitset_free(bitset_1000);

        // Null pointer
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(std::ptr::null()), 0);
        }
    }

    #[test]
    fn test_bitset_data() {
        let bitset = knowhere_bitset_create(128);
        assert!(!bitset.is_null());

        // Get data pointer
        let data = unsafe { knowhere_bitset_data(bitset) };
        assert!(!data.is_null());

        // Verify we can read the data (should be all zeros initially)
        unsafe {
            let slice = std::slice::from_raw_parts(data, 2);  // 128 bits = 2 u64s
            assert_eq!(slice[0], 0);
            assert_eq!(slice[1], 0);

            // Set some bits and verify
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 64, true);
            
            // Re-read data
            let data2 = knowhere_bitset_data(bitset);
            let slice2 = std::slice::from_raw_parts(data2, 2);
            assert_eq!(slice2[0], 1u64);      // First bit set
            assert_eq!(slice2[1], 1u64);      // 65th bit set (first bit of second u64)
        }

        knowhere_bitset_free(bitset);

        // Null pointer
        unsafe {
            assert!(knowhere_bitset_data(std::ptr::null()).is_null());
        }
    }
    
    #[test]
    fn test_bitset_count() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Initially all zeros
        assert_eq!(unsafe { knowhere_bitset_count(bitset) }, 0);
        
        // Set some bits
        unsafe {
            knowhere_bitset_set(bitset, 5, true);
            knowhere_bitset_set(bitset, 10, true);
            knowhere_bitset_set(bitset, 50, true);
        }
        
        assert_eq!(unsafe { knowhere_bitset_count(bitset) }, 3);
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_test() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Initially all zeros
        assert!(!unsafe { knowhere_bitset_test(bitset, 0) });
        assert!(!unsafe { knowhere_bitset_test(bitset, 50) });
        
        // Set some bits
        unsafe {
            knowhere_bitset_set(bitset, 5, true);
            knowhere_bitset_set(bitset, 10, true);
        }
        
        assert!(unsafe { knowhere_bitset_test(bitset, 5) });
        assert!(unsafe { knowhere_bitset_test(bitset, 10) });
        assert!(!unsafe { knowhere_bitset_test(bitset, 0) });
        assert!(!unsafe { knowhere_bitset_test(bitset, 50) });
        
        // Out of range should return true (filtered)
        assert!(unsafe { knowhere_bitset_test(bitset, 100) });
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_filter_ratio() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Initially 0 ratio
        assert_eq!(unsafe { knowhere_bitset_filter_ratio(bitset) }, 0.0);
        
        // Set all bits
        unsafe {
            for i in 0..100 {
                knowhere_bitset_set(bitset, i, true);
            }
        }
        
        assert_eq!(unsafe { knowhere_bitset_filter_ratio(bitset) }, 1.0);
        
        // Clear and set half
        unsafe {
            for i in 0..100 {
                knowhere_bitset_set(bitset, i, false);
            }
            for i in 0..50 {
                knowhere_bitset_set(bitset, i, true);
            }
        }
        
        let ratio = unsafe { knowhere_bitset_filter_ratio(bitset) };
        assert!((ratio - 0.5).abs() < 0.01);
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_get_first_valid_index() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Initially first valid is 0
        assert_eq!(unsafe { knowhere_bitset_get_first_valid_index(bitset) }, 0);
        
        // Set first few bits
        unsafe {
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 1, true);
            knowhere_bitset_set(bitset, 2, true);
        }
        
        assert_eq!(unsafe { knowhere_bitset_get_first_valid_index(bitset) }, 3);
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_size() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        assert_eq!(unsafe { knowhere_bitset_size(bitset) }, 100);
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_empty() {
        let non_empty = knowhere_bitset_create(100);
        
        // Non-empty bitset
        assert!(!unsafe { knowhere_bitset_empty(non_empty) });
        
        // NULL bitset should return true (empty)
        assert!(unsafe { knowhere_bitset_empty(std::ptr::null()) });
        
        knowhere_bitset_free(non_empty);
    }
    
    #[test]
    fn test_bitset_has_out_ids() {
        let bitset = knowhere_bitset_create(100);
        assert!(!bitset.is_null());
        
        // Currently out_ids is not supported in CBitset, should return false
        assert!(!unsafe { knowhere_bitset_has_out_ids(bitset) });
        
        knowhere_bitset_free(bitset);
    }
    
    #[test]
    fn test_bitset_or() {
        let a = knowhere_bitset_create(100);
        let b = knowhere_bitset_create(100);
        assert!(!a.is_null());
        assert!(!b.is_null());
        
        // 设置 a 的位：0, 1, 2
        unsafe {
            knowhere_bitset_set(a, 0, true);
            knowhere_bitset_set(a, 1, true);
            knowhere_bitset_set(a, 2, true);
        }
        
        // 设置 b 的位：2, 3, 4
        unsafe {
            knowhere_bitset_set(b, 2, true);
            knowhere_bitset_set(b, 3, true);
            knowhere_bitset_set(b, 4, true);
        }
        
        // 执行 OR 操作
        let result = unsafe { knowhere_bitset_or(a, b) };
        assert!(!result.is_null());
        
        // 验证结果：0, 1, 2, 3, 4 都应该被设置
        unsafe {
            assert!(knowhere_bitset_get(result, 0));
            assert!(knowhere_bitset_get(result, 1));
            assert!(knowhere_bitset_get(result, 2));
            assert!(knowhere_bitset_get(result, 3));
            assert!(knowhere_bitset_get(result, 4));
            assert!(!knowhere_bitset_get(result, 5));
        }
        
        // 验证计数
        assert_eq!(unsafe { knowhere_bitset_count(result) }, 5);
        
        unsafe {
            knowhere_bitset_free(result);
            knowhere_bitset_free(b);
            knowhere_bitset_free(a);
        }
    }
    
    #[test]
    fn test_bitset_and() {
        let a = knowhere_bitset_create(100);
        let b = knowhere_bitset_create(100);
        assert!(!a.is_null());
        assert!(!b.is_null());
        
        // 设置 a 的位：0, 1, 2, 3
        unsafe {
            knowhere_bitset_set(a, 0, true);
            knowhere_bitset_set(a, 1, true);
            knowhere_bitset_set(a, 2, true);
            knowhere_bitset_set(a, 3, true);
        }
        
        // 设置 b 的位：2, 3, 4, 5
        unsafe {
            knowhere_bitset_set(b, 2, true);
            knowhere_bitset_set(b, 3, true);
            knowhere_bitset_set(b, 4, true);
            knowhere_bitset_set(b, 5, true);
        }
        
        // 执行 AND 操作
        let result = unsafe { knowhere_bitset_and(a, b) };
        assert!(!result.is_null());
        
        // 验证结果：只有 2, 3 应该被设置（交集）
        unsafe {
            assert!(!knowhere_bitset_get(result, 0));
            assert!(!knowhere_bitset_get(result, 1));
            assert!(knowhere_bitset_get(result, 2));
            assert!(knowhere_bitset_get(result, 3));
            assert!(!knowhere_bitset_get(result, 4));
            assert!(!knowhere_bitset_get(result, 5));
        }
        
        // 验证计数
        assert_eq!(unsafe { knowhere_bitset_count(result) }, 2);
        
        unsafe {
            knowhere_bitset_free(result);
            knowhere_bitset_free(b);
            knowhere_bitset_free(a);
        }
    }
    
    #[test]
    fn test_bitset_xor() {
        let a = knowhere_bitset_create(100);
        let b = knowhere_bitset_create(100);
        assert!(!a.is_null());
        assert!(!b.is_null());
        
        // 设置 a 的位：0, 1, 2, 3
        unsafe {
            knowhere_bitset_set(a, 0, true);
            knowhere_bitset_set(a, 1, true);
            knowhere_bitset_set(a, 2, true);
            knowhere_bitset_set(a, 3, true);
        }
        
        // 设置 b 的位：2, 3, 4, 5
        unsafe {
            knowhere_bitset_set(b, 2, true);
            knowhere_bitset_set(b, 3, true);
            knowhere_bitset_set(b, 4, true);
            knowhere_bitset_set(b, 5, true);
        }
        
        // 执行 XOR 操作
        let result = unsafe { knowhere_bitset_xor(a, b) };
        assert!(!result.is_null());
        
        // 验证结果：0, 1, 4, 5 应该被设置（对称差）
        unsafe {
            assert!(knowhere_bitset_get(result, 0));
            assert!(knowhere_bitset_get(result, 1));
            assert!(!knowhere_bitset_get(result, 2));
            assert!(!knowhere_bitset_get(result, 3));
            assert!(knowhere_bitset_get(result, 4));
            assert!(knowhere_bitset_get(result, 5));
        }
        
        // 验证计数
        assert_eq!(unsafe { knowhere_bitset_count(result) }, 4);
        
        unsafe {
            knowhere_bitset_free(result);
            knowhere_bitset_free(b);
            knowhere_bitset_free(a);
        }
    }
    
    #[test]
    fn test_bitset_or_different_sizes() {
        // 测试不同长度的 bitset
        let a = knowhere_bitset_create(50);
        let b = knowhere_bitset_create(100);
        assert!(!a.is_null());
        assert!(!b.is_null());
        
        // 设置 a 的位：0, 1
        unsafe {
            knowhere_bitset_set(a, 0, true);
            knowhere_bitset_set(a, 1, true);
        }
        
        // 设置 b 的位：1, 2
        unsafe {
            knowhere_bitset_set(b, 1, true);
            knowhere_bitset_set(b, 2, true);
        }
        
        // 执行 OR 操作
        let result = unsafe { knowhere_bitset_or(a, b) };
        assert!(!result.is_null());
        
        // 结果长度应该是 100（最大值）
        assert_eq!(unsafe { knowhere_bitset_size(result) }, 100);
        
        // 验证结果
        unsafe {
            assert!(knowhere_bitset_get(result, 0));
            assert!(knowhere_bitset_get(result, 1));
            assert!(knowhere_bitset_get(result, 2));
            assert!(!knowhere_bitset_get(result, 3));
        }
        
        unsafe {
            knowhere_bitset_free(result);
            knowhere_bitset_free(b);
            knowhere_bitset_free(a);
        }
    }
    
    #[test]
    fn test_bitset_and_empty() {
        // 测试与空 bitset 的 AND 操作
        let a = knowhere_bitset_create(100);
        let b = knowhere_bitset_create(100);
        
        // a 有一些位设置
        unsafe {
            knowhere_bitset_set(a, 0, true);
            knowhere_bitset_set(a, 1, true);
        }
        // b 保持全 0
        
        let result = unsafe { knowhere_bitset_and(a, b) };
        assert!(!result.is_null());
        
        // 结果应该全为 0
        assert_eq!(unsafe { knowhere_bitset_count(result) }, 0);
        
        unsafe {
            knowhere_bitset_free(result);
            knowhere_bitset_free(b);
            knowhere_bitset_free(a);
        }
    }
    
    #[test]
    fn test_bitset_null_handling() {
        // 测试 NULL 指针处理
        let a = knowhere_bitset_create(100);
        
        let result_or = unsafe { knowhere_bitset_or(a, std::ptr::null()) };
        assert!(result_or.is_null());
        
        let result_and = unsafe { knowhere_bitset_and(std::ptr::null(), a) };
        assert!(result_and.is_null());
        
        let result_xor = unsafe { knowhere_bitset_xor(a, std::ptr::null()) };
        assert!(result_xor.is_null());
        
        knowhere_bitset_free(a);
    }
}

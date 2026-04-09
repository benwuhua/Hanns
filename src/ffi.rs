//! C API 绑定定义
//!
//! 供 Milvus C++ 调用
//!
//! Safety: This module contains FFI functions that accept raw pointers.
//! Callers must ensure pointers are valid and properly aligned.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::missing_safety_doc)]
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

pub mod interrupt_ffi;
pub mod minhash_lsh_ffi;

// Re-export interrupt FFI types and functions for C API
pub use interrupt_ffi::{
    knowhere_interrupt_clone, knowhere_interrupt_create, knowhere_interrupt_create_with_state,
    knowhere_interrupt_free, knowhere_interrupt_interrupt, knowhere_interrupt_is_interrupted,
    knowhere_interrupt_reset, knowhere_interrupt_test_and_set, CInterrupt, CInterruptError,
};

use crate::api::{
    IndexConfig, IndexParams, IndexType, MetricType, SearchRequest, SearchResult as ApiSearchResult,
};
use crate::faiss::{HnswIndex, IvfFlatIndex, IvfPqIndex, MemIndex, ScaNNConfig, ScaNNIndex};
use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::path::Path;
const FFI_FORCE_SERIAL_HNSW_ADD_ENV: &str = "HANNS_FFI_FORCE_SERIAL_HNSW_ADD";
const FFI_ENABLE_PARALLEL_HNSW_ADD_ENV: &str = "HANNS_FFI_ENABLE_PARALLEL_HNSW_ADD";

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
    Scann = 2, // Rust-only, no native parity
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
    SparseWand = 13, // Rust-only, no native parity
    BinIvfFlat = 14,
    SparseWandCc = 15, // Rust-only, no native parity
    MinHashLsh = 16,
    IvfPq = 17,
    IvfFlat = 18,
    DiskAnn = 19,
    HnswPcaSq = 20,
    HnswPcaUsq = 21,
    DiskAnnPcaUsq = 22,
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
    /// Data type (0 = Float, 100 = Binary, etc.) - matches Milvus VecType enum
    pub data_type: i32,
    // DiskANN-specific parameters
    pub pq_code_budget_gb: f32,
    pub build_dram_budget_gb: f32,
    pub disk_pq_dims: usize,
    pub beamwidth: usize,
    // IVF-PQ parameters
    pub pq_m: usize,     // number of sub-quantizers
    pub pq_nbits: usize, // bits per sub-quantizer (default 8)
    // PCA parameters
    pub pca_dim: usize, // PCA target dimensionality (0 = no PCA)
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
            data_type: 101, // Default to Float (101)
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            beamwidth: 8,
            pq_m: 32,
            pq_nbits: 8,
            pca_dim: 0,
        }
    }
}

const SPARSE_PERSISTENCE_ENVELOPE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SparsePersistenceEnvelope {
    version: u32,
    dim: usize,
    payload: Vec<u8>,
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

#[derive(Serialize)]
struct AdditionalScalarMeta<'a> {
    runtime_supported: bool,
    mv_only_query: bool,
    support_mode: &'a str,
    unsupported_reason: &'a str,
}

#[derive(Serialize)]
struct IndexCapabilitySummary<'a> {
    get_vector_by_ids: &'a str,
    ann_iterator: &'a str,
    persistence: &'a str,
}

type RangeSearchOutcome = (Vec<i64>, Vec<f32>, Vec<usize>, f64);

#[derive(Serialize)]
struct PersistenceSemantics<'a> {
    file_save_load: &'a str,
    memory_serialize: &'a str,
    deserialize_from_file: &'a str,
}

#[derive(Serialize)]
struct IndexMetaSemantics<'a> {
    family: &'a str,
    raw_data_gate: &'a str,
    persistence_mode: &'a str,
    persistence: PersistenceSemantics<'a>,
    metadata_granularity: &'a str,
}

#[derive(Serialize)]
struct RuntimeObservabilitySummary<'a> {
    schema_version: &'a str,
    build_event: &'a str,
    search_event: &'a str,
    load_event: &'a str,
    required_fields: &'a [&'a str],
    optional_fields: &'a [&'a str],
}

#[derive(Serialize)]
struct TracePropagationSummary<'a> {
    ffi_entrypoint: &'a str,
    gate_runner_entrypoint: &'a str,
    context_encoding: &'a str,
    propagation_mode: &'a str,
}

#[derive(Serialize)]
struct ResourceContractSummary<'a> {
    schema_version: &'a str,
    memory_bytes: &'a str,
    disk_bytes: &'a str,
    mmap_supported: bool,
    unsupported_reason: &'a str,
}

#[derive(Serialize)]
struct IndexMetaSummary<'a> {
    index_type: &'a str,
    dim: usize,
    count: usize,
    is_trained: bool,
    has_raw_data: bool,
    additional_scalar_supported: bool,
    additional_scalar: AdditionalScalarMeta<'a>,
    capabilities: IndexCapabilitySummary<'a>,
    semantics: IndexMetaSemantics<'a>,
    observability: RuntimeObservabilitySummary<'a>,
    trace_propagation: TracePropagationSummary<'a>,
    resource_contract: ResourceContractSummary<'a>,
}

/// 包装索引对象 - 支持 Flat, HNSW, ScaNN, HNSW-PRQ, IVF-RaBitQ, HNSW-SQ, HNSW-PQ, BinFlat, BinaryHnsw, IVF-SQ8, BinIvfFlat, SparseWand, SparseWandCC, MinHashLSH, DiskANN
struct IndexWrapper {
    flat: Option<MemIndex>,
    hnsw: Option<HnswIndex>,
    scann: Option<ScaNNIndex>,
    hnsw_prq: Option<crate::faiss::HnswPrqIndex>,
    hnsw_sq: Option<crate::faiss::HnswSqIndex>,
    hnsw_pq: Option<crate::faiss::HnswPqIndex>,
    ivf_pq: Option<crate::faiss::IvfPqIndex>,
    bin_flat: Option<crate::faiss::BinFlatIndex>,
    binary_hnsw: Option<crate::faiss::BinaryHnswIndex>,
    ivf_sq8: Option<crate::faiss::IvfSq8Index>,
    ivf_flat: Option<crate::faiss::IvfFlatIndex>,
    bin_ivf_flat: Option<crate::faiss::BinIvfFlatIndex>,
    sparse_inverted: Option<crate::faiss::SparseInvertedIndex>,
    sparse_wand: Option<crate::faiss::SparseWandIndex>,
    sparse_wand_cc: Option<crate::faiss::SparseWandIndexCC>,
    minhash_lsh: Option<crate::index::MinHashLSHIndex>,
    diskann: Option<crate::faiss::diskann_aisaq::PQFlashIndex>,
    hnsw_pca_sq: Option<crate::faiss::HnswPcaSqIndex>,
    hnsw_pca_usq: Option<crate::faiss::HnswPcaUsqIndex>,
    diskann_pca_usq: Option<crate::faiss::DiskAnnPcaUsqIndex>,
    dim: usize,
    nprobe: usize,
}

impl IndexWrapper {
    fn env_var_truthy(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(false)
    }

    fn ffi_force_serial_hnsw_add() -> bool {
        Self::env_var_truthy(FFI_FORCE_SERIAL_HNSW_ADD_ENV)
    }

    fn ffi_enable_parallel_hnsw_add() -> bool {
        Self::env_var_truthy(FFI_ENABLE_PARALLEL_HNSW_ADD_ENV)
    }

    fn should_use_parallel_hnsw_add_via_ffi(idx: &HnswIndex, count: usize) -> bool {
        !Self::ffi_force_serial_hnsw_add()
            && Self::ffi_enable_parallel_hnsw_add()
            && idx.should_use_parallel_add(count)
    }

    fn dense_chunk_to_sparse_query(
        query_chunk: &[f32],
    ) -> crate::faiss::sparse_inverted::SparseVector {
        let elements = query_chunk
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(j, &v)| crate::faiss::sparse_inverted::SparseVecElement {
                dim: j as u32,
                val: v,
            })
            .collect();
        crate::faiss::sparse_inverted::SparseVector { elements }
    }

    fn search_sparse_queries<F>(
        &self,
        query: &[f32],
        query_dim: usize,
        mut search_one: F,
    ) -> Result<ApiSearchResult, CError>
    where
        F: FnMut(&crate::faiss::sparse_inverted::SparseVector) -> Vec<(i64, f32)>,
    {
        if query_dim == 0 || query.is_empty() || query.len() % query_dim != 0 {
            return Err(CError::InvalidArg);
        }

        let start = std::time::Instant::now();
        let num_queries = query.len() / query_dim;
        let mut ids = Vec::new();
        let mut distances = Vec::new();

        ids.reserve(num_queries);
        distances.reserve(num_queries);

        for query_chunk in query.chunks(query_dim) {
            let sparse_query = Self::dense_chunk_to_sparse_query(query_chunk);
            let results = search_one(&sparse_query);
            ids.reserve(results.len());
            distances.reserve(results.len());
            for (id, distance) in results {
                ids.push(id);
                distances.push(distance);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(ApiSearchResult::new(ids, distances, elapsed_ms))
    }

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

        // Map CIndexType to IndexType for validation
        let index_type: IndexType = match config.index_type {
            CIndexType::Flat => IndexType::Flat,
            CIndexType::Hnsw => IndexType::Hnsw,
            #[cfg(feature = "scann")]
            CIndexType::Scann => IndexType::Scann,
            #[cfg(not(feature = "scann"))]
            CIndexType::Scann => {
                eprintln!("Scann index type requires 'scann' feature to be enabled");
                return None;
            }
            CIndexType::HnswPrq => IndexType::HnswPrq,
            CIndexType::IvfRabitq => IndexType::IvfUsq,
            CIndexType::HnswSq => IndexType::HnswSq,
            CIndexType::HnswPq => IndexType::HnswPq,
            CIndexType::IvfPq => IndexType::IvfPq,
            CIndexType::IvfFlat => IndexType::IvfFlat,
            CIndexType::BinFlat => IndexType::BinFlat,
            CIndexType::BinaryHnsw => IndexType::BinaryHnsw,
            CIndexType::IvfSq8 => IndexType::IvfSq8,
            CIndexType::IvfFlatCc => IndexType::IvfFlatCc,
            CIndexType::IvfSqCc => IndexType::IvfSqCc,
            CIndexType::SparseInverted => IndexType::SparseInverted,
            CIndexType::SparseWand => IndexType::SparseWand,
            CIndexType::BinIvfFlat => IndexType::BinIvfFlat,
            CIndexType::SparseWandCc => IndexType::SparseWandCc,
            CIndexType::MinHashLsh => IndexType::MinHashLsh,
            CIndexType::DiskAnn => IndexType::DiskAnn,
            CIndexType::HnswPcaSq => IndexType::HnswSq,   // PCA+SQ variant
            CIndexType::HnswPcaUsq => IndexType::HnswSq,  // PCA+USQ variant
            CIndexType::DiskAnnPcaUsq => IndexType::DiskAnn, // DiskANN+PCA+USQ variant
        };

        // Parse data_type from i32 (Milvus VecType enum)
        let data_type =
            super::api::DataType::from_i32(config.data_type).unwrap_or(super::api::DataType::Float);

        // Validate (index_type, data_type, metric_type) combination
        if let Err(e) = super::api::validate_index_config(index_type, data_type, metric) {
            eprintln!("Invalid index configuration: {}", e);
            return None;
        }

        match config.index_type {
            CIndexType::Flat => {
                let index_config = IndexConfig {
                    index_type: IndexType::Flat,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams::default(),
                };
                let flat = MemIndex::new(&index_config).ok()?;
                Some(Self {
                    flat: Some(flat),
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::Hnsw => {
                let mut index_config = IndexConfig {
                    index_type: IndexType::Hnsw,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams::default(),
                };
                if config.ef_construction > 0 {
                    index_config.params.ef_construction = Some(config.ef_construction);
                }
                if config.ef_search > 0 {
                    index_config.params.ef_search = Some(config.ef_search);
                }
                let hnsw = HnswIndex::new(&index_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: Some(hnsw),
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::Scann => {
                if metric != MetricType::L2 {
                    eprintln!(
                        "ScaNN only supports L2 metric; got metric_type={:?}",
                        metric
                    );
                    return None;
                }
                if config.ef_search > 0 {
                    eprintln!("warn: ef_search ignored for ScaNN; use reorder_k instead");
                }
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
                // ScaNN FFI only exposes num_partitions/num_centroids/reorder_k.
                // ef_search is not plumbed into the ScaNN runtime search path.
                let reorder_k = if config.reorder_k > 0 {
                    config.reorder_k
                } else {
                    100
                };
                let scann_config = ScaNNConfig::new(num_partitions, num_centroids, reorder_k);
                let scann = ScaNNIndex::new(dim, scann_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: Some(scann),
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::HnswPrq => {
                let _index_config = IndexConfig {
                    index_type: IndexType::HnswPrq,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams {
                        m: Some(16),
                        ef_construction: if config.ef_construction > 0 {
                            Some(config.ef_construction)
                        } else {
                            None
                        },
                        ef_search: if config.ef_search > 0 {
                            Some(config.ef_search)
                        } else {
                            None
                        },
                        prq_m: Some(if config.prq_nsplits > 0 {
                            config.prq_nsplits
                        } else {
                            2
                        }),
                        prq_nrq: Some(if config.prq_msub > 0 {
                            config.prq_msub
                        } else {
                            4
                        }),
                        prq_nbits: Some(if config.prq_nbits > 0 {
                            config.prq_nbits
                        } else {
                            8
                        }),
                        ..Default::default()
                    },
                };

                let hnsw_prq_config = crate::faiss::HnswPrqConfig::new(dim)
                    .with_m(16)
                    .with_ef_construction(config.ef_construction)
                    .with_ef_search(config.ef_search)
                    .with_prq_params(
                        if config.prq_nsplits > 0 {
                            config.prq_nsplits
                        } else {
                            2
                        },
                        if config.prq_msub > 0 {
                            config.prq_msub
                        } else {
                            4
                        },
                        if config.prq_nbits > 0 {
                            config.prq_nbits
                        } else {
                            8
                        },
                    )
                    .with_metric_type(metric);

                let hnsw_prq = crate::faiss::HnswPrqIndex::new(hnsw_prq_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: Some(hnsw_prq),

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::IvfRabitq => {
                eprintln!("IvfRabitq merged into IvfUsq; use ivf_usq API");
                return None;
            }
            CIndexType::HnswSq => {
                let ef_construction = if config.ef_construction > 0 {
                    config.ef_construction
                } else {
                    200
                };
                let ef_search = if config.ef_search > 0 {
                    config.ef_search
                } else {
                    50
                };
                let sq_bit = if config.prq_nbits > 0 {
                    config.prq_nbits
                } else {
                    8
                };

                let hnsw_sq = crate::faiss::HnswSqIndex::new(dim);

                // Set config parameters
                let _hnsw_config = crate::faiss::HnswQuantizeConfig {
                    ef_construction,
                    ef_search,
                    sq_bit,
                    ..Default::default()
                };

                // Store config in index (simplified - HnswSqIndex needs config support)
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: Some(hnsw_sq),
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::HnswPq => {
                let pq_m = if config.prq_nsplits > 0 {
                    config.prq_nsplits
                } else {
                    8
                };
                let pq_k = if config.prq_msub > 0 {
                    config.prq_msub
                } else {
                    256
                };

                let hnsw_pq_config = crate::faiss::HnswPqConfig::new(dim)
                    .with_m(16)
                    .with_ef_construction(config.ef_construction)
                    .with_ef_search(config.ef_search)
                    .with_pq_params(pq_m, pq_k)
                    .with_metric_type(metric);

                let hnsw_pq = crate::faiss::HnswPqIndex::new(hnsw_pq_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: Some(hnsw_pq),
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::IvfSq8 => {
                // IVF-SQ8 index with scalar quantization
                let nlist = if config.num_centroids > 0 {
                    config.num_centroids
                } else {
                    256
                };
                let nprobe = if config.nprobe > 0 { config.nprobe } else { 8 };

                let index_config = IndexConfig {
                    index_type: IndexType::IvfSq8,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams::ivf_sq8(nlist, nprobe),
                };

                let ivf_sq8 = crate::faiss::IvfSq8Index::new(&index_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: Some(ivf_sq8),
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe,
                })
            }
            CIndexType::IvfFlat => {
                let nlist = if config.num_centroids > 0 {
                    config.num_centroids
                } else {
                    256
                };
                let nprobe = if config.nprobe > 0 { config.nprobe } else { 8 };

                let index_config = IndexConfig {
                    index_type: IndexType::IvfFlat,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams::ivf(nlist, nprobe),
                };

                let ivf_flat = IvfFlatIndex::new(&index_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: Some(ivf_flat),
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe,
                })
            }
            CIndexType::IvfPq => {
                let index_config = IndexConfig {
                    index_type: IndexType::IvfPq,
                    metric_type: metric,
                    dim,
                    data_type,
                    params: IndexParams {
                        nlist: Some(config.num_clusters.max(1)),
                        nprobe: Some(config.nprobe.max(1)),
                        m: Some(config.pq_m.max(1).min(dim.max(1))),
                        nbits_per_idx: Some(if config.pq_nbits > 0 { config.pq_nbits } else { 8 }),
                        ..Default::default()
                    },
                };
                let ivf_pq = IvfPqIndex::new(&index_config).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: Some(ivf_pq),
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    dim,
                    nprobe: 8,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::BinFlat => {
                // Binary Flat index for binary vectors with Hamming distance
                let bin_flat = crate::faiss::BinFlatIndex::new(dim, metric);
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: Some(bin_flat),
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::BinaryHnsw => {
                // Binary HNSW index for binary vectors
                let mut index_config = IndexConfig {
                    index_type: IndexType::BinaryHnsw,
                    metric_type: metric,
                    dim,
                    data_type,
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
                        flat: None,
                        hnsw: None,
                        scann: None,
                        hnsw_prq: None,

                        hnsw_sq: None,
                        hnsw_pq: None,
                        ivf_pq: None,
                        bin_flat: None,
                        binary_hnsw: Some(hnsw),
                        ivf_sq8: None,
                        ivf_flat: None,
                        bin_ivf_flat: None,
                        sparse_inverted: None,
                        sparse_wand: None,
                        sparse_wand_cc: None,
                        minhash_lsh: None,
                        diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                        dim,
                        nprobe: 8,
                    })
                } else {
                    None
                }
            }
            CIndexType::BinIvfFlat => {
                // Binary IVF Flat index for binary vectors with Hamming distance
                let nlist = if config.num_clusters > 0 {
                    config.num_clusters
                } else {
                    256
                };
                let mut bin_ivf_flat = crate::faiss::BinIvfFlatIndex::new(dim, nlist, metric);
                if config.nprobe > 0 {
                    bin_ivf_flat.set_nprobe(config.nprobe);
                }
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: Some(bin_ivf_flat),
                    sparse_inverted: None,
                    dim,
                    nprobe: 8,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::SparseInverted => {
                use crate::faiss::sparse_inverted::SparseMetricType;
                if metric != MetricType::Ip {
                    eprintln!(
                        "SparseInverted only supports InnerProduct metric; got metric_type={:?}",
                        metric
                    );
                    return None;
                }
                let sparse_inverted = crate::faiss::SparseInvertedIndex::new(SparseMetricType::Ip);
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: Some(sparse_inverted),
                    dim,
                    nprobe: 8,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::SparseWand => {
                // Sparse WAND index for efficient sparse vector search
                use crate::faiss::sparse_inverted::SparseMetricType;
                if metric != MetricType::Ip {
                    eprintln!(
                        "SparseWand only supports InnerProduct metric; got metric_type={:?}",
                        metric
                    );
                    return None;
                }
                let sparse_metric = SparseMetricType::Ip;
                let sparse_wand = crate::faiss::SparseWandIndex::new(sparse_metric);
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    dim,
                    nprobe: 8,
                    sparse_wand: Some(sparse_wand),
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::SparseWandCc => {
                // Sparse WAND CC (Concurrent) index
                use crate::faiss::sparse_inverted::SparseMetricType;
                if metric != MetricType::Ip {
                    eprintln!(
                        "SparseWandCc only supports InnerProduct metric; got metric_type={:?}",
                        metric
                    );
                    return None;
                }
                let sparse_metric = SparseMetricType::Ip;
                let ssize = if config.num_partitions > 0 {
                    config.num_partitions
                } else {
                    1000
                };
                let sparse_wand_cc = crate::faiss::SparseWandIndexCC::new(sparse_metric, ssize);
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    dim,
                    nprobe: 8,
                    sparse_wand: None,
                    sparse_wand_cc: Some(sparse_wand_cc),
                    minhash_lsh: None,
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::MinHashLsh => {
                // MinHash-LSH index for Jaccard similarity
                let minhash_lsh = crate::index::MinHashLSHIndex::new();
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,

                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    dim,
                    nprobe: 8,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: Some(minhash_lsh),
                    diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                })
            }
            CIndexType::DiskAnn => {
                use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
                let max_degree = if config.ef_construction > 0 {
                    config.ef_construction
                } else {
                    48
                };
                let search_list_size = if config.ef_search > 0 {
                    config.ef_search
                } else {
                    128
                };
                let beamwidth = if config.beamwidth > 0 { config.beamwidth } else { 8 };
                let aisaq_config = AisaqConfig {
                    max_degree,
                    search_list_size,
                    disk_pq_dims: config.disk_pq_dims,
                    pq_code_budget_gb: config.pq_code_budget_gb,
                    build_dram_budget_gb: config.build_dram_budget_gb,
                    beamwidth,
                    ..AisaqConfig::default()
                };
                let diskann = PQFlashIndex::new(aisaq_config, metric, dim).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,
                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: Some(diskann),
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim,
                    nprobe: 8,
                })
            }
            CIndexType::HnswPcaSq => {
                let pca_dim = if config.pca_dim > 0 { config.pca_dim } else { dim / 2 };
                let idx = crate::faiss::HnswPcaSqIndex::new(crate::faiss::HnswPcaSqConfig {
                    dim,
                    pca_dim,
                    m: config.ef_construction.min(32).max(4),
                    ef_construction: config.ef_construction.max(50),
                    ef_search: config.ef_search.max(10),
                });
                Some(Self {
                    flat: None, hnsw: None, scann: None, hnsw_prq: None,
                    hnsw_sq: None, hnsw_pq: None, ivf_pq: None,
                    bin_flat: None, binary_hnsw: None,
                    ivf_sq8: None, ivf_flat: None, bin_ivf_flat: None,
                    sparse_inverted: None, sparse_wand: None, sparse_wand_cc: None,
                    minhash_lsh: None, diskann: None,
                    hnsw_pca_sq: Some(idx),
                    hnsw_pca_usq: None,
                    diskann_pca_usq: None,
                    dim, nprobe: 8,
                })
            }
            CIndexType::HnswPcaUsq => {
                let pca_dim = if config.pca_dim > 0 { config.pca_dim } else { dim / 2 };
                let idx = crate::faiss::HnswPcaUsqIndex::new(crate::faiss::HnswPcaUsqConfig {
                    dim,
                    pca_dim,
                    bits_per_dim: 4,
                    rotation_seed: 42,
                }).map_err(|_| CError::Internal).ok()?;
                Some(Self {
                    flat: None, hnsw: None, scann: None, hnsw_prq: None,
                    hnsw_sq: None, hnsw_pq: None, ivf_pq: None,
                    bin_flat: None, binary_hnsw: None,
                    ivf_sq8: None, ivf_flat: None, bin_ivf_flat: None,
                    sparse_inverted: None, sparse_wand: None, sparse_wand_cc: None,
                    minhash_lsh: None, diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: Some(idx),
                    diskann_pca_usq: None,
                    dim, nprobe: 8,
                })
            }
            CIndexType::DiskAnnPcaUsq => {
                use crate::faiss::diskann_aisaq::AisaqConfig;
                let pca_dim = if config.pca_dim > 0 { config.pca_dim } else { dim / 2 };
                let max_degree = if config.ef_construction > 0 { config.ef_construction } else { 48 };
                let search_list_size = if config.ef_search > 0 { config.ef_search } else { 128 };
                let idx = crate::faiss::DiskAnnPcaUsqIndex::new(dim, metric,
                    crate::faiss::DiskAnnPcaUsqConfig {
                        base: AisaqConfig {
                            max_degree,
                            search_list_size,
                            beamwidth: if config.beamwidth > 0 { config.beamwidth } else { 8 },
                            ..AisaqConfig::default()
                        },
                        pca_dim,
                        bits_per_dim: 4,
                        rotation_seed: 42,
                        rerank_k: 64,
                    }
                ).map_err(|_| CError::Internal).ok()?;
                Some(Self {
                    flat: None, hnsw: None, scann: None, hnsw_prq: None,
                    hnsw_sq: None, hnsw_pq: None, ivf_pq: None,
                    bin_flat: None, binary_hnsw: None,
                    ivf_sq8: None, ivf_flat: None, bin_ivf_flat: None,
                    sparse_inverted: None, sparse_wand: None, sparse_wand_cc: None,
                    minhash_lsh: None, diskann: None,
                    hnsw_pca_sq: None,
                    hnsw_pca_usq: None,
                    diskann_pca_usq: Some(idx),
                    dim, nprobe: 8,
                })
            }
            _ => None,
        }
    }

    fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize, CError> {
        if let Some(ref mut idx) = self.flat {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw {
            let count = vectors.len() / idx.dim();
            let result = if Self::should_use_parallel_hnsw_add_via_ffi(idx, count) {
                idx.add_parallel(vectors, ids, Some(true))
                    .or_else(|_| idx.add(vectors, ids))
            } else {
                idx.add(vectors, ids)
            };
            result.map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.scann {
            // ScaNN uses interior mutability (RwLock)
            Ok(idx.add(vectors, ids))
        } else if let Some(ref mut idx) = self.hnsw_prq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_sq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_pq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_sq8 {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_flat {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_pq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.sparse_inverted {
            let dim = self.dim;
            let n_vectors = vectors.len() / dim;
            let ids_vec: Vec<i64> = if let Some(ids) = ids {
                ids.to_vec()
            } else {
                (0..n_vectors as i64).collect()
            };

            for (i, chunk) in vectors.chunks_exact(dim).enumerate() {
                let elements: Vec<crate::faiss::sparse_inverted::SparseVecElement> = chunk
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != 0.0)
                    .map(|(j, &v)| crate::faiss::sparse_inverted::SparseVecElement {
                        dim: j as u32,
                        val: v,
                    })
                    .collect();
                let sparse_vec = crate::faiss::sparse_inverted::SparseVector { elements };
                let doc_id = ids_vec.get(i).copied().unwrap_or(i as i64);
                if idx.add(&sparse_vec, doc_id).is_err() {
                    return Err(CError::Internal);
                }
            }
            Ok(n_vectors)
        } else if let Some(ref mut idx) = self.sparse_wand {
            // Sparse WAND: interpret vectors as sparse (dim, value) pairs
            // For simplicity, treat each vector as a sparse vector with non-zero elements
            let dim = self.dim;
            let n_vectors = vectors.len() / dim;
            let ids_vec: Vec<i64> = if let Some(ids) = ids {
                ids.to_vec()
            } else {
                (0..n_vectors as i64).collect()
            };

            for (i, chunk) in vectors.chunks_exact(dim).enumerate() {
                let elements: Vec<crate::faiss::sparse_inverted::SparseVecElement> = chunk
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != 0.0)
                    .map(|(j, &v)| crate::faiss::sparse_inverted::SparseVecElement {
                        dim: j as u32,
                        val: v,
                    })
                    .collect();
                let sparse_vec = crate::faiss::sparse_inverted::SparseVector { elements };
                let doc_id = ids_vec.get(i).copied().unwrap_or(i as i64);
                if idx.add(&sparse_vec, doc_id).is_err() {
                    return Err(CError::Internal);
                }
            }
            Ok(n_vectors)
        } else if let Some(ref mut idx) = self.diskann {
            let n_vectors = vectors.len() / self.dim;
            idx.add_with_ids(vectors, ids).map_err(|_| CError::Internal)?;
            Ok(n_vectors)
        } else if let Some(ref mut idx) = self.hnsw_pca_sq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_pca_usq {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.diskann_pca_usq {
            let n = vectors.len() / self.dim;
            idx.build(vectors, n, self.dim).map_err(|_| CError::Internal)?;
            Ok(n)
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// Add binary vectors (for BinFlat, BinaryHnsw, BinIvfFlat, and MinHashLSH)
    fn add_binary(&mut self, vectors: &[u8], ids: Option<&[i64]>) -> Result<usize, CError> {
        if let Some(ref mut idx) = self.bin_flat {
            let dim_bytes = idx.dim().div_ceil(8);
            let n = vectors.len() / dim_bytes;
            idx.add(n as u32, vectors, ids)
                .map_err(|_| CError::Internal)?;
            Ok(n)
        } else if let Some(ref mut idx) = self.binary_hnsw {
            // BinaryHnswIndex::add returns the number of vectors added
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.bin_ivf_flat {
            let dim_bytes = idx.dim().div_ceil(8);
            let n = vectors.len() / dim_bytes;
            idx.add(n as u32, vectors, ids)
                .map_err(|_| CError::Internal)?;
            Ok(n)
        } else if let Some(ref mut idx) = self.minhash_lsh {
            // MinHashLSH: build index from binary data
            // C++ parity: vector byte size should follow configured dim bits.
            // Use u64 elements by default (8 bytes each), and derive mh_vec_length from dim.
            let mh_vec_element_size = std::mem::size_of::<u64>();
            let vector_bytes = self.dim.div_ceil(8);
            if vector_bytes == 0 || vectors.is_empty() || vectors.len() % vector_bytes != 0 {
                return Err(CError::InvalidArg);
            }
            if vector_bytes % mh_vec_element_size != 0 {
                return Err(CError::InvalidArg);
            }

            let mh_vec_length = vector_bytes / mh_vec_element_size;
            if mh_vec_length == 0 {
                return Err(CError::InvalidArg);
            }

            let bands = 4;
            let with_raw_data = true;
            idx.build(
                vectors,
                mh_vec_length,
                mh_vec_element_size,
                bands,
                with_raw_data,
            )
            .map_err(|_| CError::Internal)?;
            Ok(idx.count())
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
        } else if let Some(ref mut idx) = self.hnsw_sq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw_pq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_sq8 {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_flat {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_pq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if self.sparse_inverted.is_some() || self.sparse_wand.is_some() {
            let _ = vectors;
            Ok(())
        } else if let Some(ref mut idx) = self.diskann {
            idx.train(vectors).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_pca_sq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw_pca_usq {
            idx.train(vectors).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.diskann_pca_usq {
            // DiskAnnPcaUsq train is done inside build()
            let _ = vectors;
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
            // ScaNN FFI does not accept per-search ef_search; reorder_k is the search budget.
            let results = idx.search(query, top_k);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let (ids, distances): (Vec<i64>, Vec<f32>) = results.into_iter().unzip();
            let _num_visited = ids.len();
            Ok(ApiSearchResult::new(ids, distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_prq {
            let start = std::time::Instant::now();
            let results = idx
                .search(query, top_k, None)
                .map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.hnsw_sq {
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.hnsw_pq {
            let start = std::time::Instant::now();
            let results = idx
                .search(query, top_k, None)
                .map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.ivf_sq8 {
            let req = SearchRequest {
                top_k,
                nprobe: self.nprobe,
                filter: None,
                params: None,
                radius: None,
            };
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.ivf_flat {
            let req = SearchRequest {
                top_k,
                nprobe: self.nprobe,
                filter: None,
                params: None,
                radius: None,
            };
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.ivf_pq {
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
            Ok(ApiSearchResult::new(
                results.ids,
                results.distances,
                elapsed_ms,
            ))
        } else if let Some(ref idx) = self.sparse_inverted {
            self.search_sparse_queries(query, self.dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = self.sparse_wand {
            self.search_sparse_queries(query, self.dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = self.diskann {
            let start = std::time::Instant::now();
            let result = idx.search_batch(query, top_k).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(result.ids, result.distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_pca_sq {
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.hnsw_pca_usq {
            let start = std::time::Instant::now();
            let results = idx.search(query, &req).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(results.ids, results.distances, elapsed_ms))
        } else if let Some(ref idx) = self.diskann_pca_usq {
            let start = std::time::Instant::now();
            let results = idx.search(query, top_k).map_err(|_| CError::Internal)?;
            let mut ids = Vec::with_capacity(top_k);
            let mut dists = Vec::with_capacity(top_k);
            for (dist, id) in results.into_iter().take(top_k) {
                ids.push(id as i64);
                dists.push(dist);
            }
            while ids.len() < top_k { ids.push(-1); dists.push(f32::MAX); }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(ids, dists, elapsed_ms))
        } else {
            Err(CError::InvalidArg)
        }
    }

    fn set_ef_search(&mut self, ef_search: usize) -> Result<(), CError> {
        if let Some(ref mut idx) = self.hnsw {
            idx.set_ef_search(ef_search);
            Ok(())
        } else if let Some(ref mut idx) = self.diskann {
            idx.set_search_list_size(ef_search);
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }

    fn set_nprobe(&mut self, nprobe: usize) -> Result<(), CError> {
        if self.ivf_sq8.is_some() || self.ivf_flat.is_some() {
            self.nprobe = nprobe;
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }

    /// Search binary vectors (for BinFlat, BinaryHnsw, BinIvfFlat, and MinHashLSH)
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
        } else if let Some(ref idx) = self.minhash_lsh {
            // MinHashLSH search
            let start = std::time::Instant::now();
            let (ids, distances) = idx
                .search(query, top_k, None)
                .map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(ids, distances, elapsed_ms))
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
    fn range_search(&self, query: &[f32], radius: f32) -> Result<RangeSearchOutcome, CError> {
        let num_queries = query.len() / self.dim;

        if let Some(ref idx) = self.flat {
            let start = std::time::Instant::now();
            let (ids, distances) = idx
                .range_search(query, radius)
                .map_err(|_| CError::Internal)?;
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
            // TODO: wire HNSW range_search once available in src/faiss/hnsw.rs.
            eprintln!("HNSW range_search not yet implemented");
            Err(CError::InvalidArg)
        } else if let Some(ref _idx) = self.scann {
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
        } else if let Some(ref idx) = self.hnsw_sq {
            idx.count()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.count()
        } else if let Some(ref idx) = self.ivf_sq8 {
            idx.ntotal()
        } else if let Some(ref idx) = self.ivf_flat {
            idx.ntotal()
        } else if let Some(ref idx) = self.ivf_pq {
            idx.ntotal()
        } else if let Some(ref idx) = self.sparse_inverted {
            idx.n_rows()
        } else if let Some(ref idx) = self.sparse_wand {
            idx.n_rows()
        } else if let Some(ref idx) = self.sparse_wand_cc {
            idx.n_rows()
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.count()
        } else {
            0
        }
    }

    fn is_trained(&self) -> bool {
        if let Some(ref idx) = self.hnsw {
            idx.is_trained()
        } else if let Some(ref idx) = self.scann {
            idx.is_trained()
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.is_trained()
        } else if let Some(ref idx) = self.ivf_pq {
            idx.is_trained()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.is_trained()
        } else if let Some(ref idx) = self.ivf_sq8 {
            idx.is_trained()
        } else if let Some(ref idx) = self.ivf_flat {
            idx.is_trained()
        } else if let Some(ref idx) = self.sparse_inverted {
            idx.is_trained()
        } else if let Some(ref idx) = self.sparse_wand {
            idx.is_trained()
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.is_trained()
        } else {
            self.count() > 0
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
        } else if let Some(ref idx) = self.hnsw_sq {
            idx.size()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.size()
        } else if let Some(ref idx) = self.ivf_sq8 {
            // IvfSq8Index doesn't have size() method yet, estimate based on stored data
            // Use config.dim instead of private field
            idx.ntotal() * 8 // SQ8 uses 8 bits per dimension
        } else if let Some(ref idx) = self.ivf_flat {
            idx.ntotal() * self.dim * std::mem::size_of::<f32>()
        } else if let Some(ref idx) = self.ivf_pq {
            idx.ntotal() * self.dim
        } else if let Some(ref idx) = self.sparse_inverted {
            idx.size()
        } else if let Some(ref idx) = self.sparse_wand {
            idx.size()
        } else if let Some(ref idx) = self.sparse_wand_cc {
            idx.size()
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.memory_usage()
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
        } else if self.hnsw_prq.is_some() {
            "HNSW_PRQ"
        } else if self.hnsw_sq.is_some() {
            "HNSW_SQ"
        } else if self.hnsw_pq.is_some() {
            "HNSW_PQ"
        } else if self.bin_flat.is_some() {
            "BinFlat"
        } else if self.binary_hnsw.is_some() {
            "BinaryHNSW"
        } else if self.ivf_sq8.is_some() {
            "IVF_SQ8"
        } else if self.ivf_flat.is_some() {
            "IVF_FLAT"
        } else if self.ivf_pq.is_some() {
            "IVF_PQ"
        } else if self.bin_ivf_flat.is_some() {
            "BinIVFFlat"
        } else if self.sparse_inverted.is_some() {
            "SparseInverted"
        } else if self.sparse_wand.is_some() {
            "SparseWand"
        } else if self.sparse_wand_cc.is_some() {
            "SparseWandCC"
        } else if self.minhash_lsh.is_some() {
            "MinHashLSH"
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
        } else if let Some(ref _idx) = self.scann {
            // ScaNN doesn't expose metric_type directly, assume L2
            "L2"
        } else if self.sparse_inverted.is_some()
            || self.sparse_wand.is_some()
            || self.sparse_wand_cc.is_some()
        {
            "IP"
        } else {
            "Unknown"
        }
    }

    fn has_raw_data(&self) -> bool {
        if let Some(ref idx) = self.flat {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.hnsw {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.scann {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.ivf_sq8 {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.ivf_flat {
            idx.has_raw_data()
        } else if self.ivf_pq.is_some() {
            false
        } else if let Some(ref idx) = self.sparse_inverted {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.sparse_wand {
            idx.has_raw_data()
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.has_raw_data()
        } else {
            false
        }
    }

    fn additional_scalar_support_mode(&self) -> &'static str {
        if self.sparse_inverted.is_some()
            || self.sparse_wand.is_some()
            || self.sparse_wand_cc.is_some()
        {
            "partial"
        } else {
            "unsupported"
        }
    }

    fn additional_scalar_unsupported_reason(&self) -> &'static str {
        match self.additional_scalar_support_mode() {
            "supported" => "",
            "partial" => "only sparse indexes expose MV-only additional-scalar filtering via the current Rust FFI",
            _ if self.hnsw.is_some() => "HNSW does not expose additional-scalar filtering through the current Rust FFI",
            _ if self.ivf_sq8.is_some()
                || self.ivf_flat.is_some()
                || self.ivf_pq.is_some() =>
            {
                "IVF variants do not expose additional-scalar filtering through the current Rust FFI"
            }
            _ if self.scann.is_some() => "ScaNN does not expose additional-scalar filtering through the current Rust FFI",
            _ => "additional-scalar filtering is unsupported for this index type in the current Rust FFI",
        }
    }

    fn is_additional_scalar_supported(&self, is_mv_only: bool) -> bool {
        matches!(
            self.additional_scalar_support_mode(),
            "partial" | "supported"
        ) && is_mv_only
    }

    fn capability_summary(&self) -> IndexCapabilitySummary<'static> {
        if self.flat.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: "supported",
                ann_iterator: "unsupported",
                persistence: "supported",
            }
        } else if self.hnsw.is_some() || self.scann.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "supported",
                persistence: "supported",
            }
        } else if self.hnsw_prq.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "unsupported",
                persistence: "supported",
            }
        } else if self.ivf_pq.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "unsupported",
                persistence: "supported",
            }
        } else if self.hnsw_sq.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "unsupported",
                persistence: "unsupported",
            }
        } else if self.hnsw_pq.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "supported",
                persistence: "unsupported",
            }
        } else if self.bin_flat.is_some()
            || self.binary_hnsw.is_some()
            || self.bin_ivf_flat.is_some()
        {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "unsupported",
                persistence: "unsupported",
            }
        } else if self.ivf_sq8.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "unsupported",
                persistence: "unsupported",
            }
        } else if self.ivf_flat.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "unsupported",
                persistence: "supported",
            }
        } else if self.sparse_inverted.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "supported",
                persistence: "supported",
            }
        } else if self.sparse_wand.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "supported",
                persistence: "supported",
            }
        } else if self.sparse_wand_cc.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "unsupported",
                persistence: "unsupported",
            }
        } else if self.minhash_lsh.is_some() {
            IndexCapabilitySummary {
                get_vector_by_ids: if self.has_raw_data() {
                    "supported"
                } else {
                    "unsupported"
                },
                ann_iterator: "supported",
                persistence: "supported",
            }
        } else {
            IndexCapabilitySummary {
                get_vector_by_ids: "unsupported",
                ann_iterator: "unsupported",
                persistence: "unsupported",
            }
        }
    }

    fn persistence_semantics(&self) -> PersistenceSemantics<'static> {
        if self.flat.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else if self.scann.is_some() || self.hnsw_prq.is_some() || self.minhash_lsh.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "unsupported",
                deserialize_from_file: "supported",
            }
        } else if self.hnsw.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else if self.ivf_sq8.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else if self.ivf_flat.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else if self.ivf_pq.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else if self.sparse_inverted.is_some() || self.sparse_wand.is_some() {
            PersistenceSemantics {
                file_save_load: "supported",
                memory_serialize: "supported",
                deserialize_from_file: "supported",
            }
        } else {
            PersistenceSemantics {
                file_save_load: "unsupported",
                memory_serialize: "unsupported",
                deserialize_from_file: "unsupported",
            }
        }
    }

    fn meta_semantics(&self) -> IndexMetaSemantics<'static> {
        let persistence = self.persistence_semantics();
        let persistence_mode = if persistence.file_save_load == "supported"
            && persistence.memory_serialize == "supported"
        {
            "file_save_load+memory_serialize"
        } else if persistence.file_save_load == "supported" {
            "file_save_load"
        } else if persistence.file_save_load == "constrained" {
            "constrained_file_save_load"
        } else if persistence.memory_serialize == "supported" {
            "memory_serialize"
        } else {
            "unsupported"
        };

        if self.hnsw.is_some()
            || self.hnsw_prq.is_some()
            || self.hnsw_sq.is_some()
            || self.hnsw_pq.is_some()
        {
            IndexMetaSemantics {
                family: "hnsw",
                raw_data_gate: if self.has_raw_data() {
                    "raw_vectors_retained"
                } else {
                    "compressed_or_graph_only"
                },
                persistence_mode,
                persistence,
                metadata_granularity: "per-index-capability",
            }
        } else if self.ivf_sq8.is_some() || self.ivf_flat.is_some() || self.ivf_pq.is_some() {
            IndexMetaSemantics {
                family: "ivf",
                raw_data_gate: if self.has_raw_data() {
                    "raw_vectors_retained"
                } else {
                    "quantized_or_codebook_only"
                },
                persistence_mode,
                persistence,
                metadata_granularity: "per-index-capability",
            }
        } else if self.scann.is_some() {
            IndexMetaSemantics {
                family: "scann",
                raw_data_gate: if self.has_raw_data() {
                    "raw_vectors_retained"
                } else {
                    "partition_or_reorder_only"
                },
                persistence_mode,
                persistence,
                metadata_granularity: "per-index-capability",
            }
        } else if self.sparse_inverted.is_some()
            || self.sparse_wand.is_some()
            || self.sparse_wand_cc.is_some()
        {
            IndexMetaSemantics {
                family: "sparse",
                raw_data_gate: if self.has_raw_data() {
                    "sparse_postings_retained"
                } else {
                    "wand_state_only"
                },
                persistence_mode,
                persistence,
                metadata_granularity: "per-index-capability",
            }
        } else {
            IndexMetaSemantics {
                family: "generic",
                raw_data_gate: if self.has_raw_data() {
                    "raw_vectors_retained"
                } else {
                    "not_retained"
                },
                persistence_mode,
                persistence,
                metadata_granularity: "uniform-summary",
            }
        }
    }

    fn runtime_observability_summary(&self) -> RuntimeObservabilitySummary<'static> {
        RuntimeObservabilitySummary {
            schema_version: "runtime_observability.v1",
            build_event: "knowhere.index.build",
            search_event: "knowhere.index.search",
            load_event: "knowhere.index.load",
            required_fields: &["index_type", "dim", "count", "latency_ms"],
            optional_fields: &[
                "topk",
                "query_count",
                "trace_id",
                "span_id",
                "ground_truth_source",
                "recall_at_10",
                "artifact_path",
                "mmap_load",
            ],
        }
    }

    fn trace_propagation_summary(&self) -> TracePropagationSummary<'static> {
        TracePropagationSummary {
            ffi_entrypoint: "index_meta.trace_context_json",
            gate_runner_entrypoint: "OPENCLAW_TRACE_CONTEXT_JSON",
            context_encoding: "w3c-traceparent-json",
            propagation_mode: "optional_passthrough",
        }
    }

    fn resource_contract_summary(&self) -> ResourceContractSummary<'static> {
        let capability = self.capability_summary();
        let persistence = self.persistence_semantics();
        let mmap_supported = matches!(persistence.file_save_load, "supported" | "constrained");
        let unsupported_reason = if mmap_supported {
            ""
        } else if capability.persistence == "unsupported" {
            "this index family does not expose a stable file-backed persistence contract, so mmap load is not auditable"
        } else {
            "mmap support remains undefined until a stable file-backed load contract exists"
        };

        ResourceContractSummary {
            schema_version: "resource_contract.v1",
            memory_bytes: if self.has_raw_data() {
                "estimated_runtime_memory_bytes"
            } else {
                "estimated_runtime_memory_bytes_or_codebook_only"
            },
            disk_bytes: if mmap_supported {
                "estimated_file_bytes"
            } else {
                "unsupported"
            },
            mmap_supported,
            unsupported_reason,
        }
    }

    fn get_index_meta_json(&self) -> Result<String, CError> {
        let additional_scalar = AdditionalScalarMeta {
            runtime_supported: self.is_additional_scalar_supported(true),
            mv_only_query: true,
            support_mode: self.additional_scalar_support_mode(),
            unsupported_reason: self.additional_scalar_unsupported_reason(),
        };
        let summary = IndexMetaSummary {
            index_type: self.index_type(),
            dim: self.dim(),
            count: self.count(),
            is_trained: self.is_trained(),
            has_raw_data: self.has_raw_data(),
            additional_scalar_supported: additional_scalar.runtime_supported,
            additional_scalar,
            capabilities: self.capability_summary(),
            semantics: self.meta_semantics(),
            observability: self.runtime_observability_summary(),
            trace_propagation: self.trace_propagation_summary(),
            resource_contract: self.resource_contract_summary(),
        };

        serde_json::to_string(&summary).map_err(|_| CError::Internal)
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
        } else if let Some(ref idx) = self.hnsw {
            match idx.get_vector_by_ids(ids) {
                Ok(vectors) => {
                    let num_found = vectors.len() / self.dim;
                    Ok((vectors, num_found))
                }
                Err(_) => Err(CError::NotFound),
            }
        } else if let Some(ref idx) = self.scann {
            match idx.get_vector_by_ids(ids) {
                Ok(vectors) => {
                    let num_found = vectors.len() / self.dim;
                    Ok((vectors, num_found))
                }
                Err(_) => Err(CError::NotFound),
            }
        } else if let Some(ref idx) = self.sparse_inverted {
            let mut vectors = Vec::with_capacity(ids.len() * self.dim);
            let mut num_found = 0usize;
            for &id in ids {
                let sparse = idx.get_vector_by_id(id).ok_or(CError::NotFound)?;
                let mut dense = vec![0.0f32; self.dim];
                for elem in sparse.elements {
                    let dim = elem.dim as usize;
                    if dim < self.dim {
                        dense[dim] = elem.val;
                    }
                }
                vectors.extend_from_slice(&dense);
                num_found += 1;
            }
            Ok((vectors, num_found))
        } else if let Some(ref idx) = self.minhash_lsh {
            // MinHashLSH: get vectors by IDs (returns byte data)
            match idx.get_vector_by_ids(ids) {
                Ok(vectors) => {
                    // Convert byte data to f32 for compatibility
                    let num_found = vectors.len() / std::mem::size_of::<f32>();
                    let mut f32_vectors = Vec::with_capacity(num_found);
                    for chunk in vectors.chunks_exact(std::mem::size_of::<f32>()) {
                        let val = f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4]));
                        f32_vectors.push(val);
                    }
                    Ok((f32_vectors, num_found))
                }
                Err(_) => Err(CError::NotFound),
            }
        } else if let Some(ref idx) = self.ivf_flat {
            let rows = idx.get_vectors(ids);
            let mut vectors = Vec::new();
            let mut num_found = 0usize;
            for row in rows {
                if let Some(v) = row {
                    vectors.extend_from_slice(&v);
                    num_found += 1;
                }
            }
            Ok((vectors, num_found))
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
        } else if let Some(ref idx) = self.ivf_sq8 {
            idx.serialize_to_bytes().map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.ivf_flat {
            idx.serialize_to_bytes().map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw {
            idx.serialize_to_bytes().map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.ivf_pq {
            idx.serialize_to_bytes().map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.sparse_inverted {
            let payload = idx.serialize_to_bytes().map_err(|_| CError::Internal)?;
            self.serialize_sparse_payload(payload)
        } else if let Some(ref idx) = self.sparse_wand {
            let payload = idx.serialize_to_bytes().map_err(|_| CError::Internal)?;
            self.serialize_sparse_payload(payload)
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
        let sparse_fallback_dim = self.dim.max(1);
        if let Some(ref mut idx) = self.flat {
            idx.deserialize_from_memory(data)
                .map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_sq8 {
            let loaded = crate::faiss::IvfSq8Index::deserialize_from_bytes(data, idx.dim())
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_flat {
            let loaded = crate::faiss::IvfFlatIndex::deserialize_from_bytes(data, idx.dim())
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            Ok(())
        } else if let Some(ref mut idx) = self.hnsw {
            let loaded = crate::faiss::HnswIndex::deserialize_from_bytes(data)
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            Ok(())
        } else if let Some(ref mut idx) = self.ivf_pq {
            let loaded = crate::faiss::IvfPqIndex::deserialize_from_bytes(data)
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            Ok(())
        } else if let Some(ref mut idx) = self.sparse_inverted {
            let (dim, payload) = Self::deserialize_sparse_payload(sparse_fallback_dim, data)?;
            let loaded = crate::faiss::SparseInvertedIndex::deserialize_from_bytes(&payload)
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            self.dim = dim;
            Ok(())
        } else if let Some(ref mut idx) = self.sparse_wand {
            let (dim, payload) = Self::deserialize_sparse_payload(sparse_fallback_dim, data)?;
            let loaded = crate::faiss::SparseWandIndex::deserialize_from_bytes(&payload)
                .map_err(|_| CError::Internal)?;
            *idx = loaded;
            self.dim = dim;
            Ok(())
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
        } else if let Some(ref idx) = self.sparse_inverted {
            let _ = idx;
            let bytes = self.serialize()?;
            std::fs::write(path, bytes).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.sparse_wand {
            let _ = idx;
            let bytes = self.serialize()?;
            std::fs::write(path, bytes).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.ivf_flat {
            idx.save(path).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.scann {
            idx.save(path.to_str().unwrap())
                .map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw_prq {
            idx.save(path).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.ivf_pq {
            idx.save(path).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.save(path.to_str().unwrap())
                .map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.diskann {
            idx.save(path).map(|_| ()).map_err(|_| CError::Internal)
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
        } else if let Some(ref mut idx) = self.sparse_inverted {
            let _ = idx;
            let bytes = std::fs::read(path).map_err(|_| CError::Internal)?;
            self.deserialize(&bytes)
        } else if let Some(ref mut idx) = self.sparse_wand {
            let _ = idx;
            let bytes = std::fs::read(path).map_err(|_| CError::Internal)?;
            self.deserialize(&bytes)
        } else if let Some(ref mut idx) = self.ivf_flat {
            *idx =
                crate::faiss::IvfFlatIndex::load(path, idx.dim()).map_err(|_| CError::Internal)?;
            Ok(())
        } else if let Some(ref mut idx) = self.scann {
            idx.load(path.to_str().unwrap())
                .map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw_prq {
            idx.load(path.to_str().unwrap())
                .map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.ivf_pq {
            idx.load(path).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.minhash_lsh {
            idx.load(path.to_str().unwrap())
                .map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.diskann {
            *idx = crate::faiss::diskann_aisaq::PQFlashIndex::load(path)
                .map_err(|_| CError::Internal)?;
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }

    fn serialize_sparse_payload(&self, payload: Vec<u8>) -> Result<Vec<u8>, CError> {
        let envelope = SparsePersistenceEnvelope {
            version: SPARSE_PERSISTENCE_ENVELOPE_VERSION,
            dim: self.dim.max(1),
            payload,
        };
        bincode::serialize(&envelope).map_err(|_| CError::Internal)
    }

    fn deserialize_sparse_payload(
        fallback_dim: usize,
        data: &[u8],
    ) -> Result<(usize, Vec<u8>), CError> {
        match bincode::deserialize::<SparsePersistenceEnvelope>(data) {
            Ok(envelope) if envelope.version == SPARSE_PERSISTENCE_ENVELOPE_VERSION => {
                Ok((envelope.dim.max(1), envelope.payload))
            }
            Ok(_) | Err(_) => Ok((fallback_dim.max(1), data.to_vec())),
        }
    }

    /// 创建 ANN 迭代器 (AnnIterator)
    ///
    /// 用于流式返回最近邻结果，支持更灵活的搜索控制。
    /// 目前只支持实现了 Index trait 的索引（HNSW, ScaNN, HNSW-PQ）。
    fn create_ann_iterator(
        &self,
        query: &crate::dataset::Dataset,
        bitset: Option<&crate::bitset::BitsetView>,
    ) -> Result<Box<dyn crate::index::AnnIterator>, CError> {
        if let Some(ref idx) = self.hnsw {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else if let Some(ref idx) = self.scann {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else if let Some(ref idx) = self.sparse_inverted {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else if let Some(ref idx) = self.hnsw_pq {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else if let Some(ref idx) = self.minhash_lsh {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else if let Some(ref idx) = self.sparse_wand {
            idx.create_ann_iterator(query, bitset)
                .map_err(|_| CError::NotImplemented)
        } else {
            Err(CError::NotImplemented)
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
            let _ = Box::from_raw(index as *mut IndexWrapper);
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

        let t0 = std::time::Instant::now();

        let result = if let Some(ref idx) = index.sparse_inverted {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = index.sparse_wand {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = index.sparse_wand_cc {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else {
            index.search(query_slice, top_k)
        };

        if std::env::var_os("HANNS_TRACE_SEARCH").is_some() {
            eprintln!(
                "TRACE_SEARCH nq={} elapsed_us={}",
                count,
                t0.elapsed().as_micros()
            );
        }

        match result {
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

/// Override HNSW search-time ef on an existing index handle.
#[no_mangle]
pub extern "C" fn knowhere_set_ef_search(index: *mut std::ffi::c_void, ef_search: usize) -> i32 {
    if index.is_null() || ef_search == 0 {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let wrapper = &mut *(index as *mut IndexWrapper);
        match wrapper.set_ef_search(ef_search) {
            Ok(()) => CError::Success as i32,
            Err(err) => err as i32,
        }
    }
}

/// Override IVF search-time nprobe on an existing index handle.
#[no_mangle]
pub extern "C" fn knowhere_set_nprobe(index: *mut std::ffi::c_void, nprobe: usize) -> i32 {
    if index.is_null() || nprobe == 0 {
        return CError::InvalidArg as i32;
    }
    unsafe {
        let wrapper = &mut *(index as *mut IndexWrapper);
        match wrapper.set_nprobe(nprobe) {
            Ok(()) => CError::Success as i32,
            Err(err) => err as i32,
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
    if index.is_null()
        || query.is_null()
        || count == 0
        || top_k == 0
        || dim == 0
        || bitset.is_null()
    {
        return std::ptr::null_mut();
    }

    unsafe {
        let index = &*(index as *const IndexWrapper);
        let bitset_wrapper = &*bitset;

        let query_slice = std::slice::from_raw_parts(query, count * dim);

        // Zero-copy bitset for HNSW path: borrow CBitset words directly
        let bitset_words = std::slice::from_raw_parts(
            bitset_wrapper.data,
            bitset_wrapper.len.div_ceil(64),
        );
        let bitset_ref = crate::bitset::BitsetRef::new(bitset_words, bitset_wrapper.len);

        let req = SearchRequest {
            top_k,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };

        if let Some(ref idx) = index.flat {
            let bitset_view =
                crate::bitset::BitsetView::from_vec(bitset_words.to_vec(), bitset_wrapper.len);
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
        } else if let Some(ref idx) = index.hnsw {
            let hnsw_result = idx.search_with_bitset_ref(query_slice, &req, &bitset_ref);

            match hnsw_result {
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
        } else if let Some(ref idx) = index.sparse_inverted {
            let bitset_view =
                crate::bitset::BitsetView::from_vec(bitset_words.to_vec(), bitset_wrapper.len);
            let sparse_bitset = crate::faiss::sparse_inverted::bitset_to_bool_vec(&bitset_view);
            match index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, Some(&sparse_bitset))
            }) {
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
        } else if let Some(ref idx) = index.sparse_wand {
            let bitset_view =
                crate::bitset::BitsetView::from_vec(bitset_words.to_vec(), bitset_wrapper.len);
            let sparse_bitset = crate::faiss::sparse_inverted::bitset_to_bool_vec(&bitset_view);
            match index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, Some(&sparse_bitset))
            }) {
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
        } else if let Some(ref idx) = index.sparse_wand_cc {
            let bitset_view =
                crate::bitset::BitsetView::from_vec(bitset_words.to_vec(), bitset_wrapper.len);
            let sparse_bitset = crate::faiss::sparse_inverted::bitset_to_bool_vec(&bitset_view);
            match index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, Some(&sparse_bitset))
            }) {
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
        } else if let Some(ref idx) = index.diskann {
            let bitset_view =
                crate::bitset::BitsetView::from_vec(bitset_words.to_vec(), bitset_wrapper.len);
            match idx.search_batch_with_bitset(query_slice, top_k, &bitset_view) {
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
            // Do not silently drop bitset on unsupported index types.
            eprintln!("search_with_bitset not supported for this index type");
            std::ptr::null_mut()
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
pub extern "C" fn knowhere_get_index_type(
    index: *const std::ffi::c_void,
) -> *const std::os::raw::c_char {
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
        "HNSW_PRQ" => b"HNSW_PRQ\0".as_ptr() as *const std::os::raw::c_char,
        "IVF_RABITQ" => b"IVF_RABITQ\0".as_ptr() as *const std::os::raw::c_char,
        "HNSW_SQ" => b"HNSW_SQ\0".as_ptr() as *const std::os::raw::c_char,
        "HNSW_PQ" => b"HNSW_PQ\0".as_ptr() as *const std::os::raw::c_char,
        "BinFlat" => b"BinFlat\0".as_ptr() as *const std::os::raw::c_char,
        "BinaryHNSW" => b"BinaryHNSW\0".as_ptr() as *const std::os::raw::c_char,
        "IVF_SQ8" => b"IVF_SQ8\0".as_ptr() as *const std::os::raw::c_char,
        "BinIVFFlat" => b"BinIVFFlat\0".as_ptr() as *const std::os::raw::c_char,
        "SparseInverted" => b"SparseInverted\0".as_ptr() as *const std::os::raw::c_char,
        "SparseWand" => b"SparseWand\0".as_ptr() as *const std::os::raw::c_char,
        "SparseWandCC" => b"SparseWandCC\0".as_ptr() as *const std::os::raw::c_char,
        "MinHashLSH" => b"MinHashLSH\0".as_ptr() as *const std::os::raw::c_char,
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
pub extern "C" fn knowhere_get_index_metric(
    index: *const std::ffi::c_void,
) -> *const std::os::raw::c_char {
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
        if wrapper.has_raw_data() {
            1
        } else {
            0
        }
    }
}

/// 检查索引是否支持附加标量能力 (FFI ABI metadata contract)
#[no_mangle]
pub extern "C" fn knowhere_is_additional_scalar_supported(
    index: *const std::ffi::c_void,
    is_mv_only: bool,
) -> i32 {
    if index.is_null() {
        return 0;
    }

    unsafe {
        let wrapper = &*(index as *const IndexWrapper);
        if wrapper.is_additional_scalar_supported(is_mv_only) {
            1
        } else {
            0
        }
    }
}

/// 获取索引元数据 JSON (FFI ABI metadata contract)
///
/// 返回一段由 Rust 分配的 UTF-8 JSON 字符串；调用方需使用
/// `knowhere_free_cstring` 释放。
#[no_mangle]
pub extern "C" fn knowhere_get_index_meta(
    index: *const std::ffi::c_void,
) -> *mut std::os::raw::c_char {
    if index.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let wrapper = &*(index as *const IndexWrapper);
        match wrapper.get_index_meta_json() {
            Ok(json) => match std::ffi::CString::new(json) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => std::ptr::null_mut(),
            },
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 释放由 FFI 返回的 C 字符串
#[no_mangle]
pub extern "C" fn knowhere_free_cstring(ptr: *mut std::os::raw::c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}

/// 释放搜索结果
#[no_mangle]
pub extern "C" fn knowhere_free_result(result: *mut CSearchResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            if !r.ids.is_null() {
                drop(Vec::from_raw_parts(r.ids, r.num_results, r.num_results));
            }
            if !r.distances.is_null() {
                drop(Vec::from_raw_parts(
                    r.distances,
                    r.num_results,
                    r.num_results,
                ));
            }
            let _ = Box::from_raw(result);
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
                drop(Vec::from_raw_parts(
                    r.vectors,
                    r.num_vectors * r.dim,
                    r.num_vectors * r.dim,
                ));
            }
            if !r.ids.is_null() && r.num_vectors > 0 {
                drop(Vec::from_raw_parts(r.ids, r.num_vectors, r.num_vectors));
            }
            let _ = Box::from_raw(result);
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
                drop(Vec::from_raw_parts(
                    r.vectors as *mut f32,
                    r.num_ids * r.dim,
                    r.num_ids * r.dim,
                ));
            }
            // 释放 ids 数组
            if !r.ids.is_null() && r.num_ids > 0 {
                drop(Vec::from_raw_parts(r.ids, r.num_ids, r.num_ids));
            }
            // 释放结果结构体本身
            let _ = Box::from_raw(result);
        }
    }
}

// ========== AnnIterator C API ==========

/// ANN 迭代器句柄
///
/// 用于流式返回最近邻结果，支持更灵活的搜索控制。
/// 对应 C++ knowhere 的 AnnIterator 接口。
#[repr(C)]
pub struct CAnnIterator {
    /// 内部迭代器指针（类型擦除）
    inner: *mut std::ffi::c_void,
}

/// 创建 ANN 迭代器
///
/// 用于流式返回最近邻结果，支持更灵活的搜索控制。
///
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// * `query` - 查询向量数组 (n * dim)
/// * `n` - 查询向量数量
/// * `dim` - 向量维度
/// * `bitset` - 可选的 Bitset 指针 (用于过滤向量，可为 NULL)
///
/// # Returns
/// 返回 CAnnIterator 指针，失败返回 NULL
///
/// # Safety
/// 调用者需确保：
/// - index 指针有效
/// - query 指向有效的内存，长度为 n * dim
/// - 使用 knowhere_free_ann_iterator() 释放返回的迭代器
///
/// # Example
/// ```c
/// CIndex* index = knowhere_create_index(config);
/// float query[] = { ... };  // 1 * 128 dim
/// CAnnIterator* iter = knowhere_create_ann_iterator(index, query, 1, 128, NULL);
///
/// int64_t id;
/// float dist;
/// while (knowhere_ann_iterator_next(iter, &id, &dist)) {
///     printf("id=%ld, dist=%f\n", id, dist);
/// }
///
/// knowhere_free_ann_iterator(iter);
/// ```
#[no_mangle]
pub extern "C" fn knowhere_create_ann_iterator(
    index: *const std::ffi::c_void,
    query: *const f32,
    n: usize,
    dim: usize,
    bitset: *const CBitset,
) -> *mut CAnnIterator {
    if index.is_null() || query.is_null() || n == 0 || dim == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let index_wrapper = &*(index as *const IndexWrapper);

        // 验证维度匹配
        if index_wrapper.dim() != dim {
            return std::ptr::null_mut();
        }

        let query_vec = std::slice::from_raw_parts(query, n * dim).to_vec();
        let dataset = crate::dataset::Dataset::from_vectors(query_vec, dim);

        // 准备 bitset（如果提供）
        let bitset_ref = if bitset.is_null() {
            None
        } else {
            Some(&*(bitset as *const crate::bitset::BitsetView))
        };

        // 尝试创建迭代器（目前只支持实现了 Index trait 的索引）
        let iter_result = index_wrapper.create_ann_iterator(&dataset, bitset_ref);

        match iter_result {
            Ok(iter) => {
                // 将迭代器装箱并转换为原始指针
                let boxed = Box::new(iter);
                let raw = Box::into_raw(boxed);

                let result = CAnnIterator {
                    inner: raw as *mut std::ffi::c_void,
                };

                Box::into_raw(Box::new(result))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 获取下一个 ANN 结果
///
/// 从迭代器中获取下一个最近邻结果。
///
/// # Arguments
/// * `iter` - CAnnIterator 指针 (由 knowhere_create_ann_iterator 创建)
/// * `out_id` - 输出参数，用于存储返回的向量 ID
/// * `out_distance` - 输出参数，用于存储返回的距离
///
/// # Returns
/// 1 表示成功获取下一个结果，0 表示已到达末尾或出错
///
/// # Safety
/// 调用者需确保 iter 指针有效，out_id 和 out_distance 指向有效的内存
#[no_mangle]
pub extern "C" fn knowhere_ann_iterator_next(
    iter: *mut CAnnIterator,
    out_id: *mut i64,
    out_distance: *mut f32,
) -> i32 {
    if iter.is_null() || out_id.is_null() || out_distance.is_null() {
        return 0;
    }

    unsafe {
        let c_iter = &mut *iter;
        if c_iter.inner.is_null() {
            return 0;
        }

        // 将原始指针转换回 trait object
        let iter_trait = &mut *(c_iter.inner as *mut Box<dyn crate::index::AnnIterator>);

        match iter_trait.next() {
            Some((id, dist)) => {
                *out_id = id;
                *out_distance = dist;
                1
            }
            None => 0,
        }
    }
}

/// 释放 ANN 迭代器
///
/// 释放由 knowhere_create_ann_iterator 返回的 CAnnIterator 及其所有关联内存。
///
/// # Arguments
/// * `iter` - CAnnIterator 指针 (由 knowhere_create_ann_iterator 返回)
///
/// # Safety
/// 调用后 iter 指针不再有效，不应再被使用
#[no_mangle]
pub extern "C" fn knowhere_free_ann_iterator(iter: *mut CAnnIterator) {
    if !iter.is_null() {
        unsafe {
            let c_iter = &mut *iter;
            if !c_iter.inner.is_null() {
                // 释放内部迭代器
                let _ = Box::from_raw(c_iter.inner as *mut Box<dyn crate::index::AnnIterator>);
            }
            // 释放 CAnnIterator 结构体
            let _ = Box::from_raw(iter);
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
            let _ = Vec::from_raw_parts(binary.data, binary.size as usize, binary.size as usize);
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
    cap_words: usize,
}

impl From<&BitsetView> for CBitset {
    fn from(bitset: &BitsetView) -> Self {
        let slice = bitset.as_slice();
        let mut vec = slice.to_vec();
        vec.shrink_to_fit();
        let ptr = vec.as_mut_ptr();
        let cap_words = vec.capacity();
        std::mem::forget(vec);

        Self {
            data: ptr,
            len: bitset.len(),
            cap_words,
        }
    }
}

impl Drop for CBitset {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                drop(Vec::from_raw_parts(self.data, 0, self.cap_words));
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
        unsafe {
            let _ = Box::from_raw(bitset);
        }
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

        let word_idx = index >> 6; // index / 64
        let bit_idx = index & 63; // index % 64
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
        let num_words = cb.len.div_ceil(64);
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
        cb.len.div_ceil(8)
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
        let _cb = &*bitset;
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
pub extern "C" fn knowhere_bitset_set_id_offset(bitset: *mut CBitset, _offset: usize) {
    if bitset.is_null() {}

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

        let num_words = cb.len.div_ceil(64);
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
        let num_words = cb.len.div_ceil(64);
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
pub extern "C" fn knowhere_bitset_or(
    bitset1: *const CBitset,
    bitset2: *const CBitset,
) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;

        let len = cb1.len.max(cb2.len);
        let num_words = len.div_ceil(64);

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

            let w2_0 = if i < cb1.len.div_ceil(64) && i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
            let w2_1 = if i + 1 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 1)
            } else {
                0
            };
            let w2_2 = if i + 2 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 2)
            } else {
                0
            };
            let w2_3 = if i + 3 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 3)
            } else {
                0
            };

            result_data.push(w1_0 | w2_0);
            result_data.push(w1_1 | w2_1);
            result_data.push(w1_2 | w2_2);
            result_data.push(w1_3 | w2_3);

            i += 4;
        }

        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < cb1.len.div_ceil(64) {
                *cb1.data.add(i)
            } else {
                0
            };
            let w2 = if i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
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
pub extern "C" fn knowhere_bitset_and(
    bitset1: *const CBitset,
    bitset2: *const CBitset,
) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;

        let len = cb1.len.max(cb2.len);
        let num_words = len.div_ceil(64);

        // 分配结果数据
        let mut result_data: Vec<u64> = Vec::with_capacity(num_words);

        // SIMD 优化的按位与操作
        let mut i = 0;
        while i + 3 < num_words {
            let w1_0 = if i < cb1.len.div_ceil(64) {
                *cb1.data.add(i)
            } else {
                0
            };
            let w1_1 = if i + 1 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 1)
            } else {
                0
            };
            let w1_2 = if i + 2 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 2)
            } else {
                0
            };
            let w1_3 = if i + 3 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 3)
            } else {
                0
            };

            let w2_0 = if i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
            let w2_1 = if i + 1 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 1)
            } else {
                0
            };
            let w2_2 = if i + 2 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 2)
            } else {
                0
            };
            let w2_3 = if i + 3 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 3)
            } else {
                0
            };

            result_data.push(w1_0 & w2_0);
            result_data.push(w1_1 & w2_1);
            result_data.push(w1_2 & w2_2);
            result_data.push(w1_3 & w2_3);

            i += 4;
        }

        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < cb1.len.div_ceil(64) {
                *cb1.data.add(i)
            } else {
                0
            };
            let w2 = if i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
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
pub extern "C" fn knowhere_bitset_xor(
    bitset1: *const CBitset,
    bitset2: *const CBitset,
) -> *mut CBitset {
    if bitset1.is_null() || bitset2.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let cb1 = &*bitset1;
        let cb2 = &*bitset2;

        let len = cb1.len.max(cb2.len);
        let num_words = len.div_ceil(64);

        // 分配结果数据
        let mut result_data: Vec<u64> = Vec::with_capacity(num_words);

        // SIMD 优化的按位异或操作
        let mut i = 0;
        while i + 3 < num_words {
            let w1_0 = if i < cb1.len.div_ceil(64) {
                *cb1.data.add(i)
            } else {
                0
            };
            let w1_1 = if i + 1 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 1)
            } else {
                0
            };
            let w1_2 = if i + 2 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 2)
            } else {
                0
            };
            let w1_3 = if i + 3 < cb1.len.div_ceil(64) {
                *cb1.data.add(i + 3)
            } else {
                0
            };

            let w2_0 = if i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
            let w2_1 = if i + 1 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 1)
            } else {
                0
            };
            let w2_2 = if i + 2 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 2)
            } else {
                0
            };
            let w2_3 = if i + 3 < cb2.len.div_ceil(64) {
                *cb2.data.add(i + 3)
            } else {
                0
            };

            result_data.push(w1_0 ^ w2_0);
            result_data.push(w1_1 ^ w2_1);
            result_data.push(w1_2 ^ w2_2);
            result_data.push(w1_3 ^ w2_3);

            i += 4;
        }

        // 处理剩余的元素
        while i < num_words {
            let w1 = if i < cb1.len.div_ceil(64) {
                *cb1.data.add(i)
            } else {
                0
            };
            let w2 = if i < cb2.len.div_ceil(64) {
                *cb2.data.add(i)
            } else {
                0
            };
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
#[allow(unused_unsafe, unused_variables)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        old: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, old }
        }

        fn remove(key: &'static str) -> Self {
            let old = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.old.take() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    fn hnsw_wrapper_for_add_strategy_test() -> IndexWrapper {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 32,
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        };
        IndexWrapper::new(config).expect("HNSW wrapper should be created")
    }

    #[test]
    fn test_hnsw_ffi_add_defaults_to_serial_when_override_is_unset() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::remove(FFI_FORCE_SERIAL_HNSW_ADD_ENV);
        let _parallel_guard = EnvVarGuard::remove(FFI_ENABLE_PARALLEL_HNSW_ADD_ENV);
        let wrapper = hnsw_wrapper_for_add_strategy_test();
        let idx = wrapper.hnsw.as_ref().expect("wrapper should hold HNSW");

        assert!(
            !IndexWrapper::should_use_parallel_hnsw_add_via_ffi(idx, 1000),
            "FFI HNSW add should default to serial add() unless parallel is explicitly enabled"
        );
    }

    #[test]
    fn test_hnsw_ffi_add_force_serial_override_disables_parallel() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set(FFI_FORCE_SERIAL_HNSW_ADD_ENV, "1");
        let _parallel_guard = EnvVarGuard::set(FFI_ENABLE_PARALLEL_HNSW_ADD_ENV, "1");
        let wrapper = hnsw_wrapper_for_add_strategy_test();
        let idx = wrapper.hnsw.as_ref().expect("wrapper should hold HNSW");

        assert!(
            !IndexWrapper::should_use_parallel_hnsw_add_via_ffi(idx, 1000),
            "FFI HNSW add override should force large builds back to serial add()"
        );
    }

    #[test]
    fn test_hnsw_ffi_add_parallel_requires_explicit_opt_in() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::remove(FFI_FORCE_SERIAL_HNSW_ADD_ENV);
        let _parallel_guard = EnvVarGuard::set(FFI_ENABLE_PARALLEL_HNSW_ADD_ENV, "1");
        let wrapper = hnsw_wrapper_for_add_strategy_test();
        let idx = wrapper.hnsw.as_ref().expect("wrapper should hold HNSW");

        assert!(
            IndexWrapper::should_use_parallel_hnsw_add_via_ffi(idx, 1000),
            "FFI HNSW add should only use parallel build after explicit opt-in"
        );
    }

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

        let index = knowhere_create_index(config.clone());
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

        let index = knowhere_create_index(config.clone());
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
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "Flat");

        // Test metric type
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }
            .to_str()
            .unwrap();
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
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "HNSW");

        // Test metric type
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(metric_str, "IP");

        // Test size
        let size = unsafe { knowhere_get_index_size(index) };
        // Note: size is usize, always >= 0

        knowhere_free_index(index);
    }

    #[cfg(feature = "scann")]
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
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "ScaNN");

        // Test metric type (ScaNN defaults to L2)
        let metric_ptr = unsafe { knowhere_get_index_metric(index) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(metric_str, "L2");

        // Test size
        let size = unsafe { knowhere_get_index_size(index) };
        // Note: size is usize, always >= 0

        knowhere_free_index(index);
    }

    #[test]
    fn test_index_statistics_null_pointer() {
        // Test with null pointer - should return safe defaults
        let size = unsafe { knowhere_get_index_size(std::ptr::null()) };
        assert_eq!(size, 0);

        let type_ptr = unsafe { knowhere_get_index_type(std::ptr::null()) };
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "Unknown");

        let metric_ptr = unsafe { knowhere_get_index_metric(std::ptr::null()) };
        assert!(!metric_ptr.is_null());
        let metric_str = unsafe { std::ffi::CStr::from_ptr(metric_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(metric_str, "Unknown");
    }

    fn assert_index_type(config: CIndexConfig, expected: &str) {
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let type_ptr = knowhere_get_index_type(index);
        assert!(!type_ptr.is_null());
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, expected);

        knowhere_free_index(index);
    }

    #[test]
    fn test_index_type_minhash_lsh() {
        assert_index_type(
            CIndexConfig {
                index_type: CIndexType::MinHashLsh,
                metric_type: CMetricType::Hamming,
                dim: 64,
                data_type: 100, // Binary
                ..Default::default()
            },
            "MinHashLSH",
        );
    }

    #[test]
    fn test_index_type_hnsw_pq() {
        assert_index_type(
            CIndexConfig {
                index_type: CIndexType::HnswPq,
                metric_type: CMetricType::L2,
                dim: 64,
                ..Default::default()
            },
            "HNSW_PQ",
        );
    }

    #[test]
    fn test_minhash_add_binary_rejects_invalid_dim_alignment() {
        let index = knowhere_create_index(CIndexConfig {
            index_type: CIndexType::MinHashLsh,
            metric_type: CMetricType::Hamming,
            dim: 72, // 9 bytes, not aligned to u64 element size (8)
            data_type: 100,
            ..Default::default()
        });
        assert!(!index.is_null());

        // 1 vector, 9 bytes
        let vectors: Vec<u8> = (0..9).map(|v| v as u8).collect();
        let ids = [0i64];

        let rc = knowhere_add_binary_index(index, vectors.as_ptr(), ids.as_ptr(), 1, 72);
        assert_eq!(rc, CError::InvalidArg as i32);

        knowhere_free_index(index);
    }

    #[test]
    fn test_index_type_sparse_wand() {
        assert_index_type(
            CIndexConfig {
                index_type: CIndexType::SparseWand,
                metric_type: CMetricType::Ip,
                dim: 64,
                data_type: 104, // SparseFloatVector
                ..Default::default()
            },
            "SparseWand",
        );
    }

    #[test]
    fn test_index_type_sparse_inverted() {
        assert_index_type(
            CIndexConfig {
                index_type: CIndexType::SparseInverted,
                metric_type: CMetricType::Ip,
                dim: 64,
                data_type: 104, // SparseFloatVector
                ..Default::default()
            },
            "SparseInverted",
        );
    }

    #[test]
    fn test_ffi_sparse_inverted_metadata_contract() {
        let sparse = knowhere_create_index(CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 16,
            data_type: 104,
            ..Default::default()
        });
        assert!(!sparse.is_null());

        assert_eq!(knowhere_is_additional_scalar_supported(sparse, false), 0);
        assert_eq!(knowhere_is_additional_scalar_supported(sparse, true), 1);

        let sparse_meta_ptr = knowhere_get_index_meta(sparse);
        assert!(!sparse_meta_ptr.is_null());
        let sparse_meta_str = unsafe { std::ffi::CStr::from_ptr(sparse_meta_ptr) }
            .to_str()
            .unwrap();
        let sparse_meta_json: serde_json::Value = serde_json::from_str(sparse_meta_str).unwrap();
        assert_eq!(sparse_meta_json["index_type"], "SparseInverted");
        assert_eq!(sparse_meta_json["is_trained"], true);
        assert_eq!(sparse_meta_json["additional_scalar_supported"], true);
        assert_eq!(
            sparse_meta_json["additional_scalar"]["support_mode"],
            "partial"
        );
        assert_eq!(
            sparse_meta_json["capabilities"]["get_vector_by_ids"],
            "supported"
        );
        assert_eq!(
            sparse_meta_json["capabilities"]["ann_iterator"],
            "supported"
        );
        assert_eq!(sparse_meta_json["capabilities"]["persistence"], "supported");
        assert_eq!(sparse_meta_json["semantics"]["family"], "sparse");
        assert_eq!(
            sparse_meta_json["semantics"]["persistence_mode"],
            "file_save_load+memory_serialize"
        );
        assert_eq!(
            sparse_meta_json["semantics"]["persistence"]["file_save_load"],
            "supported"
        );
        assert_eq!(
            sparse_meta_json["semantics"]["persistence"]["memory_serialize"],
            "supported"
        );
        assert_eq!(
            sparse_meta_json["semantics"]["persistence"]["deserialize_from_file"],
            "supported"
        );
        assert_eq!(
            sparse_meta_json["resource_contract"]["mmap_supported"],
            true
        );

        knowhere_free_cstring(sparse_meta_ptr);
        knowhere_free_index(sparse);
    }

    #[test]
    fn test_ffi_sparse_inverted_roundtrip_and_iterator() {
        use std::ffi::CString;

        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let index = knowhere_create_index(config.clone());
        assert!(!index.is_null());

        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [10_i64, 11, 12];

        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(index), 3);

        let query_ids = [10_i64, 12];
        let vector_result = knowhere_get_vectors_by_ids(index, query_ids.as_ptr(), query_ids.len());
        assert!(!vector_result.is_null());
        let vector_result = unsafe { &mut *vector_result };
        assert_eq!(vector_result.num_vectors, 2);
        assert_eq!(vector_result.dim, 8);
        let dense = unsafe {
            std::slice::from_raw_parts(
                vector_result.vectors,
                vector_result.num_vectors * vector_result.dim,
            )
        };
        assert_eq!(dense[0], 1.0);
        assert_eq!(dense[8 + 2], 3.0);
        knowhere_free_vector_result(vector_result);

        let path = std::env::temp_dir().join(format!(
            "hanns_sparse_inverted_{}.bin",
            std::process::id()
        ));
        let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(
            knowhere_save_index(index, path_c.as_ptr()),
            CError::Success as i32
        );

        let loaded = knowhere_create_index(CIndexConfig { dim: 1, ..config });
        assert!(!loaded.is_null());
        assert_eq!(
            knowhere_load_index(loaded, path_c.as_ptr()),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(loaded), 3);
        assert_eq!(knowhere_get_index_dim(loaded), 8);

        let query = [0.0_f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let iter = knowhere_create_ann_iterator(loaded, query.as_ptr(), 1, 8, std::ptr::null());
        assert!(!iter.is_null());
        let mut id = -1_i64;
        let mut dist = -1.0_f32;
        assert_eq!(knowhere_ann_iterator_next(iter, &mut id, &mut dist), 1);
        assert_eq!(id, 11);
        assert!(dist > 0.0);
        knowhere_free_ann_iterator(iter);

        let _ = std::fs::remove_file(&path);
        knowhere_free_index(loaded);
        knowhere_free_index(index);
    }

    fn sparse_multi_query_config(index_type: CIndexType) -> CIndexConfig {
        CIndexConfig {
            index_type,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        }
    }

    fn sparse_multi_query_vectors() -> (Vec<f32>, [i64; 4]) {
        (
            vec![
                4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
            [10_i64, 11, 12, 13],
        )
    }

    #[test]
    fn test_ffi_sparse_inverted_multi_query_search_returns_results_per_query() {
        let config = sparse_multi_query_config(CIndexType::SparseInverted);
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let (vectors, ids) = sparse_multi_query_vectors();
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );

        let queries = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ];
        let result = knowhere_search(index, queries.as_ptr(), 2, 2, 8);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 4);
            let result_ids = std::slice::from_raw_parts(result_ref.ids, result_ref.num_results);
            assert_eq!(result_ids, &[10, 11, 12, 13]);
        }

        knowhere_free_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_ffi_sparse_wand_multi_query_search_returns_results_per_query() {
        let config = sparse_multi_query_config(CIndexType::SparseWand);
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let (vectors, ids) = sparse_multi_query_vectors();
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );

        let queries = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ];
        let result = knowhere_search(index, queries.as_ptr(), 2, 2, 8);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 4);
            let result_ids = std::slice::from_raw_parts(result_ref.ids, result_ref.num_results);
            assert_eq!(result_ids, &[10, 11, 12, 13]);
        }

        knowhere_free_result(result);
        knowhere_free_index(index);
    }

    #[test]
    fn test_ffi_sparse_inverted_search_accepts_query_dim_larger_than_index_dim() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 4,
            data_type: 104,
            ..Default::default()
        };
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors = vec![
            4.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            0.0, 5.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
        ];
        let ids = [10_i64, 11, 12, 13];
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), ids.len(), 4),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), ids.len(), 4),
            CError::Success as i32
        );

        let queries = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 7.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 9.0, //
        ];
        let result = knowhere_search(index, queries.as_ptr(), 2, 2, 6);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 4);
            let result_ids = std::slice::from_raw_parts(result_ref.ids, result_ref.num_results);
            assert_eq!(result_ids, &[10, 11, 12, 13]);
        }

        knowhere_free_result(result);
        knowhere_free_index(index);
    }

    #[cfg(feature = "scann")]
    #[test]
    fn test_ffi_persistence_scann_file_roundtrip() {
        use std::ffi::CString;

        let config = CIndexConfig {
            index_type: CIndexType::Scann,
            metric_type: CMetricType::L2,
            dim: 8,
            num_partitions: 4,
            num_centroids: 8,
            reorder_k: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config.clone());
        assert!(!index.is_null());

        let vectors: Vec<f32> = (0..64).map(|i| i as f32 * 0.125).collect();
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 8, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), std::ptr::null(), 8, 8),
            CError::Success as i32
        );

        let path = std::env::temp_dir().join(format!(
            "hanns_scann_persistence_{}.bin",
            std::process::id()
        ));
        let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(
            knowhere_save_index(index, path_c.as_ptr()),
            CError::Success as i32
        );

        let loaded = knowhere_create_index(config.clone());
        assert!(!loaded.is_null());
        assert_eq!(
            knowhere_load_index(loaded, path_c.as_ptr()),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(loaded), 8);

        let _ = std::fs::remove_file(&path);
        knowhere_free_index(loaded);
        knowhere_free_index(index);
    }

    #[test]
    fn test_ffi_persistence_ivf_sq8_reports_memory_serialize_supported() {
        let ivf = knowhere_create_index(CIndexConfig {
            index_type: CIndexType::IvfSq8,
            metric_type: CMetricType::L2,
            dim: 16,
            ..Default::default()
        });
        assert!(!ivf.is_null());

        let meta_ptr = knowhere_get_index_meta(ivf);
        assert!(!meta_ptr.is_null());
        let meta_str = unsafe { std::ffi::CStr::from_ptr(meta_ptr) }
            .to_str()
            .unwrap();
        let meta_json: serde_json::Value = serde_json::from_str(meta_str).unwrap();
        assert_eq!(
            meta_json["semantics"]["persistence"]["file_save_load"],
            "supported"
        );
        assert_eq!(
            meta_json["semantics"]["persistence"]["memory_serialize"],
            "supported"
        );
        assert_eq!(
            meta_json["semantics"]["persistence"]["deserialize_from_file"],
            "supported"
        );

        knowhere_free_cstring(meta_ptr);
        knowhere_free_index(ivf);
    }

    #[test]
    fn test_ffi_persistence_ivfpq_file_roundtrip() {
        use std::ffi::CString;

        let config = CIndexConfig {
            index_type: CIndexType::IvfPq,
            metric_type: CMetricType::L2,
            dim: 8,
            num_clusters: 4,
            nprobe: 2,
            ..Default::default()
        };

        let index = knowhere_create_index(config.clone());
        assert!(!index.is_null());

        let vectors: Vec<f32> = (0..96).map(|i| i as f32 * 0.25).collect();
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 12, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), std::ptr::null(), 12, 8),
            CError::Success as i32
        );

        let path = std::env::temp_dir().join(format!(
            "hanns_ivfpq_persistence_{}.bin",
            std::process::id()
        ));
        let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(
            knowhere_save_index(index, path_c.as_ptr()),
            CError::Success as i32
        );

        let loaded = knowhere_create_index(config);
        assert!(!loaded.is_null());
        assert_eq!(
            knowhere_load_index(loaded, path_c.as_ptr()),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(loaded), 12);

        let _ = std::fs::remove_file(&path);
        knowhere_free_index(loaded);
        knowhere_free_index(index);
    }

    #[test]
    fn test_ffi_persistence_flat_empty_file_load_fails() {
        use std::ffi::CString;

        let flat = knowhere_create_index(CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 8,
            ..Default::default()
        });
        assert!(!flat.is_null());

        let path = std::env::temp_dir().join(format!(
            "hanns_empty_persistence_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, []).unwrap();
        let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(
            knowhere_load_index(flat, path_c.as_ptr()),
            CError::Internal as i32
        );

        let _ = std::fs::remove_file(&path);
        knowhere_free_index(flat);
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

    #[cfg(feature = "scann")]
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

    #[cfg(feature = "scann")]
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
        let vectors_slice =
            unsafe { std::slice::from_raw_parts(result.vectors, result.num_vectors * result.dim) };
        assert_eq!(vectors_slice[0], 0.0); // First element of vector 0
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
    fn test_serialize_sparse_inverted_index() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [10_i64, 11, 12];

        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );

        let binset = knowhere_serialize_index(index);
        assert!(!binset.is_null());

        unsafe {
            let binset_ref = &*binset;
            assert_eq!(binset_ref.count, 1);
            assert!(!binset_ref.keys.is_null());
            assert!(!binset_ref.values.is_null());

            let key = std::ffi::CStr::from_ptr(*binset_ref.keys);
            assert_eq!(key.to_str().unwrap(), "index_data");

            let binary = &*binset_ref.values;
            assert!(!binary.data.is_null());
            assert!(binary.size > 0);
        }

        knowhere_free_binary_set(binset);
        knowhere_free_index(index);
    }

    #[test]
    fn test_serialize_sparse_wand_index() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseWand,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [10_i64, 11, 12];

        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );

        let binset = knowhere_serialize_index(index);
        assert!(!binset.is_null());

        unsafe {
            let binset_ref = &*binset;
            assert_eq!(binset_ref.count, 1);
            assert!(!binset_ref.keys.is_null());
            assert!(!binset_ref.values.is_null());

            let key = std::ffi::CStr::from_ptr(*binset_ref.keys);
            assert_eq!(key.to_str().unwrap(), "index_data");

            let binary = &*binset_ref.values;
            assert!(!binary.data.is_null());
            assert!(binary.size > 0);
        }

        knowhere_free_binary_set(binset);
        knowhere_free_index(index);
    }

    fn read_hnsw_ef_search_from_binset(binset: *mut CBinarySet) -> u32 {
        unsafe {
            let binset_ref = &*binset;
            assert!(binset_ref.count >= 1);
            let binary = &*binset_ref.values;
            let data = std::slice::from_raw_parts(binary.data, binary.size as usize);
            assert!(data.len() >= 24);
            assert_eq!(&data[0..4], b"HNSW");
            u32::from_le_bytes(data[20..24].try_into().unwrap())
        }
    }

    #[test]
    fn test_hnsw_set_ef_search_updates_serialized_state() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 16,
            ef_construction: 200,
            ef_search: 16,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors: Vec<f32> = (0..20 * 16).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..20).collect();

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 20, 16);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 20, 16);
        assert_eq!(add_result, CError::Success as i32);

        let before = knowhere_serialize_index(index);
        assert!(!before.is_null());
        assert_eq!(read_hnsw_ef_search_from_binset(before), 16);
        knowhere_free_binary_set(before);

        let update_result = knowhere_set_ef_search(index, 77);
        assert_eq!(update_result, CError::Success as i32);

        let after = knowhere_serialize_index(index);
        assert!(!after.is_null());
        assert_eq!(read_hnsw_ef_search_from_binset(after), 77);

        knowhere_free_binary_set(after);
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
    fn test_deserialize_sparse_inverted_index() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let source_index = knowhere_create_index(config.clone());
        assert!(!source_index.is_null());

        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [10_i64, 11, 12];

        assert_eq!(
            knowhere_train_index(source_index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(source_index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );

        let binset = knowhere_serialize_index(source_index);
        assert!(!binset.is_null());

        let target_index = knowhere_create_index(config);
        assert!(!target_index.is_null());
        assert_eq!(
            knowhere_deserialize_index(target_index, binset),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(target_index), 3);

        let query = [0.0_f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let source_result = knowhere_search(source_index, query.as_ptr(), 1, 2, 8);
        let target_result = knowhere_search(target_index, query.as_ptr(), 1, 2, 8);
        assert!(!source_result.is_null());
        assert!(!target_result.is_null());

        unsafe {
            let src_ref = &*source_result;
            let tgt_ref = &*target_result;
            assert_eq!(src_ref.num_results, tgt_ref.num_results);
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
    fn test_deserialize_sparse_wand_index() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseWand,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let source_index = knowhere_create_index(config.clone());
        assert!(!source_index.is_null());

        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [10_i64, 11, 12];

        assert_eq!(
            knowhere_train_index(source_index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(source_index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );

        let binset = knowhere_serialize_index(source_index);
        assert!(!binset.is_null());

        let target_index = knowhere_create_index(config);
        assert!(!target_index.is_null());
        assert_eq!(
            knowhere_deserialize_index(target_index, binset),
            CError::Success as i32
        );
        assert_eq!(knowhere_get_index_count(target_index), 3);

        let query = [0.0_f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let source_result = knowhere_search(source_index, query.as_ptr(), 1, 2, 8);
        let target_result = knowhere_search(target_index, query.as_ptr(), 1, 2, 8);
        assert!(!source_result.is_null());
        assert!(!target_result.is_null());

        unsafe {
            let src_ref = &*source_result;
            let tgt_ref = &*target_result;
            assert_eq!(src_ref.num_results, tgt_ref.num_results);
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
            1.0, 0.0, 0.0, 0.0, // dist=1.0 from origin
            0.0, 1.0, 0.0, 0.0, // dist=1.0 from origin
            0.0, 0.0, 1.0, 0.0, // dist=1.0 from origin
            2.0, 0.0, 0.0, 0.0, // dist=2.0 from origin
        ];
        let ids = [0i64, 1, 2, 3];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 4, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 4, 4);
        assert_eq!(add_result, CError::Success as i32);

        // Range search with radius=1.5 (should find first 3 vectors)
        let query = [0.0, 0.0, 0.0, 0.0];
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
            0.0, 0.0, 0.0, 0.0, // id=0
            1.0, 0.0, 0.0, 0.0, // id=1
            0.0, 1.0, 0.0, 0.0, // id=2
        ];
        let ids = [0i64, 1, 2];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 3, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 4);
        assert_eq!(add_result, CError::Success as i32);

        // Two query vectors
        let queries = [
            0.0, 0.0, 0.0, 0.0, // Query 1: at origin
            1.0, 0.0, 0.0, 0.0, // Query 2: at (1,0,0,0)
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
        let query = [0.0, 0.0, 0.0, 0.0];
        let result = knowhere_range_search(std::ptr::null(), query.as_ptr(), 1, 1.0, 4);
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

        let result = knowhere_range_search(index, std::ptr::null(), 1, 1.0, 4);
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
        let vectors = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let ids = [0i64, 1];

        let train_result = knowhere_train_index(index, vectors.as_ptr(), 2, 4);
        assert_eq!(train_result, CError::Success as i32);

        let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 2, 4);
        assert_eq!(add_result, CError::Success as i32);

        // HNSW range search should return NULL (NotImplemented)
        let query = [0.0, 0.0, 0.0, 0.0];
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
        let vectors_slice =
            unsafe { std::slice::from_raw_parts(result.vectors, result.num_ids * result.dim) };
        assert_eq!(vectors_slice[0], 80.0); // First element of vector 5 (5*16=80)

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
        let vectors_slice =
            unsafe { std::slice::from_raw_parts(result.vectors, result.num_ids * result.dim) };
        assert_eq!(vectors_slice[0], 0.0); // First element of vector 0
        assert_eq!(vectors_slice[16], 80.0); // First element of vector 5 (5*16=80)
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
        let result =
            knowhere_get_vector_by_ids(std::ptr::null(), query_ids.as_ptr(), query_ids.len(), 16);
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
    fn test_get_vector_by_ids_hnsw() {
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

        // HNSW get_vector_by_ids should return the stored raw vectors
        let query_ids: Vec<i64> = vec![0];
        let result = knowhere_get_vector_by_ids(index, query_ids.as_ptr(), query_ids.len(), 16);
        assert!(!result.is_null());

        let result = unsafe { &*result };
        assert_eq!(result.num_ids, 1);
        assert_eq!(result.dim, 16);
        let returned =
            unsafe { std::slice::from_raw_parts(result.vectors, result.num_ids * result.dim) };
        assert_eq!(returned, &vectors[..16]);

        knowhere_free_get_vector_result(result as *const _ as *mut _);

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
        let result = knowhere_search_with_bitset(index, query.as_ptr(), 1, 5, 4, bitset);
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
    fn test_search_with_bitset_sparse_inverted() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 8,
            data_type: 104,
            ..Default::default()
        };

        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors: Vec<f32> = vec![
            0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.8, 0.25, 0.0, 0.0, 0.0,
        ];
        let ids = [0_i64, 1, 2];

        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), 3, 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), 3, 8),
            CError::Success as i32
        );

        let query = [0.0_f32, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let bitset = knowhere_bitset_create(3);
        unsafe {
            knowhere_bitset_set(bitset, 0, true);
        }

        let result = knowhere_search_with_bitset(index, query.as_ptr(), 1, 1, 8, bitset);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 1);
            assert_eq!(*result_ref.ids.add(0), 2);
        }

        knowhere_free_result(result);
        knowhere_bitset_free(bitset);
        knowhere_free_index(index);
    }

    #[test]
    fn test_search_with_bitset_sparse_inverted_multi_query_returns_results_per_query() {
        let config = sparse_multi_query_config(CIndexType::SparseInverted);
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let (vectors, ids) = sparse_multi_query_vectors();
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), ids.len(), 8),
            CError::Success as i32
        );

        let queries = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        ];
        let bitset = knowhere_bitset_create(ids.len());
        unsafe {
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 2, true);
        }

        let result = knowhere_search_with_bitset(index, queries.as_ptr(), 2, 1, 8, bitset);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 2);
            let result_ids = std::slice::from_raw_parts(result_ref.ids, result_ref.num_results);
            assert_eq!(result_ids, &[11, 13]);
        }

        knowhere_free_result(result);
        knowhere_bitset_free(bitset);
        knowhere_free_index(index);
    }

    #[test]
    fn test_search_with_bitset_sparse_inverted_accepts_query_dim_larger_than_index_dim() {
        let config = CIndexConfig {
            index_type: CIndexType::SparseInverted,
            metric_type: CMetricType::Ip,
            dim: 4,
            data_type: 104,
            ..Default::default()
        };
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors = vec![
            4.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            0.0, 5.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
        ];
        let ids = [10_i64, 11, 12, 13];
        assert_eq!(
            knowhere_train_index(index, vectors.as_ptr(), ids.len(), 4),
            CError::Success as i32
        );
        assert_eq!(
            knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), ids.len(), 4),
            CError::Success as i32
        );

        let queries = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 7.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 9.0, //
        ];
        let bitset = knowhere_bitset_create(ids.len());
        unsafe {
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 2, true);
        }

        let result = knowhere_search_with_bitset(index, queries.as_ptr(), 2, 1, 6, bitset);
        assert!(!result.is_null());

        unsafe {
            let result_ref = &*result;
            assert_eq!(result_ref.num_results, 2);
            let result_ids = std::slice::from_raw_parts(result_ref.ids, result_ref.num_results);
            assert_eq!(result_ids, &[11, 13]);
        }

        knowhere_free_result(result);
        knowhere_bitset_free(bitset);
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
        let result =
            knowhere_search_with_bitset(std::ptr::null(), std::ptr::null(), 1, 5, 4, bitset);
        assert!(result.is_null());

        knowhere_bitset_free(bitset as *mut _);
    }

    #[test]
    fn test_bitset_byte_size() {
        // Test various sizes
        let bitset_1 = knowhere_bitset_create(1);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_1), 1); // (1+7)/8 = 1
        }
        knowhere_bitset_free(bitset_1);

        let bitset_8 = knowhere_bitset_create(8);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_8), 1); // (8+7)/8 = 1
        }
        knowhere_bitset_free(bitset_8);

        let bitset_64 = knowhere_bitset_create(64);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_64), 8); // (64+7)/8 = 8
        }
        knowhere_bitset_free(bitset_64);

        let bitset_100 = knowhere_bitset_create(100);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_100), 13); // (100+7)/8 = 13
        }
        knowhere_bitset_free(bitset_100);

        let bitset_1000 = knowhere_bitset_create(1000);
        unsafe {
            assert_eq!(knowhere_bitset_byte_size(bitset_1000), 125); // (1000+7)/8 = 125
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
            let slice = std::slice::from_raw_parts(data, 2); // 128 bits = 2 u64s
            assert_eq!(slice[0], 0);
            assert_eq!(slice[1], 0);

            // Set some bits and verify
            knowhere_bitset_set(bitset, 0, true);
            knowhere_bitset_set(bitset, 64, true);

            // Re-read data
            let data2 = knowhere_bitset_data(bitset);
            let slice2 = std::slice::from_raw_parts(data2, 2);
            assert_eq!(slice2[0], 1u64); // First bit set
            assert_eq!(slice2[1], 1u64); // 65th bit set (first bit of second u64)
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

    #[test]
    fn test_diskann_ffi_save_load_roundtrip() {
        let tmp_dir = std::env::temp_dir().join("diskann_ffi_test");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let save_path = tmp_dir.to_str().unwrap();

        unsafe {
            let dim = 8usize;
            let n = 50usize;

            let config = CIndexConfig {
                index_type: CIndexType::DiskAnn,
                metric_type: CMetricType::L2,
                dim,
                ef_construction: 12,
                ef_search: 30,
                ..CIndexConfig::default()
            };
            let index = knowhere_create_index(config);
            assert!(!index.is_null());

            let vectors: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
            let ids: Vec<i64> = (0..n as i64).collect();

            let err = knowhere_train_index(index as *mut _, vectors.as_ptr(), n, dim);
            assert_eq!(err, 0);
            let err =
                knowhere_add_index(index as *mut _, vectors.as_ptr(), ids.as_ptr(), n, dim);
            assert_eq!(err, 0);

            // Save
            let path_cstr = std::ffi::CString::new(save_path).unwrap();
            let err = knowhere_save_index(index, path_cstr.as_ptr());
            assert_eq!(err, 0, "save must succeed");

            knowhere_free_index(index as *mut _);

            // Load into a fresh index
            let config2 = CIndexConfig {
                index_type: CIndexType::DiskAnn,
                metric_type: CMetricType::L2,
                dim,
                ef_construction: 12,
                ef_search: 30,
                ..CIndexConfig::default()
            };
            let loaded = knowhere_create_index(config2);
            assert!(!loaded.is_null());

            let err = knowhere_load_index(loaded as *mut _, path_cstr.as_ptr());
            assert_eq!(err, 0, "load must succeed");

            // Search on loaded index
            let query: Vec<f32> = vec![0.0f32; dim];
            let result = knowhere_search(loaded, query.as_ptr(), 1, 3, dim);
            assert!(!result.is_null(), "search on loaded index must succeed");
            let result_ref = &*result;
            assert!(result_ref.num_results > 0);

            knowhere_free_result(result);
            knowhere_free_index(loaded as *mut _);
        }

        let _ = std::fs::remove_dir_all(tmp_dir);
    }

    #[test]
    fn test_diskann_ffi_add_and_search() {
        unsafe {
            let dim = 8usize;
            let n = 200usize;

            let config = CIndexConfig {
                index_type: CIndexType::DiskAnn,
                metric_type: CMetricType::L2,
                dim,
                ef_construction: 16,
                ef_search: 40,
                ..CIndexConfig::default()
            };
            let index = knowhere_create_index(config);
            assert!(!index.is_null());

            let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
            for i in 0..n {
                for d in 0..dim {
                    vectors.push(if d == i % dim { 1.0 } else { 0.0 });
                }
            }
            let ids: Vec<i64> = (0..n as i64).collect();

            let err = knowhere_train_index(index as *mut _, vectors.as_ptr(), n, dim);
            assert_eq!(err, 0, "train must succeed");

            let err = knowhere_add_index(index as *mut _, vectors.as_ptr(), ids.as_ptr(), n, dim);
            assert_eq!(err, 0, "add must succeed");

            // Query: unit vector at position 0 — nearest should be id=0
            let query: Vec<f32> = {
                let mut q = vec![0.0f32; dim];
                q[0] = 1.0;
                q
            };
            let result = knowhere_search(index, query.as_ptr(), 1, 3, dim);
            assert!(!result.is_null(), "search must return results");
            let result_ref = &*result;
            assert!(result_ref.num_results > 0, "must return at least 1 result");
            let top_id = *result_ref.ids;
            assert!(
                top_id >= 0 && (top_id as usize) < n,
                "top id must be valid: got {}",
                top_id
            );

            knowhere_free_result(result);
            knowhere_free_index(index as *mut _);
        }
    }

    #[test]
    fn test_diskann_ffi_create() {
        // DiskANN index creation must succeed (was returning None before this fix)
        let config = CIndexConfig {
            index_type: CIndexType::DiskAnn,
            metric_type: CMetricType::L2,
            dim: 8,
            ef_construction: 16, // reused as max_degree
            ef_search: 32,       // reused as search_list_size
            ..CIndexConfig::default()
        };
        let index = unsafe { knowhere_create_index(config) };
        assert!(
            !index.is_null(),
            "DiskANN index creation must return non-null"
        );
        unsafe { knowhere_free_index(index as *mut _) };
    }
}

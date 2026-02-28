//! Index types and configuration

use serde::{Deserialize, Serialize};

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// Flat index - brute force
    Flat,
    /// IVF-Flat
    IvfFlat,
    /// IVF-PQ
    IvfPq,
    /// HNSW
    Hnsw,
    /// DiskANN
    DiskAnn,
    /// ANNOY
    Annoy,
    /// SCANN (Google ScaNN) - for future implementation
    #[cfg(feature = "scann")]
    Scann,
    /// HNSW-PRQ (Progressive Residual Quantization)
    HnswPrq,
    /// HNSW-PQ (HNSW with Product Quantization)
    HnswPq,
    /// IVF-RaBitQ (Rotated Adaptive Bit Quantization)
    IvfRabitq,
    /// IVF-FLAT-CC (Concurrent Version)
    IvfFlatCc,
    /// IVF-SQ-CC (Concurrent Version with Scalar Quantization)
    IvfSqCc,
    /// Sparse Inverted Index (稀疏倒排索引)
    SparseInverted,
    /// Binary HNSW - HNSW for binary vectors with Hamming distance
    BinaryHnsw,
    /// Binary Flat - Exhaustive search for binary vectors (IDMAP)
    BinFlat,
    /// HNSW-SQ (HNSW with Scalar Quantization)
    HnswSq,
    /// AISAQ (Adaptive Iterative Scalar Adaptive Quantization) - DiskANN-based with PQ
    Aisaq,
    /// Sparse Inverted Index CC (Concurrent Version) - 并发稀疏倒排索引
    SparseInvertedCc,
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::Flat
    }
}

impl IndexType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "flat" => Some(IndexType::Flat),
            "ivf_flat" | "ivf-flat" => Some(IndexType::IvfFlat),
            "ivf_pq" | "ivf-pq" => Some(IndexType::IvfPq),
            "hnsw" => Some(IndexType::Hnsw),
            "diskann" | "disk_ann" => Some(IndexType::DiskAnn),
            "annoy" => Some(IndexType::Annoy),
            #[cfg(feature = "scann")]
            "scann" => Some(IndexType::Scann),
            "hnsw_prq" | "hnsw-prq" => Some(IndexType::HnswPrq),
            "hnsw_pq" | "hnsw-pq" => Some(IndexType::HnswPq),
            "ivf_rabitq" | "ivf-rabitq" | "rabitq" => Some(IndexType::IvfRabitq),
            "ivf_flat_cc" | "ivf-flat-cc" | "ivfcc" => Some(IndexType::IvfFlatCc),
            "ivf_sq_cc" | "ivf-sq-cc" | "ivfsqcc" => Some(IndexType::IvfSqCc),
            "sparse_inverted" | "sparse-inverted" | "sparse" => Some(IndexType::SparseInverted),
            "binary_hnsw" | "binary-hnsw" | "binaryhnsw" => Some(IndexType::BinaryHnsw),
            "bin_flat" | "bin-flat" | "binflat" | "binary_flat" | "binary-flat" => Some(IndexType::BinFlat),
            "hnsw_sq" | "hnsw-sq" | "hnswsq" => Some(IndexType::HnswSq),
            "aisaq" | "a_isaq" | "a-saq" => Some(IndexType::Aisaq),
            "sparse_inverted_cc" | "sparse-inverted-cc" | "sparsecc" => Some(IndexType::SparseInvertedCc),
            _ => None,
        }
    }
}

/// Distance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    /// L2 distance
    L2,
    /// Inner product
    Ip,
    /// Cosine similarity
    Cosine,
    /// Hamming distance (for binary vectors)
    Hamming,
}

impl Default for MetricType {
    fn default() -> Self {
        MetricType::L2
    }
}

impl MetricType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "l2" | "l2_distance" => Some(MetricType::L2),
            "ip" | "inner_product" => Some(MetricType::Ip),
            "cosine" | "cos" => Some(MetricType::Cosine),
            "hamming" => Some(MetricType::Hamming),
            _ => None,
        }
    }
    
    pub fn from_bytes(b: u8) -> Self {
        match b {
            0 => MetricType::L2,
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
            3 => MetricType::Hamming,
            _ => MetricType::L2,
        }
    }
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Index type
    pub index_type: IndexType,
    /// Metric type
    pub metric_type: MetricType,
    /// Vector dimension
    pub dim: usize,
    /// Index-specific parameters
    #[serde(default)]
    pub params: IndexParams,
}

impl IndexConfig {
    pub fn new(index_type: IndexType, metric_type: MetricType, dim: usize) -> Self {
        Self {
            index_type,
            metric_type,
            dim,
            params: IndexParams::default(),
        }
    }
}

/// Index-specific parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexParams {
    /// For IVF: number of clusters
    #[serde(default)]
    pub nlist: Option<usize>,
    /// For IVF: number of probes
    #[serde(default)]
    pub nprobe: Option<usize>,
    /// For PQ: number of bytes per vector
    #[serde(default)]
    pub m: Option<usize>,
    /// For PQ: number of coarse centroids
    #[serde(default)]
    pub nbits_per_idx: Option<usize>,
    /// For HNSW: number of connections
    #[serde(default)]
    pub ef_construction: Option<usize>,
    /// For HNSW: search width
    #[serde(default)]
    pub ef_search: Option<usize>,
    /// For HNSW: level factor
    #[serde(default)]
    pub ml: Option<f32>,
    /// For DiskANN: max degree (R)
    #[serde(default)]
    pub max_degree: Option<usize>,
    /// For DiskANN: search list size
    #[serde(default)]
    pub search_list_size: Option<usize>,
    /// For DiskANN: construction list size
    #[serde(default)]
    pub construction_l: Option<usize>,
    /// For DiskANN: beam width for search
    #[serde(default)]
    pub beamwidth: Option<usize>,
    /// For PRQ: number of subquantizer splits (m)
    #[serde(default)]
    pub prq_m: Option<usize>,
    /// For PRQ: number of residual quantizers (nrq)
    #[serde(default)]
    pub prq_nrq: Option<usize>,
    /// For PRQ: number of bits per subquantizer (nbits)
    #[serde(default)]
    pub prq_nbits: Option<usize>,
    /// For RaBitQ: number of bits for query
    #[serde(default)]
    pub rabitq_bits_query: Option<usize>,
    /// For IVF-CC: segment size for concurrent operations
    #[serde(default)]
    pub ssize: Option<usize>,
    /// For IVF: use Elkan algorithm for k-means
    #[serde(default)]
    pub use_elkan: Option<bool>,
}

impl IndexParams {
    pub fn ivf(nlist: usize, nprobe: usize) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        }
    }

    pub fn hnsw(ef_construction: usize, ef_search: usize, ml: f32) -> Self {
        Self {
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            ml: Some(ml),
            ..Default::default()
        }
    }

    pub fn hnsw_pq(ef_construction: usize, ef_search: usize, m: usize, nbits: usize) -> Self {
        Self {
            ef_construction: Some(ef_construction),
            ef_search: Some(ef_search),
            m: Some(m),
            nbits_per_idx: Some(nbits),
            ..Default::default()
        }
    }

    pub fn pq(m: usize, nbits_per_idx: usize) -> Self {
        Self {
            m: Some(m),
            nbits_per_idx: Some(nbits_per_idx),
            ..Default::default()
        }
    }
    
    pub fn ivf_cc(nlist: usize, nprobe: usize, ssize: usize) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ssize: Some(ssize),
            ..Default::default()
        }
    }

    pub fn bin_flat() -> Self {
        Self {
            ..Default::default()
        }
    }
}

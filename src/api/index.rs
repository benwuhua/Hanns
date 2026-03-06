//! Index types and configuration

use serde::{Deserialize, Serialize};

use super::DataType;

/// Index type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndexType {
    /// Flat index - brute force
    #[default]
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
    /// IVF-SQ8 (Scalar Quantization 8-bit)
    IvfSq8,
    /// IVF-SQ-CC (Concurrent Version with Scalar Quantization)
    IvfSqCc,
    /// Sparse Inverted Index (稀疏倒排索引)
    SparseInverted,
    /// Binary HNSW - HNSW for binary vectors with Hamming distance
    BinaryHnsw,
    /// Binary Flat - Exhaustive search for binary vectors (IDMAP)
    BinFlat,
    /// Binary IVF Flat - IVF clustering for binary vectors with Hamming distance
    BinIvfFlat,
    /// HNSW-SQ (HNSW with Scalar Quantization)
    HnswSq,
    /// AISAQ (Adaptive Iterative Scalar Adaptive Quantization) - DiskANN-based with PQ
    Aisaq,
    /// Sparse Inverted Index CC (Concurrent Version) - 并发稀疏倒排索引
    SparseInvertedCc,
    /// Sparse WAND Index - WAND (Weak AND) algorithm for sparse vectors
    SparseWand,
    /// Sparse WAND Index CC (Concurrent Version) - 并发 WAND 稀疏索引
    SparseWandCc,
    /// MinHash-LSH - Locality Sensitive Hashing for Jaccard similarity
    MinHashLsh,
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
            "ivf_sq8" | "ivf-sq8" | "ivfsq8" => Some(IndexType::IvfSq8),
            "ivf_sq_cc" | "ivf-sq-cc" | "ivfsqcc" => Some(IndexType::IvfSqCc),
            "sparse_inverted" | "sparse-inverted" | "sparse" => Some(IndexType::SparseInverted),
            "binary_hnsw" | "binary-hnsw" | "binaryhnsw" => Some(IndexType::BinaryHnsw),
            "bin_flat" | "bin-flat" | "binflat" | "binary_flat" | "binary-flat" => {
                Some(IndexType::BinFlat)
            }
            "bin_ivf_flat" | "bin-ivf-flat" | "binivfflat" | "binary_ivf_flat"
            | "binary-ivf-flat" => Some(IndexType::BinIvfFlat),
            "hnsw_sq" | "hnsw-sq" | "hnswsq" => Some(IndexType::HnswSq),
            "aisaq" | "a_isaq" | "a-saq" => Some(IndexType::Aisaq),
            "sparse_inverted_cc" | "sparse-inverted-cc" | "sparsecc" => {
                Some(IndexType::SparseInvertedCc)
            }
            "sparse_wand" | "sparse-wand" | "sparsewand" | "wand" => Some(IndexType::SparseWand),
            "sparse_wand_cc" | "sparse-wand-cc" | "sparsewandcc" | "wandcc" => {
                Some(IndexType::SparseWandCc)
            }
            "minhash_lsh" | "minhash-lsh" | "minhashlsh" => Some(IndexType::MinHashLsh),
            _ => None,
        }
    }
}

/// Distance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum MetricType {
    /// L2 distance
    #[default]
    L2,
    /// Inner product
    Ip,
    /// Cosine similarity
    Cosine,
    /// Hamming distance (for binary vectors)
    Hamming,
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
    /// Data type (float, binary, sparse, etc.)
    #[serde(default)]
    pub data_type: DataType,
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
            data_type: DataType::Float,
            params: IndexParams::default(),
        }
    }

    /// Create with explicit data type
    pub fn with_data_type(
        index_type: IndexType,
        metric_type: MetricType,
        dim: usize,
        data_type: DataType,
    ) -> Self {
        Self {
            index_type,
            metric_type,
            dim,
            data_type,
            params: IndexParams::default(),
        }
    }

    /// Validate the configuration against legal matrix
    pub fn validate(&self) -> Result<(), String> {
        super::validate_index_config(self.index_type, self.data_type, self.metric_type)
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
    /// For MinHash-LSH: number of bits for hash
    #[serde(default)]
    pub num_bit: Option<usize>,
    /// For MinHash-LSH: number of bands
    #[serde(default)]
    pub num_band: Option<usize>,
    /// For MinHash-LSH: block size in bytes
    #[serde(default)]
    pub block_size: Option<usize>,
    /// For MinHash-LSH: whether to store raw data
    #[serde(default)]
    pub with_raw_data: Option<bool>,
    /// For MinHash-LSH: whether to use Bloom filter
    #[serde(default)]
    pub use_bloom: Option<bool>,
    /// For MinHash-LSH: Bloom filter false positive rate
    #[serde(default)]
    pub bloom_fp_rate: Option<f64>,
    /// For IVF: use mini-batch k-means (faster for large datasets)
    #[serde(default)]
    pub use_mini_batch: Option<bool>,
    /// For mini-batch k-means: batch size (default: 10000)
    #[serde(default)]
    pub mini_batch_size: Option<usize>,
    /// For k-means: maximum iterations (default: 100)
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// For k-means: convergence tolerance (default: 1e-4)
    #[serde(default)]
    pub kmeans_tolerance: Option<f32>,
    /// For IVF: use k-means++ initialization (better quality, fewer iterations)
    #[serde(default)]
    pub use_kmeans_pp: Option<bool>,
    /// Random seed for k-means (default: 42)
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// OPT-024: Number of threads for parallel index construction (HNSW, etc.)
    #[serde(default)]
    pub num_threads: Option<usize>,
    /// OPT-030: Adaptive ef multiplier for HNSW search
    /// Formula: ef = max(base_ef, adaptive_k * top_k)
    /// Default: 2.0 (same as OPT-016 dynamic ef strategy)
    #[serde(default)]
    pub hnsw_adaptive_k: Option<f64>,
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

    pub fn ivf_sq8(nlist: usize, nprobe: usize) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
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

    pub fn bin_ivf_flat(nlist: usize, nprobe: usize) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            ..Default::default()
        }
    }

    pub fn ivf_mini_batch(
        nlist: usize,
        nprobe: usize,
        batch_size: usize,
        max_iter: usize,
        tol: f32,
    ) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            use_mini_batch: Some(true),
            mini_batch_size: Some(batch_size),
            max_iterations: Some(max_iter),
            kmeans_tolerance: Some(tol),
            ..Default::default()
        }
    }

    pub fn ivf_pp(nlist: usize, nprobe: usize, max_iter: usize, tol: f32, seed: u64) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            use_kmeans_pp: Some(true),
            max_iterations: Some(max_iter),
            kmeans_tolerance: Some(tol),
            random_seed: Some(seed),
            ..Default::default()
        }
    }

    pub fn ivf_elkan(nlist: usize, nprobe: usize, max_iter: usize, tol: f32, seed: u64) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            use_elkan: Some(true),
            max_iterations: Some(max_iter),
            kmeans_tolerance: Some(tol),
            random_seed: Some(seed),
            ..Default::default()
        }
    }

    /// IVF-Flat 快速构建配置
    /// 优化目标：构建速度 <2s (从 5.2s 减少 60%+)，同时保持召回率 R@100 >= 0.90
    /// 参数说明：
    /// - 不设置任何特殊参数，使用标准 K-Means 默认行为
    /// - 速度优化来自并行化的 add() 方法，而非降低聚类质量
    ///
    /// OPT-018 修复：原参数使用 Elkan K-Means 导致聚类行为不一致，召回率仅 0.347
    /// 新策略：保持聚类质量，通过其他优化（并行 add）获得速度提升
    pub fn ivf_flat_fast(nlist: usize, nprobe: usize) -> Self {
        Self {
            nlist: Some(nlist),
            nprobe: Some(nprobe),
            // 不设置 use_elkan/use_kmeans_pp/max_iterations，使用标准 K-Means 默认行为
            ..Default::default()
        }
    }

    pub fn sparse_wand() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn sparse_wand_cc(ssize: Option<usize>) -> Self {
        Self {
            ssize,
            ..Default::default()
        }
    }

    pub fn minhash_lsh(num_bit: usize, num_band: usize, block_size: usize) -> Self {
        Self {
            num_bit: Some(num_bit),
            num_band: Some(num_band),
            block_size: Some(block_size),
            with_raw_data: Some(true),
            use_bloom: Some(true),
            bloom_fp_rate: Some(0.01),
            ..Default::default()
        }
    }

    /// OPT-030: Get adaptive k multiplier for HNSW ef calculation
    /// Returns the configured value or default (2.0)
    pub fn hnsw_adaptive_k(&self) -> f64 {
        self.hnsw_adaptive_k.unwrap_or(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_type_ivf_sq8() {
        // Test enum variant exists
        let index_type = IndexType::IvfSq8;
        assert_eq!(index_type, IndexType::IvfSq8);
    }

    #[test]
    fn test_index_type_from_str_ivf_sq8() {
        assert_eq!(IndexType::from_str("ivf_sq8"), Some(IndexType::IvfSq8));
        assert_eq!(IndexType::from_str("ivf-sq8"), Some(IndexType::IvfSq8));
        assert_eq!(IndexType::from_str("ivfsq8"), Some(IndexType::IvfSq8));
        assert_eq!(IndexType::from_str("IVF_SQ8"), Some(IndexType::IvfSq8));
        assert_eq!(IndexType::from_str("IvfSq8"), Some(IndexType::IvfSq8));
    }

    #[test]
    fn test_index_params_ivf_sq8() {
        let params = IndexParams::ivf_sq8(256, 8);
        assert_eq!(params.nlist, Some(256));
        assert_eq!(params.nprobe, Some(8));
    }

    #[test]
    fn test_index_config_ivf_sq8() {
        let config = IndexConfig::new(IndexType::IvfSq8, MetricType::L2, 128);
        assert_eq!(config.index_type, IndexType::IvfSq8);
        assert_eq!(config.metric_type, MetricType::L2);
        assert_eq!(config.dim, 128);
        assert_eq!(config.data_type, DataType::Float);
    }

    #[test]
    fn test_index_config_validate() {
        // Legal combination
        let config = IndexConfig::new(IndexType::Hnsw, MetricType::L2, 128);
        assert!(config.validate().is_ok());

        // Illegal combination (binary index with L2 metric)
        let config = IndexConfig::with_data_type(
            IndexType::BinFlat,
            MetricType::L2,
            128,
            DataType::Binary,
        );
        assert!(config.validate().is_err());
    }
}

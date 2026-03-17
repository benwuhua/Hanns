//! Index types and configuration

use serde::{Deserialize, Serialize};
use std::str::FromStr;

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

impl FromStr for IndexType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "flat" => Ok(IndexType::Flat),
            "ivf_flat" | "ivf-flat" => Ok(IndexType::IvfFlat),
            "ivf_pq" | "ivf-pq" => Ok(IndexType::IvfPq),
            "hnsw" => Ok(IndexType::Hnsw),
            "diskann" | "disk_ann" => Ok(IndexType::DiskAnn),
            "annoy" => Ok(IndexType::Annoy),
            #[cfg(feature = "scann")]
            "scann" => Ok(IndexType::Scann),
            "hnsw_prq" | "hnsw-prq" => Ok(IndexType::HnswPrq),
            "hnsw_pq" | "hnsw-pq" => Ok(IndexType::HnswPq),
            "ivf_rabitq" | "ivf-rabitq" | "rabitq" => Ok(IndexType::IvfRabitq),
            "ivf_flat_cc" | "ivf-flat-cc" | "ivfcc" => Ok(IndexType::IvfFlatCc),
            "ivf_sq8" | "ivf-sq8" | "ivfsq8" => Ok(IndexType::IvfSq8),
            "ivf_sq_cc" | "ivf-sq-cc" | "ivfsqcc" => Ok(IndexType::IvfSqCc),
            "sparse_inverted" | "sparse-inverted" | "sparse" => Ok(IndexType::SparseInverted),
            "binary_hnsw" | "binary-hnsw" | "binaryhnsw" => Ok(IndexType::BinaryHnsw),
            "bin_flat" | "bin-flat" | "binflat" | "binary_flat" | "binary-flat" => {
                Ok(IndexType::BinFlat)
            }
            "bin_ivf_flat" | "bin-ivf-flat" | "binivfflat" | "binary_ivf_flat"
            | "binary-ivf-flat" => Ok(IndexType::BinIvfFlat),
            "hnsw_sq" | "hnsw-sq" | "hnswsq" => Ok(IndexType::HnswSq),
            "aisaq" | "a_isaq" | "a-saq" => Ok(IndexType::Aisaq),
            "sparse_inverted_cc" | "sparse-inverted-cc" | "sparsecc" => {
                Ok(IndexType::SparseInvertedCc)
            }
            "sparse_wand" | "sparse-wand" | "sparsewand" | "wand" => Ok(IndexType::SparseWand),
            "sparse_wand_cc" | "sparse-wand-cc" | "sparsewandcc" | "wandcc" => {
                Ok(IndexType::SparseWandCc)
            }
            "minhash_lsh" | "minhash-lsh" | "minhashlsh" => Ok(IndexType::MinHashLsh),
            _ => Err(format!("unknown index type: {s}")),
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

impl FromStr for MetricType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" | "l2_distance" => Ok(MetricType::L2),
            "ip" | "inner_product" => Ok(MetricType::Ip),
            "cosine" | "cos" => Ok(MetricType::Cosine),
            "hamming" => Ok(MetricType::Hamming),
            _ => Err(format!("unknown metric type: {s}")),
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
    /// For DiskANN: number of PQ sub-dimensions (0 = disabled)
    #[serde(default)]
    pub disk_pq_dims: Option<usize>,
    /// For DiskANN/AISAQ PQ path: compressed code budget in GB (0 = disabled)
    #[serde(default)]
    pub disk_pq_code_budget_gb: Option<f32>,
    /// For AISAQ: in-memory PQ cache entry capacity (0 = unbounded)
    #[serde(default)]
    pub disk_pq_cache_size: Option<usize>,
    /// For DiskANN PQ path: candidate pool expansion percentage (100 = off, 125 = +25%)
    #[serde(default)]
    pub disk_pq_candidate_expand_pct: Option<usize>,
    /// For DiskANN build prune: fill adjacency to max_degree after alpha pruning
    #[serde(default)]
    pub disk_saturate_after_prune: Option<bool>,
    /// For DiskANN build: additional temporal candidates from recent inserts
    #[serde(default)]
    pub disk_intra_batch_candidates: Option<usize>,
    /// For DiskANN search: number of entry points (medoid-like starts), default 1
    #[serde(default)]
    pub disk_num_entry_points: Option<usize>,
    /// For DiskANN PQ search: rerank pool expansion percentage (100 = disabled)
    #[serde(default)]
    pub disk_rerank_expand_pct: Option<usize>,
    /// For DiskANN build: temporary degree slack percentage (100 = no slack)
    #[serde(default)]
    pub disk_build_degree_slack_pct: Option<usize>,
    /// For DiskANN build: number of random initial candidate edges per inserted node (0 = disabled)
    #[serde(default)]
    pub disk_random_init_edges: Option<usize>,
    /// For DiskANN parallel build: batch size for parallel candidate search
    #[serde(default)]
    pub disk_build_parallel_batch_size: Option<usize>,
    /// For DiskANN build: DRAM budget in GB (0 = disabled)
    #[serde(default)]
    pub disk_build_dram_budget_gb: Option<f32>,
    /// For DiskANN search cache: DRAM budget in GB (0 = disabled)
    #[serde(default)]
    pub disk_search_cache_budget_gb: Option<f32>,
    /// For DiskANN persistence: write/read fixed-stride flash layout sidecar
    #[serde(default)]
    pub disk_enable_flash_layout: Option<bool>,
    /// For DiskANN flash layout: load sidecar via mmap for runtime reads
    #[serde(default)]
    pub disk_flash_mmap_mode: Option<bool>,
    /// For DiskANN flash mmap path: per-expansion prefetch batch size (0 = disabled)
    #[serde(default)]
    pub disk_flash_prefetch_batch: Option<usize>,
    /// For DiskANN/AISAQ: warm-up query-time cache state after build/load
    #[serde(default)]
    pub disk_warm_up: Option<bool>,
    /// For DiskANN/AISAQ filtered search: threshold gate for exact fallback (-1 = disabled)
    #[serde(default)]
    pub disk_filter_threshold: Option<f32>,
    /// For AISAQ: enable expanded exact rerank stage (default true)
    #[serde(default)]
    pub disk_rearrange: Option<bool>,
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
    /// For MinHash-LSH: number of bits for each MinHash element
    /// C++ parity aliases: mh_element_bit_width
    #[serde(default, alias = "mh_element_bit_width")]
    pub num_bit: Option<usize>,
    /// For MinHash-LSH: number of bands
    /// C++ parity aliases: mh_lsh_band
    #[serde(default, alias = "mh_lsh_band")]
    pub num_band: Option<usize>,
    /// For MinHash-LSH: aligned block size in bytes
    /// C++ parity aliases: mh_lsh_aligned_block_size
    #[serde(default, alias = "mh_lsh_aligned_block_size")]
    pub block_size: Option<usize>,
    /// For MinHash-LSH: whether to store raw data
    #[serde(default)]
    pub with_raw_data: Option<bool>,
    /// For MinHash-LSH: whether to use shared/global Bloom filter
    /// C++ parity aliases: mh_lsh_shared_bloom_filter
    #[serde(default, alias = "mh_lsh_shared_bloom_filter")]
    pub use_bloom: Option<bool>,
    /// For MinHash-LSH: Bloom filter false positive rate
    /// C++ parity aliases: mh_lsh_bloom_false_positive_prob
    #[serde(default, alias = "mh_lsh_bloom_false_positive_prob")]
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
    /// Set to 0.0 to disable the adaptive floor and honor the requested/base ef.
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

    /// Resolve the effective HNSW ef_search after applying the optional adaptive floor.
    pub fn effective_hnsw_ef_search(
        &self,
        base_ef_search: usize,
        requested_ef_search: usize,
        top_k: usize,
    ) -> usize {
        let adaptive_floor = (self.hnsw_adaptive_k().max(0.0) * top_k as f64) as usize;
        base_ef_search
            .max(requested_ef_search.max(1))
            .max(adaptive_floor)
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
        assert_eq!("ivf_sq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("ivf-sq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("ivfsq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("IVF_SQ8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("IvfSq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
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
        let config =
            IndexConfig::with_data_type(IndexType::BinFlat, MetricType::L2, 128, DataType::Binary);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_minhash_cpp_param_aliases_deserialize() {
        let json = r#"{
            "index_type": "min_hash_lsh",
            "metric_type": "hamming",
            "dim": 128,
            "data_type": "binary",
            "params": {
                "mh_element_bit_width": 32,
                "mh_lsh_band": 8,
                "mh_lsh_aligned_block_size": 4096,
                "with_raw_data": true,
                "mh_lsh_shared_bloom_filter": true,
                "mh_lsh_bloom_false_positive_prob": 0.01
            }
        }"#;

        let cfg: IndexConfig = serde_json::from_str(json).expect("deserialize minhash config");

        assert_eq!(cfg.params.num_bit, Some(32));
        assert_eq!(cfg.params.num_band, Some(8));
        assert_eq!(cfg.params.block_size, Some(4096));
        assert_eq!(cfg.params.with_raw_data, Some(true));
        assert_eq!(cfg.params.use_bloom, Some(true));
        assert_eq!(cfg.params.bloom_fp_rate, Some(0.01));
    }

    #[test]
    fn test_hnsw_effective_ef_search_defaults_to_adaptive_floor() {
        let params = IndexParams {
            ef_search: Some(138),
            ..Default::default()
        };

        assert_eq!(params.effective_hnsw_ef_search(138, 138, 100), 200);
    }

    #[test]
    fn test_hnsw_effective_ef_search_allows_disabling_adaptive_floor() {
        let params = IndexParams {
            ef_search: Some(138),
            hnsw_adaptive_k: Some(0.0),
            ..Default::default()
        };

        assert_eq!(params.effective_hnsw_ef_search(138, 138, 100), 138);
    }
}

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
            _ => None,
        }
    }
    
    pub fn from_bytes(b: u8) -> Self {
        match b {
            0 => MetricType::L2,
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
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

    pub fn pq(m: usize, nbits_per_idx: usize) -> Self {
        Self {
            m: Some(m),
            nbits_per_idx: Some(nbits_per_idx),
            ..Default::default()
        }
    }
}

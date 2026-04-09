//! DiskANN AISAQ skeleton with PQ Flash abstractions.
//!
//! This module provides a first Rust architecture for the SSD-oriented AISAQ path.
//! The current implementation keeps data in memory while preserving the core
//! concepts from the C++ design: flash layout, PQ-compressed payloads, beam-search
//! IO accounting, multiple entry points, and coarse-to-exact search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[cfg(test)]
use std::future::Future;
#[cfg(all(feature = "async-io", target_os = "linux"))]
use std::os::fd::AsRawFd;

use crate::api::{IndexConfig, KnowhereError, MetricType, Result, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{Index, IndexError};
use crate::quantization::hvq::{HvqConfig, HvqQuantizer, HvqQueryState};
use crate::quantization::{PQConfig, ProductQuantizer};
#[cfg(all(feature = "async-io", target_os = "linux"))]
use crate::search::VisitedList;
use crate::search::{is_visited_thread_local, mark_thread_local, reset_thread_local, with_visited};
use crate::simd;
#[cfg(all(feature = "async-io", target_os = "linux"))]
use io_uring::{opcode, types, IoUring};
use memmap2::Mmap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

const DEFAULT_PAGE_SIZE: usize = 4096;
const DEFAULT_PQ_K: usize = 256;
const PAGE_CACHE_SHARDS: usize = 256;

/// AISAQ configuration parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AisaqConfig {
    pub max_degree: usize,
    pub search_list_size: usize,
    pub beamwidth: usize,
    pub vectors_beamwidth: usize,
    pub pq_code_budget_gb: f32,
    pub build_dram_budget_gb: f32,
    pub disk_pq_dims: usize,
    pub pq_cache_size: usize,
    pub pq_read_page_cache_size: usize,
    pub rearrange: bool,
    pub inline_pq: usize,
    pub num_entry_points: usize,
    /// Exact rerank pool expansion percentage over top-k.
    pub rerank_expand_pct: usize,
    /// PQ candidate expansion percentage for search visit budget (100 = disabled)
    pub pq_candidate_expand_pct: usize,
    /// Random initial candidate edges per inserted node (0 = disabled)
    pub random_init_edges: usize,
    /// Seed for deterministic randomized build behavior.
    pub random_seed: u64,
    /// Build-time temporary degree slack percentage (100 = disabled)
    pub build_degree_slack_pct: usize,
    /// Build-time Vamana search list size (0 = auto derive from search_list_size/max_degree)
    #[serde(default)]
    pub build_search_list_size: usize,
    /// Build batch size for parallel insertion path.
    #[serde(default = "default_build_batch_size")]
    pub build_batch_size: usize,
    /// Search-time beam batch size for sync sector-batch expansion.
    #[serde(default = "default_beam_batch_size")]
    pub beam_batch_size: usize,
    /// Cross-query io_uring group size (default 8). Only used on Linux with async-io feature.
    #[serde(default = "default_uring_group_size")]
    pub uring_group_size: usize,
    pub warm_up: bool,
    pub filter_threshold: f32,
    #[serde(default)]
    pub search_io_limit: Option<usize>,
    #[serde(default)]
    pub cache_all_on_load: bool,
    #[serde(default = "default_true")]
    pub run_refine_pass: bool,
    /// Number of full-graph Vamana refinement passes after insertion (0 = disabled, default = 1)
    #[serde(default = "default_refine_passes")]
    pub num_refine_passes: usize,
    /// Frontier-bound early-stop slack factor.
    /// 1.0 = strict bound, >1.0 = conservative bound, f32::MAX = effectively disabled.
    #[serde(default = "default_early_stop_alpha")]
    pub early_stop_alpha: f32,
    /// Enable IO cutting (beam search early termination on convergence).
    #[serde(default)]
    pub io_cutting_enabled: bool,
    /// Minimum relative improvement required to reset IO cutting stale rounds.
    #[serde(default = "default_io_cutting_threshold")]
    pub io_cutting_threshold: f32,
    /// Number of stale rounds before IO cutting terminates the beam search.
    #[serde(default = "default_io_cutting_patience")]
    pub io_cutting_patience: usize,
    /// Use HVQ quantizer for coarse beam-search scoring (alternative to PQ).
    #[serde(default)]
    pub use_hvq: bool,
    /// HVQ bits per dimension (1-8).
    #[serde(default = "default_hvq_nbits")]
    pub hvq_nbits: usize,
    /// HVQ encoding refinement iterations.
    #[serde(default = "default_hvq_nrefine")]
    pub hvq_nrefine: usize,
    /// Enable SQ8 prefilter distance in beam-search hot path.
    #[serde(default)]
    pub use_sq8_prefilter: bool,
}

fn default_true() -> bool {
    true
}

fn default_refine_passes() -> usize {
    1
}

fn default_early_stop_alpha() -> f32 {
    1.5
}

fn default_io_cutting_threshold() -> f32 {
    0.02
}

fn default_io_cutting_patience() -> usize {
    3
}

fn default_hvq_nbits() -> usize {
    4
}

fn default_hvq_nrefine() -> usize {
    3
}

fn default_build_batch_size() -> usize {
    512
}

fn default_beam_batch_size() -> usize {
    8
}

fn default_uring_group_size() -> usize {
    8
}

#[allow(dead_code)] // Retained for the unfinished batch-build path that is not enabled in production yet.
const GRAPH_SLACK_FACTOR: f32 = 1.3;
const ROBUST_PRUNE_ALPHA: f32 = 1.2;

#[allow(dead_code)] // Retained for the unfinished batch-build path that is not enabled in production yet.
fn graph_slack_stride(max_degree: usize) -> usize {
    ((max_degree.max(1) as f32) * GRAPH_SLACK_FACTOR).ceil() as usize
}

impl Default for AisaqConfig {
    fn default() -> Self {
        Self {
            max_degree: 48,
            search_list_size: 128,
            beamwidth: 8,
            vectors_beamwidth: 1,
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            pq_cache_size: 0,
            pq_read_page_cache_size: 0,
            rearrange: true,
            inline_pq: 0,
            num_entry_points: 1,
            rerank_expand_pct: 200,
            pq_candidate_expand_pct: 100,
            random_init_edges: 0,
            random_seed: 42,
            build_degree_slack_pct: 100,
            build_search_list_size: 0,
            build_batch_size: 512,
            beam_batch_size: 8,
            uring_group_size: 8,
            warm_up: false,
            filter_threshold: -1.0,
            search_io_limit: None,
            cache_all_on_load: false,
            run_refine_pass: true,
            num_refine_passes: 2,
            early_stop_alpha: 1.5,
            io_cutting_enabled: false,
            io_cutting_threshold: 0.02,
            io_cutting_patience: 3,
            use_hvq: false,
            hvq_nbits: 4,
            hvq_nrefine: 3,
            use_sq8_prefilter: false,
        }
    }
}

impl AisaqConfig {
    pub fn from_index_config(config: &IndexConfig) -> Self {
        let params = &config.params;
        Self {
            max_degree: params.max_degree.unwrap_or(48),
            search_list_size: params.search_list_size.unwrap_or(128),
            beamwidth: params.beamwidth.unwrap_or(8),
            disk_pq_dims: params.disk_pq_dims.unwrap_or(0),
            pq_code_budget_gb: params.disk_pq_code_budget_gb.unwrap_or(0.0).max(0.0),
            pq_cache_size: params.disk_pq_cache_size.unwrap_or(0),
            rearrange: params.disk_rearrange.unwrap_or(true),
            num_entry_points: params.disk_num_entry_points.unwrap_or(1).clamp(1, 64),
            rerank_expand_pct: params.disk_rerank_expand_pct.unwrap_or(200).clamp(100, 400),
            pq_candidate_expand_pct: params
                .disk_pq_candidate_expand_pct
                .unwrap_or(100)
                .clamp(100, 300),
            random_init_edges: params.disk_random_init_edges.unwrap_or(0).min(64),
            random_seed: params.random_seed.unwrap_or(42),
            build_degree_slack_pct: params
                .disk_build_degree_slack_pct
                .unwrap_or(100)
                .clamp(100, 300),
            build_search_list_size: 0,
            build_dram_budget_gb: params.disk_build_dram_budget_gb.unwrap_or(0.0),
            pq_read_page_cache_size: gb_to_bytes(params.disk_search_cache_budget_gb.unwrap_or(0.0)),
            warm_up: params.disk_warm_up.unwrap_or(false),
            filter_threshold: params
                .disk_filter_threshold
                .unwrap_or(-1.0)
                .clamp(-1.0, 1.0),
            io_cutting_enabled: params.disk_io_cutting.unwrap_or(false),
            io_cutting_threshold: params.disk_io_cutting_threshold.unwrap_or(0.02).max(0.0),
            io_cutting_patience: params.disk_io_cutting_patience.unwrap_or(3).max(1),
            ..Self::default()
        }
    }

    fn validate(&self, dim: usize) -> Result<()> {
        if dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "dimension must be greater than 0".to_string(),
            ));
        }
        if self.max_degree == 0 {
            return Err(KnowhereError::InvalidArg(
                "max_degree must be greater than 0".to_string(),
            ));
        }
        if self.search_list_size == 0 {
            return Err(KnowhereError::InvalidArg(
                "search_list_size must be greater than 0".to_string(),
            ));
        }
        if self.beamwidth == 0 {
            return Err(KnowhereError::InvalidArg(
                "beamwidth must be greater than 0".to_string(),
            ));
        }
        if self.beam_batch_size == 0 {
            return Err(KnowhereError::InvalidArg(
                "beam_batch_size must be greater than 0".to_string(),
            ));
        }
        if self.num_entry_points == 0 {
            return Err(KnowhereError::InvalidArg(
                "num_entry_points must be greater than 0".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.filter_threshold) {
            return Err(KnowhereError::InvalidArg(
                "filter_threshold must be in [-1.0, 1.0]".to_string(),
            ));
        }
        if self.io_cutting_patience == 0 {
            return Err(KnowhereError::InvalidArg(
                "io_cutting_patience must be greater than 0".to_string(),
            ));
        }
        if !(1..=8).contains(&self.hvq_nbits) {
            return Err(KnowhereError::InvalidArg(
                "hvq_nbits must be in range 1..=8".to_string(),
            ));
        }
        if self.build_degree_slack_pct < 100 {
            return Err(KnowhereError::InvalidArg(
                "build_degree_slack_pct must be at least 100".to_string(),
            ));
        }
        if self.disk_pq_dims > 0 && dim < 2 {
            return Err(KnowhereError::InvalidArg(
                "disk_pq_dims requires dimension >= 2".to_string(),
            ));
        }
        Ok(())
    }
}

/// Logical flash layout for a node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlashLayout {
    pub page_size: usize,
    pub vector_bytes: usize,
    pub inline_pq_bytes: usize,
    pub neighbor_bytes: usize,
    #[serde(default)]
    pub id_bytes: usize,
    pub node_bytes: usize,
    #[serde(default)]
    pub has_separated_vectors: bool,
}

impl FlashLayout {
    fn new(dim: usize, config: &AisaqConfig) -> Self {
        let vector_bytes = dim * std::mem::size_of::<f32>();
        let inline_pq_bytes = config.inline_pq.max(config.disk_pq_dims);
        let neighbor_bytes = (config.max_degree + 1) * std::mem::size_of::<u32>();
        let id_bytes = std::mem::size_of::<i64>();
        let has_separated_vectors = config.disk_pq_dims > 0;
        let hot_vector_bytes = if has_separated_vectors {
            0
        } else {
            vector_bytes
        };
        let node_bytes = hot_vector_bytes + neighbor_bytes + inline_pq_bytes + id_bytes;

        Self {
            page_size: DEFAULT_PAGE_SIZE,
            vector_bytes,
            inline_pq_bytes,
            neighbor_bytes,
            id_bytes,
            node_bytes,
            has_separated_vectors,
        }
    }
}

/// Per-query beam-search statistics.
#[derive(Clone, Debug, Default)]
pub struct BeamSearchStats {
    pub nodes_visited: usize,
    pub nodes_loaded: usize,
    pub node_cache_hits: usize,
    pub pq_cache_hits: usize,
    pub bytes_read: usize,
    pub pages_read: usize,
}

#[derive(Clone, Debug, Default)]
pub struct PageCacheStats {
    pub requests: usize,
    pub page_hits: usize,
    pub page_misses: usize,
    pub evictions: usize,
}

impl PageCacheStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.page_hits + self.page_misses;
        if total == 0 {
            return 1.0;
        }
        self.page_hits as f32 / total as f32
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AisaqScopeAudit {
    pub dim: usize,
    pub node_count: usize,
    pub entry_point_count: usize,
    pub uses_flash_layout: bool,
    pub uses_beam_search_io: bool,
    pub uses_mmap_backed_pages: bool,
    pub has_page_cache: bool,
    pub native_comparable: bool,
    pub comparability_reason: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsyncReadEngine {
    SyncMmap,
    ReservedAio,
    ReservedIoUring,
}

impl AsyncReadEngine {
    /// Check if async IO is available on this platform
    pub fn is_available(&self) -> bool {
        match self {
            AsyncReadEngine::SyncMmap => true,
            AsyncReadEngine::ReservedAio => false,
            AsyncReadEngine::ReservedIoUring => {
                #[cfg(all(feature = "async-io", target_os = "linux"))]
                {
                    true
                }
                #[cfg(not(all(feature = "async-io", target_os = "linux")))]
                {
                    false
                }
            }
        }
    }

    /// Get the recommended engine for this platform
    pub fn recommended() -> Self {
        #[cfg(all(feature = "async-io", target_os = "linux"))]
        {
            AsyncReadEngine::ReservedIoUring
        }
        #[cfg(not(all(feature = "async-io", target_os = "linux")))]
        {
            AsyncReadEngine::SyncMmap
        }
    }
}

/// IO accounting and cache hooks for beam search.
#[derive(Clone, Debug)]
pub struct BeamSearchIO {
    page_size: usize,
    max_reads_per_iteration: usize,
    pq_cache_capacity: Option<usize>,
    pages_loaded_total: usize,
    cached_nodes: HashSet<u32>,
    cached_pq_vectors: HashSet<u32>,
    pq_lru: VecDeque<u32>,
    stats: BeamSearchStats,
}

impl BeamSearchIO {
    pub fn new(layout: &FlashLayout, config: &AisaqConfig) -> Self {
        Self {
            page_size: layout.page_size.max(1),
            max_reads_per_iteration: config.beamwidth.max(1),
            pq_cache_capacity: (config.pq_cache_size > 0).then_some(config.pq_cache_size),
            pages_loaded_total: 0,
            cached_nodes: HashSet::new(),
            cached_pq_vectors: HashSet::new(),
            pq_lru: VecDeque::new(),
            stats: BeamSearchStats::default(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.stats = BeamSearchStats::default();
        self.pages_loaded_total = 0;
    }

    pub fn max_reads_per_iteration(&self) -> usize {
        self.max_reads_per_iteration
    }

    pub fn pages_loaded_total(&self) -> usize {
        self.pages_loaded_total
    }

    pub fn cache_node(&mut self, node_id: u32) {
        self.cached_nodes.insert(node_id);
    }

    pub fn cache_pq_vector(&mut self, node_id: u32) {
        if self.cached_pq_vectors.insert(node_id) {
            self.pq_lru.push_back(node_id);
        } else {
            self.pq_lru.retain(|&id| id != node_id);
            self.pq_lru.push_back(node_id);
        }

        if let Some(capacity) = self.pq_cache_capacity {
            while self.cached_pq_vectors.len() > capacity {
                if let Some(evicted) = self.pq_lru.pop_front() {
                    self.cached_pq_vectors.remove(&evicted);
                } else {
                    break;
                }
            }
        }
    }

    pub fn record_node_read(&mut self, node_id: u32, bytes: usize) {
        self.record_node_access(node_id, bytes, bytes.div_ceil(self.page_size));
    }

    pub fn record_node_access(&mut self, node_id: u32, bytes: usize, pages_loaded: usize) {
        self.stats.nodes_visited += 1;
        if self.cached_nodes.contains(&node_id) || pages_loaded == 0 {
            self.stats.node_cache_hits += 1;
            return;
        }

        self.pages_loaded_total += pages_loaded;
        self.stats.nodes_loaded += 1;
        self.stats.bytes_read += bytes;
        self.stats.pages_read += pages_loaded;
    }

    pub fn record_pq_read(&mut self, node_id: u32, bytes: usize) {
        self.record_pq_access(node_id, bytes, bytes.div_ceil(self.page_size));
    }

    pub fn record_pq_access(&mut self, node_id: u32, bytes: usize, pages_loaded: usize) {
        if self.cached_pq_vectors.contains(&node_id) || pages_loaded == 0 {
            self.stats.pq_cache_hits += 1;
            return;
        }

        self.pages_loaded_total += pages_loaded;
        self.stats.bytes_read += bytes;
        self.stats.pages_read += pages_loaded;
    }

    pub fn stats(&self) -> &BeamSearchStats {
        &self.stats
    }
}

#[derive(Clone, Debug)]
pub struct LoadedNode {
    id: i64,
    vector: Vec<f32>,
    neighbors: Vec<u32>,
    inline_pq: Vec<u8>,
}

struct LoadedNodeRef<'a> {
    id: i64,
    vector: &'a [f32],
    neighbors: &'a [u32],
    inline_pq: &'a [u8],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedPq {
    m: usize,
    k: usize,
    nbits: usize,
    dim: usize,
    sub_dim: usize,
    centroids: Vec<f32>,
}

impl SerializedPq {
    fn from_quantizer(quantizer: &ProductQuantizer) -> Self {
        Self {
            m: quantizer.m(),
            k: quantizer.ksub(),
            nbits: quantizer.nbits(),
            dim: quantizer.dim(),
            sub_dim: quantizer.sub_dim(),
            centroids: quantizer.centroids().to_vec(),
        }
    }

    fn into_quantizer(self) -> ProductQuantizer {
        let mut quantizer = ProductQuantizer::new(PQConfig::new(self.dim, self.m, self.nbits));
        debug_assert_eq!(self.k, quantizer.ksub());
        debug_assert_eq!(self.sub_dim, quantizer.sub_dim());
        quantizer
            .set_centroids(self.centroids)
            .expect("serialized PQ centroids must match config");
        quantizer
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializedHvq {
    dim: usize,
    nbits: u8,
    rotation_matrix: Vec<f32>,
    scale: f32,
    offset: f32,
    centroid: Vec<f32>,
    rotated_centroid: Vec<f32>,
}

impl SerializedHvq {
    fn from_quantizer(quantizer: &HvqQuantizer) -> Self {
        Self {
            dim: quantizer.config.dim,
            nbits: quantizer.config.nbits,
            rotation_matrix: quantizer.rotation_matrix.clone(),
            scale: quantizer.scale,
            offset: quantizer.offset,
            centroid: quantizer.centroid.clone(),
            rotated_centroid: quantizer.rotated_centroid.clone(),
        }
    }

    fn into_quantizer(self) -> HvqQuantizer {
        HvqQuantizer {
            config: HvqConfig {
                dim: self.dim,
                nbits: self.nbits,
            },
            rotation_matrix: self.rotation_matrix,
            scale: self.scale,
            offset: self.offset,
            centroid: self.centroid,
            rotated_centroid: self.rotated_centroid,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AisaqMetadata {
    version: u32,
    config: AisaqConfig,
    metric_type: MetricType,
    dim: usize,
    flash_layout: FlashLayout,
    pq_code_size: usize,
    entry_points: Vec<u32>,
    trained: bool,
    node_count: usize,
    pq_encoder: Option<SerializedPq>,
    #[serde(default)]
    hvq_quantizer: Option<SerializedHvq>,
}

#[derive(Clone, Debug)]
pub struct FileGroup {
    root: PathBuf,
}

impl FileGroup {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn create<P: AsRef<Path>>(root: P) -> Result<Self> {
        let group = Self::new(root);
        fs::create_dir_all(&group.root)?;
        Ok(group)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn metadata_path(&self) -> PathBuf {
        self.root.join("aisaq.meta")
    }

    pub fn data_path(&self) -> PathBuf {
        self.root.join("aisaq.nodes")
    }

    pub fn vectors_path(&self) -> PathBuf {
        self.root.join("aisaq.vectors")
    }
}

#[derive(Debug)]
struct PageRead {
    bytes: Vec<u8>,
    pages_loaded: usize,
}

#[derive(Debug)]
struct PageCacheShard {
    pages: HashMap<usize, std::sync::Arc<Vec<u8>>>,
    lru: VecDeque<usize>,
    capacity: usize,
    stats: PageCacheStats,
}

#[derive(Debug)]
pub struct PageCache {
    page_size: usize,
    mmap: Mmap,
    requests: AtomicUsize,
    shards: Vec<Mutex<PageCacheShard>>,
}

impl PageCache {
    pub fn open<P: AsRef<Path>>(path: P, page_size: usize, cache_bytes: usize) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let capacity_pages = cache_bytes.div_ceil(page_size.max(1)).max(1);

        Ok(Self {
            page_size: page_size.max(1),
            mmap,
            requests: AtomicUsize::new(0),
            shards: (0..PAGE_CACHE_SHARDS)
                .map(|_| {
                    Mutex::new(PageCacheShard {
                        pages: HashMap::new(),
                        lru: VecDeque::new(),
                        capacity: (capacity_pages / PAGE_CACHE_SHARDS).max(1),
                        stats: PageCacheStats::default(),
                    })
                })
                .collect(),
        })
    }

    fn read(&self, offset: usize, len: usize) -> Result<PageRead> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| KnowhereError::Storage("page cache read overflow".to_string()))?;
        if end > self.mmap.len() {
            return Err(KnowhereError::Storage(format!(
                "page cache read [{}..{}) exceeds file size {}",
                offset,
                end,
                self.mmap.len()
            )));
        }

        let start_page = offset / self.page_size;
        let end_page = end.saturating_sub(1) / self.page_size;
        self.requests.fetch_add(1, AtomicOrdering::Relaxed);

        let mut pages_loaded = 0usize;
        let mut page_buffers = Vec::with_capacity(end_page - start_page + 1);
        for page_id in start_page..=end_page {
            let shard_idx = page_id % PAGE_CACHE_SHARDS;
            let mut shard = self.shards[shard_idx].lock();
            let page = if let Some(arc) = shard.pages.get(&page_id) {
                let page = std::sync::Arc::clone(arc); // clone while borrow is live
                shard.stats.page_hits += 1;
                touch_lru(&mut shard.lru, page_id);
                page
            } else {
                shard.stats.page_misses += 1;
                pages_loaded += 1;
                let page_start = page_id * self.page_size;
                let page_end = (page_start + self.page_size).min(self.mmap.len());
                let page = std::sync::Arc::new(self.mmap[page_start..page_end].to_vec());
                shard.pages.insert(page_id, std::sync::Arc::clone(&page));
                shard.lru.push_back(page_id);
                evict_if_needed(&mut shard);
                page
            };
            page_buffers.push((page_id, page));
        }

        let mut bytes = Vec::with_capacity(len);
        for (index, (_, page)) in page_buffers.iter().enumerate() {
            let slice_start = if index == 0 {
                offset % self.page_size
            } else {
                0
            };
            let slice_end = if index == page_buffers.len() - 1 {
                ((end - 1) % self.page_size) + 1
            } else {
                page.len()
            };
            bytes.extend_from_slice(&page[slice_start..slice_end]);
        }

        Ok(PageRead {
            bytes,
            pages_loaded,
        })
    }

    fn prefetch(&self, offset: usize, len: usize) -> Result<usize> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| KnowhereError::Storage("page cache prefetch overflow".to_string()))?;
        if end > self.mmap.len() {
            return Err(KnowhereError::Storage(format!(
                "page cache prefetch [{}..{}) exceeds file size {}",
                offset,
                end,
                self.mmap.len()
            )));
        }

        let start_page = offset / self.page_size;
        let end_page = end.saturating_sub(1) / self.page_size;
        self.requests.fetch_add(1, AtomicOrdering::Relaxed);

        let mut pages_loaded = 0usize;
        for page_id in start_page..=end_page {
            let shard_idx = page_id % PAGE_CACHE_SHARDS;
            let mut shard = self.shards[shard_idx].lock();
            if shard.pages.contains_key(&page_id) {
                shard.stats.page_hits += 1;
                touch_lru(&mut shard.lru, page_id);
            } else {
                shard.stats.page_misses += 1;
                pages_loaded += 1;
                let page_start = page_id * self.page_size;
                let page_end = (page_start + self.page_size).min(self.mmap.len());
                let page = std::sync::Arc::new(self.mmap[page_start..page_end].to_vec());
                shard.pages.insert(page_id, page);
                shard.lru.push_back(page_id);
                evict_if_needed(&mut shard);
            }
        }
        Ok(pages_loaded)
    }

    pub fn stats(&self) -> PageCacheStats {
        let mut out = PageCacheStats {
            requests: self.requests.load(AtomicOrdering::Relaxed),
            ..PageCacheStats::default()
        };
        for shard in &self.shards {
            let shard = shard.lock();
            out.page_hits += shard.stats.page_hits;
            out.page_misses += shard.stats.page_misses;
            out.evictions += shard.stats.evictions;
        }
        out
    }
}

fn touch_lru(lru: &mut VecDeque<usize>, page_id: usize) {
    if let Some(position) = lru.iter().position(|&id| id == page_id) {
        lru.remove(position);
    }
    lru.push_back(page_id);
}

fn evict_if_needed(shard: &mut PageCacheShard) {
    while shard.pages.len() > shard.capacity {
        if let Some(oldest) = shard.lru.pop_front() {
            if shard.pages.remove(&oldest).is_some() {
                shard.stats.evictions += 1;
            }
        } else {
            break;
        }
    }
}

#[inline]
fn pages_touched(offset: usize, len: usize, page_size: usize) -> usize {
    if len == 0 || page_size == 0 {
        return 0;
    }
    let start_page = offset / page_size;
    let end_page = offset
        .saturating_add(len.saturating_sub(1))
        .saturating_div(page_size);
    end_page.saturating_sub(start_page).saturating_add(1)
}

#[derive(Debug)]
struct DiskStorage {
    file_group: FileGroup,
    page_cache: PageCache,
    vectors_mmap: Option<Mmap>,
    #[cfg(all(feature = "async-io", target_os = "linux"))]
    raw_file: std::sync::Arc<File>,
}

#[derive(Debug)]
struct DirectMmapStorage {
    file_group: FileGroup,
    mmap: Mmap,
    vectors_mmap: Option<Mmap>,
}

#[derive(Clone, Copy)]
pub enum NodeAccessMode {
    None,
    Node,
    Pq,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    node_id: u32,
    score: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.node_id.cmp(&self.node_id))
    }
}

struct AisaqScratch {
    frontier: BinaryHeap<Candidate>,
    expanded: Vec<Candidate>,
    accepted: Vec<Candidate>,
}

impl AisaqScratch {
    fn new(capacity: usize) -> Self {
        Self {
            frontier: BinaryHeap::with_capacity(capacity),
            expanded: Vec::with_capacity(capacity),
            accepted: Vec::with_capacity(capacity),
        }
    }

    fn reset(&mut self) {
        self.frontier.clear();
        self.expanded.clear();
        self.accepted.clear();
    }
}

impl Default for AisaqScratch {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(all(feature = "async-io", target_os = "linux"))]
struct BeamSearchState {
    query: Vec<f32>,
    k: usize,
    pq_table: Option<Vec<f32>>,
    hvq_state: Option<HvqQueryState>,
    sq8_query: Option<Vec<i16>>,
    scratch: AisaqScratch,
    visited: VisitedList,
    io: BeamSearchIO,
    pending_candidates: Vec<Candidate>,
    done: bool,
}

#[inline]
fn kth_best_score(candidates: &[Candidate], k: usize) -> Option<f32> {
    if k == 0 || candidates.len() < k {
        return None;
    }
    let mut scores: Vec<f32> = candidates.iter().map(|c| c.score).collect();
    scores.sort_by(|a, b| a.total_cmp(b));
    scores.get(k - 1).copied()
}

/// PQ Flash index skeleton for DiskANN AISAQ.
pub struct PQFlashIndex {
    config: AisaqConfig,
    metric_type: MetricType,
    dim: usize,
    flash_layout: FlashLayout,
    vectors: Vec<f32>,
    node_ids: Vec<i64>,
    node_neighbor_ids: Vec<u32>,
    node_neighbor_counts: Vec<u32>,
    node_pq_codes: Vec<u8>,
    /// Flat PQ codes loaded in RAM for disk-backed search fast coarse scoring.
    disk_pq_codes: Vec<u8>,
    flat_stride: usize,
    pq_encoder: Option<ProductQuantizer>,
    pq_code_size: usize,
    hvq_quantizer: Option<HvqQuantizer>,
    hvq_codes: Vec<u8>,
    entry_points: Vec<u32>,
    io_template: BeamSearchIO,
    trained: bool,
    node_count: usize,
    storage: Option<DiskStorage>,
    mmap_storage: Option<DirectMmapStorage>,
    loaded_node_cache: Option<HashMap<u32, std::sync::Arc<LoadedNode>>>,
    deleted_ids: HashSet<usize>,
    sq8_quantizer: Option<crate::quantization::sq::ScalarQuantizer>,
    sq8_codes: Vec<u8>,
    scratch_pool: Mutex<Vec<AisaqScratch>>,
}

impl PQFlashIndex {
    pub fn new(config: AisaqConfig, metric_type: MetricType, dim: usize) -> Result<Self> {
        if metric_type == MetricType::Hamming {
            return Err(KnowhereError::InvalidArg(
                "AISAQ does not support Hamming metric".to_string(),
            ));
        }
        config.validate(dim)?;

        let flash_layout = FlashLayout::new(dim, &config);
        let io_template = BeamSearchIO::new(&flash_layout, &config);

        Ok(Self {
            config,
            metric_type,
            dim,
            flash_layout,
            vectors: Vec::new(),
            node_ids: Vec::new(),
            node_neighbor_ids: Vec::new(),
            node_neighbor_counts: Vec::new(),
            node_pq_codes: Vec::new(),
            disk_pq_codes: Vec::new(),
            flat_stride: 0,
            pq_encoder: None,
            pq_code_size: 0,
            hvq_quantizer: None,
            hvq_codes: Vec::new(),
            entry_points: Vec::new(),
            io_template,
            trained: false,
            node_count: 0,
            storage: None,
            mmap_storage: None,
            loaded_node_cache: None,
            deleted_ids: HashSet::new(),
            sq8_quantizer: None,
            sq8_codes: Vec::new(),
            scratch_pool: Mutex::new(Vec::new()),
        })
    }

    pub fn set_uring_group_size(&mut self, group_size: usize) {
        self.config.uring_group_size = group_size.max(1);
    }

    /// Soft-delete node by external id. Returns true if found and marked.
    pub fn soft_delete(&mut self, external_id: i64) -> bool {
        if (self.storage.is_some() || self.mmap_storage.is_some()) && self.node_ids.is_empty() {
            let _ = self.materialize_storage();
        }
        if let Some(row_id) = self.node_ids.iter().position(|&id| id == external_id) {
            self.deleted_ids.insert(row_id)
        } else {
            false
        }
    }

    pub fn is_deleted(&self, external_id: i64) -> bool {
        if let Some(row_id) = self.node_ids.iter().position(|&id| id == external_id) {
            self.deleted_ids.contains(&row_id)
        } else {
            false
        }
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted_ids.len()
    }

    /// Physically remove deleted rows and remap flat graph.
    pub fn consolidate(&mut self) -> usize {
        if self.deleted_ids.is_empty() {
            return 0;
        }
        if (self.storage.is_some() || self.mmap_storage.is_some())
            && self.node_ids.is_empty()
            && self.materialize_storage().is_err()
        {
            return 0;
        }

        let old_n = self.node_ids.len();
        if old_n == 0 {
            self.deleted_ids.clear();
            return 0;
        }
        let stride = self.flat_stride.max(1);
        let pq_size = self.pq_code_size.max(1);
        let hvq_size = self
            .hvq_quantizer
            .as_ref()
            .map(|hvq| hvq.code_size_bytes())
            .unwrap_or(0);

        let mut old_to_new: Vec<Option<u32>> = vec![None; old_n];
        let mut kept = 0usize;
        for (old, slot) in old_to_new.iter_mut().enumerate().take(old_n) {
            if !self.deleted_ids.contains(&old) {
                *slot = Some(kept as u32);
                kept += 1;
            }
        }
        let removed = old_n.saturating_sub(kept);

        let mut new_vectors = Vec::with_capacity(kept * self.dim);
        let mut new_node_ids = Vec::with_capacity(kept);
        let mut new_neighbor_counts = vec![0u32; kept];
        let mut new_neighbor_ids = vec![0u32; kept * stride];
        let mut new_node_pq_codes = vec![0u8; kept * pq_size];
        let mut new_hvq_codes = vec![0u8; kept * hvq_size];

        for old_idx in 0..old_n {
            let Some(new_idx_u32) = old_to_new[old_idx] else {
                continue;
            };
            let new_idx = new_idx_u32 as usize;

            new_node_ids.push(self.node_ids[old_idx]);
            new_vectors
                .extend_from_slice(&self.vectors[old_idx * self.dim..(old_idx + 1) * self.dim]);

            let old_pq_start = old_idx * pq_size;
            let new_pq_start = new_idx * pq_size;
            if old_pq_start + pq_size <= self.node_pq_codes.len()
                && new_pq_start + pq_size <= new_node_pq_codes.len()
            {
                new_node_pq_codes[new_pq_start..new_pq_start + pq_size]
                    .copy_from_slice(&self.node_pq_codes[old_pq_start..old_pq_start + pq_size]);
            }
            if hvq_size > 0 {
                let old_hvq_start = old_idx * hvq_size;
                let new_hvq_start = new_idx * hvq_size;
                if old_hvq_start + hvq_size <= self.hvq_codes.len()
                    && new_hvq_start + hvq_size <= new_hvq_codes.len()
                {
                    new_hvq_codes[new_hvq_start..new_hvq_start + hvq_size]
                        .copy_from_slice(&self.hvq_codes[old_hvq_start..old_hvq_start + hvq_size]);
                }
            }

            let old_start = old_idx * stride;
            let old_deg = self.node_neighbor_counts[old_idx] as usize;
            let mut remapped = Vec::with_capacity(old_deg);
            for slot in 0..old_deg {
                let nb = self.node_neighbor_ids[old_start + slot] as usize;
                if nb >= old_n || self.deleted_ids.contains(&nb) {
                    continue;
                }
                if let Some(new_nb) = old_to_new[nb] {
                    remapped.push(new_nb);
                }
            }
            let new_deg = remapped.len().min(stride);
            new_neighbor_counts[new_idx] = new_deg as u32;
            let new_start = new_idx * stride;
            for (i, nb) in remapped.into_iter().take(new_deg).enumerate() {
                new_neighbor_ids[new_start + i] = nb;
            }
        }

        let mut new_entry_points = Vec::new();
        for &old_ep in &self.entry_points {
            if let Some(new_ep) = old_to_new.get(old_ep as usize).and_then(|v| *v) {
                if !new_entry_points.contains(&new_ep) {
                    new_entry_points.push(new_ep);
                }
            }
        }
        if new_entry_points.is_empty() && kept > 0 {
            new_entry_points.push(0);
        }

        self.vectors = new_vectors;
        self.node_ids = new_node_ids;
        self.node_neighbor_counts = new_neighbor_counts;
        self.node_neighbor_ids = new_neighbor_ids;
        self.node_pq_codes = new_node_pq_codes;
        self.hvq_codes = new_hvq_codes;
        self.disk_pq_codes.clear(); // stale after ID remapping; reload if needed
        self.entry_points = new_entry_points;
        self.node_count = kept;
        self.deleted_ids.clear();
        self.sq8_quantizer = None;
        self.sq8_codes.clear();
        removed
    }

    pub fn train(&mut self, training_data: &[f32]) -> Result<()> {
        self.materialize_storage()?;
        self.validate_vectors(training_data)?;

        if training_data.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "training data must not be empty".to_string(),
            ));
        }

        if self.config.disk_pq_dims > 0 {
            let m = resolve_pq_chunks(self.dim, self.config.disk_pq_dims);
            let k = resolve_pq_centroids(training_data.len() / self.dim);
            let nbits = k.ilog2() as usize;
            let mut quantizer = ProductQuantizer::new(PQConfig::new(self.dim, m, nbits));
            quantizer.train(training_data.len() / self.dim, training_data)?;
            self.pq_code_size = quantizer.code_size();
            self.pq_encoder = Some(quantizer);
            self.flash_layout.inline_pq_bytes = self.pq_code_size.max(self.config.inline_pq);
            self.flash_layout.node_bytes = self.flash_layout.vector_bytes
                + self.flash_layout.neighbor_bytes
                + self.flash_layout.inline_pq_bytes;
        }
        if self.config.use_hvq {
            let mut hvq = HvqQuantizer::new(
                HvqConfig {
                    dim: self.dim,
                    nbits: self.config.hvq_nbits as u8,
                },
                self.config.random_seed,
            );
            hvq.train(training_data.len() / self.dim, training_data);
            self.hvq_quantizer = Some(hvq);
            self.hvq_codes.clear();
        } else {
            self.hvq_quantizer = None;
            self.hvq_codes.clear();
        }

        self.trained = true;
        Ok(())
    }

    pub fn add(&mut self, vectors: &[f32]) -> Result<()> {
        self.add_with_ids(vectors, None)
    }

    pub fn add_with_ids(&mut self, vectors: &[f32], external_ids: Option<&[i64]>) -> Result<()> {
        self.materialize_storage()?;
        self.validate_vectors(vectors)?;
        if vectors.is_empty() {
            return Ok(());
        }

        if !self.trained {
            self.train(vectors)?;
        }

        let num_vectors = vectors.len() / self.dim;
        if let Some(ids) = external_ids {
            if ids.len() != num_vectors {
                return Err(KnowhereError::InvalidArg(format!(
                    "ids count {} does not match vector count {}",
                    ids.len(),
                    num_vectors
                )));
            }
        }
        self.validate_build_dram_budget(num_vectors)?;
        self.validate_pq_code_budget(num_vectors)?;
        let build_max_degree = self.compute_build_max_degree();
        if self.flat_stride == 0 {
            self.flat_stride = build_max_degree.max(1);
        }
        let stride = self.flat_stride;
        let pq_size = self.pq_code_size.max(1);
        if self.entry_points.is_empty() && !self.node_ids.is_empty() {
            self.entry_points = vec![0];
        }

        // Inline build loop: parallel beam search per batch, sequential node
        // insertion with interleaved link_back. This matches f811cf8/5fc548e's
        // proven build path that achieves 0.952 recall on SIFT-1M.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let batch_size = self.config.build_batch_size.max(1).min(num_vectors);
            let mut row = 0usize;

            while row < num_vectors {
                let batch_end = (row + batch_size).min(num_vectors);
                let graph_size_snapshot = self.node_ids.len();

                let batch_results: Vec<(Vec<u32>, Vec<u8>, i64)> = (row..batch_end)
                    .into_par_iter()
                    .map(|r| {
                        let start = r * self.dim;
                        let vector = &vectors[start..start + self.dim];

                        let neighbors: Vec<u32> = if graph_size_snapshot < 16 {
                            self.select_neighbors(vector, stride)
                        } else {
                            let l = (stride * 3).max(32).min(graph_size_snapshot);
                            let cands =
                                self.vamana_build_search(vector, l, graph_size_snapshot, &[]);
                            cands.iter().take(stride).map(|(id, _)| *id).collect()
                        };

                        let inline_pq = self
                            .pq_encoder
                            .as_ref()
                            .map(|pq| {
                                pq.encode(vector)
                                    .expect("trained PQ quantizer must encode matching vector")
                            })
                            .unwrap_or_default();

                        let ext_id = if let Some(ids) = external_ids {
                            ids[r]
                        } else {
                            (graph_size_snapshot + (r - row)) as i64
                        };
                        (neighbors, inline_pq, ext_id)
                    })
                    .collect();

                for (r, (neighbors, inline_pq, ext_id)) in (row..batch_end).zip(batch_results) {
                    let node_id = self.node_ids.len() as u32;
                    let vector = &vectors[r * self.dim..(r + 1) * self.dim];

                    self.vectors.extend_from_slice(vector);
                    self.node_ids.push(ext_id);

                    let count = neighbors.len().min(stride);
                    self.node_neighbor_counts.push(count as u32);
                    for i in 0..stride {
                        self.node_neighbor_ids
                            .push(if i < count { neighbors[i] } else { 0 });
                    }
                    let code_len = inline_pq.len().min(pq_size);
                    for i in 0..pq_size {
                        self.node_pq_codes
                            .push(if i < code_len { inline_pq[i] } else { 0 });
                    }

                    for &neighbor in &neighbors {
                        self.link_back_with_limit(neighbor, node_id, stride);
                    }
                }

                if self.node_ids.len() % 5000 < batch_size {
                    self.refresh_entry_points();
                }

                row = batch_end;
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            let build_l = if self.config.build_search_list_size > 0 {
                self.config.build_search_list_size
            } else {
                (stride * 3).max(self.config.search_list_size)
            };
            for row in 0..num_vectors {
                let start = row * self.dim;
                let end = start + self.dim;
                let vector = &vectors[start..end];
                let node_id = self.node_ids.len() as u32;
                let neighbors = if self.node_ids.len() < 16 {
                    self.select_neighbors(vector, stride)
                } else {
                    let graph_size = self.node_ids.len();
                    let l = build_l.max(stride * 2).min(graph_size);
                    self.vamana_build_search(vector, l, graph_size, &[])
                        .into_iter()
                        .take(stride)
                        .map(|(id, _)| id)
                        .collect::<Vec<u32>>()
                };
                self.vectors.extend_from_slice(vector);
                let inline_pq = self
                    .pq_encoder
                    .as_ref()
                    .map(|pq| {
                        pq.encode(vector)
                            .expect("trained PQ quantizer must encode matching vector")
                    })
                    .unwrap_or_default();

                let ext_id = external_ids.map(|ids| ids[row]).unwrap_or(node_id as i64);
                self.node_ids.push(ext_id);
                let count = neighbors.len().min(stride);
                self.node_neighbor_counts.push(count as u32);
                for i in 0..stride {
                    self.node_neighbor_ids
                        .push(if i < count { neighbors[i] } else { 0 });
                }
                let code_len = inline_pq.len().min(pq_size);
                for i in 0..pq_size {
                    self.node_pq_codes
                        .push(if i < code_len { inline_pq[i] } else { 0 });
                }

                for neighbor in neighbors {
                    self.link_back_with_limit(neighbor, node_id, stride);
                }
            }
        }
        self.prune_graph_to_target_degree();
        if self.config.run_refine_pass {
            self.refine_flat_graph();
        }

        self.node_count = self.node_ids.len();
        if let Some(hvq) = self.hvq_quantizer.as_ref() {
            let code_size = hvq.code_size_bytes();
            let expected_old = self.node_count.saturating_sub(num_vectors) * code_size;
            if self.hvq_codes.len() == expected_old {
                let new_codes = hvq.encode_batch(num_vectors, vectors, self.config.hvq_nrefine);
                self.hvq_codes.extend_from_slice(&new_codes);
            } else {
                self.hvq_codes =
                    hvq.encode_batch(self.node_count, &self.vectors, self.config.hvq_nrefine);
            }
        } else {
            self.hvq_codes.clear();
        }
        // Pre-refine random connectivity seeding. Batch Vamana can leave late nodes
        // connected mostly to a narrow snapshot of earlier nodes, so add a few
        // random long edges before global refinement to improve graph reachability.
        let n_nodes = self.node_ids.len();
        if self.config.random_init_edges > 0 && n_nodes > 1 {
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};

            let stride = self.flat_stride.max(1);
            let k = self.config.random_init_edges;
            let mut rng = StdRng::seed_from_u64(self.config.random_seed);
            for i in 0..n_nodes {
                let nb_start = i * stride;
                let count = self.node_neighbor_counts[i] as usize;
                if count == 0 {
                    continue;
                }
                // Replace the last min(k, count) neighbors with random long-range edges.
                // This breaks chain-like connectivity from batch Vamana construction.
                let to_replace = k.min(count);
                let replace_start = nb_start + count - to_replace;
                for pos in replace_start..(nb_start + count) {
                    let mut attempts = 0usize;
                    loop {
                        let j = rng.gen_range(0..n_nodes);
                        if j != i {
                            self.node_neighbor_ids[pos] = j as u32;
                            break;
                        }
                        attempts += 1;
                        if attempts > 100 {
                            break;
                        }
                    }
                }
            }
        }
        // Full-graph Vamana refinement for large graphs. Multiple passes
        // progressively improve graph quality (5fc548e: 2 passes → recall 0.952).
        if n_nodes > 100_000 {
            let stride = self.flat_stride.max(1);
            let build_l = (stride * 3).max(32).min(n_nodes);
            let passes = self.config.num_refine_passes;
            for _ in 0..passes {
                self.vamana_refine_pass(n_nodes, build_l, 1.2);
            }
        }
        self.refresh_entry_points();
        if self.config.warm_up {
            self.warm_up_cache();
        }
        if self.config.use_sq8_prefilter && !self.vectors.is_empty() {
            let mut sq = crate::quantization::sq::ScalarQuantizer::new(self.dim, 8);
            sq.train(&self.vectors);
            self.sq8_codes = sq
                .encode_batch(&self.vectors)
                .into_iter()
                .flatten()
                .collect();
            self.sq8_quantizer = Some(sq);
        } else {
            self.sq8_quantizer = None;
            self.sq8_codes.clear();
        }
        Ok(())
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn bootstrap_insert_node(
        &mut self,
        vector: &[f32],
        ext_id: i64,
        stride: usize,
        pq_size: usize,
        build_l: usize,
    ) {
        let node_id = self.node_ids.len() as u32;
        let neighbors = if self.node_ids.len() < 16 {
            self.select_neighbors(vector, stride)
        } else {
            let graph_size = self.node_ids.len();
            let l = build_l.max(stride * 2).min(graph_size);
            self.vamana_build_search(vector, l, graph_size, &[])
                .into_iter()
                .take(stride)
                .map(|(id, _)| id)
                .collect::<Vec<u32>>()
        };
        let inline_pq = self
            .pq_encoder
            .as_ref()
            .map(|pq| {
                pq.encode(vector)
                    .expect("trained PQ quantizer must encode matching vector")
            })
            .unwrap_or_default();

        self.append_node_placeholder(vector, ext_id, &inline_pq, stride, pq_size);
        self.write_neighbor_row(node_id as usize, &neighbors, stride);
        for neighbor in neighbors {
            self.link_back_with_limit(neighbor, node_id, stride);
        }
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn append_node_placeholder(
        &mut self,
        vector: &[f32],
        ext_id: i64,
        inline_pq: &[u8],
        stride: usize,
        pq_size: usize,
    ) {
        self.vectors.extend_from_slice(vector);
        self.node_ids.push(ext_id);
        self.node_neighbor_counts.push(0);
        self.node_neighbor_ids
            .extend(std::iter::repeat_n(0u32, stride));
        let code_len = inline_pq.len().min(pq_size);
        for i in 0..pq_size {
            self.node_pq_codes
                .push(if i < code_len { inline_pq[i] } else { 0 });
        }
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn compute_batch_phase1_results(
        &self,
        batch_vectors: &[f32],
        graph_size_snapshot: usize,
        degree_limit: usize,
        build_l: usize,
    ) -> Vec<(Vec<u32>, Vec<u8>)> {
        if batch_vectors.is_empty() {
            return Vec::new();
        }

        let batch_count = batch_vectors.len() / self.dim;
        let _candidate_cap = degree_limit.saturating_mul(3).max(degree_limit).max(1);

        #[cfg(feature = "parallel")]
        let results: Vec<(Vec<u32>, Vec<u8>)> = {
            use rayon::prelude::*;
            (0..batch_count)
                .into_par_iter()
                .map(|offset| {
                    let vector = &batch_vectors[offset * self.dim..(offset + 1) * self.dim];
                    let scored = if graph_size_snapshot < 16 {
                        self.select_neighbors_scored(vector, graph_size_snapshot)
                    } else {
                        let l = build_l.max(degree_limit * 2).min(graph_size_snapshot);
                        self.vamana_build_search(vector, l, graph_size_snapshot, &[])
                    };
                    // Simple top-k selection (no robust_prune) matches f811cf8
                    // build quality — occlusion pruning at build time hurts initial
                    // graph connectivity, which cascades into poor refine results.
                    let neighbors: Vec<u32> = scored
                        .iter()
                        .take(degree_limit)
                        .map(|(id, _)| *id)
                        .collect();
                    let inline_pq = self
                        .pq_encoder
                        .as_ref()
                        .map(|pq| {
                            pq.encode(vector)
                                .expect("trained PQ quantizer must encode matching vector")
                        })
                        .unwrap_or_default();
                    (neighbors, inline_pq)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let results: Vec<(Vec<u32>, Vec<u8>)> = (0..batch_count)
            .map(|offset| {
                let vector = &batch_vectors[offset * self.dim..(offset + 1) * self.dim];
                let scored = if graph_size_snapshot < 16 {
                    self.select_neighbors_scored(vector, graph_size_snapshot)
                } else {
                    let l = build_l.max(degree_limit * 2).min(graph_size_snapshot);
                    self.vamana_build_search(vector, l, graph_size_snapshot, &[])
                };
                let neighbors: Vec<u32> = scored
                    .iter()
                    .take(degree_limit)
                    .map(|(id, _)| *id)
                    .collect();
                let inline_pq = self
                    .pq_encoder
                    .as_ref()
                    .map(|pq| {
                        pq.encode(vector)
                            .expect("trained PQ quantizer must encode matching vector")
                    })
                    .unwrap_or_default();
                (neighbors, inline_pq)
            })
            .collect();

        results
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn apply_batch_graph_updates(
        &mut self,
        batch_start_node: usize,
        forward_edges: Vec<Vec<u32>>,
        degree_limit: usize,
    ) {
        if forward_edges.is_empty() {
            return;
        }

        let stride = self.flat_stride.max(1);

        for (offset, neighbors) in forward_edges.iter().enumerate() {
            self.write_neighbor_row(batch_start_node + offset, neighbors, stride);
        }

        // Skip reverse-edge pruning at large scale (>100K nodes).
        // Premature reverse-edge pruning evicts good forward edges from existing
        // nodes, fragmenting the graph. The refine pass handles bidirectional
        // connectivity properly. This matches f811cf8's link_back_with_limit guard.
        if self.node_ids.len() > 100_000 {
            return;
        }

        let mut incoming: Vec<Vec<u32>> = vec![Vec::new(); self.node_ids.len()];
        for (offset, neighbors) in forward_edges.iter().enumerate() {
            let src = (batch_start_node + offset) as u32;
            for &dst in neighbors {
                let dst_idx = dst as usize;
                if dst_idx < incoming.len() && dst_idx != src as usize {
                    incoming[dst_idx].push(src);
                }
            }
        }

        #[cfg(feature = "parallel")]
        let updates: Vec<(usize, Vec<u32>)> = {
            use rayon::prelude::*;
            incoming
                .into_par_iter()
                .enumerate()
                .filter_map(|(dst_idx, mut backedges)| {
                    if backedges.is_empty() {
                        return None;
                    }

                    let start = dst_idx * stride;
                    let count = self.node_neighbor_counts[dst_idx] as usize;
                    backedges.extend_from_slice(&self.node_neighbor_ids[start..start + count]);

                    let mut seen = HashSet::with_capacity(backedges.len().saturating_mul(2).max(1));
                    backedges.retain(|&id| id as usize != dst_idx && seen.insert(id));
                    if backedges.is_empty() {
                        return Some((dst_idx, Vec::new()));
                    }

                    let anchor = self.node_vector(dst_idx).to_vec();
                    let mut scored: Vec<(u32, f32)> = backedges
                        .into_iter()
                        .filter(|&id| (id as usize) < self.node_ids.len())
                        .map(|id| {
                            (
                                id,
                                self.exact_distance(&anchor, self.node_vector(id as usize)),
                            )
                        })
                        .collect();
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    scored.truncate(
                        degree_limit
                            .saturating_mul(3)
                            .max(degree_limit)
                            .min(scored.len()),
                    );
                    let selected = self.robust_prune_scored(
                        dst_idx,
                        &scored,
                        degree_limit,
                        ROBUST_PRUNE_ALPHA,
                    );
                    Some((dst_idx, selected))
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let updates: Vec<(usize, Vec<u32>)> = incoming
            .into_iter()
            .enumerate()
            .filter_map(|(dst_idx, mut backedges)| {
                if backedges.is_empty() {
                    return None;
                }

                let start = dst_idx * stride;
                let count = self.node_neighbor_counts[dst_idx] as usize;
                backedges.extend_from_slice(&self.node_neighbor_ids[start..start + count]);

                let mut seen = HashSet::with_capacity(backedges.len().saturating_mul(2).max(1));
                backedges.retain(|&id| id as usize != dst_idx && seen.insert(id));
                if backedges.is_empty() {
                    return Some((dst_idx, Vec::new()));
                }

                let anchor = self.node_vector(dst_idx).to_vec();
                let mut scored: Vec<(u32, f32)> = backedges
                    .into_iter()
                    .filter(|&id| (id as usize) < self.node_ids.len())
                    .map(|id| {
                        (
                            id,
                            self.exact_distance(&anchor, self.node_vector(id as usize)),
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                scored.truncate(
                    degree_limit
                        .saturating_mul(3)
                        .max(degree_limit)
                        .min(scored.len()),
                );
                let selected =
                    self.robust_prune_scored(dst_idx, &scored, degree_limit, ROBUST_PRUNE_ALPHA);
                Some((dst_idx, selected))
            })
            .collect();

        for (dst_idx, neighbors) in updates {
            self.write_neighbor_row(dst_idx, &neighbors, stride);
        }
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn write_neighbor_row(&mut self, node_idx: usize, neighbors: &[u32], stride: usize) {
        let start = node_idx * stride;
        let count = neighbors.len().min(stride);
        self.node_neighbor_counts[node_idx] = count as u32;
        for i in 0..count {
            self.node_neighbor_ids[start + i] = neighbors[i];
        }
        for i in count..stride {
            self.node_neighbor_ids[start + i] = 0;
        }
    }

    #[allow(dead_code)] // Batch-build scaffolding is kept for future bring-up and exercised only in local experiments.
    fn select_neighbors_scored(&self, vector: &[f32], graph_size: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = (0..graph_size)
            .map(|node_id| {
                (
                    node_id as u32,
                    self.exact_distance(vector, self.node_vector(node_id)),
                )
            })
            .collect();
        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored
    }

    fn validate_build_dram_budget(&self, additional_nodes: usize) -> Result<()> {
        if self.config.build_dram_budget_gb <= 0.0 {
            return Ok(());
        }

        let budget_bytes =
            (self.config.build_dram_budget_gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize;
        let total_nodes = self.node_ids.len().saturating_add(additional_nodes);
        let projected_bytes = total_nodes.saturating_mul(self.flash_layout.node_bytes);
        if projected_bytes > budget_bytes {
            return Err(KnowhereError::InvalidArg(format!(
                "disk_build_dram_budget_gb exceeded: projected={} bytes exceeds budget={} bytes",
                projected_bytes, budget_bytes
            )));
        }
        Ok(())
    }

    fn validate_pq_code_budget(&self, additional_nodes: usize) -> Result<()> {
        if self.config.pq_code_budget_gb <= 0.0 || self.pq_code_size == 0 {
            return Ok(());
        }

        let budget_bytes = gb_to_bytes(self.config.pq_code_budget_gb);
        let total_nodes = self.node_ids.len().saturating_add(additional_nodes);
        let projected_bytes = total_nodes.saturating_mul(self.pq_code_size);
        if projected_bytes > budget_bytes {
            return Err(KnowhereError::InvalidArg(format!(
                "disk_pq_code_budget_gb exceeded: projected={} bytes exceeds budget={} bytes",
                projected_bytes, budget_bytes
            )));
        }
        Ok(())
    }

    pub fn export_native_diskann_pq(&self, prefix: &str) -> Result<()> {
        let encoder = self.pq_encoder.as_ref().ok_or_else(|| {
            KnowhereError::InvalidArg("cannot export native PQ: pq_encoder is None".to_string())
        })?;
        if self.node_pq_codes.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "cannot export native PQ: node_pq_codes is empty".to_string(),
            ));
        }

        let n = self.node_ids.len();
        let m = encoder.m();
        let k = encoder.ksub();
        let total_dim = encoder.dim();
        let sub_dim = encoder.sub_dim();
        if k != 256 {
            return Err(KnowhereError::InvalidArg(format!(
                "native DiskANN PQ export requires k=256, got k={k}"
            )));
        }
        if total_dim != self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "encoder dim {} does not match index dim {}",
                total_dim, self.dim
            )));
        }
        if self.pq_code_size != m {
            return Err(KnowhereError::InvalidArg(format!(
                "pq_code_size {} does not match encoder.m {}",
                self.pq_code_size, m
            )));
        }
        if self.node_pq_codes.len() != n.saturating_mul(m) {
            return Err(KnowhereError::InvalidArg(format!(
                "node_pq_codes size mismatch: got {}, expected {}",
                self.node_pq_codes.len(),
                n.saturating_mul(m)
            )));
        }

        let n_i32 = i32::try_from(n)
            .map_err(|_| KnowhereError::InvalidArg(format!("too many points for i32: {n}")))?;
        let dim_i32 = i32::try_from(total_dim).map_err(|_| {
            KnowhereError::InvalidArg(format!("dimension too large for i32: {total_dim}"))
        })?;
        let m_i32 = i32::try_from(m)
            .map_err(|_| KnowhereError::InvalidArg(format!("m too large for i32: {m}")))?;
        let m_plus_one_i32 = i32::try_from(m + 1)
            .map_err(|_| KnowhereError::InvalidArg(format!("m+1 too large for i32: {}", m + 1)))?;

        // 1) {prefix}_pq_pivots.bin : [256 x total_dim] f32
        let pivots_path = format!("{prefix}_pq_pivots.bin");
        let mut pivots = vec![0.0f32; 256 * total_dim];
        for centroid_id in 0..256usize {
            for mi in 0..m {
                for d in 0..sub_dim {
                    let src = mi * 256 * sub_dim + centroid_id * sub_dim + d;
                    let dst = centroid_id * total_dim + mi * sub_dim + d;
                    pivots[dst] = encoder.centroids()[src];
                }
            }
        }
        let mut file = File::create(&pivots_path)?;
        file.write_all(&(256i32).to_le_bytes())?;
        file.write_all(&dim_i32.to_le_bytes())?;
        for v in pivots {
            file.write_all(&v.to_le_bytes())?;
        }

        // 2) {prefix}_pq_pivots.bin_rearrangement_perm.bin : [total_dim x 1] u32 identity
        let perm_path = format!("{prefix}_pq_pivots.bin_rearrangement_perm.bin");
        let mut file = File::create(&perm_path)?;
        file.write_all(&dim_i32.to_le_bytes())?;
        file.write_all(&(1i32).to_le_bytes())?;
        for i in 0..total_dim {
            file.write_all(&(i as u32).to_le_bytes())?;
        }

        // 3) {prefix}_pq_pivots.bin_chunk_offsets.bin : [m+1 x 1] u32 boundaries
        let chunk_offsets_path = format!("{prefix}_pq_pivots.bin_chunk_offsets.bin");
        let mut file = File::create(&chunk_offsets_path)?;
        file.write_all(&m_plus_one_i32.to_le_bytes())?;
        file.write_all(&(1i32).to_le_bytes())?;
        for i in 0..=m {
            file.write_all(&((i * sub_dim) as u32).to_le_bytes())?;
        }

        // 4) {prefix}_pq_pivots.bin_centroid.bin : [total_dim x 1] f32 zeros
        let centroid_path = format!("{prefix}_pq_pivots.bin_centroid.bin");
        let mut file = File::create(&centroid_path)?;
        file.write_all(&dim_i32.to_le_bytes())?;
        file.write_all(&(1i32).to_le_bytes())?;
        for _ in 0..total_dim {
            file.write_all(&0.0f32.to_le_bytes())?;
        }

        // 5) {prefix}_pq_compressed.bin : [N x M] u8 codes
        let compressed_path = format!("{prefix}_pq_compressed.bin");
        let mut file = File::create(&compressed_path)?;
        file.write_all(&n_i32.to_le_bytes())?;
        file.write_all(&m_i32.to_le_bytes())?;
        file.write_all(&self.node_pq_codes)?;

        Ok(())
    }

    pub fn export_native_disk_index(&self, prefix: &str) -> Result<()> {
        const SECTOR_LEN: usize = 4096;

        if self.node_count == 0 {
            return Err(KnowhereError::InvalidArg(
                "cannot export disk.index: node_count is 0".to_string(),
            ));
        }
        if self.vectors.is_empty() || self.node_neighbor_counts.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "cannot export disk.index: vectors/neighbor_counts are empty".to_string(),
            ));
        }

        let n = self.node_count;
        let max_degree = self.config.max_degree.max(1);
        let disk_bytes_per_point = self.dim * std::mem::size_of::<f32>();
        let max_node_len = disk_bytes_per_point + (max_degree + 1) * std::mem::size_of::<u32>();
        let nnodes_per_sector = (SECTOR_LEN / max_node_len).max(1);
        let data_sectors = n.div_ceil(nnodes_per_sector);
        let num_sectors = 1 + data_sectors;
        let expected_file_size = num_sectors * SECTOR_LEN;

        let disk_index_path = format!("{prefix}_disk.index");
        let mut file = File::create(&disk_index_path)?;

        // Sector 0 header.
        let mut sector0 = vec![0u8; SECTOR_LEN];
        let mut cursor = 0usize;
        for value in [
            expected_file_size as u64,
            n as u64,
            self.entry_points.first().copied().unwrap_or(0) as u64,
            max_node_len as u64,
            nnodes_per_sector as u64,
            0u64, // num_frozen_points
            0u64, // file_frozen_id
            0u64, // reorder_data_exists
        ] {
            sector0[cursor..cursor + 8].copy_from_slice(&value.to_le_bytes());
            cursor += 8;
        }
        file.write_all(&sector0)?;

        // Sectors 1+ node payload.
        for sector_id in 0..data_sectors {
            let mut sector = vec![0u8; SECTOR_LEN];
            for slot in 0..nnodes_per_sector {
                let node_id = sector_id * nnodes_per_sector + slot;
                if node_id >= n {
                    break;
                }
                let node_off = slot * max_node_len;
                let vec_start = node_id * self.dim;
                let vec_end = vec_start + self.dim;
                if vec_end > self.vectors.len() {
                    return Err(KnowhereError::InvalidArg(format!(
                        "vector storage too small: need {}, got {}",
                        vec_end,
                        self.vectors.len()
                    )));
                }

                // Vector payload.
                let mut write_off = node_off;
                for &v in &self.vectors[vec_start..vec_end] {
                    sector[write_off..write_off + 4].copy_from_slice(&v.to_le_bytes());
                    write_off += 4;
                }

                // Neighbor count.
                let count = (self.node_neighbor_counts[node_id] as usize).min(max_degree) as u32;
                sector[write_off..write_off + 4].copy_from_slice(&count.to_le_bytes());
                write_off += 4;

                // Neighbor ids (max_degree slots, zero-padded).
                let base = node_id * self.flat_stride;
                for i in 0..max_degree {
                    let neighbor = if i < count as usize && base + i < self.node_neighbor_ids.len()
                    {
                        self.node_neighbor_ids[base + i]
                    } else {
                        0
                    };
                    sector[write_off..write_off + 4].copy_from_slice(&neighbor.to_le_bytes());
                    write_off += 4;
                }
            }
            file.write_all(&sector)?;
        }
        file.flush()?;

        // medoids file: [1i32][1i32][u32 medoid]
        let medoids_path = format!("{prefix}_disk.index_medoids.bin");
        let mut medoids = File::create(&medoids_path)?;
        medoids.write_all(&(1i32).to_le_bytes())?;
        medoids.write_all(&(1i32).to_le_bytes())?;
        medoids.write_all(&(self.entry_points.first().copied().unwrap_or(0)).to_le_bytes())?;
        medoids.flush()?;

        // centroids file: [1i32][dim i32][f32 zeros]
        let centroids_path = format!("{prefix}_disk.index_centroids.bin");
        let mut centroids = File::create(&centroids_path)?;
        centroids.write_all(&(1i32).to_le_bytes())?;
        centroids.write_all(&(self.dim as i32).to_le_bytes())?;
        for _ in 0..self.dim {
            centroids.write_all(&0.0f32.to_le_bytes())?;
        }
        centroids.flush()?;

        Ok(())
    }

    pub fn import_native_disk_index(prefix: &str) -> Result<Self> {
        const SECTOR_LEN: usize = 4096;

        fn read_matrix_header(path: &Path) -> Result<(i32, i32)> {
            let mut file = File::open(path)?;
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            let rows = i32::from_le_bytes(buf[0..4].try_into().unwrap());
            let cols = i32::from_le_bytes(buf[4..8].try_into().unwrap());
            Ok((rows, cols))
        }

        let disk_index_path = PathBuf::from(format!("{prefix}_disk.index"));
        let medoids_path = PathBuf::from(format!("{prefix}_disk.index_medoids.bin"));
        let centroids_path = PathBuf::from(format!("{prefix}_disk.index_centroids.bin"));

        let (centroid_rows, dim_i32) = read_matrix_header(&centroids_path)?;
        if centroid_rows != 1 || dim_i32 <= 0 {
            return Err(KnowhereError::Codec(format!(
                "invalid native centroid header rows={} dim={}",
                centroid_rows, dim_i32
            )));
        }
        let dim = dim_i32 as usize;

        let disk_file = OpenOptions::new().read(true).open(&disk_index_path)?;
        let mmap = unsafe { Mmap::map(&disk_file)? };
        if mmap.len() < SECTOR_LEN {
            return Err(KnowhereError::Codec(format!(
                "native disk.index too small: {} bytes",
                mmap.len()
            )));
        }

        let read_u64 = |offset: usize| -> Result<u64> {
            let bytes = mmap.get(offset..offset + 8).ok_or_else(|| {
                KnowhereError::Codec(format!("truncated disk header at {offset}"))
            })?;
            Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
        };

        let expected_file_size = read_u64(0)? as usize;
        let node_count = read_u64(8)? as usize;
        let medoid_from_header = read_u64(16)? as u32;
        let max_node_len = read_u64(24)? as usize;
        let nnodes_per_sector = read_u64(32)? as usize;

        if expected_file_size != mmap.len() {
            return Err(KnowhereError::Codec(format!(
                "native disk.index size mismatch: header={} actual={}",
                expected_file_size,
                mmap.len()
            )));
        }
        if node_count == 0 || nnodes_per_sector == 0 {
            return Err(KnowhereError::Codec(format!(
                "invalid native disk.index header: node_count={} nnodes_per_sector={}",
                node_count, nnodes_per_sector
            )));
        }
        let vector_bytes = dim
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| KnowhereError::Codec("native vector byte overflow".to_string()))?;
        if max_node_len < vector_bytes + std::mem::size_of::<u32>() {
            return Err(KnowhereError::Codec(format!(
                "invalid native max_node_len {} for dim {}",
                max_node_len, dim
            )));
        }
        let tail_bytes = max_node_len - vector_bytes;
        if tail_bytes % std::mem::size_of::<u32>() != 0 {
            return Err(KnowhereError::Codec(format!(
                "native max_node_len {} leaves misaligned neighbor payload",
                max_node_len
            )));
        }
        let max_degree = tail_bytes / std::mem::size_of::<u32>() - 1;

        let mut medoid = medoid_from_header;
        if medoids_path.exists() {
            let mut file = File::open(&medoids_path)?;
            let mut buf = [0u8; 12];
            file.read_exact(&mut buf)?;
            let rows = i32::from_le_bytes(buf[0..4].try_into().unwrap());
            let cols = i32::from_le_bytes(buf[4..8].try_into().unwrap());
            if rows == 1 && cols == 1 {
                medoid = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            }
        }

        let config = AisaqConfig {
            max_degree: max_degree.max(1),
            search_list_size: max_degree.max(1).saturating_mul(4),
            ..AisaqConfig::default()
        };
        let mut index = Self::new(config, MetricType::L2, dim)?;
        index.vectors = Vec::with_capacity(node_count * dim);
        index.node_ids = (0..node_count as i64).collect();
        let stride = max_degree.max(1);
        index.node_neighbor_counts = vec![0; node_count];
        index.node_neighbor_ids = vec![0; node_count * stride];
        index.flat_stride = stride;
        index.node_count = node_count;
        index.trained = true;
        index.entry_points = vec![medoid.min(node_count.saturating_sub(1) as u32)];

        for node_id in 0..node_count {
            let sector = 1 + node_id / nnodes_per_sector;
            let slot = node_id % nnodes_per_sector;
            let node_offset = sector
                .checked_mul(SECTOR_LEN)
                .and_then(|v| v.checked_add(slot.saturating_mul(max_node_len)))
                .ok_or_else(|| KnowhereError::Codec("native node offset overflow".to_string()))?;
            let node_bytes = mmap
                .get(node_offset..node_offset + max_node_len)
                .ok_or_else(|| {
                    KnowhereError::Codec(format!(
                        "native node {} extends past file bounds at [{}..{})",
                        node_id,
                        node_offset,
                        node_offset + max_node_len
                    ))
                })?;

            let mut cursor = 0usize;
            for _ in 0..dim {
                let raw = node_bytes
                    .get(cursor..cursor + 4)
                    .ok_or_else(|| KnowhereError::Codec("truncated native vector".to_string()))?;
                index
                    .vectors
                    .push(f32::from_le_bytes(raw.try_into().unwrap()));
                cursor += 4;
            }

            let count_raw = u32::from_le_bytes(
                node_bytes[cursor..cursor + 4]
                    .try_into()
                    .map_err(|_| KnowhereError::Codec("truncated native degree".to_string()))?,
            ) as usize;
            cursor += 4;
            let count = count_raw.min(max_degree);
            index.node_neighbor_counts[node_id] = count as u32;
            let row_start = node_id * index.flat_stride;
            for slot in 0..max_degree {
                let neighbor =
                    u32::from_le_bytes(node_bytes[cursor..cursor + 4].try_into().map_err(
                        |_| KnowhereError::Codec("truncated native neighbor".to_string()),
                    )?);
                cursor += 4;
                if slot < count {
                    index.node_neighbor_ids[row_start + slot] =
                        neighbor.min(node_count.saturating_sub(1) as u32);
                }
            }
        }

        Ok(index)
    }

    pub fn load_with_mmap<P: AsRef<Path>>(root: P) -> Result<Self> {
        let file_group = FileGroup::new(root);
        let mut metadata_bytes = Vec::new();
        File::open(file_group.metadata_path())?.read_to_end(&mut metadata_bytes)?;
        let metadata: AisaqMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|error| KnowhereError::Codec(error.to_string()))?;
        if metadata.version != 2 && metadata.version != 3 && metadata.version != 4 {
            return Err(KnowhereError::Codec(format!(
                "unsupported AISAQ metadata version {}",
                metadata.version
            )));
        }

        let mut io_template = BeamSearchIO::new(&metadata.flash_layout, &metadata.config);
        if metadata.config.warm_up {
            for &entry in &metadata.entry_points {
                io_template.cache_node(entry);
                if metadata.pq_code_size > 0 {
                    io_template.cache_pq_vector(entry);
                }
            }
        }

        let data_file = OpenOptions::new().read(true).open(file_group.data_path())?;
        let mmap = unsafe { Mmap::map(&data_file)? };
        let vectors_mmap = if metadata.flash_layout.has_separated_vectors {
            let vectors_file = File::open(file_group.vectors_path())?;
            Some(unsafe { Mmap::map(&vectors_file)? })
        } else {
            None
        };
        let mut index = Self {
            config: metadata.config,
            metric_type: metadata.metric_type,
            dim: metadata.dim,
            flash_layout: metadata.flash_layout,
            vectors: Vec::new(),
            node_ids: Vec::new(),
            node_neighbor_ids: Vec::new(),
            node_neighbor_counts: Vec::new(),
            node_pq_codes: Vec::new(),
            disk_pq_codes: Vec::new(),
            flat_stride: 0,
            pq_encoder: metadata.pq_encoder.map(SerializedPq::into_quantizer),
            pq_code_size: metadata.pq_code_size,
            hvq_quantizer: metadata.hvq_quantizer.map(SerializedHvq::into_quantizer),
            hvq_codes: Vec::new(),
            entry_points: metadata.entry_points,
            io_template,
            trained: metadata.trained,
            node_count: metadata.node_count,
            storage: None,
            mmap_storage: Some(DirectMmapStorage {
                file_group,
                mmap,
                vectors_mmap,
            }),
            loaded_node_cache: None,
            deleted_ids: HashSet::new(),
            sq8_quantizer: None,
            sq8_codes: Vec::new(),
            scratch_pool: Mutex::new(Vec::new()),
        };

        let expected_node_bytes = index
            .node_count
            .saturating_mul(index.flash_layout.node_bytes) as u64;
        let mut data_file = File::open(
            index
                .mmap_storage
                .as_ref()
                .expect("mmap storage set")
                .file_group
                .data_path(),
        )?;
        let file_len = data_file.metadata()?.len();
        let mut cursor = expected_node_bytes;
        if file_len >= expected_node_bytes + 8 {
            data_file.seek(SeekFrom::Start(cursor))?;
            let mut count_buf = [0u8; 8];
            if data_file.read_exact(&mut count_buf).is_ok() {
                let count = u64::from_le_bytes(count_buf) as usize;
                let needed = expected_node_bytes
                    .saturating_add(8)
                    .saturating_add((count as u64).saturating_mul(8));
                if file_len >= needed {
                    for _ in 0..count {
                        let mut id_buf = [0u8; 8];
                        if data_file.read_exact(&mut id_buf).is_err() {
                            break;
                        }
                        let row = u64::from_le_bytes(id_buf) as usize;
                        if row < index.node_count {
                            index.deleted_ids.insert(row);
                        }
                    }
                    cursor = needed;
                }
            }
        }

        index.read_optional_sq8_payload(&mut data_file, &mut cursor, file_len)?;
        index.read_optional_hvq_payload(&mut data_file, &mut cursor, file_len)?;
        if index.pq_code_size > 0 {
            index.disk_pq_codes = index.load_disk_pq_codes()?;
        }
        if index.config.cache_all_on_load {
            index.prime_loaded_node_cache()?;
        }
        if index.config.warm_up {
            index.warm_up_cache();
        }
        Ok(index)
    }

    fn compute_build_max_degree(&self) -> usize {
        let slack = self.config.build_degree_slack_pct.max(100);
        self.config
            .max_degree
            .saturating_mul(slack)
            .div_ceil(100)
            .max(self.config.max_degree)
    }

    fn write_optional_sq8_payload(&self, data_file: &mut File) -> Result<()> {
        let use_sq8 = (!self.sq8_codes.is_empty() && self.sq8_quantizer.is_some()) as u8;
        data_file.write_all(&[use_sq8])?;
        if use_sq8 == 1 {
            data_file.write_all(&(self.sq8_codes.len() as u64).to_le_bytes())?;
            data_file.write_all(&self.sq8_codes)?;
            let sq = self
                .sq8_quantizer
                .as_ref()
                .expect("sq8 quantizer present when use_sq8 is set");
            data_file.write_all(&(sq.dim as u64).to_le_bytes())?;
            data_file.write_all(&(sq.bit as u64).to_le_bytes())?;
            data_file.write_all(&sq.min_val.to_le_bytes())?;
            data_file.write_all(&sq.max_val.to_le_bytes())?;
            data_file.write_all(&sq.scale.to_le_bytes())?;
            data_file.write_all(&sq.offset.to_le_bytes())?;
        }
        Ok(())
    }

    fn write_optional_hvq_payload(&self, data_file: &mut File) -> Result<()> {
        let use_hvq = (!self.hvq_codes.is_empty() && self.hvq_quantizer.is_some()) as u8;
        data_file.write_all(&[use_hvq])?;
        if use_hvq == 1 {
            data_file.write_all(&(self.hvq_codes.len() as u64).to_le_bytes())?;
            data_file.write_all(&self.hvq_codes)?;
        }
        Ok(())
    }

    fn read_optional_sq8_payload(
        &mut self,
        data_file: &mut File,
        cursor: &mut u64,
        file_len: u64,
    ) -> Result<()> {
        if file_len <= *cursor {
            return Ok(());
        }

        data_file.seek(SeekFrom::Start(*cursor))?;
        let mut flag = [0u8; 1];
        if data_file.read_exact(&mut flag).is_err() {
            return Ok(());
        }
        *cursor += 1;

        if flag[0] != 1 {
            return Ok(());
        }

        let mut len_buf = [0u8; 8];
        if data_file.read_exact(&mut len_buf).is_err() {
            return Ok(());
        }
        *cursor += 8;
        let code_len = u64::from_le_bytes(len_buf) as usize;

        let mut codes = vec![0u8; code_len];
        if data_file.read_exact(&mut codes).is_err() {
            return Ok(());
        }
        *cursor += code_len as u64;

        let mut dim_buf = [0u8; 8];
        let mut bit_buf = [0u8; 8];
        let mut min_buf = [0u8; 4];
        let mut max_buf = [0u8; 4];
        let mut scale_buf = [0u8; 4];
        let mut offset_buf = [0u8; 4];
        if data_file.read_exact(&mut dim_buf).is_ok()
            && data_file.read_exact(&mut bit_buf).is_ok()
            && data_file.read_exact(&mut min_buf).is_ok()
            && data_file.read_exact(&mut max_buf).is_ok()
            && data_file.read_exact(&mut scale_buf).is_ok()
            && data_file.read_exact(&mut offset_buf).is_ok()
        {
            *cursor += 8 + 8 + 4 + 4 + 4 + 4;
            let sq_dim = u64::from_le_bytes(dim_buf) as usize;
            let sq_bit = u64::from_le_bytes(bit_buf) as usize;
            let mut sq = crate::quantization::sq::ScalarQuantizer::new(sq_dim, sq_bit);
            sq.min_val = f32::from_le_bytes(min_buf);
            sq.max_val = f32::from_le_bytes(max_buf);
            sq.scale = f32::from_le_bytes(scale_buf);
            sq.offset = f32::from_le_bytes(offset_buf);
            self.sq8_codes = codes;
            self.sq8_quantizer = Some(sq);
        }

        Ok(())
    }

    fn read_optional_hvq_payload(
        &mut self,
        data_file: &mut File,
        cursor: &mut u64,
        file_len: u64,
    ) -> Result<()> {
        if file_len <= *cursor {
            return Ok(());
        }

        data_file.seek(SeekFrom::Start(*cursor))?;
        let mut flag = [0u8; 1];
        if data_file.read_exact(&mut flag).is_err() {
            return Ok(());
        }
        *cursor += 1;

        if flag[0] != 1 {
            return Ok(());
        }

        let mut len_buf = [0u8; 8];
        if data_file.read_exact(&mut len_buf).is_err() {
            return Ok(());
        }
        *cursor += 8;
        let code_len = u64::from_le_bytes(len_buf) as usize;
        let mut codes = vec![0u8; code_len];
        if data_file.read_exact(&mut codes).is_err() {
            return Ok(());
        }
        *cursor += code_len as u64;
        self.hvq_codes = codes;
        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&self, root: P) -> Result<FileGroup> {
        let file_group = FileGroup::create(root)?;
        let metadata = AisaqMetadata {
            version: 4,
            config: self.config.clone(),
            metric_type: self.metric_type,
            dim: self.dim,
            flash_layout: self.flash_layout.clone(),
            pq_code_size: self.pq_code_size,
            entry_points: self.entry_points.clone(),
            trained: self.trained,
            node_count: self.len(),
            pq_encoder: self.pq_encoder.as_ref().map(SerializedPq::from_quantizer),
            hvq_quantizer: self
                .hvq_quantizer
                .as_ref()
                .map(SerializedHvq::from_quantizer),
        };
        let metadata_bytes = bincode::serialize(&metadata)
            .map_err(|error| KnowhereError::Codec(error.to_string()))?;
        let mut metadata_file = File::create(file_group.metadata_path())?;
        metadata_file.write_all(&metadata_bytes)?;
        metadata_file.flush()?;

        let mut data_file = File::create(file_group.data_path())?;
        if let Some(storage) = &self.storage {
            let input = File::open(storage.file_group.data_path())?;
            let node_bytes_len = self.len().saturating_mul(self.flash_layout.node_bytes) as u64;
            std::io::copy(&mut input.take(node_bytes_len), &mut data_file)?;
        } else if let Some(storage) = &self.mmap_storage {
            let input = File::open(storage.file_group.data_path())?;
            let node_bytes_len = self.len().saturating_mul(self.flash_layout.node_bytes) as u64;
            std::io::copy(&mut input.take(node_bytes_len), &mut data_file)?;
        } else {
            for node_id in 0..self.node_ids.len() {
                let bytes = self.serialize_node(node_id as u32);
                data_file.write_all(&bytes)?;
            }
        }
        if self.flash_layout.has_separated_vectors {
            if let Some(storage) = &self.storage {
                fs::copy(storage.file_group.vectors_path(), file_group.vectors_path())?;
            } else if let Some(storage) = &self.mmap_storage {
                fs::copy(storage.file_group.vectors_path(), file_group.vectors_path())?;
            } else {
                let mut vectors_file = File::create(file_group.vectors_path())?;
                for node_id in 0..self.node_ids.len() {
                    for &value in self.node_vector(node_id) {
                        vectors_file.write_all(&value.to_le_bytes())?;
                    }
                }
                vectors_file.flush()?;
            }
        }
        data_file.write_all(&(self.deleted_ids.len() as u64).to_le_bytes())?;
        for &id in &self.deleted_ids {
            data_file.write_all(&(id as u64).to_le_bytes())?;
        }
        self.write_optional_sq8_payload(&mut data_file)?;
        self.write_optional_hvq_payload(&mut data_file)?;
        data_file.flush()?;

        Ok(file_group)
    }

    pub fn load<P: AsRef<Path>>(root: P) -> Result<Self> {
        let file_group = FileGroup::new(root);
        let mut metadata_bytes = Vec::new();
        File::open(file_group.metadata_path())?.read_to_end(&mut metadata_bytes)?;
        let metadata: AisaqMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|error| KnowhereError::Codec(error.to_string()))?;
        if metadata.version != 2 && metadata.version != 3 && metadata.version != 4 {
            return Err(KnowhereError::Codec(format!(
                "unsupported AISAQ metadata version {}",
                metadata.version
            )));
        }

        let mut io_template = BeamSearchIO::new(&metadata.flash_layout, &metadata.config);
        if metadata.config.warm_up {
            for &entry in &metadata.entry_points {
                io_template.cache_node(entry);
                if metadata.pq_code_size > 0 {
                    io_template.cache_pq_vector(entry);
                }
            }
        }

        let page_cache = PageCache::open(
            file_group.data_path(),
            metadata.flash_layout.page_size,
            metadata
                .config
                .pq_read_page_cache_size
                .max(metadata.flash_layout.page_size),
        )?;
        let vectors_mmap = if metadata.flash_layout.has_separated_vectors {
            let vectors_file = File::open(file_group.vectors_path())?;
            Some(unsafe { Mmap::map(&vectors_file)? })
        } else {
            None
        };
        #[cfg(all(feature = "async-io", target_os = "linux"))]
        let raw_file =
            std::sync::Arc::new(OpenOptions::new().read(true).open(file_group.data_path())?);

        let mut index = Self {
            config: metadata.config,
            metric_type: metadata.metric_type,
            dim: metadata.dim,
            flash_layout: metadata.flash_layout,
            vectors: Vec::new(),
            node_ids: Vec::new(),
            node_neighbor_ids: Vec::new(),
            node_neighbor_counts: Vec::new(),
            node_pq_codes: Vec::new(),
            disk_pq_codes: Vec::new(),
            flat_stride: 0,
            pq_encoder: metadata.pq_encoder.map(SerializedPq::into_quantizer),
            pq_code_size: metadata.pq_code_size,
            hvq_quantizer: metadata.hvq_quantizer.map(SerializedHvq::into_quantizer),
            hvq_codes: Vec::new(),
            entry_points: metadata.entry_points,
            io_template,
            trained: metadata.trained,
            node_count: metadata.node_count,
            storage: Some(DiskStorage {
                file_group,
                page_cache,
                vectors_mmap,
                #[cfg(all(feature = "async-io", target_os = "linux"))]
                raw_file,
            }),
            mmap_storage: None,
            loaded_node_cache: None,
            deleted_ids: HashSet::new(),
            sq8_quantizer: None,
            sq8_codes: Vec::new(),
            scratch_pool: Mutex::new(Vec::new()),
        };
        // Optional trailing deleted-id payload: [count:u64][row_id:u64 * count]
        let expected_node_bytes = index
            .node_count
            .saturating_mul(index.flash_layout.node_bytes) as u64;
        let mut data_file = File::open(
            index
                .storage
                .as_ref()
                .expect("storage set")
                .file_group
                .data_path(),
        )?;
        let file_len = data_file.metadata()?.len();
        let mut cursor = expected_node_bytes;
        if file_len >= expected_node_bytes + 8 {
            data_file.seek(SeekFrom::Start(cursor))?;
            let mut count_buf = [0u8; 8];
            if data_file.read_exact(&mut count_buf).is_ok() {
                let count = u64::from_le_bytes(count_buf) as usize;
                let needed = expected_node_bytes
                    .saturating_add(8)
                    .saturating_add((count as u64).saturating_mul(8));
                if file_len >= needed {
                    for _ in 0..count {
                        let mut id_buf = [0u8; 8];
                        if data_file.read_exact(&mut id_buf).is_err() {
                            break;
                        }
                        let row = u64::from_le_bytes(id_buf) as usize;
                        if row < index.node_count {
                            index.deleted_ids.insert(row);
                        }
                    }
                    cursor = needed;
                }
            }
        }

        index.read_optional_sq8_payload(&mut data_file, &mut cursor, file_len)?;
        index.read_optional_hvq_payload(&mut data_file, &mut cursor, file_len)?;
        if index.config.cache_all_on_load {
            index.prime_loaded_node_cache()?;
        }
        if index.pq_code_size > 0 {
            index.disk_pq_codes = index.load_disk_pq_codes()?;
        }
        if index.config.warm_up {
            index.warm_up_cache();
        }
        // NoPQ in-memory mode: convert disk-backed DiskStorage to direct array access.
        // load() always creates storage=Some(DiskStorage) which routes search through
        // PageCache reads. materialize_storage() populates self.vectors/node_neighbor_ids
        // and sets storage=None, enabling the same fast path used after add().
        if index.pq_code_size == 0 {
            index.materialize_storage()?;
        }
        Ok(index)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        self.search_internal(query, k, None)
    }

    pub fn set_search_list_size(&mut self, l: usize) {
        self.config.search_list_size = l.max(1);
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
        self.search_internal(query, k, Some(bitset))
    }

    /// Range search: return all vectors within `radius` distance from `query`.
    /// For L2/Cosine: radius is a distance upper bound.
    /// For IP: radius is the internal distance upper bound (i.e., -similarity).
    pub fn range_search_raw(&self, query: &[f32], radius: f32) -> Result<Vec<(i64, f32)>> {
        // Use a large k so search_internal explores enough candidates.
        // search_list_size * 4 balances quality vs cost; for higher recall increase search_list_size.
        let k = self
            .config
            .search_list_size
            .saturating_mul(4)
            .max(64)
            .min(self.len().max(1));
        let result = self.search_internal(query, k, None)?;
        let pairs: Vec<(i64, f32)> = result
            .ids
            .into_iter()
            .zip(result.distances)
            .filter(|(_, dist)| *dist <= radius)
            .collect();
        Ok(pairs)
    }

    #[cfg(feature = "parallel")]
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Result<SearchResult> {
        use rayon::prelude::*;
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        #[cfg(all(feature = "async-io", target_os = "linux"))]
        if self.storage.is_some() && self.loaded_node_cache.is_none() && n_queries >= 2 {
            return self.search_batch_grouped_uring(queries, k);
        }
        let results: Result<Vec<SearchResult>> = queries
            .par_chunks(self.dim)
            .map(|q| self.search_internal(q, k, None))
            .collect();
        let results = results?;
        let total = n_queries * k;
        let mut ids = Vec::with_capacity(total);
        let mut distances = Vec::with_capacity(total);
        for r in &results {
            let row_len = k.min(r.ids.len()).min(r.distances.len());
            ids.extend_from_slice(&r.ids[..row_len]);
            for _ in row_len..k {
                ids.push(-1);
            }
            distances.extend_from_slice(&r.distances[..row_len]);
            for _ in row_len..k {
                distances.push(f32::MAX);
            }
        }
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    #[cfg(feature = "parallel")]
    pub fn search_batch_with_bitset(
        &self,
        queries: &[f32],
        k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
        use rayon::prelude::*;
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        let results: Result<Vec<SearchResult>> = queries
            .par_chunks(self.dim)
            .map(|q| self.search_internal(q, k, Some(bitset)))
            .collect();
        let results = results?;
        let total = n_queries * k;
        let mut ids = Vec::with_capacity(total);
        let mut distances = Vec::with_capacity(total);
        for r in &results {
            let row_len = k.min(r.ids.len()).min(r.distances.len());
            ids.extend_from_slice(&r.ids[..row_len]);
            for _ in row_len..k {
                ids.push(-1);
            }
            distances.extend_from_slice(&r.distances[..row_len]);
            for _ in row_len..k {
                distances.push(f32::MAX);
            }
        }
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    #[cfg(not(feature = "parallel"))]
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Result<SearchResult> {
        use rayon::prelude::*;
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        #[cfg(all(feature = "async-io", target_os = "linux"))]
        if self.storage.is_some() && self.loaded_node_cache.is_none() && n_queries >= 2 {
            return self.search_batch_grouped_uring(queries, k);
        }
        let results: Result<Vec<SearchResult>> = queries
            .par_chunks(self.dim)
            .map(|q| self.search_internal(q, k, None))
            .collect();
        let results = results?;
        let total = n_queries * k;
        let mut ids = Vec::with_capacity(total);
        let mut distances = Vec::with_capacity(total);
        for r in &results {
            let row_len = k.min(r.ids.len()).min(r.distances.len());
            ids.extend_from_slice(&r.ids[..row_len]);
            for _ in row_len..k {
                ids.push(-1);
            }
            distances.extend_from_slice(&r.distances[..row_len]);
            for _ in row_len..k {
                distances.push(f32::MAX);
            }
        }
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    #[cfg(not(feature = "parallel"))]
    pub fn search_batch_with_bitset(
        &self,
        queries: &[f32],
        k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }
        let total = n_queries * k;
        let mut ids = Vec::with_capacity(total);
        let mut distances = Vec::with_capacity(total);
        for q in queries.chunks(self.dim) {
            let r = self.search_internal(q, k, Some(bitset))?;
            let row_len = k.min(r.ids.len()).min(r.distances.len());
            ids.extend_from_slice(&r.ids[..row_len]);
            for _ in row_len..k {
                ids.push(-1);
            }
            distances.extend_from_slice(&r.distances[..row_len]);
            for _ in row_len..k {
                distances.push(f32::MAX);
            }
        }
        Ok(SearchResult::new(ids, distances, 0.0))
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn prepare_pending_candidates(&self, state: &mut BeamSearchState) {
        state.pending_candidates.clear();

        let max_visit = self.compute_max_visit(state.k);
        if state.scratch.expanded.len() >= max_visit {
            state.done = true;
            return;
        }

        let remaining_budget = max_visit.saturating_sub(state.scratch.expanded.len());
        let batch_cap = self
            .config
            .beam_batch_size
            .max(1)
            .min(16)
            .min(remaining_budget.max(1))
            .min(state.scratch.frontier.len().max(1));

        while state.pending_candidates.len() < batch_cap {
            let Some(candidate) = state.scratch.frontier.pop() else {
                break;
            };
            if state.visited.is_visited(candidate.node_id) {
                continue;
            }
            state.visited.mark(candidate.node_id);
            state.scratch.expanded.push(candidate);
            if self.node_allowed(candidate.node_id, None) {
                state.scratch.accepted.push(candidate);
            }
            state.pending_candidates.push(candidate);
        }

        state.done = state.pending_candidates.is_empty();
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn init_beam_state(&self, query: &[f32], k: usize) -> Result<BeamSearchState> {
        if !self.trained {
            return Err(KnowhereError::IndexNotTrained(
                "AISAQ index is not trained".to_string(),
            ));
        }
        if self.is_empty() {
            return Err(KnowhereError::InternalError(
                "AISAQ index has no vectors".to_string(),
            ));
        }
        if query.len() != self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.dim
            )));
        }

        let mut io = self.io_template.clone();
        io.reset_stats();

        let pq_table = self
            .pq_encoder
            .as_ref()
            .map(|encoder| self.build_pq_table(encoder, query));
        let pq_table_slice = pq_table.as_ref().map(|v| v.as_slice());
        let hvq_state = self.hvq_state_for_query(query);
        let sq8_query = if self.config.use_sq8_prefilter && self.metric_type == MetricType::L2 {
            self.sq8_quantizer
                .as_ref()
                .map(|sq| sq.precompute_query(query))
        } else {
            None
        };

        let max_visit = self.compute_max_visit(k);
        let mut scratch = {
            let mut pool = self.scratch_pool.lock();
            pool.pop()
                .unwrap_or_else(|| AisaqScratch::new(max_visit.saturating_mul(2)))
        };
        scratch.reset();

        for candidate in
            self.rank_entry_candidates(query, pq_table_slice, hvq_state.as_ref(), &mut io)?
        {
            scratch.frontier.push(candidate);
        }

        let mut state = BeamSearchState {
            query: query.to_vec(),
            k,
            pq_table,
            hvq_state,
            sq8_query,
            scratch,
            visited: VisitedList::new(self.node_count),
            io,
            pending_candidates: Vec::new(),
            done: false,
        };
        self.prepare_pending_candidates(&mut state);
        Ok(state)
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn advance_beam_state(
        &self,
        state: &mut BeamSearchState,
        prefetched: &HashMap<u32, LoadedNode>,
    ) -> Result<()> {
        let pq_table_slice = state.pq_table.as_ref().map(|v| v.as_slice());

        for candidate in state.pending_candidates.drain(..) {
            let node = prefetched.get(&candidate.node_id).ok_or_else(|| {
                KnowhereError::InternalError(format!(
                    "missing prefetched node {} in grouped io_uring search",
                    candidate.node_id
                ))
            })?;

            let unseen: Vec<u32> = node
                .neighbors
                .iter()
                .filter(|&&neighbor| !state.visited.is_visited(neighbor))
                .copied()
                .collect();
            let mut neighbor_scores = Vec::with_capacity(unseen.len());
            for &neighbor in &unseen {
                let score = if let Some(neighbor_node) = self.node_ref(neighbor) {
                    self.loaded_node_coarse_distance(
                        neighbor,
                        &state.query,
                        neighbor_node,
                        pq_table_slice,
                        state.hvq_state.as_ref(),
                        state.sq8_query.as_deref(),
                        &mut state.io,
                    )
                } else {
                    self.coarse_distance(
                        neighbor,
                        &state.query,
                        pq_table_slice,
                        state.hvq_state.as_ref(),
                        &mut state.io,
                    )?
                };
                neighbor_scores.push(Candidate {
                    node_id: neighbor,
                    score,
                });
            }

            neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
            let expand_limit = self.compute_expand_limit(&state.io);
            for neighbor in neighbor_scores.into_iter().take(expand_limit) {
                state.scratch.frontier.push(neighbor);
            }
        }

        if let Some(io_limit) = self.config.search_io_limit {
            if state.io.pages_loaded_total() > io_limit {
                state.done = true;
                return Ok(());
            }
        }

        if state.k > 0 && state.scratch.accepted.len() >= state.k {
            if let (Some(frontier_best_dist), Some(result_kth_dist)) = (
                state.scratch.frontier.peek().map(|c| c.score),
                kth_best_score(&state.scratch.accepted, state.k),
            ) {
                if frontier_best_dist >= result_kth_dist * self.config.early_stop_alpha {
                    state.done = true;
                    return Ok(());
                }
            }
        }

        if state.scratch.expanded.len() >= self.compute_max_visit(state.k)
            || state.scratch.frontier.is_empty()
        {
            state.done = true;
            return Ok(());
        }

        self.prepare_pending_candidates(state);
        Ok(())
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn finalize_beam_state(&self, state: &mut BeamSearchState) -> Result<SearchResult> {
        if state.scratch.accepted.len() < state.k {
            self.fill_from_allowed_exact(
                &state.query,
                None,
                |id| state.visited.is_visited(id),
                state.k,
                &mut state.scratch.accepted,
                &mut state.io,
            )?;
        }

        let rerank_pool = self.compute_rerank_pool_size(state.k, state.scratch.accepted.len());
        state
            .scratch
            .accepted
            .sort_by(|left, right| left.score.total_cmp(&right.score));
        let mut scored = Vec::with_capacity(rerank_pool);
        for candidate in state.scratch.accepted.iter().take(rerank_pool) {
            if let Some(node) = self.node_ref(candidate.node_id) {
                let distance = self.exact_distance(&state.query, node.vector);
                scored.push((node.id, distance));
            } else {
                let node =
                    self.load_node(candidate.node_id, NodeAccessMode::None, &mut state.io)?;
                let distance = self.exact_distance_for_node_vector(
                    &state.query,
                    candidate.node_id,
                    &node.vector,
                )?;
                scored.push((node.id, distance));
            }
        }

        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(state.k.min(scored.len()));
        let mut result = SearchResult::new(
            scored.iter().map(|(id, _)| *id).collect(),
            scored.iter().map(|(_, distance)| *distance).collect(),
            0.0,
        );
        result.num_visited = state.io.stats().nodes_visited;
        Ok(result)
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn search_batch_grouped_uring(&self, queries: &[f32], k: usize) -> Result<SearchResult> {
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }

        let group_size = self.config.uring_group_size.max(1);
        let mut all_ids = Vec::with_capacity(n_queries * k);
        let mut all_distances = Vec::with_capacity(n_queries * k);
        let mut total_visited = 0usize;
        let mut total_rounds = 0usize;
        let mut total_sqe = 0usize;
        let mut total_groups = 0usize;

        for group in queries
            .chunks(self.dim)
            .collect::<Vec<_>>()
            .chunks(group_size)
        {
            let mut states = Vec::with_capacity(group.len());
            for query in group {
                match self.init_beam_state(query, k) {
                    Ok(state) => states.push(state),
                    Err(error) => {
                        let mut pool = self.scratch_pool.lock();
                        for state in &mut states {
                            pool.push(std::mem::take(&mut state.scratch));
                        }
                        return Err(error);
                    }
                }
            }

            let group_result = (|| -> Result<()> {
                while states.iter().any(|state| !state.done) {
                    let mut all_node_ids: Vec<u32> = states
                        .iter()
                        .filter(|state| !state.done)
                        .flat_map(|state| {
                            state
                                .pending_candidates
                                .iter()
                                .map(|candidate| candidate.node_id)
                        })
                        .collect();
                    all_node_ids.sort_unstable();
                    all_node_ids.dedup();

                    if all_node_ids.is_empty() {
                        break;
                    }

                    total_rounds += 1;
                    total_sqe += all_node_ids.len();
                    let raw_results = self.read_nodes_batch_ioring(&all_node_ids)?;
                    let mut node_map = HashMap::with_capacity(all_node_ids.len());
                    let mut pages_map = HashMap::with_capacity(all_node_ids.len());
                    for (node_id, (bytes, pages_loaded)) in
                        all_node_ids.iter().copied().zip(raw_results.into_iter())
                    {
                        let node = self.deserialize_node(node_id, &bytes)?;
                        node_map.insert(node_id, node);
                        pages_map.insert(node_id, pages_loaded);
                    }

                    for state in states.iter_mut().filter(|state| !state.done) {
                        for candidate in &state.pending_candidates {
                            if let Some(&pages_loaded) = pages_map.get(&candidate.node_id) {
                                state.io.record_node_access(
                                    candidate.node_id,
                                    self.flash_layout.node_bytes,
                                    pages_loaded,
                                );
                            }
                        }
                        self.advance_beam_state(state, &node_map)?;
                    }
                }

                for state in &mut states {
                    let result = self.finalize_beam_state(state)?;
                    total_visited += result.num_visited;

                    let row_len = k.min(result.ids.len()).min(result.distances.len());
                    all_ids.extend_from_slice(&result.ids[..row_len]);
                    for _ in row_len..k {
                        all_ids.push(-1);
                    }
                    all_distances.extend_from_slice(&result.distances[..row_len]);
                    for _ in row_len..k {
                        all_distances.push(f32::MAX);
                    }
                }

                Ok(())
            })();

            let mut pool = self.scratch_pool.lock();
            for state in &mut states {
                pool.push(std::mem::take(&mut state.scratch));
            }
            group_result?;
            total_groups += 1;
        }

        let avg_sqe_per_round = if total_rounds == 0 {
            0.0
        } else {
            total_sqe as f64 / total_rounds as f64
        };
        let avg_rounds_per_group = if total_groups == 0 {
            0.0
        } else {
            total_rounds as f64 / total_groups as f64
        };
        eprintln!(
            "[grouped_uring] groups={} rounds={} avg_sqe_per_round={:.1} avg_rounds_per_group={:.1}",
            total_groups,
            total_rounds,
            avg_sqe_per_round,
            avg_rounds_per_group
        );

        let mut result = SearchResult::new(all_ids, all_distances, 0.0);
        result.num_visited = total_visited;
        Ok(result)
    }

    fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        bitset: Option<&BitsetView>,
    ) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::IndexNotTrained(
                "AISAQ index is not trained".to_string(),
            ));
        }
        if self.is_empty() {
            return Err(KnowhereError::InternalError(
                "AISAQ index has no vectors".to_string(),
            ));
        }
        if query.len() != self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.dim
            )));
        }

        let mut io = self.io_template.clone();
        io.reset_stats();

        if self.should_force_exact_filter_scan(bitset) {
            return self.exact_scan_allowed_sync(query, k, bitset, &mut io);
        }

        let pq_table = self
            .pq_encoder
            .as_ref()
            .map(|encoder| self.build_pq_table(encoder, query));
        let pq_table_slice = pq_table.as_ref().map(|v| v.as_slice());
        let hvq_state = self.hvq_state_for_query(query);
        let sq8_q = if self.config.use_sq8_prefilter && self.metric_type == MetricType::L2 {
            self.sq8_quantizer
                .as_ref()
                .map(|sq| sq.precompute_query(query))
        } else {
            None
        };

        let max_visit = self.compute_max_visit(k);
        let mut scratch = {
            let mut pool = self.scratch_pool.lock();
            pool.pop()
                .unwrap_or_else(|| AisaqScratch::new(max_visit.saturating_mul(2)))
        };
        scratch.reset();

        let outcome: Result<SearchResult> = with_visited(self.node_count, |visited| {
            (|| {
                let prefetch_enabled = self.storage.is_some();
                let mut next_prefetch = Vec::new();
                let mut io_cutting_best_score = f32::INFINITY;
                let mut io_cutting_stale_rounds = 0usize;
                for candidate in
                    self.rank_entry_candidates(query, pq_table_slice, hvq_state.as_ref(), &mut io)?
                {
                    scratch.frontier.push(candidate);
                }

                while scratch.expanded.len() < max_visit {
                    if let Some(io_limit) = self.config.search_io_limit {
                        if io.pages_loaded_total() > io_limit {
                            break;
                        }
                    }

                    if k > 0 && scratch.accepted.len() >= k {
                        if let (Some(frontier_best_dist), Some(result_kth_dist)) = (
                            scratch.frontier.peek().map(|c| c.score),
                            kth_best_score(&scratch.accepted, k),
                        ) {
                            if frontier_best_dist >= result_kth_dist * self.config.early_stop_alpha
                            {
                                break;
                            }
                        }
                    }

                    let expand_limit = self.compute_expand_limit(&io);
                    let remaining_budget = max_visit.saturating_sub(scratch.expanded.len());
                    let use_sector_batch = self.storage.is_some();
                    #[cfg(feature = "parallel")]
                    let batch_limit = if use_sector_batch && rayon::current_thread_index().is_none()
                    {
                        self.config.beam_batch_size.max(1).min(16)
                    } else {
                        1usize
                    };
                    #[cfg(not(feature = "parallel"))]
                    let batch_limit = 1usize;
                    let batch_cap = batch_limit
                        .min(remaining_budget.max(1))
                        .min(scratch.frontier.len().max(1));

                    let mut batch = Vec::with_capacity(batch_cap);
                    while batch.len() < batch_cap {
                        let Some(candidate) = scratch.frontier.pop() else {
                            break;
                        };
                        if visited.is_visited(candidate.node_id) {
                            continue;
                        }
                        visited.mark(candidate.node_id);
                        scratch.expanded.push(candidate);
                        if self.node_allowed(candidate.node_id, bitset) {
                            scratch.accepted.push(candidate);
                        }
                        batch.push(candidate);
                    }

                    if batch.is_empty() {
                        break;
                    }

                    if prefetch_enabled && !next_prefetch.is_empty() {
                        self.batch_prefetch_neighbors(&next_prefetch);
                        next_prefetch.clear();
                    }

                    let loaded_batch = self.load_node_batch_sync(&batch, &mut io)?;
                    for (_, node) in loaded_batch {
                        let unseen: Vec<u32> = node
                            .neighbors
                            .iter()
                            .filter(|&&neighbor| !visited.is_visited(neighbor))
                            .copied()
                            .collect();
                        let mut neighbor_scores = Vec::with_capacity(unseen.len());
                        for &neighbor in &unseen {
                            let score = if let Some(neighbor_node) = self.node_ref(neighbor) {
                                self.loaded_node_coarse_distance(
                                    neighbor,
                                    query,
                                    neighbor_node,
                                    pq_table_slice,
                                    hvq_state.as_ref(),
                                    sq8_q.as_deref(),
                                    &mut io,
                                )
                            } else {
                                self.coarse_distance(
                                    neighbor,
                                    query,
                                    pq_table_slice,
                                    hvq_state.as_ref(),
                                    &mut io,
                                )?
                            };
                            neighbor_scores.push(Candidate {
                                node_id: neighbor,
                                score,
                            });
                        }
                        neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
                        for neighbor in neighbor_scores.into_iter().take(expand_limit) {
                            if prefetch_enabled {
                                next_prefetch.push(neighbor.node_id);
                            }
                            scratch.frontier.push(neighbor);
                        }
                    }

                    if self.should_cut_io_on_convergence(
                        scratch
                            .accepted
                            .iter()
                            .map(|candidate| candidate.score)
                            .fold(f32::INFINITY, f32::min),
                        &mut io_cutting_best_score,
                        &mut io_cutting_stale_rounds,
                        scratch.accepted.len(),
                        k,
                    ) {
                        break;
                    }
                }

                if scratch.accepted.len() < k {
                    self.fill_from_allowed_exact(
                        query,
                        bitset,
                        |id| visited.is_visited(id),
                        k,
                        &mut scratch.accepted,
                        &mut io,
                    )?;
                }

                let rerank_pool = self.compute_rerank_pool_size(k, scratch.accepted.len());
                scratch
                    .accepted
                    .sort_by(|left, right| left.score.total_cmp(&right.score));
                let mut scored: Vec<(i64, f32)> = scratch
                    .accepted
                    .iter()
                    .take(rerank_pool)
                    .map(|candidate| {
                        if let Some(node) = self.node_ref(candidate.node_id) {
                            let distance = self.exact_distance(query, node.vector);
                            Ok((node.id, distance))
                        } else {
                            let node =
                                self.load_node(candidate.node_id, NodeAccessMode::None, &mut io)?;
                            let distance = self.exact_distance_for_node_vector(
                                query,
                                candidate.node_id,
                                &node.vector,
                            )?;
                            Ok((node.id, distance))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                scored.sort_by(|left, right| left.1.total_cmp(&right.1));
                scored.truncate(k.min(scored.len()));

                let mut result = SearchResult::new(
                    scored.iter().map(|(id, _)| *id).collect(),
                    scored.iter().map(|(_, distance)| *distance).collect(),
                    0.0,
                );
                result.num_visited = io.stats().nodes_visited;
                Ok(result)
            })()
        });

        let mut pool = self.scratch_pool.lock();
        pool.push(scratch);
        outcome
    }

    pub async fn search_async(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        self.search_async_internal(query, k, None).await
    }

    pub async fn search_async_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
        self.search_async_internal(query, k, Some(bitset)).await
    }

    fn load_node_batch_sync(
        &self,
        batch: &[Candidate],
        io: &mut BeamSearchIO,
    ) -> Result<Vec<(Candidate, std::sync::Arc<LoadedNode>)>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(all(feature = "async-io", target_os = "linux"))]
        if self.storage.is_some() && batch.len() > 1 {
            let node_ids: Vec<u32> = batch.iter().map(|candidate| candidate.node_id).collect();
            let raw_results = self.read_nodes_batch_ioring(&node_ids)?;
            let mut out = Vec::with_capacity(batch.len());
            for (i, candidate) in batch.iter().enumerate() {
                let (bytes, pages_loaded) = &raw_results[i];
                let node = self.deserialize_node(candidate.node_id, bytes)?;
                io.record_node_access(
                    candidate.node_id,
                    self.flash_layout.node_bytes,
                    *pages_loaded,
                );
                out.push((*candidate, std::sync::Arc::new(node)));
            }
            return Ok(out);
        }

        #[cfg(feature = "parallel")]
        {
            if self.storage.is_some() && batch.len() > 1 {
                use rayon::prelude::*;

                let loaded: Vec<Result<(Candidate, std::sync::Arc<LoadedNode>, usize)>> = batch
                    .par_iter()
                    .copied()
                    .map(|candidate| {
                        let mut local_io = self.io_template.clone();
                        local_io.reset_stats();
                        let node =
                            self.load_node(candidate.node_id, NodeAccessMode::Node, &mut local_io)?;
                        Ok((candidate, node, local_io.stats().pages_read))
                    })
                    .collect();

                let mut out = Vec::with_capacity(loaded.len());
                for item in loaded {
                    let (candidate, node, pages_read) = item?;
                    io.record_node_access(
                        candidate.node_id,
                        self.flash_layout.node_bytes,
                        pages_read,
                    );
                    out.push((candidate, node));
                }
                return Ok(out);
            }
        }

        let mut out = Vec::with_capacity(batch.len());
        for &candidate in batch {
            let node = self.load_node(candidate.node_id, NodeAccessMode::Node, io)?;
            out.push((candidate, node));
        }
        Ok(out)
    }

    async fn load_node_batch_async(
        &self,
        batch: &[Candidate],
        io: &mut BeamSearchIO,
    ) -> Result<Vec<(Candidate, std::sync::Arc<LoadedNode>)>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        if batch.len() <= 1 {
            let candidate = batch[0];
            let node = self
                .load_node_async(candidate.node_id, NodeAccessMode::Node, io)
                .await?;
            return Ok(vec![(candidate, node)]);
        }

        let loaded = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(batch.len());
            for &candidate in batch {
                handles.push(scope.spawn(move || {
                    let mut local_io = self.io_template.clone();
                    local_io.reset_stats();
                    let node =
                        self.load_node(candidate.node_id, NodeAccessMode::Node, &mut local_io)?;
                    Ok::<_, KnowhereError>((candidate, node, local_io.stats().pages_read))
                }));
            }

            let mut out = Vec::with_capacity(batch.len());
            for handle in handles {
                let joined = handle.join().map_err(|_| {
                    KnowhereError::InternalError("async batch load thread panicked".to_string())
                })?;
                let (candidate, node, pages_read) = joined?;
                io.record_node_access(candidate.node_id, self.flash_layout.node_bytes, pages_read);
                out.push((candidate, node));
            }
            Ok::<_, KnowhereError>(out)
        })?;

        Ok(loaded)
    }

    async fn search_async_internal(
        &self,
        query: &[f32],
        k: usize,
        bitset: Option<&BitsetView>,
    ) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::IndexNotTrained(
                "AISAQ index is not trained".to_string(),
            ));
        }
        if self.is_empty() {
            return Err(KnowhereError::InternalError(
                "AISAQ index has no vectors".to_string(),
            ));
        }
        if query.len() != self.dim {
            return Err(KnowhereError::InvalidArg(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.dim
            )));
        }

        let mut io = self.io_template.clone();
        io.reset_stats();

        if self.should_force_exact_filter_scan(bitset) {
            return self
                .exact_scan_allowed_async(query, k, bitset, &mut io)
                .await;
        }

        let pq_table = self
            .pq_encoder
            .as_ref()
            .map(|encoder| self.build_pq_table(encoder, query));
        let hvq_state = self.hvq_state_for_query(query);

        let mut frontier = BinaryHeap::new();
        let mut expanded = Vec::new();
        let mut accepted = Vec::new();
        reset_thread_local(self.node_count);

        for candidate in self
            .rank_entry_candidates_async(
                query,
                pq_table.as_ref().map(|v| v.as_slice()),
                hvq_state.as_ref(),
                &mut io,
            )
            .await?
        {
            frontier.push(candidate);
        }

        let max_visit = self.compute_max_visit(k);
        let mut io_cutting_best_score = f32::INFINITY;
        let mut io_cutting_stale_rounds = 0usize;
        while expanded.len() < max_visit {
            let remaining_budget = max_visit.saturating_sub(expanded.len());
            let batch_cap = self.config.beamwidth.max(1).min(remaining_budget);
            let mut batch = Vec::with_capacity(batch_cap);
            while batch.len() < batch_cap {
                let Some(candidate) = frontier.pop() else {
                    break;
                };
                if is_visited_thread_local(candidate.node_id) {
                    continue;
                }
                mark_thread_local(candidate.node_id);
                expanded.push(candidate);
                if self.node_allowed(candidate.node_id, bitset) {
                    accepted.push(candidate);
                }
                batch.push(candidate);
            }

            if batch.is_empty() {
                break;
            }

            let loaded_batch = self.load_node_batch_async(&batch, &mut io).await?;
            for (_, node) in loaded_batch {
                for &neighbor in &node.neighbors {
                    if is_visited_thread_local(neighbor) {
                        continue;
                    }
                    self.prefetch_node(neighbor);
                }

                let mut neighbor_scores = Vec::with_capacity(node.neighbors.len());
                for &neighbor in &node.neighbors {
                    if is_visited_thread_local(neighbor) {
                        continue;
                    }
                    let score = self
                        .coarse_distance_async(
                            neighbor,
                            query,
                            pq_table.as_ref().map(|v| v.as_slice()),
                            hvq_state.as_ref(),
                            &mut io,
                        )
                        .await?;
                    neighbor_scores.push(Candidate {
                        node_id: neighbor,
                        score,
                    });
                }

                neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
                let expand_limit = self.compute_expand_limit(&io);
                for neighbor in neighbor_scores.into_iter().take(expand_limit) {
                    frontier.push(neighbor);
                }
            }

            if self.should_cut_io_on_convergence(
                accepted
                    .iter()
                    .map(|candidate| candidate.score)
                    .fold(f32::INFINITY, f32::min),
                &mut io_cutting_best_score,
                &mut io_cutting_stale_rounds,
                accepted.len(),
                k,
            ) {
                break;
            }
        }

        if accepted.len() < k {
            self.fill_from_allowed_exact(
                query,
                bitset,
                is_visited_thread_local,
                k,
                &mut accepted,
                &mut io,
            )?;
        }

        let rerank_pool = self.compute_rerank_pool_size(k, accepted.len());
        accepted.sort_by(|left, right| left.score.total_cmp(&right.score));
        let mut scored = Vec::with_capacity(rerank_pool);
        // TODO: true concurrent IO rerank requires refactoring BeamSearchIO to not require &mut
        for candidate in accepted.iter().take(rerank_pool) {
            let node = self.load_node(candidate.node_id, NodeAccessMode::None, &mut io)?;
            let distance =
                self.exact_distance_for_node_vector(query, candidate.node_id, &node.vector)?;
            scored.push((node.id, distance));
        }

        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(k.min(scored.len()));
        let mut result = SearchResult::new(
            scored.iter().map(|(id, _)| *id).collect(),
            scored.iter().map(|(_, distance)| *distance).collect(),
            0.0,
        );
        result.num_visited = io.stats().nodes_visited;
        Ok(result)
    }

    #[inline]
    fn compute_rerank_pool_size(&self, k: usize, expanded_len: usize) -> usize {
        if expanded_len == 0 {
            return 0;
        }
        if !self.config.rearrange {
            return k.min(expanded_len).max(1);
        }
        let target = k
            .saturating_mul(self.config.rerank_expand_pct)
            .saturating_add(99)
            / 100;
        target.max(k).min(expanded_len).max(1)
    }

    #[inline]
    fn compute_max_visit(&self, k: usize) -> usize {
        let base = self.config.search_list_size.max(k);
        if self.pq_encoder.is_some() {
            let expanded = base
                .saturating_mul(self.config.pq_candidate_expand_pct)
                .saturating_add(99)
                / 100;
            expanded.max(base).min(self.len())
        } else {
            base.min(self.len())
        }
    }

    #[inline]
    fn compute_expand_limit(&self, io: &BeamSearchIO) -> usize {
        let io_limit = io.max_reads_per_iteration();
        if self.config.vectors_beamwidth <= 1 {
            return io_limit;
        }
        io_limit.min(self.config.vectors_beamwidth)
    }

    #[inline]
    fn should_cut_io_on_convergence(
        &self,
        current_best: f32,
        best_score: &mut f32,
        stale_rounds: &mut usize,
        accepted_len: usize,
        k: usize,
    ) -> bool {
        if !self.config.io_cutting_enabled || !current_best.is_finite() {
            return false;
        }

        let improvement = if best_score.is_finite() {
            (*best_score - current_best) / best_score.abs().max(1e-10)
        } else {
            1.0
        };

        if improvement > self.config.io_cutting_threshold {
            *best_score = current_best;
            *stale_rounds = 0;
            false
        } else {
            *stale_rounds += 1;
            *stale_rounds >= self.config.io_cutting_patience && accepted_len >= k
        }
    }

    #[inline]
    fn hvq_state_for_query(&self, query: &[f32]) -> Option<HvqQueryState> {
        if !matches!(self.metric_type, MetricType::Ip | MetricType::Cosine) {
            return None;
        }
        let hvq = self.hvq_quantizer.as_ref()?;
        let q_rot = hvq.rotate_query(query);
        Some(hvq.precompute_query_state(&q_rot))
    }

    #[inline]
    fn hvq_coarse_distance(&self, node_id: u32, state: &HvqQueryState) -> Option<f32> {
        let hvq = self.hvq_quantizer.as_ref()?;
        let code_size = hvq.code_size_bytes();
        let start = node_id as usize * code_size;
        let code = self.hvq_codes.get(start..start + code_size)?;
        let score = hvq.score_code(state, code);
        Some(match self.metric_type {
            MetricType::Ip => -score,
            MetricType::Cosine => 1.0 - score,
            _ => return None,
        })
    }

    #[inline]
    fn loaded_node_coarse_distance(
        &self,
        node_id: u32,
        query: &[f32],
        node: LoadedNodeRef<'_>,
        pq_table: Option<&[f32]>,
        hvq_state: Option<&HvqQueryState>,
        sq8_query: Option<&[i16]>,
        io: &mut BeamSearchIO,
    ) -> f32 {
        if let Some(state) = hvq_state {
            if let Some(score) = self.hvq_coarse_distance(node_id, state) {
                return score;
            }
        }
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            if !node.inline_pq.is_empty() {
                io.record_pq_read(node_id, node.inline_pq.len());
                return encoder.compute_distance_with_table(table, node.inline_pq);
            }
        }
        if let Some(q_i16) = sq8_query {
            if let Some(score) = self.sq8_prefilter_distance(q_i16, node_id as usize) {
                return score;
            }
        }
        self.exact_distance(query, node.vector)
    }

    #[inline]
    fn node_allowed(&self, node_id: u32, bitset: Option<&BitsetView>) -> bool {
        if self.deleted_ids.contains(&(node_id as usize)) {
            return false;
        }
        !bitset.is_some_and(|b| (node_id as usize) < b.len() && b.get(node_id as usize))
    }

    #[inline]
    fn should_force_exact_filter_scan(&self, bitset: Option<&BitsetView>) -> bool {
        let bitset = match bitset {
            Some(bitset) => bitset,
            None => return false,
        };
        if self.config.filter_threshold < 0.0 {
            return false;
        }
        let total = self.len();
        if total == 0 {
            return false;
        }
        let filtered = bitset.count().min(total);
        let allowed = total.saturating_sub(filtered);
        let allowed_ratio = allowed as f32 / total as f32;
        let threshold = self.config.filter_threshold.clamp(0.0, 1.0);
        allowed_ratio <= threshold
    }

    fn exact_scan_allowed_sync(
        &self,
        query: &[f32],
        k: usize,
        bitset: Option<&BitsetView>,
        io: &mut BeamSearchIO,
    ) -> Result<SearchResult> {
        let mut scored = Vec::new();
        for node_id in 0..self.len() as u32 {
            if !self.node_allowed(node_id, bitset) {
                continue;
            }
            let node = self.load_node(node_id, NodeAccessMode::Node, io)?;
            let distance = self.exact_distance_for_node_vector(query, node_id, &node.vector)?;
            scored.push((node.id, distance));
        }
        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(k.min(scored.len()));
        let mut result = SearchResult::new(
            scored.iter().map(|(id, _)| *id).collect(),
            scored.iter().map(|(_, distance)| *distance).collect(),
            0.0,
        );
        result.num_visited = io.stats().nodes_visited;
        Ok(result)
    }

    async fn exact_scan_allowed_async(
        &self,
        query: &[f32],
        k: usize,
        bitset: Option<&BitsetView>,
        io: &mut BeamSearchIO,
    ) -> Result<SearchResult> {
        let mut scored = Vec::new();
        for node_id in 0..self.len() as u32 {
            if !self.node_allowed(node_id, bitset) {
                continue;
            }
            let node = self
                .load_node_async(node_id, NodeAccessMode::Node, io)
                .await?;
            let distance = self.exact_distance_for_node_vector(query, node_id, &node.vector)?;
            scored.push((node.id, distance));
        }
        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(k.min(scored.len()));
        let mut result = SearchResult::new(
            scored.iter().map(|(id, _)| *id).collect(),
            scored.iter().map(|(_, distance)| *distance).collect(),
            0.0,
        );
        result.num_visited = io.stats().nodes_visited;
        Ok(result)
    }

    fn fill_from_allowed_exact<F>(
        &self,
        query: &[f32],
        bitset: Option<&BitsetView>,
        mut seen_contains: F,
        k: usize,
        accepted: &mut Vec<Candidate>,
        io: &mut BeamSearchIO,
    ) -> Result<()>
    where
        F: FnMut(u32) -> bool,
    {
        if accepted.len() >= k {
            return Ok(());
        }

        let mut in_accepted = HashSet::with_capacity(accepted.len() * 2 + 1);
        for c in accepted.iter() {
            in_accepted.insert(c.node_id);
        }

        for node_id in 0..self.len() as u32 {
            if seen_contains(node_id) || in_accepted.contains(&node_id) {
                continue;
            }
            if !self.node_allowed(node_id, bitset) {
                continue;
            }
            let node = self.load_node(node_id, NodeAccessMode::None, io)?;
            let score = self.exact_distance_for_node_vector(query, node_id, &node.vector)?;
            accepted.push(Candidate { node_id, score });
            if accepted.len() >= k {
                break;
            }
        }
        Ok(())
    }

    fn rank_entry_candidates(
        &self,
        query: &[f32],
        pq_table: Option<&[f32]>,
        hvq_state: Option<&HvqQueryState>,
        io: &mut BeamSearchIO,
    ) -> Result<Vec<Candidate>> {
        let mut ranked = Vec::new();
        for &entry in &self.entry_points {
            let score = self.coarse_distance(entry, query, pq_table, hvq_state, io)?;
            ranked.push(Candidate {
                node_id: entry,
                score,
            });
        }
        ranked.sort_by(|a, b| a.score.total_cmp(&b.score));
        Ok(ranked)
    }

    async fn rank_entry_candidates_async(
        &self,
        query: &[f32],
        pq_table: Option<&[f32]>,
        hvq_state: Option<&HvqQueryState>,
        io: &mut BeamSearchIO,
    ) -> Result<Vec<Candidate>> {
        let mut ranked = Vec::new();
        for &entry in &self.entry_points {
            let score = self
                .coarse_distance_async(entry, query, pq_table, hvq_state, io)
                .await?;
            ranked.push(Candidate {
                node_id: entry,
                score,
            });
        }
        ranked.sort_by(|a, b| a.score.total_cmp(&b.score));
        Ok(ranked)
    }

    pub fn scope_audit(&self) -> AisaqScopeAudit {
        AisaqScopeAudit {
            dim: self.dim,
            node_count: self.len(),
            entry_point_count: self.entry_points.len(),
            uses_flash_layout: true,
            uses_beam_search_io: true,
            uses_mmap_backed_pages: self.storage.is_some() || self.mmap_storage.is_some(),
            has_page_cache: self.storage.is_some(),
            native_comparable: false,
            comparability_reason: "PQFlashIndex exposes a real flash-layout/page-cache AISAQ skeleton, but graph construction and IO remain simplified rather than native-comparable DiskANN",
        }
    }

    pub fn flash_layout(&self) -> &FlashLayout {
        &self.flash_layout
    }

    pub fn beam_search_io(&self) -> &BeamSearchIO {
        &self.io_template
    }

    pub fn len(&self) -> usize {
        self.node_count
    }

    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    pub fn file_group(&self) -> Option<&FileGroup> {
        self.storage
            .as_ref()
            .map(|storage| &storage.file_group)
            .or_else(|| {
                self.mmap_storage
                    .as_ref()
                    .map(|storage| &storage.file_group)
            })
    }

    pub fn page_cache(&self) -> Option<&PageCache> {
        self.storage.as_ref().map(|storage| &storage.page_cache)
    }

    pub fn page_cache_stats(&self) -> Option<PageCacheStats> {
        self.page_cache().map(PageCache::stats)
    }

    pub fn async_read_engine(&self) -> AsyncReadEngine {
        AsyncReadEngine::recommended()
    }

    /// Asynchronously load a node from disk (io_uring on Linux, sync fallback otherwise)
    #[cfg(all(feature = "async-io", target_os = "linux"))]
    pub async fn load_node_async(
        &self,
        node_id: u32,
        access_mode: NodeAccessMode,
        io: &mut BeamSearchIO,
    ) -> Result<std::sync::Arc<LoadedNode>> {
        if node_id as usize >= self.len() {
            return Err(KnowhereError::InvalidArg(format!(
                "node_id {} out of bounds for {} nodes",
                node_id,
                self.len()
            )));
        }

        if self.storage.is_none() {
            return self.load_node(node_id, access_mode, io);
        }

        let (bytes, pages_loaded) = self.read_node_bytes_io_uring(node_id)?;
        let loaded = self.deserialize_node(node_id, &bytes)?;
        match access_mode {
            NodeAccessMode::None => {}
            NodeAccessMode::Node => {
                io.record_node_access(node_id, self.flash_layout.node_bytes, pages_loaded)
            }
            NodeAccessMode::Pq => {
                if !loaded.inline_pq.is_empty() {
                    io.record_pq_access(node_id, loaded.inline_pq.len(), pages_loaded);
                }
            }
        }
        Ok(std::sync::Arc::new(loaded))
    }

    #[cfg(not(all(feature = "async-io", target_os = "linux")))]
    pub async fn load_node_async(
        &self,
        node_id: u32,
        access_mode: NodeAccessMode,
        io: &mut BeamSearchIO,
    ) -> Result<std::sync::Arc<LoadedNode>> {
        // Fallback to sync on non-Linux platforms
        self.load_node(node_id, access_mode, io)
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn read_node_bytes_io_uring(&self, node_id: u32) -> Result<(Vec<u8>, usize)> {
        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| KnowhereError::Storage("missing disk storage".to_string()))?;
        let file = &*storage.raw_file;
        let offset = node_id as usize * self.flash_layout.node_bytes;
        let node_bytes = self.flash_layout.node_bytes;
        let mut bytes = vec![0u8; node_bytes];

        let mut ring = IoUring::new(2)
            .map_err(|e| KnowhereError::Storage(format!("io_uring init failed: {e}")))?;
        let entry = opcode::Read::new(
            types::Fd(file.as_raw_fd()),
            bytes.as_mut_ptr(),
            node_bytes as u32,
        )
        .offset(offset as u64)
        .build()
        .user_data(1);

        unsafe {
            ring.submission().push(&entry).map_err(|_| {
                KnowhereError::Storage("io_uring submission queue full".to_string())
            })?;
        }

        ring.submit_and_wait(1)
            .map_err(|e| KnowhereError::Storage(format!("io_uring submit failed: {e}")))?;

        let cqe = ring
            .completion()
            .next()
            .ok_or_else(|| KnowhereError::Storage("io_uring completion missing".to_string()))?;
        let res = cqe.result();
        if res < 0 {
            return Err(KnowhereError::Storage(format!(
                "io_uring read failed errno={}",
                -res
            )));
        }
        if res as usize != node_bytes {
            return Err(KnowhereError::Storage(format!(
                "io_uring short read: got {} expected {}",
                res, node_bytes
            )));
        }

        let pages_loaded = node_bytes.div_ceil(self.flash_layout.page_size.max(1));
        Ok((bytes, pages_loaded))
    }

    #[cfg(all(feature = "async-io", target_os = "linux"))]
    fn read_nodes_batch_ioring(&self, node_ids: &[u32]) -> Result<Vec<(Vec<u8>, usize)>> {
        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| KnowhereError::Storage("missing disk storage".to_string()))?;
        let file = &*storage.raw_file;
        let node_bytes = self.flash_layout.node_bytes;
        let page_size = self.flash_layout.page_size.max(1);
        let n = node_ids.len();

        let ring_cap = n.min(128).max(1) as u32;
        let mut ring = IoUring::new(ring_cap)
            .map_err(|e| KnowhereError::Storage(format!("io_uring batch init failed: {e}")))?;
        let mut buffers: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; node_bytes]).collect();

        let batch = ring_cap as usize;
        for chunk_start in (0..n).step_by(batch) {
            let chunk_end = (chunk_start + batch).min(n);
            let chunk_len = chunk_end - chunk_start;

            for i in 0..chunk_len {
                let nid = node_ids[chunk_start + i];
                let offset = nid as u64 * node_bytes as u64;
                let buf = &mut buffers[chunk_start + i];
                let entry = opcode::Read::new(
                    types::Fd(file.as_raw_fd()),
                    buf.as_mut_ptr(),
                    node_bytes as u32,
                )
                .offset(offset)
                .build()
                .user_data(i as u64);
                unsafe {
                    ring.submission().push(&entry).map_err(|_| {
                        KnowhereError::Storage("io_uring batch SQ full".to_string())
                    })?;
                }
            }

            ring.submit_and_wait(chunk_len).map_err(|e| {
                KnowhereError::Storage(format!("io_uring batch submit failed: {e}"))
            })?;

            for cqe in ring.completion() {
                if cqe.result() < 0 {
                    return Err(KnowhereError::Storage(format!(
                        "io_uring batch read errno={}",
                        -cqe.result()
                    )));
                }
                if cqe.result() as usize != node_bytes {
                    return Err(KnowhereError::Storage(format!(
                        "io_uring batch short read: got {} expected {}",
                        cqe.result(),
                        node_bytes
                    )));
                }
            }
        }

        let pages_per_node = node_bytes.div_ceil(page_size);
        Ok(buffers
            .into_iter()
            .map(|buf| (buf, pages_per_node))
            .collect())
    }

    fn validate_vectors(&self, vectors: &[f32]) -> Result<()> {
        if vectors.len() % self.dim != 0 {
            return Err(KnowhereError::InvalidArg(format!(
                "input length {} is not divisible by dimension {}",
                vectors.len(),
                self.dim
            )));
        }
        Ok(())
    }

    fn coarse_distance(
        &self,
        node_id: u32,
        query: &[f32],
        pq_table: Option<&[f32]>,
        hvq_state: Option<&HvqQueryState>,
        io: &mut BeamSearchIO,
    ) -> Result<f32> {
        if let Some(state) = hvq_state {
            if let Some(score) = self.hvq_coarse_distance(node_id, state) {
                return Ok(score);
            }
        }
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            let pq_size = self.pq_code_size.max(1);
            let start = node_id as usize * pq_size;
            if start + pq_size <= self.disk_pq_codes.len() {
                io.record_pq_access(node_id, pq_size, 0);
                return Ok(encoder.compute_distance_with_table(
                    table,
                    &self.disk_pq_codes[start..start + pq_size],
                ));
            }
        }
        let node = self.load_node(node_id, NodeAccessMode::Pq, io)?;
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            let pq = &node.inline_pq;
            if !pq.is_empty() {
                return Ok(encoder.compute_distance_with_table(table, pq));
            }
        }

        self.exact_distance_for_node_vector(query, node_id, &node.vector)
    }

    #[inline]
    fn prefetch_node(&self, node_id: u32) {
        if node_id as usize >= self.len() {
            return;
        }
        if let Some(storage) = &self.storage {
            let offset = node_id as usize * self.flash_layout.node_bytes;
            let _ = storage
                .page_cache
                .prefetch(offset, self.flash_layout.node_bytes);
        }
    }

    /// Prefetch upcoming sectors for the next beam-search round.
    /// On the sync/PageCache path this warms the page cache; on direct mmap it is a no-op.
    #[cfg(feature = "async-io")]
    pub async fn prefetch_sectors(&self, node_ids: &[u32]) -> Result<()> {
        for &node_id in node_ids {
            self.prefetch_node(node_id);
        }
        Ok(())
    }

    #[cfg(not(feature = "async-io"))]
    pub async fn prefetch_sectors(&self, node_ids: &[u32]) -> Result<()> {
        let _ = node_ids;
        Ok(())
    }

    fn batch_prefetch_neighbors(&self, neighbor_ids: &[u32]) {
        let Some(storage) = &self.storage else {
            return;
        };
        if neighbor_ids.is_empty() {
            return;
        }

        let node_bytes = self.flash_layout.node_bytes;
        let page_size = storage.page_cache.page_size.max(1);
        let mmap_len = storage.page_cache.mmap.len();
        let mut page_ids: Vec<usize> = Vec::with_capacity(neighbor_ids.len().saturating_mul(2));

        for &node_id in neighbor_ids {
            if node_id as usize >= self.len() {
                continue;
            }
            let offset = node_id as usize * node_bytes;
            if offset >= mmap_len || node_bytes == 0 {
                continue;
            }
            let end_offset = offset
                .saturating_add(node_bytes.saturating_sub(1))
                .min(mmap_len.saturating_sub(1));
            let start_page = offset / page_size;
            let end_page = end_offset / page_size;
            for page_id in start_page..=end_page {
                page_ids.push(page_id);
            }
        }

        page_ids.sort_unstable();
        page_ids.dedup();

        for page_id in page_ids {
            let offset = page_id * page_size;
            if offset >= mmap_len {
                continue;
            }
            let len = page_size.min(mmap_len.saturating_sub(offset));
            if len > 0 {
                let _ = storage.page_cache.prefetch(offset, len);
            }
        }
    }

    async fn coarse_distance_async(
        &self,
        node_id: u32,
        query: &[f32],
        pq_table: Option<&[f32]>,
        hvq_state: Option<&HvqQueryState>,
        io: &mut BeamSearchIO,
    ) -> Result<f32> {
        if let Some(state) = hvq_state {
            if let Some(score) = self.hvq_coarse_distance(node_id, state) {
                return Ok(score);
            }
        }
        let node = self
            .load_node_async(node_id, NodeAccessMode::Pq, io)
            .await?;
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            let pq = &node.inline_pq;
            if !pq.is_empty() {
                return Ok(encoder.compute_distance_with_table(table, pq));
            }
        }
        self.exact_distance_for_node_vector(query, node_id, &node.vector)
    }

    fn exact_distance(&self, query: &[f32], vector: &[f32]) -> f32 {
        match self.metric_type {
            MetricType::L2 => simd::l2_distance(query, vector),
            MetricType::Ip => -dot(query, vector),
            MetricType::Cosine => 1.0 - cosine_similarity(query, vector),
            MetricType::Hamming => f32::MAX,
        }
    }

    /// Build PQ distance table with metric-aware dispatch
    fn build_pq_table(&self, encoder: &ProductQuantizer, query: &[f32]) -> Vec<f32> {
        match self.metric_type {
            MetricType::Ip | MetricType::Cosine => encoder.build_distance_table_ip(query),
            _ => encoder.build_distance_table_l2(query),
        }
    }

    fn separated_vectors_mmap(&self) -> Option<&Mmap> {
        self.storage
            .as_ref()
            .and_then(|storage| storage.vectors_mmap.as_ref())
            .or_else(|| {
                self.mmap_storage
                    .as_ref()
                    .and_then(|storage| storage.vectors_mmap.as_ref())
            })
    }

    fn load_separated_vector(&self, node_id: u32) -> Result<Vec<f32>> {
        let mmap = self.separated_vectors_mmap().ok_or_else(|| {
            KnowhereError::Codec("no vectors_mmap in separated layout".to_string())
        })?;
        let vector_bytes = self.flash_layout.vector_bytes;
        let start = node_id as usize * vector_bytes;
        let end = start + vector_bytes;
        if end > mmap.len() {
            return Err(KnowhereError::Codec(
                "vectors_mmap out of range".to_string(),
            ));
        }
        Ok(mmap[start..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn exact_distance_for_node_vector(
        &self,
        query: &[f32],
        node_id: u32,
        vector: &[f32],
    ) -> Result<f32> {
        if !vector.is_empty() || !self.flash_layout.has_separated_vectors {
            return Ok(self.exact_distance(query, vector));
        }
        let vector = self.load_separated_vector(node_id)?;
        Ok(self.exact_distance(query, &vector))
    }

    #[inline]
    fn sq8_prefilter_distance(&self, q_i16: &[i16], node_idx: usize) -> Option<f32> {
        let sq = self.sq8_quantizer.as_ref()?;
        let start = node_idx.checked_mul(self.dim)?;
        let end = start.checked_add(self.dim)?;
        let code = self.sq8_codes.get(start..end)?;
        Some(sq.sq_l2_precomputed(q_i16, code))
    }

    /// Build-time beam search over currently built prefix graph.
    /// Returns up to `l` candidates sorted by distance ascending.
    fn vamana_build_search(
        &self,
        query: &[f32],
        l: usize,
        graph_size: usize,
        extra_seeds: &[u32],
    ) -> Vec<(u32, f32)> {
        if graph_size == 0 || l == 0 {
            return Vec::new();
        }
        // Build PQ distance table if available (ADC lookup — 4-8x faster than exact distance)
        let pq_table: Option<Vec<f32>> = if self.pq_code_size > 0 && !self.node_pq_codes.is_empty()
        {
            self.pq_encoder
                .as_ref()
                .map(|pq| self.build_pq_table(pq, query))
        } else {
            None
        };
        let hvq_state = self.hvq_state_for_query(query);
        let stride = self.flat_stride.max(1);
        let start_node = self.entry_points.first().copied().unwrap_or(0) as usize;
        let start_node = start_node.min(graph_size - 1);
        let score_node = |node_idx: usize| -> f32 {
            if let Some(ref state) = hvq_state {
                if let Some(score) = self.hvq_coarse_distance(node_idx as u32, state) {
                    return score;
                }
            }
            if let Some(ref table) = pq_table {
                let code_start = node_idx * self.pq_code_size;
                let code_end = code_start + self.pq_code_size;
                if code_end <= self.node_pq_codes.len() {
                    if let Some(ref pq) = self.pq_encoder {
                        return pq.compute_distance_with_table(
                            table,
                            &self.node_pq_codes[code_start..code_end],
                        );
                    }
                }
            }
            self.exact_distance(query, self.node_vector(node_idx))
        };
        let start_dist = score_node(start_node);

        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::with_capacity(l.saturating_mul(2));
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(l);

        frontier.push(Candidate {
            node_id: start_node as u32,
            score: start_dist,
        });
        visited.insert(start_node as u32);

        for &seed in extra_seeds {
            let seed_idx = seed as usize;
            if seed_idx >= graph_size || !visited.insert(seed) {
                continue;
            }
            frontier.push(Candidate {
                node_id: seed,
                score: score_node(seed_idx),
            });
        }

        while let Some(candidate) = frontier.pop() {
            best.push((candidate.node_id, candidate.score));
            if best.len() >= l {
                break;
            }
            let node = candidate.node_id as usize;
            let nb_start = node * stride;
            let count = self.node_neighbor_counts.get(node).copied().unwrap_or(0) as usize;
            for k in 0..count {
                let nb = self.node_neighbor_ids[nb_start + k];
                if nb as usize >= graph_size || !visited.insert(nb) {
                    continue;
                }
                frontier.push(Candidate {
                    node_id: nb,
                    score: score_node(nb as usize),
                });
            }
        }

        best.sort_by(|a, b| a.1.total_cmp(&b.1));
        best.truncate(l);
        best
    }

    fn select_neighbors(&self, vector: &[f32], max_degree: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = (0..self.node_ids.len())
            .map(|node_id| {
                (
                    node_id as u32,
                    self.exact_distance(vector, self.node_vector(node_id)),
                )
            })
            .collect();

        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(max_degree.min(scored.len()));
        scored.into_iter().map(|(node_id, _)| node_id).collect()
    }

    #[cfg(test)]
    fn link_back(&mut self, neighbor: u32, node_id: u32) {
        self.link_back_with_limit(neighbor, node_id, self.config.max_degree);
    }

    fn candidate_pair_distance(&self, a: u32, b: u32) -> Option<f32> {
        let a_idx = a as usize;
        let b_idx = b as usize;
        if a_idx >= self.node_ids.len() || b_idx >= self.node_ids.len() {
            return None;
        }
        Some(self.exact_distance(self.node_vector(a_idx), self.node_vector(b_idx)))
    }

    fn run_robust_prune_pass(
        &self,
        pool: &[(u32, f32)],
        selected: &mut Vec<u32>,
        selected_ids: &mut HashSet<u32>,
        current_alpha: f32,
        degree_limit: usize,
    ) {
        for &(candidate_id, anchor_distance) in pool {
            if selected.len() >= degree_limit {
                break;
            }
            if selected_ids.contains(&candidate_id) {
                continue;
            }

            let mut occluded = false;
            for &selected_id in selected.iter() {
                let Some(d_jk) = self.candidate_pair_distance(selected_id, candidate_id) else {
                    continue;
                };
                if d_jk <= f32::EPSILON || anchor_distance / d_jk > current_alpha {
                    occluded = true;
                    break;
                }
            }

            if !occluded {
                selected.push(candidate_id);
                selected_ids.insert(candidate_id);
            }
        }
    }

    fn robust_prune_scored(
        &self,
        anchor_idx: usize,
        pool: &[(u32, f32)],
        degree_limit: usize,
        alpha: f32,
    ) -> Vec<u32> {
        if pool.is_empty() || degree_limit == 0 {
            return Vec::new();
        }

        let anchor_id = anchor_idx as u32;
        let mut dedup = HashSet::with_capacity(pool.len().saturating_mul(2));
        let filtered: Vec<(u32, f32)> = pool
            .iter()
            .copied()
            .filter(|&(candidate_id, _)| candidate_id != anchor_id)
            .filter(|&(candidate_id, _)| (candidate_id as usize) < self.node_ids.len())
            .filter(|&(candidate_id, _)| dedup.insert(candidate_id))
            .collect();

        if filtered.len() <= degree_limit {
            return filtered
                .into_iter()
                .map(|(candidate_id, _)| candidate_id)
                .collect();
        }

        let final_alpha = alpha.max(1.0);
        let mut selected = Vec::with_capacity(degree_limit.min(filtered.len()));
        let mut selected_ids = HashSet::with_capacity(degree_limit.saturating_mul(2));

        self.run_robust_prune_pass(
            &filtered,
            &mut selected,
            &mut selected_ids,
            1.0,
            degree_limit,
        );
        if selected.len() < degree_limit && final_alpha > 1.0 {
            self.run_robust_prune_pass(
                &filtered,
                &mut selected,
                &mut selected_ids,
                final_alpha,
                degree_limit,
            );
        }

        if final_alpha > 1.0 {
            for &(candidate_id, _) in &filtered {
                if selected.len() >= degree_limit {
                    break;
                }
                if selected_ids.insert(candidate_id) {
                    selected.push(candidate_id);
                }
            }
        }

        selected
    }

    fn link_back_with_limit(&mut self, neighbor: u32, node_id: u32, degree_limit: usize) {
        let neighbor_idx = neighbor as usize;
        if neighbor_idx >= self.node_ids.len() || node_id == neighbor {
            return;
        }
        if self.node_ids.len() > 100_000 {
            // Guard: removing this causes -41% QPS and 9x build time regression
            // at 1M scale. The vamana_refine_pass() handles full-graph reverse-edge
            // recovery beyond this size.
            return;
        }
        let stride = self.flat_stride.max(1);
        let start = neighbor_idx * stride;
        let count = self.node_neighbor_counts[neighbor_idx] as usize;
        let effective_limit = stride.min(degree_limit.max(1));
        let current = &self.node_neighbor_ids[start..start + count];
        if current.contains(&node_id) {
            return;
        }
        if count < effective_limit {
            self.node_neighbor_ids[start + count] = node_id;
            self.node_neighbor_counts[neighbor_idx] += 1;
            return;
        }
        let mut neighbor_list: Vec<u32> = current.to_vec();
        neighbor_list.push(node_id);
        let anchor = self.node_vector(neighbor_idx).to_vec();
        let mut scored: Vec<(u32, f32)> = neighbor_list
            .iter()
            .map(|&nb| {
                (
                    nb,
                    self.exact_distance(&anchor, self.node_vector(nb as usize)),
                )
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let selected = self.robust_prune_scored(neighbor_idx, &scored, effective_limit, 1.0);
        let new_count = selected.len();
        for (i, nb) in selected.into_iter().enumerate() {
            self.node_neighbor_ids[start + i] = nb;
        }
        for i in new_count..stride {
            self.node_neighbor_ids[start + i] = 0;
        }
        self.node_neighbor_counts[neighbor_idx] = new_count as u32;
    }

    #[allow(dead_code)]
    fn link_back_no_guard(&mut self, neighbor: u32, node_id: u32, degree_limit: usize) {
        let neighbor_idx = neighbor as usize;
        if neighbor_idx >= self.node_ids.len() || node_id == neighbor {
            return;
        }
        let stride = self.flat_stride.max(1);
        let start = neighbor_idx * stride;
        let count = self.node_neighbor_counts[neighbor_idx] as usize;
        let effective_limit = stride.min(degree_limit.max(1));
        let current = &self.node_neighbor_ids[start..start + count];
        if current.contains(&node_id) {
            return;
        }
        if count < effective_limit {
            self.node_neighbor_ids[start + count] = node_id;
            self.node_neighbor_counts[neighbor_idx] += 1;
            return;
        }
        let mut neighbor_list: Vec<u32> = current.to_vec();
        neighbor_list.push(node_id);
        let anchor = self.node_vector(neighbor_idx).to_vec();
        let mut scored: Vec<(u32, f32)> = neighbor_list
            .iter()
            .map(|&nb| {
                (
                    nb,
                    self.exact_distance(&anchor, self.node_vector(nb as usize)),
                )
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let selected =
            self.robust_prune_scored(neighbor_idx, &scored, effective_limit, ROBUST_PRUNE_ALPHA);
        let new_count = selected.len();
        for (i, nb) in selected.into_iter().enumerate() {
            self.node_neighbor_ids[start + i] = nb;
        }
        for i in new_count..stride {
            self.node_neighbor_ids[start + i] = 0;
        }
        self.node_neighbor_counts[neighbor_idx] = new_count as u32;
    }

    fn vamana_refine_pass(&mut self, n_nodes: usize, build_l: usize, alpha: f32) {
        let stride = self.flat_stride.max(1);
        let target_degree = self.config.max_degree.min(stride).max(1);
        let refine_start = std::time::Instant::now();

        // Phase 1: parallel beam search + prune (reads from same graph snapshot)
        #[cfg(feature = "parallel")]
        let mut rows: Vec<Vec<u32>> = {
            use rayon::prelude::*;
            (0..n_nodes)
                .into_par_iter()
                .map(|i| {
                    let vector_i = self.node_vector(i).to_vec();
                    let cands = self.vamana_build_search(&vector_i, build_l, n_nodes, &[]);
                    self.robust_prune_scored(i, &cands, stride, alpha)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let mut rows: Vec<Vec<u32>> = (0..n_nodes)
            .map(|i| {
                let vector_i = self.node_vector(i).to_vec();
                let cands = self.vamana_build_search(&vector_i, build_l, n_nodes, &[]);
                self.robust_prune_scored(i, &cands, stride, alpha)
            })
            .collect();

        // Phase 2: native-like batch reverse-edge handling. Aggregate reverse
        // edges per target node, append lazily, and only prune once for rows
        // that overflow physical storage.
        let mut reverse_incoming: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for (src_idx, neighbors) in rows.iter().enumerate() {
            let src = src_idx as u32;
            for &dst in neighbors {
                let dst_idx = dst as usize;
                if dst_idx < n_nodes && dst_idx != src_idx {
                    reverse_incoming[dst_idx].push(src);
                }
            }
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            rows.par_iter_mut().enumerate().for_each(|(node_idx, row)| {
                let incoming = &reverse_incoming[node_idx];
                if incoming.is_empty() {
                    return;
                }

                let mut merged = Vec::with_capacity(row.len() + incoming.len());
                let mut seen =
                    HashSet::with_capacity((row.len() + incoming.len()).saturating_mul(2).max(1));

                for &neighbor in row.iter() {
                    let neighbor_idx = neighbor as usize;
                    if neighbor_idx < n_nodes && neighbor_idx != node_idx && seen.insert(neighbor) {
                        merged.push(neighbor);
                    }
                }
                for &src in incoming.iter() {
                    let src_idx = src as usize;
                    if src_idx < n_nodes && src_idx != node_idx && seen.insert(src) {
                        merged.push(src);
                    }
                }

                if merged.len() <= stride {
                    *row = merged;
                    return;
                }

                let anchor = self.node_vector(node_idx).to_vec();
                let mut scored: Vec<(u32, f32)> = merged
                    .into_iter()
                    .map(|id| {
                        (
                            id,
                            self.exact_distance(&anchor, self.node_vector(id as usize)),
                        )
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                *row =
                    self.robust_prune_scored(node_idx, &scored, target_degree, ROBUST_PRUNE_ALPHA);
            });

            let counts = &mut self.node_neighbor_counts[..n_nodes];
            let neighbor_ids = &mut self.node_neighbor_ids[..n_nodes * stride];
            neighbor_ids
                .par_chunks_mut(stride)
                .zip(counts.par_iter_mut())
                .zip(rows.par_iter())
                .for_each(|((row_slots, count_slot), row)| {
                    let count = row.len().min(stride);
                    *count_slot = count as u32;
                    row_slots[..count].copy_from_slice(&row[..count]);
                    row_slots[count..].fill(0);
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (node_idx, row) in rows.iter_mut().enumerate() {
                let incoming = &reverse_incoming[node_idx];
                if !incoming.is_empty() {
                    let mut merged = Vec::with_capacity(row.len() + incoming.len());
                    let mut seen = HashSet::with_capacity(
                        (row.len() + incoming.len()).saturating_mul(2).max(1),
                    );

                    for &neighbor in row.iter() {
                        let neighbor_idx = neighbor as usize;
                        if neighbor_idx < n_nodes
                            && neighbor_idx != node_idx
                            && seen.insert(neighbor)
                        {
                            merged.push(neighbor);
                        }
                    }
                    for &src in incoming.iter() {
                        let src_idx = src as usize;
                        if src_idx < n_nodes && src_idx != node_idx && seen.insert(src) {
                            merged.push(src);
                        }
                    }

                    if merged.len() <= stride {
                        *row = merged;
                    } else {
                        let anchor = self.node_vector(node_idx).to_vec();
                        let mut scored: Vec<(u32, f32)> = merged
                            .into_iter()
                            .map(|id| {
                                (
                                    id,
                                    self.exact_distance(&anchor, self.node_vector(id as usize)),
                                )
                            })
                            .collect();
                        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                        *row = self.robust_prune_scored(
                            node_idx,
                            &scored,
                            target_degree,
                            ROBUST_PRUNE_ALPHA,
                        );
                    }
                }

                self.write_neighbor_row(node_idx, row, stride);
            }
        }

        self.prune_graph_to_target_degree();
        let _ = refine_start;
    }

    fn prune_graph_to_target_degree(&mut self) {
        let target_degree = self.config.max_degree;
        let stride = self.flat_stride.max(1);
        for idx in 0..self.node_ids.len() {
            let count = self.node_neighbor_counts[idx] as usize;
            if count <= target_degree {
                continue;
            }
            self.node_neighbor_counts[idx] = target_degree as u32;
            let start = idx * stride;
            for i in target_degree..count {
                self.node_neighbor_ids[start + i] = 0;
            }
        }
    }

    fn refine_flat_graph(&mut self) {
        let n = self.node_ids.len();
        if n == 0 || n > 50_000 {
            return;
        }
        let stride = self.flat_stride.max(1);
        let target = self.config.max_degree.min(stride).max(1);
        let mut new_neighbor_ids = vec![0u32; n * stride];
        let mut new_neighbor_counts = vec![0u32; n];

        for i in 0..n {
            let vec_i = self.node_vector(i).to_vec();
            let mut scored: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, self.exact_distance(&vec_i, self.node_vector(j))))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(target);
            let nb_start = i * stride;
            let count = scored.len();
            for (k, (nb, _)) in scored.into_iter().enumerate() {
                new_neighbor_ids[nb_start + k] = nb;
            }
            new_neighbor_counts[i] = count as u32;
        }

        self.node_neighbor_ids = new_neighbor_ids;
        self.node_neighbor_counts = new_neighbor_counts;
    }

    fn refresh_entry_points(&mut self) {
        let n = self.len();
        if n == 0 {
            self.entry_points.clear();
            return;
        }

        let count = self.config.num_entry_points.min(n).max(1);
        if self.vectors.is_empty() || self.node_ids.is_empty() {
            self.entry_points = (0..count as u32).collect();
            return;
        }

        let mut centroid = vec![0.0f32; self.dim];
        for i in 0..self.node_ids.len() {
            let vec = self.node_vector(i);
            for d in 0..self.dim {
                centroid[d] += vec[d];
            }
        }
        let inv_n = 1.0f32 / n as f32;
        for v in &mut centroid {
            *v *= inv_n;
        }

        let first = (0..n)
            .min_by(|&a, &b| {
                self.exact_distance(&centroid, self.node_vector(a))
                    .total_cmp(&self.exact_distance(&centroid, self.node_vector(b)))
            })
            .unwrap_or(0);

        let mut selected = Vec::with_capacity(count);
        selected.push(first as u32);
        while selected.len() < count {
            let next = (0..n)
                .filter(|idx| !selected.contains(&(*idx as u32)))
                .max_by(|&a, &b| {
                    let min_a = selected
                        .iter()
                        .map(|&sid| {
                            self.exact_distance(self.node_vector(sid as usize), self.node_vector(a))
                        })
                        .fold(f32::MAX, f32::min);
                    let min_b = selected
                        .iter()
                        .map(|&sid| {
                            self.exact_distance(self.node_vector(sid as usize), self.node_vector(b))
                        })
                        .fold(f32::MAX, f32::min);
                    min_a.total_cmp(&min_b)
                });
            if let Some(idx) = next {
                selected.push(idx as u32);
            } else {
                break;
            }
        }

        if selected.is_empty() {
            selected.push(0);
        }
        self.entry_points = selected;
    }

    #[inline]
    fn node_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline]
    fn node_ref(&self, node_id: u32) -> Option<LoadedNodeRef<'_>> {
        if self.node_ids.is_empty() {
            return None;
        }
        let id = node_id as usize;
        if id >= self.node_ids.len() {
            return None;
        }
        let stride = self.flat_stride.max(1);
        let nb_start = id * stride;
        let count = self.node_neighbor_counts.get(id).copied().unwrap_or(0) as usize;
        let pq_size = self.pq_code_size.max(1);
        let pq_start = id * pq_size;
        Some(LoadedNodeRef {
            id: self.node_ids[id],
            vector: self.node_vector(id),
            neighbors: &self.node_neighbor_ids[nb_start..nb_start + count.min(stride)],
            inline_pq: if pq_start + pq_size <= self.node_pq_codes.len() {
                &self.node_pq_codes[pq_start..pq_start + pq_size]
            } else {
                &[]
            },
        })
    }

    fn bfs_cache_candidates(&self, num_levels: usize, io: &mut BeamSearchIO) -> Vec<u32> {
        if self.entry_points.is_empty() || self.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::with_capacity(self.entry_points.len().saturating_mul(8).max(1));
        let mut queue = VecDeque::with_capacity(self.entry_points.len().max(1));
        let mut ordered = Vec::new();

        for &entry in &self.entry_points {
            if entry as usize >= self.len() || !visited.insert(entry) {
                continue;
            }
            queue.push_back((entry, 0usize));
        }

        while let Some((node_id, level)) = queue.pop_front() {
            let Ok(node) = self.load_node(node_id, NodeAccessMode::Node, io) else {
                continue;
            };
            ordered.push(node_id);

            if level >= num_levels {
                continue;
            }

            for &neighbor in &node.neighbors {
                if neighbor as usize >= self.len() || !visited.insert(neighbor) {
                    continue;
                }
                queue.push_back((neighbor, level + 1));
            }
        }

        ordered
    }

    fn warm_up_cache(&mut self) {
        let _ = self.cache_bfs_levels(2);
    }

    pub fn cache_bfs_levels(&mut self, num_levels: usize) -> usize {
        let mut io = self.io_template.clone();
        let cached = self.bfs_cache_candidates(num_levels, &mut io);
        for &node_id in &cached {
            self.io_template.cache_node(node_id);
            if let Ok(node) = self.load_node(node_id, NodeAccessMode::None, &mut io) {
                if !node.inline_pq.is_empty() {
                    self.io_template.cache_pq_vector(node_id);
                }
            }
        }
        cached.len()
    }

    pub fn generate_cache_list_from_sample_queries(
        &mut self,
        sample_queries: &[Vec<f32>],
        cache_k: usize,
    ) -> usize {
        if cache_k == 0 || sample_queries.is_empty() || self.is_empty() {
            return 0;
        }

        let mut bfs_io = self.io_template.clone();
        let bfs_candidates = self.bfs_cache_candidates(2, &mut bfs_io);
        if bfs_candidates.is_empty() {
            return 0;
        }

        let mut frequency: HashMap<u32, (usize, f32)> = HashMap::new();
        let per_query = cache_k.min(bfs_candidates.len());
        let mut distance_io = self.io_template.clone();

        for query in sample_queries {
            if query.len() != self.dim {
                continue;
            }

            let mut scored = Vec::with_capacity(bfs_candidates.len());
            for &node_id in &bfs_candidates {
                let Ok(node) = self.load_node(node_id, NodeAccessMode::None, &mut distance_io)
                else {
                    continue;
                };
                let Ok(distance) =
                    self.exact_distance_for_node_vector(query, node_id, &node.vector)
                else {
                    continue;
                };
                scored.push((node_id, distance));
            }

            scored.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });

            for &(node_id, distance) in scored.iter().take(per_query) {
                let entry = frequency.entry(node_id).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += distance;
            }
        }

        let mut ranked: Vec<(u32, usize, f32)> = frequency
            .into_iter()
            .map(|(node_id, (count, total_distance))| (node_id, count, total_distance))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| left.2.total_cmp(&right.2))
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut warm_io = self.io_template.clone();
        let mut cached = 0usize;
        for (node_id, _, _) in ranked.into_iter().take(cache_k) {
            self.io_template.cache_node(node_id);
            if let Ok(node) = self.load_node(node_id, NodeAccessMode::Node, &mut warm_io) {
                if !node.inline_pq.is_empty() {
                    self.io_template.cache_pq_vector(node_id);
                }
                cached += 1;
            }
        }

        cached
    }

    fn materialize_storage(&mut self) -> Result<()> {
        if self.storage.is_none() && self.mmap_storage.is_none() {
            return Ok(());
        }

        let node_count = self.node_count;
        let mut vectors = Vec::with_capacity(node_count * self.dim);
        let mut node_data: Vec<std::sync::Arc<LoadedNode>> = Vec::with_capacity(node_count);
        let mut io = self.io_template.clone();
        for node_id in 0..node_count {
            let loaded = self.load_node(node_id as u32, NodeAccessMode::None, &mut io)?;
            if loaded.vector.is_empty() && self.flash_layout.has_separated_vectors {
                vectors.extend_from_slice(&self.load_separated_vector(node_id as u32)?);
            } else {
                vectors.extend_from_slice(&loaded.vector);
            }
            node_data.push(loaded);
        }

        let n = node_data.len();
        let stride = self.compute_build_max_degree().max(1);
        self.flat_stride = stride;
        self.node_ids = node_data.iter().map(|node| node.id).collect();
        self.node_neighbor_counts = node_data
            .iter()
            .map(|node| node.neighbors.len().min(stride) as u32)
            .collect();
        self.node_neighbor_ids = vec![0u32; n * stride];
        let pq_size = self.pq_code_size.max(1);
        self.node_pq_codes = vec![0u8; n * pq_size];
        for (i, node) in node_data.iter().enumerate() {
            let nb_start = i * stride;
            let count = node.neighbors.len().min(stride);
            self.node_neighbor_ids[nb_start..nb_start + count]
                .copy_from_slice(&node.neighbors[..count]);
            let pq_start = i * pq_size;
            let code_len = node.inline_pq.len().min(pq_size);
            self.node_pq_codes[pq_start..pq_start + code_len]
                .copy_from_slice(&node.inline_pq[..code_len]);
        }
        self.vectors = vectors;
        self.storage = None;
        self.mmap_storage = None;
        self.loaded_node_cache = None;
        self.disk_pq_codes.clear();
        if let Some(hvq) = self.hvq_quantizer.as_ref() {
            let expected = n.saturating_mul(hvq.code_size_bytes());
            if self.hvq_codes.len() != expected {
                self.hvq_codes = hvq.encode_batch(n, &self.vectors, self.config.hvq_nrefine);
            }
        } else {
            self.hvq_codes.clear();
        }
        Ok(())
    }

    fn load_disk_pq_codes(&self) -> Result<Vec<u8>> {
        if self.pq_code_size == 0 || self.node_count == 0 {
            return Ok(Vec::new());
        }

        let bytes = if let Some(storage) = &self.storage {
            fs::read(storage.file_group.data_path())?
        } else if let Some(storage) = &self.mmap_storage {
            storage.mmap.to_vec()
        } else {
            return Ok(Vec::new());
        };
        let expected = self.node_count.saturating_mul(self.flash_layout.node_bytes);
        if bytes.len() < expected {
            return Err(KnowhereError::Codec(format!(
                "node file too small for pq extraction: got {} expected at least {}",
                bytes.len(),
                expected
            )));
        }

        let mut out = vec![0u8; self.node_count * self.pq_code_size];
        let hot_vector_bytes = if self.flash_layout.has_separated_vectors {
            0
        } else {
            self.flash_layout.vector_bytes
        };
        let inline_offset = hot_vector_bytes + self.flash_layout.neighbor_bytes;
        let copy_len = self.pq_code_size.min(self.flash_layout.inline_pq_bytes);
        for i in 0..self.node_count {
            if copy_len == 0 {
                continue;
            }
            let node_start = i * self.flash_layout.node_bytes;
            let src_start = node_start + inline_offset;
            let dst_start = i * self.pq_code_size;
            out[dst_start..dst_start + copy_len]
                .copy_from_slice(&bytes[src_start..src_start + copy_len]);
        }
        Ok(out)
    }

    /// Materialize all disk nodes into in-memory flat arrays, giving the same search
    /// performance as a purely in-memory index (no PageCache, no HashMap, no Mutex contention).
    pub fn enable_node_cache(&mut self) -> Result<()> {
        self.materialize_storage()
    }

    fn prime_loaded_node_cache(&mut self) -> Result<()> {
        if self.storage.is_none() && self.mmap_storage.is_none() {
            return Ok(());
        }
        let mut cache = HashMap::with_capacity(self.node_count);
        let mut io = BeamSearchIO::new(&self.flash_layout, &self.config);
        for node_id in 0..self.node_count {
            let loaded = self.load_node(node_id as u32, NodeAccessMode::None, &mut io)?;
            cache.insert(node_id as u32, loaded);
        }
        self.loaded_node_cache = Some(cache);
        Ok(())
    }

    fn load_node(
        &self,
        node_id: u32,
        access_mode: NodeAccessMode,
        io: &mut BeamSearchIO,
    ) -> Result<std::sync::Arc<LoadedNode>> {
        if node_id as usize >= self.len() {
            return Err(KnowhereError::InvalidArg(format!(
                "node_id {} out of bounds for {} nodes",
                node_id,
                self.len()
            )));
        }

        if let Some(cache) = &self.loaded_node_cache {
            if let Some(loaded) = cache.get(&node_id) {
                match access_mode {
                    NodeAccessMode::None => {}
                    NodeAccessMode::Node => {
                        io.record_node_access(node_id, self.flash_layout.node_bytes, 0)
                    }
                    NodeAccessMode::Pq => {
                        if !loaded.inline_pq.is_empty() {
                            io.record_pq_access(node_id, loaded.inline_pq.len(), 0);
                        }
                    }
                }
                return Ok(std::sync::Arc::clone(loaded));
            }
        }

        if let Some(storage) = &self.storage {
            let offset = node_id as usize * self.flash_layout.node_bytes;
            let page_read = storage
                .page_cache
                .read(offset, self.flash_layout.node_bytes)?;
            let loaded = self.deserialize_node(node_id, &page_read.bytes)?;
            match access_mode {
                NodeAccessMode::None => {}
                NodeAccessMode::Node => io.record_node_access(
                    node_id,
                    self.flash_layout.node_bytes,
                    page_read.pages_loaded,
                ),
                NodeAccessMode::Pq => {
                    if !loaded.inline_pq.is_empty() {
                        io.record_pq_access(
                            node_id,
                            loaded.inline_pq.len(),
                            page_read.pages_loaded,
                        );
                    }
                }
            }
            return Ok(std::sync::Arc::new(loaded));
        }

        if let Some(storage) = &self.mmap_storage {
            let offset = node_id as usize * self.flash_layout.node_bytes;
            let end = offset + self.flash_layout.node_bytes;
            let bytes = storage.mmap.get(offset..end).ok_or_else(|| {
                KnowhereError::Storage(format!(
                    "direct mmap read [{}..{}) exceeds file size {}",
                    offset,
                    end,
                    storage.mmap.len()
                ))
            })?;
            let loaded = self.deserialize_node(node_id, bytes)?;
            let pages_loaded = pages_touched(
                offset,
                self.flash_layout.node_bytes,
                self.flash_layout.page_size,
            );
            match access_mode {
                NodeAccessMode::None => {}
                NodeAccessMode::Node => {
                    io.record_node_access(node_id, self.flash_layout.node_bytes, pages_loaded)
                }
                NodeAccessMode::Pq => {
                    if !loaded.inline_pq.is_empty() {
                        io.record_pq_access(node_id, loaded.inline_pq.len(), pages_loaded);
                    }
                }
            }
            return Ok(std::sync::Arc::new(loaded));
        }

        if let Some(node) = self.node_ref(node_id) {
            match access_mode {
                NodeAccessMode::None => {}
                NodeAccessMode::Node => io.record_node_read(node_id, self.flash_layout.node_bytes),
                NodeAccessMode::Pq => {
                    if !node.inline_pq.is_empty() {
                        io.record_pq_read(node_id, node.inline_pq.len());
                    }
                }
            }
            return Ok(std::sync::Arc::new(LoadedNode {
                id: node.id,
                vector: node.vector.to_vec(),
                neighbors: node.neighbors.to_vec(),
                inline_pq: node.inline_pq.to_vec(),
            }));
        }
        Err(KnowhereError::InvalidArg(format!(
            "node_id {} out of bounds for {} nodes",
            node_id,
            self.len()
        )))
    }

    fn serialize_node(&self, node_id: u32) -> Vec<u8> {
        let id = node_id as usize;
        let stride = self.flat_stride.max(1);
        let nb_start = id * stride;
        let count = self.node_neighbor_counts[id] as usize;
        let pq_size = self.pq_code_size.max(1);
        let pq_start = id * pq_size;
        let mut bytes = Vec::with_capacity(self.flash_layout.node_bytes);
        if !self.flash_layout.has_separated_vectors {
            for value in self.node_vector(id) {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }

        let degree = count.min(self.config.max_degree) as u32;
        bytes.extend_from_slice(&degree.to_le_bytes());
        for i in 0..self.config.max_degree {
            let neighbor = if i < count {
                self.node_neighbor_ids[nb_start + i]
            } else {
                0
            };
            bytes.extend_from_slice(&neighbor.to_le_bytes());
        }
        let inline_len = self.flash_layout.inline_pq_bytes.min(pq_size);
        for i in 0..inline_len {
            let v = if pq_start + i < self.node_pq_codes.len() {
                self.node_pq_codes[pq_start + i]
            } else {
                0
            };
            bytes.push(v);
        }
        let external_id = self.node_ids[id];
        bytes.extend_from_slice(&external_id.to_le_bytes());
        bytes.resize(self.flash_layout.node_bytes, 0);
        bytes
    }

    fn deserialize_node(&self, node_id: u32, bytes: &[u8]) -> Result<LoadedNode> {
        if bytes.len() != self.flash_layout.node_bytes {
            return Err(KnowhereError::Codec(format!(
                "invalid node size {}, expected {}",
                bytes.len(),
                self.flash_layout.node_bytes
            )));
        }

        let mut cursor = 0usize;
        let mut vector = Vec::new();
        if !self.flash_layout.has_separated_vectors {
            vector = Vec::with_capacity(self.dim);
            for _ in 0..self.dim {
                let raw = bytes
                    .get(cursor..cursor + 4)
                    .ok_or_else(|| KnowhereError::Codec("truncated vector payload".to_string()))?;
                vector.push(f32::from_le_bytes(raw.try_into().unwrap()));
                cursor += 4;
            }
        }

        let degree = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| KnowhereError::Codec("truncated degree payload".to_string()))?,
        ) as usize;
        cursor += 4;

        let mut neighbors = Vec::with_capacity(degree.min(self.config.max_degree));
        for slot in 0..self.config.max_degree {
            let neighbor = u32::from_le_bytes(
                bytes[cursor..cursor + 4]
                    .try_into()
                    .map_err(|_| KnowhereError::Codec("truncated neighbor payload".to_string()))?,
            );
            cursor += 4;
            if slot < degree {
                neighbors.push(neighbor);
            }
        }

        let inline_pq = bytes[cursor..cursor + self.flash_layout.inline_pq_bytes]
            .iter()
            .copied()
            .take(self.pq_code_size)
            .collect();
        cursor += self.flash_layout.inline_pq_bytes;

        let external_id = if self.flash_layout.id_bytes >= std::mem::size_of::<i64>()
            && cursor + std::mem::size_of::<i64>() <= bytes.len()
        {
            i64::from_le_bytes(
                bytes[cursor..cursor + std::mem::size_of::<i64>()]
                    .try_into()
                    .map_err(|_| KnowhereError::Codec("truncated id payload".to_string()))?,
            )
        } else {
            node_id as i64
        };

        Ok(LoadedNode {
            id: external_id,
            vector,
            neighbors,
            inline_pq,
        })
    }
}

fn resolve_pq_chunks(dim: usize, requested: usize) -> usize {
    let mut chunks = requested.min(dim).max(1);
    while chunks > 1 && dim % chunks != 0 {
        chunks -= 1;
    }
    chunks
}

fn resolve_pq_centroids(num_vectors: usize) -> usize {
    let capped = num_vectors.clamp(2, DEFAULT_PQ_K);
    let mut power = 1usize;
    while power.saturating_mul(2) <= capped {
        power *= 2;
    }
    power.max(2)
}

fn gb_to_bytes(gb: f32) -> usize {
    if gb <= 0.0 {
        return 0;
    }
    let bytes = gb as f64 * 1024.0 * 1024.0 * 1024.0;
    bytes.clamp(0.0, usize::MAX as f64) as usize
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot(left, right) / (left_norm * right_norm)
}

impl Index for PQFlashIndex {
    fn index_type(&self) -> &str {
        "DiskANN-AISAQ"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        let vectors = dataset.vectors();
        PQFlashIndex::train(self, vectors).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        let vectors = dataset.vectors();
        let count_before = self.len();
        PQFlashIndex::add_with_ids(self, vectors, dataset.ids())
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(self.len().saturating_sub(count_before))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let queries = query.vectors();
        let n_queries = queries.len() / self.dim;
        let mut all_ids = Vec::with_capacity(n_queries * top_k);
        let mut all_dists = Vec::with_capacity(n_queries * top_k);
        for i in 0..n_queries {
            let q = &queries[i * self.dim..(i + 1) * self.dim];
            let result = PQFlashIndex::search(self, q, top_k)
                .map_err(|e| IndexError::Unsupported(e.to_string()))?;
            all_ids.extend_from_slice(&result.ids);
            all_dists.extend_from_slice(&result.distances);
        }
        Ok(crate::index::SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let queries = query.vectors();
        let n_queries = queries.len() / self.dim;
        let mut all_ids = Vec::with_capacity(n_queries * top_k);
        let mut all_dists = Vec::with_capacity(n_queries * top_k);
        for i in 0..n_queries {
            let q = &queries[i * self.dim..(i + 1) * self.dim];
            let result = PQFlashIndex::search_with_bitset(self, q, top_k, bitset)
                .map_err(|e| IndexError::Unsupported(e.to_string()))?;
            all_ids.extend_from_slice(&result.ids);
            all_dists.extend_from_slice(&result.distances);
        }
        Ok(crate::index::SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn range_search(
        &self,
        query: &Dataset,
        radius: f32,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let queries = query.vectors();
        let n_queries = queries.len() / self.dim;
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        for i in 0..n_queries {
            let q = &queries[i * self.dim..(i + 1) * self.dim];
            let pairs = PQFlashIndex::range_search_raw(self, q, radius)
                .map_err(|e| IndexError::Unsupported(e.to_string()))?;
            for (id, dist) in pairs {
                all_ids.push(id);
                all_dists.push(dist);
            }
        }
        Ok(crate::index::SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        PQFlashIndex::save(self, path)
            .map(|_| ())
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        *self = PQFlashIndex::load(path).map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
fn block_on<F: Future>(future: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            raw_waker()
        }
        fn wake(_: *const ()) {}
        fn wake_by_ref(_: *const ()) {}
        fn drop(_: *const ()) {}
        RawWaker::new(
            std::ptr::null(),
            &RawWakerVTable::new(clone, wake, wake_by_ref, drop),
        )
    }

    let waker = unsafe { Waker::from_raw(raw_waker()) };
    let mut ctx = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::as_mut(&mut future).poll(&mut ctx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DataType, IndexParams, IndexType};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn set_flat_graph_for_tests(index: &mut PQFlashIndex, node_count: usize, stride: usize) {
        index.node_ids = (0..node_count as i64).collect();
        index.flat_stride = stride.max(1);
        index.node_neighbor_counts = vec![0; node_count];
        index.node_neighbor_ids = vec![0; node_count * index.flat_stride];
        let pq_size = index.pq_code_size.max(1);
        index.node_pq_codes = vec![0; node_count * pq_size];
        index.node_count = node_count;
    }

    #[test]
    fn beam_io_tracks_reads() {
        let config = AisaqConfig {
            beamwidth: 4,
            ..AisaqConfig::default()
        };
        let layout = FlashLayout::new(8, &config);
        let mut io = BeamSearchIO::new(&layout, &config);

        io.record_node_read(1, layout.node_bytes);
        io.cache_node(2);
        io.record_node_read(2, layout.node_bytes);

        assert_eq!(io.stats().nodes_visited, 2);
        assert_eq!(io.stats().nodes_loaded, 1);
        assert_eq!(io.stats().node_cache_hits, 1);
    }

    #[test]
    fn beam_io_pq_cache_respects_capacity() {
        let config = AisaqConfig {
            beamwidth: 4,
            pq_cache_size: 2,
            ..AisaqConfig::default()
        };
        let layout = FlashLayout::new(8, &config);
        let mut io = BeamSearchIO::new(&layout, &config);

        io.cache_pq_vector(1);
        io.cache_pq_vector(2);
        io.cache_pq_vector(3); // evicts 1 under capacity=2

        io.record_pq_read(2, 16);
        io.record_pq_read(1, 16);

        assert_eq!(io.stats().pq_cache_hits, 1);
        assert_eq!(io.stats().bytes_read, 16);
    }

    #[test]
    fn aisaq_config_maps_rerank_expand_pct() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 8,
            data_type: DataType::Float,
            params: IndexParams {
                disk_rerank_expand_pct: Some(250),
                disk_pq_candidate_expand_pct: Some(150),
                disk_num_entry_points: Some(3),
                disk_pq_dims: Some(4),
                disk_pq_code_budget_gb: Some(0.5),
                disk_pq_cache_size: Some(64),
                disk_rearrange: Some(false),
                disk_random_init_edges: Some(5),
                disk_build_dram_budget_gb: Some(1.25),
                disk_search_cache_budget_gb: Some(0.01),
                disk_build_degree_slack_pct: Some(130),
                disk_warm_up: Some(true),
                disk_filter_threshold: Some(0.25),
                disk_io_cutting: Some(true),
                disk_io_cutting_threshold: Some(0.05),
                disk_io_cutting_patience: Some(4),
                random_seed: Some(9),
                ..Default::default()
            },
        };
        let mapped = AisaqConfig::from_index_config(&config);
        assert_eq!(mapped.rerank_expand_pct, 250);
        assert_eq!(mapped.pq_candidate_expand_pct, 150);
        assert_eq!(mapped.num_entry_points, 3);
        assert_eq!(mapped.disk_pq_dims, 4);
        assert_eq!(mapped.pq_code_budget_gb, 0.5);
        assert_eq!(mapped.pq_cache_size, 64);
        assert!(!mapped.rearrange);
        assert_eq!(mapped.random_init_edges, 5);
        assert_eq!(mapped.random_seed, 9);
        assert_eq!(mapped.build_dram_budget_gb, 1.25);
        assert_eq!(mapped.pq_read_page_cache_size, gb_to_bytes(0.01));
        assert_eq!(mapped.build_degree_slack_pct, 130);
        assert!(mapped.warm_up);
        assert_eq!(mapped.filter_threshold, 0.25);
        assert!(mapped.io_cutting_enabled);
        assert_eq!(mapped.io_cutting_threshold, 0.05);
        assert_eq!(mapped.io_cutting_patience, 4);
    }

    #[test]
    fn aisaq_rerank_pool_size_is_bounded() {
        let index = PQFlashIndex::new(
            AisaqConfig {
                rearrange: true,
                ..AisaqConfig::default()
            },
            MetricType::L2,
            8,
        )
        .unwrap();
        assert_eq!(index.compute_rerank_pool_size(10, 0), 0);
        assert_eq!(index.compute_rerank_pool_size(10, 5), 5);
        assert_eq!(index.compute_rerank_pool_size(10, 30), 20);
    }

    #[test]
    fn aisaq_rerank_pool_size_without_rearrange_uses_topk_only() {
        let index = PQFlashIndex::new(
            AisaqConfig {
                rearrange: false,
                rerank_expand_pct: 300,
                ..AisaqConfig::default()
            },
            MetricType::L2,
            8,
        )
        .unwrap();
        assert_eq!(index.compute_rerank_pool_size(10, 30), 10);
    }

    #[test]
    fn aisaq_search_runs_with_rerank_stage() {
        let config = AisaqConfig {
            max_degree: 4,
            search_list_size: 16,
            beamwidth: 4,
            disk_pq_dims: 2,
            rerank_expand_pct: 200,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let result = index.search(&[0.9, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.distances.len(), 2);
        assert!(result.ids.iter().all(|&id| id >= 0));
    }

    #[test]
    fn aisaq_search_async_matches_sync() {
        let config = AisaqConfig {
            max_degree: 4,
            search_list_size: 16,
            beamwidth: 4,
            disk_pq_dims: 2,
            rerank_expand_pct: 200,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let sync = index.search(&[0.9, 0.0, 0.0, 0.0], 3).unwrap();
        let async_res = block_on(index.search_async(&[0.9, 0.0, 0.0, 0.0], 3)).unwrap();
        assert_eq!(sync.ids, async_res.ids);
        assert_eq!(sync.distances.len(), async_res.distances.len());
        for (a, b) in sync.distances.iter().zip(async_res.distances.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn aisaq_io_cutting_preserves_top1_on_simple_search() {
        let base_config = AisaqConfig {
            max_degree: 4,
            search_list_size: 32,
            beamwidth: 4,
            ..AisaqConfig::default()
        };
        let mut cut_config = base_config.clone();
        cut_config.io_cutting_enabled = true;
        cut_config.io_cutting_threshold = 1.0;
        cut_config.io_cutting_patience = 1;

        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0, //
            5.0, 0.0, 0.0, 0.0,
        ];

        let mut baseline = PQFlashIndex::new(base_config, MetricType::L2, 4).unwrap();
        baseline.train(&vectors).unwrap();
        baseline.add(&vectors).unwrap();

        let mut cutting = PQFlashIndex::new(cut_config, MetricType::L2, 4).unwrap();
        cutting.train(&vectors).unwrap();
        cutting.add(&vectors).unwrap();

        let query = [0.9f32, 0.0, 0.0, 0.0];
        let baseline_result = baseline.search(&query, 3).unwrap();
        let cutting_result = cutting.search(&query, 3).unwrap();

        assert_eq!(cutting_result.ids.first(), baseline_result.ids.first());
        assert!(cutting_result.num_visited <= baseline_result.num_visited);
    }

    #[test]
    fn aisaq_entry_points_use_centroid_anchor_instead_of_sequential_ids() {
        let config = AisaqConfig {
            num_entry_points: 2,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        let vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            100.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        assert_eq!(index.entry_points.len(), 2);
        assert_eq!(
            index.entry_points[0], 2,
            "first entry should be centroid-nearest anchor, not sequential id"
        );
    }

    #[test]
    fn aisaq_entry_points_include_far_diverse_seed() {
        let config = AisaqConfig {
            num_entry_points: 2,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        let vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            100.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        assert_eq!(index.entry_points.len(), 2);
        assert_eq!(
            index.entry_points[1], 3,
            "second entry should be a far diverse seed from the centroid anchor"
        );
    }

    #[test]
    fn aisaq_link_back_robust_prune_applies_alpha_occlusion() {
        let config = AisaqConfig {
            max_degree: 2,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        index.vectors = vec![
            0.0f32, 0.0, // node 0
            100.0, 0.0, // node 1 (far)
            1.0, 0.0, // node 2 (near)
            2.0, 0.0, // node 3 (near)
        ];
        set_flat_graph_for_tests(&mut index, 4, 2);
        index.node_neighbor_counts[0] = 2;
        index.node_neighbor_ids[0] = 1;
        index.node_neighbor_ids[1] = 2;
        index.node_neighbor_counts[1] = 1;
        index.node_neighbor_ids[2] = 0;
        index.node_neighbor_counts[2] = 1;
        index.node_neighbor_ids[4] = 0;

        index.link_back(0, 3);
        let cnt = index.node_neighbor_counts[0] as usize;
        let nbrs = &index.node_neighbor_ids[0..cnt];
        assert_eq!(cnt, 1);
        assert!(nbrs.contains(&2));
        assert!(
            !nbrs.contains(&1),
            "standard alpha-robust prune should occlude the far edge once a closer neighbor explains it"
        );
        assert!(
            !nbrs.contains(&3),
            "near-but-occluded edge should be pruned under alpha-occlusion"
        );
    }

    #[test]
    fn aisaq_link_back_no_guard_appends_into_empty_slots_without_prune() {
        let config = AisaqConfig {
            max_degree: 2,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        index.vectors = vec![
            0.0f32, 0.0, // node 0
            100.0, 0.0, // node 1 (far late node)
            1.0, 0.0, // node 2 (near existing neighbor)
        ];
        set_flat_graph_for_tests(&mut index, 3, 2);
        index.node_neighbor_counts[0] = 1;
        index.node_neighbor_ids[0] = 2;

        index.link_back_no_guard(0, 1, 2);

        let cnt = index.node_neighbor_counts[0] as usize;
        let nbrs = &index.node_neighbor_ids[0..cnt];
        assert_eq!(cnt, 2);
        assert!(nbrs.contains(&2));
        assert!(
            nbrs.contains(&1),
            "reverse-edge recovery should preserve the far back-link while slots remain"
        );
    }

    #[test]
    fn aisaq_add_fails_when_build_budget_too_small() {
        let config = AisaqConfig {
            max_degree: 4,
            build_dram_budget_gb: 0.000_000_001,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ];

        let err = index.add(&vectors).unwrap_err();
        match err {
            KnowhereError::InvalidArg(message) => {
                assert!(message.contains("disk_build_dram_budget_gb exceeded"));
            }
            other => panic!("expected InvalidArg, got {other:?}"),
        }
    }

    #[test]
    fn aisaq_add_succeeds_when_build_budget_sufficient() {
        let config = AisaqConfig {
            max_degree: 4,
            build_dram_budget_gb: 0.1,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0,
        ];

        index.add(&vectors).unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn aisaq_add_fails_when_pq_code_budget_too_small() {
        let config = AisaqConfig {
            max_degree: 4,
            disk_pq_dims: 2,
            pq_code_budget_gb: 0.000_000_001,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];

        let err = index.add(&vectors).unwrap_err();
        match err {
            KnowhereError::InvalidArg(message) => {
                assert!(message.contains("disk_pq_code_budget_gb exceeded"));
            }
            other => panic!("expected InvalidArg, got {other:?}"),
        }
    }

    #[test]
    fn aisaq_build_degree_slack_computes_expanded_build_limit() {
        let config = AisaqConfig {
            max_degree: 10,
            build_degree_slack_pct: 130,
            ..AisaqConfig::default()
        };
        let index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        assert_eq!(index.compute_build_max_degree(), 13);
    }

    #[test]
    fn aisaq_add_with_degree_slack_prunes_back_to_target_degree() {
        let config = AisaqConfig {
            max_degree: 2,
            build_degree_slack_pct: 200,
            search_list_size: 32,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        let vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        index.add(&vectors).unwrap();
        assert!(index
            .node_neighbor_counts
            .iter()
            .all(|&count| count as usize <= index.config.max_degree));
    }

    #[test]
    fn aisaq_vectors_beamwidth_default_keeps_io_beamwidth_limit() {
        let config = AisaqConfig {
            beamwidth: 8,
            vectors_beamwidth: 1,
            ..AisaqConfig::default()
        };
        let index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let io = BeamSearchIO::new(&index.flash_layout, &index.config);
        assert_eq!(index.compute_expand_limit(&io), 8);
    }

    #[test]
    fn aisaq_vectors_beamwidth_caps_neighbor_expansion_when_larger_than_one() {
        let config = AisaqConfig {
            beamwidth: 8,
            vectors_beamwidth: 3,
            ..AisaqConfig::default()
        };
        let index = PQFlashIndex::new(config, MetricType::L2, 4).unwrap();
        let io = BeamSearchIO::new(&index.flash_layout, &index.config);
        assert_eq!(index.compute_expand_limit(&io), 3);
    }

    #[test]
    fn aisaq_compute_max_visit_without_pq_ignores_candidate_expand_pct() {
        let config = AisaqConfig {
            search_list_size: 10,
            pq_candidate_expand_pct: 200,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        index.trained = true;
        index.vectors = vec![0.0; 100 * 2];
        let stride = index.config.max_degree.max(1);
        set_flat_graph_for_tests(&mut index, 100, stride);
        assert_eq!(index.compute_max_visit(5), 10);
    }

    #[test]
    fn aisaq_compute_max_visit_with_pq_applies_candidate_expand_pct() {
        let config = AisaqConfig {
            search_list_size: 10,
            pq_candidate_expand_pct: 150,
            disk_pq_dims: 2,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        let train = vec![0.0f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        index.train(&train).unwrap();
        index.vectors = vec![0.0; 100 * 2];
        let stride = index.config.max_degree.max(1);
        set_flat_graph_for_tests(&mut index, 100, stride);
        assert_eq!(index.compute_max_visit(5), 15);
    }

    #[test]
    fn aisaq_filter_threshold_triggers_exact_scan_when_allowed_ratio_is_small() {
        let config = AisaqConfig {
            max_degree: 4,
            filter_threshold: 0.4,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        let vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        index.add(&vectors).unwrap();
        let mut bitset = BitsetView::new(index.len());
        bitset.set(0, true);
        bitset.set(1, true);
        bitset.set(2, true);
        bitset.set(3, true);

        let result = index
            .search_with_bitset(&[4.1, 0.0], 1, &bitset)
            .expect("search_with_bitset should succeed");
        assert_eq!(result.ids, vec![4]);
        assert_eq!(result.num_visited, 1);
    }

    #[test]
    fn aisaq_filter_threshold_gate_defaults_to_disabled() {
        let config = AisaqConfig::default();
        let mut index = PQFlashIndex::new(config, MetricType::L2, 2).unwrap();
        index.vectors = vec![0.0; 4 * 2];
        let stride = index.config.max_degree.max(1);
        set_flat_graph_for_tests(&mut index, 4, stride);
        let mut bitset = BitsetView::new(index.len());
        bitset.set(0, true);
        bitset.set(1, true);
        assert!(!index.should_force_exact_filter_scan(Some(&bitset)));
    }

    #[test]
    fn aisaq_cache_bfs_levels_warms_two_hops_from_entry_points() {
        let mut index = PQFlashIndex::new(AisaqConfig::default(), MetricType::L2, 2).unwrap();
        index.vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        set_flat_graph_for_tests(&mut index, 5, 2);
        index.entry_points = vec![0];
        index.node_neighbor_counts[0] = 2;
        index.node_neighbor_ids[0] = 1;
        index.node_neighbor_ids[1] = 2;
        index.node_neighbor_counts[1] = 1;
        index.node_neighbor_ids[2] = 3;
        index.node_neighbor_counts[2] = 1;
        index.node_neighbor_ids[4] = 4;

        let warmed = index.cache_bfs_levels(2);

        assert_eq!(warmed, 5);
        for node_id in 0..5u32 {
            assert!(index.io_template.cached_nodes.contains(&node_id));
            assert!(index.io_template.cached_pq_vectors.contains(&node_id));
        }
    }

    #[test]
    fn aisaq_generate_cache_list_from_sample_queries_promotes_frequent_bfs_hubs() {
        let mut index = PQFlashIndex::new(AisaqConfig::default(), MetricType::L2, 2).unwrap();
        index.vectors = vec![
            0.0f32, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        set_flat_graph_for_tests(&mut index, 5, 2);
        index.entry_points = vec![0];
        index.node_neighbor_counts[0] = 2;
        index.node_neighbor_ids[0] = 1;
        index.node_neighbor_ids[1] = 2;
        index.node_neighbor_counts[1] = 1;
        index.node_neighbor_ids[2] = 3;
        index.node_neighbor_counts[2] = 1;
        index.node_neighbor_ids[4] = 4;

        let cached = index.generate_cache_list_from_sample_queries(
            &[vec![3.0, 0.0], vec![3.1, 0.0], vec![2.9, 0.0]],
            1,
        );

        assert_eq!(cached, 1);
        assert!(index.io_template.cached_nodes.contains(&3));
        assert!(index.io_template.cached_pq_vectors.contains(&3));
    }

    fn brute_force_topk_l2(
        vectors: &[f32],
        queries: &[f32],
        dim: usize,
        k: usize,
    ) -> Vec<Vec<i64>> {
        let n = vectors.len() / dim;
        let nq = queries.len() / dim;
        let mut out = Vec::with_capacity(nq);
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];
            let mut scored = Vec::with_capacity(n);
            for idx in 0..n {
                let v = &vectors[idx * dim..(idx + 1) * dim];
                scored.push((idx as i64, simd::l2_distance_sq(q, v)));
            }
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            out.push(scored.into_iter().take(k).map(|(id, _)| id).collect());
        }
        out
    }

    fn brute_force_topk_ip(
        vectors: &[f32],
        queries: &[f32],
        dim: usize,
        k: usize,
    ) -> Vec<Vec<i64>> {
        let n = vectors.len() / dim;
        let nq = queries.len() / dim;
        let mut out = Vec::with_capacity(nq);
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];
            let mut scored = Vec::with_capacity(n);
            for idx in 0..n {
                let v = &vectors[idx * dim..(idx + 1) * dim];
                scored.push((idx as i64, -dot(q, v)));
            }
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            out.push(scored.into_iter().take(k).map(|(id, _)| id).collect());
        }
        out
    }

    fn compute_recall(results: &SearchResult, ground_truth: &[Vec<i64>], k: usize) -> f64 {
        let mut hits = 0usize;
        for (query_idx, gt_ids) in ground_truth.iter().enumerate() {
            let start = query_idx * k;
            let end = ((query_idx + 1) * k).min(results.ids.len());
            let result_ids: HashSet<i64> = results.ids[start..end].iter().copied().collect();
            for &gt_id in gt_ids.iter().take(k) {
                if result_ids.contains(&gt_id) {
                    hits += 1;
                }
            }
        }
        hits as f64 / (k * ground_truth.len()) as f64
    }

    fn brute_force_range_l2(
        vectors: &[f32],
        query: &[f32],
        dim: usize,
        radius: f32,
    ) -> Vec<(i64, f32)> {
        let mut out = Vec::new();
        for idx in 0..(vectors.len() / dim) {
            let v = &vectors[idx * dim..(idx + 1) * dim];
            let dist = simd::l2_distance_sq(query, v);
            if dist <= radius {
                out.push((idx as i64, dist));
            }
        }
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    fn run_batch_search(
        index: &PQFlashIndex,
        queries: &[f32],
        dim: usize,
        k: usize,
    ) -> SearchResult {
        let nq = queries.len() / dim;
        let mut ids = Vec::with_capacity(nq * k);
        let mut dists = Vec::with_capacity(nq * k);
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];
            let res = index.search(q, k).unwrap();
            ids.extend(res.ids);
            dists.extend(res.distances);
        }
        SearchResult::new(ids, dists, 0.0)
    }

    fn run_batch_search_with_bitset(
        index: &PQFlashIndex,
        queries: &[f32],
        dim: usize,
        k: usize,
        bitset: &BitsetView,
    ) -> SearchResult {
        let nq = queries.len() / dim;
        let mut ids = Vec::with_capacity(nq * k);
        let mut dists = Vec::with_capacity(nq * k);
        for q_idx in 0..nq {
            let q = &queries[q_idx * dim..(q_idx + 1) * dim];
            let res = index.search_with_bitset(q, k, bitset).unwrap();
            ids.extend(res.ids);
            dists.extend(res.distances);
        }
        SearchResult::new(ids, dists, 0.0)
    }

    fn make_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}"))
    }

    #[test]
    fn test_pqflash_recall_quality() {
        let dim = 16usize;
        let n = 500usize;
        let nq = 30usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let results = run_batch_search(&index, &queries, dim, k);
        let gt = brute_force_topk_l2(&vectors, &queries, dim, k);
        let recall = compute_recall(&results, &gt, k);
        assert!(recall >= 0.70, "recall@5 too low: {recall:.4}");
    }

    #[test]
    fn test_pqflash_hvq_coarse_scoring() {
        let dim = 32usize;
        let n = 1000usize;
        let nq = 24usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(123);
        let normalize_rows = |data: &[f32], dim: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(data.len());
            for row in data.chunks_exact(dim) {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                out.extend(row.iter().map(|x| x / norm));
            }
            out
        };
        let raw_vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let raw_queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let vectors = normalize_rows(&raw_vectors, dim);
        let queries = normalize_rows(&raw_queries, dim);

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 64,
            disk_pq_dims: 4,
            use_hvq: true,
            hvq_nbits: 4,
            hvq_nrefine: 3,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::Ip, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let hvq = index
            .hvq_quantizer
            .as_ref()
            .expect("hvq quantizer should be present");
        let hvq_code_size = hvq.code_size_bytes();
        assert_eq!(index.hvq_codes.len(), n * hvq_code_size);

        let results = run_batch_search(&index, &queries, dim, k);
        let gt = brute_force_topk_ip(&vectors, &queries, dim, k);
        let recall = compute_recall(&results, &gt, k);
        assert!(recall >= 0.35, "hvq coarse recall@{k} too low: {recall:.4}");

        let dir = make_temp_dir("pqflash_hvq");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let loaded = PQFlashIndex::load(&dir).unwrap();

        assert!(
            loaded.hvq_quantizer.is_some(),
            "hvq quantizer should roundtrip"
        );
        assert_eq!(loaded.hvq_codes.len(), n * hvq_code_size);

        let loaded_results = run_batch_search(&loaded, &queries, dim, k);
        let loaded_recall = compute_recall(&loaded_results, &gt, k);
        assert!(
            loaded_recall >= 0.35,
            "loaded hvq coarse recall@{k} too low: {loaded_recall:.4}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_aisaq_range_search() {
        let dim = 16usize;
        let n = 100usize;
        let query_idx = 17usize;
        let radius = 1e-6f32;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(if j == 0 { i as f32 * 0.25 } else { 0.0 });
            }
        }
        let query = vectors[query_idx * dim..(query_idx + 1) * dim].to_vec();

        let config = AisaqConfig {
            max_degree: 12,
            search_list_size: 64,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let hits = index.range_search_raw(&query, radius).unwrap();
        let gt = brute_force_range_l2(&vectors, &query, dim, radius);
        let hit_ids: HashSet<i64> = hits.iter().map(|(id, _)| *id).collect();

        assert!(
            !gt.is_empty(),
            "brute-force range search should find at least self"
        );
        assert!(
            !hits.is_empty(),
            "aisaq range search should return at least one hit"
        );
        assert!(hits.iter().all(|(_, dist)| *dist <= radius + 1e-6));
        assert!(
            gt.iter().any(|(id, _)| hit_ids.contains(id)),
            "aisaq range search should overlap brute-force hits"
        );
    }

    #[test]
    fn test_pqflash_bitset_filter() {
        let dim = 16usize;
        let n = 300usize;
        let nq = 20usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let mut bitset = BitsetView::new(index.len());
        for i in (0..index.len()).step_by(2) {
            bitset.set(i, true);
        }

        let results = run_batch_search_with_bitset(&index, &queries, dim, k, &bitset);
        for &id in &results.ids {
            if id >= 0 {
                assert_ne!(
                    (id as usize) % 2,
                    0,
                    "deleted id {id} should not appear in filtered results"
                );
            }
        }
    }

    #[test]
    fn test_pqflash_save_load_roundtrip() {
        let dim = 16usize;
        let n = 300usize;
        let nq = 20usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();
        let original = run_batch_search(&index, &queries, dim, k);

        let dir = make_temp_dir("pqflash_roundtrip");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let loaded = PQFlashIndex::load(&dir).unwrap();
        let loaded_results = run_batch_search(&loaded, &queries, dim, k);

        assert_eq!(loaded_results.ids, original.ids);
        assert_eq!(loaded_results.distances, original.distances);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pqflash_save_load_with_pq() {
        let dim = 16usize;
        let n = 300usize;
        let nq = 20usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 4,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_roundtrip_pq");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let loaded = PQFlashIndex::load(&dir).unwrap();
        let loaded_results = run_batch_search(&loaded, &queries, dim, k);

        let gt = brute_force_topk_l2(&vectors, &queries, dim, k);
        let recall = compute_recall(&loaded_results, &gt, k);
        assert!(recall >= 0.65, "loaded pq recall@5 too low: {recall:.4}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_native_diskann_pq_format() {
        use std::io::Read;

        fn read_header(path: &std::path::Path) -> (i32, i32) {
            let mut f = std::fs::File::open(path).unwrap();
            let mut b = [0u8; 8];
            f.read_exact(&mut b).unwrap();
            let npts = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let ndims = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            (npts, ndims)
        }

        let dim = 16usize;
        let n = 300usize; // ensure resolve_pq_centroids -> 256
        let mut rng = StdRng::seed_from_u64(123);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            disk_pq_dims: 4,
            max_degree: 16,
            search_list_size: 32,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_native_export");
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("native");
        index
            .export_native_diskann_pq(prefix.to_str().unwrap())
            .unwrap();

        let pivots = dir.join("native_pq_pivots.bin");
        let perm = dir.join("native_pq_pivots.bin_rearrangement_perm.bin");
        let chunk_offsets = dir.join("native_pq_pivots.bin_chunk_offsets.bin");
        let centroid = dir.join("native_pq_pivots.bin_centroid.bin");
        let compressed = dir.join("native_pq_compressed.bin");

        let (npts, ndims) = read_header(&pivots);
        assert_eq!(npts, 256);
        assert_eq!(ndims, dim as i32);
        assert_eq!(
            std::fs::metadata(&pivots).unwrap().len() as usize,
            8 + 256 * dim * std::mem::size_of::<f32>()
        );

        let (npts, ndims) = read_header(&perm);
        assert_eq!(npts, dim as i32);
        assert_eq!(ndims, 1);
        assert_eq!(
            std::fs::metadata(&perm).unwrap().len() as usize,
            8 + dim * std::mem::size_of::<u32>()
        );

        let m = index.pq_code_size;
        let (npts, ndims) = read_header(&chunk_offsets);
        assert_eq!(npts, (m + 1) as i32);
        assert_eq!(ndims, 1);
        assert_eq!(
            std::fs::metadata(&chunk_offsets).unwrap().len() as usize,
            8 + (m + 1) * std::mem::size_of::<u32>()
        );

        let (npts, ndims) = read_header(&centroid);
        assert_eq!(npts, dim as i32);
        assert_eq!(ndims, 1);
        assert_eq!(
            std::fs::metadata(&centroid).unwrap().len() as usize,
            8 + dim * std::mem::size_of::<f32>()
        );

        let (npts, ndims) = read_header(&compressed);
        assert_eq!(npts, index.node_ids.len() as i32);
        assert_eq!(ndims, m as i32);
        assert_eq!(
            std::fs::metadata(&compressed).unwrap().len() as usize,
            8 + index.node_ids.len() * m
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_native_disk_index_format() {
        use std::io::Read;

        let dim = 4usize;
        let n = 10usize;
        let mut rng = StdRng::seed_from_u64(77);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 8,
            search_list_size: 16,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_native_disk_index");
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("native");
        index
            .export_native_disk_index(prefix.to_str().unwrap())
            .unwrap();

        let disk_index_path = dir.join("native_disk.index");
        let medoids_path = dir.join("native_disk.index_medoids.bin");
        let centroids_path = dir.join("native_disk.index_centroids.bin");

        let mut f = std::fs::File::open(&disk_index_path).unwrap();
        let mut b = [0u8; 8];
        f.read_exact(&mut b).unwrap();
        let expected_file_size = u64::from_le_bytes(b);
        let actual_file_size = std::fs::metadata(&disk_index_path).unwrap().len();
        assert_eq!(actual_file_size, expected_file_size);

        let mut sector0 = [0u8; 64];
        let mut f = std::fs::File::open(&disk_index_path).unwrap();
        f.read_exact(&mut sector0).unwrap();
        let header: Vec<u64> = sector0
            .chunks_exact(8)
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(chunk);
                u64::from_le_bytes(bytes)
            })
            .collect();
        assert_eq!(header[1], index.node_ids.len() as u64);
        assert!(
            header[2] < index.node_ids.len() as u64,
            "medoid id in sector0 header should be a valid node id"
        );

        let mut centroid_header = [0u8; 8];
        let mut centroids_file = std::fs::File::open(&centroids_path).unwrap();
        centroids_file.read_exact(&mut centroid_header).unwrap();
        let centroid_rows = i32::from_le_bytes([
            centroid_header[0],
            centroid_header[1],
            centroid_header[2],
            centroid_header[3],
        ]);
        let centroid_dim = i32::from_le_bytes([
            centroid_header[4],
            centroid_header[5],
            centroid_header[6],
            centroid_header[7],
        ]);
        assert_eq!(centroid_rows, 1);
        assert_eq!(centroid_dim, dim as i32);

        assert_eq!(std::fs::metadata(&medoids_path).unwrap().len(), 12);
        assert_eq!(
            std::fs::metadata(&centroids_path).unwrap().len(),
            (8 + dim * 4) as u64
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_import_native_disk_index_roundtrip() {
        let dim = 8usize;
        let n = 24usize;
        let k = 4usize;
        let mut rng = StdRng::seed_from_u64(91);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..8 * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 8,
            search_list_size: 16,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_native_disk_roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("native");
        index
            .export_native_disk_index(prefix.to_str().unwrap())
            .unwrap();

        let imported = PQFlashIndex::import_native_disk_index(prefix.to_str().unwrap()).unwrap();
        assert_eq!(imported.len(), index.len());
        assert_eq!(imported.entry_points, index.entry_points);
        assert_eq!(imported.node_neighbor_counts, index.node_neighbor_counts);
        assert_eq!(imported.node_neighbor_ids, index.node_neighbor_ids);
        assert_eq!(imported.vectors.len(), index.vectors.len());
        for (a, b) in imported.vectors.iter().zip(index.vectors.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let original_results = run_batch_search(&index, &queries, dim, k);
        let imported_results = run_batch_search(&imported, &queries, dim, k);
        assert_eq!(original_results.ids, imported_results.ids);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pqflash_disk_path_after_load() {
        let dim = 16usize;
        let n = 300usize;
        let nq = 20usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_disk_path");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let loaded = PQFlashIndex::load(&dir).unwrap();

        // NoPQ mode: load() materializes vectors into memory for fast access
        assert!(
            !loaded.vectors.is_empty(),
            "NoPQ load should materialize vectors"
        );
        assert!(
            loaded.storage.is_none(),
            "NoPQ load should clear disk storage after materialization"
        );
        let results = run_batch_search(&loaded, &queries, dim, k);
        assert_eq!(results.ids.len(), nq * k);
        assert!(results.ids.iter().all(|&id| id >= 0));
        let gt = brute_force_topk_l2(&vectors, &queries, dim, k);
        let recall = compute_recall(&results, &gt, k);
        assert!(
            recall >= 0.50,
            "NoPQ load recall@{k} too low after load: {recall:.4}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pqflash_disk_path_after_load_with_mmap() {
        let dim = 16usize;
        let n = 300usize;
        let nq = 20usize;
        let k = 5usize;
        let mut rng = StdRng::seed_from_u64(84);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 50,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_disk_path_mmap");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let loaded = PQFlashIndex::load_with_mmap(&dir).unwrap();

        assert!(
            loaded.vectors.is_empty(),
            "mmap-backed load should stay disk-backed"
        );
        assert!(
            loaded.mmap_storage.is_some(),
            "direct mmap storage should be set"
        );
        let results = run_batch_search(&loaded, &queries, dim, k);
        assert_eq!(results.ids.len(), nq * k);
        assert!(results.ids.iter().all(|&id| id >= 0));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pqflash_disk_search_path() {
        let dim = 64usize;
        let n = 1000usize;
        let nq = 50usize;
        let k = 10usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(0.0..1.0)).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen_range(0.0..1.0)).collect();

        let build_config = AisaqConfig {
            max_degree: 24,
            search_list_size: 100,
            disk_pq_dims: 0,
            cache_all_on_load: false,
            pq_read_page_cache_size: 128 * 1024,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(build_config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        let dir = make_temp_dir("pqflash_disk_search_path");
        fs::create_dir_all(&dir).unwrap();
        index.save(&dir).unwrap();
        let disk_index = PQFlashIndex::load(&dir).unwrap();

        // NoPQ mode: load() materializes vectors into memory for fast access
        assert!(
            !disk_index.vectors.is_empty(),
            "NoPQ load should materialize vectors"
        );
        assert!(
            disk_index.storage.is_none(),
            "NoPQ load should clear disk storage after materialization"
        );

        let results = run_batch_search(&disk_index, &queries, dim, k);
        assert_eq!(results.ids.len(), nq * k);
        assert!(
            results.ids.iter().any(|&id| id >= 0),
            "disk search returned no valid ids"
        );

        let gt = brute_force_topk_l2(&vectors, &queries, dim, k);
        let recall = compute_recall(&results, &gt, k);
        assert!(recall >= 0.85, "disk path recall@10 too low: {recall:.4}");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pqflash_external_ids() {
        use crate::dataset::Dataset;
        use crate::index::Index;

        let n = 100usize;
        let dim = 16usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(0.0..1.0)).collect();
        let ext_ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 64,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        let dataset = Dataset::from_f32_slice_with_ids(&vectors, dim, ext_ids);
        Index::train(&mut index, &dataset).unwrap();
        Index::add(&mut index, &dataset).unwrap();

        let query = Dataset::from_f32_slice(&vectors[0..dim], dim);
        let result = Index::search(&index, &query, 5).unwrap();
        for &id in &result.ids {
            if id >= 0 {
                assert!(
                    (1000..1100).contains(&id),
                    "id {} out of external range [1000, 1100)",
                    id
                );
            }
        }
    }

    #[test]
    fn test_soft_delete_and_consolidate() {
        let dim = 32usize;
        let n = 1000usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(0.0..1.0)).collect();

        let config = AisaqConfig {
            max_degree: 16,
            search_list_size: 64,
            disk_pq_dims: 0,
            ..AisaqConfig::default()
        };
        let mut index = PQFlashIndex::new(config, MetricType::L2, dim).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors).unwrap();

        for id in 0i64..100i64 {
            assert!(index.soft_delete(id), "failed to soft_delete id={id}");
        }
        assert!(index.is_deleted(0));
        assert!(!index.is_deleted(500));
        assert_eq!(index.deleted_count(), 100);

        let q = &vectors[0..dim];
        let result = index.search(q, 10).unwrap();
        for &id in &result.ids {
            assert!(
                !(0..100).contains(&id),
                "soft-deleted id {} appeared in search result",
                id
            );
        }

        let removed = index.consolidate();
        assert_eq!(removed, 100);
        assert_eq!(index.node_count, 900);
        assert_eq!(index.deleted_count(), 0);
        assert!(!index.is_deleted(0));
        assert!(!index.is_deleted(500));

        let result2 = index.search(q, 10).unwrap();
        for &id in &result2.ids {
            assert!(
                !(0..100).contains(&id),
                "deleted-range id {} appeared after consolidate",
                id
            );
        }
    }
}

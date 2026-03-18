//! DiskANN AISAQ skeleton with PQ Flash abstractions.
//!
//! This module provides a first Rust architecture for the SSD-oriented AISAQ path.
//! The current implementation keeps data in memory while preserving the core
//! concepts from the C++ design: flash layout, PQ-compressed payloads, beam-search
//! IO accounting, multiple entry points, and coarse-to-exact search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[cfg(test)]
use std::future::Future;
#[cfg(all(feature = "async-io", target_os = "linux"))]
use std::os::fd::AsRawFd;

use crate::api::{IndexConfig, KnowhereError, MetricType, Result, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::faiss::pq::PqEncoder;
use crate::index::{Index, IndexError};
use crate::simd;
#[cfg(all(feature = "async-io", target_os = "linux"))]
use io_uring::{opcode, types, IoUring};
use memmap2::Mmap;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

const DEFAULT_PAGE_SIZE: usize = 4096;
const DEFAULT_PQ_K: usize = 256;
const PAGE_CACHE_SHARDS: usize = 16;

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
    pub warm_up: bool,
    pub filter_threshold: f32,
    #[serde(default)]
    pub cache_all_on_load: bool,
    #[serde(default = "default_true")]
    pub run_refine_pass: bool,
}

fn default_true() -> bool {
    true
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
            warm_up: false,
            filter_threshold: -1.0,
            cache_all_on_load: false,
            run_refine_pass: true,
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
            build_degree_slack_pct: params.disk_build_degree_slack_pct.unwrap_or(100).clamp(100, 300),
            build_search_list_size: 0,
            build_dram_budget_gb: params.disk_build_dram_budget_gb.unwrap_or(0.0),
            pq_read_page_cache_size: gb_to_bytes(params.disk_search_cache_budget_gb.unwrap_or(0.0)),
            warm_up: params.disk_warm_up.unwrap_or(false),
            filter_threshold: params.disk_filter_threshold.unwrap_or(-1.0).clamp(-1.0, 1.0),
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
    pub node_bytes: usize,
}

impl FlashLayout {
    fn new(dim: usize, config: &AisaqConfig) -> Self {
        let vector_bytes = dim * std::mem::size_of::<f32>();
        let inline_pq_bytes = config.inline_pq.max(config.disk_pq_dims);
        let neighbor_bytes = (config.max_degree + 1) * std::mem::size_of::<u32>();
        let node_bytes = vector_bytes + neighbor_bytes + inline_pq_bytes;

        Self {
            page_size: DEFAULT_PAGE_SIZE,
            vector_bytes,
            inline_pq_bytes,
            neighbor_bytes,
            node_bytes,
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
            cached_nodes: HashSet::new(),
            cached_pq_vectors: HashSet::new(),
            pq_lru: VecDeque::new(),
            stats: BeamSearchStats::default(),
        }
    }

    pub fn reset_stats(&mut self) {
        self.stats = BeamSearchStats::default();
    }

    pub fn max_reads_per_iteration(&self) -> usize {
        self.max_reads_per_iteration
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
struct SerializedPqEncoder {
    m: usize,
    k: usize,
    nbits: usize,
    dim: usize,
    sub_dim: usize,
    codebooks: Vec<f32>,
}

impl SerializedPqEncoder {
    fn from_encoder(encoder: &PqEncoder) -> Self {
        Self {
            m: encoder.m,
            k: encoder.k,
            nbits: encoder.nbits,
            dim: encoder.dim,
            sub_dim: encoder.sub_dim,
            codebooks: encoder.codebooks.clone(),
        }
    }

    fn into_encoder(self) -> PqEncoder {
        let mut encoder = PqEncoder::new(self.dim, self.m, self.k);
        encoder.nbits = self.nbits;
        encoder.sub_dim = self.sub_dim;
        encoder.codebooks = self.codebooks;
        encoder
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
    pq_encoder: Option<SerializedPqEncoder>,
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
                let page = std::sync::Arc::clone(arc);  // clone while borrow is live
                drop(arc);  // release immutable borrow
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

#[derive(Debug)]
struct DiskStorage {
    file_group: FileGroup,
    page_cache: PageCache,
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
    seen: HashSet<u32>,
    expanded: Vec<Candidate>,
    accepted: Vec<Candidate>,
}

impl AisaqScratch {
    fn new(capacity: usize) -> Self {
        Self {
            frontier: BinaryHeap::with_capacity(capacity),
            seen: HashSet::with_capacity(capacity.saturating_mul(2)),
            expanded: Vec::with_capacity(capacity),
            accepted: Vec::with_capacity(capacity),
        }
    }

    fn reset(&mut self) {
        self.frontier.clear();
        self.seen.clear();
        self.expanded.clear();
        self.accepted.clear();
    }
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
    pq_encoder: Option<PqEncoder>,
    pq_code_size: usize,
    entry_points: Vec<u32>,
    io_template: BeamSearchIO,
    trained: bool,
    node_count: usize,
    storage: Option<DiskStorage>,
    loaded_node_cache: Option<HashMap<u32, std::sync::Arc<LoadedNode>>>,
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
            entry_points: Vec::new(),
            io_template,
            trained: false,
            node_count: 0,
            storage: None,
            loaded_node_cache: None,
            scratch_pool: Mutex::new(Vec::new()),
        })
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
            let mut encoder = PqEncoder::new(self.dim, m, k);
            encoder.train(training_data, 25);
            self.pq_code_size = encoder.m;
            self.pq_encoder = Some(encoder);
            self.flash_layout.inline_pq_bytes = self.pq_code_size.max(self.config.inline_pq);
            self.flash_layout.node_bytes = self.flash_layout.vector_bytes
                + self.flash_layout.neighbor_bytes
                + self.flash_layout.inline_pq_bytes;
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

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let batch_size = 512usize.min(num_vectors).max(1);
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
                            let cands = self.vamana_build_search(vector, l, graph_size_snapshot);
                            cands
                                .iter()
                                .take(stride)
                                .map(|(id, _)| *id)
                                .collect()
                        };

                        let inline_pq = self
                            .pq_encoder
                            .as_ref()
                            .map(|pq| pq.encode(vector))
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
                    self.vamana_build_search(vector, l, graph_size)
                        .into_iter()
                        .take(stride)
                        .map(|(id, _)| id)
                        .collect::<Vec<u32>>()
                };
                self.vectors.extend_from_slice(vector);
                let inline_pq = self
                    .pq_encoder
                    .as_ref()
                    .map(|pq| pq.encode(vector))
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
        self.refresh_entry_points();
        if self.config.warm_up {
            self.warm_up_cache();
        }
        Ok(())
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

    fn compute_build_max_degree(&self) -> usize {
        let slack = self.config.build_degree_slack_pct.max(100);
        self.config
            .max_degree
            .saturating_mul(slack)
            .div_ceil(100)
            .max(self.config.max_degree)
    }

    pub fn save<P: AsRef<Path>>(&self, root: P) -> Result<FileGroup> {
        let file_group = FileGroup::create(root)?;
        let metadata = AisaqMetadata {
            version: 2,
            config: self.config.clone(),
            metric_type: self.metric_type,
            dim: self.dim,
            flash_layout: self.flash_layout.clone(),
            pq_code_size: self.pq_code_size,
            entry_points: self.entry_points.clone(),
            trained: self.trained,
            node_count: self.len(),
            pq_encoder: self
                .pq_encoder
                .as_ref()
                .map(SerializedPqEncoder::from_encoder),
        };
        let metadata_bytes = bincode::serialize(&metadata)
            .map_err(|error| KnowhereError::Codec(error.to_string()))?;
        let mut metadata_file = File::create(file_group.metadata_path())?;
        metadata_file.write_all(&metadata_bytes)?;
        metadata_file.flush()?;

        let mut data_file = File::create(file_group.data_path())?;
        if let Some(storage) = &self.storage {
            let mut input = File::open(storage.file_group.data_path())?;
            std::io::copy(&mut input, &mut data_file)?;
        } else {
            for node_id in 0..self.node_ids.len() {
                let bytes = self.serialize_node(node_id as u32);
                data_file.write_all(&bytes)?;
            }
        }
        data_file.flush()?;

        Ok(file_group)
    }

    pub fn load<P: AsRef<Path>>(root: P) -> Result<Self> {
        let file_group = FileGroup::new(root);
        let mut metadata_bytes = Vec::new();
        File::open(file_group.metadata_path())?.read_to_end(&mut metadata_bytes)?;
        let metadata: AisaqMetadata = bincode::deserialize(&metadata_bytes)
            .map_err(|error| KnowhereError::Codec(error.to_string()))?;
        if metadata.version != 2 {
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
            pq_encoder: metadata.pq_encoder.map(SerializedPqEncoder::into_encoder),
            pq_code_size: metadata.pq_code_size,
            entry_points: metadata.entry_points,
            io_template,
            trained: metadata.trained,
            node_count: metadata.node_count,
            storage: Some(DiskStorage {
                file_group,
                page_cache,
            }),
            loaded_node_cache: None,
            scratch_pool: Mutex::new(Vec::new()),
        };
        if index.config.cache_all_on_load {
            index.prime_loaded_node_cache()?;
        }
        if index.pq_code_size > 0 {
            index.disk_pq_codes = index.load_disk_pq_codes()?;
        }
        Ok(index)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
        self.search_internal(query, k, None)
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        k: usize,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
        self.search_internal(query, k, Some(bitset))
    }

    #[cfg(feature = "parallel")]
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Result<SearchResult> {
        use rayon::prelude::*;
        let n_queries = queries.len() / self.dim;
        if n_queries == 0 {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
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
            .map(|encoder| encoder.build_distance_table(query));
        let pq_table_slice = pq_table.as_ref().map(|v| v.as_slice());

        let max_visit = self.compute_max_visit(k);
        let mut scratch = {
            let mut pool = self.scratch_pool.lock();
            pool.pop()
                .unwrap_or_else(|| AisaqScratch::new(max_visit.saturating_mul(2)))
        };
        scratch.reset();

        let outcome: Result<SearchResult> = (|| {
            for candidate in self.rank_entry_candidates(
                query,
                pq_table_slice,
                &mut io,
            )? {
                scratch.frontier.push(candidate);
            }

            while scratch.expanded.len() < max_visit {
                let candidate = match scratch.frontier.pop() {
                    Some(candidate) => candidate,
                    None => break,
                };

                if !scratch.seen.insert(candidate.node_id) {
                    continue;
                }

                scratch.expanded.push(candidate);
                if self.node_allowed(candidate.node_id, bitset) {
                    scratch.accepted.push(candidate);
                }

                let mut neighbor_scores = if let Some(node) = self.node_ref(candidate.node_id) {
                    io.record_node_read(candidate.node_id, self.flash_layout.node_bytes);
                    let mut scores = Vec::with_capacity(node.neighbors.len());
                    for &neighbor in node.neighbors {
                        if scratch.seen.contains(&neighbor) {
                            continue;
                        }
                        let score = if let Some(neighbor_node) = self.node_ref(neighbor) {
                            if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table_slice) {
                                if !neighbor_node.inline_pq.is_empty() {
                                    io.record_pq_read(neighbor, neighbor_node.inline_pq.len());
                                    encoder.compute_distance_with_table(table, neighbor_node.inline_pq)
                                } else {
                                    self.exact_distance(query, neighbor_node.vector)
                                }
                            } else {
                                self.exact_distance(query, neighbor_node.vector)
                            }
                        } else {
                            self.coarse_distance(neighbor, query, pq_table_slice, &mut io)?
                        };
                        scores.push(Candidate {
                            node_id: neighbor,
                            score,
                        });
                    }
                    scores
                } else {
                    let node = self.load_node(candidate.node_id, NodeAccessMode::Node, &mut io)?;
                    for &neighbor in &node.neighbors {
                        if scratch.seen.contains(&neighbor) {
                            continue;
                        }
                        self.prefetch_node(neighbor);
                    }
                    let mut scores = Vec::with_capacity(node.neighbors.len());
                    for &neighbor in &node.neighbors {
                        if scratch.seen.contains(&neighbor) {
                            continue;
                        }
                        let score =
                            self.coarse_distance(neighbor, query, pq_table_slice, &mut io)?;
                        scores.push(Candidate {
                            node_id: neighbor,
                            score,
                        });
                    }
                    scores
                };

                neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
                let expand_limit = self.compute_expand_limit(&io);
                for neighbor in neighbor_scores.into_iter().take(expand_limit) {
                    scratch.frontier.push(neighbor);
                }
            }

            if scratch.accepted.len() < k {
                self.fill_from_allowed_exact(
                    query,
                    bitset,
                    &scratch.seen,
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
                        let distance = self.exact_distance(query, &node.vector);
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
        })();

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
            return self.exact_scan_allowed_async(query, k, bitset, &mut io).await;
        }

        let pq_table = self
            .pq_encoder
            .as_ref()
            .map(|encoder| encoder.build_distance_table(query));

        let mut frontier = BinaryHeap::new();
        let mut seen = HashSet::new();
        let mut expanded = Vec::new();
        let mut accepted = Vec::new();

        for candidate in self
            .rank_entry_candidates_async(
                query,
                pq_table.as_ref().map(|v| v.as_slice()),
                &mut io,
            )
            .await?
        {
            frontier.push(candidate);
        }

        let max_visit = self.compute_max_visit(k);
        while expanded.len() < max_visit {
            let candidate = match frontier.pop() {
                Some(candidate) => candidate,
                None => break,
            };
            if !seen.insert(candidate.node_id) {
                continue;
            }
            expanded.push(candidate);
            if self.node_allowed(candidate.node_id, bitset) {
                accepted.push(candidate);
            }

            let node = self
                .load_node_async(candidate.node_id, NodeAccessMode::Node, &mut io)
                .await?;
            for &neighbor in &node.neighbors {
                if seen.contains(&neighbor) {
                    continue;
                }
                self.prefetch_node(neighbor);
            }
            let mut neighbor_scores = Vec::with_capacity(node.neighbors.len());
            for &neighbor in &node.neighbors {
                if seen.contains(&neighbor) {
                    continue;
                }
                let score = self
                    .coarse_distance_async(
                        neighbor,
                        query,
                        pq_table.as_ref().map(|v| v.as_slice()),
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

        if accepted.len() < k {
            self.fill_from_allowed_exact(query, bitset, &seen, k, &mut accepted, &mut io)?;
        }

        let rerank_pool = self.compute_rerank_pool_size(k, accepted.len());
        accepted.sort_by(|left, right| left.score.total_cmp(&right.score));
        let mut scored = Vec::with_capacity(rerank_pool);
        // TODO: true concurrent IO rerank requires refactoring BeamSearchIO to not require &mut
        for candidate in accepted.iter().take(rerank_pool) {
            let node = self.load_node(candidate.node_id, NodeAccessMode::None, &mut io)?;
            let distance = self.exact_distance(query, &node.vector);
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
    fn node_allowed(&self, node_id: u32, bitset: Option<&BitsetView>) -> bool {
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
            let distance = self.exact_distance(query, &node.vector);
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
            let distance = self.exact_distance(query, &node.vector);
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

    fn fill_from_allowed_exact(
        &self,
        query: &[f32],
        bitset: Option<&BitsetView>,
        seen: &HashSet<u32>,
        k: usize,
        accepted: &mut Vec<Candidate>,
        io: &mut BeamSearchIO,
    ) -> Result<()> {
        if accepted.len() >= k {
            return Ok(());
        }

        let mut in_accepted = HashSet::with_capacity(accepted.len() * 2 + 1);
        for c in accepted.iter() {
            in_accepted.insert(c.node_id);
        }

        for node_id in 0..self.len() as u32 {
            if seen.contains(&node_id) || in_accepted.contains(&node_id) {
                continue;
            }
            if !self.node_allowed(node_id, bitset) {
                continue;
            }
            let node = self.load_node(node_id, NodeAccessMode::None, io)?;
            let score = self.exact_distance(query, &node.vector);
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
        io: &mut BeamSearchIO,
    ) -> Result<Vec<Candidate>> {
        let mut ranked = Vec::new();
        for &entry in &self.entry_points {
            let score = self.coarse_distance(entry, query, pq_table, io)?;
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
        io: &mut BeamSearchIO,
    ) -> Result<Vec<Candidate>> {
        let mut ranked = Vec::new();
        for &entry in &self.entry_points {
            let score = self
                .coarse_distance_async(entry, query, pq_table, io)
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
            uses_mmap_backed_pages: self.storage.is_some(),
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
        self.storage.as_ref().map(|storage| &storage.file_group)
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
        let data_path = storage.file_group.data_path();
        let file = OpenOptions::new().read(true).open(data_path)?;
        let offset = node_id as usize * self.flash_layout.node_bytes;
        let mut bytes = vec![0u8; self.flash_layout.node_bytes];

        let mut ring = IoUring::new(2)
            .map_err(|e| KnowhereError::Storage(format!("io_uring init failed: {e}")))?;
        let entry = opcode::Read::new(
            types::Fd(file.as_raw_fd()),
            bytes.as_mut_ptr(),
            bytes.len() as u32,
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
        if res as usize != self.flash_layout.node_bytes {
            return Err(KnowhereError::Storage(format!(
                "io_uring short read: got {} expected {}",
                res, self.flash_layout.node_bytes
            )));
        }

        let pages_loaded = self
            .flash_layout
            .node_bytes
            .div_ceil(self.flash_layout.page_size.max(1));
        Ok((bytes, pages_loaded))
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
        io: &mut BeamSearchIO,
    ) -> Result<f32> {
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

        Ok(self.exact_distance(query, &node.vector))
    }

    #[inline]
    fn prefetch_node(&self, node_id: u32) {
        let Some(storage) = &self.storage else {
            return;
        };
        if node_id as usize >= self.len() {
            return;
        }
        let offset = node_id as usize * self.flash_layout.node_bytes;
        let _ = storage
            .page_cache
            .prefetch(offset, self.flash_layout.node_bytes);
    }

    async fn coarse_distance_async(
        &self,
        node_id: u32,
        query: &[f32],
        pq_table: Option<&[f32]>,
        io: &mut BeamSearchIO,
    ) -> Result<f32> {
        let node = self.load_node_async(node_id, NodeAccessMode::Pq, io).await?;
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            let pq = &node.inline_pq;
            if !pq.is_empty() {
                return Ok(encoder.compute_distance_with_table(table, pq));
            }
        }
        Ok(self.exact_distance(query, &node.vector))
    }

    fn exact_distance(&self, query: &[f32], vector: &[f32]) -> f32 {
        match self.metric_type {
            MetricType::L2 => simd::l2_distance(query, vector),
            MetricType::Ip => -dot(query, vector),
            MetricType::Cosine => 1.0 - cosine_similarity(query, vector),
            MetricType::Hamming => f32::MAX,
        }
    }

    /// Build-time beam search over currently built prefix graph.
    /// Returns up to `l` candidates sorted by distance ascending.
    fn vamana_build_search(&self, query: &[f32], l: usize, graph_size: usize) -> Vec<(u32, f32)> {
        if graph_size == 0 || l == 0 {
            return Vec::new();
        }
        // Build PQ distance table if available (ADC lookup — 4-8x faster than exact distance)
        let pq_table: Option<Vec<f32>> = if self.pq_code_size > 0 && !self.node_pq_codes.is_empty() {
            self.pq_encoder.as_ref().map(|pq| pq.build_distance_table(query))
        } else {
            None
        };
        let stride = self.flat_stride.max(1);
        let start_node = self
            .entry_points
            .first()
            .copied()
            .unwrap_or(0) as usize;
        let start_node = start_node.min(graph_size - 1);
        let start_dist = if let Some(ref table) = pq_table {
            let code_start = start_node * self.pq_code_size;
            let code_end = code_start + self.pq_code_size;
            if code_end <= self.node_pq_codes.len() {
                if let Some(ref pq) = self.pq_encoder {
                    pq.compute_distance_with_table(table, &self.node_pq_codes[code_start..code_end])
                } else {
                    self.exact_distance(query, self.node_vector(start_node))
                }
            } else {
                self.exact_distance(query, self.node_vector(start_node))
            }
        } else {
            self.exact_distance(query, self.node_vector(start_node))
        };

        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::with_capacity(l.saturating_mul(2));
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(l);

        frontier.push(Candidate {
            node_id: start_node as u32,
            score: start_dist,
        });
        visited.insert(start_node as u32);

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
                let nb_dist = if let Some(ref table) = pq_table {
                    let code_start = nb as usize * self.pq_code_size;
                    let code_end = code_start + self.pq_code_size;
                    if code_end <= self.node_pq_codes.len() {
                        if let Some(ref pq) = self.pq_encoder {
                            pq.compute_distance_with_table(table, &self.node_pq_codes[code_start..code_end])
                        } else {
                            self.exact_distance(query, self.node_vector(nb as usize))
                        }
                    } else {
                        // node not yet encoded (shouldn't happen, but safe fallback)
                        self.exact_distance(query, self.node_vector(nb as usize))
                    }
                } else {
                    self.exact_distance(query, self.node_vector(nb as usize))
                };
                frontier.push(Candidate {
                    node_id: nb,
                    score: nb_dist,
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

    fn select_neighbors_with_random(
        &self,
        node_id: usize,
        vector: &[f32],
        max_degree: usize,
    ) -> Vec<u32> {
        let mut selected = self.select_neighbors(vector, max_degree);
        if self.config.random_init_edges == 0 || node_id == 0 || self.node_ids.is_empty() {
            return selected;
        }

        let limit = node_id.min(self.node_ids.len());
        if limit == 0 {
            return selected;
        }

        let sample = self.config.random_init_edges.min(limit);
        let seed = self.config.random_seed ^ ((node_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pool: Vec<usize> = (0..limit).collect();
        pool.shuffle(&mut rng);

        for idx in pool.into_iter().take(sample) {
            let rid = idx as u32;
            if !selected.contains(&rid) {
                selected.push(rid);
            }
        }

        selected.sort_by(|&a, &b| {
            self.exact_distance(vector, self.node_vector(a as usize))
                .total_cmp(&self.exact_distance(vector, self.node_vector(b as usize)))
        });
        selected.truncate(max_degree.min(selected.len()));
        selected
    }

    #[cfg(test)]
    fn link_back(&mut self, neighbor: u32, node_id: u32) {
        self.link_back_with_limit(neighbor, node_id, self.config.max_degree);
    }

    fn link_back_with_limit(&mut self, neighbor: u32, node_id: u32, degree_limit: usize) {
        let neighbor_idx = neighbor as usize;
        if neighbor_idx >= self.node_ids.len() {
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
        if self.node_ids.len() > 50_000 {
            // Build-time fast path on very large graphs: when reverse adjacency is already
            // full, skip costly O(R log R) re-pruning to keep insertion complexity near linear.
            return;
        }

        let mut neighbor_list: Vec<u32> = current.to_vec();
        neighbor_list.push(node_id);
        let anchor = self.node_vector(neighbor_idx).to_vec();
        let mut scored: Vec<(u32, f32)> = neighbor_list
            .iter()
            .map(|&nb| (nb, self.exact_distance(&anchor, self.node_vector(nb as usize))))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(effective_limit);
        let new_count = scored.len();
        for (i, (nb, _)) in scored.into_iter().enumerate() {
            self.node_neighbor_ids[start + i] = nb;
        }
        for i in new_count..stride {
            self.node_neighbor_ids[start + i] = 0;
        }
        self.node_neighbor_counts[neighbor_idx] = new_count as u32;
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

    fn warm_up_cache(&mut self) {
        for &entry in &self.entry_points {
            self.io_template.cache_node(entry);
            let pq_size = self.pq_code_size.max(1);
            let pq_start = entry as usize * pq_size;
            let has_pq = pq_start + pq_size <= self.node_pq_codes.len();
            if has_pq {
                self.io_template.cache_pq_vector(entry);
            }
        }
    }

    fn materialize_storage(&mut self) -> Result<()> {
        if self.storage.is_none() {
            return Ok(());
        }

        let node_count = self.node_count;
        let mut vectors = Vec::with_capacity(node_count * self.dim);
        let mut node_data: Vec<std::sync::Arc<LoadedNode>> = Vec::with_capacity(node_count);
        let mut io = self.io_template.clone();
        for node_id in 0..node_count {
            let loaded = self.load_node(node_id as u32, NodeAccessMode::None, &mut io)?;
            vectors.extend_from_slice(&loaded.vector);
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
        self.loaded_node_cache = None;
        self.disk_pq_codes.clear();
        Ok(())
    }

    fn load_disk_pq_codes(&self) -> Result<Vec<u8>> {
        let Some(storage) = &self.storage else {
            return Ok(Vec::new());
        };
        if self.pq_code_size == 0 || self.node_count == 0 {
            return Ok(Vec::new());
        }

        let bytes = fs::read(storage.file_group.data_path())?;
        let expected = self.node_count.saturating_mul(self.flash_layout.node_bytes);
        if bytes.len() < expected {
            return Err(KnowhereError::Codec(format!(
                "node file too small for pq extraction: got {} expected at least {}",
                bytes.len(),
                expected
            )));
        }

        let mut out = vec![0u8; self.node_count * self.pq_code_size];
        let inline_offset = self.flash_layout.vector_bytes + self.flash_layout.neighbor_bytes;
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

    fn prime_loaded_node_cache(&mut self) -> Result<()> {
        let Some(storage) = &self.storage else {
            return Ok(());
        };
        let mut cache = HashMap::with_capacity(self.node_count);
        for node_id in 0..self.node_count {
            let offset = node_id * self.flash_layout.node_bytes;
            let page_read = storage
                .page_cache
                .read(offset, self.flash_layout.node_bytes)?;
            let loaded = self.deserialize_node(node_id as u32, &page_read.bytes)?;
            cache.insert(node_id as u32, std::sync::Arc::new(loaded));
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
                    NodeAccessMode::Node => io.record_node_access(node_id, self.flash_layout.node_bytes, 0),
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
        for value in self.node_vector(id) {
            bytes.extend_from_slice(&value.to_le_bytes());
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
        let mut vector = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            let raw = bytes
                .get(cursor..cursor + 4)
                .ok_or_else(|| KnowhereError::Codec("truncated vector payload".to_string()))?;
            vector.push(f32::from_le_bytes(raw.try_into().unwrap()));
            cursor += 4;
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

        Ok(LoadedNode {
            id: node_id as i64,
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
            let result =
                PQFlashIndex::search(self, q, top_k).map_err(|e| IndexError::Unsupported(e.to_string()))?;
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
    fn aisaq_random_initial_neighbors_are_seeded() {
        let config = AisaqConfig {
            max_degree: 4,
            random_init_edges: 2,
            random_seed: 7,
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
        index.train(&vectors).unwrap();
        index.add(&vectors[..8]).unwrap(); // add first 4 nodes
        let query = [4.0f32, 0.0f32];

        let n1 = index.select_neighbors_with_random(4, &query, 4);
        let n2 = index.select_neighbors_with_random(4, &query, 4);
        assert_eq!(n1, n2, "random init neighbors should be deterministic under seed");
    }

    #[test]
    fn aisaq_link_back_keeps_closer_reverse_neighbors() {
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
        assert_eq!(cnt, 2);
        assert!(nbrs.contains(&2));
        assert!(nbrs.contains(&3));
        assert!(
            !nbrs.contains(&1),
            "far neighbor should be evicted when reverse list overflows"
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

    fn run_batch_search(index: &PQFlashIndex, queries: &[f32], dim: usize, k: usize) -> SearchResult {
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
    fn test_pqflash_basic_search() {
        let dim = 16usize;
        let n = 200usize;
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

        let results = run_batch_search(&index, &queries, dim, k);
        assert_eq!(results.ids.len(), nq * k);
        assert!(results.ids.iter().all(|&id| id >= 0));
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

        assert!(loaded.vectors.is_empty(), "loaded index should use disk-backed path");
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

        assert!(disk_index.vectors.is_empty(), "disk path should not materialize vectors");
        assert!(disk_index.storage.is_some(), "disk path should keep disk storage");

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
}

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

#[cfg(test)]
use std::future::Future;

use crate::api::{IndexConfig, KnowhereError, MetricType, Result, SearchResult};
use crate::faiss::pq::PqEncoder;
use crate::simd;
use memmap2::Mmap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

const DEFAULT_PAGE_SIZE: usize = 4096;
const DEFAULT_PQ_K: usize = 256;

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
    pub warm_up: bool,
    pub filter_threshold: f32,
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
            rearrange: false,
            inline_pq: 0,
            num_entry_points: 1,
            rerank_expand_pct: 200,
            warm_up: false,
            filter_threshold: -1.0,
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
            num_entry_points: params.disk_num_entry_points.unwrap_or(1).clamp(1, 64),
            rerank_expand_pct: params.disk_rerank_expand_pct.unwrap_or(200).clamp(100, 400),
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
    cached_nodes: HashSet<u32>,
    cached_pq_vectors: HashSet<u32>,
    stats: BeamSearchStats,
}

impl BeamSearchIO {
    pub fn new(layout: &FlashLayout, config: &AisaqConfig) -> Self {
        Self {
            page_size: layout.page_size.max(1),
            max_reads_per_iteration: config.beamwidth.max(1),
            cached_nodes: HashSet::new(),
            cached_pq_vectors: HashSet::new(),
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
        self.cached_pq_vectors.insert(node_id);
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
struct FlashNode {
    id: i64,
    vector_offset: usize,
    neighbors: Vec<u32>,
    inline_pq: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LoadedNode {
    id: i64,
    vector: Vec<f32>,
    neighbors: Vec<u32>,
    inline_pq: Vec<u8>,
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
struct PageCacheState {
    pages: HashMap<usize, Vec<u8>>,
    lru: VecDeque<usize>,
    stats: PageCacheStats,
}

#[derive(Debug)]
pub struct PageCache {
    page_size: usize,
    capacity_pages: usize,
    mmap: Mmap,
    state: Mutex<PageCacheState>,
}

impl PageCache {
    pub fn open<P: AsRef<Path>>(path: P, page_size: usize, cache_bytes: usize) -> Result<Self> {
        let file = OpenOptions::new().read(true).open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let capacity_pages = cache_bytes.div_ceil(page_size.max(1)).max(1);

        Ok(Self {
            page_size: page_size.max(1),
            capacity_pages,
            mmap,
            state: Mutex::new(PageCacheState {
                pages: HashMap::new(),
                lru: VecDeque::new(),
                stats: PageCacheStats::default(),
            }),
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
        let mut state = self.state.lock();
        state.stats.requests += 1;

        let mut pages_loaded = 0usize;
        let mut page_buffers = Vec::with_capacity(end_page - start_page + 1);
        for page_id in start_page..=end_page {
            if let Some(page) = state.pages.get(&page_id).cloned() {
                state.stats.page_hits += 1;
                touch_lru(&mut state.lru, page_id);
                page_buffers.push((page_id, page));
                continue;
            }

            state.stats.page_misses += 1;
            pages_loaded += 1;
            let page_start = page_id * self.page_size;
            let page_end = (page_start + self.page_size).min(self.mmap.len());
            let page = self.mmap[page_start..page_end].to_vec();
            state.pages.insert(page_id, page.clone());
            state.lru.push_back(page_id);
            evict_if_needed(&mut state, self.capacity_pages);
            page_buffers.push((page_id, page));
        }
        drop(state);

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

    pub fn stats(&self) -> PageCacheStats {
        self.state.lock().stats.clone()
    }
}

fn touch_lru(lru: &mut VecDeque<usize>, page_id: usize) {
    if let Some(position) = lru.iter().position(|&id| id == page_id) {
        lru.remove(position);
    }
    lru.push_back(page_id);
}

fn evict_if_needed(state: &mut PageCacheState, capacity_pages: usize) {
    while state.pages.len() > capacity_pages {
        if let Some(oldest) = state.lru.pop_front() {
            if state.pages.remove(&oldest).is_some() {
                state.stats.evictions += 1;
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

/// PQ Flash index skeleton for DiskANN AISAQ.
pub struct PQFlashIndex {
    config: AisaqConfig,
    metric_type: MetricType,
    dim: usize,
    flash_layout: FlashLayout,
    vectors: Vec<f32>,
    nodes: Vec<FlashNode>,
    pq_encoder: Option<PqEncoder>,
    pq_code_size: usize,
    entry_points: Vec<u32>,
    io_template: BeamSearchIO,
    trained: bool,
    node_count: usize,
    storage: Option<DiskStorage>,
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
            nodes: Vec::new(),
            pq_encoder: None,
            pq_code_size: 0,
            entry_points: Vec::new(),
            io_template,
            trained: false,
            node_count: 0,
            storage: None,
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
        self.materialize_storage()?;
        self.validate_vectors(vectors)?;
        if vectors.is_empty() {
            return Ok(());
        }

        if !self.trained {
            self.train(vectors)?;
        }

        let num_vectors = vectors.len() / self.dim;
        for row in 0..num_vectors {
            let start = row * self.dim;
            let end = start + self.dim;
            let vector = &vectors[start..end];
            let neighbors = self.select_neighbors(vector, self.config.max_degree);
            let vector_offset = self.vectors.len();
            self.vectors.extend_from_slice(vector);

            let node_id = self.nodes.len() as u32;
            let inline_pq = self
                .pq_encoder
                .as_ref()
                .map(|pq| pq.encode(vector))
                .unwrap_or_default();

            let node = FlashNode {
                id: node_id as i64,
                vector_offset,
                neighbors: neighbors.clone(),
                inline_pq,
            };
            self.nodes.push(node);

            for neighbor in neighbors {
                self.link_back(neighbor, node_id);
            }
        }

        self.node_count = self.nodes.len();
        self.refresh_entry_points();
        if self.config.warm_up {
            self.warm_up_cache();
        }
        Ok(())
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
            for node_id in 0..self.nodes.len() {
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

        Ok(Self {
            config: metadata.config,
            metric_type: metadata.metric_type,
            dim: metadata.dim,
            flash_layout: metadata.flash_layout,
            vectors: Vec::new(),
            nodes: Vec::new(),
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
        })
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult> {
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
            .map(|encoder| encoder.build_distance_table(query));

        let mut frontier = BinaryHeap::new();
        let mut seen = HashSet::new();
        let mut expanded = Vec::new();

        for &entry in &self.entry_points {
            let score = self.coarse_distance(entry, query, pq_table.as_ref(), &mut io)?;
            frontier.push(Candidate {
                node_id: entry,
                score,
            });
        }

        let max_visit = self.config.search_list_size.max(k).min(self.len());

        while expanded.len() < max_visit {
            let candidate = match frontier.pop() {
                Some(candidate) => candidate,
                None => break,
            };

            if !seen.insert(candidate.node_id) {
                continue;
            }

            expanded.push(candidate);

            let node = self.load_node(candidate.node_id, NodeAccessMode::Node, &mut io)?;
            let mut neighbor_scores = Vec::with_capacity(node.neighbors.len());
            for &neighbor in &node.neighbors {
                if seen.contains(&neighbor) {
                    continue;
                }
                let score = self.coarse_distance(neighbor, query, pq_table.as_ref(), &mut io)?;
                neighbor_scores.push(Candidate {
                    node_id: neighbor,
                    score,
                });
            }

            neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
            for neighbor in neighbor_scores
                .into_iter()
                .take(io.max_reads_per_iteration())
            {
                frontier.push(neighbor);
            }
        }

        let rerank_pool = self.compute_rerank_pool_size(k, expanded.len());
        expanded.sort_by(|left, right| left.score.total_cmp(&right.score));
        let mut scored: Vec<(i64, f32)> = expanded
            .iter()
            .take(rerank_pool)
            .map(|candidate| {
                let node = self.load_node(candidate.node_id, NodeAccessMode::None, &mut io)?;
                let distance = self.exact_distance(query, &node.vector);
                Ok((node.id, distance))
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
    }

    pub async fn search_async(&self, query: &[f32], k: usize) -> Result<SearchResult> {
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
            .map(|encoder| encoder.build_distance_table(query));

        let mut frontier = BinaryHeap::new();
        let mut seen = HashSet::new();
        let mut expanded = Vec::new();

        for &entry in &self.entry_points {
            let score = self
                .coarse_distance_async(entry, query, pq_table.as_ref(), &mut io)
                .await?;
            frontier.push(Candidate {
                node_id: entry,
                score,
            });
        }

        let max_visit = self.config.search_list_size.max(k).min(self.len());
        while expanded.len() < max_visit {
            let candidate = match frontier.pop() {
                Some(candidate) => candidate,
                None => break,
            };
            if !seen.insert(candidate.node_id) {
                continue;
            }
            expanded.push(candidate);

            let node = self
                .load_node_async(candidate.node_id, NodeAccessMode::Node, &mut io)
                .await?;
            let mut neighbor_scores = Vec::with_capacity(node.neighbors.len());
            for &neighbor in &node.neighbors {
                if seen.contains(&neighbor) {
                    continue;
                }
                let score = self
                    .coarse_distance_async(neighbor, query, pq_table.as_ref(), &mut io)
                    .await?;
                neighbor_scores.push(Candidate {
                    node_id: neighbor,
                    score,
                });
            }
            neighbor_scores.sort_by(|left, right| left.score.total_cmp(&right.score));
            for neighbor in neighbor_scores
                .into_iter()
                .take(io.max_reads_per_iteration())
            {
                frontier.push(neighbor);
            }
        }

        let rerank_pool = self.compute_rerank_pool_size(k, expanded.len());
        expanded.sort_by(|left, right| left.score.total_cmp(&right.score));
        let mut scored = Vec::with_capacity(rerank_pool);
        for candidate in expanded.iter().take(rerank_pool) {
            let node = self
                .load_node_async(candidate.node_id, NodeAccessMode::None, &mut io)
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

    #[inline]
    fn compute_rerank_pool_size(&self, k: usize, expanded_len: usize) -> usize {
        if expanded_len == 0 {
            return 0;
        }
        let target = k
            .saturating_mul(self.config.rerank_expand_pct)
            .saturating_add(99)
            / 100;
        target.max(k).min(expanded_len).max(1)
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
    ) -> Result<LoadedNode> {
        // For now, delegate to sync implementation
        // Full io_uring implementation would use IoUring::submit() here
        self.load_node(node_id, access_mode, io)
    }

    #[cfg(not(all(feature = "async-io", target_os = "linux")))]
    pub async fn load_node_async(
        &self,
        node_id: u32,
        access_mode: NodeAccessMode,
        io: &mut BeamSearchIO,
    ) -> Result<LoadedNode> {
        // Fallback to sync on non-Linux platforms
        self.load_node(node_id, access_mode, io)
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
        pq_table: Option<&Vec<Vec<f32>>>,
        io: &mut BeamSearchIO,
    ) -> Result<f32> {
        let node = self.load_node(node_id, NodeAccessMode::Pq, io)?;
        if let (Some(encoder), Some(table)) = (&self.pq_encoder, pq_table) {
            let pq = &node.inline_pq;
            if !pq.is_empty() {
                return Ok(encoder.compute_distance_with_table(table, pq));
            }
        }

        Ok(self.exact_distance(query, &node.vector))
    }

    async fn coarse_distance_async(
        &self,
        node_id: u32,
        query: &[f32],
        pq_table: Option<&Vec<Vec<f32>>>,
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

    fn select_neighbors(&self, vector: &[f32], max_degree: usize) -> Vec<u32> {
        let mut scored: Vec<(u32, f32)> = (0..self.nodes.len())
            .map(|node_id| {
                (
                    node_id as u32,
                    self.exact_distance(
                        vector,
                        &self.vectors[self.nodes[node_id].vector_offset
                            ..self.nodes[node_id].vector_offset + self.dim],
                    ),
                )
            })
            .collect();

        scored.sort_by(|left, right| left.1.total_cmp(&right.1));
        scored.truncate(max_degree.min(scored.len()));
        scored.into_iter().map(|(node_id, _)| node_id).collect()
    }

    fn link_back(&mut self, neighbor: u32, node_id: u32) {
        let neighbor_list = &mut self.nodes[neighbor as usize].neighbors;
        if !neighbor_list.contains(&node_id) {
            neighbor_list.push(node_id);
        }
        if neighbor_list.len() > self.config.max_degree {
            neighbor_list.sort_unstable();
            neighbor_list.truncate(self.config.max_degree);
        }
    }

    fn refresh_entry_points(&mut self) {
        let count = self.config.num_entry_points.min(self.len()).max(1);
        self.entry_points = (0..count as u32).collect();
    }

    fn warm_up_cache(&mut self) {
        for &entry in &self.entry_points {
            self.io_template.cache_node(entry);
            if !self.nodes[entry as usize].inline_pq.is_empty() {
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
        let mut nodes = Vec::with_capacity(node_count);
        let mut io = self.io_template.clone();
        for node_id in 0..node_count {
            let loaded = self.load_node(node_id as u32, NodeAccessMode::None, &mut io)?;
            let vector_offset = vectors.len();
            vectors.extend_from_slice(&loaded.vector);
            nodes.push(FlashNode {
                id: loaded.id,
                vector_offset,
                neighbors: loaded.neighbors,
                inline_pq: loaded.inline_pq,
            });
        }

        self.vectors = vectors;
        self.nodes = nodes;
        self.storage = None;
        Ok(())
    }

    fn load_node(
        &self,
        node_id: u32,
        access_mode: NodeAccessMode,
        io: &mut BeamSearchIO,
    ) -> Result<LoadedNode> {
        if node_id as usize >= self.len() {
            return Err(KnowhereError::InvalidArg(format!(
                "node_id {} out of bounds for {} nodes",
                node_id,
                self.len()
            )));
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
            return Ok(loaded);
        }

        let node = &self.nodes[node_id as usize];
        match access_mode {
            NodeAccessMode::None => {}
            NodeAccessMode::Node => io.record_node_read(node_id, self.flash_layout.node_bytes),
            NodeAccessMode::Pq => {
                if !node.inline_pq.is_empty() {
                    io.record_pq_read(node_id, node.inline_pq.len());
                }
            }
        }

        let start = node.vector_offset;
        Ok(LoadedNode {
            id: node.id,
            vector: self.vectors[start..start + self.dim].to_vec(),
            neighbors: node.neighbors.clone(),
            inline_pq: node.inline_pq.clone(),
        })
    }

    fn serialize_node(&self, node_id: u32) -> Vec<u8> {
        let node = &self.nodes[node_id as usize];
        let mut bytes = Vec::with_capacity(self.flash_layout.node_bytes);
        let start = node.vector_offset;
        for value in &self.vectors[start..start + self.dim] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let degree = node.neighbors.len().min(self.config.max_degree) as u32;
        bytes.extend_from_slice(&degree.to_le_bytes());
        for &neighbor in node.neighbors.iter().take(self.config.max_degree) {
            bytes.extend_from_slice(&neighbor.to_le_bytes());
        }
        for _ in node.neighbors.len().min(self.config.max_degree)..self.config.max_degree {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }

        let inline_len = self.flash_layout.inline_pq_bytes.min(node.inline_pq.len());
        bytes.extend_from_slice(&node.inline_pq[..inline_len]);
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
    fn aisaq_config_maps_rerank_expand_pct() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 8,
            data_type: DataType::Float,
            params: IndexParams {
                disk_rerank_expand_pct: Some(250),
                disk_num_entry_points: Some(3),
                disk_pq_dims: Some(4),
                ..Default::default()
            },
        };
        let mapped = AisaqConfig::from_index_config(&config);
        assert_eq!(mapped.rerank_expand_pct, 250);
        assert_eq!(mapped.num_entry_points, 3);
        assert_eq!(mapped.disk_pq_dims, 4);
    }

    #[test]
    fn aisaq_rerank_pool_size_is_bounded() {
        let index = PQFlashIndex::new(AisaqConfig::default(), MetricType::L2, 8).unwrap();
        assert_eq!(index.compute_rerank_pool_size(10, 0), 0);
        assert_eq!(index.compute_rerank_pool_size(10, 5), 5);
        assert_eq!(index.compute_rerank_pool_size(10, 30), 20);
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
}

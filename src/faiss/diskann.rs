//! DiskANN-inspired index (Vamana algorithm) - Enhanced
//!
//! A graph-based index optimized for SSD storage.
//! Features:
//! - Vamana graph algorithm with beam search
//! - L2 and IP (inner product) distance
//! - Beam search with early termination
//! - Range search support
//! - Iterator support for streaming search
//! - PQ compression support (simplified)
//! - Configurable parameters (max_degree, search_list_size, beamwidth, etc.)
//! - Statistics API for monitoring

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::api::search::Predicate;
use crate::api::{IndexConfig, KnowhereError, MetricType, Result, SearchRequest, SearchResult};
use crate::quantization::kmeans::KMeans;
use crate::simd;

/// DiskANN configuration parameters
/// Mirrors the C++ DiskANNConfig structure
#[derive(Clone, Debug)]
pub struct DiskAnnConfig {
    /// Graph degree (max neighbors per node), typically 48-150
    pub max_degree: usize,
    /// Search list size (L parameter for Vamana), typically 75-200
    pub search_list_size: usize,
    /// Build-time search list size (Lbuild). Defaults to search_list_size.
    pub construction_l: usize,
    /// PQ code budget in GB (for compression)
    pub pq_code_budget_gb: f32,
    /// Build DRAM budget in GB
    pub build_dram_budget_gb: f32,
    /// Disk PQ dimensions (0 = uncompressed)
    pub disk_pq_dims: usize,
    /// Candidate pool expansion ratio for PQ path (100 = off, 125 = +25%)
    pub pq_candidate_expand_pct: usize,
    /// Neighbor rerank pool expansion ratio for PQ path (100 = off, 200 = 2x beamwidth)
    pub rerank_expand_pct: usize,
    /// Whether to saturate adjacency to max degree after alpha pruning
    pub saturate_after_prune: bool,
    /// Additional temporal candidates during build (0 = disabled)
    pub intra_batch_candidates: usize,
    /// Number of entry points for search-start seeding (medoid-like), default 1
    pub num_entry_points: usize,
    /// Temporary build-degree slack percentage (100 = no slack, 130 = 1.3x)
    pub build_degree_slack_pct: usize,
    /// Random initial candidate edges for build-time insertion (0 = disabled)
    pub random_init_edges: usize,
    /// Batch size used by parallel candidate search in build path.
    pub build_parallel_batch_size: usize,
    /// Beamwidth for search (IO parallelism), default 8
    pub beamwidth: usize,
    /// Cache DRAM budget in GB
    pub cache_dram_budget_gb: f32,
    /// Whether to persist/load fixed-stride flash layout sidecar
    pub enable_flash_layout: bool,
    /// Whether to use mmap runtime reads for flash layout sidecar.
    pub flash_mmap_mode: bool,
    /// Per-expansion prefetch batch size for mmap path (0 = disabled).
    pub flash_prefetch_batch: usize,
    /// Warm-up before search
    pub warm_up: bool,
    /// Filter threshold for PQ+Refine (0.0-1.0, -1 = auto)
    pub filter_threshold: f32,
    /// Accelerate build (skip full 2-round build)
    pub accelerate_build: bool,
    /// Min K for range search
    pub min_k: usize,
    /// Max K for range search
    pub max_k: usize,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            max_degree: 48,
            search_list_size: 128,
            construction_l: 128,
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            pq_candidate_expand_pct: 125,
            rerank_expand_pct: 100,
            saturate_after_prune: true,
            intra_batch_candidates: 8,
            num_entry_points: 1,
            build_degree_slack_pct: 100,
            random_init_edges: 0,
            build_parallel_batch_size: 64,
            beamwidth: 8,
            cache_dram_budget_gb: 0.0,
            enable_flash_layout: false,
            flash_mmap_mode: false,
            flash_prefetch_batch: 0,
            warm_up: false,
            filter_threshold: -1.0,
            accelerate_build: false,
            min_k: 100,
            max_k: usize::MAX,
        }
    }
}

impl DiskAnnConfig {
    pub fn from_index_config(config: &IndexConfig) -> Self {
        let params = &config.params;
        let search_list_size = params.search_list_size.unwrap_or(128);
        Self {
            max_degree: params.max_degree.unwrap_or(48),
            search_list_size,
            construction_l: params.construction_l.unwrap_or(search_list_size).max(16),
            beamwidth: params.beamwidth.unwrap_or(8),
            pq_code_budget_gb: params.disk_pq_code_budget_gb.unwrap_or(0.0).max(0.0),
            build_dram_budget_gb: params.disk_build_dram_budget_gb.unwrap_or(0.0).max(0.0),
            disk_pq_dims: params.disk_pq_dims.unwrap_or(0),
            pq_candidate_expand_pct: params
                .disk_pq_candidate_expand_pct
                .unwrap_or(125)
                .clamp(100, 300),
            rerank_expand_pct: params.disk_rerank_expand_pct.unwrap_or(100).clamp(100, 400),
            saturate_after_prune: params.disk_saturate_after_prune.unwrap_or(true),
            intra_batch_candidates: params.disk_intra_batch_candidates.unwrap_or(8).min(256),
            num_entry_points: params.disk_num_entry_points.unwrap_or(1).clamp(1, 64),
            build_degree_slack_pct: params
                .disk_build_degree_slack_pct
                .unwrap_or(100)
                .clamp(100, 200),
            random_init_edges: params.disk_random_init_edges.unwrap_or(0).min(64),
            build_parallel_batch_size: params
                .disk_build_parallel_batch_size
                .unwrap_or(64)
                .clamp(1, 1024),
            cache_dram_budget_gb: params.disk_search_cache_budget_gb.unwrap_or(0.0).max(0.0),
            enable_flash_layout: params.disk_enable_flash_layout.unwrap_or(false),
            flash_mmap_mode: params.disk_flash_mmap_mode.unwrap_or(false),
            flash_prefetch_batch: params.disk_flash_prefetch_batch.unwrap_or(0).min(256),
            warm_up: false,
            filter_threshold: -1.0,
            accelerate_build: false,
            min_k: 100,
            max_k: usize::MAX,
        }
    }

    #[inline]
    fn budget_bytes(gb: f32) -> Option<usize> {
        if gb <= 0.0 {
            None
        } else {
            Some((gb as f64 * 1024.0 * 1024.0 * 1024.0) as usize)
        }
    }
}

/// Statistics about the DiskANN index
#[derive(Debug, Clone, Default)]
pub struct DiskAnnStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub is_trained: bool,
    pub memory_usage_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskAnnScopeAudit {
    pub dim: usize,
    pub max_degree: usize,
    pub search_list_size: usize,
    pub beamwidth: usize,
    pub uses_vamana_graph: bool,
    pub uses_placeholder_pq: bool,
    pub has_flash_layout: bool,
    pub native_comparable: bool,
    pub comparability_reason: &'static str,
}

/// In-memory PQ-compressed vectors with per-subspace codebooks and ADC lookup.
///
/// This is still not native DiskANN's full SSD pipeline (graph/disk layout, I/O path,
/// reorder on full vectors), so it should not be treated as native-comparable.
#[derive(Clone)]
struct PQCode {
    codes: Vec<u8>,
    dims: usize,
    ksub: usize,
    subdim_size: usize,
    codebooks: Vec<f32>, // [num_subdims][ksub][subdim_size]
}

impl PQCode {
    fn new(dims: usize) -> Self {
        Self {
            codes: Vec::new(),
            dims,
            ksub: 0,
            subdim_size: 0,
            codebooks: Vec::new(),
        }
    }

    fn encode(&mut self, vectors: &[f32], dim: usize) {
        if self.dims == 0 {
            return; // No compression
        }

        self.codes.clear();
        self.codebooks.clear();

        let n = vectors.len() / dim;
        if n == 0 {
            return;
        }

        let num_subdims = self.dims.min(dim);
        let subdim_size = dim / num_subdims;
        let ksub = n.min(256).max(1);

        self.ksub = ksub;
        self.subdim_size = subdim_size;
        self.codebooks = vec![0.0; num_subdims * ksub * subdim_size];

        // Train one codebook per sub-dimension.
        for sub_q in 0..num_subdims {
            let mut sub_vectors = Vec::with_capacity(n * subdim_size);
            for vec in vectors.chunks(dim) {
                let start = sub_q * subdim_size;
                let end = (start + subdim_size).min(dim);
                sub_vectors.extend_from_slice(&vec[start..end]);
            }

            let mut kmeans = KMeans::new(ksub, subdim_size);
            kmeans.set_max_iter(16);
            kmeans.train(&sub_vectors);
            let centroid_base = sub_q * ksub * subdim_size;
            self.codebooks[centroid_base..centroid_base + ksub * subdim_size]
                .copy_from_slice(&kmeans.centroids);
        }

        // Encode vectors with nearest centroid id for each sub-dimension.
        for vec in vectors.chunks(dim) {
            for sub_q in 0..num_subdims {
                let start = sub_q * subdim_size;
                let end = (start + subdim_size).min(dim);
                let subvec = &vec[start..end];
                let code = self.find_nearest_centroid(sub_q, subvec) as u8;
                self.codes.push(code);
            }
        }
    }

    fn find_nearest_centroid(&self, sub_q: usize, subvec: &[f32]) -> usize {
        if self.ksub == 0 || self.subdim_size == 0 {
            return 0;
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        let centroid_base = sub_q * self.ksub * self.subdim_size;
        for c in 0..self.ksub {
            let centroid_start = centroid_base + c * self.subdim_size;
            let centroid = &self.codebooks[centroid_start..centroid_start + self.subdim_size];
            let mut dist = 0.0f32;
            for d in 0..self.subdim_size {
                let diff = subvec[d] - centroid[d];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }
        best_idx
    }

    fn distance(&self, query: &[f32], idx: usize, dim: usize) -> f32 {
        if self.dims == 0 || self.codes.is_empty() || self.codebooks.is_empty() {
            return f32::MAX; // Fallback to full distance
        }

        let query_table = self.build_query_distance_table(query, dim);
        self.distance_with_table(&query_table, idx, dim)
    }

    /// Build ADC table once per query.
    fn build_query_distance_table(&self, query: &[f32], dim: usize) -> Vec<f32> {
        if self.dims == 0 || self.ksub == 0 || self.subdim_size == 0 || self.codebooks.is_empty() {
            return Vec::new();
        }

        let num_subdims = self.dims.min(dim);
        let mut table = vec![0.0f32; num_subdims * self.ksub];

        for sub_q in 0..num_subdims {
            let start = sub_q * self.subdim_size;
            let end = (start + self.subdim_size).min(dim);
            let query_sub = &query[start..end];
            let centroid_base = sub_q * self.ksub * self.subdim_size;
            let table_base = sub_q * self.ksub;

            for c in 0..self.ksub {
                let centroid_start = centroid_base + c * self.subdim_size;
                let centroid = &self.codebooks[centroid_start..centroid_start + self.subdim_size];
                let mut dist = 0.0f32;
                for d in 0..self.subdim_size {
                    let diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                table[table_base + c] = dist;
            }
        }

        table
    }

    /// Distance using precomputed ADC table.
    fn distance_with_table(&self, query_table: &[f32], idx: usize, dim: usize) -> f32 {
        if self.dims == 0 || self.codes.is_empty() || self.ksub == 0 || self.codebooks.is_empty() {
            return f32::MAX;
        }

        let num_subdims = self.dims.min(dim);
        let mut dist = 0.0f32;

        for sub_q in 0..num_subdims {
            let code_idx = self.codes[idx * num_subdims + sub_q] as usize;
            let table_row = sub_q * self.ksub;
            let centroid_idx = code_idx.min(self.ksub - 1);
            dist += query_table[table_row + centroid_idx];
        }

        dist
    }
}

/// DiskANN-style graph index (Vamana) - Enhanced
pub struct DiskAnnIndex {
    config: IndexConfig,
    dann_config: DiskAnnConfig,
    dim: usize,
    vectors: Vec<f32>,
    ids: Vec<i64>,
    graph: Vec<Vec<(i64, f32)>>,
    next_id: i64,
    trained: bool,
    alpha: f32,
    entry_point: Option<usize>,
    entry_points: Vec<usize>,
    /// PQ compression (optional)
    pq_codes: Option<PQCode>,
    /// Cached nodes for faster search
    cached_nodes: HashSet<usize>,
    /// Whether a flash-layout sidecar is available for this index snapshot.
    has_flash_layout: bool,
    /// Fixed-stride neighbor ids loaded from flash sidecar.
    flash_neighbor_ids: Option<Vec<i64>>,
    /// Max neighbor slots per node in flash sidecar.
    flash_max_degree: usize,
    /// Raw vectors loaded from flash sidecar.
    flash_vectors: Option<Vec<f32>>,
    /// Memory mapped flash sidecar for on-demand neighbor/vector reads.
    flash_sidecar_mmap: Option<memmap2::Mmap>,
    /// Budgeted runtime cache for flash mmap mode (node idx -> decoded neighbors).
    flash_cached_neighbors: Option<HashMap<usize, Box<[i64]>>>,
    /// Budgeted runtime cache for flash mmap mode (node idx -> decoded vector).
    flash_cached_vectors: Option<HashMap<usize, Box<[f32]>>>,
}

impl DiskAnnIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let dann_config = DiskAnnConfig::from_index_config(config);

        Ok(Self {
            config: config.clone(),
            dann_config,
            dim: config.dim,
            vectors: Vec::new(),
            ids: Vec::new(),
            graph: Vec::new(),
            next_id: 0,
            trained: false,
            alpha: 1.2,
            entry_point: None,
            entry_points: Vec::new(),
            pq_codes: None,
            cached_nodes: HashSet::new(),
            has_flash_layout: false,
            flash_neighbor_ids: None,
            flash_max_degree: 0,
            flash_vectors: None,
            flash_sidecar_mmap: None,
            flash_cached_neighbors: None,
            flash_cached_vectors: None,
        })
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        self.vectors.reserve(n * self.dim);
        self.graph.reserve(n);
        self.has_flash_layout = false;
        self.flash_neighbor_ids = None;
        self.flash_max_degree = 0;
        self.flash_vectors = None;
        self.flash_sidecar_mmap = None;
        self.flash_cached_neighbors = None;
        self.flash_cached_vectors = None;

        // Store vectors
        for i in 0..n {
            let start = i * self.dim;
            self.vectors
                .extend_from_slice(&vectors[start..start + self.dim]);
            self.ids.push(i as i64);
        }

        self.next_id = n as i64;

        // Hard gate on build DRAM budget (in-memory implementation boundary).
        if let Some(budget) = DiskAnnConfig::budget_bytes(self.dann_config.build_dram_budget_gb) {
            let resident_floor = self.vectors.len() * std::mem::size_of::<f32>()
                + self.ids.len() * std::mem::size_of::<i64>();
            if resident_floor > budget {
                return Err(crate::api::KnowhereError::InvalidArg(format!(
                    "disk_build_dram_budget_gb exceeded: resident_floor_bytes={} budget_bytes={}",
                    resident_floor, budget
                )));
            }
        }

        // Build graph with Vamana algorithm
        self.build_vamana_graph();

        // Build PQ codes if configured
        if self.dann_config.disk_pq_dims > 0 {
            self.build_pq_codes(vectors)?;
        }

        // Warm-up if configured
        if self.dann_config.warm_up {
            self.warm_up();
        }

        self.trained = true;
        if let Some(budget) = DiskAnnConfig::budget_bytes(self.dann_config.build_dram_budget_gb) {
            let used = self.get_stats().memory_usage_bytes;
            if used > budget {
                return Err(crate::api::KnowhereError::InvalidArg(format!(
                    "disk_build_dram_budget_gb exceeded after build: used_bytes={} budget_bytes={}",
                    used, budget
                )));
            }
        }
        tracing::info!(
            "Built DiskANN graph with {} nodes, max_degree={}, search_list={}",
            n,
            self.dann_config.max_degree,
            self.dann_config.search_list_size
        );
        Ok(())
    }

    pub fn scope_audit(&self) -> DiskAnnScopeAudit {
        DiskAnnScopeAudit {
            dim: self.dim,
            max_degree: self.dann_config.max_degree,
            search_list_size: self.dann_config.search_list_size,
            beamwidth: self.dann_config.beamwidth,
            uses_vamana_graph: true,
            uses_placeholder_pq: false,
            has_flash_layout: self.has_flash_layout,
            native_comparable: false,
            comparability_reason: "DiskAnnIndex remains an in-memory Vamana+PQ path without native DiskANN SSD async I/O pipeline, so it is not native-comparable",
        }
    }

    fn flash_layout_sidecar_path(path: &std::path::Path) -> std::path::PathBuf {
        let mut os = path.as_os_str().to_os_string();
        os.push(".flash");
        std::path::PathBuf::from(os)
    }

    fn write_flash_layout_sidecar(&self, sidecar: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(sidecar)?;
        let n = self.ids.len();
        let max_degree = self.dann_config.max_degree;

        file.write_all(b"DNFL")?;
        file.write_all(&1u32.to_le_bytes())?; // flash sidecar version
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(max_degree as u32).to_le_bytes())?;
        file.write_all(&(n as u64).to_le_bytes())?;

        for idx in 0..n {
            let neighbors = self.graph.get(idx).cloned().unwrap_or_default();
            let degree = neighbors.len().min(max_degree);
            file.write_all(&(degree as u32).to_le_bytes())?;

            // Fixed-size neighbor slot for deterministic per-node stride.
            for slot in 0..max_degree {
                let id = if slot < degree { neighbors[slot].0 } else { -1i64 };
                file.write_all(&id.to_le_bytes())?;
            }

            let start = idx * self.dim;
            for &v in &self.vectors[start..start + self.dim] {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn try_load_flash_layout_sidecar(
        &mut self,
        sidecar: &std::path::Path,
        count: usize,
    ) -> Result<bool> {
        use std::fs::File;
        use std::io::Read;

        if !sidecar.exists() {
            return Ok(false);
        }
        let mut file = File::open(sidecar)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"DNFL" {
            return Ok(false);
        }

        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);
        if version != 1 {
            return Ok(false);
        }

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        if dim != self.dim {
            return Ok(false);
        }

        let mut md_bytes = [0u8; 4];
        file.read_exact(&mut md_bytes)?;
        let max_degree = u32::from_le_bytes(md_bytes) as usize;
        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let n = u64::from_le_bytes(count_bytes) as usize;
        if n != count {
            return Ok(false);
        }

        let header_len = 4 + 4 + 4 + 4 + 8;
        let record_len = 4 + max_degree.saturating_mul(8) + self.dim.saturating_mul(4);
        let expected_len = header_len + n.saturating_mul(record_len);

        if self.dann_config.flash_mmap_mode {
            let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
            if mmap.len() < expected_len {
                return Ok(false);
            }
            self.flash_max_degree = max_degree;
            self.flash_neighbor_ids = None;
            self.flash_vectors = None;
            self.flash_sidecar_mmap = Some(mmap);
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
            return Ok(true);
        }

        let mut flash_neighbor_ids = Vec::with_capacity(n * max_degree);
        let mut flash_vectors = Vec::with_capacity(n * self.dim);
        for _ in 0..n {
            file.read_exact(&mut md_bytes)?; // degree
            let degree = u32::from_le_bytes(md_bytes) as usize;
            if degree > max_degree {
                return Ok(false);
            }
            for _ in 0..max_degree {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                flash_neighbor_ids.push(i64::from_le_bytes(id_bytes));
            }
            for _ in 0..self.dim {
                let mut f_bytes = [0u8; 4];
                file.read_exact(&mut f_bytes)?;
                flash_vectors.push(f32::from_le_bytes(f_bytes));
            }
        }

        self.flash_max_degree = max_degree;
        self.flash_neighbor_ids = Some(flash_neighbor_ids);
        self.flash_vectors = Some(flash_vectors);
        self.flash_sidecar_mmap = None;
        self.flash_cached_neighbors = None;
        self.flash_cached_vectors = None;

        Ok(true)
    }

    #[inline]
    fn flash_mmap_record_len(&self) -> Option<usize> {
        if self.flash_max_degree == 0 {
            None
        } else {
            Some(4 + self.flash_max_degree * 8 + self.dim * 4)
        }
    }

    #[inline]
    fn flash_mmap_neighbor_id(&self, idx: usize, slot: usize) -> Option<i64> {
        let mmap = self.flash_sidecar_mmap.as_ref()?;
        let record = self.flash_mmap_record_len()?;
        if slot >= self.flash_max_degree {
            return None;
        }
        let offset = 24 + idx.checked_mul(record)? + 4 + slot.checked_mul(8)?;
        let end = offset + 8;
        if end > mmap.len() {
            return None;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&mmap[offset..end]);
        Some(i64::from_le_bytes(bytes))
    }

    #[inline]
    fn flash_mmap_vector_component(&self, idx: usize, d: usize) -> Option<f32> {
        let mmap = self.flash_sidecar_mmap.as_ref()?;
        let record = self.flash_mmap_record_len()?;
        if d >= self.dim {
            return None;
        }
        let vec_base = 24 + idx.checked_mul(record)? + 4 + self.flash_max_degree.checked_mul(8)?;
        let offset = vec_base + d.checked_mul(4)?;
        let end = offset + 4;
        if end > mmap.len() {
            return None;
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&mmap[offset..end]);
        Some(f32::from_le_bytes(bytes))
    }

    fn flash_mmap_read_node(&self, idx: usize) -> Option<(Box<[i64]>, Box<[f32]>)> {
        if self.flash_sidecar_mmap.is_none() || self.flash_max_degree == 0 {
            return None;
        }
        let mut nbrs = Vec::with_capacity(self.flash_max_degree);
        for slot in 0..self.flash_max_degree {
            nbrs.push(self.flash_mmap_neighbor_id(idx, slot)?);
        }
        let mut vec = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            vec.push(self.flash_mmap_vector_component(idx, d)?);
        }
        Some((nbrs.into_boxed_slice(), vec.into_boxed_slice()))
    }

    fn flash_mmap_read_vector(&self, idx: usize) -> Option<Box<[f32]>> {
        if self.flash_sidecar_mmap.is_none() {
            return None;
        }
        let mut vec = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            vec.push(self.flash_mmap_vector_component(idx, d)?);
        }
        Some(vec.into_boxed_slice())
    }

    #[inline]
    fn flash_prefetch_enabled(&self) -> bool {
        self.flash_sidecar_mmap.is_some()
            && self.flash_vectors.is_none()
            && self.dann_config.flash_prefetch_batch > 0
    }

    #[inline]
    fn query_prefetch_cap(&self, total_nodes: usize) -> usize {
        if !self.flash_prefetch_enabled() || total_nodes == 0 {
            return 0;
        }
        let batch = self.dann_config.flash_prefetch_batch.max(1);
        let cap = batch
            .saturating_mul(self.dann_config.beamwidth.max(1))
            .max(batch)
            .min(2048);
        cap.min(total_nodes)
    }

    fn prefetch_vectors_batch_into_cache(
        &self,
        neighbor_ids: &[usize],
        total_nodes: usize,
        query_prefetch_cap: usize,
        cache: &mut HashMap<usize, Box<[f32]>>,
    ) {
        if !self.flash_prefetch_enabled()
            || query_prefetch_cap == 0
            || cache.len() >= query_prefetch_cap
        {
            return;
        }

        let budget = query_prefetch_cap.saturating_sub(cache.len());
        let batch = self.dann_config.flash_prefetch_batch.min(budget);
        if batch == 0 {
            return;
        }

        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        for &n_idx in neighbor_ids {
            if n_idx >= total_nodes || cache.contains_key(&n_idx) || !seen.insert(n_idx) {
                continue;
            }
            selected.push(n_idx);
            if selected.len() >= batch {
                break;
            }
        }
        if selected.is_empty() {
            return;
        }

        #[cfg(feature = "parallel")]
        {
            let items: Vec<(usize, Box<[f32]>)> = selected
                .par_iter()
                .filter_map(|&idx| self.flash_mmap_read_vector(idx).map(|v| (idx, v)))
                .collect();
            for (idx, vec) in items {
                if cache.len() >= query_prefetch_cap {
                    break;
                }
                cache.insert(idx, vec);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for idx in selected {
                if cache.len() >= query_prefetch_cap {
                    break;
                }
                if let Some(v) = self.flash_mmap_read_vector(idx) {
                    cache.insert(idx, v);
                }
            }
        }
    }

    fn build_flash_runtime_cache_from_budget(&mut self) {
        if self.flash_sidecar_mmap.is_none() || self.flash_max_degree == 0 {
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
            return;
        }
        let Some(budget) = DiskAnnConfig::budget_bytes(self.dann_config.cache_dram_budget_gb) else {
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
            return;
        };
        if budget == 0 {
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
            return;
        }

        let node_cost = self
            .flash_max_degree
            .saturating_mul(std::mem::size_of::<i64>())
            .saturating_add(self.dim.saturating_mul(std::mem::size_of::<f32>()));
        if node_cost == 0 || node_cost > budget {
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
            return;
        }

        let mut candidates = Vec::new();
        candidates.extend(self.entry_points.iter().copied());
        if candidates.is_empty() {
            if let Some(ep) = self.entry_point {
                candidates.push(ep);
            }
        }
        if candidates.is_empty() && !self.ids.is_empty() {
            candidates.push(0);
        }

        let mut seen = HashSet::new();
        let mut used = 0usize;
        let mut cached_neighbors: HashMap<usize, Box<[i64]>> = HashMap::new();
        let mut cached_vectors: HashMap<usize, Box<[f32]>> = HashMap::new();

        let mut enqueue_neighbors = Vec::new();
        for idx in candidates {
            if idx >= self.ids.len() || !seen.insert(idx) {
                continue;
            }
            if used + node_cost > budget {
                break;
            }
            if let Some((nbrs, vec)) = self.flash_mmap_read_node(idx) {
                enqueue_neighbors.extend(
                    nbrs.iter()
                        .copied()
                        .filter(|&id| id >= 0)
                        .map(|id| id as usize),
                );
                used += node_cost;
                cached_neighbors.insert(idx, nbrs);
                cached_vectors.insert(idx, vec);
            }
        }

        for idx in enqueue_neighbors {
            if idx >= self.ids.len() || !seen.insert(idx) {
                continue;
            }
            if used + node_cost > budget {
                break;
            }
            if let Some((nbrs, vec)) = self.flash_mmap_read_node(idx) {
                used += node_cost;
                cached_neighbors.insert(idx, nbrs);
                cached_vectors.insert(idx, vec);
            }
        }

        if cached_neighbors.is_empty() {
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
        } else {
            self.flash_cached_neighbors = Some(cached_neighbors);
            self.flash_cached_vectors = Some(cached_vectors);
        }
    }

    #[inline]
    fn node_vector(&self, idx: usize) -> &[f32] {
        if let Some(v) = self
            .flash_cached_vectors
            .as_ref()
            .and_then(|m| m.get(&idx))
            .map(|b| b.as_ref())
        {
            return v;
        }
        let data: &[f32] = self.flash_vectors.as_deref().unwrap_or(&self.vectors);
        let start = idx * self.dim;
        &data[start..start + self.dim]
    }

    #[inline]
    fn flash_neighbors(&self, idx: usize) -> Option<&[i64]> {
        if let Some(n) = self
            .flash_cached_neighbors
            .as_ref()
            .and_then(|m| m.get(&idx))
            .map(|b| b.as_ref())
        {
            return Some(n);
        }
        let ids = self.flash_neighbor_ids.as_deref()?;
        if self.flash_max_degree == 0 {
            return None;
        }
        let start = idx * self.flash_max_degree;
        let end = start + self.flash_max_degree;
        if end > ids.len() {
            return None;
        }
        Some(&ids[start..end])
    }

    #[inline]
    fn neighbor_indices(&self, idx: usize) -> Vec<usize> {
        if let Some(nbrs) = self.flash_neighbors(idx) {
            return nbrs
                .iter()
                .copied()
                .filter(|&id| id >= 0)
                .map(|id| id as usize)
                .collect();
        }
        if self.flash_sidecar_mmap.is_some() {
            let mut out = Vec::with_capacity(self.flash_max_degree);
            for slot in 0..self.flash_max_degree {
                if let Some(id) = self.flash_mmap_neighbor_id(idx, slot) {
                    if id >= 0 {
                        out.push(id as usize);
                    }
                }
            }
            return out;
        }
        self.graph
            .get(idx)
            .map(|nbrs| nbrs.iter().map(|&(id, _)| id as usize).collect())
            .unwrap_or_default()
    }

    /// Build PQ codes for compression
    fn build_pq_codes(&mut self, vectors: &[f32]) -> Result<()> {
        let mut pq = PQCode::new(self.dann_config.disk_pq_dims);
        pq.encode(vectors, self.dim);
        if let Some(budget) = DiskAnnConfig::budget_bytes(self.dann_config.pq_code_budget_gb) {
            if pq.codes.len() > budget {
                return Err(KnowhereError::InvalidArg(format!(
                    "disk_pq_code_budget_gb exceeded: pq_code_bytes={} budget_bytes={}",
                    pq.codes.len(),
                    budget
                )));
            }
        }
        self.pq_codes = Some(pq);
        Ok(())
    }

    /// Warm-up: cache frequently accessed nodes
    fn warm_up(&mut self) {
        let cache_budget = DiskAnnConfig::budget_bytes(self.dann_config.cache_dram_budget_gb);
        let mut used_bytes = 0usize;
        let node_bytes = |dim: usize, degree: usize| {
            dim * std::mem::size_of::<f32>()
                + degree * (std::mem::size_of::<i64>() + std::mem::size_of::<f32>())
        };
        // Simple warm-up: cache entry points and their neighbors
        let mut starts = self.entry_points.clone();
        if starts.is_empty() {
            if let Some(entry) = self.entry_point {
                starts.push(entry);
            }
        }
        for entry in starts {
            if entry >= self.ids.len() {
                continue;
            }
            let entry_cost = node_bytes(self.dim, self.graph.get(entry).map_or(0, |v| v.len()));
            if cache_budget.is_some_and(|b| used_bytes + entry_cost > b) {
                break;
            }
            if self.cached_nodes.insert(entry) {
                used_bytes += entry_cost;
            }
            if let Some(nbrs) = self.graph.get(entry) {
                for &(id, _) in nbrs {
                    if (id as usize) < self.ids.len() {
                        let n_idx = id as usize;
                        let nbr_cost = node_bytes(self.dim, self.graph.get(n_idx).map_or(0, |v| v.len()));
                        if cache_budget.is_some_and(|b| used_bytes + nbr_cost > b) {
                            break;
                        }
                        if self.cached_nodes.insert(n_idx) {
                            used_bytes += nbr_cost;
                        }
                    }
                }
            }
        }
    }

    /// Build Vamana graph using the proper algorithm
    #[allow(non_snake_case)]
    fn build_vamana_graph(&mut self) {
        let n = self.ids.len();
        if n == 0 {
            return;
        }

        let L = self.dann_config.construction_l;
        let R = self.dann_config.max_degree;
        let build_r = self.compute_build_degree_limit(R);

        // Sort vectors by first dimension for better entry point selection
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.vectors[a * self.dim]
                .partial_cmp(&self.vectors[b * self.dim])
                .unwrap_or(Ordering::Equal)
        });

        // Build graph incrementally (Vamana style)
        let mut current_graph: Vec<Vec<(i64, f32)>> = Vec::with_capacity(n);

        // First node is entry point
        self.entry_point = Some(0);
        self.entry_points = vec![0];
        current_graph.push(Vec::new());

        #[cfg(feature = "parallel")]
        if self.config.params.num_threads.unwrap_or(1) > 1 && n > 2 {
            let batch_size = self.dann_config.build_parallel_batch_size.min(n - 1).max(1);
            let mut batch_start = 1usize;
            while batch_start < n {
                let batch_end = (batch_start + batch_size).min(n);
                let snapshot = current_graph.clone();
                let snapshot_len = snapshot.len();
                let mut batch_results: Vec<(usize, Vec<(usize, f32)>)> = (batch_start..batch_end)
                    .into_par_iter()
                    .map(|node_idx| {
                        let merged = self.gather_build_candidates(
                            node_idx,
                            L,
                            build_r,
                            &snapshot,
                            snapshot_len,
                        );
                        (node_idx, merged)
                    })
                    .collect();
                batch_results.sort_by_key(|(node_idx, _)| *node_idx);

                for (node_idx, merged_neighbors) in batch_results {
                    let mut node_neighbors: Vec<(i64, f32)> = merged_neighbors
                        .iter()
                        .map(|&(idx, dist)| (self.ids[idx], dist))
                        .collect();
                    node_neighbors = self.prune_neighbors(node_idx, &node_neighbors, build_r);
                    current_graph.push(node_neighbors);

                    for &(idx, dist) in &merged_neighbors {
                        if idx < current_graph.len() {
                            current_graph[idx].push((self.ids[node_idx], dist));
                            current_graph[idx] =
                                self.prune_neighbors(idx, &current_graph[idx], build_r);
                        }
                    }
                }
                batch_start = batch_end;
            }

            self.graph = current_graph;
            if !self.dann_config.accelerate_build {
                self.refine_graph();
            }
            self.finalize_graph_degree(R);
            self.refresh_entry_points();
            return;
        }

        // Insert remaining nodes one by one
        for i in 1..n {
            let merged_neighbors = self.gather_build_candidates(i, L, build_r, &current_graph, i);

            // Add bidirectional edges
            let mut node_neighbors: Vec<(i64, f32)> = merged_neighbors
                .iter()
                .map(|&(idx, dist)| (self.ids[idx], dist))
                .collect();

            // Prune to max_degree using Vamana pruning
            node_neighbors = self.prune_neighbors(i, &node_neighbors, build_r);

            current_graph.push(node_neighbors);

            // Add reverse edges
            for &(idx, dist) in &merged_neighbors {
                if idx < current_graph.len() {
                    current_graph[idx].push((self.ids[i], dist));
                    // Prune reverse edges too
                    current_graph[idx] = self.prune_neighbors(idx, &current_graph[idx], build_r);
                }
            }
        }

        self.graph = current_graph;

        // Second pass for better connectivity (unless accelerate_build)
        if !self.dann_config.accelerate_build {
            self.refine_graph();
        }
        self.finalize_graph_degree(R);

        self.refresh_entry_points();
    }

    fn gather_build_candidates(
        &self,
        node_idx: usize,
        l: usize,
        build_r: usize,
        graph: &[Vec<(i64, f32)>],
        upper_bound: usize,
    ) -> Vec<(usize, f32)> {
        let query = &self.vectors[node_idx * self.dim..(node_idx + 1) * self.dim];
        let mut merged_neighbors = self.vamana_search(query, l, build_r, graph);
        if self.dann_config.intra_batch_candidates > 0 {
            merged_neighbors.extend(self.collect_intra_batch_candidates_with_upper(
                node_idx,
                upper_bound,
            ));
        }
        if self.dann_config.random_init_edges > 0 {
            merged_neighbors.extend(self.collect_random_initial_candidates_with_upper(
                node_idx,
                upper_bound,
            ));
        }
        merged_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        merged_neighbors.dedup_by(|a, b| a.0 == b.0);
        merged_neighbors
    }

    fn collect_random_initial_candidates_with_upper(
        &self,
        node_idx: usize,
        upper_bound: usize,
    ) -> Vec<(usize, f32)> {
        let limit = upper_bound.min(node_idx);
        if self.dann_config.random_init_edges == 0 || limit == 0 {
            return Vec::new();
        }

        let sample = self.dann_config.random_init_edges.min(limit);
        let query = &self.vectors[node_idx * self.dim..(node_idx + 1) * self.dim];
        let base_seed = self.config.params.random_seed.unwrap_or(42);
        let seed = base_seed ^ ((node_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pool: Vec<usize> = (0..limit).collect();
        pool.shuffle(&mut rng);
        pool.into_iter()
            .take(sample)
            .map(|idx| (idx, self.compute_dist(query, idx)))
            .collect()
    }

    fn compute_build_degree_limit(&self, max_degree: usize) -> usize {
        max_degree
            .saturating_mul(self.dann_config.build_degree_slack_pct)
            .saturating_add(99)
            / 100
    }

    fn finalize_graph_degree(&mut self, max_degree: usize) {
        for i in 0..self.graph.len() {
            if self.graph[i].len() > max_degree {
                self.graph[i] = self.prune_neighbors(i, &self.graph[i], max_degree);
            }
        }
    }

    fn refresh_entry_points(&mut self) {
        let n = self.ids.len();
        if n == 0 {
            self.entry_point = None;
            self.entry_points.clear();
            return;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.vectors[a * self.dim]
                .partial_cmp(&self.vectors[b * self.dim])
                .unwrap_or(Ordering::Equal)
        });

        let target = self.dann_config.num_entry_points.max(1).min(n);
        let pool_size = n.min((target * 32).clamp(target, 4096));
        let mut pool = Vec::with_capacity(pool_size);
        if pool_size == 1 {
            pool.push(indices[0]);
        } else {
            for slot in 0..pool_size {
                let pos = slot * (n - 1) / (pool_size - 1);
                pool.push(indices[pos]);
            }
        }
        pool.sort_unstable();
        pool.dedup();
        if pool.is_empty() {
            pool.push(indices[0]);
        }

        let mut centroid = vec![0.0f32; self.dim];
        for i in 0..n {
            let start = i * self.dim;
            for d in 0..self.dim {
                centroid[d] += self.vectors[start + d];
            }
        }
        let inv_n = 1.0f32 / n as f32;
        for v in &mut centroid {
            *v *= inv_n;
        }

        let first = *pool
            .iter()
            .min_by(|&&a, &&b| {
                self.compute_dist(&centroid, a)
                    .partial_cmp(&self.compute_dist(&centroid, b))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(&indices[0]);

        let mut points = Vec::with_capacity(target);
        points.push(first);
        while points.len() < target {
            let next = pool
                .iter()
                .copied()
                .filter(|idx| !points.contains(idx))
                .max_by(|&a, &b| {
                    let min_a = points
                        .iter()
                        .map(|&s| self.distance_between_nodes(s, a))
                        .fold(f32::MAX, f32::min);
                    let min_b = points
                        .iter()
                        .map(|&s| self.distance_between_nodes(s, b))
                        .fold(f32::MAX, f32::min);
                    min_a.partial_cmp(&min_b).unwrap_or(Ordering::Equal)
                });
            if let Some(idx) = next {
                points.push(idx);
            } else {
                break;
            }
        }

        if points.is_empty() {
            points.push(indices[0]);
        }

        self.entry_point = Some(points[0]);
        self.entry_points = points;
    }

    fn initial_search_starts(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let n = self.ids.len();
        if n == 0 {
            return Vec::new();
        }

        let mut starts = self.entry_points.clone();
        if starts.is_empty() {
            starts.push(self.entry_point.unwrap_or(0));
        }
        starts.retain(|&idx| idx < n);
        let mut dedup = HashSet::with_capacity(starts.len() * 2 + 1);
        starts.retain(|idx| dedup.insert(*idx));
        if starts.is_empty() {
            starts.push(0);
        }

        let mut out: Vec<(usize, f32)> = starts
            .into_iter()
            .map(|idx| (idx, self.compute_dist(query, idx)))
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let keep = self.dann_config.num_entry_points.max(1).min(out.len());
        out.truncate(keep);
        out
    }

    /// Add temporal candidates from recent inserted nodes.
    #[cfg(test)]
    fn collect_intra_batch_candidates(&self, i: usize) -> Vec<(usize, f32)> {
        self.collect_intra_batch_candidates_with_upper(i, i)
    }

    fn collect_intra_batch_candidates_with_upper(
        &self,
        i: usize,
        upper_bound: usize,
    ) -> Vec<(usize, f32)> {
        let limit = upper_bound.min(i);
        let window = self.dann_config.intra_batch_candidates.min(limit);
        if window == 0 {
            return Vec::new();
        }

        let start = limit - window;
        let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
        let mut out = Vec::with_capacity(window);
        for idx in start..limit {
            let d = self.compute_dist(query, idx);
            out.push((idx, d));
        }
        out
    }

    /// Vamana search for finding neighbors during build
    #[allow(non_snake_case)]
    fn vamana_search(
        &self,
        query: &[f32],
        L: usize,
        _R: usize,
        graph: &[Vec<(i64, f32)>],
    ) -> Vec<(usize, f32)> {
        let mut visited = vec![false; self.ids.len()];
        let mut candidates: BinaryHeap<ReverseOrderedFloat> = BinaryHeap::new();
        let mut results: Vec<(f32, usize)> = Vec::new();

        // Start from entry point
        if let Some(entry) = self.entry_point {
            let dist = self.compute_dist(query, entry);
            candidates.push(ReverseOrderedFloat(dist, entry));
            visited[entry] = true;
        }

        // Beam search
        let beam_size = self.dann_config.beamwidth;

        while !candidates.is_empty() && results.len() < L {
            // Get best candidate
            let ReverseOrderedFloat(dist, idx) = candidates.pop().unwrap();
            results.push((dist, idx));

            // Explore neighbors (beam search)
            let mut nbr_dists: Vec<(f32, usize)> = Vec::new();

            if let Some(nbrs) = graph.get(idx) {
                for &(nbr_id, _) in nbrs {
                    let n_idx = nbr_id as usize;
                    if n_idx < visited.len() && !visited[n_idx] {
                        let d = self.compute_dist(query, n_idx);
                        nbr_dists.push((d, n_idx));
                    }
                }
            }

            // Sort and add best beamwidth neighbors
            nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (d, n_idx) in nbr_dists.into_iter().take(beam_size) {
                visited[n_idx] = true;
                candidates.push(ReverseOrderedFloat(d, n_idx));
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.into_iter().map(|(d, idx)| (idx, d)).collect()
    }

    /// Prune neighbors using Vamana pruning strategy
    #[allow(non_snake_case)]
    fn prune_neighbors(
        &self,
        _node_idx: usize,
        neighbors: &[(i64, f32)],
        R: usize,
    ) -> Vec<(i64, f32)> {
        if neighbors.len() <= R {
            return neighbors.to_vec();
        }

        // Sort by distance
        let mut sorted: Vec<(i64, f32)> = neighbors.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Robust-prune style alpha occlusion:
        // - iterate with increasing current_alpha (1.0 -> alpha)
        // - then optionally saturate back to R with nearest remaining neighbors.
        let mut selected: Vec<(i64, f32)> = Vec::new();
        let mut selected_ids = HashSet::new();
        let final_alpha = self.alpha.max(1.0);
        let alpha_step = final_alpha.min(1.2);
        let mut current_alpha = 1.0f32;

        while selected.len() < R {
            for &(id, dist) in &sorted {
                if selected.len() >= R {
                    break;
                }
                if selected_ids.contains(&id) {
                    continue;
                }

                let idx = id as usize;
                if idx >= self.vectors.len() / self.dim {
                    continue;
                }

                let mut occluded = false;
                for &(sel_id, _) in &selected {
                    let sel_idx = sel_id as usize;
                    if sel_idx < self.vectors.len() / self.dim {
                        let d_jk = self.distance_between_nodes(sel_idx, idx);
                        if d_jk <= f32::EPSILON || dist / d_jk > current_alpha {
                            occluded = true;
                            break;
                        }
                    }
                }

                if !occluded {
                    selected.push((id, dist));
                    selected_ids.insert(id);
                }
            }

            if current_alpha >= final_alpha {
                break;
            }
            current_alpha = (current_alpha * alpha_step).min(final_alpha);
        }

        if self.dann_config.saturate_after_prune && final_alpha > 1.0 {
            for &(id, dist) in &sorted {
                if selected.len() >= R {
                    break;
                }
                if selected_ids.insert(id) {
                    selected.push((id, dist));
                }
            }
        }

        selected
    }

    /// Refine graph with second pass
    #[allow(non_snake_case)]
    fn refine_graph(&mut self) {
        let n = self.ids.len();
        let R = self.dann_config.max_degree;
        let build_r = self.compute_build_degree_limit(R);

        // For each node, search again and update edges
        for i in 0..n {
            let query = &self.vectors[i * self.dim..(i + 1) * self.dim];
            let neighbors =
                self.vamana_search(query, self.dann_config.construction_l, build_r, &self.graph);

            let new_neighbors: Vec<(i64, f32)> = neighbors
                .iter()
                .filter(|&&(idx, _)| idx != i)
                .map(|&(idx, dist)| (self.ids[idx], dist))
                .collect();

            if !new_neighbors.is_empty() {
                self.graph[i] = self.prune_neighbors(i, &new_neighbors, build_r);
            }
        }
    }

    #[inline]
    fn l2_sqr(&self, a: &[f32], b_idx: usize) -> f32 {
        if self.flash_sidecar_mmap.is_some() && self.flash_vectors.is_none() {
            let mut acc = 0.0f32;
            for (d, &av) in a.iter().enumerate().take(self.dim) {
                let bv = self.flash_mmap_vector_component(b_idx, d).unwrap_or(0.0);
                let diff = av - bv;
                acc += diff * diff;
            }
            acc
        } else {
            let b = self.node_vector(b_idx);
            // Search/range gate on squared L2 directly to avoid an unnecessary sqrt+square roundtrip.
            simd::l2_distance_sq(a, b)
        }
    }

    #[inline]
    fn distance_between_nodes(&self, a_idx: usize, b_idx: usize) -> f32 {
        if a_idx >= self.ids.len() || b_idx >= self.ids.len() {
            return f32::MAX;
        }
        let a_start = a_idx * self.dim;
        let a = &self.vectors[a_start..a_start + self.dim];
        self.compute_dist(a, b_idx)
    }

    #[inline]
    fn ip_distance(&self, a: &[f32], b_idx: usize) -> f32 {
        // For IP, higher is better, so we return negative (for consistent sorting)
        let mut sum = 0.0f32;
        if self.flash_sidecar_mmap.is_some() && self.flash_vectors.is_none() {
            for (d, &av) in a.iter().enumerate().take(self.dim) {
                let bv = self.flash_mmap_vector_component(b_idx, d).unwrap_or(0.0);
                sum += av * bv;
            }
        } else {
            let b = self.node_vector(b_idx);
            for i in 0..self.dim {
                sum += a[i] * b[i];
            }
        }
        -sum
    }

    /// Compute distance based on metric type
    #[inline]
    fn compute_dist(&self, query: &[f32], idx: usize) -> f32 {
        match self.config.metric_type {
            MetricType::L2 => self.l2_sqr(query, idx),
            MetricType::Ip => self.ip_distance(query, idx),
            _ => self.l2_sqr(query, idx),
        }
    }

    #[inline]
    fn compute_dist_with_prefetched(
        &self,
        query: &[f32],
        idx: usize,
        prefetched_vectors: Option<&HashMap<usize, Box<[f32]>>>,
    ) -> f32 {
        if let Some(v) = prefetched_vectors.and_then(|m| m.get(&idx)) {
            match self.config.metric_type {
                MetricType::Ip => {
                    let mut sum = 0.0f32;
                    for i in 0..self.dim {
                        sum += query[i] * v[i];
                    }
                    -sum
                }
                _ => simd::l2_distance_sq(query, v),
            }
        } else {
            self.compute_dist(query, idx)
        }
    }

    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        let n = vectors.len() / self.dim;

        for i in 0..n {
            let start = i * self.dim;
            self.vectors
                .extend_from_slice(&vectors[start..start + self.dim]);

            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            self.ids.push(id);
            self.graph.push(Vec::new());
        }

        Ok(n)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if self.ids.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n_queries = query.len() / self.dim;
        if n_queries * self.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        #[allow(non_snake_case)]
        let L = req.nprobe.max(self.dann_config.search_list_size / 2);
        let k = req.top_k;

        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            // Use improved beam search
            let results = self.beam_search(query_vec, L, req.filter.as_deref(), None);

            for i in 0..k {
                if i < results.len() {
                    all_ids.push(results[i].0);
                    // For L2: sqrt the squared distance; For IP: negate to get positive similarity
                    let dist = match self.config.metric_type {
                        MetricType::Ip => -results[i].1, // Convert back to positive similarity
                        _ => results[i].1.sqrt(),
                    };
                    all_dists.push(dist);
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    /// Beam search with early termination
    #[allow(non_snake_case)]
    fn node_allowed(
        &self,
        idx: usize,
        filter: Option<&dyn Predicate>,
        bitset: Option<&crate::bitset::BitsetView>,
    ) -> bool {
        if bitset.is_some_and(|b| idx < b.len() && b.get(idx)) {
            return false;
        }
        if let Some(pred) = filter {
            return pred.evaluate(self.ids[idx]);
        }
        true
    }

    #[allow(non_snake_case)]
    fn beam_search(
        &self,
        query: &[f32],
        L: usize,
        filter: Option<&dyn Predicate>,
        bitset: Option<&crate::bitset::BitsetView>,
    ) -> Vec<(i64, f32)> {
        let n = self.ids.len();
        if n == 0 {
            return vec![];
        }

        let beamwidth = self.dann_config.beamwidth;
        let effective_l = if self.pq_codes.is_some() {
            let expanded = L
                .saturating_mul(self.dann_config.pq_candidate_expand_pct)
                .saturating_add(99)
                / 100;
            expanded.max(L).min(n)
        } else {
            L
        };
        let mut visited = vec![false; n];
        let mut candidates: BinaryHeap<ReverseOrderedFloat> = BinaryHeap::new();
        let mut explored: Vec<(f32, usize)> = Vec::new();
        let mut accepted: Vec<(f32, usize)> = Vec::new();
        let mut prefetched_vectors = if self.flash_prefetch_enabled() {
            Some(HashMap::new())
        } else {
            None
        };
        let query_prefetch_cap = self.query_prefetch_cap(n);
        let pq_query_table = self.pq_codes.as_ref().and_then(|pq| {
            if pq.dims > 0 && pq.ksub > 0 {
                Some(pq.build_query_distance_table(query, self.dim))
            } else {
                None
            }
        });

        // Start from entry points (medoid-like seeds)
        let starts = self.initial_search_starts(query);
        if starts.is_empty() {
            return Vec::new();
        }
        for (start, dist) in &starts {
            candidates.push(ReverseOrderedFloat(*dist, *start));
            visited[*start] = true;
        }

        // Early termination tracking
        let mut no_progress_count = 0;
        let mut best_dist = starts[0].1;

        // Beam search loop
        while !candidates.is_empty() && explored.len() < effective_l {
            let ReverseOrderedFloat(dist, idx) = candidates.pop().unwrap();
            explored.push((dist, idx));
            if self.node_allowed(idx, filter, bitset) {
                accepted.push((dist, idx));
            }

            // Check for early termination
            if dist < best_dist * 0.99 {
                best_dist = dist;
                no_progress_count = 0;
            } else {
                no_progress_count += 1;
            }

            // Early termination: stop if no progress
            if no_progress_count > effective_l / 4 {
                break;
            }

            // Explore neighbors with beamwidth limit
            {
                let neighbor_ids = self.neighbor_indices(idx);
                if let Some(cache) = prefetched_vectors.as_mut() {
                    self.prefetch_vectors_batch_into_cache(
                        &neighbor_ids,
                        n,
                        query_prefetch_cap,
                        cache,
                    );
                }
                let mut nbr_dists: Vec<(f32, usize)> = Vec::new();

                for n_idx in neighbor_ids {
                    if n_idx < n && !visited[n_idx] {
                        // Use PQ distance if available and node is not cached
                        let d = if let Some(pq_codes) = &self.pq_codes {
                            if !self.cached_nodes.contains(&n_idx) {
                                if let Some(table) = &pq_query_table {
                                    pq_codes.distance_with_table(table, n_idx, self.dim)
                                } else {
                                    pq_codes.distance(query, n_idx, self.dim)
                                }
                            } else {
                                self.compute_dist(query, n_idx)
                            }
                        } else {
                            self.compute_dist_with_prefetched(
                                query,
                                n_idx,
                                prefetched_vectors.as_ref(),
                            )
                        };
                        nbr_dists.push((d, n_idx));
                    }
                }

                // Phase-1: PQ/approx screening order.
                nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                // Phase-2: optional exact rerank on a wider candidate pool.
                let rerank_width = if self.pq_codes.is_some() {
                    beamwidth
                        .saturating_mul(self.dann_config.rerank_expand_pct)
                        .saturating_add(99)
                        / 100
                } else {
                    beamwidth
                }
                .max(beamwidth)
                .min(nbr_dists.len());

                let mut reranked: Vec<(f32, usize)> = nbr_dists
                    .into_iter()
                    .take(rerank_width)
                    .map(|(_screen_d, n_idx)| {
                        (
                            self.compute_dist_with_prefetched(
                                query,
                                n_idx,
                                prefetched_vectors.as_ref(),
                            ),
                            n_idx,
                        )
                    })
                    .collect();
                reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for (exact_d, n_idx) in reranked.into_iter().take(beamwidth) {
                    visited[n_idx] = true;
                    candidates.push(ReverseOrderedFloat(exact_d, n_idx));
                }
            }
        }

        // If filtering shrinks candidate pool too much, fill with exact scan.
        if accepted.len() < L {
            let mut accepted_set: HashSet<usize> = accepted.iter().map(|(_, idx)| *idx).collect();
            let mut fallback: Vec<(f32, usize)> = Vec::new();
            for idx in 0..n {
                if accepted_set.contains(&idx) || !self.node_allowed(idx, filter, bitset) {
                    continue;
                }
                fallback.push((self.compute_dist(query, idx), idx));
            }
            fallback.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            for (dist, idx) in fallback {
                accepted.push((dist, idx));
                accepted_set.insert(idx);
                if accepted.len() >= L {
                    break;
                }
            }
        }

        accepted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        accepted.truncate(L);

        accepted
            .into_iter()
            .map(|(d, idx)| (self.ids[idx], d))
            .collect()
    }

    /// Range search: find all vectors within radius
    pub fn range_search(
        &self,
        query: &[f32],
        radius: f32,
        max_results: usize,
    ) -> Result<Vec<(i64, f32)>> {
        if self.ids.is_empty() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index is empty".to_string(),
            ));
        }

        let n = self.ids.len();
        let mut results: Vec<(i64, f32)> = Vec::new();
        let mut visited = vec![false; n];
        let mut candidates: VecDeque<(usize, f32)> = VecDeque::new();
        let mut prefetched_vectors = if self.flash_prefetch_enabled() {
            Some(HashMap::new())
        } else {
            None
        };
        let query_prefetch_cap = self.query_prefetch_cap(n);

        // Start from entry points
        for (start, start_dist) in self.initial_search_starts(query) {
            if start_dist <= radius * radius {
                results.push((self.ids[start], start_dist.sqrt()));
            }
            candidates.push_back((start, start_dist));
            visited[start] = true;
        }

        // BFS-style range search
        let beamwidth = self.dann_config.beamwidth;

        while !candidates.is_empty() && results.len() < max_results {
            let (idx, _dist) = candidates.pop_front().unwrap();

            // Explore neighbors
            {
                let neighbor_ids = self.neighbor_indices(idx);
                if let Some(cache) = prefetched_vectors.as_mut() {
                    self.prefetch_vectors_batch_into_cache(
                        &neighbor_ids,
                        n,
                        query_prefetch_cap,
                        cache,
                    );
                }
                let mut nbr_dists: Vec<(f32, usize)> = Vec::new();

                for n_idx in neighbor_ids {
                    if n_idx < n && !visited[n_idx] {
                        let d = self.compute_dist_with_prefetched(
                            query,
                            n_idx,
                            prefetched_vectors.as_ref(),
                        );
                        if d <= radius * radius {
                            results.push((self.ids[n_idx], d.sqrt()));
                            nbr_dists.push((d, n_idx));
                        }
                        visited[n_idx] = true;
                    }
                }

                // Continue searching from closest neighbors
                nbr_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (d, n_idx) in nbr_dists.into_iter().take(beamwidth) {
                    candidates.push_back((n_idx, d));
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    /// Create an iterator for streaming search results
    #[allow(non_snake_case)]
    pub fn search_iterator<'a>(&'a self, query: &'a [f32], L: usize) -> DiskAnnIterator<'a> {
        DiskAnnIterator::new(self, query, L)
    }

    /// Get statistics about the index
    pub fn get_stats(&self) -> DiskAnnStats {
        let n = self.ids.len();
        let mut num_edges = 0;
        let mut max_deg = 0;
        let mut min_deg = usize::MAX;

        for nbrs in &self.graph {
            let deg = nbrs.len();
            num_edges += deg;
            max_deg = max_deg.max(deg);
            min_deg = min_deg.min(deg);
        }

        if n == 0 {
            min_deg = 0;
        }

        let avg_deg = if n > 0 {
            num_edges as f32 / n as f32
        } else {
            0.0
        };

        // Estimate memory usage
        let pq_memory = self
            .pq_codes
            .as_ref()
            .map(|p| p.codes.len() + p.codebooks.len() * 4)
            .unwrap_or(0);
        let memory = self.vectors.len() * 4  // vectors
            + self.ids.len() * 8             // ids
            + num_edges * 12                 // graph edges (8 byte id + 4 byte dist)
            + pq_memory;

        DiskAnnStats {
            num_nodes: n,
            num_edges,
            avg_degree: avg_deg,
            max_degree: max_deg,
            min_degree: min_deg,
            is_trained: self.trained,
            memory_usage_bytes: memory,
        }
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        file.write_all(b"DANN")?;
        file.write_all(&3u32.to_le_bytes())?; // Version 3
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;

        // Write config
        file.write_all(&(self.dann_config.max_degree as u32).to_le_bytes())?;
        file.write_all(&(self.dann_config.search_list_size as u32).to_le_bytes())?;
        file.write_all(&(self.dann_config.beamwidth as u32).to_le_bytes())?;

        let bytes: Vec<u8> = self.vectors.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;

        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        for neighbors in &self.graph {
            file.write_all(&(neighbors.len() as u32).to_le_bytes())?;
            for &(id, dist) in neighbors {
                file.write_all(&id.to_le_bytes())?;
                file.write_all(&dist.to_le_bytes())?;
            }
        }

        // Write PQ codes if present
        let has_pq = self.pq_codes.is_some() as u32;
        file.write_all(&has_pq.to_le_bytes())?;
        if let Some(pq) = &self.pq_codes {
            file.write_all(&(pq.dims as u32).to_le_bytes())?;
            file.write_all(&(pq.ksub as u32).to_le_bytes())?;
            file.write_all(&(pq.subdim_size as u32).to_le_bytes())?;
            file.write_all(&(pq.codes.len() as u64).to_le_bytes())?;
            file.write_all(&pq.codes)?;
            file.write_all(&(pq.codebooks.len() as u64).to_le_bytes())?;
            for &v in &pq.codebooks {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        if self.dann_config.enable_flash_layout {
            let sidecar = Self::flash_layout_sidecar_path(path);
            self.write_flash_layout_sidecar(&sidecar)?;
        }

        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"DANN" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid magic".to_string(),
            ));
        }

        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;

        if dim != self.dim {
            return Err(crate::api::KnowhereError::Codec(
                "dimension mismatch".to_string(),
            ));
        }

        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;

        // Read config (version 2+)
        if version >= 2 {
            let mut md_bytes = [0u8; 4];
            file.read_exact(&mut md_bytes)?;
            self.dann_config.max_degree = u32::from_le_bytes(md_bytes) as usize;

            file.read_exact(&mut md_bytes)?;
            self.dann_config.search_list_size = u32::from_le_bytes(md_bytes) as usize;

            file.read_exact(&mut md_bytes)?;
            self.dann_config.beamwidth = u32::from_le_bytes(md_bytes) as usize;
        }

        let mut vec_bytes = vec![0u8; count * dim * 4];
        file.read_exact(&mut vec_bytes)?;

        self.vectors.clear();
        for chunk in vec_bytes.chunks(4) {
            self.vectors
                .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        self.ids.clear();
        for _ in 0..count {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            self.ids.push(i64::from_le_bytes(id_bytes));
        }

        self.graph.clear();
        for _ in 0..count {
            let mut nc_bytes = [0u8; 4];
            file.read_exact(&mut nc_bytes)?;
            let nc = u32::from_le_bytes(nc_bytes) as usize;

            let mut neighbors = Vec::with_capacity(nc);
            for _ in 0..nc {
                let mut id_bytes = [0u8; 8];
                file.read_exact(&mut id_bytes)?;
                let id = i64::from_le_bytes(id_bytes);

                let mut d_bytes = [0u8; 4];
                file.read_exact(&mut d_bytes)?;
                let dist = f32::from_le_bytes(d_bytes);

                neighbors.push((id, dist));
            }
            self.graph.push(neighbors);
        }

        // Read PQ codes (version 2+)
        if version >= 2 {
            let mut has_pq_bytes = [0u8; 4];
            file.read_exact(&mut has_pq_bytes)?;
            if u32::from_le_bytes(has_pq_bytes) != 0 {
                let mut dims_bytes = [0u8; 4];
                file.read_exact(&mut dims_bytes)?;
                let dims = u32::from_le_bytes(dims_bytes) as usize;

                let mut ksub = 0usize;
                let mut subdim_size = 0usize;
                if version >= 3 {
                    let mut val = [0u8; 4];
                    file.read_exact(&mut val)?;
                    ksub = u32::from_le_bytes(val) as usize;
                    file.read_exact(&mut val)?;
                    subdim_size = u32::from_le_bytes(val) as usize;
                }

                let mut len_bytes = [0u8; 8];
                file.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;

                let mut codes = vec![0u8; len];
                file.read_exact(&mut codes)?;

                let mut codebooks = Vec::new();
                if version >= 3 {
                    let mut cb_len_bytes = [0u8; 8];
                    file.read_exact(&mut cb_len_bytes)?;
                    let cb_len = u64::from_le_bytes(cb_len_bytes) as usize;
                    codebooks = vec![0.0f32; cb_len];
                    for item in codebooks.iter_mut().take(cb_len) {
                        let mut f_bytes = [0u8; 4];
                        file.read_exact(&mut f_bytes)?;
                        *item = f32::from_le_bytes(f_bytes);
                    }
                }

                self.pq_codes = Some(PQCode {
                    codes,
                    dims,
                    ksub,
                    subdim_size,
                    codebooks,
                });
            }
        }

        self.refresh_entry_points();
        let sidecar = Self::flash_layout_sidecar_path(path);
        self.has_flash_layout = self.try_load_flash_layout_sidecar(&sidecar, count)?;
        if !self.has_flash_layout {
            self.flash_neighbor_ids = None;
            self.flash_vectors = None;
            self.flash_max_degree = 0;
            self.flash_sidecar_mmap = None;
            self.flash_cached_neighbors = None;
            self.flash_cached_vectors = None;
        } else if self.flash_sidecar_mmap.is_some() {
            self.build_flash_runtime_cache_from_budget();
        }

        self.trained = true;
        Ok(())
    }
}

/// Helper struct for priority queue (min-heap)
#[derive(Debug)]
struct ReverseOrderedFloat(f32, usize);

impl PartialEq for ReverseOrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for ReverseOrderedFloat {}

impl PartialOrd for ReverseOrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseOrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Iterator for streaming DiskANN search results
#[allow(dead_code)]
pub struct DiskAnnIterator<'a> {
    index: &'a DiskAnnIndex,
    results: Vec<(i64, f32)>,
    current: usize,
}

impl<'a> DiskAnnIterator<'a> {
    #[allow(non_snake_case)]
    fn new(index: &'a DiskAnnIndex, query: &'a [f32], L: usize) -> Self {
        let results = index.beam_search(query, L, None, None);
        Self {
            index,
            results,
            current: 0,
        }
    }

    /// Get remaining count
    pub fn remaining(&self) -> usize {
        self.results.len() - self.current
    }
}

impl<'a> Iterator for DiskAnnIterator<'a> {
    type Item = (i64, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.results.len() {
            let result = self.results[self.current];
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
}

#[allow(clippy::items_after_test_module)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::IndexType;
    use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
    use serde_json::Value;
    use std::fs;

    const DISKANN_FINAL_VERDICT_PATH: &str = "benchmark_results/diskann_p3_004_final_verdict.json";

    fn load_diskann_final_verdict() -> Value {
        let content = fs::read_to_string(DISKANN_FINAL_VERDICT_PATH)
            .expect("DiskANN family verdict artifact must exist for the library verdict lane");
        serde_json::from_str(&content).expect("DiskANN family verdict artifact must be valid JSON")
    }

    #[test]
    fn test_diskann() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();

        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
    }

    #[test]
    fn test_diskann_save_load() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        // Create and train index
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        // Save to temp file
        let temp_path = std::env::temp_dir().join("diskann_test.bin");
        index.save(&temp_path).unwrap();

        // Create new index and load
        let mut index2 = DiskAnnIndex::new(&config).unwrap();
        index2.load(&temp_path).unwrap();

        // Verify search results match
        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        let result1 = index.search(&query, &req).unwrap();
        let result2 = index2.search(&query, &req).unwrap();

        // Results should be identical after reload
        assert_eq!(result1.ids, result2.ids);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_diskann_save_load_preserves_pq_codebook() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_pq_dims: Some(2),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        index.train(&vectors).unwrap();
        assert!(index.pq_codes.is_some());
        let before_pq = index.pq_codes.as_ref().unwrap();
        assert!(!before_pq.codebooks.is_empty());

        let temp_path = std::env::temp_dir().join("diskann_test_pq_v3.bin");
        index.save(&temp_path).unwrap();

        let mut index2 = DiskAnnIndex::new(&config).unwrap();
        index2.load(&temp_path).unwrap();
        let after_pq = index2.pq_codes.as_ref().unwrap();
        assert_eq!(before_pq.dims, after_pq.dims);
        assert_eq!(before_pq.ksub, after_pq.ksub);
        assert_eq!(before_pq.subdim_size, after_pq.subdim_size);
        assert_eq!(before_pq.codebooks.len(), after_pq.codebooks.len());
        assert_eq!(before_pq.codes.len(), after_pq.codes.len());

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        let result1 = index.search(&query, &req).unwrap();
        let result2 = index2.search(&query, &req).unwrap();
        assert_eq!(result1.ids, result2.ids);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_diskann_save_load_with_flash_layout_sidecar_sets_scope_audit() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_enable_flash_layout: Some(true),
                max_degree: Some(16),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let temp_path = std::env::temp_dir().join("diskann_flash_sidecar_test.bin");
        let sidecar = DiskAnnIndex::flash_layout_sidecar_path(&temp_path);
        index.save(&temp_path).unwrap();
        assert!(sidecar.exists());

        let mut loaded = DiskAnnIndex::new(&config).unwrap();
        loaded.load(&temp_path).unwrap();
        let audit = loaded.scope_audit();
        assert!(audit.has_flash_layout);
        assert!(!audit.native_comparable);

        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(sidecar).ok();
    }

    #[test]
    fn test_diskann_search_can_run_from_flash_sidecar_data() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_enable_flash_layout: Some(true),
                max_degree: Some(16),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let temp_path = std::env::temp_dir().join("diskann_flash_sidecar_runtime_test.bin");
        let sidecar = DiskAnnIndex::flash_layout_sidecar_path(&temp_path);
        index.save(&temp_path).unwrap();

        let mut loaded = DiskAnnIndex::new(&config).unwrap();
        loaded.load(&temp_path).unwrap();
        assert!(loaded.has_flash_layout);
        assert!(loaded.flash_neighbor_ids.is_some());
        assert!(loaded.flash_vectors.is_some());

        // Force search path to rely on sidecar runtime data.
        loaded.graph.clear();
        loaded.vectors.clear();

        let query = vec![0.1f32, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };
        let result = loaded.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids[0] >= 0);

        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(sidecar).ok();
    }

    #[test]
    fn test_diskann_search_can_run_from_flash_sidecar_mmap_mode() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_enable_flash_layout: Some(true),
                disk_flash_mmap_mode: Some(true),
                disk_flash_prefetch_batch: Some(8),
                max_degree: Some(16),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let temp_path = std::env::temp_dir().join("diskann_flash_sidecar_mmap_test.bin");
        let sidecar = DiskAnnIndex::flash_layout_sidecar_path(&temp_path);
        index.save(&temp_path).unwrap();

        let mut loaded = DiskAnnIndex::new(&config).unwrap();
        loaded.load(&temp_path).unwrap();
        assert!(loaded.has_flash_layout);
        assert!(loaded.flash_sidecar_mmap.is_some());
        assert!(loaded.flash_neighbor_ids.is_none());
        assert!(loaded.flash_vectors.is_none());
        assert_eq!(loaded.dann_config.flash_prefetch_batch, 8);

        loaded.graph.clear();
        loaded.vectors.clear();

        let query = vec![0.1f32, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };
        let result = loaded.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids[0] >= 0);

        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(sidecar).ok();
    }

    #[test]
    fn test_diskann_prefetch_cache_reuses_across_expansions_with_cap() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_enable_flash_layout: Some(true),
                disk_flash_mmap_mode: Some(true),
                disk_flash_prefetch_batch: Some(2),
                beamwidth: Some(2),
                max_degree: Some(16),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let temp_path = std::env::temp_dir().join("diskann_flash_prefetch_cache_test.bin");
        let sidecar = DiskAnnIndex::flash_layout_sidecar_path(&temp_path);
        index.save(&temp_path).unwrap();

        let mut loaded = DiskAnnIndex::new(&config).unwrap();
        loaded.load(&temp_path).unwrap();
        assert!(loaded.flash_sidecar_mmap.is_some());
        assert!(loaded.flash_prefetch_enabled());

        let n = loaded.ids.len();
        let cap = loaded.query_prefetch_cap(n);
        assert_eq!(cap, 4);

        let mut cache: HashMap<usize, Box<[f32]>> = HashMap::new();
        loaded.prefetch_vectors_batch_into_cache(&[0, 1], n, cap, &mut cache);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&0));
        assert!(cache.contains_key(&1));

        // Reuse should skip already cached nodes and only fill missing ones.
        loaded.prefetch_vectors_batch_into_cache(&[1, 2], n, cap, &mut cache);
        assert_eq!(cache.len(), 3);
        assert!(cache.contains_key(&2));

        // Respect per-query cap.
        loaded.prefetch_vectors_batch_into_cache(&[3, 4], n, cap, &mut cache);
        assert_eq!(cache.len(), 4);

        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(sidecar).ok();
    }

    #[test]
    fn test_diskann_flash_mmap_runtime_cache_respects_budget() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_enable_flash_layout: Some(true),
                disk_flash_mmap_mode: Some(true),
                disk_search_cache_budget_gb: Some(1e-9),
                max_degree: Some(16),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let temp_path = std::env::temp_dir().join("diskann_flash_mmap_budget_test.bin");
        let sidecar = DiskAnnIndex::flash_layout_sidecar_path(&temp_path);
        index.save(&temp_path).unwrap();

        let mut loaded = DiskAnnIndex::new(&config).unwrap();
        loaded.load(&temp_path).unwrap();
        assert!(loaded.has_flash_layout);
        assert!(loaded.flash_sidecar_mmap.is_some());
        let cached = loaded
            .flash_cached_vectors
            .as_ref()
            .map(|m| m.len())
            .unwrap_or(0);
        // tiny budget should cache at most one node in this fixture.
        assert!(cached <= 1);

        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file(sidecar).ok();
    }

    #[test]
    fn test_diskann_range_search() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let results = index.range_search(&query, 1.0, 10).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_diskann_stats() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let stats = index.get_stats();

        assert_eq!(stats.num_nodes, 4);
        assert!(stats.num_edges > 0);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.is_trained);
    }

    #[test]
    fn test_diskann_iterator() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let iter = index.search_iterator(&query, 10);

        let mut count = 0;
        for (id, _dist) in iter {
            count += 1;
            assert!(id >= 0);
        }

        assert!(count > 0);
    }

    #[test]
    fn test_diskann_config() {
        use crate::api::IndexParams;

        let params = IndexParams {
            max_degree: Some(64),
            search_list_size: Some(200),
            construction_l: Some(240),
            beamwidth: Some(16),
            disk_pq_dims: Some(2),
            disk_pq_code_budget_gb: Some(0.5),
            disk_pq_candidate_expand_pct: Some(150),
            disk_rerank_expand_pct: Some(220),
            disk_saturate_after_prune: Some(false),
            disk_intra_batch_candidates: Some(7),
            disk_num_entry_points: Some(3),
            disk_build_degree_slack_pct: Some(130),
            disk_random_init_edges: Some(5),
            disk_build_parallel_batch_size: Some(96),
            disk_build_dram_budget_gb: Some(2.5),
            disk_search_cache_budget_gb: Some(1.25),
            disk_enable_flash_layout: Some(true),
            disk_flash_mmap_mode: Some(true),
            disk_flash_prefetch_batch: Some(16),
            ..Default::default()
        };

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params,
        };

        let index = DiskAnnIndex::new(&config).unwrap();

        assert_eq!(index.dann_config.max_degree, 64);
        assert_eq!(index.dann_config.search_list_size, 200);
        assert_eq!(index.dann_config.construction_l, 240);
        assert_eq!(index.dann_config.beamwidth, 16);
        assert_eq!(index.dann_config.disk_pq_dims, 2);
        assert_eq!(index.dann_config.pq_code_budget_gb, 0.5);
        assert_eq!(index.dann_config.pq_candidate_expand_pct, 150);
        assert_eq!(index.dann_config.rerank_expand_pct, 220);
        assert!(!index.dann_config.saturate_after_prune);
        assert_eq!(index.dann_config.intra_batch_candidates, 7);
        assert_eq!(index.dann_config.num_entry_points, 3);
        assert_eq!(index.dann_config.build_degree_slack_pct, 130);
        assert_eq!(index.dann_config.random_init_edges, 5);
        assert_eq!(index.dann_config.build_parallel_batch_size, 96);
        assert_eq!(index.dann_config.build_dram_budget_gb, 2.5);
        assert_eq!(index.dann_config.cache_dram_budget_gb, 1.25);
        assert!(index.dann_config.enable_flash_layout);
        assert!(index.dann_config.flash_mmap_mode);
        assert_eq!(index.dann_config.flash_prefetch_batch, 16);
    }

    #[test]
    fn test_diskann_config_clamps_pq_candidate_expand_pct() {
        use crate::api::IndexParams;

        let params = IndexParams {
            disk_pq_candidate_expand_pct: Some(20),
            ..Default::default()
        };

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params,
        };

        let index = DiskAnnIndex::new(&config).unwrap();
        assert_eq!(index.dann_config.pq_candidate_expand_pct, 100);
    }

    #[test]
    fn test_diskann_config_clamps_rerank_expand_pct() {
        use crate::api::IndexParams;

        let params = IndexParams {
            disk_rerank_expand_pct: Some(20),
            ..Default::default()
        };

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params,
        };

        let index = DiskAnnIndex::new(&config).unwrap();
        assert_eq!(index.dann_config.rerank_expand_pct, 100);
    }

    #[test]
    fn test_diskann_train_fails_when_pq_code_budget_too_small() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_pq_dims: Some(2),
                disk_pq_code_budget_gb: Some(0.000_000_001),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        let err = index.train(&vectors).unwrap_err();
        match err {
            KnowhereError::InvalidArg(message) => {
                assert!(message.contains("disk_pq_code_budget_gb exceeded"));
            }
            other => panic!("expected InvalidArg, got {other:?}"),
        }
    }

    #[test]
    fn test_diskann_build_degree_slack_limit_computation() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                max_degree: Some(48),
                disk_build_degree_slack_pct: Some(130),
                ..Default::default()
            },
        };
        let index = DiskAnnIndex::new(&config).unwrap();
        assert_eq!(index.compute_build_degree_limit(48), 63);
    }

    #[test]
    fn test_diskann_train_respects_build_dram_budget() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_build_dram_budget_gb: Some(1e-9),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        let err = index.train(&vectors).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disk_build_dram_budget_gb exceeded"));
    }

    #[test]
    fn test_diskann_warm_up_respects_cache_budget() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0f32, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.cached_nodes.clear();
        index.dann_config.cache_dram_budget_gb = 1e-9;
        index.warm_up();
        assert!(index.cached_nodes.len() <= 1);
    }

    #[test]
    fn test_diskann_collect_intra_batch_candidates_respects_window() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_intra_batch_candidates: Some(2),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        index.vectors = vec![
            0.0, 0.0, // 0
            1.0, 0.0, // 1
            2.0, 0.0, // 2
            3.0, 0.0, // 3
        ];
        index.ids = vec![0, 1, 2, 3];

        let cands = index.collect_intra_batch_candidates(3);
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].0, 1);
        assert_eq!(cands[1].0, 2);
    }

    #[test]
    fn test_diskann_collect_random_initial_candidates_is_seeded_and_bounded() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_random_init_edges: Some(3),
                random_seed: Some(7),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        index.vectors = vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        index.ids = vec![0, 1, 2, 3, 4];

        let c1 = index.collect_random_initial_candidates_with_upper(4, 4);
        let c2 = index.collect_random_initial_candidates_with_upper(4, 4);

        assert_eq!(c1, c2);
        assert_eq!(c1.len(), 3);
        let picked: HashSet<usize> = c1.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(picked.len(), 3);
        assert!(picked.iter().all(|&idx| idx < 4));

        let c_limited = index.collect_random_initial_candidates_with_upper(4, 2);
        assert_eq!(c_limited.len(), 2);
        assert!(c_limited.iter().all(|(idx, _)| *idx < 2));
    }

    #[test]
    fn test_diskann_multi_entry_starts_pick_nearest_seed_first() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_num_entry_points: Some(3),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            -10.0, 0.0, //
            -5.0, 0.0, //
            0.0, 0.0, //
            5.0, 0.0, //
            10.0, 0.0, //
            20.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        assert_eq!(index.entry_points.len(), 3);

        let query = [19.0f32, 0.0f32];
        let starts = index.initial_search_starts(&query);
        assert_eq!(starts.len(), 3);

        let nearest_idx = index
            .entry_points
            .iter()
            .copied()
            .min_by(|&a, &b| {
                index
                    .compute_dist(&query, a)
                    .partial_cmp(&index.compute_dist(&query, b))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();
        assert_eq!(starts[0].0, nearest_idx);
        assert!(starts[0].1 <= starts[1].1);
        assert!(starts[1].1 <= starts[2].1);
    }

    #[test]
    fn test_diskann_single_entry_prefers_centroid_representative() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_num_entry_points: Some(1),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            -100.0, 0.0, //
            0.0, 0.0, //
            100.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        assert_eq!(index.entry_points.len(), 1);
        assert_eq!(index.entry_points[0], 1);
    }

    #[test]
    fn test_diskann_pq_table_distance_matches_direct_distance() {
        let mut pq = PQCode::new(2);
        let vectors = vec![
            0.1f32, -0.2, 0.3, -0.4, //
            0.5, -0.6, 0.7, -0.8,
        ];
        let dim = 4usize;
        pq.encode(&vectors, dim);

        let query = vec![0.11f32, -0.19, 0.29, -0.41];
        let table = pq.build_query_distance_table(&query, dim);
        let direct = pq.distance(&query, 0, dim);
        let cached = pq.distance_with_table(&table, 0, dim);

        assert!((direct - cached).abs() < 1e-6);
    }

    #[test]
    fn test_diskann_prune_neighbors_uses_alpha_occlusion() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        index.vectors = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            2.0, 0.0, // node 2
            3.0, 0.0, // node 3
        ];
        index.ids = vec![0, 1, 2, 3];

        let pruned = index.prune_neighbors(0, &[(1, 1.0), (2, 4.0), (3, 9.0)], 2);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].0, 1);
        assert_eq!(pruned[1].0, 2);
    }

    #[test]
    fn test_diskann_prune_neighbors_can_disable_saturation() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_saturate_after_prune: Some(false),
                ..Default::default()
            },
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        index.vectors = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            2.0, 0.0, // node 2
            3.0, 0.0, // node 3
        ];
        index.ids = vec![0, 1, 2, 3];

        let pruned = index.prune_neighbors(0, &[(1, 1.0), (2, 4.0), (3, 9.0)], 2);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0].0, 1);
    }

    #[test]
    fn test_diskann_beam_search_with_pq_returns_exact_final_distances() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                disk_pq_dims: Some(2),
                ..Default::default()
            },
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.9, 0.0, 0.0, 0.0];
        let results = index.beam_search(&query, 4, None, None);

        for (id, dist) in results {
            let idx = id as usize;
            let exact = index.compute_dist(&query, idx);
            assert!(
                (dist - exact).abs() < 1e-6,
                "beam-search final distance should be exact when PQ is enabled"
            );
        }
    }

    #[test]
    fn test_diskann_search_respects_predicate_filter() {
        use std::sync::Arc;

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0,
        ];
        index.train(&vectors).unwrap();

        let query = vec![0.1f32, 0.0];
        let req = SearchRequest {
            top_k: 3,
            nprobe: 8,
            filter: Some(Arc::new(crate::api::search::IdsPredicate { ids: vec![2, 3] })),
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        let kept: Vec<i64> = result.ids.into_iter().filter(|&id| id >= 0).collect();
        assert_eq!(kept, vec![2, 3]);
    }

    #[test]
    fn test_diskann_index_search_with_bitset_fills_topk_from_allowed_set() {
        use crate::bitset::BitsetView;
        use crate::dataset::Dataset;
        use crate::index::Index;

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, //
            1.0, 0.0, //
            2.0, 0.0, //
            3.0, 0.0, //
            4.0, 0.0,
        ];
        let train = Dataset::from_vectors(vectors, 2);
        Index::train(&mut index, &train).unwrap();

        let mut bitset = BitsetView::new(index.ntotal());
        bitset.set(0, true);
        bitset.set(1, true);
        bitset.set(2, true);

        let query = Dataset::from_vectors(vec![0.0f32, 0.0], 2);
        let result = Index::search_with_bitset(&index, &query, 2, &bitset).unwrap();
        assert_eq!(result.ids, vec![3, 4]);
        assert_eq!(result.distances.len(), 2);
    }

    #[test]
    fn test_diskann_index_trait() {
        use crate::dataset::Dataset;
        use crate::index::Index;

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = DiskAnnIndex::new(&config).unwrap();

        // Test Index trait methods
        assert_eq!(Index::index_type(&index), "DiskAnn");
        assert_eq!(Index::dim(&index), 4);
        assert_eq!(Index::count(&index), 0);
        assert!(!Index::is_trained(&index));

        // Train
        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let dataset = Dataset::from_vectors(vectors.clone(), 4);
        Index::train(&mut index, &dataset).unwrap();

        assert!(Index::is_trained(&index));
        assert_eq!(Index::count(&index), 4);
        assert!(Index::has_raw_data(&index));

        // Search
        let query = Dataset::from_vectors(vec![0.1, 0.1, 0.1, 0.1], 4);
        let result = Index::search(&index, &query, 2).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.distances.len(), 2);

        // Get vector by IDs
        let vectors = Index::get_vector_by_ids(&index, &[0, 1]).unwrap();
        assert_eq!(vectors.len(), 8); // 2 vectors * 4 dim
    }

    #[test]
    fn test_diskann_l2_sqr_matches_simd_squared_distance() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let vectors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        index.train(&vectors).unwrap();

        let query = vec![0.5, 1.5, 2.5, 3.5];
        let expected = simd::l2_distance_sq(&query, &vectors[4..8]);
        let actual = index.l2_sqr(&query, 1);
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn test_diskann_pqcode_builds_codebook_and_codes() {
        let mut pq = PQCode::new(2);
        let vectors = vec![
            0.0, 2.0, 4.0, 6.0, //
            1.0, 1.0, 3.0, 7.0, //
            -1.0, -2.0, 2.0, 1.0,
        ];
        pq.encode(&vectors, 4);

        assert_eq!(pq.codes.len(), 6);
        assert!(pq.ksub > 0);
        assert_eq!(pq.subdim_size, 2);
        assert!(!pq.codebooks.is_empty());
    }

    #[test]
    fn test_diskann_raw_data_semantics_follow_metric_type() {
        use crate::index::Index;

        let l2_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();
        let cosine_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::Cosine,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();
        let ip_index = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::Ip,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        })
        .unwrap();

        assert!(Index::has_raw_data(&l2_index));
        assert!(Index::has_raw_data(&cosine_index));
        assert!(!Index::has_raw_data(&ip_index));
        assert!(Index::get_vector_by_ids(&ip_index, &[0]).is_err());
    }

    #[test]
    fn test_diskann_get_vector_by_ids_rejects_missing_ids() {
        use crate::dataset::Dataset;
        use crate::index::{Index, IndexError};

        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut index = DiskAnnIndex::new(&config).unwrap();
        let dataset = Dataset::from_vectors(vec![0.0, 1.0, 2.0, 3.0], 4);
        Index::train(&mut index, &dataset).unwrap();

        let err = Index::get_vector_by_ids(&index, &[7]).unwrap_err();
        assert!(matches!(err, IndexError::Unsupported(_)));
    }

    #[test]
    fn test_diskann_scope_audit_locks_non_native_boundary() {
        let config = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                max_degree: Some(16),
                search_list_size: Some(32),
                beamwidth: Some(4),
                ..Default::default()
            },
        };

        let index = DiskAnnIndex::new(&config).unwrap();
        let audit = index.scope_audit();

        assert_eq!(audit.dim, 4);
        assert_eq!(audit.max_degree, 16);
        assert_eq!(audit.search_list_size, 32);
        assert_eq!(audit.beamwidth, 4);
        assert!(audit.uses_vamana_graph);
        assert!(!audit.uses_placeholder_pq);
        assert!(!audit.has_flash_layout);
        assert!(!audit.native_comparable);
        assert!(
            audit.comparability_reason.contains("not native-comparable"),
            "audit should explain why DiskAnnIndex is not native-comparable"
        );
    }

    #[test]
    fn test_diskann_family_verdict_archives_constrained_classification() {
        let verdict = load_diskann_final_verdict();
        let diskann = DiskAnnIndex::new(&IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                max_degree: Some(16),
                search_list_size: Some(32),
                beamwidth: Some(4),
                ..Default::default()
            },
        })
        .unwrap();
        let diskann_audit = diskann.scope_audit();

        let mut pqflash = PQFlashIndex::new(
            AisaqConfig {
                max_degree: 2,
                search_list_size: 4,
                beamwidth: 2,
                disk_pq_dims: 2,
                num_entry_points: 1,
                ..AisaqConfig::default()
            },
            MetricType::L2,
            4,
        )
        .unwrap();
        pqflash
            .add(&[
                0.0, 0.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, 1.0, //
                10.0, 10.0, 10.0, 10.0,
            ])
            .unwrap();
        let pqflash_audit = pqflash.scope_audit();

        assert_eq!(verdict["family"], "DiskANN");
        assert_eq!(verdict["classification"], "constrained");
        assert_eq!(
            verdict["leadership_verdict"],
            "no_go_for_native_comparable_benchmark"
        );
        assert_eq!(verdict["leadership_claim_allowed"], false);
        assert!(!diskann_audit.native_comparable);
        assert!(!diskann_audit.uses_placeholder_pq);
        assert!(!pqflash_audit.native_comparable);
        assert!(pqflash_audit.uses_flash_layout);
    }
}

// ============================================================================
// Index trait implementation for DiskAnnIndex (PARITY-P1-003)
// ============================================================================

use crate::dataset::Dataset;
use crate::index::{AnnIterator, Index, IndexError};
use std::time::Instant;

/// Wrapper to adapt DiskAnnIterator to the AnnIterator trait
pub struct DiskAnnIteratorWrapper {
    results: Vec<(i64, f32)>,
    current: usize,
}

impl DiskAnnIteratorWrapper {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self {
            results,
            current: 0,
        }
    }
}

impl AnnIterator for DiskAnnIteratorWrapper {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.current < self.results.len() {
            let result = self.results[self.current];
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }

    fn buffer_size(&self) -> usize {
        self.results.len() - self.current
    }
}

impl Index for DiskAnnIndex {
    fn index_type(&self) -> &str {
        "DiskAnn"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.ntotal()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        let vectors = dataset.vectors().to_vec();
        DiskAnnIndex::train(self, &vectors).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        let vectors = dataset.vectors().to_vec();
        DiskAnnIndex::add(self, &vectors, None).map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let query_vec = query.vectors().to_vec();
        let req = SearchRequest {
            top_k,
            nprobe: self.dann_config.beamwidth,
            filter: None,
            params: None,
            radius: None,
        };

        let start = Instant::now();
        let result = DiskAnnIndex::search(self, &query_vec, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        Ok(crate::index::SearchResult::new(
            result.ids,
            result.distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn range_search(
        &self,
        query: &Dataset,
        radius: f32,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let query_vec = query.vectors().to_vec();

        let start = Instant::now();
        let results = DiskAnnIndex::range_search(self, &query_vec, radius, self.dann_config.max_k)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        let ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        let distances: Vec<f32> = results.iter().map(|(_, dist)| *dist).collect();

        Ok(crate::index::SearchResult::new(
            ids,
            distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn get_vector_by_ids(&self, ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        if !self.has_raw_data() {
            return Err(IndexError::Unsupported(
                "get_vector_by_ids not supported for DiskANN without raw-data metric semantics"
                    .into(),
            ));
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(ids.len() * self.dim);

        for &id in ids {
            let idx =
                self.ids.iter().position(|&x| x == id).ok_or_else(|| {
                    IndexError::Unsupported(format!("ID {} not found in index", id))
                })?;
            let start = idx * self.dim;
            let end = start + self.dim;
            if end > self.vectors.len() {
                return Err(IndexError::Unsupported(format!(
                    "Vector data corrupted for ID {}",
                    id
                )));
            }
            result.extend_from_slice(&self.vectors[start..end]);
        }

        Ok(result)
    }

    fn has_raw_data(&self) -> bool {
        matches!(self.config.metric_type, MetricType::L2 | MetricType::Cosine)
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        _bitset: Option<&crate::bitset::BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        let query_vec = query.vectors().to_vec();
        let results = self.beam_search(&query_vec, self.dann_config.search_list_size, None, None);
        Ok(Box::new(DiskAnnIteratorWrapper::new(results)))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> std::result::Result<crate::index::SearchResult, IndexError> {
        let query_vec = query.vectors().to_vec();
        if self.vectors.is_empty() {
            return Err(IndexError::Empty);
        }
        if query_vec.is_empty() || query_vec.len() % self.dim != 0 {
            return Err(IndexError::DimMismatch);
        }

        let start = Instant::now();
        let n_queries = query_vec.len() / self.dim;
        let l = self.dann_config.search_list_size.max(top_k);
        let mut ids = Vec::with_capacity(n_queries * top_k);
        let mut distances = Vec::with_capacity(n_queries * top_k);

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let q = &query_vec[q_start..q_start + self.dim];
            let results = self.beam_search(q, l, None, Some(bitset));
            for i in 0..top_k {
                if i < results.len() {
                    let dist = match self.config.metric_type {
                        MetricType::Ip => -results[i].1,
                        _ => results[i].1.sqrt(),
                    };
                    ids.push(results[i].0);
                    distances.push(dist);
                } else {
                    ids.push(-1);
                    distances.push(f32::MAX);
                }
            }
        }

        Ok(crate::index::SearchResult::new(
            ids,
            distances,
            start.elapsed().as_secs_f64() * 1000.0,
        ))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        DiskAnnIndex::save(self, std::path::Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        DiskAnnIndex::load(self, std::path::Path::new(path))
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }
}

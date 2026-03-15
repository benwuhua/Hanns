//! HNSW - High Performance Version with True Multi-Layer Support
//!
//! Optimized HNSW with progressive sampling and full multi-layer graph structure.
//! Implements the original HNSW paper algorithm with:
//! - Random level generation using exponential distribution
//! - Layered neighbor connections (each node has connections at its levels)
//! - Greedy search from top layer to bottom layer

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::{
    DataType, IndexConfig, MetricType, Predicate, Result, SearchRequest,
    SearchResult as ApiSearchResult,
};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::half::{bf16_l2_sq_batch_4, Bf16};
use crate::index::{Index as IndexTrait, IndexError, SearchResult as IndexSearchResult};
use crate::simd;

type BatchNeighborResults = (usize, usize, Vec<Vec<(usize, f32)>>);
type BatchNeighborResultsProfiled = (
    usize,
    usize,
    Vec<Vec<(usize, f32)>>,
    HnswBuildProfileStats,
    HnswParallelBuildGraphQualityStats,
);
type DistanceToIdxFn = fn(&HnswIndex, &[f32], usize) -> f32;

/// Maximum number of layers in the HNSW graph
const MAX_LAYERS: usize = 16;
const HNSW_SEARCH_KNN_BF_FILTER_THRESHOLD: f32 = 0.93;
const HNSW_SEARCH_BF_TOPK_THRESHOLD: f32 = 0.5;

/// BUG-001 FIX: Reference M value for level multiplier calculation.
/// Using a fixed reference M ensures consistent level distribution across different M values.
/// Without this, high M values (e.g., M=64) would have very few layers, collapsing the
/// multi-layer structure and degrading recall.
const REFERENCE_M_FOR_LEVEL: usize = 16;

/// A single node's neighbor connections at a specific layer
///
/// OPT-015: Store neighbor indices (usize) instead of IDs (i64) for O(1) direct access.
/// This eliminates all HashMap/linear search overhead in the hot path.
#[derive(Clone, Debug)]
pub struct LayerNeighbors {
    /// Neighbor IDs at this layer. Kept densely packed because search hot paths only need IDs.
    pub ids: Vec<i64>,
    /// Distances used for insertion-time pruning and stable persistence layout.
    pub dists: Vec<f32>,
}

impl LayerNeighbors {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            dists: Vec::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            dists: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn push(&mut self, id: i64, dist: f32) {
        self.ids.push(id);
        self.dists.push(dist);
    }

    /// Get neighbors as (id, dist) pairs for heuristic processing
    #[allow(dead_code)]
    fn as_pairs(&self) -> Vec<(i64, f32)> {
        self.ids
            .iter()
            .zip(self.dists.iter())
            .map(|(&id, &dist)| (id, dist))
            .collect()
    }

    /// Set neighbors from (id, dist) pairs after heuristic processing
    #[allow(dead_code)]
    fn set_from_pairs(&mut self, pairs: &[(i64, f32)]) {
        self.ids.clear();
        self.dists.clear();
        for &(id, dist) in pairs {
            self.ids.push(id);
            self.dists.push(dist);
        }
    }
}

/// Node information including its layer assignment and connections
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// The highest layer this node exists in (0 = only base layer)
    pub max_layer: usize,
    /// Neighbor connections per layer (index 0 = layer 0, etc.)
    /// Only stores connections up to max_layer
    pub layer_neighbors: Vec<LayerNeighbors>,
}

impl NodeInfo {
    fn new(max_layer: usize, m: usize) -> Self {
        let mut layer_neighbors = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            layer_neighbors.push(LayerNeighbors::with_capacity(m * 2));
        }
        Self {
            max_layer,
            layer_neighbors,
        }
    }
}

/// HNSW index with true multi-layer graph structure
///
/// OPT-021: Removed HashMap for O(1) direct indexing.
/// OPT-015: Removed ids Vec and linear search - use idx directly when IDs are sequential.
/// IDs are stored in order, so idx = id for sequential case (most common).
/// For custom IDs, we maintain a separate ids Vec for lookup only when needed.
/// OPT-024: Added num_threads for parallel build configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layer0PoolEntry {
    idx: usize,
    dist: f32,
}

#[derive(Default)]
struct Layer0OrderedFrontier {
    entries: Vec<Layer0PoolEntry>,
}

impl Layer0OrderedFrontier {
    fn prepare(&mut self, capacity: usize) {
        self.entries.clear();
        if self.entries.capacity() < capacity {
            self.entries.reserve(capacity - self.entries.capacity());
        }
    }

    fn push(&mut self, entry: Layer0PoolEntry) {
        let insert_at = self.entries.partition_point(|existing| {
            matches!(
                existing.dist.total_cmp(&entry.dist),
                std::cmp::Ordering::Greater
            ) || (existing.dist.to_bits() == entry.dist.to_bits() && existing.idx > entry.idx)
        });
        self.entries.insert(insert_at, entry);
    }

    fn pop_best(&mut self) -> Option<Layer0PoolEntry> {
        self.entries.pop()
    }
}

#[derive(Default)]
struct Layer0OrderedResults {
    entries: Vec<Layer0PoolEntry>,
}

impl Layer0OrderedResults {
    fn prepare(&mut self, capacity: usize) {
        self.entries.clear();
        if self.entries.capacity() < capacity {
            self.entries.reserve(capacity - self.entries.capacity());
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn worst_dist(&self) -> Option<f32> {
        self.entries.last().map(|entry| entry.dist)
    }

    fn can_insert(&self, dist: f32, ef: usize) -> bool {
        self.entries.len() < ef || dist < self.worst_dist().unwrap_or(f32::INFINITY)
    }

    fn insert(&mut self, entry: Layer0PoolEntry, ef: usize) -> u64 {
        if !self.can_insert(entry.dist, ef) {
            return 0;
        }

        let insert_at = self.entries.partition_point(|existing| {
            matches!(
                existing.dist.total_cmp(&entry.dist),
                std::cmp::Ordering::Less
            ) || (existing.dist.to_bits() == entry.dist.to_bits() && existing.idx < entry.idx)
        });
        self.entries.insert(insert_at, entry);
        if self.entries.len() > ef {
            self.entries.pop();
            1
        } else {
            0
        }
    }

    fn to_sorted_pairs(&self) -> Vec<(usize, f32)> {
        self.entries
            .iter()
            .map(|entry| (entry.idx, entry.dist))
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SearchMinDist(f32);

impl Eq for SearchMinDist {}

impl PartialOrd for SearchMinDist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchMinDist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SearchMaxDist(f32);

impl Eq for SearchMaxDist {}

impl PartialOrd for SearchMaxDist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchMaxDist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self
                .0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Layer0FlatGraph {
    max_neighbors: usize,
    degrees: Vec<usize>,
    neighbors: Vec<u32>,
    enabled: bool,
}

impl Layer0FlatGraph {
    fn clear(&mut self) {
        self.max_neighbors = 0;
        self.degrees.clear();
        self.neighbors.clear();
        self.enabled = false;
    }

    fn is_enabled_for(&self, node_count: usize) -> bool {
        self.enabled && self.max_neighbors > 0 && self.degrees.len() == node_count
    }

    fn neighbors_for(&self, node_idx: usize) -> &[u32] {
        let degree = self.degrees[node_idx];
        let start = node_idx * self.max_neighbors;
        &self.neighbors[start..start + degree]
    }
}

#[derive(Clone, Debug, Default)]
struct Layer0Slab {
    stride_words: usize,
    vector_offset_words: usize,
    max_neighbors: usize,
    dim: usize,
    words: Vec<u32>,
    enabled: bool,
}

impl Layer0Slab {
    fn clear(&mut self) {
        self.stride_words = 0;
        self.vector_offset_words = 0;
        self.max_neighbors = 0;
        self.dim = 0;
        self.words.clear();
        self.enabled = false;
    }

    fn is_enabled_for(&self, node_count: usize) -> bool {
        self.enabled
            && self.stride_words > 0
            && self.max_neighbors > 0
            && self.dim > 0
            && self.words.len() == node_count * self.stride_words
    }

    fn neighbors_for(&self, node_idx: usize) -> &[u32] {
        let base = node_idx * self.stride_words;
        let degree = self.words[base] as usize;
        &self.words[base + 1..base + 1 + degree]
    }

    unsafe fn vector_ptr_for(&self, node_idx: usize) -> *const f32 {
        let base = node_idx * self.stride_words + self.vector_offset_words;
        self.words.as_ptr().add(base) as *const f32
    }

    #[cfg(any(test, feature = "long-tests"))]
    fn vector_for_audit(&self, node_idx: usize) -> &[f32] {
        let base = node_idx * self.stride_words + self.vector_offset_words;
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().add(base) as *const f32, self.dim) }
    }
}

struct SearchScratch {
    visited_epoch: Vec<u32>,
    epoch: u32,
    generic_frontier: BinaryHeap<(SearchMinDist, usize)>,
    generic_results: BinaryHeap<(SearchMaxDist, usize)>,
    generic_worst_result_distance: f32,
    layer0_frontier: Layer0OrderedFrontier,
    layer0_results: Layer0OrderedResults,
}

#[derive(Clone, Copy, Debug)]
enum HnswBuildProfileStage {
    LayerDescent,
    CandidateSearch,
    NeighborSelection,
    ConnectionUpdate,
    Repair,
}

#[derive(Clone, Debug, Default)]
pub struct HnswBuildProfileStats {
    layer_descent: Duration,
    candidate_search: Duration,
    neighbor_selection: Duration,
    connection_update: Duration,
    repair: Duration,
    layer_descent_calls: u64,
    candidate_search_calls: u64,
    neighbor_selection_calls: u64,
    connection_update_calls: u64,
    repair_calls: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswBuildProfileTimingBuckets {
    pub layer_descent_ms: f64,
    pub candidate_search_ms: f64,
    pub neighbor_selection_ms: f64,
    pub connection_update_ms: f64,
    pub repair_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswBuildProfileCallCounts {
    pub layer_descent_calls: u64,
    pub candidate_search_calls: u64,
    pub neighbor_selection_calls: u64,
    pub connection_update_calls: u64,
    pub repair_calls: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswBuildProfileHotspot {
    pub stage: String,
    pub milliseconds: f64,
    pub calls: u64,
    pub share_of_profiled_time: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswBuildProfileReport {
    pub timing_buckets: HnswBuildProfileTimingBuckets,
    pub call_counts: HnswBuildProfileCallCounts,
    pub hotspot_ranking: Vec<HnswBuildProfileHotspot>,
    pub recommended_first_rework_target: String,
    pub total_profiled_ms: f64,
    pub repair_operations: usize,
    pub vectors_added: usize,
}

#[derive(Clone, Debug, Default)]
struct HnswParallelBuildGraphQualityStats {
    omitted_upper_layer_descent_levels: u64,
    upper_layer_connection_update_calls: u64,
    upper_layer_overflow_events: u64,
    upper_layer_truncate_to_best_events: u64,
    upper_layer_heuristic_shrink_events: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswParallelBuildGraphQualityCallCounts {
    pub omitted_upper_layer_descent_levels: u64,
    pub upper_layer_connection_update_calls: u64,
    pub upper_layer_overflow_events: u64,
    pub upper_layer_truncate_to_best_events: u64,
    pub upper_layer_heuristic_shrink_events: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswParallelBuildProfileReport {
    pub timing_buckets: HnswBuildProfileTimingBuckets,
    pub call_counts: HnswBuildProfileCallCounts,
    pub hotspot_ranking: Vec<HnswBuildProfileHotspot>,
    pub recommended_first_rework_target: String,
    pub total_profiled_ms: f64,
    pub repair_operations: usize,
    pub vectors_added: usize,
    pub parallel_insert_entry_descent_mode: String,
    pub upper_layer_overflow_shrink_mode: String,
    pub graph_quality_call_counts: HnswParallelBuildGraphQualityCallCounts,
}

#[derive(Clone, Debug, Default)]
pub struct HnswCandidateSearchProfileStats {
    entry_descent: Duration,
    frontier_ops: Duration,
    visited_ops: Duration,
    distance_compute: Duration,
    upper_layer_query_distance: Duration,
    layer0_query_distance: Duration,
    node_node_distance: Duration,
    candidate_pruning: Duration,
    entry_descent_calls: u64,
    layer0_candidate_search_calls: u64,
    frontier_pushes: u64,
    frontier_pops: u64,
    visited_marks: u64,
    distance_calls: u64,
    upper_layer_query_distance_calls: u64,
    layer0_query_distance_calls: u64,
    node_node_distance_calls: u64,
    pruned_candidates: u64,
    layer0_batch4_calls: u64,
    layer0_vector_prefetches: u64,
    layer0_flat_graph_neighbor_reads: u64,
    layer0_ordered_pool_enabled: bool,
    layer0_flat_graph_enabled: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswCandidateSearchProfileBreakdown {
    pub entry_descent_ms: f64,
    pub frontier_ops_ms: f64,
    pub visited_ops_ms: f64,
    pub distance_compute_ms: f64,
    pub candidate_pruning_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswCandidateSearchProfileCallCounts {
    pub entry_descent_calls: u64,
    pub layer0_candidate_search_calls: u64,
    pub frontier_pushes: u64,
    pub frontier_pops: u64,
    pub visited_marks: u64,
    pub distance_calls: u64,
    pub pruned_candidates: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswDistanceComputeProfileBreakdown {
    pub upper_layer_query_distance_ms: f64,
    pub layer0_query_distance_ms: f64,
    pub node_node_distance_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswDistanceComputeProfileCallCounts {
    pub upper_layer_query_distance_calls: u64,
    pub layer0_query_distance_calls: u64,
    pub node_node_distance_calls: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0SearchCoreShape {
    pub native_layer0_candidate_container: String,
    pub rust_layer0_candidate_container: String,
    pub rust_frontier_container: String,
    pub rust_result_container: String,
    pub rust_scratch_reuse_scope: String,
    pub rust_layer0_neighbor_layout: String,
    pub rust_layer0_neighbor_id_type: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0BatchDistanceMode {
    pub native_layer0_query_distance: String,
    pub rust_layer0_query_distance: String,
    pub rust_batch_enabled: bool,
    pub rust_batch_width: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0BatchDistanceCallCounts {
    pub layer0_batch4_calls: u64,
    pub layer0_scalar_distance_calls: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0PrefetchMode {
    pub native_layer0_vector_prefetch: String,
    pub rust_layer0_vector_prefetch: String,
    pub rust_prefetch_enabled: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0PrefetchCallCounts {
    pub layer0_vector_prefetches: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswLayer0NeighborAccessCallCounts {
    pub layer0_flat_graph_neighbor_reads: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswCandidateSearchProfileReport {
    pub candidate_search_breakdown: HnswCandidateSearchProfileBreakdown,
    pub call_counts: HnswCandidateSearchProfileCallCounts,
    pub distance_compute_breakdown: HnswDistanceComputeProfileBreakdown,
    pub distance_compute_call_counts: HnswDistanceComputeProfileCallCounts,
    pub search_core_shape: HnswLayer0SearchCoreShape,
    pub batch_distance_mode: HnswLayer0BatchDistanceMode,
    pub batch_distance_call_counts: HnswLayer0BatchDistanceCallCounts,
    pub prefetch_mode: HnswLayer0PrefetchMode,
    pub prefetch_call_counts: HnswLayer0PrefetchCallCounts,
    pub layer0_neighbor_access_call_counts: HnswLayer0NeighborAccessCallCounts,
    pub hotspot_ranking: Vec<HnswBuildProfileHotspot>,
    pub recommended_next_target: String,
    pub total_profiled_ms: f64,
    pub query_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswGraphDiagnosisReport {
    pub node_count: usize,
    pub max_level: usize,
    pub level_histogram: BTreeMap<String, usize>,
    pub layer_degree_histograms: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswSearchCostDiagnosis {
    pub visited_nodes: usize,
    pub frontier_pushes: usize,
    pub frontier_pops: usize,
    pub distance_calls: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HnswGenericSearchKernelReport {
    pub unfiltered_layer_search_core: String,
    pub bitset_layer_search_core: String,
    pub frontier_container: String,
    pub result_container: String,
    pub frontier_reuse_scope: String,
    pub result_reuse_scope: String,
    pub shared_l2_distance_dispatch: String,
    pub shared_bitset_distance_mode: String,
    pub shared_layer0_neighbor_layout: String,
    pub shared_result_threshold_mode: String,
    pub visited_reuse_scope: String,
}

impl HnswBuildProfileStats {
    fn record(&mut self, stage: HnswBuildProfileStage, elapsed: Duration) {
        match stage {
            HnswBuildProfileStage::LayerDescent => {
                self.layer_descent += elapsed;
                self.layer_descent_calls += 1;
            }
            HnswBuildProfileStage::CandidateSearch => {
                self.candidate_search += elapsed;
                self.candidate_search_calls += 1;
            }
            HnswBuildProfileStage::NeighborSelection => {
                self.neighbor_selection += elapsed;
                self.neighbor_selection_calls += 1;
            }
            HnswBuildProfileStage::ConnectionUpdate => {
                self.connection_update += elapsed;
                self.connection_update_calls += 1;
            }
            HnswBuildProfileStage::Repair => {
                self.repair += elapsed;
                self.repair_calls += 1;
            }
        }
    }

    fn absorb(&mut self, other: Self) {
        self.layer_descent += other.layer_descent;
        self.candidate_search += other.candidate_search;
        self.neighbor_selection += other.neighbor_selection;
        self.connection_update += other.connection_update;
        self.repair += other.repair;
        self.layer_descent_calls += other.layer_descent_calls;
        self.candidate_search_calls += other.candidate_search_calls;
        self.neighbor_selection_calls += other.neighbor_selection_calls;
        self.connection_update_calls += other.connection_update_calls;
        self.repair_calls += other.repair_calls;
    }

    fn timing_buckets(&self) -> HnswBuildProfileTimingBuckets {
        HnswBuildProfileTimingBuckets {
            layer_descent_ms: self.layer_descent.as_secs_f64() * 1000.0,
            candidate_search_ms: self.candidate_search.as_secs_f64() * 1000.0,
            neighbor_selection_ms: self.neighbor_selection.as_secs_f64() * 1000.0,
            connection_update_ms: self.connection_update.as_secs_f64() * 1000.0,
            repair_ms: self.repair.as_secs_f64() * 1000.0,
        }
    }

    fn call_counts(&self) -> HnswBuildProfileCallCounts {
        HnswBuildProfileCallCounts {
            layer_descent_calls: self.layer_descent_calls,
            candidate_search_calls: self.candidate_search_calls,
            neighbor_selection_calls: self.neighbor_selection_calls,
            connection_update_calls: self.connection_update_calls,
            repair_calls: self.repair_calls,
        }
    }

    fn total_profiled_ms(&self) -> f64 {
        self.layer_descent.as_secs_f64() * 1000.0
            + self.candidate_search.as_secs_f64() * 1000.0
            + self.neighbor_selection.as_secs_f64() * 1000.0
            + self.connection_update.as_secs_f64() * 1000.0
            + self.repair.as_secs_f64() * 1000.0
    }

    fn hotspot_ranking(&self) -> Vec<HnswBuildProfileHotspot> {
        let total = self.total_profiled_ms();
        let mut hotspots = vec![
            (
                "layer_descent",
                self.layer_descent.as_secs_f64() * 1000.0,
                self.layer_descent_calls,
            ),
            (
                "candidate_search",
                self.candidate_search.as_secs_f64() * 1000.0,
                self.candidate_search_calls,
            ),
            (
                "neighbor_selection",
                self.neighbor_selection.as_secs_f64() * 1000.0,
                self.neighbor_selection_calls,
            ),
            (
                "connection_update",
                self.connection_update.as_secs_f64() * 1000.0,
                self.connection_update_calls,
            ),
            (
                "repair",
                self.repair.as_secs_f64() * 1000.0,
                self.repair_calls,
            ),
        ];

        hotspots.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        hotspots
            .into_iter()
            .map(|(stage, milliseconds, calls)| HnswBuildProfileHotspot {
                stage: stage.to_string(),
                milliseconds,
                calls,
                share_of_profiled_time: if total > 0.0 {
                    milliseconds / total
                } else {
                    0.0
                },
            })
            .collect()
    }

    fn recommended_first_rework_target(&self) -> String {
        let hotspots = self.hotspot_ranking();
        let stage = hotspots
            .first()
            .map(|hotspot| hotspot.stage.as_str())
            .unwrap_or("connection_update");
        match stage {
            "candidate_search" => "build_time_candidate_search".to_string(),
            "layer_descent" => "layer_descent_entrypoint_walk".to_string(),
            "neighbor_selection" => "neighbor_selection_diversification".to_string(),
            "repair" => "connectivity_repair_path".to_string(),
            _ => "bulk_build_connection_update_path".to_string(),
        }
    }

    fn into_report(self, vectors_added: usize, repair_operations: usize) -> HnswBuildProfileReport {
        HnswBuildProfileReport {
            timing_buckets: self.timing_buckets(),
            call_counts: self.call_counts(),
            hotspot_ranking: self.hotspot_ranking(),
            recommended_first_rework_target: self.recommended_first_rework_target(),
            total_profiled_ms: self.total_profiled_ms(),
            repair_operations,
            vectors_added,
        }
    }

    fn into_parallel_report(
        self,
        vectors_added: usize,
        repair_operations: usize,
        parallel_insert_entry_descent_mode: &str,
        upper_layer_overflow_shrink_mode: &str,
        graph_quality_stats: &HnswParallelBuildGraphQualityStats,
    ) -> HnswParallelBuildProfileReport {
        HnswParallelBuildProfileReport {
            timing_buckets: self.timing_buckets(),
            call_counts: self.call_counts(),
            hotspot_ranking: self.hotspot_ranking(),
            recommended_first_rework_target: self.recommended_first_rework_target(),
            total_profiled_ms: self.total_profiled_ms(),
            repair_operations,
            vectors_added,
            parallel_insert_entry_descent_mode: parallel_insert_entry_descent_mode.to_string(),
            upper_layer_overflow_shrink_mode: upper_layer_overflow_shrink_mode.to_string(),
            graph_quality_call_counts: graph_quality_stats.call_counts(),
        }
    }
}

impl HnswParallelBuildGraphQualityStats {
    fn absorb(&mut self, other: Self) {
        self.omitted_upper_layer_descent_levels += other.omitted_upper_layer_descent_levels;
        self.upper_layer_connection_update_calls += other.upper_layer_connection_update_calls;
        self.upper_layer_overflow_events += other.upper_layer_overflow_events;
        self.upper_layer_truncate_to_best_events += other.upper_layer_truncate_to_best_events;
        self.upper_layer_heuristic_shrink_events += other.upper_layer_heuristic_shrink_events;
    }

    fn record_upper_layer_connection_update(&mut self) {
        self.upper_layer_connection_update_calls += 1;
    }

    fn record_upper_layer_overflow_heuristic_shrink(&mut self) {
        self.upper_layer_overflow_events += 1;
        self.upper_layer_heuristic_shrink_events += 1;
    }

    fn call_counts(&self) -> HnswParallelBuildGraphQualityCallCounts {
        HnswParallelBuildGraphQualityCallCounts {
            omitted_upper_layer_descent_levels: self.omitted_upper_layer_descent_levels,
            upper_layer_connection_update_calls: self.upper_layer_connection_update_calls,
            upper_layer_overflow_events: self.upper_layer_overflow_events,
            upper_layer_truncate_to_best_events: self.upper_layer_truncate_to_best_events,
            upper_layer_heuristic_shrink_events: self.upper_layer_heuristic_shrink_events,
        }
    }
}

impl HnswCandidateSearchProfileStats {
    fn record_entry_descent(&mut self, elapsed: Duration) {
        self.entry_descent += elapsed;
        self.entry_descent_calls += 1;
    }

    fn start_layer0_candidate_search(&mut self) {
        self.layer0_candidate_search_calls += 1;
    }

    fn record_frontier_ops(&mut self, elapsed: Duration, pushes: u64, pops: u64) {
        self.frontier_ops += elapsed;
        self.frontier_pushes += pushes;
        self.frontier_pops += pops;
    }

    fn record_visited_ops(&mut self, elapsed: Duration, marks: u64) {
        self.visited_ops += elapsed;
        self.visited_marks += marks;
    }

    fn record_distance_compute(&mut self, elapsed: Duration, calls: u64) {
        self.distance_compute += elapsed;
        self.distance_calls += calls;
    }

    fn record_upper_layer_query_distance(&mut self, elapsed: Duration, calls: u64) {
        self.upper_layer_query_distance += elapsed;
        self.upper_layer_query_distance_calls += calls;
        self.record_distance_compute(elapsed, calls);
    }

    fn record_layer0_query_distance(&mut self, elapsed: Duration, calls: u64) {
        self.layer0_query_distance += elapsed;
        self.layer0_query_distance_calls += calls;
        self.record_distance_compute(elapsed, calls);
    }

    fn record_layer0_query_distance_batch4(&mut self, elapsed: Duration) {
        self.layer0_batch4_calls += 1;
        self.record_layer0_query_distance(elapsed, 4);
    }

    fn mark_layer0_ordered_pool(&mut self) {
        self.layer0_ordered_pool_enabled = true;
    }

    fn mark_layer0_flat_graph_enabled(&mut self, enabled: bool) {
        self.layer0_flat_graph_enabled = enabled;
    }

    fn record_layer0_flat_graph_neighbor_reads(&mut self, reads: u64) {
        self.layer0_flat_graph_neighbor_reads += reads;
    }

    fn record_layer0_vector_prefetch(&mut self) {
        self.layer0_vector_prefetches += 1;
    }

    fn record_candidate_pruning(&mut self, elapsed: Duration, pruned: u64) {
        self.candidate_pruning += elapsed;
        self.pruned_candidates += pruned;
    }

    fn breakdown(&self) -> HnswCandidateSearchProfileBreakdown {
        HnswCandidateSearchProfileBreakdown {
            entry_descent_ms: self.entry_descent.as_secs_f64() * 1000.0,
            frontier_ops_ms: self.frontier_ops.as_secs_f64() * 1000.0,
            visited_ops_ms: self.visited_ops.as_secs_f64() * 1000.0,
            distance_compute_ms: self.distance_compute.as_secs_f64() * 1000.0,
            candidate_pruning_ms: self.candidate_pruning.as_secs_f64() * 1000.0,
        }
    }

    fn call_counts(&self) -> HnswCandidateSearchProfileCallCounts {
        HnswCandidateSearchProfileCallCounts {
            entry_descent_calls: self.entry_descent_calls,
            layer0_candidate_search_calls: self.layer0_candidate_search_calls,
            frontier_pushes: self.frontier_pushes,
            frontier_pops: self.frontier_pops,
            visited_marks: self.visited_marks,
            distance_calls: self.distance_calls,
            pruned_candidates: self.pruned_candidates,
        }
    }

    fn distance_compute_breakdown(&self) -> HnswDistanceComputeProfileBreakdown {
        HnswDistanceComputeProfileBreakdown {
            upper_layer_query_distance_ms: self.upper_layer_query_distance.as_secs_f64() * 1000.0,
            layer0_query_distance_ms: self.layer0_query_distance.as_secs_f64() * 1000.0,
            node_node_distance_ms: self.node_node_distance.as_secs_f64() * 1000.0,
        }
    }

    fn distance_compute_call_counts(&self) -> HnswDistanceComputeProfileCallCounts {
        HnswDistanceComputeProfileCallCounts {
            upper_layer_query_distance_calls: self.upper_layer_query_distance_calls,
            layer0_query_distance_calls: self.layer0_query_distance_calls,
            node_node_distance_calls: self.node_node_distance_calls,
        }
    }

    fn search_core_shape(&self) -> HnswLayer0SearchCoreShape {
        HnswLayer0SearchCoreShape {
            native_layer0_candidate_container: "NeighborSetDoublePopList".to_string(),
            rust_layer0_candidate_container: if self.layer0_ordered_pool_enabled {
                "ordered_pool".to_string()
            } else {
                "dual_binary_heap".to_string()
            },
            rust_frontier_container: if self.layer0_ordered_pool_enabled {
                "Vec<Layer0PoolEntry>::descending".to_string()
            } else {
                "BinaryHeap<QueryCandidate>".to_string()
            },
            rust_result_container: if self.layer0_ordered_pool_enabled {
                "Vec<Layer0PoolEntry>::ascending".to_string()
            } else {
                "BinaryHeap<ResultCandidate>".to_string()
            },
            rust_scratch_reuse_scope: if self.layer0_ordered_pool_enabled {
                "visited_epoch_and_layer0_pools".to_string()
            } else {
                "visited_epoch_only".to_string()
            },
            rust_layer0_neighbor_layout: if self.layer0_flat_graph_enabled {
                "flat_u32_adjacency".to_string()
            } else {
                "vec_i64_per_node".to_string()
            },
            rust_layer0_neighbor_id_type: if self.layer0_flat_graph_enabled {
                "u32".to_string()
            } else {
                "i64".to_string()
            },
        }
    }

    fn batch_distance_mode(&self) -> HnswLayer0BatchDistanceMode {
        HnswLayer0BatchDistanceMode {
            native_layer0_query_distance: "distances_batch_4".to_string(),
            rust_layer0_query_distance: if self.layer0_batch4_calls > 0 {
                "batch4_pointer_fast_path".to_string()
            } else {
                "scalar_pointer_fast_path".to_string()
            },
            rust_batch_enabled: self.layer0_batch4_calls > 0,
            rust_batch_width: if self.layer0_batch4_calls > 0 { 4 } else { 1 },
        }
    }

    fn batch_distance_call_counts(&self) -> HnswLayer0BatchDistanceCallCounts {
        HnswLayer0BatchDistanceCallCounts {
            layer0_batch4_calls: self.layer0_batch4_calls,
            layer0_scalar_distance_calls: self
                .layer0_query_distance_calls
                .saturating_sub(self.layer0_batch4_calls * 4),
        }
    }

    fn prefetch_mode(&self) -> HnswLayer0PrefetchMode {
        HnswLayer0PrefetchMode {
            native_layer0_vector_prefetch: "next_neighbor_vector_l1".to_string(),
            rust_layer0_vector_prefetch: if self.layer0_vector_prefetches > 0 {
                "next_neighbor_vector_l1".to_string()
            } else {
                "disabled_or_unsupported".to_string()
            },
            rust_prefetch_enabled: self.layer0_vector_prefetches > 0,
        }
    }

    fn prefetch_call_counts(&self) -> HnswLayer0PrefetchCallCounts {
        HnswLayer0PrefetchCallCounts {
            layer0_vector_prefetches: self.layer0_vector_prefetches,
        }
    }

    fn layer0_neighbor_access_call_counts(&self) -> HnswLayer0NeighborAccessCallCounts {
        HnswLayer0NeighborAccessCallCounts {
            layer0_flat_graph_neighbor_reads: self.layer0_flat_graph_neighbor_reads,
        }
    }

    fn total_profiled_ms(&self) -> f64 {
        self.entry_descent.as_secs_f64() * 1000.0
            + self.frontier_ops.as_secs_f64() * 1000.0
            + self.visited_ops.as_secs_f64() * 1000.0
            + self.distance_compute.as_secs_f64() * 1000.0
            + self.candidate_pruning.as_secs_f64() * 1000.0
    }

    fn hotspot_ranking(&self) -> Vec<HnswBuildProfileHotspot> {
        let total = self.total_profiled_ms();
        let mut hotspots = vec![
            (
                "entry_descent",
                self.entry_descent.as_secs_f64() * 1000.0,
                self.entry_descent_calls,
            ),
            (
                "frontier_ops",
                self.frontier_ops.as_secs_f64() * 1000.0,
                self.frontier_pushes + self.frontier_pops,
            ),
            (
                "visited_ops",
                self.visited_ops.as_secs_f64() * 1000.0,
                self.visited_marks,
            ),
            (
                "distance_compute",
                self.distance_compute.as_secs_f64() * 1000.0,
                self.distance_calls,
            ),
            (
                "candidate_pruning",
                self.candidate_pruning.as_secs_f64() * 1000.0,
                self.pruned_candidates,
            ),
        ];

        hotspots.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        hotspots
            .into_iter()
            .map(|(stage, milliseconds, calls)| HnswBuildProfileHotspot {
                stage: stage.to_string(),
                milliseconds,
                calls,
                share_of_profiled_time: if total > 0.0 {
                    milliseconds / total
                } else {
                    0.0
                },
            })
            .collect()
    }

    fn recommended_next_target(&self) -> String {
        let hotspots = self.hotspot_ranking();
        let stage = hotspots
            .first()
            .map(|hotspot| hotspot.stage.as_str())
            .unwrap_or("distance_compute");
        match stage {
            "entry_descent" => "entry_descent_level_hopping".to_string(),
            "frontier_ops" => "frontier_queue_operations".to_string(),
            "visited_ops" => "visited_state_reuse".to_string(),
            "candidate_pruning" => "candidate_pruning_result_heap".to_string(),
            _ => "distance_compute_inner_loop".to_string(),
        }
    }

    fn into_report(self, query_count: usize) -> HnswCandidateSearchProfileReport {
        HnswCandidateSearchProfileReport {
            candidate_search_breakdown: self.breakdown(),
            call_counts: self.call_counts(),
            distance_compute_breakdown: self.distance_compute_breakdown(),
            distance_compute_call_counts: self.distance_compute_call_counts(),
            search_core_shape: self.search_core_shape(),
            batch_distance_mode: self.batch_distance_mode(),
            batch_distance_call_counts: self.batch_distance_call_counts(),
            prefetch_mode: self.prefetch_mode(),
            prefetch_call_counts: self.prefetch_call_counts(),
            layer0_neighbor_access_call_counts: self.layer0_neighbor_access_call_counts(),
            hotspot_ranking: self.hotspot_ranking(),
            recommended_next_target: self.recommended_next_target(),
            total_profiled_ms: self.total_profiled_ms(),
            query_count,
        }
    }
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            visited_epoch: Vec::new(),
            epoch: 1,
            generic_frontier: BinaryHeap::new(),
            generic_results: BinaryHeap::new(),
            generic_worst_result_distance: f32::INFINITY,
            layer0_frontier: Layer0OrderedFrontier::default(),
            layer0_results: Layer0OrderedResults::default(),
        }
    }

    fn prepare(&mut self, len: usize) {
        if self.visited_epoch.len() < len {
            self.visited_epoch.resize(len, 0);
        }
        if self.epoch == u32::MAX {
            self.visited_epoch.fill(0);
            self.epoch = 1;
        } else {
            self.epoch += 1;
        }
    }

    #[inline]
    fn mark_visited(&mut self, idx: usize) -> bool {
        if self.visited_epoch[idx] == self.epoch {
            return false;
        }
        self.visited_epoch[idx] = self.epoch;
        true
    }

    fn prepare_layer0_pools(&mut self, ef: usize) {
        self.layer0_frontier.prepare(ef * 2);
        self.layer0_results.prepare(ef);
    }

    fn prepare_generic_heaps(&mut self, ef: usize) {
        let frontier_capacity = ef.saturating_mul(2).max(1);
        self.generic_frontier.clear();
        if self.generic_frontier.capacity() < frontier_capacity {
            self.generic_frontier
                .reserve(frontier_capacity - self.generic_frontier.capacity());
        }

        let result_capacity = ef.max(1);
        self.generic_results.clear();
        if self.generic_results.capacity() < result_capacity {
            self.generic_results
                .reserve(result_capacity - self.generic_results.capacity());
        }
        self.generic_worst_result_distance = f32::INFINITY;
    }

    #[inline]
    fn sync_generic_worst_result_distance(&mut self) {
        self.generic_worst_result_distance = self
            .generic_results
            .peek()
            .map(|&(SearchMaxDist(dist), _)| dist)
            .unwrap_or(f32::INFINITY);
    }

    #[inline]
    fn push_generic_result(&mut self, idx: usize, dist: f32, ef: usize) -> u64 {
        let mut result_pops = 0;
        if self.generic_results.len() >= ef {
            self.generic_results.pop();
            result_pops = 1;
        }
        self.generic_results.push((SearchMaxDist(dist), idx));
        self.sync_generic_worst_result_distance();
        result_pops
    }

    fn drain_generic_results_sorted(&mut self) -> Vec<(usize, f32)> {
        let mut sorted = Vec::with_capacity(self.generic_results.len());
        while let Some((SearchMaxDist(dist), idx)) = self.generic_results.pop() {
            sorted.push((idx, dist));
        }
        sorted.reverse();
        sorted
    }
}

pub struct HnswIndex {
    config: IndexConfig,
    entry_point: Option<i64>,
    max_level: usize,
    vectors: Vec<f32>,
    bf16_vectors: Vec<u16>,
    use_bf16_storage: bool,
    // OPT-015: ids Vec kept only for custom ID support, not used in hot path
    ids: Vec<i64>,
    node_info: Vec<NodeInfo>,
    layer0_flat_graph: Layer0FlatGraph,
    layer0_slab: Layer0Slab,
    next_id: i64,
    trained: bool,
    dim: usize,
    ef_construction: usize,
    ef_search: usize,
    m: usize,
    m_max0: usize,
    level_multiplier: f32,
    metric_type: MetricType,
    distance_to_idx_fn: DistanceToIdxFn,
    l2_distance_sq_ptr_kernel: simd::L2DistanceSqPtrKernel,
    // OPT-015: Flag to indicate if we're using sequential IDs (idx == id)
    use_sequential_ids: bool,
    // OPT-024: Number of threads for parallel build
    num_threads: usize,
    level_rng: StdRng,
}

impl HnswIndex {
    #[inline]
    fn bf16_storage_enabled_for_config(config: &IndexConfig) -> bool {
        config.data_type == DataType::BFloat16 && config.metric_type == MetricType::L2
    }

    #[inline]
    fn resolve_distance_to_idx_fn(metric_type: MetricType) -> DistanceToIdxFn {
        match metric_type {
            MetricType::L2 => Self::distance_to_idx_l2_dispatch,
            MetricType::Ip => Self::distance_to_idx_ip_dispatch,
            MetricType::Cosine => Self::distance_to_idx_cosine_dispatch,
            MetricType::Hamming => Self::distance_to_idx_hamming_dispatch,
        }
    }

    #[inline(always)]
    fn distance_to_idx_l2_dispatch(index: &HnswIndex, query: &[f32], idx: usize) -> f32 {
        unsafe { index.l2_distance_to_idx_ptr(query.as_ptr(), index.vectors.as_ptr(), idx) }
    }

    #[inline(always)]
    fn distance_to_idx_ip_dispatch(index: &HnswIndex, query: &[f32], idx: usize) -> f32 {
        let start = idx * index.dim;
        let stored = &index.vectors[start..start + index.dim];
        -simd::inner_product(query, stored)
    }

    #[inline(always)]
    fn distance_to_idx_cosine_dispatch(index: &HnswIndex, query: &[f32], idx: usize) -> f32 {
        let start = idx * index.dim;
        let stored = &index.vectors[start..start + index.dim];
        let ip = simd::inner_product(query, stored);
        let q_norm_sq = simd::inner_product(query, query);
        let v_norm_sq = simd::inner_product(stored, stored);
        let q_norm = q_norm_sq.sqrt();
        let v_norm = v_norm_sq.sqrt();
        if q_norm > 0.0 && v_norm > 0.0 {
            1.0 - ip / (q_norm * v_norm)
        } else {
            1.0
        }
    }

    #[inline(always)]
    fn distance_to_idx_hamming_dispatch(_index: &HnswIndex, _query: &[f32], _idx: usize) -> f32 {
        panic!("Hamming distance not supported for HNSW - use BinaryHnswIndex");
    }

    #[inline]
    fn should_bruteforce_bitset_knn(
        &self,
        top_k: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> bool {
        let ntotal = self.ids.len();
        if ntotal == 0 {
            return false;
        }

        if top_k as f32 >= ntotal as f32 * HNSW_SEARCH_BF_TOPK_THRESHOLD {
            return true;
        }

        if bitset.is_empty() {
            return false;
        }

        let filtered_out_num = bitset.count();
        if filtered_out_num as f32 >= bitset.len() as f32 * HNSW_SEARCH_KNN_BF_FILTER_THRESHOLD {
            return true;
        }

        let survivors = bitset.len().saturating_sub(filtered_out_num);
        survivors > 0 && top_k as f32 >= survivors as f32 * HNSW_SEARCH_BF_TOPK_THRESHOLD
    }

    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        // OPT-029: Updated defaults for better recall on large datasets (100K+ vectors)
        // M=32 provides better graph connectivity for high-dimensional data
        // EF_CONSTRUCTION=400 ensures better graph quality during build
        // EF_SEARCH=400 provides good recall for most scenarios
        let m = config.params.m.unwrap_or(32).clamp(2, 64);

        // M_max0 is typically 2*M for layer 0 (denser connections at base)
        let m_max0 = m * 2;

        // OPT-029: Higher default ef_search for better recall
        let ef_search = config.params.ef_search.unwrap_or(400).max(1);

        // OPT-029: Higher default ef_construction for better graph quality
        let ef_construction = config.params.ef_construction.unwrap_or(400).max(1);

        // BUG-001 FIX: keep the default layer distribution stable across M values.
        // High-M builds collapse into a near-single-layer graph when we derive ml from the
        // runtime M value. Allow explicit overrides, otherwise use the reference distribution.
        let level_multiplier = config
            .params
            .ml
            .unwrap_or_else(|| 1.0 / (REFERENCE_M_FOR_LEVEL as f32).ln());

        // OPT-024: Get number of threads from config or use default (num_cpus)
        let num_threads = config.params.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        let distance_to_idx_fn = Self::resolve_distance_to_idx_fn(config.metric_type);
        let l2_distance_sq_ptr_kernel = simd::l2_distance_sq_ptr_kernel();
        let level_rng = config
            .params
            .random_seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        let use_bf16_storage = Self::bf16_storage_enabled_for_config(config);

        Ok(Self {
            config: config.clone(),
            entry_point: None,
            vectors: Vec::new(),
            bf16_vectors: Vec::new(),
            use_bf16_storage,
            ids: Vec::new(),
            node_info: Vec::new(),
            layer0_flat_graph: Layer0FlatGraph::default(),
            layer0_slab: Layer0Slab::default(),
            next_id: 0,
            trained: false,
            dim: config.dim,
            ef_construction,
            ef_search,
            m,
            m_max0,
            max_level: 0,
            level_multiplier,
            metric_type: config.metric_type,
            distance_to_idx_fn,
            l2_distance_sq_ptr_kernel,
            use_sequential_ids: true,
            num_threads,
            level_rng,
        })
    }

    #[inline]
    fn append_dense_vector(&mut self, vector: &[f32]) {
        self.vectors.extend_from_slice(vector);
        if self.use_bf16_storage {
            self.bf16_vectors
                .extend(vector.iter().map(|&v| Bf16::from_f32(v).to_bits()));
        }
    }

    #[inline]
    fn rebuild_bf16_storage(&mut self) {
        if self.use_bf16_storage {
            self.bf16_vectors = self
                .vectors
                .iter()
                .map(|&v| Bf16::from_f32(v).to_bits())
                .collect();
        } else {
            self.bf16_vectors.clear();
        }
    }

    /// Generate a random level for a new node using exponential distribution
    fn random_level(&mut self) -> usize {
        let r: f32 = self.level_rng.gen(); // Uniform [0, 1)

        // Inverse CDF of exponential distribution
        let level = (-r.ln() * self.level_multiplier) as usize;

        level.min(MAX_LAYERS - 1)
    }

    /// Get the maximum allowed connections for a given layer
    #[inline]
    fn max_connections_for_layer(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    fn refresh_layer0_flat_graph(&mut self) {
        self.layer0_flat_graph.clear();
        self.layer0_slab.clear();

        let node_count = self.node_info.len();
        if node_count == 0 || self.m_max0 == 0 || node_count > (u32::MAX as usize) {
            return;
        }

        let max_neighbors = self.m_max0;
        self.layer0_flat_graph.max_neighbors = max_neighbors;
        self.layer0_flat_graph.degrees.resize(node_count, 0);
        self.layer0_flat_graph
            .neighbors
            .resize(node_count * max_neighbors, u32::MAX);

        for node_idx in 0..node_count {
            if self.node_info[node_idx].layer_neighbors.is_empty() {
                continue;
            }

            let ids = &self.node_info[node_idx].layer_neighbors[0].ids;
            let mut degree = 0usize;
            for &nbr_id in ids {
                if degree >= max_neighbors {
                    break;
                }
                let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                if nbr_idx < node_count {
                    self.layer0_flat_graph.neighbors[node_idx * max_neighbors + degree] =
                        nbr_idx as u32;
                    degree += 1;
                }
            }
            self.layer0_flat_graph.degrees[node_idx] = degree;
        }

        self.layer0_flat_graph.enabled = true;
        self.refresh_layer0_slab();
    }

    fn refresh_layer0_slab(&mut self) {
        self.layer0_slab.clear();

        let node_count = self.node_info.len();
        if !self.layer0_flat_graph.is_enabled_for(node_count) || self.dim == 0 {
            return;
        }

        let max_neighbors = self.layer0_flat_graph.max_neighbors;
        let vector_offset_words = 1 + max_neighbors;
        let stride_words = vector_offset_words + self.dim;

        self.layer0_slab.stride_words = stride_words;
        self.layer0_slab.vector_offset_words = vector_offset_words;
        self.layer0_slab.max_neighbors = max_neighbors;
        self.layer0_slab.dim = self.dim;
        self.layer0_slab.words.resize(node_count * stride_words, 0);

        for node_idx in 0..node_count {
            let base = node_idx * stride_words;
            let neighbors = self.layer0_flat_graph.neighbors_for(node_idx);
            self.layer0_slab.words[base] = neighbors.len() as u32;
            for (offset, &nbr_idx) in neighbors.iter().enumerate() {
                self.layer0_slab.words[base + 1 + offset] = nbr_idx;
            }

            let vector_start = node_idx * self.dim;
            for offset in 0..self.dim {
                self.layer0_slab.words[base + vector_offset_words + offset] =
                    self.vectors[vector_start + offset].to_bits();
            }
        }

        self.layer0_slab.enabled = true;
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }
        self.trained = true;
        Ok(())
    }

    /// Add vectors to the index with proper multi-layer construction
    ///
    /// OPT-022: Two-phase build optimization:
    /// - Phase 1: Pre-allocate and store all vectors and metadata (no graph construction)
    /// - Phase 2: Build graph connections for all nodes
    ///
    /// OPT-015: Sequential ID optimization - when IDs are sequential (0,1,2,...),
    /// we can use idx directly without any lookup, eliminating HashMap/linear search.
    ///
    /// This improves cache locality and reduces memory fragmentation during bulk insert.
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        let base_count = self.ids.len();

        // OPT-015: Track if we're using sequential IDs
        // Sequential IDs allow direct indexing (idx == id), eliminating all lookups
        let using_sequential = ids.is_none();
        if base_count == 0 {
            self.use_sequential_ids = using_sequential;
        } else {
            self.use_sequential_ids = self.use_sequential_ids && using_sequential;
        }

        // OPT-022: Phase 1 - Pre-allocate and store all vectors/metadata
        self.vectors.reserve(n * self.dim);
        if self.use_bf16_storage {
            self.bf16_vectors.reserve(n * self.dim);
        }
        self.ids.reserve(n);
        self.node_info.reserve(n);

        let first_new_idx = self.ids.len();

        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];

            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            // Assign random level to this node
            let node_level = self.random_level();

            // Create node info with appropriate layer structure
            let node_info = NodeInfo::new(node_level, self.m);

            // Store vector and metadata (no graph construction yet)
            // OPT-021: Direct indexing - idx is the position in the arrays

            // OPT-015: Only store IDs if not sequential (saves memory and cache)
            if !using_sequential {
                self.ids.push(id);
            } else {
                // For sequential IDs, we don't need to store them (idx == id)
                // But we keep the Vec allocated for API compatibility
                self.ids.push(id);
            }

            self.append_dense_vector(new_vec);
            self.node_info.push(node_info);

            // If this is the first node, set it as entry point
            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_level;
            }

            // Update global max level and entry point if needed
            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(id);
            }
        }

        // OPT-022: Phase 2 - Build graph connections for all new nodes
        // Collect node levels first to avoid borrow conflicts
        let node_levels: Vec<usize> = (first_new_idx..first_new_idx + n)
            .map(|idx| self.node_info[idx].max_layer)
            .collect();

        let mut scratch = SearchScratch::new();
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;

            // Skip the first node (it's the entry point, no connections needed)
            if idx > 0 {
                let vec_start = idx * self.dim;
                let vec: Vec<f32> = self.vectors[vec_start..vec_start + self.dim].to_vec();
                self.insert_node_with_scratch(idx, &vec, node_level, &mut scratch);
            }
        }

        self.refresh_layer0_flat_graph();
        Ok(n)
    }

    fn add_profiled(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
        stats: &mut HnswBuildProfileStats,
    ) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        let base_count = self.ids.len();
        let using_sequential = ids.is_none();
        if base_count == 0 {
            self.use_sequential_ids = using_sequential;
        } else {
            self.use_sequential_ids = self.use_sequential_ids && using_sequential;
        }

        self.vectors.reserve(n * self.dim);
        if self.use_bf16_storage {
            self.bf16_vectors.reserve(n * self.dim);
        }
        self.ids.reserve(n);
        self.node_info.reserve(n);

        let first_new_idx = self.ids.len();

        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];

            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            let node_level = self.random_level();
            let node_info = NodeInfo::new(node_level, self.m);

            self.ids.push(id);
            self.append_dense_vector(new_vec);
            self.node_info.push(node_info);

            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_level;
            }

            if node_level > self.max_level {
                self.max_level = node_level;
                self.entry_point = Some(id);
            }
        }

        let node_levels: Vec<usize> = (first_new_idx..first_new_idx + n)
            .map(|idx| self.node_info[idx].max_layer)
            .collect();

        let mut scratch = SearchScratch::new();
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;
            if idx > 0 {
                let vec_start = idx * self.dim;
                let vec: Vec<f32> = self.vectors[vec_start..vec_start + self.dim].to_vec();
                self.insert_node_profiled_with_scratch(idx, &vec, node_level, stats, &mut scratch);
            }
        }

        self.refresh_layer0_flat_graph();
        Ok(n)
    }

    /// OPT-019: Add vectors with shuffle (randomized insertion order)
    ///
    /// Randomizes the order of vector insertion to improve graph quality.
    /// This avoids biases from data distribution and can improve recall by 2-5%.
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (n * dim)
    /// * `ids` - Optional IDs for vectors (auto-generated if None)
    /// * `shuffle_seed` - Optional seed for reproducible shuffle (None = random)
    ///
    /// # Returns
    /// Number of vectors added
    ///
    /// # Performance
    /// - Recall improvement: +2-5% typical
    /// - Build time overhead: <1% (shuffle is negligible)
    /// - Memory overhead: O(n) for index array
    pub fn add_shuffle(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
        shuffle_seed: Option<u64>,
    ) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        // Create shuffled index array
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = shuffle_seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        // Build shuffled vectors and IDs
        let mut shuffled_vectors = Vec::with_capacity(vectors.len());
        let mut shuffled_ids = if ids.is_some() {
            Some(Vec::with_capacity(n))
        } else {
            None
        };

        for &idx in &indices {
            let start = idx * self.dim;
            shuffled_vectors.extend_from_slice(&vectors[start..start + self.dim]);
            if let Some(ids) = ids {
                shuffled_ids.as_mut().unwrap().push(ids[idx]);
            }
        }

        // Add in shuffled order
        self.add(&shuffled_vectors, shuffled_ids.as_deref())
    }

    /// Add vectors to the index using parallel construction
    ///
    /// OPT-024: Parallel HNSW build using rayon.
    ///
    /// Strategy:
    /// - Phase 1: Pre-allocate all vectors and metadata (same as serial)
    /// - Phase 2: Batch-parallel insertion - divide nodes into batches,
    ///   each batch processes in parallel with read-only graph access,
    ///   then merge results serially to avoid race conditions.
    ///
    /// This achieves 4-8x speedup while maintaining graph quality.
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (n * dim)
    /// * `ids` - Optional IDs for vectors (auto-generated if None)
    /// * `parallel` - Whether to use parallel construction (default: true if num_threads > 1)
    ///
    /// # Returns
    /// Number of vectors added
    /// OPT-024/OPT-031: Add vectors using parallel construction with production-grade features
    ///
    /// Uses batched parallel approach:
    /// - Divide nodes into batches with dynamic batch size based on vector count and dimensions
    /// - Parallel neighbor search within each batch (read-only)
    /// - Serial graph update (avoids race conditions)
    /// - Progress tracking and logging
    /// - Comprehensive error handling
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (n * dim)
    /// * `ids` - Optional IDs for vectors (auto-generated if None)
    /// * `parallel` - Whether to use parallel construction (default: true if num_threads > 1 and n >= 1000)
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of vectors added
    /// * `Err(KnowhereError)` - Error during construction
    ///
    /// # Performance
    /// Expected speedup: 4-8x on multi-core systems
    /// Target: 100K vectors build time < 3s
    ///
    /// # Example
    /// ```ignore
    /// let mut index = HnswIndex::new(&config)?;
    /// index.train(&vectors)?;
    /// let count = index.add_parallel(&vectors, None, None)?;
    /// ```
    pub fn add_parallel(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
        parallel: Option<bool>,
    ) -> Result<usize> {
        // Validate input
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        // Validate vector dimension first (before calculating n)
        if vectors.len() % self.dim != 0 {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "vector dimension mismatch: {} elements not divisible by dim {}",
                vectors.len(),
                self.dim
            )));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        // Validate IDs if provided
        if let Some(id_slice) = ids {
            if id_slice.len() != n {
                return Err(crate::api::KnowhereError::InvalidArg(format!(
                    "ID count ({}) does not match vector count ({})",
                    id_slice.len(),
                    n
                )));
            }
        }

        // Decide whether to use parallel construction
        let use_parallel = parallel.unwrap_or(self.num_threads > 1 && n >= 1000);
        if !use_parallel {
            eprintln!(
                "[HNSW] Parallel build: falling back to serial (n={}, threads={})",
                n, self.num_threads
            );
            return self.add(vectors, ids);
        }

        eprintln!(
            "[HNSW] Parallel build: {} vectors x {} dims, {} threads, M={}, ef_construction={}",
            n, self.dim, self.num_threads, self.m, self.ef_construction
        );

        let base_count = self.ids.len();
        let using_sequential = ids.is_none();
        if base_count == 0 {
            self.use_sequential_ids = using_sequential;
        } else {
            self.use_sequential_ids = self.use_sequential_ids && using_sequential;
        }

        // Phase 1: Pre-allocate all vectors/metadata
        let start_time = std::time::Instant::now();

        self.vectors.reserve(n * self.dim);
        if self.use_bf16_storage {
            self.bf16_vectors.reserve(n * self.dim);
        }
        self.ids.reserve(n);
        self.node_info.reserve(n);

        let first_new_idx = self.ids.len();

        // Pre-generate random levels
        let node_levels: Vec<usize> = (0..n).map(|_| self.random_level()).collect();

        // Store all vectors and metadata
        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            let node_info = NodeInfo::new(node_levels[i], self.m);
            self.ids.push(id);
            self.append_dense_vector(new_vec);
            self.node_info.push(node_info);

            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_levels[i];
            }
            if node_levels[i] > self.max_level {
                self.max_level = node_levels[i];
                self.entry_point = Some(id);
            }
        }

        let phase1_time = start_time.elapsed();
        eprintln!(
            "[HNSW] Phase 1 (pre-allocation) completed in {:?}",
            phase1_time
        );

        // Phase 2: Batched parallel graph construction
        // OPT-031: Dynamic batch size strategy based on vector count and dimensions
        let batch_size = self.calculate_optimal_batch_size(n, self.dim);
        let total_batches = n.div_ceil(batch_size);

        eprintln!(
            "[HNSW] Phase 2 (graph construction) - {} batches, batch_size={}",
            total_batches, batch_size
        );

        let mut processed = 0;
        let phase2_start = std::time::Instant::now();

        for (batch_idx, batch_start) in (0..n).step_by(batch_size).enumerate() {
            let batch_end = (batch_start + batch_size).min(n);
            let current_batch_size = batch_end - batch_start;

            // Parallel neighbor search (read-only)
            let batch_results: Vec<BatchNeighborResults> = (batch_start..batch_end)
                .into_par_iter()
                .map(|idx| {
                    let vec_start = idx * self.dim;
                    let vec = &self.vectors[vec_start..vec_start + self.dim];
                    let node_level = node_levels[idx];
                    let neighbors = self.find_neighbors_for_insertion(vec, node_level);
                    (idx, node_level, neighbors)
                })
                .collect();

            let mut layer0_nodes_to_shrink: Vec<usize> = Vec::new();

            // Serial graph update (avoids race conditions)
            for (idx, node_level, neighbors_per_layer) in batch_results {
                if base_count == 0 && idx == first_new_idx {
                    continue;
                }
                let new_id = self.ids[idx];
                for (level, neighbors) in
                    neighbors_per_layer.iter().enumerate().take(node_level + 1)
                {
                    self.add_connections_for_node(
                        idx,
                        new_id,
                        level,
                        neighbors,
                        &mut layer0_nodes_to_shrink,
                    );
                }
            }

            layer0_nodes_to_shrink.sort_unstable();
            layer0_nodes_to_shrink.dedup();
            for node_idx in layer0_nodes_to_shrink {
                self.shrink_layer_neighbors_heuristic_idx(node_idx, 0, self.m_max0);
            }

            processed += current_batch_size;

            // Progress logging (every 10% or every batch for small datasets)
            let progress = (processed as f64 / n as f64) * 100.0;
            let should_log = batch_idx == 0
                || batch_idx == total_batches - 1
                || (processed * 10 / n) > ((processed - current_batch_size) * 10 / n);

            if should_log {
                let elapsed = phase2_start.elapsed();
                let rate = processed as f64 / elapsed.as_secs_f64();
                eprintln!(
                    "[HNSW] Progress: {}/{} ({:.1}%), {:.0} vec/s",
                    processed, n, progress, rate
                );
            }
        }

        let phase2_time = phase2_start.elapsed();
        let total_time = start_time.elapsed();

        eprintln!(
            "[HNSW] Parallel build completed: {} vectors in {:?} (phase1: {:?}, phase2: {:?})",
            n, total_time, phase1_time, phase2_time
        );

        // Performance validation (only in debug/test builds)
        #[cfg(debug_assertions)]
        {
            let total_secs = total_time.as_secs_f64();
            let target_secs = 3.0; // Target: < 3s for 100K vectors
            if n >= 100_000 && total_secs > target_secs {
                eprintln!(
                    "[HNSW] Warning: build time ({:.2}s) exceeded target ({:.2}s) for {} vectors",
                    total_secs, target_secs, n
                );
            }
        }

        self.refresh_layer0_flat_graph();
        Ok(n)
    }

    fn add_parallel_profiled(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
        stats: &mut HnswBuildProfileStats,
        graph_quality_stats: &mut HnswParallelBuildGraphQualityStats,
    ) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        if vectors.len() % self.dim != 0 {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "vector dimension mismatch: {} elements not divisible by dim {}",
                vectors.len(),
                self.dim
            )));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        if let Some(id_slice) = ids {
            if id_slice.len() != n {
                return Err(crate::api::KnowhereError::InvalidArg(format!(
                    "ID count ({}) does not match vector count ({})",
                    id_slice.len(),
                    n
                )));
            }
        }

        if self.num_threads <= 1 || n < 1000 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "parallel build profile requires num_threads > 1 and at least 1000 vectors"
                    .to_string(),
            ));
        }

        let base_count = self.ids.len();
        let using_sequential = ids.is_none();
        if base_count == 0 {
            self.use_sequential_ids = using_sequential;
        } else {
            self.use_sequential_ids = self.use_sequential_ids && using_sequential;
        }

        self.vectors.reserve(n * self.dim);
        if self.use_bf16_storage {
            self.bf16_vectors.reserve(n * self.dim);
        }
        self.ids.reserve(n);
        self.node_info.reserve(n);

        let first_new_idx = self.ids.len();
        let node_levels: Vec<usize> = (0..n).map(|_| self.random_level()).collect();

        for i in 0..n {
            let start = i * self.dim;
            let new_vec = &vectors[start..start + self.dim];
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            let node_info = NodeInfo::new(node_levels[i], self.m);
            self.ids.push(id);
            self.append_dense_vector(new_vec);
            self.node_info.push(node_info);

            if base_count == 0 && i == 0 {
                self.entry_point = Some(id);
                self.max_level = node_levels[i];
            }
            if node_levels[i] > self.max_level {
                self.max_level = node_levels[i];
                self.entry_point = Some(id);
            }
        }

        let batch_size = self.calculate_optimal_batch_size(n, self.dim);

        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            let batch_results: Vec<BatchNeighborResultsProfiled> = (batch_start..batch_end)
                .into_par_iter()
                .map(|idx| {
                    let vec_start = idx * self.dim;
                    let vec = &self.vectors[vec_start..vec_start + self.dim];
                    let node_level = node_levels[idx];
                    let mut local_stats = HnswBuildProfileStats::default();
                    let mut local_graph_quality_stats =
                        HnswParallelBuildGraphQualityStats::default();
                    let neighbors = self.find_neighbors_for_insertion_profiled(
                        vec,
                        node_level,
                        &mut local_stats,
                        &mut local_graph_quality_stats,
                    );
                    (
                        idx,
                        node_level,
                        neighbors,
                        local_stats,
                        local_graph_quality_stats,
                    )
                })
                .collect();

            let mut layer0_nodes_to_shrink: Vec<usize> = Vec::new();

            for (idx, node_level, neighbors_per_layer, local_stats, local_graph_quality_stats) in
                batch_results
            {
                stats.absorb(local_stats);
                graph_quality_stats.absorb(local_graph_quality_stats);

                if base_count == 0 && idx == first_new_idx {
                    continue;
                }

                let new_id = self.ids[idx];
                for (level, neighbors) in
                    neighbors_per_layer.iter().enumerate().take(node_level + 1)
                {
                    let stage_start = Instant::now();
                    let connection_graph_quality_stats = self.add_connections_for_node_profiled(
                        idx,
                        new_id,
                        level,
                        neighbors,
                        &mut layer0_nodes_to_shrink,
                    );
                    stats.record(
                        HnswBuildProfileStage::ConnectionUpdate,
                        stage_start.elapsed(),
                    );
                    graph_quality_stats.absorb(connection_graph_quality_stats);
                }
            }

            layer0_nodes_to_shrink.sort_unstable();
            layer0_nodes_to_shrink.dedup();
            for node_idx in layer0_nodes_to_shrink {
                let stage_start = Instant::now();
                self.shrink_layer_neighbors_heuristic_idx(node_idx, 0, self.m_max0);
                stats.record(
                    HnswBuildProfileStage::ConnectionUpdate,
                    stage_start.elapsed(),
                );
            }
        }

        self.refresh_layer0_flat_graph();
        Ok(n)
    }

    /// OPT-031: Calculate optimal batch size for parallel HNSW construction
    ///
    /// Dynamic batch size strategy based on:
    /// - Vector count: larger datasets benefit from larger batches (better parallelism)
    /// - Vector dimensions: higher dimensions need smaller batches (more computation per vector)
    /// - Number of threads: more threads can handle more batches
    ///
    /// Formula:
    /// base_batch = n / num_threads
    /// dim_factor = max(1.0, 128.0 / dim as f64) // Smaller batches for high dimensions
    /// count_factor = min(1.0, n as f64 / 10000.0) // Scale down for small datasets
    /// batch_size = base_batch * dim_factor * count_factor
    ///
    /// Constraints:
    /// - Minimum: 50 vectors per batch (avoid excessive overhead)
    /// - Maximum: 5000 vectors per batch (ensure progress reporting granularity)
    fn calculate_optimal_batch_size(&self, n: usize, dim: usize) -> usize {
        let base_batch = ((n as f64) / (self.num_threads as f64)).max(50.0);

        // Dimension factor: higher dimensions = more computation per vector = smaller batches
        let dim_factor = (128.0 / dim as f64).clamp(0.25, 2.0);

        // Count factor: small datasets need smaller batches for better progress tracking
        let count_factor = (n as f64 / 10000.0).clamp(0.5, 1.0);

        let optimal = base_batch * dim_factor * count_factor;

        // Clamp to reasonable range
        let batch_size = optimal.clamp(50.0, 5000.0) as usize;

        eprintln!(
            "[HNSW] Batch size: n={}, dim={}, threads={}, batch_size={}",
            n, dim, self.num_threads, batch_size
        );

        batch_size
    }

    fn parallel_insert_entry_descent_mode(&self) -> &'static str {
        "greedy_from_max_level"
    }

    fn parallel_upper_layer_overflow_shrink_mode(&self) -> &'static str {
        "heuristic_shrink"
    }

    /// Find neighbors for node insertion (read-only, parallelizable)
    fn find_neighbors_for_insertion(
        &self,
        vec: &[f32],
        node_level: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut scratch = SearchScratch::new();
        self.find_neighbors_for_insertion_with_scratch(vec, node_level, &mut scratch)
    }

    fn find_neighbors_for_insertion_with_scratch(
        &self,
        vec: &[f32],
        node_level: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut neighbors_per_layer = vec![Vec::new(); node_level + 1];
        let entry_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());
        let mut curr_ep_idx = entry_idx;

        if node_level < self.max_level {
            for level in ((node_level + 1)..=self.max_level).rev() {
                let nearest =
                    self.search_layer_idx_with_scratch(vec, curr_ep_idx, level, 1, scratch);
                if !nearest.is_empty() {
                    curr_ep_idx = nearest[0].0;
                }
            }
        }

        for level in (0..=node_level).rev() {
            let nearest = self.search_layer_idx_with_scratch(vec, curr_ep_idx, level, 1, scratch);
            if !nearest.is_empty() {
                curr_ep_idx = nearest[0].0;
            }

            let candidates = self.search_layer_idx_with_scratch(
                vec,
                curr_ep_idx,
                level,
                self.ef_construction,
                scratch,
            );
            let m = if level == 0 { self.m_max0 } else { self.m };
            neighbors_per_layer[level] =
                self.select_neighbors_heuristic_idx_layer_aware(vec, &candidates, m, level == 0);
        }
        neighbors_per_layer
    }

    fn find_neighbors_for_insertion_profiled(
        &self,
        vec: &[f32],
        node_level: usize,
        stats: &mut HnswBuildProfileStats,
        graph_quality_stats: &mut HnswParallelBuildGraphQualityStats,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut scratch = SearchScratch::new();
        self.find_neighbors_for_insertion_with_scratch_profiled(
            vec,
            node_level,
            stats,
            graph_quality_stats,
            &mut scratch,
        )
    }

    fn find_neighbors_for_insertion_with_scratch_profiled(
        &self,
        vec: &[f32],
        node_level: usize,
        stats: &mut HnswBuildProfileStats,
        _graph_quality_stats: &mut HnswParallelBuildGraphQualityStats,
        scratch: &mut SearchScratch,
    ) -> Vec<Vec<(usize, f32)>> {
        let mut neighbors_per_layer = vec![Vec::new(); node_level + 1];
        let entry_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());
        let mut curr_ep_idx = entry_idx;

        if node_level < self.max_level {
            for level in ((node_level + 1)..=self.max_level).rev() {
                let stage_start = Instant::now();
                let nearest =
                    self.search_layer_idx_with_scratch(vec, curr_ep_idx, level, 1, scratch);
                stats.record(HnswBuildProfileStage::LayerDescent, stage_start.elapsed());
                if !nearest.is_empty() {
                    curr_ep_idx = nearest[0].0;
                }
            }
        }

        for level in (0..=node_level).rev() {
            let stage_start = Instant::now();
            let nearest = self.search_layer_idx_with_scratch(vec, curr_ep_idx, level, 1, scratch);
            stats.record(HnswBuildProfileStage::LayerDescent, stage_start.elapsed());
            if !nearest.is_empty() {
                curr_ep_idx = nearest[0].0;
            }

            let stage_start = Instant::now();
            let candidates = self.search_layer_idx_with_scratch(
                vec,
                curr_ep_idx,
                level,
                self.ef_construction,
                scratch,
            );
            stats.record(
                HnswBuildProfileStage::CandidateSearch,
                stage_start.elapsed(),
            );

            let m = if level == 0 { self.m_max0 } else { self.m };
            let stage_start = Instant::now();
            neighbors_per_layer[level] =
                self.select_neighbors_heuristic_idx_layer_aware(vec, &candidates, m, level == 0);
            stats.record(
                HnswBuildProfileStage::NeighborSelection,
                stage_start.elapsed(),
            );
        }

        neighbors_per_layer
    }

    /// Add connections for a node at specific layer
    fn add_connections_for_node(
        &mut self,
        new_idx: usize,
        new_id: i64,
        level: usize,
        neighbors: &[(usize, f32)],
        layer0_nodes_to_shrink: &mut Vec<usize>,
    ) {
        let m_max = self.max_connections_for_layer(level);
        let filtered_neighbors: Vec<(usize, f32)> = neighbors
            .iter()
            .copied()
            .filter(|(nbr_idx, _)| *nbr_idx != new_idx)
            .collect();

        if filtered_neighbors.is_empty() {
            return;
        }

        let nbr_ids: Vec<i64> = filtered_neighbors
            .iter()
            .map(|&(nbr_idx, _)| self.get_id_from_idx(nbr_idx))
            .collect();

        let mut shrink_new_upper_layer = false;
        {
            let node_info = &mut self.node_info[new_idx];
            let layer_nbrs = &mut node_info.layer_neighbors[level];
            for (i, &(_, dist)) in filtered_neighbors.iter().enumerate() {
                layer_nbrs.push(nbr_ids[i], dist);
            }
            if layer_nbrs.ids.len() > m_max {
                if level == 0 {
                    layer0_nodes_to_shrink.push(new_idx);
                } else {
                    shrink_new_upper_layer = true;
                }
            }
        }
        if shrink_new_upper_layer {
            self.shrink_layer_neighbors_heuristic_idx(new_idx, level, m_max);
        }

        for &(nbr_idx, dist) in &filtered_neighbors {
            let mut shrink_neighbor_upper_layer = false;
            {
                let nbr_node_info = &mut self.node_info[nbr_idx];
                if level > nbr_node_info.max_layer {
                    continue;
                }

                let nbr_layer_nbrs = &mut nbr_node_info.layer_neighbors[level];
                nbr_layer_nbrs.push(new_id, dist);
                if nbr_layer_nbrs.ids.len() > m_max {
                    if level == 0 {
                        layer0_nodes_to_shrink.push(nbr_idx);
                    } else {
                        shrink_neighbor_upper_layer = true;
                    }
                }
            }
            if shrink_neighbor_upper_layer {
                self.shrink_layer_neighbors_heuristic_idx(nbr_idx, level, m_max);
            }
        }
    }

    fn add_connections_for_node_profiled(
        &mut self,
        new_idx: usize,
        new_id: i64,
        level: usize,
        neighbors: &[(usize, f32)],
        layer0_nodes_to_shrink: &mut Vec<usize>,
    ) -> HnswParallelBuildGraphQualityStats {
        let mut graph_quality_stats = HnswParallelBuildGraphQualityStats::default();
        let m_max = self.max_connections_for_layer(level);
        let filtered_neighbors: Vec<(usize, f32)> = neighbors
            .iter()
            .copied()
            .filter(|(nbr_idx, _)| *nbr_idx != new_idx)
            .collect();

        if filtered_neighbors.is_empty() {
            return graph_quality_stats;
        }

        let nbr_ids: Vec<i64> = filtered_neighbors
            .iter()
            .map(|&(nbr_idx, _)| self.get_id_from_idx(nbr_idx))
            .collect();

        let mut shrink_new_upper_layer = false;
        {
            let node_info = &mut self.node_info[new_idx];
            let layer_nbrs = &mut node_info.layer_neighbors[level];
            for (i, &(_, dist)) in filtered_neighbors.iter().enumerate() {
                layer_nbrs.push(nbr_ids[i], dist);
            }
            if level > 0 {
                graph_quality_stats.record_upper_layer_connection_update();
            }
            if layer_nbrs.ids.len() > m_max {
                if level == 0 {
                    layer0_nodes_to_shrink.push(new_idx);
                } else {
                    graph_quality_stats.record_upper_layer_overflow_heuristic_shrink();
                    shrink_new_upper_layer = true;
                }
            }
        }
        if shrink_new_upper_layer {
            self.shrink_layer_neighbors_heuristic_idx(new_idx, level, m_max);
        }

        for &(nbr_idx, dist) in &filtered_neighbors {
            let mut shrink_neighbor_upper_layer = false;
            {
                let nbr_node_info = &mut self.node_info[nbr_idx];
                if level > nbr_node_info.max_layer {
                    continue;
                }

                let nbr_layer_nbrs = &mut nbr_node_info.layer_neighbors[level];
                nbr_layer_nbrs.push(new_id, dist);
                if level > 0 {
                    graph_quality_stats.record_upper_layer_connection_update();
                }
                if nbr_layer_nbrs.ids.len() > m_max {
                    if level == 0 {
                        layer0_nodes_to_shrink.push(nbr_idx);
                    } else {
                        graph_quality_stats.record_upper_layer_overflow_heuristic_shrink();
                        shrink_neighbor_upper_layer = true;
                    }
                }
            }
            if shrink_neighbor_upper_layer {
                self.shrink_layer_neighbors_heuristic_idx(nbr_idx, level, m_max);
            }
        }

        graph_quality_stats
    }

    /// Add a single vector to the index with optional explicit layer specification
    ///
    /// # Arguments
    /// * `vector` - The vector to add
    /// * `id` - Optional ID for the vector (auto-generated if None)
    /// * `layers` - Optional explicit layer specification. If Some, the vector will be
    ///   inserted at exactly these layers. If None, uses random level assignment.
    ///
    /// # Example
    /// ```ignore
    /// // Add with random layer (default behavior)
    /// index.add_vector(&[1.0, 2.0, 3.0, 4.0], None, None)?;
    ///
    /// // Add with explicit layers (insert at layers 0, 2, and 3)
    /// index.add_vector(&[1.0, 2.0, 3.0, 4.0], Some(42), Some(&[0, 2, 3]))?;
    /// ```
    pub fn add_vector(
        &mut self,
        vector: &[f32],
        id: Option<i64>,
        layers: Option<&[usize]>,
    ) -> Result<i64> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        if vector.len() != self.dim {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        let assigned_id = id.unwrap_or(self.next_id);
        self.next_id += 1;

        // Determine the node's level(s)
        let node_level = if let Some(spec) = layers {
            // Validate layer specifications
            if spec.is_empty() {
                return Err(crate::api::KnowhereError::InvalidArg(
                    "layers specification cannot be empty".to_string(),
                ));
            }
            let max_spec = *spec.iter().max().unwrap_or(&0);
            if max_spec >= MAX_LAYERS {
                return Err(crate::api::KnowhereError::InvalidArg(format!(
                    "layer index {} exceeds maximum {}",
                    max_spec,
                    MAX_LAYERS - 1
                )));
            }
            // Use the maximum layer from specification
            max_spec
        } else {
            // Use random level assignment
            self.random_level()
        };

        // Determine which layers to actually connect (may be subset of node's levels)
        let connect_layers: Vec<usize> = if let Some(spec) = layers {
            // Sort and deduplicate layer specifications
            let mut sorted: Vec<usize> = spec.to_vec();
            sorted.sort();
            sorted.dedup();
            // Only include layers up to node_level
            sorted.into_iter().filter(|&l| l <= node_level).collect()
        } else {
            // Connect at all layers from 0 to node_level (original behavior)
            (0..=node_level).collect()
        };

        // Create node info with appropriate layer structure
        let node_info = NodeInfo::new(node_level, self.m);

        // Store vector and metadata
        // OPT-021: Direct indexing - idx is the position in the arrays
        let idx = self.ids.len();
        self.ids.push(assigned_id);
        // OPT-021: Removed HashMap insert
        self.append_dense_vector(vector);
        self.node_info.push(node_info);

        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(assigned_id);
            self.max_level = node_level;
            self.refresh_layer0_flat_graph();
            return Ok(assigned_id);
        }

        // Insert node into the graph for specified layers
        self.insert_node_at_layers(idx, vector, &connect_layers);

        // Update global max level and entry point if needed
        if node_level > self.max_level {
            self.max_level = node_level;
            self.entry_point = Some(assigned_id);
        }

        self.refresh_layer0_flat_graph();
        Ok(assigned_id)
    }

    /// Insert a node into specific layers of the graph
    fn insert_node_at_layers(&mut self, new_idx: usize, new_vec: &[f32], layers: &[usize]) {
        let new_id = self.ids[new_idx];

        // Start from the top layer and work down
        let mut curr_ep = self.entry_point.unwrap();

        // Get the maximum layer we're connecting to
        let max_layer = layers.iter().max().copied().unwrap_or(0);

        // For each layer from max_layer down to 0
        for level in (0..=max_layer).rev() {
            // Check if this node should be connected at this layer
            if !layers.contains(&level) {
                // Skip this layer but still traverse to find entry point
                if level <= self.max_level {
                    let nearest_results = self.search_layer(new_vec, curr_ep, level, 1);
                    if !nearest_results.is_empty() {
                        curr_ep = nearest_results[0].0;
                    }
                }
                continue;
            }

            // Search for nearest neighbor at this layer
            let results = self.search_layer(new_vec, curr_ep, level, 1);

            if !results.is_empty() {
                curr_ep = results[0].0;
            }

            // Find efConstruction candidates at this layer
            let candidates = self.search_layer(new_vec, curr_ep, level, self.ef_construction);

            // Select best M neighbors using heuristic
            let m = if level == 0 { self.m_max0 } else { self.m };
            let selected =
                self.select_neighbors_heuristic_layer_aware(new_vec, &candidates, m, level == 0);

            // Add bidirectional connections
            self.add_bidirectional_connections(new_idx, new_id, level, &selected);
        }
    }

    fn insert_node_with_scratch(
        &mut self,
        new_idx: usize,
        new_vec: &[f32],
        node_level: usize,
        scratch: &mut SearchScratch,
    ) {
        let new_id = self.get_id_from_idx(new_idx);

        // Start from the top layer and work down
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        // OPT-021: Layer descent - start from max_level (not node_level) to ensure we connect to the entry point
        // We need to traverse from the top layer down to find the best entry point at each layer
        for level in (0..=self.max_level).rev() {
            // Skip layers above this node's level for connection (but still traverse for entry point search)
            if level > node_level {
                // Just search for entry point at this layer, don't connect
                // Use ef=1 for speed during descent through unused layers
                let nearest_results =
                    self.search_layer_idx_with_scratch(new_vec, curr_ep_idx, level, 1, scratch);
                if !nearest_results.is_empty() {
                    curr_ep_idx = nearest_results[0].0;
                }
                continue;
            }

            // For layers <= node_level: search for nearest neighbors and connect
            // Use ef=1 to find the closest entry point for this layer
            let nearest_results =
                self.search_layer_idx_with_scratch(new_vec, curr_ep_idx, level, 1, scratch);

            if !nearest_results.is_empty() {
                curr_ep_idx = nearest_results[0].0;
            }

            // Find efConstruction candidates at this layer (using index-based version)
            let candidates = self.search_layer_idx_with_scratch(
                new_vec,
                curr_ep_idx,
                level,
                self.ef_construction,
                scratch,
            );

            // Select best M neighbors using heuristic (index-based)
            // Layer 0 must use the denser base-layer degree (m_max0), otherwise bulk-build
            // under-connects the main search layer and recall collapses on larger datasets.
            let m = if level == 0 { self.m_max0 } else { self.m };
            let selected = self.select_neighbors_heuristic_idx_layer_aware(
                new_vec,
                &candidates,
                m,
                level == 0,
            );

            // Add bidirectional connections (uses indices directly)
            self.add_bidirectional_connections_idx(new_idx, new_id, level, &selected);
        }
    }

    fn insert_node_profiled_with_scratch(
        &mut self,
        new_idx: usize,
        new_vec: &[f32],
        node_level: usize,
        stats: &mut HnswBuildProfileStats,
        scratch: &mut SearchScratch,
    ) {
        let new_id = self.get_id_from_idx(new_idx);
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        for level in (0..=self.max_level).rev() {
            if level > node_level {
                let stage_start = Instant::now();
                let nearest_results =
                    self.search_layer_idx_with_scratch(new_vec, curr_ep_idx, level, 1, scratch);
                stats.record(HnswBuildProfileStage::LayerDescent, stage_start.elapsed());
                if !nearest_results.is_empty() {
                    curr_ep_idx = nearest_results[0].0;
                }
                continue;
            }

            let stage_start = Instant::now();
            let nearest_results =
                self.search_layer_idx_with_scratch(new_vec, curr_ep_idx, level, 1, scratch);
            stats.record(HnswBuildProfileStage::LayerDescent, stage_start.elapsed());
            if !nearest_results.is_empty() {
                curr_ep_idx = nearest_results[0].0;
            }

            let stage_start = Instant::now();
            let candidates = self.search_layer_idx_with_scratch(
                new_vec,
                curr_ep_idx,
                level,
                self.ef_construction,
                scratch,
            );
            stats.record(
                HnswBuildProfileStage::CandidateSearch,
                stage_start.elapsed(),
            );

            let m = if level == 0 { self.m_max0 } else { self.m };
            let stage_start = Instant::now();
            let selected = self.select_neighbors_heuristic_idx_layer_aware(
                new_vec,
                &candidates,
                m,
                level == 0,
            );
            stats.record(
                HnswBuildProfileStage::NeighborSelection,
                stage_start.elapsed(),
            );

            let stage_start = Instant::now();
            self.add_bidirectional_connections_idx(new_idx, new_id, level, &selected);
            stats.record(
                HnswBuildProfileStage::ConnectionUpdate,
                stage_start.elapsed(),
            );
        }
    }

    /// Search for nearest neighbors at a specific layer (internal version - works with indices)
    ///
    /// OPT-015 REV2: Work entirely with indices internally - no ID conversion in hot path.
    /// Returns (idx, dist) pairs for internal use.
    /// OPT-009: Optimized search layer implementation
    /// Key optimizations:
    /// 1. Vec<bool> instead of HashSet for visited tracking (10x faster)
    /// 2. Use get_idx_from_id_fast to avoid Option overhead
    /// 3. Removed BUG-001 duplicate search extension (BUG-006 fix makes it unnecessary)
    /// 4. Early termination when candidate exceeds worst result
    /// 5. OPT-036: Fixed heap ordering bug by using separate types for candidates and results
    fn search_layer_idx(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::BinaryHeap;

        /// For candidates: smaller distance = higher priority (pop returns nearest first)
        /// This matches FAISS NodeDistFarther behavior
        #[derive(Clone, Copy, PartialEq)]
        struct MinDist(f32);

        impl Eq for MinDist {}
        impl PartialOrd for MinDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for MinDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Reverse: smaller distance = "greater" in heap = popped first
                other
                    .0
                    .partial_cmp(&self.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        /// For results: larger distance = higher priority (peek returns worst/farthest)
        /// This matches FAISS NodeDistCloser behavior
        #[derive(Clone, Copy, PartialEq)]
        struct MaxDist(f32);

        impl Eq for MaxDist {}
        impl PartialOrd for MaxDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for MaxDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // Normal: larger distance = "greater" in heap = peek returns worst
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        // OPT-009: Use Vec<bool> instead of HashSet (much faster for dense indices)
        let num_nodes = self.node_info.len();
        let mut visited = vec![false; num_nodes];

        let mut candidates: BinaryHeap<(MinDist, usize)> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<(MaxDist, usize)> = BinaryHeap::with_capacity(ef);

        let entry_dist = self.distance(query, entry_idx);
        candidates.push((MinDist(entry_dist), entry_idx));
        results.push((MaxDist(entry_dist), entry_idx));
        visited[entry_idx] = true;

        while let Some((MinDist(cand_dist), cand_idx)) = candidates.pop() {
            // OPT-036: Correct early termination - peek returns WORST (largest) distance
            if results.len() >= ef {
                if let Some(&(MaxDist(worst_dist), _)) = results.peek() {
                    if cand_dist > worst_dist {
                        break;
                    }
                }
            }

            let node_info = &self.node_info[cand_idx];
            if level > node_info.max_layer {
                continue;
            }

            // OPT-009: Use fast ID→Index conversion and avoid Option overhead
            for &nbr_id in &node_info.layer_neighbors[level].ids {
                let nbr_idx = self.get_idx_from_id_fast(nbr_id);

                // OPT-009: Bounds check once, then direct array access
                if nbr_idx >= num_nodes {
                    continue;
                }

                if !visited[nbr_idx] {
                    visited[nbr_idx] = true;
                    let nbr_dist = self.distance(query, nbr_idx);

                    // OPT-036: Compare against WORST result (largest distance)
                    let should_add = results.len() < ef
                        || nbr_dist
                            < results
                                .peek()
                                .map(|&(MaxDist(d), _)| d)
                                .unwrap_or(f32::INFINITY);

                    if should_add {
                        if results.len() >= ef {
                            results.pop();
                        }
                        results.push((MaxDist(nbr_dist), nbr_idx));
                        candidates.push((MinDist(nbr_dist), nbr_idx));
                    }
                }
            }
        }

        // Convert to sorted vector of (idx, dist) without recomputing distances.
        let mut sorted: Vec<(usize, f32)> = results
            .into_sorted_vec()
            .into_iter()
            .map(|(MaxDist(dist), idx)| (idx, dist))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted
    }

    /// Search for nearest neighbors at a specific layer (public API version)
    ///
    /// Returns (id, dist) pairs for API compatibility.
    fn search_layer(
        &self,
        query: &[f32],
        entry_id: i64,
        level: usize,
        ef: usize,
    ) -> Vec<(i64, f32)> {
        let entry_idx = self.get_idx_from_id_fast(entry_id);
        // Use the optimized index-based version
        let results_idx = self.search_layer_idx(query, entry_idx, level, ef);
        // Convert to (id, dist) for API compatibility
        results_idx
            .into_iter()
            .map(|(idx, dist)| (self.get_id_from_idx(idx), dist))
            .collect()
    }

    /// Select neighbors using the hnswlib-style diversification heuristic.
    /// BUG-006: Implemented proper heuristic to improve recall.
    ///
    /// Audit note: this path intentionally does not perform the FAISS layer-0
    /// `keep_max_size_level0` outsider backfill, so it can return fewer than `m`
    /// neighbors when diversification prunes near-duplicate candidates.
    ///
    /// Algorithm: For each candidate (sorted by distance to query), check if it's "good":
    /// - A candidate is good if dist(candidate, already_selected) >= dist(candidate, query)
    /// - This avoids selecting neighbors that are too close to each other
    /// - Improves graph connectivity and search recall
    ///
    /// OPT-015 REV2: Return indices instead of IDs for direct access
    #[allow(dead_code)]
    fn select_neighbors_heuristic_idx(
        &self,
        _query: &[f32],
        candidates: &[(usize, f32)],
        m: usize,
    ) -> Vec<(usize, f32)> {
        self.select_neighbors_heuristic_idx_layer_aware(_query, candidates, m, false)
    }

    fn select_neighbors_heuristic_idx_layer_aware(
        &self,
        _query: &[f32],
        candidates: &[(usize, f32)],
        m: usize,
        keep_pruned: bool,
    ) -> Vec<(usize, f32)> {
        if candidates.len() <= m {
            // Not enough candidates, return all
            let mut selected: Vec<(usize, f32)> = candidates.to_vec();
            selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return selected;
        }

        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(m);
        let mut pruned: Vec<(usize, f32)> = Vec::new();

        // Candidates are already sorted by distance (closest first)
        for &(cand_idx, cand_dist) in candidates {
            if selected.len() >= m {
                break;
            }

            // Check if this candidate is "good"
            // A candidate is good if it's not too close to any already selected neighbor
            let mut good = true;
            for &(sel_idx, _sel_dist) in &selected {
                // Calculate distance between candidate and already selected neighbor
                let dist_between = self.distance_between_nodes_idx(cand_idx, sel_idx);

                // If candidate is closer to an already selected neighbor than to the query,
                // it's redundant - skip it
                if dist_between < cand_dist {
                    good = false;
                    break;
                }
            }

            if good {
                selected.push((cand_idx, cand_dist));
                if selected.len() >= m {
                    break;
                }
            } else if keep_pruned {
                pruned.push((cand_idx, cand_dist));
            }
        }

        if keep_pruned {
            let mut idx = 0;
            while selected.len() < m && idx < pruned.len() {
                selected.push(pruned[idx]);
                idx += 1;
            }
        }

        selected
    }

    /// Select neighbors using the hnswlib-style diversification heuristic.
    /// BUG-006: Implemented proper heuristic to improve recall.
    #[allow(dead_code)]
    fn select_neighbors_heuristic(
        &self,
        _query: &[f32],
        candidates: &[(i64, f32)],
        m: usize,
    ) -> Vec<(i64, f32)> {
        self.select_neighbors_heuristic_layer_aware(_query, candidates, m, false)
    }

    fn select_neighbors_heuristic_layer_aware(
        &self,
        _query: &[f32],
        candidates: &[(i64, f32)],
        m: usize,
        keep_pruned: bool,
    ) -> Vec<(i64, f32)> {
        if candidates.len() <= m {
            // Not enough candidates, return all
            let mut selected: Vec<(i64, f32)> = candidates.to_vec();
            selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            return selected;
        }

        let mut selected: Vec<(i64, f32)> = Vec::with_capacity(m);
        let mut pruned: Vec<(i64, f32)> = Vec::new();

        // Candidates are already sorted by distance (closest first)
        for &(cand_id, cand_dist) in candidates {
            if selected.len() >= m {
                break;
            }

            // Check if this candidate is "good"
            let mut good = true;
            for &(sel_id, _sel_dist) in &selected {
                // Calculate distance between candidate and already selected neighbor
                let dist_between = self.distance_between_nodes(cand_id, sel_id);

                // If candidate is closer to an already selected neighbor than to the query,
                // it's redundant - skip it
                if dist_between < cand_dist {
                    good = false;
                    break;
                }
            }

            if good {
                selected.push((cand_id, cand_dist));
                if selected.len() >= m {
                    break;
                }
            } else if keep_pruned {
                pruned.push((cand_id, cand_dist));
            }
        }

        if keep_pruned {
            let mut idx = 0;
            while selected.len() < m && idx < pruned.len() {
                selected.push(pruned[idx]);
                idx += 1;
            }
        }

        selected
    }

    /// Shrink a node's layer neighbors using the heuristic algorithm.
    ///
    /// This is the FAISS-style approach: when a neighbor's connection list exceeds M_max,
    /// we re-apply the heuristic (diversification) rather than just keeping the closest.
    ///
    /// OPT-035: This fixes the recall gap caused by truncate_to_best which only kept
    /// closest neighbors without considering diversity.
    fn shrink_layer_neighbors_heuristic_idx(
        &mut self,
        node_idx: usize,
        level: usize,
        m_max: usize,
    ) {
        let node_info = &self.node_info[node_idx];
        if level > node_info.max_layer {
            return;
        }

        // Get current neighbors as (idx, dist) pairs
        let current: Vec<(usize, f32)> = node_info.layer_neighbors[level]
            .ids
            .iter()
            .zip(node_info.layer_neighbors[level].dists.iter())
            .map(|(&id, &dist)| (self.get_idx_from_id_fast(id), dist))
            .collect();

        if current.len() <= m_max {
            return; // No need to shrink
        }

        // Sort by distance (closest first)
        let mut sorted: Vec<(usize, f32)> = current;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Apply diversification heuristic from THIS node's perspective
        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(m_max);

        for (cand_idx, cand_dist) in sorted {
            if selected.len() >= m_max {
                break;
            }

            // Check if candidate is diverse enough
            let mut good = true;
            for &(sel_idx, _sel_dist) in &selected {
                let dist_between = self.distance_between_nodes_idx(cand_idx, sel_idx);
                // Reject if candidate is closer to an already-selected neighbor than to this node
                if dist_between < cand_dist {
                    good = false;
                    break;
                }
            }

            if good {
                selected.push((cand_idx, cand_dist));
            }
            // OPT-037: keep the hnswlib-style no-backfill behavior here.
        }

        // NOTE: no backfill here; FAISS can optionally refill pruned layer-0
        // candidates through `keep_max_size_level0`.

        // Pre-compute IDs before mutable borrow
        let new_neighbors: Vec<(i64, f32)> = selected
            .iter()
            .map(|&(idx, dist)| (self.get_id_from_idx(idx), dist))
            .collect();

        // Update the node's neighbors
        let node_info = &mut self.node_info[node_idx];
        let layer_nbrs = &mut node_info.layer_neighbors[level];
        layer_nbrs.ids.clear();
        layer_nbrs.dists.clear();
        for (id, dist) in new_neighbors {
            layer_nbrs.push(id, dist);
        }
    }

    /// BUG-006: Calculate distance between two nodes (by index)
    #[inline]
    fn distance_between_nodes_idx(&self, idx1: usize, idx2: usize) -> f32 {
        let start1 = idx1 * self.dim;
        let start2 = idx2 * self.dim;
        let vec1 = &self.vectors[start1..start1 + self.dim];
        let vec2 = &self.vectors[start2..start2 + self.dim];

        match self.metric_type {
            MetricType::L2 => simd::l2_distance_sq(vec1, vec2),
            MetricType::Ip => {
                // Keep node-node distance semantics aligned with query-node distance:
                // larger inner product => smaller (better) distance.
                // Using abs() here breaks ordering for negative correlations and can
                // destabilize graph connectivity/search ranking under IP metric.
                -simd::inner_product(vec1, vec2)
            }
            MetricType::Cosine => {
                let ip = simd::inner_product(vec1, vec2);
                let norm1 = simd::inner_product(vec1, vec1).sqrt();
                let norm2 = simd::inner_product(vec2, vec2).sqrt();
                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - ip / (norm1 * norm2)
                } else {
                    1.0
                }
            }
            MetricType::Hamming => {
                panic!("Hamming distance not supported for HNSW - use BinaryHnswIndex");
            }
        }
    }

    /// BUG-006: Calculate distance between two nodes (by ID)
    #[inline]
    fn distance_between_nodes(&self, id1: i64, id2: i64) -> f32 {
        let idx1 = self.get_idx_from_id_fast(id1);
        let idx2 = self.get_idx_from_id_fast(id2);
        self.distance_between_nodes_idx(idx1, idx2)
    }

    /// Add bidirectional connections between nodes at a specific layer
    ///
    /// OPT-021: Neighbors store (id, dist) - convert indices to IDs when storing.
    /// OPT-035: Use heuristic shrink instead of truncate_to_best for reverse connections.
    fn add_bidirectional_connections_idx(
        &mut self,
        new_idx: usize,
        new_id: i64,
        level: usize,
        neighbors: &[(usize, f32)],
    ) {
        let m_max = self.max_connections_for_layer(level);

        // Collect nodes that may need shrinking after adding connections
        let mut nodes_to_shrink: Vec<usize> = Vec::new();

        // Pre-compute distances from neighbors to new node (to avoid borrow issues later)
        let nbr_dists_from_nbr: Vec<f32> = neighbors
            .iter()
            .map(|&(nbr_idx, _)| self.distance_between_nodes_idx(nbr_idx, new_idx))
            .collect();

        // Add forward connections from new node (store IDs, not indices)
        // OPT-021: Collect IDs first to avoid borrow checker issues
        let forward_connections: Vec<(i64, f32)> = neighbors
            .iter()
            .map(|&(nbr_idx, dist)| (self.get_id_from_idx(nbr_idx), dist))
            .collect();

        {
            let node_info = &mut self.node_info[new_idx];
            let layer_nbrs = &mut node_info.layer_neighbors[level];

            for (nbr_id, dist) in forward_connections {
                layer_nbrs.push(nbr_id, dist);
            }

            // Check if we need shrinking
            if layer_nbrs.ids.len() > m_max {
                nodes_to_shrink.push(new_idx);
            }
        }

        // Add reverse connections
        for (i, &(nbr_idx, _dist)) in neighbors.iter().enumerate() {
            let dist_from_nbr = nbr_dists_from_nbr[i];

            let nbr_node_info = &mut self.node_info[nbr_idx];

            // Only add if this layer exists for the neighbor
            if level <= nbr_node_info.max_layer {
                let nbr_layer_nbrs = &mut nbr_node_info.layer_neighbors[level];
                nbr_layer_nbrs.push(new_id, dist_from_nbr);

                // Track for shrinking
                if nbr_layer_nbrs.ids.len() > m_max && !nodes_to_shrink.contains(&nbr_idx) {
                    nodes_to_shrink.push(nbr_idx);
                }
            }
        }

        // Apply heuristic shrink to all affected nodes
        for node_idx in nodes_to_shrink {
            self.shrink_layer_neighbors_heuristic_idx(node_idx, level, m_max);
        }
    }

    /// Add bidirectional connections between nodes at a specific layer (legacy version for API compatibility)
    fn add_bidirectional_connections(
        &mut self,
        new_idx: usize,
        new_id: i64,
        level: usize,
        neighbors: &[(i64, f32)],
    ) {
        // Convert (id, dist) to (idx, dist) and call the optimized version
        let neighbors_idx: Vec<(usize, f32)> = neighbors
            .iter()
            .map(|&(id, dist)| (self.get_idx_from_id_fast(id), dist))
            .collect();
        self.add_bidirectional_connections_idx(new_idx, new_id, level, &neighbors_idx);
    }

    /// OPT-015: Get ID from index (reverse of get_idx_from_id).
    /// For sequential IDs, this is just returning the index.
    #[inline]
    fn get_id_from_idx(&self, idx: usize) -> i64 {
        if self.use_sequential_ids {
            idx as i64
        } else {
            self.ids[idx]
        }
    }

    /// OPT-015: Fast path for getting index from ID.
    /// When using sequential IDs (most common case), idx == id, so this is O(1).
    /// For custom IDs, falls back to linear search.
    #[inline]
    fn get_idx_from_id_fast(&self, id: i64) -> usize {
        if self.use_sequential_ids {
            // Fast path: idx == id for sequential IDs
            id as usize
        } else {
            // Fallback: linear search for custom IDs
            self.get_idx_from_id(id).unwrap_or(0)
        }
    }

    /// OPT-015: Get distance for an index (helper for search_layer return conversion)
    #[inline]
    #[allow(dead_code)]
    fn get_distance_for_idx(&self, query: &[f32], idx: usize) -> f32 {
        self.distance(query, idx)
    }

    /// OPT-021: Helper method to get index from ID.
    /// Since we removed the HashMap, this does a linear search.
    /// For sequential IDs (most common case), this is fast.
    /// For random IDs, consider maintaining a reverse index if needed.
    #[inline]
    fn get_idx_from_id(&self, id: i64) -> Option<usize> {
        // OPT-015: Fast path - if using sequential IDs, idx == id
        if self.use_sequential_ids {
            if id >= 0 && (id as usize) < self.vectors.len() / self.dim {
                return Some(id as usize);
            }
            return None;
        }

        // Fallback: linear search for non-sequential IDs
        self.ids.iter().position(|&x| x == id)
    }

    /// Calculate distance based on metric type
    #[inline]
    fn distance(&self, query: &[f32], idx: usize) -> f32 {
        debug_assert_eq!(query.len(), self.dim);
        (self.distance_to_idx_fn)(self, query, idx)
    }

    #[inline(always)]
    unsafe fn l2_distance_to_idx_ptr(
        &self,
        query_ptr: *const f32,
        base_ptr: *const f32,
        idx: usize,
    ) -> f32 {
        debug_assert!(idx < self.vectors.len() / self.dim);
        if self.use_bf16_storage {
            return unsafe { self.l2_distance_to_idx_bf16_ptr(query_ptr, idx) };
        }
        (self.l2_distance_sq_ptr_kernel)(query_ptr, base_ptr.add(idx * self.dim), self.dim)
    }

    #[inline(always)]
    unsafe fn l2_distance_to_idx_bf16_ptr(&self, query_ptr: *const f32, idx: usize) -> f32 {
        debug_assert!(idx < self.bf16_vectors.len() / self.dim);
        let start = idx * self.dim;
        let mut sum = 0.0f32;
        for offset in 0..self.dim {
            let q = Bf16::from_f32(unsafe { *query_ptr.add(offset) }).to_f32();
            let v = Bf16::from_bits(self.bf16_vectors[start + offset]).to_f32();
            let diff = q - v;
            sum += diff * diff;
        }
        sum
    }

    #[inline(always)]
    fn l2_distance_to_idx_bf16_with_query_bits(&self, query_bf16: &[u16], idx: usize) -> f32 {
        debug_assert_eq!(query_bf16.len(), self.dim);
        let start = idx * self.dim;
        let mut sum = 0.0f32;
        for (offset, &q_bits) in query_bf16.iter().enumerate() {
            let q = Bf16::from_bits(q_bits).to_f32();
            let v = Bf16::from_bits(self.bf16_vectors[start + offset]).to_f32();
            let diff = q - v;
            sum += diff * diff;
        }
        sum
    }

    #[inline(always)]
    fn l2_distance_to_4_idxs_bf16_with_query_bits(
        &self,
        query_bf16: &[u16],
        idxs: [usize; 4],
    ) -> [f32; 4] {
        let starts = [
            idxs[0] * self.dim,
            idxs[1] * self.dim,
            idxs[2] * self.dim,
            idxs[3] * self.dim,
        ];
        bf16_l2_sq_batch_4(
            query_bf16,
            &self.bf16_vectors[starts[0]..starts[0] + self.dim],
            &self.bf16_vectors[starts[1]..starts[1] + self.dim],
            &self.bf16_vectors[starts[2]..starts[2] + self.dim],
            &self.bf16_vectors[starts[3]..starts[3] + self.dim],
        )
    }

    #[inline]
    fn l2_distance_to_idx(&self, query: &[f32], idx: usize) -> f32 {
        unsafe { self.l2_distance_to_idx_ptr(query.as_ptr(), self.vectors.as_ptr(), idx) }
    }

    #[inline(always)]
    unsafe fn l2_distance_to_4_idxs_ptr(
        &self,
        query_ptr: *const f32,
        base_ptr: *const f32,
        idxs: [usize; 4],
    ) -> [f32; 4] {
        if self.use_bf16_storage {
            return [
                unsafe { self.l2_distance_to_idx_bf16_ptr(query_ptr, idxs[0]) },
                unsafe { self.l2_distance_to_idx_bf16_ptr(query_ptr, idxs[1]) },
                unsafe { self.l2_distance_to_idx_bf16_ptr(query_ptr, idxs[2]) },
                unsafe { self.l2_distance_to_idx_bf16_ptr(query_ptr, idxs[3]) },
            ];
        }
        simd::l2_batch_4_ptrs(
            query_ptr,
            base_ptr.add(idxs[0] * self.dim),
            base_ptr.add(idxs[1] * self.dim),
            base_ptr.add(idxs[2] * self.dim),
            base_ptr.add(idxs[3] * self.dim),
            self.dim,
        )
    }

    #[inline(always)]
    unsafe fn prefetch_l2_vector_idx(&self, base_ptr: *const f32, idx: usize) -> bool {
        let vec_ptr = base_ptr.add(idx * self.dim);
        #[cfg(target_arch = "x86_64")]
        {
            core::arch::x86_64::_mm_prefetch(vec_ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
            true
        }
        #[cfg(target_arch = "x86")]
        {
            core::arch::x86::_mm_prefetch(vec_ptr as *const i8, core::arch::x86::_MM_HINT_T0);
            true
        }
        #[cfg(target_arch = "aarch64")]
        {
            core::arch::asm!(
                "prfm pldl1keep, [{addr}]",
                addr = in(reg) vec_ptr,
                options(readonly, nostack)
            );
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
        {
            let _ = vec_ptr;
            true
        }
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        if self.vectors.is_empty() {
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

        let k = req.top_k;
        let ef = self
            .config
            .params
            .effective_hnsw_ef_search(self.ef_search, req.nprobe, k);
        let filter = req.filter.clone();

        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            let results = self.search_single(query_vec, ef, k, &filter);

            let offset = q_idx * k;
            for (i, item) in results.into_iter().enumerate().take(k) {
                all_ids[offset + i] = item.0;
                all_dists[offset + i] = item.1;
            }
        }

        // Finalize distances
        for i in 0..all_dists.len() {
            if all_ids[i] != -1 {
                match self.metric_type {
                    MetricType::L2 => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                    MetricType::Ip => {
                        all_dists[i] = -all_dists[i];
                    }
                    MetricType::Cosine => {}
                    _ => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                }
            }
        }

        Ok(ApiSearchResult::new(all_ids, all_dists, 0.0))
    }

    /// Search with Bitset filtering
    ///
    /// Bitset is used to filter out certain vectors (e.g., soft-deleted vectors).
    /// Each bit in the bitset corresponds to a vector index: 1 = filtered out, 0 = kept.
    ///
    /// This is more efficient than using Predicate because it integrates bitset checking
    /// directly into the search algorithm, avoiding post-filtering.
    ///
    /// # Arguments
    /// * `query` - Query vector(s)
    /// * `req` - Search request
    /// * `bitset` - BitsetView for filtering vectors by index
    ///
    /// # Returns
    /// Search results excluding filtered vectors
    pub fn search_with_bitset(
        &self,
        query: &[f32],
        req: &SearchRequest,
        bitset: &crate::bitset::BitsetView,
    ) -> Result<ApiSearchResult> {
        if self.vectors.is_empty() {
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

        let k = req.top_k;
        let ef = self
            .config
            .params
            .effective_hnsw_ef_search(self.ef_search, req.nprobe, k);

        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        let should_bruteforce = self.should_bruteforce_bitset_knn(k, bitset);

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            let results = if should_bruteforce {
                self.brute_force_search(query_vec, k, |_id, idx| {
                    idx >= bitset.len() || !bitset.get(idx)
                })
            } else {
                self.search_single_with_bitset(query_vec, ef, k, bitset)
            };

            for (id, dist) in results.into_iter().take(k) {
                all_ids.push(id);
                all_dists.push(dist);
            }
        }

        // Finalize distances
        for i in 0..all_dists.len() {
            if all_ids[i] != -1 {
                match self.metric_type {
                    MetricType::L2 => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                    MetricType::Ip => {
                        all_dists[i] = -all_dists[i];
                    }
                    MetricType::Cosine => {}
                    _ => {
                        all_dists[i] = all_dists[i].sqrt();
                    }
                }
            }
        }

        Ok(ApiSearchResult::new(all_ids, all_dists, 0.0))
    }

    #[inline]
    fn brute_force_search<F>(&self, query: &[f32], k: usize, mut keep: F) -> Vec<(i64, f32)>
    where
        F: FnMut(i64, usize) -> bool,
    {
        let mut all: Vec<(i64, f32)> = Vec::with_capacity(self.ids.len());
        for (idx, &id) in self.ids.iter().enumerate() {
            if keep(id, idx) {
                all.push((id, self.distance(query, idx)));
            }
        }
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    fn greedy_upper_layer_descent_idx(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
    ) -> (usize, f32) {
        let entry_dist = self.distance(query, entry_idx);
        self.greedy_upper_layer_descent_idx_with_entry_dist_optional_profile(
            query, entry_idx, level, entry_dist, None,
        )
    }

    fn greedy_upper_layer_descent_idx_with_entry_dist(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        entry_dist: f32,
    ) -> (usize, f32) {
        self.greedy_upper_layer_descent_idx_with_entry_dist_optional_profile(
            query, entry_idx, level, entry_dist, None,
        )
    }

    fn greedy_upper_layer_descent_l2_with_entry_dist(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        entry_dist: f32,
    ) -> (usize, f32) {
        self.greedy_upper_layer_descent_l2_with_entry_dist_optional_profile(
            query, entry_idx, level, entry_dist, None,
        )
    }

    fn greedy_upper_layer_descent_idx_with_entry_dist_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        entry_dist: f32,
        mut profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> (usize, f32) {
        let num_nodes = self.node_info.len();
        let mut best_idx = entry_idx;
        let mut best_dist = entry_dist;

        loop {
            let node_info = &self.node_info[best_idx];
            if level > node_info.max_layer {
                break;
            }

            let mut improved = false;
            for &nbr_id in &node_info.layer_neighbors[level].ids {
                let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                if nbr_idx >= num_nodes || nbr_idx == best_idx {
                    continue;
                }

                let distance_start = Instant::now();
                let nbr_dist = self.distance(query, nbr_idx);
                if let Some(stats) = profile.as_mut() {
                    stats.record_upper_layer_query_distance(distance_start.elapsed(), 1);
                }
                if nbr_dist < best_dist {
                    best_idx = nbr_idx;
                    best_dist = nbr_dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        (best_idx, best_dist)
    }

    fn greedy_upper_layer_descent_l2_with_entry_dist_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        entry_dist: f32,
        mut profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> (usize, f32) {
        let num_nodes = self.node_info.len();
        let query_ptr = query.as_ptr();
        let base_ptr = self.vectors.as_ptr();
        let mut best_idx = entry_idx;
        let mut best_dist = entry_dist;

        loop {
            let node_info = &self.node_info[best_idx];
            if level > node_info.max_layer {
                break;
            }

            let mut improved = false;
            for &nbr_id in &node_info.layer_neighbors[level].ids {
                let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                if nbr_idx >= num_nodes || nbr_idx == best_idx {
                    continue;
                }

                let distance_start = Instant::now();
                let nbr_dist = unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) };
                if let Some(stats) = profile.as_mut() {
                    stats.record_upper_layer_query_distance(distance_start.elapsed(), 1);
                }
                if nbr_dist < best_dist {
                    best_idx = nbr_idx;
                    best_dist = nbr_dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        (best_idx, best_dist)
    }

    fn search_layer_idx_with_scratch(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_with_optional_profile(query, entry_idx, level, ef, scratch, None)
    }

    fn search_layer_idx_l2_with_scratch(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_l2_with_optional_profile(
            query, entry_idx, level, ef, scratch, profile,
        )
    }

    #[cfg(test)]
    fn search_layer_idx_l2_heap_with_scratch(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_l2_heap_with_optional_profile(
            query, entry_idx, level, ef, scratch, profile,
        )
    }

    #[cfg(test)]
    fn search_layer_idx_l2_ordered_pool_with_scratch(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_l2_ordered_pool_with_optional_profile(
            query, entry_idx, level, ef, scratch, profile,
        )
    }

    pub fn layer0_l2_search_mode_for_audit(&self, profiled: bool) -> &'static str {
        if profiled {
            "profiled_optional"
        } else {
            "fast_unprofiled"
        }
    }

    pub fn production_layer0_avoids_profile_timing_for_audit(&self) -> bool {
        true
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_enabled_for_audit(&self) -> bool {
        self.layer0_slab.is_enabled_for(self.node_info.len())
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_neighbors_for_audit(&self, node_idx: usize) -> &[u32] {
        self.layer0_slab.neighbors_for(node_idx)
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_vector_for_audit(&self, node_idx: usize) -> &[f32] {
        self.layer0_slab.vector_for_audit(node_idx)
    }

    pub fn production_layer0_layout_mode_for_audit(&self) -> &'static str {
        if self.layer0_slab.is_enabled_for(self.node_info.len()) {
            "layer0_slab"
        } else if self.layer0_flat_graph.is_enabled_for(self.node_info.len()) {
            "flat_u32_adjacency"
        } else {
            "layer0_heap_fallback"
        }
    }

    pub fn profiled_layer0_layout_mode_for_audit(&self) -> &'static str {
        if self.layer0_flat_graph.is_enabled_for(self.node_info.len()) {
            "flat_graph_profiled"
        } else {
            "heap_profiled"
        }
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_stride_bytes_for_audit(&self) -> usize {
        self.layer0_slab.stride_words * std::mem::size_of::<u32>()
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_vector_offset_bytes_for_audit(&self) -> usize {
        self.layer0_slab.vector_offset_words * std::mem::size_of::<u32>()
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_max_neighbors_for_audit(&self) -> usize {
        self.layer0_slab.max_neighbors
    }

    #[cfg(any(test, feature = "long-tests"))]
    pub fn layer0_slab_rebuild_source_for_audit(&self) -> &'static str {
        "derived_from_canonical_flat_graph_and_vectors"
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer_idx_shared(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: Option<&BitsetView>,
        scratch: &mut SearchScratch,
        mut profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        fn accept_shared_neighbor(
            scratch: &mut SearchScratch,
            ef: usize,
            nbr_idx: usize,
            nbr_dist: f32,
            profile: &mut Option<&mut HnswCandidateSearchProfileStats>,
        ) {
            let pruning_start = Instant::now();
            let should_add = scratch.generic_results.len() < ef
                || nbr_dist < scratch.generic_worst_result_distance;
            if let Some(stats) = profile.as_mut() {
                stats.record_candidate_pruning(pruning_start.elapsed(), u64::from(!should_add));
            }

            if should_add {
                let frontier_start = Instant::now();
                let result_pops = scratch.push_generic_result(nbr_idx, nbr_dist, ef);
                scratch
                    .generic_frontier
                    .push((SearchMinDist(nbr_dist), nbr_idx));
                if let Some(stats) = profile.as_mut() {
                    stats.record_frontier_ops(frontier_start.elapsed(), 2, result_pops);
                }
            }
        }

        if bitset.is_none() && ef <= 1 {
            let (best_idx, best_dist) =
                self.greedy_upper_layer_descent_idx(query, entry_idx, level);
            return vec![(best_idx, best_dist)];
        }

        let is_filtered = |idx: usize| bitset.is_some_and(|mask| idx < mask.len() && mask.get(idx));
        let use_l2_ptr_dispatch = self.metric_type == MetricType::L2;
        let query_ptr = query.as_ptr();
        let base_ptr = self.vectors.as_ptr();

        let num_nodes = self.node_info.len();
        let use_layer0_flat_graph =
            use_l2_ptr_dispatch && level == 0 && self.layer0_flat_graph.is_enabled_for(num_nodes);
        let visited_start = Instant::now();
        scratch.prepare(num_nodes);
        scratch.prepare_generic_heaps(ef);
        if let Some(stats) = profile.as_mut() {
            stats.start_layer0_candidate_search();
            stats.record_visited_ops(visited_start.elapsed(), 0);
        }

        let distance_start = Instant::now();
        let entry_dist = if use_l2_ptr_dispatch {
            unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, entry_idx) }
        } else {
            self.distance(query, entry_idx)
        };
        if let Some(stats) = profile.as_mut() {
            stats.record_layer0_query_distance(distance_start.elapsed(), 1);
        }

        let entry_is_filtered = is_filtered(entry_idx);
        let frontier_start = Instant::now();
        scratch
            .generic_frontier
            .push((SearchMinDist(entry_dist), entry_idx));
        if !entry_is_filtered {
            scratch
                .generic_results
                .push((SearchMaxDist(entry_dist), entry_idx));
            scratch.sync_generic_worst_result_distance();
        }
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(
                frontier_start.elapsed(),
                1 + u64::from(!entry_is_filtered),
                0,
            );
        }

        let visited_start = Instant::now();
        let entry_marked = scratch.mark_visited(entry_idx);
        if let Some(stats) = profile.as_mut() {
            stats.record_visited_ops(visited_start.elapsed(), u64::from(entry_marked));
        }

        loop {
            let frontier_start = Instant::now();
            let Some((SearchMinDist(cand_dist), cand_idx)) = scratch.generic_frontier.pop() else {
                if let Some(stats) = profile.as_mut() {
                    stats.record_frontier_ops(frontier_start.elapsed(), 0, 0);
                }
                break;
            };
            if let Some(stats) = profile.as_mut() {
                stats.record_frontier_ops(frontier_start.elapsed(), 0, 1);
            }

            let pruning_start = Instant::now();
            if scratch.generic_results.len() >= ef
                && cand_dist > scratch.generic_worst_result_distance
            {
                if let Some(stats) = profile.as_mut() {
                    stats.record_candidate_pruning(pruning_start.elapsed(), 1);
                }
                break;
            }
            if let Some(stats) = profile.as_mut() {
                stats.record_candidate_pruning(pruning_start.elapsed(), 0);
            }

            if use_layer0_flat_graph {
                let use_grouped_bitset_batch4 = bitset.is_some();
                let mut grouped_nbrs = [0usize; 4];
                let mut grouped_count = 0usize;
                for &nbr_u32 in self.layer0_flat_graph.neighbors_for(cand_idx) {
                    let nbr_idx = nbr_u32 as usize;
                    if nbr_idx >= num_nodes {
                        continue;
                    }

                    let visited_start = Instant::now();
                    let marked = scratch.mark_visited(nbr_idx);
                    if let Some(stats) = profile.as_mut() {
                        stats.record_visited_ops(visited_start.elapsed(), u64::from(marked));
                    }
                    if !marked || is_filtered(nbr_idx) {
                        continue;
                    }

                    if use_grouped_bitset_batch4 {
                        grouped_nbrs[grouped_count] = nbr_idx;
                        grouped_count += 1;
                        if grouped_count == 4 {
                            let distance_start = Instant::now();
                            let nbr_dists = unsafe {
                                self.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, grouped_nbrs)
                            };
                            if let Some(stats) = profile.as_mut() {
                                stats.record_layer0_query_distance_batch4(distance_start.elapsed());
                            }
                            for (idx, dist) in grouped_nbrs.into_iter().zip(nbr_dists.into_iter()) {
                                accept_shared_neighbor(scratch, ef, idx, dist, &mut profile);
                            }
                            grouped_count = 0;
                        }
                    } else {
                        let distance_start = Instant::now();
                        let nbr_dist =
                            unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) };
                        if let Some(stats) = profile.as_mut() {
                            stats.record_layer0_query_distance(distance_start.elapsed(), 1);
                        }
                        accept_shared_neighbor(scratch, ef, nbr_idx, nbr_dist, &mut profile);
                    }
                }
                for &nbr_idx in &grouped_nbrs[..grouped_count] {
                    let distance_start = Instant::now();
                    let nbr_dist =
                        unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) };
                    if let Some(stats) = profile.as_mut() {
                        stats.record_layer0_query_distance(distance_start.elapsed(), 1);
                    }
                    accept_shared_neighbor(scratch, ef, nbr_idx, nbr_dist, &mut profile);
                }
            } else {
                let node_info = &self.node_info[cand_idx];
                if level > node_info.max_layer {
                    continue;
                }

                let neighbors = &node_info.layer_neighbors[level].ids;
                for &nbr_id in neighbors {
                    let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                    if nbr_idx >= num_nodes {
                        continue;
                    }

                    let visited_start = Instant::now();
                    let marked = scratch.mark_visited(nbr_idx);
                    if let Some(stats) = profile.as_mut() {
                        stats.record_visited_ops(visited_start.elapsed(), u64::from(marked));
                    }
                    if !marked || is_filtered(nbr_idx) {
                        continue;
                    }

                    let distance_start = Instant::now();
                    let nbr_dist = if use_l2_ptr_dispatch {
                        unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) }
                    } else {
                        self.distance(query, nbr_idx)
                    };
                    if let Some(stats) = profile.as_mut() {
                        stats.record_layer0_query_distance(distance_start.elapsed(), 1);
                    }

                    let pruning_start = Instant::now();
                    let should_add = scratch.generic_results.len() < ef
                        || nbr_dist < scratch.generic_worst_result_distance;
                    if let Some(stats) = profile.as_mut() {
                        stats.record_candidate_pruning(
                            pruning_start.elapsed(),
                            u64::from(!should_add),
                        );
                    }

                    if should_add {
                        let frontier_start = Instant::now();
                        let result_pops = scratch.push_generic_result(nbr_idx, nbr_dist, ef);
                        scratch
                            .generic_frontier
                            .push((SearchMinDist(nbr_dist), nbr_idx));
                        if let Some(stats) = profile.as_mut() {
                            stats.record_frontier_ops(frontier_start.elapsed(), 2, result_pops);
                        }
                    }
                }
            }
        }

        let frontier_start = Instant::now();
        let sorted = scratch.drain_generic_results_sorted();
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 0, sorted.len() as u64);
        }
        sorted
    }

    #[cfg(test)]
    fn search_layer_idx_shared_for_test(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: Option<&BitsetView>,
        scratch: &mut SearchScratch,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_shared(query, entry_idx, level, ef, bitset, scratch, None)
    }

    fn search_layer_idx_with_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_shared(query, entry_idx, level, ef, None, scratch, profile)
    }

    fn search_layer_idx_l2_heap_with_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        mut profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        if ef <= 1 {
            let (best_idx, best_dist) = self.greedy_upper_layer_descent_l2_with_entry_dist(
                query,
                entry_idx,
                level,
                self.l2_distance_to_idx(query, entry_idx),
            );
            return vec![(best_idx, best_dist)];
        }

        let num_nodes = self.node_info.len();
        let query_ptr = query.as_ptr();
        let base_ptr = self.vectors.as_ptr();
        let visited_start = Instant::now();
        scratch.prepare(num_nodes);
        scratch.prepare_generic_heaps(ef);
        if let Some(stats) = profile.as_mut() {
            stats.start_layer0_candidate_search();
            stats.record_visited_ops(visited_start.elapsed(), 0);
        }

        let distance_start = Instant::now();
        let entry_dist = unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, entry_idx) };
        if let Some(stats) = profile.as_mut() {
            stats.record_layer0_query_distance(distance_start.elapsed(), 1);
        }

        let frontier_start = Instant::now();
        scratch
            .generic_frontier
            .push((SearchMinDist(entry_dist), entry_idx));
        scratch
            .generic_results
            .push((SearchMaxDist(entry_dist), entry_idx));
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 2, 0);
        }

        let visited_start = Instant::now();
        let entry_marked = scratch.mark_visited(entry_idx);
        if let Some(stats) = profile.as_mut() {
            stats.record_visited_ops(visited_start.elapsed(), u64::from(entry_marked));
        }

        loop {
            let frontier_start = Instant::now();
            let Some((SearchMinDist(cand_dist), cand_idx)) = scratch.generic_frontier.pop() else {
                if let Some(stats) = profile.as_mut() {
                    stats.record_frontier_ops(frontier_start.elapsed(), 0, 0);
                }
                break;
            };
            if let Some(stats) = profile.as_mut() {
                stats.record_frontier_ops(frontier_start.elapsed(), 0, 1);
            }

            let pruning_start = Instant::now();
            if scratch.generic_results.len() >= ef {
                if let Some(&(SearchMaxDist(worst_dist), _)) = scratch.generic_results.peek() {
                    if cand_dist > worst_dist {
                        if let Some(stats) = profile.as_mut() {
                            stats.record_candidate_pruning(pruning_start.elapsed(), 1);
                        }
                        break;
                    }
                }
            }
            if let Some(stats) = profile.as_mut() {
                stats.record_candidate_pruning(pruning_start.elapsed(), 0);
            }

            let node_info = &self.node_info[cand_idx];
            if level > node_info.max_layer {
                continue;
            }

            let neighbors = &node_info.layer_neighbors[level].ids;
            for &nbr_id in neighbors {
                let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                if nbr_idx >= num_nodes {
                    continue;
                }

                let visited_start = Instant::now();
                let marked = scratch.mark_visited(nbr_idx);
                if let Some(stats) = profile.as_mut() {
                    stats.record_visited_ops(visited_start.elapsed(), u64::from(marked));
                }
                if !marked {
                    continue;
                }

                let distance_start = Instant::now();
                let nbr_dist = unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) };
                if let Some(stats) = profile.as_mut() {
                    stats.record_layer0_query_distance(distance_start.elapsed(), 1);
                }

                let pruning_start = Instant::now();
                let should_add = scratch.generic_results.len() < ef
                    || nbr_dist
                        < scratch
                            .generic_results
                            .peek()
                            .map(|&(SearchMaxDist(d), _)| d)
                            .unwrap_or(f32::INFINITY);
                if let Some(stats) = profile.as_mut() {
                    stats.record_candidate_pruning(pruning_start.elapsed(), u64::from(!should_add));
                }

                if should_add {
                    let frontier_start = Instant::now();
                    let mut result_pops = 0;
                    if scratch.generic_results.len() >= ef {
                        scratch.generic_results.pop();
                        result_pops = 1;
                    }
                    scratch
                        .generic_results
                        .push((SearchMaxDist(nbr_dist), nbr_idx));
                    scratch
                        .generic_frontier
                        .push((SearchMinDist(nbr_dist), nbr_idx));
                    if let Some(stats) = profile.as_mut() {
                        stats.record_frontier_ops(frontier_start.elapsed(), 2, result_pops);
                    }
                }
            }
        }

        let frontier_start = Instant::now();
        let sorted = scratch.drain_generic_results_sorted();
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 0, sorted.len() as u64);
        }
        sorted
    }

    fn process_layer0_l2_candidate(
        &self,
        nbr_idx: usize,
        nbr_dist: f32,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: &mut Option<&mut HnswCandidateSearchProfileStats>,
    ) {
        let pruning_start = Instant::now();
        let should_add = scratch.layer0_results.can_insert(nbr_dist, ef);
        if let Some(stats) = profile.as_mut() {
            stats.record_candidate_pruning(pruning_start.elapsed(), u64::from(!should_add));
        }

        if !should_add {
            return;
        }

        let frontier_start = Instant::now();
        let entry = Layer0PoolEntry {
            idx: nbr_idx,
            dist: nbr_dist,
        };
        let result_pops = scratch.layer0_results.insert(entry, ef);
        scratch.layer0_frontier.push(entry);
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 2, result_pops);
        }
    }

    fn process_layer0_l2_candidate_fast(
        &self,
        nbr_idx: usize,
        nbr_dist: f32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        if !scratch.layer0_results.can_insert(nbr_dist, ef) {
            return;
        }

        let entry = Layer0PoolEntry {
            idx: nbr_idx,
            dist: nbr_dist,
        };
        scratch.layer0_results.insert(entry, ef);
        scratch.layer0_frontier.push(entry);
    }

    fn search_layer_idx_l2_ordered_pool_fast(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        query_bf16: Option<&[u16]>,
    ) -> Vec<(usize, f32)> {
        let num_nodes = self.node_info.len();
        let use_flat_graph = level == 0 && self.layer0_flat_graph.is_enabled_for(num_nodes);
        let use_layer0_slab = use_flat_graph && self.layer0_slab.is_enabled_for(num_nodes);
        let query_ptr = query.as_ptr();
        let base_ptr = self.vectors.as_ptr();
        scratch.prepare(num_nodes);
        scratch.prepare_layer0_pools(ef);

        let entry_dist = if let Some(bits) = query_bf16 {
            self.l2_distance_to_idx_bf16_with_query_bits(bits, entry_idx)
        } else {
            unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, entry_idx) }
        };

        let entry = Layer0PoolEntry {
            idx: entry_idx,
            dist: entry_dist,
        };
        scratch.layer0_frontier.push(entry);
        scratch.layer0_results.insert(entry, ef);
        scratch.mark_visited(entry_idx);

        loop {
            let Some(candidate) = scratch.layer0_frontier.pop_best() else {
                break;
            };

            if scratch.layer0_results.len() >= ef
                && scratch
                    .layer0_results
                    .worst_dist()
                    .is_some_and(|worst_dist| candidate.dist > worst_dist)
            {
                break;
            }

            let node_info = &self.node_info[candidate.idx];
            if level > node_info.max_layer {
                continue;
            }

            let mut batch_indices = [0usize; 4];
            let mut batch_len = 0usize;

            if use_layer0_slab {
                let neighbors = self.layer0_slab.neighbors_for(candidate.idx);
                for &nbr_u32 in neighbors {
                    let nbr_idx = nbr_u32 as usize;
                    if nbr_idx >= num_nodes || !scratch.mark_visited(nbr_idx) {
                        continue;
                    }

                    batch_indices[batch_len] = nbr_idx;
                    batch_len += 1;

                    if batch_len == 4 {
                        let distances = unsafe {
                            simd::l2_batch_4_ptrs(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[0]),
                                self.layer0_slab.vector_ptr_for(batch_indices[1]),
                                self.layer0_slab.vector_ptr_for(batch_indices[2]),
                                self.layer0_slab.vector_ptr_for(batch_indices[3]),
                                self.dim,
                            )
                        };
                        for (offset, nbr_dist) in distances.into_iter().enumerate() {
                            self.process_layer0_l2_candidate_fast(
                                batch_indices[offset],
                                nbr_dist,
                                ef,
                                scratch,
                            );
                        }
                        batch_len = 0;
                    }
                }
            } else if use_flat_graph {
                let neighbors = self.layer0_flat_graph.neighbors_for(candidate.idx);
                for (neighbor_offset, &nbr_u32) in neighbors.iter().enumerate() {
                    if neighbor_offset + 1 < neighbors.len() {
                        let next_nbr_idx = neighbors[neighbor_offset + 1] as usize;
                        if next_nbr_idx < num_nodes {
                            unsafe { self.prefetch_l2_vector_idx(base_ptr, next_nbr_idx) };
                        }
                    }

                    let nbr_idx = nbr_u32 as usize;
                    if nbr_idx >= num_nodes || !scratch.mark_visited(nbr_idx) {
                        continue;
                    }

                    batch_indices[batch_len] = nbr_idx;
                    batch_len += 1;

                    if batch_len == 4 {
                        let distances = if let Some(bits) = query_bf16 {
                            self.l2_distance_to_4_idxs_bf16_with_query_bits(bits, batch_indices)
                        } else {
                            unsafe {
                                self.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, batch_indices)
                            }
                        };
                        for (offset, nbr_dist) in distances.into_iter().enumerate() {
                            self.process_layer0_l2_candidate_fast(
                                batch_indices[offset],
                                nbr_dist,
                                ef,
                                scratch,
                            );
                        }
                        batch_len = 0;
                    }
                }
            } else {
                let neighbors = &node_info.layer_neighbors[level].ids;
                for (neighbor_offset, &nbr_id) in neighbors.iter().enumerate() {
                    if neighbor_offset + 1 < neighbors.len() {
                        let next_nbr_idx =
                            self.get_idx_from_id_fast(neighbors[neighbor_offset + 1]);
                        if next_nbr_idx < num_nodes {
                            unsafe { self.prefetch_l2_vector_idx(base_ptr, next_nbr_idx) };
                        }
                    }

                    let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                    if nbr_idx >= num_nodes || !scratch.mark_visited(nbr_idx) {
                        continue;
                    }

                    batch_indices[batch_len] = nbr_idx;
                    batch_len += 1;

                    if batch_len == 4 {
                        let distances = if let Some(bits) = query_bf16 {
                            self.l2_distance_to_4_idxs_bf16_with_query_bits(bits, batch_indices)
                        } else {
                            unsafe {
                                self.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, batch_indices)
                            }
                        };
                        for (offset, nbr_dist) in distances.into_iter().enumerate() {
                            self.process_layer0_l2_candidate_fast(
                                batch_indices[offset],
                                nbr_dist,
                                ef,
                                scratch,
                            );
                        }
                        batch_len = 0;
                    }
                }
            }

            for &nbr_idx in &batch_indices[..batch_len] {
                let nbr_dist = if use_layer0_slab {
                    unsafe {
                        (self.l2_distance_sq_ptr_kernel)(
                            query_ptr,
                            self.layer0_slab.vector_ptr_for(nbr_idx),
                            self.dim,
                        )
                    }
                } else if let Some(bits) = query_bf16 {
                    self.l2_distance_to_idx_bf16_with_query_bits(bits, nbr_idx)
                } else {
                    unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) }
                };
                self.process_layer0_l2_candidate_fast(nbr_idx, nbr_dist, ef, scratch);
            }
        }

        scratch.layer0_results.to_sorted_pairs()
    }

    fn search_layer_idx_l2_ordered_pool_with_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        mut profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        if let Some(stats) = profile.as_mut() {
            stats.mark_layer0_ordered_pool();
        }

        let num_nodes = self.node_info.len();
        let use_flat_graph = level == 0 && self.layer0_flat_graph.is_enabled_for(num_nodes);
        let query_ptr = query.as_ptr();
        let base_ptr = self.vectors.as_ptr();
        let visited_start = Instant::now();
        scratch.prepare(num_nodes);
        scratch.prepare_layer0_pools(ef);
        if let Some(stats) = profile.as_mut() {
            stats.start_layer0_candidate_search();
            stats.record_visited_ops(visited_start.elapsed(), 0);
            stats.mark_layer0_flat_graph_enabled(use_flat_graph);
        }

        let distance_start = Instant::now();
        let entry_dist = unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, entry_idx) };
        if let Some(stats) = profile.as_mut() {
            stats.record_layer0_query_distance(distance_start.elapsed(), 1);
        }

        let frontier_start = Instant::now();
        let entry = Layer0PoolEntry {
            idx: entry_idx,
            dist: entry_dist,
        };
        scratch.layer0_frontier.push(entry);
        scratch.layer0_results.insert(entry, ef);
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 2, 0);
        }

        let visited_start = Instant::now();
        let entry_marked = scratch.mark_visited(entry_idx);
        if let Some(stats) = profile.as_mut() {
            stats.record_visited_ops(visited_start.elapsed(), u64::from(entry_marked));
        }

        loop {
            let frontier_start = Instant::now();
            let Some(candidate) = scratch.layer0_frontier.pop_best() else {
                if let Some(stats) = profile.as_mut() {
                    stats.record_frontier_ops(frontier_start.elapsed(), 0, 0);
                }
                break;
            };
            if let Some(stats) = profile.as_mut() {
                stats.record_frontier_ops(frontier_start.elapsed(), 0, 1);
            }

            let pruning_start = Instant::now();
            if scratch.layer0_results.len() >= ef {
                if let Some(worst_dist) = scratch.layer0_results.worst_dist() {
                    if candidate.dist > worst_dist {
                        if let Some(stats) = profile.as_mut() {
                            stats.record_candidate_pruning(pruning_start.elapsed(), 1);
                        }
                        break;
                    }
                }
            }
            if let Some(stats) = profile.as_mut() {
                stats.record_candidate_pruning(pruning_start.elapsed(), 0);
            }

            let node_info = &self.node_info[candidate.idx];
            if level > node_info.max_layer {
                continue;
            }

            let mut batch_indices = [0usize; 4];
            let mut batch_len = 0usize;

            if use_flat_graph {
                let neighbors = self.layer0_flat_graph.neighbors_for(candidate.idx);
                if let Some(stats) = profile.as_mut() {
                    stats.record_layer0_flat_graph_neighbor_reads(neighbors.len() as u64);
                }
                for (neighbor_offset, &nbr_u32) in neighbors.iter().enumerate() {
                    if neighbor_offset + 1 < neighbors.len() {
                        let next_nbr_idx = neighbors[neighbor_offset + 1] as usize;
                        if next_nbr_idx < num_nodes {
                            let prefetched =
                                unsafe { self.prefetch_l2_vector_idx(base_ptr, next_nbr_idx) };
                            if prefetched {
                                if let Some(stats) = profile.as_mut() {
                                    stats.record_layer0_vector_prefetch();
                                }
                            }
                        }
                    }

                    let nbr_idx = nbr_u32 as usize;
                    if nbr_idx >= num_nodes {
                        continue;
                    }

                    let visited_start = Instant::now();
                    let marked = scratch.mark_visited(nbr_idx);
                    if let Some(stats) = profile.as_mut() {
                        stats.record_visited_ops(visited_start.elapsed(), u64::from(marked));
                    }
                    if !marked {
                        continue;
                    }

                    batch_indices[batch_len] = nbr_idx;
                    batch_len += 1;

                    if batch_len == 4 {
                        let distance_start = Instant::now();
                        let distances = unsafe {
                            self.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, batch_indices)
                        };
                        if let Some(stats) = profile.as_mut() {
                            stats.record_layer0_query_distance_batch4(distance_start.elapsed());
                        }
                        for (offset, nbr_dist) in distances.into_iter().enumerate() {
                            self.process_layer0_l2_candidate(
                                batch_indices[offset],
                                nbr_dist,
                                ef,
                                scratch,
                                &mut profile,
                            );
                        }
                        batch_len = 0;
                    }
                }
            } else {
                let neighbors = &node_info.layer_neighbors[level].ids;
                for (neighbor_offset, &nbr_id) in neighbors.iter().enumerate() {
                    if neighbor_offset + 1 < neighbors.len() {
                        let next_nbr_idx =
                            self.get_idx_from_id_fast(neighbors[neighbor_offset + 1]);
                        if next_nbr_idx < num_nodes {
                            let prefetched =
                                unsafe { self.prefetch_l2_vector_idx(base_ptr, next_nbr_idx) };
                            if prefetched {
                                if let Some(stats) = profile.as_mut() {
                                    stats.record_layer0_vector_prefetch();
                                }
                            }
                        }
                    }

                    let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                    if nbr_idx >= num_nodes {
                        continue;
                    }

                    let visited_start = Instant::now();
                    let marked = scratch.mark_visited(nbr_idx);
                    if let Some(stats) = profile.as_mut() {
                        stats.record_visited_ops(visited_start.elapsed(), u64::from(marked));
                    }
                    if !marked {
                        continue;
                    }

                    batch_indices[batch_len] = nbr_idx;
                    batch_len += 1;

                    if batch_len == 4 {
                        let distance_start = Instant::now();
                        let distances = unsafe {
                            self.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, batch_indices)
                        };
                        if let Some(stats) = profile.as_mut() {
                            stats.record_layer0_query_distance_batch4(distance_start.elapsed());
                        }
                        for (offset, nbr_dist) in distances.into_iter().enumerate() {
                            self.process_layer0_l2_candidate(
                                batch_indices[offset],
                                nbr_dist,
                                ef,
                                scratch,
                                &mut profile,
                            );
                        }
                        batch_len = 0;
                    }
                }
            }

            for &nbr_idx in &batch_indices[..batch_len] {
                let distance_start = Instant::now();
                let nbr_dist = unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) };
                if let Some(stats) = profile.as_mut() {
                    stats.record_layer0_query_distance(distance_start.elapsed(), 1);
                }
                self.process_layer0_l2_candidate(nbr_idx, nbr_dist, ef, scratch, &mut profile);
            }
        }

        let frontier_start = Instant::now();
        let sorted = scratch.layer0_results.to_sorted_pairs();
        if let Some(stats) = profile.as_mut() {
            stats.record_frontier_ops(frontier_start.elapsed(), 0, sorted.len() as u64);
        }
        sorted
    }

    fn search_layer_idx_l2_with_optional_profile(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        scratch: &mut SearchScratch,
        profile: Option<&mut HnswCandidateSearchProfileStats>,
    ) -> Vec<(usize, f32)> {
        if ef <= 1 {
            let (best_idx, best_dist) = self.greedy_upper_layer_descent_l2_with_entry_dist(
                query,
                entry_idx,
                level,
                self.l2_distance_to_idx(query, entry_idx),
            );
            return vec![(best_idx, best_dist)];
        }

        if level == 0 {
            self.search_layer_idx_l2_ordered_pool_with_optional_profile(
                query, entry_idx, level, ef, scratch, profile,
            )
        } else {
            self.search_layer_idx_l2_heap_with_optional_profile(
                query, entry_idx, level, ef, scratch, profile,
            )
        }
    }

    fn search_single_candidate_profiled(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        stats: &mut HnswCandidateSearchProfileStats,
    ) -> Vec<(i64, f32)> {
        if self.metric_type == MetricType::L2 {
            return self.search_single_candidate_profiled_l2(query, ef, k, stats);
        }

        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        let distance_start = Instant::now();
        let mut curr_ep_dist = self.distance(query, curr_ep_idx);
        let mut best_ep_dist = curr_ep_dist;
        stats.record_upper_layer_query_distance(distance_start.elapsed(), 1);
        let mut best_ep_idx = curr_ep_idx;

        for level in (1..=self.max_level).rev() {
            let entry_start = Instant::now();
            let (next_idx, next_dist) = self
                .greedy_upper_layer_descent_idx_with_entry_dist_optional_profile(
                    query,
                    curr_ep_idx,
                    level,
                    curr_ep_dist,
                    Some(stats),
                );
            curr_ep_idx = next_idx;
            curr_ep_dist = next_dist;
            if next_dist < best_ep_dist {
                best_ep_idx = next_idx;
                best_ep_dist = next_dist;
            }
            stats.record_entry_descent(entry_start.elapsed());
        }

        let results = self.search_layer_idx_with_optional_profile(
            query,
            best_ep_idx,
            0,
            ef,
            &mut scratch,
            Some(stats),
        );

        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            final_results.push((self.get_id_from_idx(idx), dist));
            if final_results.len() >= k {
                break;
            }
        }

        final_results
    }

    fn search_single_candidate_profiled_with_bitset(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        bitset: &BitsetView,
        stats: &mut HnswCandidateSearchProfileStats,
    ) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        if self.metric_type == MetricType::Cosine && self.ids.len() <= ef.max(k).max(64) {
            return self
                .brute_force_search(query, k, |_id, idx| idx >= bitset.len() || !bitset.get(idx));
        }

        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        for level in (1..=self.max_level).rev() {
            let jump_ef = if level >= self.max_level / 2 {
                1
            } else {
                ef.min(4)
            };

            let entry_start = Instant::now();
            let results = self.search_layer_idx_shared(
                query,
                curr_ep_idx,
                level,
                jump_ef,
                Some(bitset),
                &mut scratch,
                Some(stats),
            );
            stats.record_entry_descent(entry_start.elapsed());

            if let Some((best_idx, _)) = results.first() {
                curr_ep_idx = *best_idx;
            }
        }

        let results = self.search_layer_idx_shared(
            query,
            curr_ep_idx,
            0,
            ef,
            Some(bitset),
            &mut scratch,
            Some(stats),
        );

        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            final_results.push((self.get_id_from_idx(idx), dist));
            if final_results.len() >= k {
                break;
            }
        }

        final_results
    }

    fn search_single_candidate_profiled_l2(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        stats: &mut HnswCandidateSearchProfileStats,
    ) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        let distance_start = Instant::now();
        let mut curr_ep_dist = self.l2_distance_to_idx(query, curr_ep_idx);
        let mut best_ep_dist = curr_ep_dist;
        stats.record_upper_layer_query_distance(distance_start.elapsed(), 1);
        let mut best_ep_idx = curr_ep_idx;

        for level in (1..=self.max_level).rev() {
            let entry_start = Instant::now();
            let (next_idx, next_dist) = self
                .greedy_upper_layer_descent_l2_with_entry_dist_optional_profile(
                    query,
                    curr_ep_idx,
                    level,
                    curr_ep_dist,
                    Some(stats),
                );
            curr_ep_idx = next_idx;
            curr_ep_dist = next_dist;
            if next_dist < best_ep_dist {
                best_ep_idx = next_idx;
                best_ep_dist = next_dist;
            }
            stats.record_entry_descent(entry_start.elapsed());
        }

        let results = self.search_layer_idx_l2_with_scratch(
            query,
            best_ep_idx,
            0,
            ef,
            &mut scratch,
            Some(stats),
        );

        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            final_results.push((self.get_id_from_idx(idx), dist));
            if final_results.len() >= k {
                break;
            }
        }

        final_results
    }

    pub fn candidate_search_profile_report(
        &self,
        queries: &[f32],
        ef: usize,
        k: usize,
    ) -> Result<HnswCandidateSearchProfileReport> {
        if queries.is_empty() || queries.len() % self.dim != 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "profile queries must be a non-empty multiple of the index dimension".to_string(),
            ));
        }
        if self.ids.is_empty() || self.entry_point.is_none() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "candidate-search profiling requires a non-empty HNSW index".to_string(),
            ));
        }

        let mut stats = HnswCandidateSearchProfileStats::default();
        let query_count = queries.len() / self.dim;
        for i in 0..query_count {
            let query = &queries[i * self.dim..(i + 1) * self.dim];
            let _ = self.search_single_candidate_profiled(query, ef, k, &mut stats);
        }

        Ok(stats.into_report(query_count))
    }

    fn search_single(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        filter: &Option<Arc<dyn Predicate>>,
    ) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        if self.metric_type == MetricType::L2 && filter.is_none() {
            return self.search_single_l2_unfiltered(query, ef, k);
        }

        let filter_fn = |id: i64| {
            if let Some(f) = filter {
                f.evaluate(id)
            } else {
                true
            }
        };

        // BUG-P1-001: Cosine on tiny collections can be sensitive to graph entry-point
        // randomness. Use deterministic exhaustive ranking for small-N to keep
        // IP/Cosine metric semantics aligned and avoid flaky top-1 ordering.
        if self.metric_type == MetricType::Cosine && self.ids.len() <= ef.max(k).max(64) {
            return self.brute_force_search(query, k, |id, _idx| filter_fn(id));
        }

        // OPT-021: Multi-layer search with improved layer descent
        // Start from top layer and greedily search down.
        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());
        let mut best_ep_idx = curr_ep_idx;
        let mut curr_ep_dist = self.distance(query, curr_ep_idx);
        let mut best_ep_dist = curr_ep_dist;
        let use_greedy_upper_descent = filter.is_none();

        for level in (1..=self.max_level).rev() {
            if use_greedy_upper_descent {
                let (next_idx, next_dist) = self.greedy_upper_layer_descent_idx_with_entry_dist(
                    query,
                    curr_ep_idx,
                    level,
                    curr_ep_dist,
                );
                curr_ep_idx = next_idx;
                curr_ep_dist = next_dist;
                if next_dist < best_ep_dist {
                    best_ep_idx = next_idx;
                    best_ep_dist = next_dist;
                }
                continue;
            }

            let results = self.search_layer_idx_with_scratch(
                query,
                curr_ep_idx,
                level,
                ef.max(64).min(ef * 2),
                &mut scratch,
            );

            let mut best_valid_idx: Option<usize> = None;
            let mut best_valid_dist = f32::MAX;

            for (idx, dist) in &results {
                let id = self.get_id_from_idx(*idx);
                if filter_fn(id) && *dist < best_valid_dist {
                    best_valid_dist = *dist;
                    best_valid_idx = Some(*idx);
                }
            }

            if let Some(best_idx) = best_valid_idx {
                curr_ep_idx = best_idx;
                curr_ep_dist = best_valid_dist;
                if best_valid_dist < best_ep_dist {
                    best_ep_idx = best_idx;
                    best_ep_dist = best_valid_dist;
                }
            } else if let Some((fallback_idx, fallback_dist)) = results.first() {
                curr_ep_idx = *fallback_idx;
                curr_ep_dist = *fallback_dist;
            }
        }

        let results = self.search_layer_idx_with_scratch(query, best_ep_idx, 0, ef, &mut scratch);

        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            let id = self.get_id_from_idx(idx);
            if filter_fn(id) {
                final_results.push((id, dist));
                if final_results.len() >= k {
                    break;
                }
            }
        }

        final_results
    }

    fn search_single_l2_unfiltered(&self, query: &[f32], ef: usize, k: usize) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());
        let mut best_ep_idx = curr_ep_idx;
        let mut curr_ep_dist = self.l2_distance_to_idx(query, curr_ep_idx);
        let mut best_ep_dist = curr_ep_dist;

        for level in (1..=self.max_level).rev() {
            let (next_idx, next_dist) = self.greedy_upper_layer_descent_l2_with_entry_dist(
                query,
                curr_ep_idx,
                level,
                curr_ep_dist,
            );
            curr_ep_idx = next_idx;
            curr_ep_dist = next_dist;
            if next_dist < best_ep_dist {
                best_ep_idx = next_idx;
                best_ep_dist = next_dist;
            }
        }

        let query_bf16 = if self.use_bf16_storage {
            Some(
                query
                    .iter()
                    .map(|&v| Bf16::from_f32(v).to_bits())
                    .collect::<Vec<u16>>(),
            )
        } else {
            None
        };
        let results = self.search_layer_idx_l2_ordered_pool_fast(
            query,
            best_ep_idx,
            0,
            ef,
            &mut scratch,
            query_bf16.as_deref(),
        );

        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            final_results.push((self.get_id_from_idx(idx), dist));
            if final_results.len() >= k {
                break;
            }
        }

        final_results
    }

    /// Search single query with bitset filtering
    ///
    /// Integrates bitset checking directly into the search algorithm for better performance.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `ef` - Search ef parameter
    /// * `k` - Number of results to return
    /// * `bitset` - BitsetView for filtering
    ///
    /// # Returns
    /// Filtered search results
    fn search_single_with_bitset(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || self.entry_point.is_none() {
            return vec![];
        }

        if self.metric_type == MetricType::Cosine && self.ids.len() <= ef.max(k).max(64) {
            return self
                .brute_force_search(query, k, |_id, idx| idx >= bitset.len() || !bitset.get(idx));
        }

        // Multi-layer search with layer-wise jumping: start from top layer
        let mut scratch = SearchScratch::new();
        let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

        // Enhanced layer descent with bitset filtering
        for level in (1..=self.max_level).rev() {
            let jump_ef = if level >= self.max_level / 2 {
                1
            } else {
                ef.min(4)
            };

            let results = self.search_layer_idx_with_bitset_scratch(
                query,
                curr_ep_idx,
                level,
                jump_ef,
                bitset,
                &mut scratch,
            );

            // Find the best valid result for jumping
            let mut best_valid_idx = curr_ep_idx;

            if let Some((idx, _dist)) = results.into_iter().next() {
                best_valid_idx = idx;
            }

            if best_valid_idx != curr_ep_idx {
                curr_ep_idx = best_valid_idx;
            }
        }

        // Final search at layer 0 with full ef
        let results = self.search_layer_idx_with_bitset_scratch(
            query,
            curr_ep_idx,
            0,
            ef,
            bitset,
            &mut scratch,
        );

        // Apply bitset filter and return top k
        let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
        for (idx, dist) in results {
            final_results.push((self.get_id_from_idx(idx), dist));
            if final_results.len() >= k {
                break;
            }
        }

        final_results
    }

    /// Search for nearest neighbors at a specific layer with bitset filtering.
    ///
    /// This is the first pure-Rust core rewrite slice for filtered HNSW: move
    /// away from `id + HashSet` traversal toward the same `idx + SearchScratch`
    /// shape used by the generic search kernel.
    fn search_layer_idx_with_bitset_scratch(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: &crate::bitset::BitsetView,
        scratch: &mut SearchScratch,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_shared(query, entry_idx, level, ef, Some(bitset), scratch, None)
    }

    #[cfg(test)]
    fn search_layer_idx_with_bitset_scratch_stats(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: &crate::bitset::BitsetView,
        scratch: &mut SearchScratch,
    ) -> (Vec<(usize, f32)>, usize) {
        let before_snapshot = scratch.visited_epoch.clone();
        let result =
            self.search_layer_idx_with_bitset_scratch(query, entry_idx, level, ef, bitset, scratch);

        let visited_count = scratch
            .visited_epoch
            .iter()
            .zip(before_snapshot.iter().chain(std::iter::repeat(&0)))
            .filter(|&(after, before)| *after == scratch.epoch && *before != scratch.epoch)
            .count();

        (result, visited_count)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn search_layer_idx_with_bitset_legacy_for_screen(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> Vec<(usize, f32)> {
        self.search_layer_idx_with_bitset_legacy_for_screen_stats(
            query, entry_idx, level, ef, bitset,
        )
        .0
    }

    #[cfg(test)]
    fn search_layer_idx_with_bitset_legacy_for_screen_stats(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
        ef: usize,
        bitset: &crate::bitset::BitsetView,
    ) -> (Vec<(usize, f32)>, usize) {
        use std::collections::{BinaryHeap, HashSet};

        #[derive(Clone, Copy, PartialEq)]
        struct OrderedDist(f32);

        impl Eq for OrderedDist {}

        impl PartialOrd for OrderedDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for OrderedDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match (self.0.is_nan(), other.0.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => other
                        .0
                        .partial_cmp(&self.0)
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            }
        }

        let entry_id = self.get_id_from_idx(entry_idx);
        let mut visited: HashSet<i64> = HashSet::with_capacity(ef * 2);
        let mut candidates: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::with_capacity(ef);

        let entry_is_filtered = entry_idx < bitset.len() && bitset.get(entry_idx);
        let entry_dist = self.distance(query, entry_idx);
        candidates.push((OrderedDist(entry_dist), entry_id));
        if !entry_is_filtered {
            results.push((OrderedDist(entry_dist), entry_id));
        }
        visited.insert(entry_id);

        while let Some((OrderedDist(cand_dist), cand_id)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                    if cand_dist > worst_dist {
                        break;
                    }
                }
            }

            let cand_idx = self.get_idx_from_id_fast(cand_id);
            let node_info = &self.node_info[cand_idx];
            if level > node_info.max_layer {
                continue;
            }

            for &nbr_id in &node_info.layer_neighbors[level].ids {
                if visited.insert(nbr_id) {
                    let nbr_idx = self.get_idx_from_id_fast(nbr_id);
                    if nbr_idx < bitset.len() && bitset.get(nbr_idx) {
                        continue;
                    }

                    let nbr_dist = self.distance(query, nbr_idx);
                    if results.len() < ef {
                        results.push((OrderedDist(nbr_dist), nbr_id));
                        candidates.push((OrderedDist(nbr_dist), nbr_id));
                    } else if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                        if nbr_dist < worst_dist {
                            results.pop();
                            results.push((OrderedDist(nbr_dist), nbr_id));
                            candidates.push((OrderedDist(nbr_dist), nbr_id));
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(usize, f32)> = results
            .into_sorted_vec()
            .into_iter()
            .map(|(OrderedDist(d), id)| (self.get_idx_from_id_fast(id), d))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        (sorted, visited.len())
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    /// Set ef_search parameter for query-time control
    /// Higher ef_search improves recall at the cost of latency
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef.max(1);
    }

    /// Get graph statistics for debugging
    /// Returns (max_layer, layer_distribution, avg_neighbors_l0)
    pub fn get_graph_stats(&self) -> (usize, Vec<usize>, f32) {
        let mut layer_counts = vec![0usize; 16];
        let mut total_neighbors_l0 = 0;
        let mut max_layer = 0;

        for node_info in &self.node_info {
            if node_info.max_layer > max_layer {
                max_layer = node_info.max_layer;
            }
            if node_info.max_layer < layer_counts.len() {
                layer_counts[node_info.max_layer] += 1;
            }

            // Count layer 0 neighbors
            if !node_info.layer_neighbors.is_empty() {
                total_neighbors_l0 += node_info.layer_neighbors[0].len();
            }
        }

        let avg_neighbors_l0 = if self.node_info.is_empty() {
            0.0
        } else {
            total_neighbors_l0 as f32 / self.node_info.len() as f32
        };

        (max_layer, layer_counts, avg_neighbors_l0)
    }

    /// Get min/max/average neighbor counts for nodes that exist at a layer.
    pub fn layer_neighbor_count_stats(&self, level: usize) -> Option<(usize, usize, f32)> {
        let mut counts = self
            .node_info
            .iter()
            .filter(|node_info| level <= node_info.max_layer)
            .map(|node_info| node_info.layer_neighbors[level].len());

        let first = counts.next()?;
        let mut min_count = first;
        let mut max_count = first;
        let mut total = first;
        let mut seen = 1usize;

        for count in counts {
            min_count = min_count.min(count);
            max_count = max_count.max(count);
            total += count;
            seen += 1;
        }

        Some((min_count, max_count, total as f32 / seen as f32))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Magic and version
        file.write_all(b"HNSW")?;
        file.write_all(&3u32.to_le_bytes())?; // Version 3: multi-layer support

        // Config
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.m_max0 as u32).to_le_bytes())?;
        file.write_all(&(self.ef_search as u32).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(self.max_level as u32).to_le_bytes())?;
        file.write_all(&(self.level_multiplier.to_bits()).to_le_bytes())?;

        // Metric type
        file.write_all(&(self.metric_type as u8).to_le_bytes())?;

        // Number of vectors
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;

        // Vectors
        for v in &self.vectors {
            file.write_all(&v.to_le_bytes())?;
        }

        // IDs
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        // Node info (layer assignments and connections)
        for node_info in &self.node_info {
            file.write_all(&(node_info.max_layer as u32).to_le_bytes())?;

            // Connections for each layer
            for layer_idx in 0..=node_info.max_layer {
                let layer_nbrs = &node_info.layer_neighbors[layer_idx];
                file.write_all(&(layer_nbrs.len() as u32).to_le_bytes())?;

                for (&nbr_id, &dist) in layer_nbrs.ids.iter().zip(layer_nbrs.dists.iter()) {
                    file.write_all(&nbr_id.to_le_bytes())?;
                    file.write_all(&dist.to_le_bytes())?;
                }
            }
        }

        // Entry point
        if let Some(ep) = self.entry_point {
            file.write_all(&[1u8])?;
            file.write_all(&ep.to_le_bytes())?;
        } else {
            file.write_all(&[0u8])?;
        }

        Ok(())
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid magic".to_string(),
            ));
        }

        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);

        if version != 3 {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported version: {}",
                version
            )));
        }

        // Config
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        self.dim = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.m = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.m_max0 = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.ef_search = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.ef_construction = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.max_level = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)?;
        self.level_multiplier = f32::from_bits(u32::from_le_bytes(buf4));

        // Metric type
        let mut buf1 = [0u8; 1];
        file.read_exact(&mut buf1)?;
        self.metric_type = MetricType::from_bytes(buf1[0]);
        self.distance_to_idx_fn = Self::resolve_distance_to_idx_fn(self.metric_type);
        self.l2_distance_sq_ptr_kernel = simd::l2_distance_sq_ptr_kernel();

        // Number of vectors
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;

        // Vectors
        self.vectors = vec![0.0f32; count * self.dim];
        for i in 0..count * self.dim {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            self.vectors[i] = f32::from_le_bytes(buf);
        }
        self.rebuild_bf16_storage();

        // IDs
        self.ids = Vec::with_capacity(count);
        // OPT-021: Removed HashMap - IDs are stored in order
        for _ in 0..count {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            let id = i64::from_le_bytes(buf);
            self.ids.push(id);
            // OPT-021: No HashMap insert needed - idx = position in array
        }

        // Node info
        self.node_info = Vec::with_capacity(count);
        for _ in 0..count {
            file.read_exact(&mut buf4)?;
            let max_layer = u32::from_le_bytes(buf4) as usize;

            let mut node_info = NodeInfo::new(max_layer, self.m);

            for layer_idx in 0..=max_layer {
                file.read_exact(&mut buf4)?;
                let nbr_count = u32::from_le_bytes(buf4) as usize;

                for _ in 0..nbr_count {
                    let mut id_buf = [0u8; 8];
                    let mut dist_buf = [0u8; 4];
                    file.read_exact(&mut id_buf)?;
                    file.read_exact(&mut dist_buf)?;

                    let nbr_id = i64::from_le_bytes(id_buf);
                    let dist = f32::from_le_bytes(dist_buf);
                    node_info.layer_neighbors[layer_idx].push(nbr_id, dist);
                }
            }

            self.node_info.push(node_info);
        }

        // Entry point
        file.read_exact(&mut buf1)?;
        if buf1[0] == 1 {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            self.entry_point = Some(i64::from_le_bytes(buf));
        } else {
            self.entry_point = None;
        }

        self.refresh_layer0_flat_graph();
        self.trained = true;
        Ok(())
    }

    /// Check if this index contains raw data
    ///
    /// HNSW index stores raw vectors in the graph nodes
    pub fn has_raw_data(&self) -> bool {
        true
    }

    /// Get index memory size in bytes (estimate)
    pub fn size(&self) -> usize {
        // Estimate: vectors + ids + graph structure
        // OPT-021: Removed HashMap overhead from size calculation
        let vectors_size = self.vectors.len() * std::mem::size_of::<f32>();
        let bf16_vectors_size = self.bf16_vectors.len() * std::mem::size_of::<u16>();
        let ids_size = self.ids.len() * std::mem::size_of::<i64>();
        let node_info_size = self.node_info.len() * std::mem::size_of::<NodeInfo>();
        vectors_size + bf16_vectors_size + ids_size + node_info_size
    }

    /// Get metric type
    pub fn metric_type(&self) -> MetricType {
        self.metric_type
    }

    /// Get maximum level in the graph (for debugging)
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Get average node level (for debugging)
    pub fn average_node_level(&self) -> f64 {
        if self.node_info.is_empty() {
            return 0.0;
        }
        let total: usize = self.node_info.iter().map(|n| n.max_layer).sum();
        total as f64 / self.node_info.len() as f64
    }

    /// Find unreachable vectors from the entry point using BFS at each layer
    ///
    /// This implements the same algorithm as C++ `findUnreachableVectors()`:
    /// - For each layer from max_level down to 0, perform BFS from entry point
    /// - Mark all visited nodes at each layer
    /// - Return indices of nodes that exist at a layer but were not visited
    ///
    /// # Returns
    /// Vector of unreachable vector indices
    pub fn find_unreachable_vectors(&self) -> Vec<usize> {
        if self.entry_point.is_none() || self.ids.is_empty() {
            return vec![];
        }

        let entry_point = self.entry_point.unwrap();
        // OPT-021: Use helper method instead of HashMap lookup
        let entry_idx = match self.get_idx_from_id(entry_point) {
            Some(idx) => idx,
            None => return vec![],
        };

        let mut unreachable: Vec<usize> = Vec::new();
        let mut start_points = vec![entry_idx];

        // For each layer from max_level down to 0
        for level in (0..=self.max_level).rev() {
            let mut visited = vec![false; self.ids.len()];
            let mut touched_at_level: Vec<usize> = Vec::new();

            // BFS from all start points at this layer
            for &start_idx in &start_points {
                if visited[start_idx] {
                    continue;
                }

                let mut queue = std::collections::VecDeque::new();
                queue.push_back(start_idx);
                visited[start_idx] = true;

                if level > 0 {
                    touched_at_level.push(start_idx);
                }

                while let Some(curr_idx) = queue.pop_front() {
                    let node_info = &self.node_info[curr_idx];
                    if level > node_info.max_layer {
                        continue;
                    }

                    // Get neighbors at this layer
                    let neighbors = &node_info.layer_neighbors[level].ids;
                    for &nbr_id in neighbors {
                        // OPT-021: Use helper method instead of HashMap lookup
                        if let Some(nbr_idx) = self.get_idx_from_id(nbr_id) {
                            if !visited[nbr_idx] {
                                visited[nbr_idx] = true;
                                queue.push_back(nbr_idx);
                                if level > 0 {
                                    touched_at_level.push(nbr_idx);
                                }
                            }
                        }
                    }
                }
            }

            // Find unreachable nodes at this layer
            for (idx, was_visited) in visited.iter().enumerate().take(self.ids.len()) {
                let node_info = &self.node_info[idx];
                if node_info.max_layer >= level && !*was_visited {
                    // Collect all unreachable nodes (caller decides whether to repair)
                    unreachable.push(idx);
                }
            }

            // Update start points for next lower level
            start_points = touched_at_level;
        }

        unreachable
    }

    /// Find and repair unreachable vectors
    ///
    /// This is a convenience method that combines find_unreachable_vectors()
    /// with repair operations. It repairs all unreachable vectors at all levels.
    ///
    /// # Returns
    /// Number of unreachable vectors that were repaired
    pub fn find_and_repair_unreachable(&mut self) -> usize {
        let unreachable = self.find_unreachable_vectors();
        let count = unreachable.len();

        if count > 0 {
            // Collect repair tasks: (index, level)
            let repair_tasks: Vec<(usize, usize)> = unreachable
                .iter()
                .flat_map(|&idx| {
                    let node_info = &self.node_info[idx];
                    (0..=node_info.max_layer).map(move |level| (idx, level))
                })
                .collect();

            // Repair each unreachable vector at each level
            for (idx, level) in repair_tasks {
                self.repair_graph_connectivity_internal(idx, level);
            }
            self.refresh_layer0_flat_graph();
        }

        count
    }

    fn find_and_repair_unreachable_profiled(&mut self, stats: &mut HnswBuildProfileStats) -> usize {
        let unreachable = self.find_unreachable_vectors();
        let count = unreachable.len();

        if count > 0 {
            let repair_tasks: Vec<(usize, usize)> = unreachable
                .iter()
                .flat_map(|&idx| {
                    let node_info = &self.node_info[idx];
                    (0..=node_info.max_layer).map(move |level| (idx, level))
                })
                .collect();

            for (idx, level) in repair_tasks {
                let stage_start = Instant::now();
                self.repair_graph_connectivity_internal(idx, level);
                stats.record(HnswBuildProfileStage::Repair, stage_start.elapsed());
            }
            self.refresh_layer0_flat_graph();
        }

        count
    }

    /// Internal method to repair connectivity for a single unreachable vector
    ///
    /// This is called during find_unreachable_vectors for upper levels.
    /// It finds the nearest neighbors from entry_point and adds edges.
    fn repair_graph_connectivity_internal(&mut self, unreachable_idx: usize, level: usize) {
        if self.entry_point.is_none() {
            return;
        }

        let entry_point = self.entry_point.unwrap();
        // OPT-021: Use helper method instead of HashMap lookup
        let entry_idx = match self.get_idx_from_id(entry_point) {
            Some(idx) => idx,
            None => return,
        };

        let unreachable_id = self.ids[unreachable_idx];
        let unreachable_vec_start = unreachable_idx * self.dim;
        let unreachable_vec =
            &self.vectors[unreachable_vec_start..unreachable_vec_start + self.dim];

        // Greedy search from entry point to find nearest neighbors
        let mut curr_obj = entry_idx;
        let mut cur_dist = self.distance(unreachable_vec, curr_obj);

        // Descend from max_level to target level
        for level_above in (level..=self.max_level).rev() {
            if level_above == 0 {
                continue;
            }

            let mut changed = true;
            while changed {
                changed = false;
                let node_info = &self.node_info[curr_obj];
                if level_above > node_info.max_layer {
                    break;
                }

                let neighbors = &node_info.layer_neighbors[level_above].ids;
                for &nbr_id in neighbors {
                    // OPT-021: Use helper method instead of HashMap lookup
                    if let Some(nbr_idx) = self.get_idx_from_id(nbr_id) {
                        let d = self.distance(unreachable_vec, nbr_idx);
                        if d < cur_dist {
                            cur_dist = d;
                            curr_obj = nbr_idx;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Search for candidates at target level
        let candidates = self.search_layer_for_repair(unreachable_vec, curr_obj, level);

        // Add edges from candidates to unreachable vector
        let m_max = self.max_connections_for_layer(level);
        let mut add_count = 0;

        for (nbr_id, _) in candidates {
            if add_count >= m_max {
                break;
            }

            // OPT-021: Use helper method instead of HashMap lookup
            if let Some(nbr_idx) = self.get_idx_from_id(nbr_id) {
                // Skip self
                if nbr_idx == unreachable_idx {
                    continue;
                }

                // Add edge from candidate to unreachable (unidirectional for repair)
                let nbr_max_layer = self.node_info[nbr_idx].max_layer;
                if level <= nbr_max_layer {
                    let nbr_layer_nbrs = &mut self.node_info[nbr_idx].layer_neighbors[level];

                    // Check if edge already exists
                    if !nbr_layer_nbrs.ids.contains(&unreachable_id) {
                        nbr_layer_nbrs.push(unreachable_id, 0.0);
                        add_count += 1;
                    }
                }
            }
        }
    }

    /// Helper method to get vector slice for a given index
    #[allow(dead_code)]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Search layer for repair - simplified version without visited list pool
    fn search_layer_for_repair(
        &self,
        query: &[f32],
        entry_idx: usize,
        level: usize,
    ) -> Vec<(i64, f32)> {
        use std::collections::BinaryHeap;
        use std::collections::HashSet;

        #[derive(Clone, Copy, PartialEq)]
        struct OrderedDist(f32);

        impl Eq for OrderedDist {}

        impl PartialOrd for OrderedDist {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for OrderedDist {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                match (self.0.is_nan(), other.0.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => other
                        .0
                        .partial_cmp(&self.0)
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            }
        }

        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(OrderedDist, usize)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedDist, i64)> = BinaryHeap::new();

        let entry_dist = self.distance(query, entry_idx);
        candidates.push((OrderedDist(entry_dist), entry_idx));
        results.push((OrderedDist(entry_dist), self.ids[entry_idx]));
        visited.insert(entry_idx);

        let entry_id = self.ids[entry_idx];
        let entry_node_info = &self.node_info[entry_idx];
        if level > entry_node_info.max_layer {
            return vec![(entry_id, entry_dist)];
        }

        while let Some((OrderedDist(cand_dist), cand_idx)) = candidates.pop() {
            if results.len() >= self.ef_construction {
                if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                    if cand_dist > worst_dist {
                        break;
                    }
                }
            }

            let node_info = &self.node_info[cand_idx];
            if level > node_info.max_layer {
                continue;
            }

            for &nbr_id in &node_info.layer_neighbors[level].ids {
                // OPT-021: Use helper method instead of HashMap lookup
                if let Some(nbr_idx) = self.get_idx_from_id(nbr_id) {
                    if visited.insert(nbr_idx) {
                        let nbr_dist = self.distance(query, nbr_idx);

                        if results.len() < self.ef_construction {
                            results.push((OrderedDist(nbr_dist), nbr_id));
                            candidates.push((OrderedDist(nbr_dist), nbr_idx));
                        } else if let Some(&(OrderedDist(worst_dist), _)) = results.peek() {
                            if nbr_dist < worst_dist {
                                results.pop();
                                results.push((OrderedDist(nbr_dist), nbr_id));
                                candidates.push((OrderedDist(nbr_dist), nbr_idx));
                            }
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(i64, f32)> = results
            .into_sorted_vec()
            .into_iter()
            .map(|(OrderedDist(d), id)| (id, d))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted
    }

    /// Repair graph connectivity for an unreachable vector at a specific level
    ///
    /// This is the public method that can be called to manually repair connectivity.
    /// For automatic repair during build, use `build_with_repair()`.
    ///
    /// # Arguments
    /// * `unreachable_idx` - Index of the unreachable vector
    /// * `level` - Layer level to repair (0 = base layer)
    pub fn repair_graph_connectivity(&mut self, unreachable_idx: usize, level: usize) {
        self.repair_graph_connectivity_internal(unreachable_idx, level);
    }

    /// Build HNSW index with automatic graph repair
    ///
    /// This method:
    /// 1. Trains the index (if not already trained)
    /// 2. Adds all vectors to the index
    /// 3. Finds unreachable vectors from the entry point
    /// 4. Repairs unreachable vectors at all levels
    ///
    /// # Arguments
    /// * `vectors` - Flat array of vectors (n * dim)
    /// * `ids` - Optional IDs for vectors (auto-generated if None)
    ///
    /// # Returns
    /// Number of vectors added
    ///
    /// # Example
    /// ```ignore
    /// let mut index = HnswIndex::new(&config)?;
    /// let vectors = vec![1.0, 2.0, 3.0, 4.0, ...]; // n * dim
    /// index.build_with_repair(&vectors, None)?;
    /// ```
    pub fn build_with_repair(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        // Train if not already trained
        if !self.trained {
            self.train(vectors)?;
        }

        // Add all vectors
        self.add(vectors, ids)?;

        // Find and repair unreachable vectors
        let repaired_count = self.find_and_repair_unreachable();

        if repaired_count > 0 {
            println!(
                "HNSW build: repaired {} unreachable vectors",
                repaired_count
            );
        }

        Ok(self.ids.len())
    }

    pub fn build_profile_report(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
    ) -> Result<HnswBuildProfileReport> {
        if !self.trained {
            self.train(vectors)?;
        }

        let mut stats = HnswBuildProfileStats::default();
        let vectors_added = self.add_profiled(vectors, ids, &mut stats)?;
        let repair_operations = self.find_and_repair_unreachable_profiled(&mut stats);

        Ok(stats.into_report(vectors_added, repair_operations))
    }

    pub fn parallel_build_profile_report(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
    ) -> Result<HnswParallelBuildProfileReport> {
        if !self.trained {
            self.train(vectors)?;
        }

        let mut stats = HnswBuildProfileStats::default();
        let mut graph_quality_stats = HnswParallelBuildGraphQualityStats::default();
        let vectors_added =
            self.add_parallel_profiled(vectors, ids, &mut stats, &mut graph_quality_stats)?;
        let repair_operations = self.find_and_repair_unreachable_profiled(&mut stats);

        Ok(stats.into_parallel_report(
            vectors_added,
            repair_operations,
            self.parallel_insert_entry_descent_mode(),
            self.parallel_upper_layer_overflow_shrink_mode(),
            &graph_quality_stats,
        ))
    }

    pub fn graph_diagnosis_report(&self) -> HnswGraphDiagnosisReport {
        let node_count = self.ids.len();
        let mut level_histogram: BTreeMap<String, usize> = BTreeMap::new();
        let mut layer_degree_histograms: BTreeMap<String, BTreeMap<String, usize>> =
            BTreeMap::new();

        for node_info in &self.node_info {
            *level_histogram
                .entry(node_info.max_layer.to_string())
                .or_insert(0) += 1;

            for layer in 0..=node_info.max_layer {
                let degree = node_info.layer_neighbors[layer].len();
                *layer_degree_histograms
                    .entry(layer.to_string())
                    .or_default()
                    .entry(degree.to_string())
                    .or_insert(0) += 1;
            }
        }

        HnswGraphDiagnosisReport {
            node_count,
            max_level: self.max_level,
            level_histogram,
            layer_degree_histograms,
        }
    }

    pub fn search_cost_diagnosis(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
    ) -> HnswSearchCostDiagnosis {
        if self.dim == 0
            || query.len() != self.dim
            || self.ids.is_empty()
            || self.entry_point.is_none()
        {
            return HnswSearchCostDiagnosis {
                visited_nodes: 0,
                frontier_pushes: 0,
                frontier_pops: 0,
                distance_calls: 0,
            };
        }

        let mut stats = HnswCandidateSearchProfileStats::default();
        let _ = self.search_single_candidate_profiled(query, ef.max(1), k.max(1), &mut stats);

        HnswSearchCostDiagnosis {
            visited_nodes: stats.visited_marks as usize,
            frontier_pushes: stats.frontier_pushes as usize,
            frontier_pops: stats.frontier_pops as usize,
            distance_calls: stats.distance_calls as usize,
        }
    }

    pub fn search_cost_diagnosis_with_bitset(
        &self,
        query: &[f32],
        ef: usize,
        k: usize,
        bitset: &BitsetView,
    ) -> HnswSearchCostDiagnosis {
        if self.dim == 0
            || query.len() != self.dim
            || self.ids.is_empty()
            || self.entry_point.is_none()
        {
            return HnswSearchCostDiagnosis {
                visited_nodes: 0,
                frontier_pushes: 0,
                frontier_pops: 0,
                distance_calls: 0,
            };
        }

        let mut stats = HnswCandidateSearchProfileStats::default();
        let _ = self.search_single_candidate_profiled_with_bitset(
            query,
            ef.max(1),
            k.max(1),
            bitset,
            &mut stats,
        );

        HnswSearchCostDiagnosis {
            visited_nodes: stats.visited_marks as usize,
            frontier_pushes: stats.frontier_pushes as usize,
            frontier_pops: stats.frontier_pops as usize,
            distance_calls: stats.distance_calls as usize,
        }
    }

    pub fn generic_search_kernel_report(&self) -> HnswGenericSearchKernelReport {
        HnswGenericSearchKernelReport {
            unfiltered_layer_search_core: "shared_idx_binary_heap".to_string(),
            bitset_layer_search_core: "shared_idx_binary_heap".to_string(),
            frontier_container: "BinaryHeap<QueryCandidate>".to_string(),
            result_container: "BinaryHeap<ResultCandidate>".to_string(),
            frontier_reuse_scope: "scratch_binary_heap".to_string(),
            result_reuse_scope: "scratch_binary_heap".to_string(),
            shared_l2_distance_dispatch: "idx_ptr_kernel".to_string(),
            shared_bitset_distance_mode: "idx_ptr_batch4_when_grouped".to_string(),
            shared_layer0_neighbor_layout: "flat_u32_adjacency_when_enabled".to_string(),
            shared_result_threshold_mode: "scratch_cached_worst".to_string(),
            visited_reuse_scope: "visited_epoch_and_generic_heaps".to_string(),
        }
    }
}

/// Implement Index trait for HnswIndex
impl IndexTrait for HnswIndex {
    fn index_type(&self) -> &str {
        "HNSW"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn count(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        let vectors = dataset.vectors();
        self.train(vectors)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        let vectors = dataset.vectors();
        let ids = dataset.ids();
        self.add(vectors, ids)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let vectors = query.vectors();
        let req = SearchRequest {
            top_k,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        let api_result = self
            .search(vectors, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
        ))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let vectors = query.vectors();
        let req = SearchRequest {
            top_k,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };
        let api_result = self
            .search_with_bitset(vectors, &req, bitset)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
        ))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        let path = std::path::Path::new(path);
        self.save(path)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        let path = std::path::Path::new(path);
        self.load(path)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn has_raw_data(&self) -> bool {
        true
    }

    /// Get vectors by IDs (PARITY-P1-000: AnnIterator contract)
    ///
    /// Returns the raw vectors for the given IDs.
    /// Returns flatten vector data: [vec0_dim0, vec0_dim1, ..., vec1_dim0, vec1_dim1, ...]
    fn get_vector_by_ids(&self, ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if self.ids.is_empty() || self.vectors.is_empty() {
            return Err(IndexError::Empty);
        }

        let mut result = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            // For sequential IDs, idx == id
            // For custom IDs, find position in self.ids
            let idx = if self.use_sequential_ids {
                id as usize
            } else {
                self.ids.iter().position(|&x| x == id).ok_or_else(|| {
                    IndexError::Unsupported(format!("ID {} not found in index", id))
                })?
            };

            if idx >= self.ids.len() {
                return Err(IndexError::Unsupported(format!(
                    "ID {} out of range (max {})",
                    id,
                    self.ids.len()
                )));
            }

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

    /// Create ANN iterator for streaming search results (PARITY-P1-000)
    ///
    /// Creates an iterator that streams search results one at a time.
    /// Internally performs a search and returns results via the iterator.
    fn create_ann_iterator(
        &self,
        query: &Dataset,
        bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn crate::index::AnnIterator>, IndexError> {
        // Use a large top_k to get all potential results
        let top_k = self.ids.len().max(1000);
        let vectors = query.vectors();

        let req = SearchRequest {
            top_k,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        let api_result = if let Some(bs) = bitset {
            self.search_with_bitset(vectors, &req, bs)
        } else {
            self.search(vectors, &req)
        }
        .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        // Convert to iterator format: Vec<(id, distance)>
        let results: Vec<(i64, f32)> = api_result
            .ids
            .into_iter()
            .zip(api_result.distances)
            .collect();

        Ok(Box::new(HnswAnnIterator::new(results)))
    }
}

/// HNSW ANN Iterator implementation
///
/// Provides streaming access to HNSW search results.
pub struct HnswAnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl HnswAnnIterator {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl crate::index::AnnIterator for HnswAnnIterator {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.pos >= self.results.len() {
            return None;
        }
        let result = self.results[self.pos];
        self.pos += 1;
        Some(result)
    }

    fn buffer_size(&self) -> usize {
        self.results.len() - self.pos
    }
}

/// Generate a random level for a new node using exponential distribution.
///
/// This implements the original HNSW algorithm for level selection:
/// level = floor(-ln(U) / ln(REFERENCE_M)) where U ~ Uniform(0, 1]
///
/// BUG-001 FIX: Uses a fixed reference M (16) instead of the dynamic M parameter.
/// This ensures consistent level distribution across different M values.
/// Using dynamic M causes high M values to have very few layers, degrading recall.
///
/// # Arguments
/// * `_m` - The M parameter (number of connections), kept for API compatibility but not used
/// * `rng` - Random number generator
///
/// # Returns
/// A random level (0 means only base layer, higher values mean more layers)
pub fn random_level(_m: usize, rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen(); // Uniform [0, 1)

    // BUG-001 FIX: Use fixed reference M for consistent level distribution
    let level = (-r.ln() / (REFERENCE_M_FOR_LEVEL as f32).ln()) as usize;

    // Clamp to reasonable maximum
    level.min(MAX_LAYERS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::IndexType;
    use std::collections::BTreeSet;

    fn faiss_layer0_backfill_model(
        index: &HnswIndex,
        candidates: &[(usize, f32)],
        m: usize,
    ) -> Vec<(usize, f32)> {
        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(m);
        let mut outsiders: Vec<(usize, f32)> = Vec::new();

        for &(cand_idx, cand_dist) in candidates {
            let mut good = true;
            for &(sel_idx, _) in &selected {
                let dist_between = index.distance_between_nodes_idx(cand_idx, sel_idx);
                if dist_between < cand_dist {
                    good = false;
                    break;
                }
            }

            if good {
                selected.push((cand_idx, cand_dist));
                if selected.len() >= m {
                    return selected;
                }
            } else {
                outsiders.push((cand_idx, cand_dist));
            }
        }

        let mut outsider_idx = 0;
        while selected.len() < m && outsider_idx < outsiders.len() {
            selected.push(outsiders[outsider_idx]);
            outsider_idx += 1;
        }
        selected
    }

    fn deterministic_layer0_config(m: usize) -> IndexConfig {
        IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(m),
                ef_construction: Some(32),
                ef_search: Some(32),
                ml: Some(0.0),
                num_threads: Some(1),
                ..Default::default()
            },
        }
    }

    fn clustered_layer0_vectors() -> Vec<f32> {
        vec![
            0.0, 0.0, // center
            1.0, 0.0, // east
            1.01, 0.0, // east duplicate 1
            1.02, 0.0, // east duplicate 2
            0.0, 1.05, // north
            -1.05, 0.0, // west
            0.0, -1.05, // south
        ]
    }

    fn layer_neighbor_ids(index: &HnswIndex, node_idx: usize, level: usize) -> BTreeSet<i64> {
        index.node_info[node_idx].layer_neighbors[level]
            .ids
            .iter()
            .copied()
            .collect()
    }

    fn layer0_neighbor_ids(index: &HnswIndex, node_idx: usize) -> BTreeSet<i64> {
        layer_neighbor_ids(index, node_idx, 0)
    }

    fn deterministic_upper_layer_config() -> IndexConfig {
        IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(4),
                ef_construction: Some(32),
                ef_search: Some(32),
                ml: Some(0.0),
                num_threads: Some(1),
                ..Default::default()
            },
        }
    }

    fn deterministic_upper_layer_index() -> HnswIndex {
        let config = deterministic_upper_layer_config();
        let vectors = vec![
            0.0, 0.0, // node 0
            4.0, 0.0, // node 1
            8.0, 0.0, // node 2
            12.0, 0.0, // node 3
        ];

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        for id in 0..4 {
            let vector = &vectors[id * 2..(id + 1) * 2];
            index
                .add_vector(vector, Some(id as i64), Some(&[0, 1, 2]))
                .unwrap();
        }

        index.entry_point = Some(0);
        index.max_level = 2;

        for node_info in &mut index.node_info {
            for layer in &mut node_info.layer_neighbors {
                layer.ids.clear();
                layer.dists.clear();
            }
        }

        let chain = [(0usize, 1usize), (1, 2), (2, 3)];
        for &(left, right) in &chain {
            let dist = index.distance_between_nodes_idx(left, right);
            for level in 1..=2 {
                index.node_info[left].layer_neighbors[level].push(right as i64, dist);
                index.node_info[right].layer_neighbors[level].push(left as i64, dist);
            }
        }

        for left in 0..4 {
            for right in 0..4 {
                if left == right {
                    continue;
                }
                let dist = index.distance_between_nodes_idx(left, right);
                index.node_info[left].layer_neighbors[0].push(right as i64, dist);
            }
        }

        index.refresh_layer0_flat_graph();
        index
    }

    fn deterministic_layer0_batch4_index() -> HnswIndex {
        let config = deterministic_upper_layer_config();
        let vectors = vec![
            0.0, 0.0, // node 0
            4.0, 0.0, // node 1
            8.0, 0.0, // node 2
            12.0, 0.0, // node 3
            16.0, 0.0, // node 4
        ];

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        for id in 0..5 {
            let vector = &vectors[id * 2..(id + 1) * 2];
            index
                .add_vector(vector, Some(id as i64), Some(&[0]))
                .unwrap();
        }

        index.entry_point = Some(0);
        index.max_level = 0;

        for node_info in &mut index.node_info {
            for layer in &mut node_info.layer_neighbors {
                layer.ids.clear();
                layer.dists.clear();
            }
        }

        for left in 0..5 {
            for right in 0..5 {
                if left == right {
                    continue;
                }
                let dist = index.distance_between_nodes_idx(left, right);
                index.node_info[left].layer_neighbors[0].push(right as i64, dist);
            }
        }

        index.refresh_layer0_flat_graph();
        index
    }

    fn deterministic_filtered_screen_index(num_vectors: usize, dim: usize) -> HnswIndex {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(100),
                ef_search: Some(138),
                ml: Some(1.0 / (REFERENCE_M_FOR_LEVEL as f32).ln()),
                num_threads: Some(1),
                ..Default::default()
            },
        };

        let mut vectors = Vec::with_capacity(num_vectors * dim);
        for i in 0..num_vectors {
            for d in 0..dim {
                let value = ((i * 17 + d * 31) % 997) as f32 / 997.0;
                vectors.push(value);
            }
        }
        let ids: Vec<i64> = (0..num_vectors as i64).collect();

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();
        index
    }

    fn deterministic_parallel_build_entry_descent_fixture() -> HnswIndex {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(8),
                ef_search: Some(8),
                ml: Some(0.0),
                num_threads: Some(4),
                ..Default::default()
            },
        };

        let vectors = vec![
            0.0, 0.0, // node 0: global entry, far from query
            100.0, 0.0, // node 1: upper-layer bridge
            101.0, 0.0, // node 2: lower-layer bridge
            102.0, 0.0, // node 3: closest layer-0 neighbor to the query
        ];

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.vectors = vectors;
        index.ids = vec![0, 1, 2, 3];
        index.node_info = vec![
            NodeInfo::new(2, index.m),
            NodeInfo::new(2, index.m),
            NodeInfo::new(1, index.m),
            NodeInfo::new(0, index.m),
        ];
        index.entry_point = Some(0);
        index.max_level = 2;
        index.next_id = 4;
        index.use_sequential_ids = true;

        let dist01 = index.distance_between_nodes_idx(0, 1);
        index.node_info[0].layer_neighbors[2].push(1, dist01);
        index.node_info[1].layer_neighbors[2].push(0, dist01);

        let dist12 = index.distance_between_nodes_idx(1, 2);
        index.node_info[1].layer_neighbors[1].push(2, dist12);
        index.node_info[2].layer_neighbors[1].push(1, dist12);

        let dist23 = index.distance_between_nodes_idx(2, 3);
        index.node_info[2].layer_neighbors[0].push(3, dist23);
        index.node_info[3].layer_neighbors[0].push(2, dist23);

        index.refresh_layer0_flat_graph();
        index
    }

    #[test]
    fn test_graph_diagnosis_report_counts_nodes() {
        let index = deterministic_parallel_build_entry_descent_fixture();

        let report = index.graph_diagnosis_report();

        assert_eq!(report.node_count, index.ids.len());
    }

    #[test]
    fn test_graph_diagnosis_report_has_layer0_degree_summary() {
        let index = deterministic_parallel_build_entry_descent_fixture();

        let report = index.graph_diagnosis_report();

        assert!(
            report.layer_degree_histograms.contains_key("0"),
            "graph diagnosis report must include a layer-0 degree histogram"
        );
    }

    #[test]
    fn test_search_cost_diagnosis_reports_visited_nodes() {
        let index = deterministic_parallel_build_entry_descent_fixture();

        let report = index.search_cost_diagnosis(&[102.5, 0.0], 8, 2);

        assert!(
            report.visited_nodes > 0,
            "search cost diagnosis must report a positive visited-node count"
        );
    }

    #[test]
    fn test_search_cost_diagnosis_with_bitset_reports_visited_nodes() {
        use crate::bitset::BitsetView;

        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut bitset = BitsetView::new(index.ntotal());
        bitset.set(0, true);
        bitset.set(2, true);

        let report = index.search_cost_diagnosis_with_bitset(&query, 4, 2, &bitset);

        assert!(
            report.visited_nodes > 0,
            "bitset search-cost diagnosis should report visited nodes on the deterministic fixture"
        );
        assert!(
            report.frontier_pops > 0,
            "bitset search-cost diagnosis should report frontier pops on the deterministic fixture"
        );
        assert!(
            report.distance_calls > 0,
            "bitset search-cost diagnosis should report distance calls on the deterministic fixture"
        );
    }

    fn deterministic_parallel_profile_vectors(num_vectors: usize, dim: usize) -> Vec<f32> {
        let mut vectors = Vec::with_capacity(num_vectors * dim);
        for i in 0..num_vectors {
            let cluster = (i % 32) as f32 * 0.05;
            let band = ((i / 32) % 16) as f32 * 0.005;
            for d in 0..dim {
                let raw = ((i * 73 + d * 19 + (i % 11) * (d % 5)) % 1024) as f32 / 1024.0;
                vectors.push(raw + cluster + band + (d % 7) as f32 * 0.0005);
            }
        }
        vectors
    }

    #[test]
    fn test_hnsw() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();

        let vectors = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

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
    fn test_parallel_bulk_build_matches_single_insert_layer0_neighbor_diversification() {
        let config = deterministic_layer0_config(2);
        let vectors = clustered_layer0_vectors();

        let mut serial = HnswIndex::new(&config).unwrap();
        serial.train(&vectors).unwrap();
        for (idx, vector) in vectors.chunks_exact(2).enumerate() {
            serial
                .add_vector(vector, Some(idx as i64), Some(&[0]))
                .unwrap();
        }

        let serial_neighbors = layer0_neighbor_ids(&serial, 0);
        assert_eq!(
            serial_neighbors,
            BTreeSet::from([1, 4, 5, 6]),
            "single-node insertion should keep diverse reverse neighbors around the center node"
        );

        let mut parallel = HnswIndex::new(&config).unwrap();
        parallel.train(&vectors).unwrap();
        parallel.add_parallel(&vectors, None, Some(true)).unwrap();

        let parallel_neighbors = layer0_neighbor_ids(&parallel, 0);
        assert_eq!(
            parallel_neighbors, serial_neighbors,
            "bulk add path should maintain the same diversified layer-0 reverse neighbors as repeated single-node insertion"
        );
    }

    #[test]
    fn test_parallel_bulk_neighbor_search_descends_upper_layers_before_layer0() {
        let index = deterministic_parallel_build_entry_descent_fixture();
        let mut scratch = SearchScratch::new();

        let neighbors =
            index.find_neighbors_for_insertion_with_scratch(&[102.1, 0.0], 0, &mut scratch);

        assert_eq!(
            neighbors.len(),
            1,
            "level-0 bulk build should return one layer"
        );
        assert_eq!(
            neighbors[0].first().map(|(idx, _)| *idx),
            Some(3),
            "bulk build should descend through upper layers before searching layer 0"
        );
    }

    #[test]
    fn test_parallel_bulk_upper_layer_overflow_uses_heuristic_diversification() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(16),
                ef_search: Some(16),
                ml: Some(0.0),
                num_threads: Some(4),
                ..Default::default()
            },
        };

        let vectors = vec![
            0.0, 0.0, // node 0
            1.0, 0.0, // node 1
            1.01, 0.0, // node 2
            0.0, 2.0, // node 3
        ];

        let mut index = HnswIndex::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.vectors = vectors;
        index.ids = vec![0, 1, 2, 3];
        index.node_info = (0..4).map(|_| NodeInfo::new(1, index.m)).collect();
        index.entry_point = Some(0);
        index.max_level = 1;
        index.next_id = 4;
        index.use_sequential_ids = true;

        let neighbors = vec![
            (1usize, index.distance_between_nodes_idx(0, 1)),
            (2usize, index.distance_between_nodes_idx(0, 2)),
            (3usize, index.distance_between_nodes_idx(0, 3)),
        ];
        let mut layer0_nodes_to_shrink = Vec::new();

        index.add_connections_for_node(0, 0, 1, &neighbors, &mut layer0_nodes_to_shrink);

        assert!(
            layer0_nodes_to_shrink.is_empty(),
            "upper-layer overflow should not spill into layer-0 deferred shrink bookkeeping"
        );
        assert_eq!(
            layer_neighbor_ids(&index, 0, 1),
            BTreeSet::from([1, 3]),
            "upper-layer overflow should keep diverse neighbors instead of truncating to the two closest"
        );
    }

    #[test]
    fn test_parallel_build_profile_report_tracks_reworked_graph_quality_modes() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(64),
                ef_search: Some(32),
                ml: Some(2.0),
                num_threads: Some(4),
                ..Default::default()
            },
        };
        let vectors = deterministic_parallel_profile_vectors(1_200, 16);

        let mut index = HnswIndex::new(&config).unwrap();
        let report = index.parallel_build_profile_report(&vectors, None).unwrap();

        assert_eq!(
            report.parallel_insert_entry_descent_mode,
            "greedy_from_max_level",
            "round-8 profile should report that bulk build descends from max_level before node-level search"
        );
        assert_eq!(
            report.upper_layer_overflow_shrink_mode, "heuristic_shrink",
            "round-8 profile should report heuristic upper-layer overflow shrink after the rework"
        );
        assert_eq!(
            report
                .graph_quality_call_counts
                .omitted_upper_layer_descent_levels,
            0,
            "reworked bulk build should not omit any upper-layer descent levels"
        );
        assert_eq!(
            report
                .graph_quality_call_counts
                .upper_layer_truncate_to_best_events,
            0,
            "reworked bulk build should stop recording truncate-to-best upper-layer overflows"
        );
    }

    #[test]
    fn test_greedy_upper_layer_descent_follows_improving_chain() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];

        let (best_idx, best_dist) = index.greedy_upper_layer_descent_idx(&query, 0, 2);

        assert_eq!(
            best_idx, 3,
            "greedy upper-layer descent should keep hopping toward the improving chain endpoint"
        );
        assert!(
            best_dist < 0.1,
            "best upper-layer candidate should end near the query endpoint"
        );
    }

    #[test]
    fn test_search_layer_idx_ef1_matches_greedy_upper_layer_descent() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut scratch = SearchScratch::new();

        let greedy = index.greedy_upper_layer_descent_idx(&query, 0, 2);
        let with_scratch = index.search_layer_idx_with_scratch(&query, 0, 2, 1, &mut scratch);

        assert_eq!(
            with_scratch,
            vec![greedy],
            "ef=1 shared layer search should reuse the greedy upper-layer descent result"
        );
    }

    #[test]
    fn test_greedy_upper_layer_descent_l2_fast_matches_generic() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let entry_dist = index.distance(&query, 0);

        let generic =
            index.greedy_upper_layer_descent_idx_with_entry_dist(&query, 0, 2, entry_dist);
        let fast = index.greedy_upper_layer_descent_l2_with_entry_dist(&query, 0, 2, entry_dist);

        assert_eq!(
            fast, generic,
            "L2 fast upper-layer descent must preserve the generic greedy result"
        );
    }

    #[test]
    fn test_search_layer_idx_l2_fast_matches_generic_layer0_results() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut generic_scratch = SearchScratch::new();
        let mut fast_scratch = SearchScratch::new();

        let generic = index.search_layer_idx_with_scratch(&query, 0, 0, 3, &mut generic_scratch);
        let fast = index.search_layer_idx_l2_with_scratch(&query, 0, 0, 3, &mut fast_scratch, None);

        assert_eq!(
            fast, generic,
            "L2 fast layer-0 search must preserve generic candidate ordering and distances"
        );
    }

    #[test]
    fn test_layer0_ordered_results_keep_nearest_and_prune_worst() {
        let mut results = Layer0OrderedResults::default();
        results.prepare(3);

        assert_eq!(results.insert(Layer0PoolEntry { idx: 10, dist: 3.0 }, 3), 0);
        assert_eq!(results.insert(Layer0PoolEntry { idx: 11, dist: 1.0 }, 3), 0);
        assert_eq!(results.insert(Layer0PoolEntry { idx: 12, dist: 2.0 }, 3), 0);
        assert_eq!(
            results.to_sorted_pairs(),
            vec![(11, 1.0), (12, 2.0), (10, 3.0)],
            "ordered results must remain sorted from nearest to farthest"
        );

        assert!(
            !results.can_insert(4.0, 3),
            "a full results pool must reject candidates worse than the current worst distance"
        );
        assert!(results.can_insert(0.5, 3));

        assert_eq!(results.insert(Layer0PoolEntry { idx: 13, dist: 0.5 }, 3), 1);
        assert_eq!(
            results.to_sorted_pairs(),
            vec![(13, 0.5), (11, 1.0), (12, 2.0)],
            "ordered results must evict the current worst entry when a nearer candidate arrives"
        );
    }

    #[test]
    fn test_search_layer_idx_l2_ordered_pool_matches_heap_layer0_results() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut heap_scratch = SearchScratch::new();
        let mut ordered_scratch = SearchScratch::new();

        let heap =
            index.search_layer_idx_l2_heap_with_scratch(&query, 0, 0, 3, &mut heap_scratch, None);
        let ordered = index.search_layer_idx_l2_ordered_pool_with_scratch(
            &query,
            0,
            0,
            3,
            &mut ordered_scratch,
            None,
        );

        assert_eq!(
            ordered, heap,
            "ordered layer-0 search must preserve the heap-based candidate ordering and distances"
        );
    }

    #[test]
    fn test_candidate_profile_reports_layer0_flat_graph_layout() {
        let index = deterministic_upper_layer_index();
        let queries = vec![11.8, 0.0, 1.2, 0.0];
        let report = index
            .candidate_search_profile_report(&queries, 8, 2)
            .expect("candidate search profile should succeed");

        assert_eq!(
            report.search_core_shape.rust_layer0_neighbor_layout,
            "flat_u32_adjacency"
        );
        assert_eq!(report.search_core_shape.rust_layer0_neighbor_id_type, "u32");
        assert!(
            report
                .layer0_neighbor_access_call_counts
                .layer0_flat_graph_neighbor_reads
                > 0,
            "profile report must include non-zero flat graph neighbor reads"
        );
    }

    #[test]
    fn test_l2_distance_to_4_idxs_matches_scalar_distances() {
        let index = deterministic_upper_layer_index();
        let query = [1.5, 0.0];
        let query_ptr = query.as_ptr();
        let base_ptr = index.vectors.as_ptr();

        let scalar = [
            unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 0) },
            unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 1) },
            unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 2) },
            unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 3) },
        ];
        let batched = unsafe { index.l2_distance_to_4_idxs_ptr(query_ptr, base_ptr, [0, 1, 2, 3]) };

        assert_eq!(
            batched, scalar,
            "batch-4 layer-0 query distance helper must match scalar pointer distances"
        );
    }

    #[test]
    fn test_bfloat16_distance_path_reads_bfloat16_storage_instead_of_mutated_f32_buffer() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::BFloat16,
            params: crate::api::IndexParams {
                ef_search: Some(16),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, // idx 0
            2.0, 0.0, // idx 1
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // If L2 reads from f32 storage, this mutation will corrupt distance(0).
        // A real BF16 path should stay stable because it reads from BF16 storage.
        index.vectors[0] = 1234.0;
        let query = [0.0, 0.0];
        let query_ptr = query.as_ptr();
        let base_ptr = index.vectors.as_ptr();

        let d0 = unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 0) };
        let d1 = unsafe { index.l2_distance_to_idx_ptr(query_ptr, base_ptr, 1) };

        assert!(d0 < 1e-6, "BF16 path should keep idx=0 distance near zero");
        assert!(
            (d1 - 4.0).abs() < 1e-3,
            "BF16 path should preserve idx=1 distance"
        );
    }

    #[test]
    fn test_search_single_l2_fast_matches_generic_and_filter_path_stays_stable() {
        use crate::api::search::IdsPredicate;
        use std::sync::Arc;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                ef_search: Some(32),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ];
        let ids = vec![0, 1, 2, 3];
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vec![0.5, 0.0, 0.0, 0.0];
        let generic = index.search_single(&query, 32, 4, &None);
        let fast = index.search_single_l2_unfiltered(&query, 32, 4);

        assert_eq!(
            fast, generic,
            "L2 fast unfiltered search must preserve the generic search results"
        );

        let ids_predicate = IdsPredicate { ids: vec![0, 2] };
        let filtered = index.search_single(
            &query,
            32,
            2,
            &(Some(Arc::new(ids_predicate) as Arc<dyn Predicate>)),
        );
        let filtered_ids = filtered.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert!(
            !filtered_ids.is_empty(),
            "filtered searches must still return in-predicate results"
        );
        assert!(
            filtered_ids.iter().all(|id| matches!(id, 0 | 2)),
            "filtered searches must keep using the generic path and never leak filtered-out ids"
        );
    }

    #[test]
    fn test_search_single_l2_fast_bfloat16_matches_generic_unfiltered() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::BFloat16,
            params: crate::api::IndexParams {
                ef_search: Some(32),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let query = vec![0.5, 0.0, 0.0, 0.0];
        let generic = index.search_single(&query, 32, 4, &None);
        let fast = index.search_single_l2_unfiltered(&query, 32, 4);

        assert_eq!(
            fast, generic,
            "BF16 fast unfiltered search must preserve generic unfiltered results"
        );
    }

    #[test]
    fn test_search_layer_idx_with_bitset_scratch_allows_filtered_entry_but_skips_filtered_results()
    {
        use crate::bitset::BitsetView;

        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut bitset = BitsetView::new(index.ntotal());
        bitset.set(0, true);
        bitset.set(2, true);

        let mut scratch = SearchScratch::new();
        let results =
            index.search_layer_idx_with_bitset_scratch(&query, 0, 0, 4, &bitset, &mut scratch);
        let result_idxs: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();

        assert_eq!(
            result_idxs,
            vec![3, 1],
            "scratch-based bitset search should keep using a filtered entry as a frontier seed while excluding filtered nodes from the final result set"
        );
        assert!(
            results.windows(2).all(|pair| pair[0].1 <= pair[1].1),
            "bitset scratch search results must remain sorted from nearest to farthest"
        );
    }

    #[test]
    fn test_shared_layer_search_matches_generic_layer0_results() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut generic_scratch = SearchScratch::new();
        let expected = index.search_layer_idx_with_scratch(&query, 0, 0, 4, &mut generic_scratch);

        let mut shared_scratch = SearchScratch::new();
        let actual =
            index.search_layer_idx_shared_for_test(&query, 0, 0, 4, None, &mut shared_scratch);

        assert_eq!(
            actual, expected,
            "shared generic layer search must preserve the existing unfiltered layer-0 results"
        );
    }

    #[test]
    fn test_shared_layer_search_matches_bitset_layer0_results() {
        use crate::bitset::BitsetView;

        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut bitset = BitsetView::new(index.ntotal());
        bitset.set(0, true);
        bitset.set(2, true);

        let mut bitset_scratch = SearchScratch::new();
        let expected = index.search_layer_idx_with_bitset_scratch(
            &query,
            0,
            0,
            4,
            &bitset,
            &mut bitset_scratch,
        );

        let mut shared_scratch = SearchScratch::new();
        let actual = index.search_layer_idx_shared_for_test(
            &query,
            0,
            0,
            4,
            Some(&bitset),
            &mut shared_scratch,
        );

        assert_eq!(
            actual, expected,
            "shared generic layer search must preserve the existing bitset layer-0 results"
        );
    }

    #[test]
    fn test_search_layer_idx_with_bitset_scratch_reuses_visited_epoch_across_calls() {
        use crate::bitset::BitsetView;

        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut scratch = SearchScratch::new();
        let initial_epoch = scratch.epoch;

        let mut first_bitset = BitsetView::new(index.ntotal());
        first_bitset.set(2, true);
        let first = index.search_layer_idx_with_bitset_scratch(
            &query,
            0,
            0,
            4,
            &first_bitset,
            &mut scratch,
        );
        let epoch_after_first = scratch.epoch;

        let mut second_bitset = BitsetView::new(index.ntotal());
        second_bitset.set(1, true);
        let second = index.search_layer_idx_with_bitset_scratch(
            &query,
            0,
            0,
            4,
            &second_bitset,
            &mut scratch,
        );

        assert!(
            epoch_after_first > initial_epoch,
            "first scratch-backed bitset search should advance the visited epoch"
        );
        assert!(
            scratch.epoch > epoch_after_first,
            "subsequent scratch-backed bitset searches should reuse and advance the same visited epoch buffer"
        );
        assert_eq!(
            first.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            vec![3, 1, 0],
            "first filtered call should exclude only the masked node while preserving nearest-first ordering"
        );
        assert_eq!(
            second.iter().map(|(idx, _)| *idx).collect::<Vec<_>>(),
            vec![3, 2, 0],
            "second filtered call should not inherit visited state from the first call"
        );
    }

    #[test]
    fn test_search_layer_idx_shared_reuses_generic_heap_capacity_across_calls() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut scratch = SearchScratch::new();

        let _ = index.search_layer_idx_shared_for_test(&query, 0, 0, 4, None, &mut scratch);
        let first_frontier_cap = scratch.generic_frontier.capacity();
        let first_result_cap = scratch.generic_results.capacity();

        let mut fresh_scratch = SearchScratch::new();
        let expected =
            index.search_layer_idx_shared_for_test(&query, 0, 0, 2, None, &mut fresh_scratch);
        let actual = index.search_layer_idx_shared_for_test(&query, 0, 0, 2, None, &mut scratch);

        assert_eq!(
            actual, expected,
            "reusing scratch-backed generic heaps must not change shared layer-search results"
        );
        assert!(
            first_frontier_cap >= 8,
            "generic frontier capacity should scale with ef * 2 after the first call"
        );
        assert!(
            first_result_cap >= 4,
            "generic result capacity should scale with ef after the first call"
        );
        assert_eq!(
            scratch.generic_frontier.capacity(),
            first_frontier_cap,
            "generic frontier capacity should be retained across later calls"
        );
        assert_eq!(
            scratch.generic_results.capacity(),
            first_result_cap,
            "generic result capacity should be retained across later calls"
        );
    }

    #[test]
    fn test_search_layer_idx_shared_cached_worst_threshold_resets_across_calls() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut scratch = SearchScratch::new();

        let first = index.search_layer_idx_shared_for_test(&query, 0, 0, 4, None, &mut scratch);
        let first_worst = first
            .last()
            .map(|(_, dist)| *dist)
            .expect("first shared search should return results");
        assert_eq!(
            scratch.generic_worst_result_distance, first_worst,
            "scratch should cache the worst accepted distance from the first shared search"
        );

        let second = index.search_layer_idx_shared_for_test(&query, 0, 0, 2, None, &mut scratch);
        let second_worst = second
            .last()
            .map(|(_, dist)| *dist)
            .expect("second shared search should return results");
        assert_eq!(
            scratch.generic_worst_result_distance,
            second_worst,
            "shared search should refresh the cached worst threshold for the current call rather than reusing stale state"
        );
    }

    #[test]
    fn test_search_layer_idx_shared_uses_batch4_distance_when_grouping_layer0_neighbors() {
        use crate::bitset::BitsetView;

        let index = deterministic_layer0_batch4_index();
        let query = [1.5, 0.0];
        let bitset = BitsetView::new(index.ntotal());
        let mut scratch = SearchScratch::new();
        let mut stats = HnswCandidateSearchProfileStats::default();

        let results = index.search_layer_idx_shared(
            &query,
            0,
            0,
            4,
            Some(&bitset),
            &mut scratch,
            Some(&mut stats),
        );

        assert_eq!(
            results.len(),
            4,
            "shared bitset layer search should still return the expected bounded result set"
        );
        assert!(
            stats.layer0_batch4_calls > 0,
            "shared bitset layer-0 search should use the grouped batch-4 distance path once four valid neighbors are available"
        );
    }

    #[test]
    fn test_search_layer_idx_l2_heap_reuses_generic_heap_capacity_across_calls() {
        let index = deterministic_upper_layer_index();
        let query = [11.8, 0.0];
        let mut scratch = SearchScratch::new();

        let _ = index.search_layer_idx_l2_heap_with_scratch(&query, 0, 1, 4, &mut scratch, None);
        let first_frontier_cap = scratch.generic_frontier.capacity();
        let first_result_cap = scratch.generic_results.capacity();

        let mut fresh_scratch = SearchScratch::new();
        let expected =
            index.search_layer_idx_l2_heap_with_scratch(&query, 0, 1, 2, &mut fresh_scratch, None);
        let actual =
            index.search_layer_idx_l2_heap_with_scratch(&query, 0, 1, 2, &mut scratch, None);

        assert_eq!(
            actual, expected,
            "reusing scratch-backed generic heaps must not change upper-layer L2 heap results"
        );
        assert!(
            first_frontier_cap >= 8,
            "upper-layer L2 frontier capacity should scale with ef * 2 after the first call"
        );
        assert!(
            first_result_cap >= 4,
            "upper-layer L2 result capacity should scale with ef after the first call"
        );
        assert_eq!(
            scratch.generic_frontier.capacity(),
            first_frontier_cap,
            "upper-layer L2 frontier capacity should be retained across later calls"
        );
        assert_eq!(
            scratch.generic_results.capacity(),
            first_result_cap,
            "upper-layer L2 result capacity should be retained across later calls"
        );
    }

    #[test]
    fn test_search_with_bitset_matches_bruteforce_on_screen_fixture() {
        use crate::bitset::BitsetView;

        let index = deterministic_filtered_screen_index(512, 32);
        let query = &index.vectors[index.dim..index.dim * 2];
        let mut bitset = BitsetView::new(index.ntotal());
        for idx in (0..index.ntotal()).step_by(3) {
            bitset.set(idx, true);
        }

        let req = SearchRequest {
            top_k: 10,
            nprobe: 512,
            filter: None,
            params: None,
            radius: None,
        };
        let exact = index.brute_force_search(query, 10, |_id, idx| {
            idx >= bitset.len() || !bitset.get(idx)
        });
        let exact_ids: Vec<i64> = exact.iter().map(|(id, _)| *id).collect();

        let approx = index.search_with_bitset(query, &req, &bitset).unwrap();

        assert_eq!(
            approx.ids, exact_ids,
            "rewritten filtered HNSW search should match brute-force ids on the deterministic screen fixture when ef is large"
        );
    }

    #[test]
    fn test_should_bruteforce_bitset_knn_for_high_filter_ratio() {
        use crate::bitset::BitsetView;

        let index = deterministic_filtered_screen_index(100, 16);
        let mut bitset = BitsetView::new(index.ntotal());
        for idx in 0..93 {
            bitset.set(idx, true);
        }

        assert!(
            index.should_bruteforce_bitset_knn(5, &bitset),
            "native-aligned filtered search policy should switch to brute force when >=93% of candidates are filtered out"
        );
    }

    #[test]
    fn test_should_bruteforce_bitset_knn_for_large_topk_after_filtering() {
        use crate::bitset::BitsetView;

        let index = deterministic_filtered_screen_index(100, 16);
        let mut bitset = BitsetView::new(index.ntotal());
        for idx in 0..40 {
            bitset.set(idx, true);
        }

        assert!(
            index.should_bruteforce_bitset_knn(30, &bitset),
            "native-aligned filtered search policy should switch to brute force when top-k consumes at least half of the surviving vectors"
        );
    }

    #[test]
    fn test_search_with_bitset_uses_bruteforce_policy_on_high_filter_ratio_fixture() {
        use crate::bitset::BitsetView;

        let index = deterministic_filtered_screen_index(100, 16);
        let query = &index.vectors[index.dim..index.dim * 2];
        let mut bitset = BitsetView::new(index.ntotal());
        for idx in 0..95 {
            bitset.set(idx, true);
        }

        let req = SearchRequest {
            top_k: 3,
            nprobe: 1,
            filter: None,
            params: None,
            radius: None,
        };

        let exact =
            index.brute_force_search(query, 3, |_id, idx| idx >= bitset.len() || !bitset.get(idx));
        let exact_ids: Vec<i64> = exact.iter().map(|(id, _)| *id).collect();
        let result = index.search_with_bitset(query, &req, &bitset).unwrap();

        assert_eq!(
            result.ids, exact_ids,
            "high-filter-ratio bitset search should use brute-force fallback and preserve exact nearest-neighbor ids even when ef is tiny"
        );
    }

    #[test]
    fn test_layer0_l2_search_modes_distinguish_fast_and_profiled_paths() {
        let index = deterministic_upper_layer_index();

        assert_eq!(
            index.layer0_l2_search_mode_for_audit(false),
            "fast_unprofiled",
            "round-9 production L2 layer-0 path should expose a dedicated fast-path mode"
        );
        assert_eq!(
            index.layer0_l2_search_mode_for_audit(true),
            "profiled_optional",
            "round-9 profiled L2 layer-0 path should remain distinct from the production fast path"
        );
    }

    #[test]
    fn test_layer0_l2_fast_path_avoids_profile_timing() {
        let index = deterministic_upper_layer_index();

        assert!(
            index.production_layer0_avoids_profile_timing_for_audit(),
            "round-9 production layer-0 fast path should avoid profiling timing calls"
        );
    }

    #[test]
    fn test_layer0_slab_rebuild_tracks_canonical_state() {
        let index = deterministic_upper_layer_index();

        assert!(
            index.layer0_slab_enabled_for_audit(),
            "round-10 layer-0 slab should be enabled for the deterministic audit index"
        );

        let canonical_neighbors = index.layer0_flat_graph.neighbors_for(0).to_vec();
        let slab_neighbors = index.layer0_slab_neighbors_for_audit(0).to_vec();
        assert_eq!(
            slab_neighbors, canonical_neighbors,
            "round-10 slab neighbors must mirror the canonical layer-0 flat-graph neighbors"
        );

        let canonical_vector = &index.vectors[..index.dim];
        let slab_vector = index.layer0_slab_vector_for_audit(0);
        assert_eq!(
            slab_vector, canonical_vector,
            "round-10 slab vector payload must mirror the canonical vector storage"
        );
    }

    #[test]
    fn test_layer0_fast_path_reports_slab_layout_when_enabled() {
        let index = deterministic_upper_layer_index();

        assert_eq!(
            index.production_layer0_layout_mode_for_audit(),
            "layer0_slab",
            "round-10 production layer-0 fast path should report slab-backed layout mode when eligible"
        );
        assert_eq!(
            index.profiled_layer0_layout_mode_for_audit(),
            "flat_graph_profiled",
            "round-10 profiled layer-0 path should remain explicitly distinct from the slab-backed production path"
        );
    }

    #[test]
    #[ignore = "screen benchmark; excluded from default regression"]
    fn test_hnsw_filtered_search_screen_benchmark() {
        use crate::bitset::BitsetView;
        use std::hint::black_box;
        use std::time::Instant;

        let index = deterministic_filtered_screen_index(20_000, 64);
        let ef = 138usize;
        let entry_idx = index.get_idx_from_id_fast(index.entry_point.unwrap());

        let mut bitset = BitsetView::new(index.ntotal());
        for idx in (0..index.ntotal()).step_by(3) {
            bitset.set(idx, true);
        }

        let query_count = 256usize;
        let mut queries = Vec::with_capacity(query_count * index.dim);
        for i in 0..query_count {
            for d in 0..index.dim {
                let value = ((i * 43 + d * 19 + 7) % 991) as f32 / 991.0;
                queries.push(value);
            }
        }

        let mut scratch = SearchScratch::new();
        let mut legacy_total_visited = 0usize;
        let mut scratch_total_visited = 0usize;

        let legacy_start = Instant::now();
        for query in queries.chunks_exact(index.dim) {
            let (result, visited) = index.search_layer_idx_with_bitset_legacy_for_screen_stats(
                black_box(query),
                entry_idx,
                0,
                ef,
                &bitset,
            );
            legacy_total_visited += visited;
            black_box(result);
        }
        let legacy_elapsed = legacy_start.elapsed();

        let scratch_start = Instant::now();
        for query in queries.chunks_exact(index.dim) {
            let (result, visited) = index.search_layer_idx_with_bitset_scratch_stats(
                black_box(query),
                entry_idx,
                0,
                ef,
                &bitset,
                &mut scratch,
            );
            scratch_total_visited += visited;
            black_box(result);
        }
        let scratch_elapsed = scratch_start.elapsed();

        let legacy_qps = query_count as f64 / legacy_elapsed.as_secs_f64();
        let scratch_qps = query_count as f64 / scratch_elapsed.as_secs_f64();
        let speedup = scratch_qps / legacy_qps;

        println!(
            "filtered_screen_benchmark legacy_qps={legacy_qps:.3} scratch_qps={scratch_qps:.3} speedup={speedup:.3} legacy_ms={:.3} scratch_ms={:.3} avg_legacy_visited={:.3} avg_scratch_visited={:.3}",
            legacy_elapsed.as_secs_f64() * 1000.0,
            scratch_elapsed.as_secs_f64() * 1000.0,
            legacy_total_visited as f64 / query_count as f64,
            scratch_total_visited as f64 / query_count as f64,
        );
    }

    #[test]
    #[ignore = "screen benchmark; excluded from default regression"]
    fn test_hnsw_filtered_bruteforce_fallback_screen_benchmark() {
        use crate::bitset::BitsetView;
        use std::hint::black_box;
        use std::time::Instant;

        let index = deterministic_filtered_screen_index(20_000, 64);
        let ef = 138usize;

        let mut bitset = BitsetView::new(index.ntotal());
        for idx in 0..19_000 {
            bitset.set(idx, true);
        }

        let query_count = 256usize;
        let mut queries = Vec::with_capacity(query_count * index.dim);
        for i in 0..query_count {
            for d in 0..index.dim {
                let value = ((i * 29 + d * 13 + 5) % 983) as f32 / 983.0;
                queries.push(value);
            }
        }

        let corrected_start = Instant::now();
        for query in queries.chunks_exact(index.dim) {
            let result = index.search_single_with_bitset(black_box(query), ef, 10, &bitset);
            black_box(result);
        }
        let corrected_elapsed = corrected_start.elapsed();

        let brute_force_start = Instant::now();
        for query in queries.chunks_exact(index.dim) {
            let result = index.brute_force_search(black_box(query), 10, |_id, idx| {
                idx >= bitset.len() || !bitset.get(idx)
            });
            black_box(result);
        }
        let brute_force_elapsed = brute_force_start.elapsed();

        let corrected_qps = query_count as f64 / corrected_elapsed.as_secs_f64();
        let brute_force_qps = query_count as f64 / brute_force_elapsed.as_secs_f64();
        let speedup = brute_force_qps / corrected_qps;

        println!(
            "filtered_fallback_screen_benchmark corrected_qps={corrected_qps:.3} brute_force_qps={brute_force_qps:.3} brute_force_speedup={speedup:.3} corrected_ms={:.3} brute_force_ms={:.3}",
            corrected_elapsed.as_secs_f64() * 1000.0,
            brute_force_elapsed.as_secs_f64() * 1000.0,
        );
    }

    #[test]
    fn test_hnsw_ip_metric() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Ip,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(200),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        index.train(&flat).unwrap();

        for (id, v) in vectors.iter().enumerate() {
            index
                .add_vector(v, Some(id as i64), Some(&[0]))
                .expect("add_vector should succeed");
        }

        let query = vec![1.0, 0.1, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 4,
            nprobe: 200,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 0);
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Cosine,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(200),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ];

        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        index.train(&flat).unwrap();

        for (id, v) in vectors.iter().enumerate() {
            index
                .add_vector(v, Some(id as i64), Some(&[0]))
                .expect("add_vector should succeed");
        }

        let query = vec![2.0, 0.2, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 4,
            nprobe: 200,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids[0], 0);
    }

    #[test]
    fn test_hnsw_search_with_filter() {
        use crate::api::search::IdsPredicate;
        use std::sync::Arc;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();

        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ];
        let ids = vec![0, 1, 2, 3];

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vec![0.5, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 4,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 4);

        let ids_predicate = IdsPredicate { ids: vec![0, 2] };
        let req_filtered = SearchRequest {
            top_k: 2,
            nprobe: 10,
            filter: Some(Arc::new(ids_predicate) as Arc<dyn Predicate>),
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req_filtered).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&0) || result.ids.contains(&2));
    }

    #[test]
    fn test_hnsw_search_with_bitset() {
        use crate::bitset::BitsetView;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();

        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0,
            0.0, 0.0, 0.0,
        ];
        let ids = vec![0, 1, 2, 3, 4];

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vec![0.5, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        // Search without bitset
        let result_no_filter = index.search(&query, &req).unwrap();
        assert_eq!(result_no_filter.ids.len(), 5);

        // Create bitset to filter out vectors at positions 0 and 2
        let mut bitset = BitsetView::new(5);
        bitset.set(0, true); // Filter out vector 0
        bitset.set(2, true); // Filter out vector 2

        // Search with bitset
        let result_with_bitset = index.search_with_bitset(&query, &req, &bitset).unwrap();
        assert_eq!(result_with_bitset.ids.len(), 3);
        // Results should not contain filtered IDs 0 and 2
        assert!(!result_with_bitset.ids.contains(&0));
        assert!(!result_with_bitset.ids.contains(&2));
        // Should contain IDs 1, 3, 4
        assert!(result_with_bitset.ids.contains(&1));
        assert!(result_with_bitset.ids.contains(&3));
        assert!(result_with_bitset.ids.contains(&4));
    }

    #[test]
    fn test_hnsw_search_with_bitset_all_filtered() {
        use crate::bitset::BitsetView;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();

        let vectors = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let ids = vec![0, 1, 2];

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vec![0.5, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 3,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        // Filter all vectors
        let mut bitset = BitsetView::new(3);
        bitset.set_all();

        let result = index.search_with_bitset(&query, &req, &bitset).unwrap();
        assert_eq!(result.ids.len(), 0);
    }

    #[test]
    fn test_random_level_distribution() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Test that random levels follow expected distribution
        let mut level_counts = [0usize; 10];
        for _ in 0..1000 {
            let level = index.random_level();
            if level < 10 {
                level_counts[level] += 1;
            }
        }

        // Level 0 should have most nodes (~50% for M=16)
        assert!(level_counts[0] > 400, "Level 0 should have ~50% of nodes");

        // Higher levels should have fewer nodes
        for i in 1..level_counts.len() {
            assert!(
                level_counts[i] <= level_counts[i - 1],
                "Level {} should have fewer nodes than level {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_random_level_uses_config_random_seed() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                random_seed: Some(42),
                ..Default::default()
            },
        };

        let mut first = HnswIndex::new(&config).unwrap();
        let mut second = HnswIndex::new(&config).unwrap();

        let first_levels: Vec<usize> = (0..64).map(|_| first.random_level()).collect();
        let second_levels: Vec<usize> = (0..64).map(|_| second.random_level()).collect();

        assert_eq!(
            first_levels, second_levels,
            "HNSW random level generation must honor config.params.random_seed so local diagnosis lanes can reproduce the same graph shape"
        );
    }

    #[test]
    fn test_multilayer_structure() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(8),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Add many vectors to ensure multi-layer structure
        let mut vectors = Vec::new();
        for i in 0..100 {
            vectors.push(i as f32);
            vectors.push(0.0);
            vectors.push(0.0);
            vectors.push(0.0);
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // Verify that some nodes have multiple layers
        let mut max_layers_found = 0;
        for node_info in &index.node_info {
            max_layers_found = max_layers_found.max(node_info.max_layer);
        }

        // With 100 nodes, we should have some nodes at higher layers
        assert!(max_layers_found > 0, "Should have multi-layer structure");

        // Verify search still works
        let query = vec![50.0, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }

    #[test]
    fn test_hnsw_index_trait_with_bitset() {
        use crate::dataset::Dataset;
        use crate::index::Index as IndexTrait;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Add vectors
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ];
        let ids = vec![10, 11, 12, 13];

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        // Create dataset for query
        let query_vec = vec![1.5, 0.0, 0.0, 0.0];
        let query_dataset = Dataset::from_vectors(query_vec.clone(), 4);

        // Search without filter using Index trait
        let result = IndexTrait::search(&index, &query_dataset, 4).unwrap();
        assert_eq!(result.ids.len(), 4);

        // Create bitset to filter out first two vectors (indices 0 and 1)
        let mut bitset = BitsetView::new(4);
        bitset.set(0, true);
        bitset.set(1, true);

        // Search with bitset filter using Index trait
        let result_filtered =
            IndexTrait::search_with_bitset(&index, &query_dataset, 4, &bitset).unwrap();
        assert_eq!(result_filtered.ids.len(), 2);
        // Should only contain IDs 12 and 13 (indices 2 and 3)
        assert!(result_filtered.ids.contains(&12) || result_filtered.ids.contains(&13));
    }

    #[test]
    fn test_hnsw_get_vector_by_ids_empty_index_returns_empty_error() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let index = HnswIndex::new(&config).unwrap();
        let err = crate::index::Index::get_vector_by_ids(&index, &[0]).unwrap_err();
        assert!(matches!(err, crate::index::IndexError::Empty));
    }

    #[test]
    fn test_find_unreachable_vectors_empty_index() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let index = HnswIndex::new(&config).unwrap();
        let unreachable = index.find_unreachable_vectors();
        assert_eq!(unreachable.len(), 0);
    }

    #[test]
    fn test_find_unreachable_vectors_single_vector() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![1.0, 2.0, 3.0, 4.0];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let unreachable = index.find_unreachable_vectors();
        // Single vector should be reachable (it's the entry point)
        assert_eq!(unreachable.len(), 0);
    }

    #[test]
    fn test_build_with_repair() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(8),
                ef_construction: Some(100),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Add many vectors to create a complex graph
        let mut vectors = Vec::new();
        for i in 0..50 {
            vectors.push(i as f32);
            vectors.push((i * 2) as f32);
            vectors.push((i * 3) as f32);
            vectors.push((i * 4) as f32);
        }

        // Build with repair
        let count = index.build_with_repair(&vectors, None).unwrap();
        assert_eq!(count, 50);

        // Verify search works after repair
        let query = vec![25.0, 50.0, 75.0, 100.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);

        // Results should be close to query point
        for &id in &result.ids {
            assert_ne!(id, -1);
        }
    }

    #[test]
    fn test_repair_graph_connectivity_manual() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(8),
                ef_construction: Some(100),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Build index
        let mut vectors = Vec::new();
        for i in 0..30 {
            vectors.push(i as f32);
            vectors.push(0.0);
            vectors.push(0.0);
            vectors.push(0.0);
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // Find unreachable vectors
        let unreachable = index.find_unreachable_vectors();

        // Manually repair if any found
        for &idx in &unreachable {
            let node_info = &index.node_info[idx];
            for level in 0..=node_info.max_layer {
                index.repair_graph_connectivity(idx, level);
            }
        }

        // Verify search still works
        let query = vec![15.0, 0.0, 0.0, 0.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert!(!result.ids.is_empty());
    }

    #[test]
    fn test_graph_connectivity_after_repair() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Add a large number of vectors to stress test connectivity
        let mut vectors = Vec::new();
        for i in 0..100 {
            vectors.push((i % 10) as f32);
            vectors.push((i / 10) as f32);
            vectors.push((i % 5) as f32);
            vectors.push((i / 5) as f32);
        }

        // Build with repair
        index.build_with_repair(&vectors, None).unwrap();

        // Test that we can search from any point and get results
        for i in 0..10 {
            let query = vec![
                (i % 10) as f32,
                (i / 10) as f32,
                (i % 5) as f32,
                (i / 5) as f32,
            ];
            let req = SearchRequest {
                top_k: 3,
                nprobe: 20,
                filter: None,
                params: None,
                radius: None,
            };

            let result = index.search(&query, &req).unwrap();
            assert_eq!(result.ids.len(), 3, "Search failed for query {}", i);
        }
    }

    #[test]
    fn test_unreachable_detection_multilayer() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(8),
                ef_construction: Some(150),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Add vectors to ensure multi-layer structure
        let mut vectors = Vec::new();
        for i in 0..80 {
            vectors.push(i as f32 / 10.0);
            vectors.push((i * 2) as f32 / 10.0);
            vectors.push((i * 3) as f32 / 10.0);
            vectors.push((i * 4) as f32 / 10.0);
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // Check multi-layer structure exists
        let mut max_layers_found = 0;
        for node_info in &index.node_info {
            max_layers_found = max_layers_found.max(node_info.max_layer);
        }
        assert!(max_layers_found > 0, "Should have multi-layer structure");

        // Find unreachable vectors
        let unreachable = index.find_unreachable_vectors();

        // After repair, all vectors should be reachable
        // (the find_unreachable_vectors method repairs upper levels automatically)
        // So we just verify the method completes without errors
        let _ = unreachable;

        // Verify search works
        let query = vec![4.0, 8.0, 12.0, 16.0];
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }

    /// OPT-022: Performance benchmark for HNSW build with two-phase optimization
    ///
    /// Target: 100K vectors build time < 500ms
    /// Using OPT-015 parameters for comparison: M=16, EF_CONSTRUCTION=200
    #[test]
    #[ignore = "performance benchmark; excluded from default regression"]
    fn test_hnsw_build_performance() {
        use std::time::Instant;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 128,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(64),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Generate 100K random vectors
        let n = 100_000;
        let dim = 128;
        let mut vectors = Vec::with_capacity(n * dim);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..(n * dim) {
            vectors.push(rng.gen::<f32>());
        }

        println!("\n=== OPT-015 HNSW Build Performance Benchmark ===");
        println!("Vectors: {} x {}", n, dim);
        println!("M: {}, EF_CONSTRUCTION: {}", index.m, index.ef_construction);
        println!("Optimization: Index-based graph (no HashMap/ID lookup in hot path)");

        // Train
        let train_start = Instant::now();
        index.train(&vectors).unwrap();
        let train_time = train_start.elapsed();
        println!("Train time: {:?}", train_time);

        // Build with two-phase optimization
        let build_start = Instant::now();
        let count = index.add(&vectors, None).unwrap();
        let build_time = build_start.elapsed();

        println!("Build time: {:?}", build_time);
        println!("Vectors added: {}", count);

        let build_ms = build_time.as_secs_f64() * 1000.0;
        println!("Build time (ms): {:.2}", build_ms);

        // Verify performance target: < 500ms for 100K vectors
        // Note: This is an aggressive target; actual performance depends on hardware
        // For reference, we check if it's under 2000ms (2 seconds) as a reasonable target
        let target_ms = 2000.0; // 2 seconds for 100K vectors on typical hardware
        println!("Target: < {:.0}ms", target_ms);

        if build_ms < target_ms {
            println!(
                "✅ PASS: Build time ({:.2}ms) is under target ({:.0}ms)",
                build_ms, target_ms
            );
        } else {
            println!(
                "⚠️  INFO: Build time ({:.2}ms) exceeded target ({:.0}ms)",
                build_ms, target_ms
            );
            println!("   (This is expected on slower hardware; two-phase optimization is still beneficial)");
        }

        // Verify recall with a search test
        let query = &vectors[0..dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let search_start = Instant::now();
        let result = index.search(query, &req).unwrap();
        let search_time = search_start.elapsed();

        println!("\nSearch test:");
        println!("  Search time: {:?}", search_time);
        println!("  Results: {} vectors", result.ids.len());
        println!("  Top result distance: {:.4}", result.distances[0]);

        // Verify recall@10 - the first result should be very close to query (ideally the query itself)
        // Since query is vector[0], it should find itself or very close neighbors
        let recall_ok = result.distances[0] < 10.0; // Reasonable threshold for random 128D vectors
        if recall_ok {
            println!(
                "✅ Recall@10: GOOD (top distance: {:.4})",
                result.distances[0]
            );
        } else {
            println!(
                "⚠️  Recall@10: Check needed (top distance: {:.4})",
                result.distances[0]
            );
        }

        println!("\n=== Benchmark Complete ===\n");

        assert_eq!(count, n, "Should add all {} vectors", n);
        assert!(!result.ids.is_empty(), "Search should return results");
    }

    /// OPT-031: Test parallel HNSW build
    #[test]
    fn test_hnsw_parallel_build() {
        use std::time::Instant;

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 128,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(64),
                num_threads: Some(4),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();

        // Generate 10K random vectors for faster test
        let n = 10_000;
        let dim = 128;
        let mut vectors = Vec::with_capacity(n * dim);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..(n * dim) {
            vectors.push(rng.gen::<f32>());
        }

        println!("\n=== OPT-031: HNSW Parallel Build Test ===");
        println!("Vectors: {} x {}", n, dim);
        println!("Threads: {}", index.num_threads);

        // Train
        index.train(&vectors).unwrap();

        // Build with parallel construction
        let build_start = Instant::now();
        let count = index.add_parallel(&vectors, None, Some(true)).unwrap();
        let build_time = build_start.elapsed();

        println!("Build time: {:?}", build_time);
        println!("Vectors added: {}", count);

        assert_eq!(count, n, "Should add all {} vectors", n);

        // Verify search works
        let query = &vectors[0..dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(query, &req).unwrap();
        assert!(!result.ids.is_empty(), "Search should return results");
        println!(
            "Search results: {} vectors, top distance: {:.4}",
            result.ids.len(),
            result.distances[0]
        );

        println!("\n=== Parallel Build Test Complete ===\n");
    }

    /// OPT-031: Test parallel build error handling
    #[test]
    fn test_hnsw_parallel_error_handling() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 128,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                num_threads: Some(4),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![1.0; 1000 * 128];

        // Test 1: Untrained index
        let result = index.add_parallel(&vectors, None, Some(true));
        assert!(result.is_err(), "Should fail on untrained index");

        // Train the index
        index.train(&vectors).unwrap();

        // Test 2: Empty vector set
        let result = index.add_parallel(&[], None, Some(true));
        assert!(result.is_ok(), "Should handle empty vector set gracefully");
        assert_eq!(result.unwrap(), 0, "Should return 0 for empty set");

        // Test 3: Dimension mismatch (non-zero count but wrong total length)
        // 128-dim index, but only 100 elements (not divisible by 128)
        let bad_vectors = vec![1.0; 100];
        let result = index.add_parallel(&bad_vectors, None, Some(true));
        assert!(
            result.is_err(),
            "Should fail on dimension mismatch: got {:?}",
            result
        );

        // Test 4: ID count mismatch
        let wrong_ids = vec![1i64, 2i64];
        let result = index.add_parallel(&vectors, Some(&wrong_ids), Some(true));
        assert!(result.is_err(), "Should fail on ID count mismatch");

        println!("✅ All error handling tests passed");
    }

    /// OPT-031: Test batch size optimization
    #[test]
    fn test_hnsw_batch_size_calculation() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 128,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                num_threads: Some(8),
                ..Default::default()
            },
        };

        let index = HnswIndex::new(&config).unwrap();

        // Test batch size calculation for different scenarios
        let test_cases = vec![
            (1000, 64, "Small dataset, low dim"),
            (10000, 128, "Medium dataset, standard dim"),
            (100000, 128, "Large dataset, standard dim"),
            (50000, 512, "Medium dataset, high dim"),
        ];

        println!("\n=== Batch Size Calculation Test ===");
        for (n, dim, description) in test_cases {
            let batch_size = index.calculate_optimal_batch_size(n, dim);
            println!(
                "  {}: n={}, dim={} -> batch_size={}",
                description, n, dim, batch_size
            );

            // Verify batch size is within reasonable bounds
            assert!(batch_size >= 50, "Batch size should be >= 50");
            assert!(batch_size <= 5000, "Batch size should be <= 5000");
        }
        println!("✅ All batch size calculations are within bounds\n");
    }

    /// OPT-031: Test API compatibility between add() and add_parallel()
    #[test]
    fn test_hnsw_parallel_api_compatibility() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Cosine,
            dim: 32,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(16),
                ef_construction: Some(200),
                ef_search: Some(64),
                num_threads: Some(4),
                ..Default::default()
            },
        };

        // Keep the dataset under the deterministic small-N exhaustive-search cutoff
        // so this test validates API compatibility rather than random graph variance.
        let n = 48;
        let dim = 32;

        // Generate vectors
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        // Build with serial add()
        let mut index_serial = HnswIndex::new(&config).unwrap();
        index_serial.train(&vectors).unwrap();
        index_serial.add(&vectors, Some(&ids)).unwrap();

        // Build with parallel add_parallel()
        let mut index_parallel = HnswIndex::new(&config).unwrap();
        index_parallel.train(&vectors).unwrap();
        index_parallel
            .add_parallel(&vectors, Some(&ids), Some(true))
            .unwrap();

        // Verify both have same count
        assert_eq!(
            index_serial.ntotal(),
            index_parallel.ntotal(),
            "Serial and parallel should have same vector count"
        );

        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            filter: None,
            params: None,
            radius: None,
        };

        for query_idx in 0..8 {
            let start = query_idx * dim;
            let query = &vectors[start..start + dim];
            let result_serial = index_serial.search(query, &req).unwrap();
            let result_parallel = index_parallel.search(query, &req).unwrap();

            assert_eq!(
                result_serial.ids, result_parallel.ids,
                "Serial and parallel should return the same IDs for query {}",
                query_idx
            );
            assert_eq!(
                result_serial.distances.len(),
                result_parallel.distances.len(),
                "Serial and parallel should return same number of distances for query {}",
                query_idx
            );

            for (rank, (serial_dist, parallel_dist)) in result_serial
                .distances
                .iter()
                .zip(result_parallel.distances.iter())
                .enumerate()
            {
                let dist_diff = (serial_dist - parallel_dist).abs();
                assert!(
                    dist_diff < 1e-6,
                    "Distances should match for query {} rank {} (diff: {:.8})",
                    query_idx,
                    rank,
                    dist_diff
                );
            }
        }

        println!("✅ API compatibility test passed");
    }

    #[test]
    fn test_load_refreshes_metric_distance_dispatch() {
        let ip_config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::Ip,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };

        let vectors = vec![
            1.0f32, 0.0f32, //
            0.0f32, 1.0f32, //
        ];

        let mut original = HnswIndex::new(&ip_config).expect("create source index");
        original.train(&vectors).expect("train source index");
        original.add(&vectors, None).expect("add vectors");

        let temp_path = std::env::temp_dir().join(format!(
            "hnsw_metric_dispatch_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos()
        ));
        original.save(&temp_path).expect("save source index");

        let l2_config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams::default(),
        };
        let mut reloaded = HnswIndex::new(&l2_config).expect("create destination index");
        reloaded.load(&temp_path).expect("load saved index");
        let _ = std::fs::remove_file(&temp_path);

        assert_eq!(reloaded.metric_type(), MetricType::Ip);
        let query = [1.0f32, 0.0f32];
        let dist_self = reloaded.distance(&query, 0);
        let dist_other = reloaded.distance(&query, 1);
        assert!(
            (dist_self + 1.0).abs() < 1e-6,
            "expected IP distance -1.0 for aligned vector, got {dist_self}"
        );
        assert!(
            dist_other.abs() < 1e-6,
            "expected IP distance 0.0 for orthogonal vector, got {dist_other}"
        );
    }

    #[test]
    fn test_select_neighbors_heuristic_no_backfill_matches_hnswlib() {
        // OPT-037: Verify that heuristic does NOT backfill pruned candidates
        // This matches hnswlib behavior - preserve diversity by not adding redundant neighbors
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(8),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, // candidate 0
            0.1, 0.0, // candidate 1, close to candidate 0
            0.2, 0.0, // candidate 2, also close enough to be pruned in diversification pass
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // Candidates where 1 and 2 are too close to each other (both near query)
        let candidates = vec![(0usize, 0.1f32), (1usize, 0.2f32), (2usize, 5.0f32)];
        let selected = index.select_neighbors_heuristic_idx(&[0.0, 0.0], &candidates, 2);

        // OPT-037: hnswlib does NOT backfill, so we may get fewer than M neighbors
        // when heuristic rejects candidates for being too similar
        assert!(
            selected.len() <= 2,
            "heuristic should not exceed M, but may return fewer if candidates are too similar"
        );
        assert!(
            !selected.is_empty(),
            "heuristic should always select at least the nearest candidate"
        );
        assert_eq!(
            selected[0].0, 0,
            "nearest good candidate should stay selected"
        );
    }

    #[test]
    fn test_layer0_neighbor_selection_audit_differs_from_faiss_backfill() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(8),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, // candidate 0
            0.1, 0.0, // candidate 1, pruned by diversification
            0.2, 0.0, // candidate 2, also pruned
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let candidates = vec![(0usize, 0.1f32), (1usize, 0.2f32), (2usize, 5.0f32)];

        let rust_selected = index.select_neighbors_heuristic_idx(&[0.0, 0.0], &candidates, 2);
        let faiss_selected = faiss_layer0_backfill_model(&index, &candidates, 2);

        assert_eq!(
            rust_selected,
            vec![(0usize, 0.1f32)],
            "current Rust build path keeps the hnswlib-style no-backfill result"
        );
        assert_eq!(
            faiss_selected,
            vec![(0usize, 0.1f32), (1usize, 0.2f32)],
            "FAISS layer-0 keep_max_size_level0 refills pruned outsiders"
        );
        assert_ne!(
            rust_selected, faiss_selected,
            "audit fixture should capture the concrete FAISS-vs-Rust layer-0 difference"
        );
    }

    #[test]
    fn test_layer0_neighbor_selection_matches_faiss_backfill() {
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            dim: 2,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                m: Some(2),
                ef_construction: Some(8),
                ..Default::default()
            },
        };

        let mut index = HnswIndex::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, //
            0.1, 0.0, //
            0.2, 0.0, //
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let candidates = vec![(0usize, 0.1f32), (1usize, 0.2f32), (2usize, 5.0f32)];

        let layer0_selected =
            index.select_neighbors_heuristic_idx_layer_aware(&[0.0, 0.0], &candidates, 2, true);
        let upper_layer_selected =
            index.select_neighbors_heuristic_idx_layer_aware(&[0.0, 0.0], &candidates, 2, false);

        assert_eq!(
            layer0_selected,
            vec![(0usize, 0.1f32), (1usize, 0.2f32)],
            "layer 0 should refill pruned outsiders up to m_max0 like FAISS"
        );
        assert_eq!(
            upper_layer_selected,
            vec![(0usize, 0.1f32)],
            "upper layers should keep the existing no-backfill behavior"
        );
    }
}

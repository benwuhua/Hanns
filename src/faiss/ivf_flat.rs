//! IVF-Flat Index Implementation
//!
//! IVF (Inverted File) + Flat (no quantization)
//! 内存索引，倒排列表中存储原始向量
//!
//! OPT-005: 大规模 IVF-Flat 性能优化
//! - 扁平连续倒排表布局
//! - 两阶段 add()，批量构建 cluster-local 连续数据
//! - 固定容量 top-k 聚合，避免候选全量 materialize
//! - 保留 SIMD 批量距离计算

use std::cmp::Ordering;

use crate::api::{IndexConfig, Result, SearchRequest, SearchResult};
use crate::simd::{l2_batch_4_ptr, l2_distance_sq, l2_distance_sq_ptr};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct SearchHit {
    id: i64,
    dist: f32,
}

impl PartialEq for SearchHit {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist.to_bits() == other.dist.to_bits()
    }
}

impl Eq for SearchHit {}

impl PartialOrd for SearchHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

const INLINE_TOPK_CAP: usize = 32;
const EMPTY_HIT: SearchHit = SearchHit {
    id: 0,
    dist: f32::INFINITY,
};

struct TopKAccumulator {
    inline_hits: [SearchHit; INLINE_TOPK_CAP],
    heap_hits: Vec<SearchHit>,
    limit: usize,
    len: usize,
    visited: usize,
}

impl TopKAccumulator {
    fn new(limit: usize) -> Self {
        Self {
            inline_hits: [EMPTY_HIT; INLINE_TOPK_CAP],
            heap_hits: if limit > INLINE_TOPK_CAP {
                Vec::with_capacity(limit)
            } else {
                Vec::new()
            },
            limit,
            len: 0,
            visited: 0,
        }
    }

    #[inline]
    fn push(&mut self, id: i64, dist: f32) {
        if self.limit == 0 {
            return;
        }

        let candidate = SearchHit { id, dist };
        if self.limit <= INLINE_TOPK_CAP {
            insert_sorted(
                &mut self.inline_hits[..self.limit],
                &mut self.len,
                self.limit,
                candidate,
            );
        } else {
            insert_sorted_vec(&mut self.heap_hits, self.limit, candidate);
        }
    }

    fn merge(&mut self, other: Self) {
        self.visited += other.visited;
        for hit in other.hits() {
            self.push(hit.id, hit.dist);
        }
    }

    fn into_result(self) -> SearchResult {
        let visited = self.visited;
        let hits = self.into_hits();
        let mut ids = Vec::with_capacity(hits.len());
        let mut distances = Vec::with_capacity(hits.len());

        for hit in hits {
            ids.push(hit.id);
            distances.push(hit.dist);
        }

        SearchResult {
            ids,
            distances,
            elapsed_ms: 0.0,
            num_visited: visited,
        }
    }

    #[inline]
    fn hits(&self) -> &[SearchHit] {
        if self.limit <= INLINE_TOPK_CAP {
            &self.inline_hits[..self.len]
        } else {
            &self.heap_hits
        }
    }

    #[inline]
    fn into_hits(self) -> Vec<SearchHit> {
        if self.limit <= INLINE_TOPK_CAP {
            self.inline_hits[..self.len].to_vec()
        } else {
            self.heap_hits
        }
    }
}

#[inline]
fn insert_sorted(storage: &mut [SearchHit], len: &mut usize, limit: usize, candidate: SearchHit) {
    debug_assert_eq!(storage.len(), limit);

    if *len == limit && !is_better(candidate, storage[*len - 1]) {
        return;
    }

    let search_len = (*len).min(limit);
    let pos = storage[..search_len]
        .binary_search_by(|hit| hit.cmp(&candidate))
        .unwrap_or_else(|pos| pos);

    if *len < limit {
        for idx in (pos..*len).rev() {
            storage[idx + 1] = storage[idx];
        }
        storage[pos] = candidate;
        *len += 1;
        return;
    }

    for idx in (pos..limit - 1).rev() {
        storage[idx + 1] = storage[idx];
    }
    storage[pos] = candidate;
}

#[inline]
fn insert_sorted_vec(storage: &mut Vec<SearchHit>, limit: usize, candidate: SearchHit) {
    if storage.len() == limit && !is_better(candidate, storage[storage.len() - 1]) {
        return;
    }

    let pos = storage
        .binary_search_by(|hit| hit.cmp(&candidate))
        .unwrap_or_else(|pos| pos);

    if storage.len() < limit {
        storage.insert(pos, candidate);
    } else if pos < limit {
        storage.insert(pos, candidate);
        storage.truncate(limit);
    }
}

#[inline]
fn is_better(candidate: SearchHit, existing: SearchHit) -> bool {
    candidate < existing
}

/// IVF-Flat Index - stores raw vectors in flattened inverted lists
pub struct IvfFlatIndex {
    dim: usize,
    nlist: usize,
    nprobe: usize,

    /// Cluster centroids (连续内存)
    centroids: Vec<f32>,
    /// Flattened inverted list IDs
    invlist_ids: Vec<i64>,
    /// Flattened inverted list vectors
    invlist_vectors: Vec<f32>,
    /// Per-cluster starting vector index in flattened arrays
    invlist_offsets: Vec<usize>,
    /// Per-cluster vector count
    invlist_sizes: Vec<usize>,
    /// Insertion-order vectors retained for API compatibility
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl IvfFlatIndex {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);

        Ok(Self {
            dim: config.dim,
            nlist,
            nprobe,
            centroids: Vec::new(),
            invlist_ids: Vec::new(),
            invlist_vectors: Vec::new(),
            invlist_offsets: vec![0; nlist],
            invlist_sizes: vec![0; nlist],
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }

    /// Train the index (k-means for IVF)
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "empty training data".to_string(),
            ));
        }

        self.train_ivf(vectors)?;
        self.trained = true;
        Ok(n)
    }

    /// Train IVF (clustering)
    fn train_ivf(&mut self, vectors: &[f32]) -> Result<()> {
        use crate::quantization::KMeans;

        let mut km = KMeans::new(self.nlist, self.dim);
        km.train(vectors);

        self.centroids = km.centroids().to_vec();
        Ok(())
    }

    /// Add vectors
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        if n == 0 {
            return Ok(0);
        }

        let mut cluster_assignments = Vec::with_capacity(n);
        let mut batch_ids = Vec::with_capacity(n);
        let mut batch_sizes = vec![0usize; self.nlist];

        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            let cluster_id = self.find_nearest_centroid(vector);
            cluster_assignments.push(cluster_id);
            batch_sizes[cluster_id] += 1;

            let id = ids.map(|provided| provided[i]).unwrap_or(self.next_id);
            self.next_id += 1;
            batch_ids.push(id);
        }

        let mut batch_offsets = vec![0usize; self.nlist];
        let mut running = 0usize;
        for cluster_id in 0..self.nlist {
            batch_offsets[cluster_id] = running;
            running += batch_sizes[cluster_id];
        }

        let mut batch_fill = batch_offsets.clone();
        let mut batch_invlist_ids = vec![0i64; n];
        let mut batch_invlist_vectors = vec![0.0f32; n * self.dim];

        for i in 0..n {
            let cluster_id = cluster_assignments[i];
            let dst_idx = batch_fill[cluster_id];
            batch_fill[cluster_id] += 1;

            batch_invlist_ids[dst_idx] = batch_ids[i];

            let src_start = i * self.dim;
            let src_end = src_start + self.dim;
            let dst_start = dst_idx * self.dim;
            let dst_end = dst_start + self.dim;
            batch_invlist_vectors[dst_start..dst_end].copy_from_slice(&vectors[src_start..src_end]);
        }

        let old_total = self.ids.len();
        let new_total = old_total + n;
        let mut new_sizes = vec![0usize; self.nlist];
        for cluster_id in 0..self.nlist {
            new_sizes[cluster_id] = self.invlist_sizes[cluster_id] + batch_sizes[cluster_id];
        }

        let mut new_offsets = vec![0usize; self.nlist];
        let mut new_running = 0usize;
        for cluster_id in 0..self.nlist {
            new_offsets[cluster_id] = new_running;
            new_running += new_sizes[cluster_id];
        }

        let mut new_invlist_ids = vec![0i64; new_total];
        let mut new_invlist_vectors = vec![0.0f32; new_total * self.dim];

        for cluster_id in 0..self.nlist {
            let old_size = self.invlist_sizes[cluster_id];
            let batch_size = batch_sizes[cluster_id];
            if old_size == 0 && batch_size == 0 {
                continue;
            }

            let dst_start = new_offsets[cluster_id];

            if old_size > 0 {
                let old_start = self.invlist_offsets[cluster_id];
                let old_end = old_start + old_size;
                new_invlist_ids[dst_start..dst_start + old_size]
                    .copy_from_slice(&self.invlist_ids[old_start..old_end]);

                let old_vec_start = old_start * self.dim;
                let old_vec_end = old_end * self.dim;
                let dst_vec_start = dst_start * self.dim;
                let dst_vec_end = (dst_start + old_size) * self.dim;
                new_invlist_vectors[dst_vec_start..dst_vec_end]
                    .copy_from_slice(&self.invlist_vectors[old_vec_start..old_vec_end]);
            }

            if batch_size > 0 {
                let batch_start = batch_offsets[cluster_id];
                let batch_end = batch_start + batch_size;
                let dst_batch_start = dst_start + old_size;
                let dst_batch_end = dst_batch_start + batch_size;

                new_invlist_ids[dst_batch_start..dst_batch_end]
                    .copy_from_slice(&batch_invlist_ids[batch_start..batch_end]);

                let batch_vec_start = batch_start * self.dim;
                let batch_vec_end = batch_end * self.dim;
                let dst_vec_start = dst_batch_start * self.dim;
                let dst_vec_end = dst_batch_end * self.dim;
                new_invlist_vectors[dst_vec_start..dst_vec_end]
                    .copy_from_slice(&batch_invlist_vectors[batch_vec_start..batch_vec_end]);
            }
        }

        self.invlist_ids = new_invlist_ids;
        self.invlist_vectors = new_invlist_vectors;
        self.invlist_offsets = new_offsets;
        self.invlist_sizes = new_sizes;

        self.vectors.extend_from_slice(vectors);
        self.ids.extend_from_slice(&batch_ids);

        Ok(n)
    }

    /// Find nearest centroid (使用 l2_distance_sq)
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;

        for c in 0..self.nlist {
            let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
            let dist = l2_distance_sq(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = c;
            }
        }

        best
    }

    #[inline]
    fn scan_cluster(
        &self,
        cluster_id: usize,
        query: &[f32],
        top_k: usize,
        filter: Option<&dyn crate::api::Predicate>,
    ) -> TopKAccumulator {
        let size = self.invlist_sizes[cluster_id];
        let mut acc = TopKAccumulator::new(top_k);
        acc.visited = size;

        if size == 0 {
            return acc;
        }

        let offset = self.invlist_offsets[cluster_id];
        let ids = &self.invlist_ids[offset..offset + size];
        let vectors = &self.invlist_vectors[offset * self.dim..(offset + size) * self.dim];

        match filter {
            Some(predicate) => {
                let query_ptr = query.as_ptr();
                let base_ptr = vectors.as_ptr();
                for (idx, &id) in ids.iter().enumerate().take(size) {
                    if predicate.evaluate(id) {
                        // SAFETY:
                        // - `base_ptr` 指向当前 cluster 的连续向量区间，长度为 `size * dim`
                        // - `idx < size`，因此 `idx * dim .. idx * dim + dim` 始终落在该区间内
                        // - `query_ptr` 来自 `query` 切片，长度与 `self.dim` 一致
                        let dist = unsafe {
                            l2_distance_sq_ptr(query_ptr, base_ptr.add(idx * self.dim), self.dim)
                        };
                        acc.push(id, dist);
                    }
                }
            }
            None => unsafe {
                // SAFETY:
                // - `vectors` 是紧凑连续布局，每个向量占 `dim` 个 `f32`
                // - 当 `idx + 4 <= size` 时，`l2_batch_4_ptr` 读取的 4 个向量都在该切片边界内
                // - remainder 路径使用相同边界条件逐个读取，不会越界
                let query_ptr = query.as_ptr();
                let base_ptr = vectors.as_ptr();
                let mut idx = 0usize;

                while idx + 4 <= size {
                    let dists =
                        l2_batch_4_ptr(query_ptr, base_ptr.add(idx * self.dim), self.dim, self.dim);
                    acc.push(ids[idx], dists[0]);
                    acc.push(ids[idx + 1], dists[1]);
                    acc.push(ids[idx + 2], dists[2]);
                    acc.push(ids[idx + 3], dists[3]);
                    idx += 4;
                }

                while idx < size {
                    let dist =
                        l2_distance_sq_ptr(query_ptr, base_ptr.add(idx * self.dim), self.dim);
                    acc.push(ids[idx], dist);
                    idx += 1;
                }
            },
        }

        acc
    }

    /// Search
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }

        let top_k = req.top_k;
        if top_k == 0 {
            return Ok(SearchResult {
                ids: Vec::new(),
                distances: Vec::new(),
                elapsed_ms: 0.0,
                num_visited: 0,
            });
        }

        let nprobe = if req.nprobe > 0 {
            req.nprobe
        } else {
            self.nprobe
        }
        .min(self.nlist)
        .max(1);

        let mut cluster_dists = Vec::with_capacity(self.nlist);
        for c in 0..self.nlist {
            let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
            cluster_dists.push((c, l2_distance_sq(query, centroid)));
        }

        if nprobe < cluster_dists.len() {
            cluster_dists.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
            cluster_dists.truncate(nprobe);
        }

        let filter = req.filter.as_deref();

        #[cfg(feature = "parallel")]
        {
            let partials: Vec<TopKAccumulator> = cluster_dists
                .par_iter()
                .map(|(cluster_id, _)| self.scan_cluster(*cluster_id, query, top_k, filter))
                .collect();

            let mut merged = TopKAccumulator::new(top_k);
            for partial in partials {
                merged.merge(partial);
            }
            Ok(merged.into_result())
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut merged = TopKAccumulator::new(top_k);
            for (cluster_id, _) in cluster_dists {
                merged.merge(self.scan_cluster(cluster_id, query, top_k, filter));
            }
            Ok(merged.into_result())
        }
    }

    /// Get number of vectors
    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get raw vectors by IDs
    pub fn get_vectors(&self, ids: &[i64]) -> Vec<Option<Vec<f32>>> {
        ids.iter()
            .map(|target_id| {
                self.ids
                    .iter()
                    .position(|stored_id| stored_id == target_id)
                    .map(|idx| {
                        let start = idx * self.dim;
                        let end = start + self.dim;
                        self.vectors[start..end].to_vec()
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexParams, IndexType, MetricType};

    #[test]
    fn test_ivf_flat_new() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 128,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf(100, 10),
        };

        let index = IvfFlatIndex::new(&config).unwrap();
        assert_eq!(index.dim(), 128);
    }

    #[test]
    fn test_ivf_flat_train_add_search() {
        let config = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf(2, 2), // nprobe=2 to search both clusters
        };

        let mut index = IvfFlatIndex::new(&config).unwrap();

        let train_data = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        ];

        index.train(&train_data).unwrap();
        index.add(&train_data, None).unwrap();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 2, // Search both clusters
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert!(!result.ids.is_empty() && result.ids.len() <= 2); // IVF may return fewer than top_k
    }
}

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
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::api::{IndexConfig, MetricType, Result, SearchRequest, SearchResult};
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

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// IVF-Flat Index - stores raw vectors in flattened inverted lists
pub struct IvfFlatIndex {
    dim: usize,
    nlist: usize,
    nprobe: usize,
    metric_type: MetricType,

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
            metric_type: config.metric_type,
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

    /// Add vectors with external IDs.
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[i64]) -> Result<usize> {
        self.add(vectors, Some(ids))
    }

    /// Find nearest centroid based on configured metric.
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        match self.metric_type {
            MetricType::L2 | MetricType::Hamming => {
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
            MetricType::Ip | MetricType::Cosine => {
                let mut best_score = f32::NEG_INFINITY;
                let mut best = 0;
                for c in 0..self.nlist {
                    let centroid = &self.centroids[c * self.dim..(c + 1) * self.dim];
                    let score = dot_product(vector, centroid);
                    if score > best_score {
                        best_score = score;
                        best = c;
                    }
                }
                best
            }
        }
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
                match self.metric_type {
                    MetricType::L2 | MetricType::Hamming => {
                        let query_ptr = query.as_ptr();
                        let base_ptr = vectors.as_ptr();
                        for (idx, &id) in ids.iter().enumerate().take(size) {
                            if predicate.evaluate(id) {
                                // SAFETY:
                                // - `base_ptr` 指向当前 cluster 的连续向量区间，长度为 `size * dim`
                                // - `idx < size`，因此 `idx * dim .. idx * dim + dim` 始终落在该区间内
                                // - `query_ptr` 来自 `query` 切片，长度与 `self.dim` 一致
                                let dist = unsafe {
                                    l2_distance_sq_ptr(
                                        query_ptr,
                                        base_ptr.add(idx * self.dim),
                                        self.dim,
                                    )
                                };
                                acc.push(id, dist);
                            }
                        }
                    }
                    MetricType::Ip | MetricType::Cosine => {
                        for (idx, &id) in ids.iter().enumerate().take(size) {
                            if predicate.evaluate(id) {
                                let start = idx * self.dim;
                                let end = start + self.dim;
                                let vec = &vectors[start..end];
                                // Store negative dot so that smaller is better in TopKAccumulator.
                                let dist = -dot_product(query, vec);
                                acc.push(id, dist);
                            }
                        }
                    }
                }
            }
            None => match self.metric_type {
                MetricType::L2 | MetricType::Hamming => unsafe {
                    // SAFETY:
                    // - `vectors` 是紧凑连续布局，每个向量占 `dim` 个 `f32`
                    // - 当 `idx + 4 <= size` 时，`l2_batch_4_ptr` 读取的 4 个向量都在该切片边界内
                    // - remainder 路径使用相同边界条件逐个读取，不会越界
                    let query_ptr = query.as_ptr();
                    let base_ptr = vectors.as_ptr();
                    let mut idx = 0usize;

                    while idx + 4 <= size {
                        let dists = l2_batch_4_ptr(
                            query_ptr,
                            base_ptr.add(idx * self.dim),
                            self.dim,
                            self.dim,
                        );
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
                MetricType::Ip | MetricType::Cosine => {
                    for (idx, &id) in ids.iter().enumerate().take(size) {
                        let start = idx * self.dim;
                        let end = start + self.dim;
                        let vec = &vectors[start..end];
                        let dist = -dot_product(query, vec);
                        acc.push(id, dist);
                    }
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
            let score = match self.metric_type {
                MetricType::L2 | MetricType::Hamming => l2_distance_sq(query, centroid),
                MetricType::Ip | MetricType::Cosine => -dot_product(query, centroid),
            };
            cluster_dists.push((c, score));
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

    /// Search multiple queries in parallel (requires rayon)
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &self,
        queries: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> std::result::Result<Vec<Vec<(usize, f32)>>, Box<dyn std::error::Error>> {
        if !queries.len().is_multiple_of(self.dim) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "query dimension mismatch: len {} not divisible by dim {}",
                    queries.len(),
                    self.dim
                ),
            )));
        }

        let n_queries = queries.len() / self.dim;
        let req = SearchRequest {
            top_k,
            nprobe,
            ..Default::default()
        };

        let results: Result<Vec<Vec<(usize, f32)>>> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let start = q_idx * self.dim;
                let end = start + self.dim;
                let query = &queries[start..end];
                let result = self.search(query, &req)?;
                Ok(result
                    .ids
                    .into_iter()
                    .zip(result.distances.into_iter())
                    .filter_map(|(id, dist)| {
                        if id >= 0 {
                            Some((id as usize, dist))
                        } else {
                            None
                        }
                    })
                    .collect())
            })
            .collect();

        results.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    /// Get number of vectors
    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn has_raw_data(&self) -> bool {
        true
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

    fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(b"IVFFLAT")?;
        writer.write_all(&(self.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.nlist as u32).to_le_bytes())?;
        writer.write_all(&(self.nprobe as u32).to_le_bytes())?;
        let metric_byte = match self.metric_type {
            MetricType::L2 => 0u8,
            MetricType::Ip => 1u8,
            MetricType::Cosine => 2u8,
            MetricType::Hamming => 3u8,
        };
        writer.write_all(&[metric_byte])?;
        writer.write_all(&self.next_id.to_le_bytes())?;

        for &v in &self.centroids {
            writer.write_all(&v.to_le_bytes())?;
        }

        for &off in &self.invlist_offsets {
            writer.write_all(&(off as u64).to_le_bytes())?;
        }
        for &sz in &self.invlist_sizes {
            writer.write_all(&(sz as u64).to_le_bytes())?;
        }

        let n_total = self.invlist_ids.len() as u64;
        writer.write_all(&n_total.to_le_bytes())?;
        for &id in &self.invlist_ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        for &v in &self.invlist_vectors {
            writer.write_all(&v.to_le_bytes())?;
        }

        let n_ordered = self.ids.len() as u64;
        writer.write_all(&n_ordered.to_le_bytes())?;
        for &id in &self.ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        for &v in &self.vectors {
            writer.write_all(&v.to_le_bytes())?;
        }

        Ok(())
    }

    fn read_from<R: Read>(reader: &mut R, dim: usize) -> Result<Self> {
        let mut magic = [0u8; 7];
        reader.read_exact(&mut magic)?;
        if &magic != b"IVFFLAT" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid IVFFLAT magic".to_string(),
            ));
        }

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let stored_dim = u32::from_le_bytes(u32_buf) as usize;
        if stored_dim != dim {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "dim mismatch: load dim {} vs stored dim {}",
                dim, stored_dim
            )));
        }

        reader.read_exact(&mut u32_buf)?;
        let nlist = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let nprobe = u32::from_le_bytes(u32_buf) as usize;
        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let metric_type = MetricType::from_bytes(metric_buf[0]);

        let mut i64_buf = [0u8; 8];
        reader.read_exact(&mut i64_buf)?;
        let next_id = i64::from_le_bytes(i64_buf);

        let mut centroids = vec![0.0f32; nlist * stored_dim];
        for value in &mut centroids {
            let mut fbuf = [0u8; 4];
            reader.read_exact(&mut fbuf)?;
            *value = f32::from_le_bytes(fbuf);
        }

        let mut invlist_offsets = vec![0usize; nlist];
        for off in &mut invlist_offsets {
            let mut ubuf = [0u8; 8];
            reader.read_exact(&mut ubuf)?;
            *off = u64::from_le_bytes(ubuf) as usize;
        }

        let mut invlist_sizes = vec![0usize; nlist];
        for sz in &mut invlist_sizes {
            let mut ubuf = [0u8; 8];
            reader.read_exact(&mut ubuf)?;
            *sz = u64::from_le_bytes(ubuf) as usize;
        }

        let mut u64_buf = [0u8; 8];
        reader.read_exact(&mut u64_buf)?;
        let n_total = u64::from_le_bytes(u64_buf) as usize;

        let mut invlist_ids = vec![0i64; n_total];
        for id in &mut invlist_ids {
            let mut ibuf = [0u8; 8];
            reader.read_exact(&mut ibuf)?;
            *id = i64::from_le_bytes(ibuf);
        }

        let mut invlist_vectors = vec![0.0f32; n_total * stored_dim];
        for value in &mut invlist_vectors {
            let mut fbuf = [0u8; 4];
            reader.read_exact(&mut fbuf)?;
            *value = f32::from_le_bytes(fbuf);
        }

        reader.read_exact(&mut u64_buf)?;
        let n_ordered = u64::from_le_bytes(u64_buf) as usize;

        let mut ids = vec![0i64; n_ordered];
        for id in &mut ids {
            let mut ibuf = [0u8; 8];
            reader.read_exact(&mut ibuf)?;
            *id = i64::from_le_bytes(ibuf);
        }

        let mut vectors = vec![0.0f32; n_ordered * stored_dim];
        for value in &mut vectors {
            let mut fbuf = [0u8; 4];
            reader.read_exact(&mut fbuf)?;
            *value = f32::from_le_bytes(fbuf);
        }

        Ok(Self {
            dim: stored_dim,
            nlist,
            nprobe,
            metric_type,
            centroids,
            invlist_ids,
            invlist_vectors,
            invlist_offsets,
            invlist_sizes,
            vectors,
            ids,
            next_id,
            trained: true,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        self.write_to(&mut file)
    }

    pub fn load(path: &Path, dim: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        Self::read_from(&mut file, dim)
    }

    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    pub fn deserialize_from_bytes(bytes: &[u8], dim: usize) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::read_from(&mut cursor, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexParams, IndexType, MetricType};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

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

    #[test]
    fn test_ivf_flat_save_load() {
        let dim = 64usize;
        let n = 1000usize;
        let nlist = 32usize;
        let mut rng = StdRng::seed_from_u64(42);
        let n_clusters = 20usize;
        let centers: Vec<f32> = (0..n_clusters * dim)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            let c = i % n_clusters;
            let center = &centers[c * dim..(c + 1) * dim];
            for &v in center {
                let noise = (rng.r#gen::<f32>() - 0.5) * 0.1;
                vectors.push(v + noise);
            }
        }

        let cfg = IndexConfig {
            index_type: IndexType::IvfFlat,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf(nlist, 8),
        };
        let mut index = IvfFlatIndex::new(&cfg).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let path = Path::new("/tmp/ivf_flat_test.bin");
        let _ = std::fs::remove_file(path);
        index.save(path).unwrap();

        let loaded = IvfFlatIndex::load(path, dim).unwrap();

        let req = SearchRequest {
            top_k: 10,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };
        let num_queries = 100usize;
        let mut total_hits = 0usize;
        for qi in 0..num_queries {
            let query = &vectors[qi * dim..(qi + 1) * dim];
            let result = loaded.search(query, &req).unwrap();
            let mut gt: Vec<(i64, f32)> = (0..n)
                .map(|i| {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    let d = v
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum::<f32>();
                    (i as i64, d)
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_top10: Vec<i64> = gt.into_iter().take(10).map(|(id, _)| id).collect();
            total_hits += gt_top10
                .iter()
                .filter(|id| result.ids.iter().take(10).any(|rid| rid == *id))
                .count();
        }
        let recall = total_hits as f64 / (num_queries as f64 * 10.0);
        assert!(
            recall >= 0.8,
            "save/load ivf-flat recall@10 too low: {:.3}",
            recall
        );
    }
}

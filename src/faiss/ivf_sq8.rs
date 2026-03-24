//! IVF-SQ8 Index Implementation
//!
//! IVF (Inverted File) + SQ8 (Scalar Quantization 8-bit)
//! 内存优化索引，适合大规模数据

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::api::{IndexConfig, MetricType, Result, SearchRequest, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::executor::l2_distance;
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::quantization::ScalarQuantizer;
use crate::simd::dot_product_f32;

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

#[allow(dead_code)]
#[inline]
fn prefetch_sq_code(codes: &[u8], i: usize, n: usize, dim: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if i + 2 < n {
            let base = (i + 2) * dim;
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            // SAFETY: base is in-bounds because i + 2 < n and n <= codes.len() / dim.
            unsafe {
                let ptr = codes.as_ptr().add(base) as *const i8;
                _mm_prefetch(ptr, _MM_HINT_T0);
                if dim > 64 {
                    _mm_prefetch(ptr.add(64), _MM_HINT_T0);
                }
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (codes, i, n, dim);
    }
}

/// IVF-SQ8 Index
#[allow(dead_code)]
pub struct IvfSq8Index {
    config: IndexConfig,
    dim: usize,
    nlist: usize,  // Number of clusters
    nprobe: usize, // Number of clusters to search
    metric_type: MetricType,

    /// Cluster centroids
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id -> (ids, flat quantized residual codes)
    inverted_lists: HashMap<usize, (Vec<i64>, Vec<u8>)>,
    /// Internal row ids aligned with each inverted-list entry.
    inverted_list_rows: HashMap<usize, Vec<usize>>,
    /// Scalar quantizer for residuals
    quantizer: ScalarQuantizer,
    /// All vectors (for decoding)
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl IvfSq8Index {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            nlist,
            nprobe,
            metric_type: config.metric_type,
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
            inverted_list_rows: HashMap::new(),
            quantizer: ScalarQuantizer::new(config.dim, 8),
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }

    /// Train the index (k-means for IVF, SQ for quantization)
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "empty training data".to_string(),
            ));
        }

        // Step 1: train IVF centroids first
        self.train_ivf(vectors)?;

        // Step 2: build residual training set using nearest centroids
        let mut residuals = Vec::with_capacity(n * self.dim);
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            let cluster = self.find_nearest_centroid(vector);
            let residual = self.compute_residual(vector, cluster);
            residuals.extend_from_slice(&residual);
        }

        // Step 3: train scalar quantizer on residuals (not raw vectors)
        self.quantizer.train(&residuals);

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

        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];
            let internal_row = self.ids.len();

            // Find nearest centroid
            let cluster = self.find_nearest_centroid(vector);

            // Compute residual
            let residual = self.compute_residual(vector, cluster);

            // Quantize residual
            let quantized = self.quantizer.encode(&residual);

            // Get ID
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            // Store
            self.ids.push(id);
            self.vectors.extend_from_slice(vector);

            let entry = self
                .inverted_lists
                .entry(cluster)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.push(id);
            entry.1.extend_from_slice(&quantized);
            self.inverted_list_rows
                .entry(cluster)
                .or_insert_with(Vec::new)
                .push(internal_row);
        }

        Ok(n)
    }

    /// Search
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }

        if self.ids.is_empty() {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }

        let n_queries = query.len() / self.dim;
        if n_queries * self.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let k = req.top_k;
        let nprobe = req.nprobe.min(self.nlist);

        if k == 0 {
            return Ok(SearchResult::new(Vec::new(), Vec::new(), 0.0));
        }

        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];
            let clusters = self.search_clusters(query_vec, nprobe);

            #[cfg(feature = "parallel")]
            let merged = {
                use rayon::prelude::*;
                let dim = self.dim;
                let partials: Vec<TopKAccumulator> = clusters
                    .par_iter()
                    .map_init(
                        || (vec![0.0f32; dim], vec![0i16; dim]),
                        |(q_residual_buf, q_precomputed_buf), &cluster_id| {
                            self.scan_cluster_with_buf(
                                cluster_id,
                                query_vec,
                                k,
                                q_residual_buf,
                                q_precomputed_buf,
                                None,
                                None,
                            )
                        },
                    )
                    .collect();
                let mut merged = TopKAccumulator::new(k);
                for partial in partials {
                    merged.merge(partial);
                }
                merged
            };

            #[cfg(not(feature = "parallel"))]
            let merged = {
                let mut q_residual_buf = vec![0.0f32; self.dim];
                let mut q_precomputed_buf = vec![0i16; self.dim];
                let mut merged = TopKAccumulator::new(k);
                for cluster_id in clusters {
                    merged.merge(self.scan_cluster_with_buf(
                        cluster_id,
                        query_vec,
                        k,
                        &mut q_residual_buf,
                        &mut q_precomputed_buf,
                        None,
                        None,
                    ));
                }
                merged
            };

            let hits = merged.into_hits();
            let offset = q_idx * k;
            for (i, hit) in hits.into_iter().enumerate().take(k) {
                all_ids[offset + i] = hit.id;
                all_dists[offset + i] = hit.dist;
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    fn scan_cluster_with_buf(
        &self,
        cluster_id: usize,
        query_vec: &[f32],
        top_k: usize,
        q_residual_buf: &mut Vec<f32>,
        q_precomputed_buf: &mut Vec<i16>,
        bitset: Option<&BitsetView>,
        cluster_rows: Option<&[usize]>,
    ) -> TopKAccumulator {
        let mut acc = TopKAccumulator::new(top_k);
        if let Some((ids, codes)) = self.inverted_lists.get(&cluster_id) {
            let centroid_vec = &self.centroids[cluster_id * self.dim..(cluster_id + 1) * self.dim];
            let n = ids.len().min(codes.len() / self.dim);
            acc.visited = n;

            match self.metric_type {
                MetricType::L2 | MetricType::Hamming => {
                    q_residual_buf.clear();
                    q_residual_buf.extend(
                        query_vec
                            .iter()
                            .zip(centroid_vec.iter())
                            .map(|(a, b)| a - b),
                    );
                    if q_precomputed_buf.len() != q_residual_buf.len() {
                        q_precomputed_buf.resize(q_residual_buf.len(), 0);
                    }
                    self.quantizer
                        .precompute_query_into(q_residual_buf, q_precomputed_buf.as_mut_slice());

                    for i in 0..n {
                        if !Self::row_allowed_by_bitset(
                            cluster_rows.and_then(|rows| rows.get(i)).copied(),
                            bitset,
                        ) {
                            continue;
                        }
                        let code = &codes[i * self.dim..(i + 1) * self.dim];
                        let dist = self
                            .quantizer
                            .sq_l2_precomputed(q_precomputed_buf.as_slice(), code);
                        acc.push(ids[i], dist);
                    }
                }
                MetricType::Ip | MetricType::Cosine => {
                    // Scheme A: decode SQ8 residual and score with negative dot product.
                    // Keep "smaller is better" convention for TopKAccumulator.
                    let centroid_dot = dot_product_f32(query_vec, centroid_vec);
                    for i in 0..n {
                        if !Self::row_allowed_by_bitset(
                            cluster_rows.and_then(|rows| rows.get(i)).copied(),
                            bitset,
                        ) {
                            continue;
                        }
                        let code = &codes[i * self.dim..(i + 1) * self.dim];
                        let residual_dot = self.quantizer.decode_dot_f32(code, query_vec);
                        let dist = -(centroid_dot + residual_dot);
                        acc.push(ids[i], dist);
                    }
                }
            }
        }
        acc
    }

    #[allow(dead_code)]
    fn scan_cluster(&self, cluster_id: usize, query_vec: &[f32], top_k: usize) -> TopKAccumulator {
        let mut q_residual_buf = vec![0.0f32; self.dim];
        let mut q_precomputed_buf = vec![0i16; self.dim];
        self.scan_cluster_with_buf(
            cluster_id,
            query_vec,
            top_k,
            &mut q_residual_buf,
            &mut q_precomputed_buf,
            None,
            None,
        )
    }

    #[inline]
    fn row_allowed_by_bitset(
        internal_row: Option<usize>,
        bitset: Option<&BitsetView>,
    ) -> bool {
        let Some(bitset) = bitset else {
            return true;
        };
        let Some(row) = internal_row else {
            return true;
        };
        if row < bitset.len() {
            return !bitset.get(row);
        }
        true
    }

    #[inline]
    fn dist_in_range(&self, dist: f32, radius: f32, range_filter: f32) -> bool {
        match self.metric_type {
            MetricType::L2 | MetricType::Hamming => radius <= dist && dist <= range_filter,
            MetricType::Ip | MetricType::Cosine => range_filter <= dist && dist <= radius,
        }
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        match self.metric_type {
            MetricType::L2 | MetricType::Hamming => {
                let mut min_dist = f32::MAX;
                let mut best = 0;
                for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
                    let dist = l2_distance(vector, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best = i;
                    }
                }
                best
            }
            MetricType::Ip | MetricType::Cosine => {
                let mut best_score = f32::NEG_INFINITY;
                let mut best = 0;
                for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
                    let score = dot_product_f32(vector, centroid);
                    if score > best_score {
                        best_score = score;
                        best = i;
                    }
                }
                best
            }
        }
    }

    /// Search clusters
    fn search_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.nlist)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let dist = match self.metric_type {
                    MetricType::L2 | MetricType::Hamming => l2_distance(query, centroid),
                    MetricType::Ip | MetricType::Cosine => -dot_product_f32(query, centroid),
                };
                (i, dist)
            })
            .collect();

        if nprobe >= distances.len() {
            distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            return distances.into_iter().map(|(i, _)| i).collect();
        }

        distances.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
        distances[..nprobe].sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances[..nprobe].iter().map(|(i, _)| *i).collect()
    }

    /// Compute residual
    fn compute_residual(&self, vector: &[f32], cluster: usize) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        vector.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }

    fn rebuild_inverted_list_rows_from_vectors(&self) -> HashMap<usize, Vec<usize>> {
        let mut rows: HashMap<usize, Vec<usize>> =
            HashMap::with_capacity(self.inverted_lists.len());
        let total_rows = self.ids.len().min(self.vectors.len() / self.dim);
        for row in 0..total_rows {
            let start = row * self.dim;
            let vector = &self.vectors[start..start + self.dim];
            let cluster = self.find_nearest_centroid(vector);
            rows.entry(cluster).or_insert_with(Vec::new).push(row);
        }
        rows
    }

    fn synthetic_inverted_list_rows(
        inverted_lists: &HashMap<usize, (Vec<i64>, Vec<u8>)>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut rows: HashMap<usize, Vec<usize>> = HashMap::with_capacity(inverted_lists.len());
        let mut cluster_ids: Vec<usize> = inverted_lists.keys().copied().collect();
        cluster_ids.sort_unstable();

        let mut next_row = 0usize;
        for cluster_id in cluster_ids {
            let Some((ids, _)) = inverted_lists.get(&cluster_id) else {
                continue;
            };
            let cluster_rows: Vec<usize> = (next_row..next_row + ids.len()).collect();
            next_row += ids.len();
            rows.insert(cluster_id, cluster_rows);
        }
        rows
    }

    /// Add vectors in parallel (requires rayon)
    #[cfg(feature = "parallel")]
    pub fn add_parallel(
        &mut self,
        vectors: &[f32],
        ids: Option<&[i64]>,
        _num_threads: usize,
    ) -> Result<usize> {
        use rayon::prelude::*;

        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        let dim = self.dim;
        let nlist = self.nlist;
        let centroids = self.centroids.clone();
        let quantizer = &self.quantizer;
        let base_row = self.ids.len();

        // Parallel: assign vectors to clusters
        let assignments: Vec<(usize, i64, usize, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let vector = &vectors[start..start + dim];

                // Find nearest centroid
                let mut cluster = 0usize;
                match self.metric_type {
                    MetricType::L2 | MetricType::Hamming => {
                        let mut min_dist = f32::MAX;
                        for c in 0..nlist {
                            let dist = l2_distance(vector, &centroids[c * dim..(c + 1) * dim]);
                            if dist < min_dist {
                                min_dist = dist;
                                cluster = c;
                            }
                        }
                    }
                    MetricType::Ip | MetricType::Cosine => {
                        let mut best_score = f32::NEG_INFINITY;
                        for c in 0..nlist {
                            let score =
                                dot_product_f32(vector, &centroids[c * dim..(c + 1) * dim]);
                            if score > best_score {
                                best_score = score;
                                cluster = c;
                            }
                        }
                    }
                }

                // Compute residual
                let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
                let residual: Vec<f32> = vector.iter().zip(centroid).map(|(a, b)| a - b).collect();

                // Quantize residual
                let quantized = quantizer.encode(&residual);

                let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
                (cluster, id, base_row + i, quantized)
            })
            .collect();

        // Collect by cluster
        let mut cluster_data: HashMap<usize, (Vec<i64>, Vec<u8>, Vec<usize>)> = HashMap::new();
        for (cluster, id, internal_row, quantized) in assignments {
            let entry = cluster_data
                .entry(cluster)
                .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            entry.0.push(id);
            entry.1.extend_from_slice(&quantized);
            entry.2.push(internal_row);
        }

        // Merge into inverted lists
        for (cluster, (mut ids_buf, mut codes_buf, mut rows_buf)) in cluster_data {
            let entry = self
                .inverted_lists
                .entry(cluster)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            entry.0.append(&mut ids_buf);
            entry.1.append(&mut codes_buf);
            self.inverted_list_rows
                .entry(cluster)
                .or_insert_with(Vec::new)
                .append(&mut rows_buf);
        }

        // Update ids and vectors
        for i in 0..n {
            let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
            self.ids.push(id);
            let start = i * self.dim;
            self.vectors
                .extend_from_slice(&vectors[start..start + self.dim]);
        }
        self.next_id = n as i64;

        tracing::debug!("Added {} vectors to IVF-SQ8 (parallel)", n);
        Ok(n)
    }

    /// Search multiple queries in parallel (requires rayon)
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &self,
        query: &[f32],
        req: &SearchRequest,
        _num_threads: usize,
    ) -> Result<SearchResult> {
        use rayon::prelude::*;

        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }

        if self.ids.is_empty() {
            return Ok(SearchResult::new(vec![], vec![], 0.0));
        }

        let n_queries = query.len() / self.dim;
        if n_queries * self.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let k = req.top_k;
        let nprobe = req.nprobe.min(self.nlist);
        let dim = self.dim;

        // Parallel search for each query
        let results: Vec<Vec<(i64, f32)>> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * dim;
                let query_vec = &query[q_start..q_start + dim];
                let clusters = self.search_clusters(query_vec, nprobe);
                let mut q_residual_buf = vec![0.0f32; dim];
                let mut q_precomputed_buf = vec![0i16; dim];
                let mut acc = TopKAccumulator::new(k);

                for cluster_id in clusters {
                    acc.merge(self.scan_cluster_with_buf(
                        cluster_id,
                        query_vec,
                        k,
                        &mut q_residual_buf,
                        &mut q_precomputed_buf,
                        None,
                        None,
                    ));
                }

                acc.into_hits()
                    .into_iter()
                    .map(|h| (h.id, h.dist))
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];

        for (q_idx, res) in results.into_iter().enumerate() {
            let offset = q_idx * k;
            for (i, item) in res.into_iter().enumerate().take(k) {
                all_ids[offset + i] = item.0;
                all_dists[offset + i] = item.1;
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    pub fn range_search(&self, query: &[f32], radius: f32, range_filter: f32) -> Vec<(i64, f32)> {
        if !self.trained || self.ids.is_empty() || query.len() != self.dim {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut q_residual_buf = vec![0.0f32; self.dim];
        let mut q_precomputed_buf = vec![0i16; self.dim];

        for cluster_id in 0..self.nlist {
            let Some((cluster_ids, _)) = self.inverted_lists.get(&cluster_id) else {
                continue;
            };
            if cluster_ids.is_empty() {
                continue;
            }

            let hits = self
                .scan_cluster_with_buf(
                    cluster_id,
                    query,
                    cluster_ids.len(),
                    &mut q_residual_buf,
                    &mut q_precomputed_buf,
                    None,
                    None,
                )
                .into_hits();

            results.extend(
                hits.into_iter()
                    .filter(|hit| self.dist_in_range(hit.dist, radius, range_filter))
                    .map(|hit| (hit.id, hit.dist)),
            );
        }

        results.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results
    }

    /// Get total number of vectors
    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> Result<()> {
        const HEADER_VERSION_V3: u32 = 3;

        // Magic
        writer.write_all(b"IVFSQ8")?;
        writer.write_all(&HEADER_VERSION_V3.to_le_bytes())?;

        // Dim
        writer.write_all(&(self.dim as u32).to_le_bytes())?;

        // Nlist
        writer.write_all(&(self.nlist as u32).to_le_bytes())?;
        // Nprobe (persist runtime search breadth)
        writer.write_all(&(self.nprobe as u32).to_le_bytes())?;
        // Metric type
        let metric_byte = match self.metric_type {
            MetricType::L2 => 0u8,
            MetricType::Ip => 1u8,
            MetricType::Cosine => 2u8,
            MetricType::Hamming => 3u8,
        };
        writer.write_all(&[metric_byte])?;

        // Centroids
        let centroid_bytes: Vec<u8> = self
            .centroids
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        writer.write_all(&centroid_bytes)?;

        // Quantizer params
        writer.write_all(&self.quantizer.min_val.to_le_bytes())?;
        writer.write_all(&self.quantizer.max_val.to_le_bytes())?;
        writer.write_all(&self.quantizer.scale.to_le_bytes())?;

        // IDs
        writer.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            writer.write_all(&id.to_le_bytes())?;
        }

        // Vectors
        let vec_bytes: Vec<u8> = self.vectors.iter().flat_map(|&f| f.to_le_bytes()).collect();
        writer.write_all(&vec_bytes)?;

        // Inverted lists (flat layout): cluster_id + n + ids + codes
        writer.write_all(&(self.inverted_lists.len() as u64).to_le_bytes())?;
        for (&cluster_id, (ids, codes)) in &self.inverted_lists {
            let n = ids.len();
            writer.write_all(&(cluster_id as u64).to_le_bytes())?;
            writer.write_all(&(n as u64).to_le_bytes())?;

            for &id in ids {
                writer.write_all(&id.to_le_bytes())?;
            }

            let expected = n * self.dim;
            if codes.len() != expected {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "invalid inverted list code length for cluster {}: {} != {}",
                    cluster_id,
                    codes.len(),
                    expected
                )));
            }
            writer.write_all(codes)?;
        }

        Ok(())
    }

    fn read_from<R: std::io::Read>(reader: &mut R, dim: usize) -> Result<Self> {
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;
        if &magic != b"IVFSQ8" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid IVFSQ8 magic".to_string(),
            ));
        }

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?;
        let first_word = u32::from_le_bytes(u32_buf);
        let (stored_dim, nlist, nprobe, metric_type) = match first_word {
            // v2: [magic][version=2][dim][nlist]..., metric defaults to L2, nprobe defaults to 8
            2 => {
                reader.read_exact(&mut u32_buf)?;
                let stored_dim = u32::from_le_bytes(u32_buf) as usize;
                reader.read_exact(&mut u32_buf)?;
                let nlist = u32::from_le_bytes(u32_buf) as usize;
                (stored_dim, nlist, 8usize, MetricType::L2)
            }
            // v3: [magic][version=3][dim][nlist][nprobe][metric]...
            3 => {
                reader.read_exact(&mut u32_buf)?;
                let stored_dim = u32::from_le_bytes(u32_buf) as usize;
                reader.read_exact(&mut u32_buf)?;
                let nlist = u32::from_le_bytes(u32_buf) as usize;
                reader.read_exact(&mut u32_buf)?;
                let nprobe = u32::from_le_bytes(u32_buf) as usize;
                let mut metric_buf = [0u8; 1];
                reader.read_exact(&mut metric_buf)?;
                let metric_type = MetricType::from_bytes(metric_buf[0]);
                (stored_dim, nlist, nprobe, metric_type)
            }
            // Legacy (unversioned) format:
            // [magic][dim][nlist][metric]...
            stored_dim => {
                let stored_dim = stored_dim as usize;
                reader.read_exact(&mut u32_buf)?;
                let nlist = u32::from_le_bytes(u32_buf) as usize;
                let mut metric_buf = [0u8; 1];
                reader.read_exact(&mut metric_buf)?;
                let metric_type = MetricType::from_bytes(metric_buf[0]);
                (stored_dim, nlist, 8usize, metric_type)
            }
        };
        if stored_dim != dim {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "dim mismatch: load dim {} vs stored dim {}",
                dim, stored_dim
            )));
        }

        let mut centroids = vec![0.0f32; nlist * stored_dim];
        for value in &mut centroids {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *value = f32::from_le_bytes(buf);
        }

        let mut f32_buf = [0u8; 4];
        reader.read_exact(&mut f32_buf)?;
        let min_val = f32::from_le_bytes(f32_buf);
        reader.read_exact(&mut f32_buf)?;
        let max_val = f32::from_le_bytes(f32_buf);
        reader.read_exact(&mut f32_buf)?;
        let scale = f32::from_le_bytes(f32_buf);

        let mut u64_buf = [0u8; 8];
        reader.read_exact(&mut u64_buf)?;
        let n = u64::from_le_bytes(u64_buf) as usize;

        let mut ids = vec![0i64; n];
        for id in &mut ids {
            let mut ibuf = [0u8; 8];
            reader.read_exact(&mut ibuf)?;
            *id = i64::from_le_bytes(ibuf);
        }

        let mut vectors = vec![0.0f32; n * stored_dim];
        for value in &mut vectors {
            let mut vbuf = [0u8; 4];
            reader.read_exact(&mut vbuf)?;
            *value = f32::from_le_bytes(vbuf);
        }

        reader.read_exact(&mut u64_buf)?;
        let inverted_count = u64::from_le_bytes(u64_buf) as usize;
        let mut inverted_lists: HashMap<usize, (Vec<i64>, Vec<u8>)> =
            HashMap::with_capacity(inverted_count);
        for _ in 0..inverted_count {
            reader.read_exact(&mut u64_buf)?;
            let cluster_id = u64::from_le_bytes(u64_buf) as usize;

            reader.read_exact(&mut u64_buf)?;
            let list_n = u64::from_le_bytes(u64_buf) as usize;

            let mut list_ids = vec![0i64; list_n];
            for id in &mut list_ids {
                let mut ibuf = [0u8; 8];
                reader.read_exact(&mut ibuf)?;
                *id = i64::from_le_bytes(ibuf);
            }

            let mut list_codes = vec![0u8; list_n * stored_dim];
            reader.read_exact(&mut list_codes)?;
            inverted_lists.insert(cluster_id, (list_ids, list_codes));
        }

        let mut quantizer = ScalarQuantizer::new(stored_dim, 8);
        quantizer.min_val = min_val;
        quantizer.max_val = max_val;
        quantizer.scale = scale;
        quantizer.offset = min_val;

        let mut index = Self {
            config: IndexConfig {
                index_type: crate::api::IndexType::IvfSq8,
                metric_type,
                dim: stored_dim,
                data_type: crate::api::DataType::Float,
                params: crate::api::IndexParams::ivf(nlist, nprobe),
            },
            dim: stored_dim,
            nlist,
            nprobe,
            metric_type,
            centroids,
            inverted_lists,
            inverted_list_rows: HashMap::new(),
            quantizer,
            vectors,
            ids,
            next_id: 0,
            trained: true,
        };

        index.inverted_list_rows = index.rebuild_inverted_list_rows_from_vectors();
        index.next_id = index.ids.iter().copied().max().map_or(0, |m| m + 1);

        Ok(index)
    }

    /// Save index
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;

        let mut file = File::create(path)?;
        self.write_to(&mut file)
    }

    /// Load index
    pub fn load(path: &std::path::Path, dim: usize) -> Result<Self> {
        use std::fs::File;

        let mut file = File::open(path)?;
        Self::read_from(&mut file, dim)
    }

    /// Serialize index into an in-memory byte buffer.
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    /// Deserialize index from an in-memory byte buffer.
    pub fn deserialize_from_bytes(bytes: &[u8], dim: usize) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::read_from(&mut cursor, dim)
    }

    /// Import IVF-SQ8 index from FAISS standard "IwSq" binary.
    pub fn import_from_faiss_file(path: &str) -> Result<Self> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(b[0])
        }
        fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(i32::from_le_bytes(b))
        }
        fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        }
        fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
            Ok(i64::from_le_bytes(b))
        }
        fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        }
        fn read_f32_vec_with_count<R: Read>(r: &mut R) -> Result<Vec<f32>> {
            let n = read_i64(r)? as usize;
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(read_f32(r)?);
            }
            Ok(out)
        }
        fn metric_from_faiss_i32(v: i32) -> Result<MetricType> {
            match v {
                0 => Ok(MetricType::Ip),
                1 => Ok(MetricType::L2),
                // fallback: keep parser permissive for uncommon FAISS metrics
                _ => Ok(MetricType::L2),
            }
        }

        let mut file = File::open(path)?;

        // Top-level fourcc: IwSq
        let mut fourcc = [0u8; 4];
        file.read_exact(&mut fourcc)?;
        if &fourcc != b"IwSq" {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported top-level fourcc {:?}, expected IwSq",
                String::from_utf8_lossy(&fourcc)
            )));
        }

        // write_index_header
        let dim = read_i32(&mut file)? as usize;
        let ntotal = read_i64(&mut file)? as usize;
        let _dummy1 = read_i64(&mut file)?;
        let _dummy2 = read_i64(&mut file)?;
        let _is_trained = read_u8(&mut file)?;
        let metric_i32 = read_i32(&mut file)?;
        let metric_type = metric_from_faiss_i32(metric_i32)?;
        if metric_i32 > 1 {
            let _metric_arg = read_f32(&mut file)?;
        }

        // write_ivf_header
        let nlist = read_i64(&mut file)? as usize;
        let nprobe = read_i64(&mut file)? as usize;

        // coarse quantizer (IxF2 / IxFI)
        let mut cq_fourcc = [0u8; 4];
        file.read_exact(&mut cq_fourcc)?;
        if &cq_fourcc != b"IxF2" && &cq_fourcc != b"IxFI" {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported coarse quantizer fourcc {:?}, expected IxF2/IxFI",
                String::from_utf8_lossy(&cq_fourcc)
            )));
        }
        let cq_dim = read_i32(&mut file)? as usize;
        if cq_dim != dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "coarse dim mismatch: cq_dim={} dim={}",
                cq_dim, dim
            )));
        }
        let _cq_ntotal = read_i64(&mut file)?;
        let _cq_dummy1 = read_i64(&mut file)?;
        let _cq_dummy2 = read_i64(&mut file)?;
        let _cq_trained = read_u8(&mut file)?;
        let _cq_metric = read_i32(&mut file)?;
        let centroids = read_f32_vec_with_count(&mut file)?;
        if centroids.len() != nlist * dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "invalid centroid payload: {} != nlist*dim {}",
                centroids.len(),
                nlist * dim
            )));
        }

        // direct_map
        let dm_type = read_u8(&mut file)?;
        let dm_arr_len = read_i64(&mut file)? as usize;
        match dm_type {
            0 => {}
            1 => {
                // Array direct-map payload (i64[])
                let skip = dm_arr_len
                    .checked_mul(8)
                    .ok_or_else(|| crate::api::KnowhereError::Codec("direct_map overflow".to_string()))?;
                file.seek(SeekFrom::Current(skip as i64))?;
            }
            2 | 3 => {
                // Hashtable payload
                let hash_len = read_i64(&mut file)? as usize;
                let skip = hash_len
                    .checked_mul(16)
                    .ok_or_else(|| crate::api::KnowhereError::Codec("direct_map hash overflow".to_string()))?;
                file.seek(SeekFrom::Current(skip as i64))?;
            }
            _ => {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "unsupported direct_map type {dm_type}"
                )));
            }
        }

        // ScalarQuantizer
        let _qtype = read_i32(&mut file)?;
        let _rangestat = read_i32(&mut file)?;
        let _rangestat_arg = read_f32(&mut file)?;
        let sq_d = read_i32(&mut file)? as usize;
        let sq_code_size = read_i64(&mut file)? as usize;
        let trained_vals = read_f32_vec_with_count(&mut file)?;

        // IVFSQ fields
        let ivsc_code_size = read_i64(&mut file)? as usize;
        let _by_residual = read_u8(&mut file)?;

        // Inverted lists
        let mut il_fourcc = [0u8; 4];
        file.read_exact(&mut il_fourcc)?;
        if &il_fourcc != b"ilar" {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported invlists fourcc {:?}, expected ilar",
                String::from_utf8_lossy(&il_fourcc)
            )));
        }
        let il_nlist = read_i64(&mut file)? as usize;
        let il_code_size = read_i64(&mut file)? as usize;
        if il_nlist != nlist {
            return Err(crate::api::KnowhereError::Codec(format!(
                "invlists nlist mismatch: {} != {}",
                il_nlist, nlist
            )));
        }

        let mut list_sizes = vec![0usize; nlist];
        let mut list_type_buf = [0u8; 4];
        file.read_exact(&mut list_type_buf)?;
        if &list_type_buf == b"full" {
            let sizes_count = read_i64(&mut file)? as usize;
            if sizes_count != nlist {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "full list sizes count mismatch: {} != {}",
                    sizes_count, nlist
                )));
            }
            for slot in &mut list_sizes {
                *slot = read_i64(&mut file)? as usize;
            }
        } else if &list_type_buf == b"sprs" {
            let sparse_count = read_i64(&mut file)? as usize;
            for _ in 0..sparse_count {
                let list_id = read_i64(&mut file)? as usize;
                let list_size = read_i64(&mut file)? as usize;
                if list_id >= nlist {
                    return Err(crate::api::KnowhereError::Codec(format!(
                        "sparse list id out of range: {} >= {}",
                        list_id, nlist
                    )));
                }
                list_sizes[list_id] = list_size;
            }
        } else {
            // Legacy simplified layout: rewind and read sizes directly as i64[nlist]
            file.seek(SeekFrom::Current(-4))?;
            let sizes_count = read_i64(&mut file)? as usize;
            if sizes_count == nlist {
                for slot in &mut list_sizes {
                    *slot = read_i64(&mut file)? as usize;
                }
            } else {
                list_sizes[0] = sizes_count;
                for slot in list_sizes.iter_mut().skip(1) {
                    *slot = read_i64(&mut file)? as usize;
                }
            }
        }

        let mut inverted_lists: HashMap<usize, (Vec<i64>, Vec<u8>)> =
            HashMap::with_capacity(nlist);
        let mut total_ids = 0usize;
        for (list_id, &sz) in list_sizes.iter().enumerate() {
            if sz == 0 {
                continue;
            }
            let mut codes = vec![0u8; sz * il_code_size];
            file.read_exact(&mut codes)?;
            let mut ids = vec![0i64; sz];
            for id in &mut ids {
                *id = read_i64(&mut file)?;
            }
            total_ids += sz;
            inverted_lists.insert(list_id, (ids, codes));
        }

        // Build quantizer params from serialized SQ training payload.
        let mut quantizer = ScalarQuantizer::new(dim, 8);
        let (min_val, max_val) = if sq_d == dim && trained_vals.len() >= 2 * dim {
            let mins = &trained_vals[..dim];
            let maxs = &trained_vals[dim..2 * dim];
            (
                mins.iter().copied().fold(f32::INFINITY, f32::min),
                maxs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        } else if !trained_vals.is_empty() {
            (
                trained_vals.iter().copied().fold(f32::INFINITY, f32::min),
                trained_vals
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
            )
        } else {
            (0.0, 1.0)
        };
        let safe_min = if min_val.is_finite() { min_val } else { 0.0 };
        let safe_max = if max_val.is_finite() && max_val > safe_min {
            max_val
        } else {
            safe_min + 1.0
        };
        quantizer.min_val = safe_min;
        quantizer.max_val = safe_max;
        quantizer.offset = safe_min;
        quantizer.scale = 255.0 / (safe_max - safe_min);

        let mut cluster_ids: Vec<usize> = inverted_lists.keys().copied().collect();
        cluster_ids.sort_unstable();
        let mut all_ids = Vec::with_capacity(total_ids);
        for cluster_id in &cluster_ids {
            if let Some((ids, _)) = inverted_lists.get(cluster_id) {
                all_ids.extend_from_slice(ids);
            }
        }
        let inverted_list_rows = Self::synthetic_inverted_list_rows(&inverted_lists);

        let mut index = Self {
            config: IndexConfig {
                index_type: crate::api::IndexType::IvfSq8,
                metric_type,
                dim,
                data_type: crate::api::DataType::Float,
                params: crate::api::IndexParams::ivf(nlist, nprobe.max(1)),
            },
            dim,
            nlist,
            nprobe: nprobe.max(1),
            metric_type,
            centroids,
            inverted_lists,
            inverted_list_rows,
            quantizer,
            vectors: vec![0.0; ntotal * dim],
            ids: all_ids,
            next_id: 0,
            trained: true,
        };
        index.next_id = index.ids.iter().copied().max().map_or(0, |m| m + 1);

        if il_code_size != dim && sq_code_size != dim && ivsc_code_size != dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unexpected code size: il={} sq={} ivsc={} dim={}",
                il_code_size, sq_code_size, ivsc_code_size, dim
            )));
        }

        Ok(index)
    }
}

/// Index trait implementation for IvfSq8Index
///
/// This wrapper enables IvfSq8Index to be used through the unified Index trait interface,
/// allowing consistent access to advanced features (AnnIterator, get_vector_by_ids, etc.)
/// across all index types.
impl IndexTrait for IvfSq8Index {
    fn index_type(&self) -> &str {
        "IVF-SQ8"
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
        let vectors = dataset.vectors();
        self.train(vectors)
            .map(|_| ())
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
            nprobe: self.nprobe,
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
        if !self.trained {
            return Err(IndexError::Unsupported("index not trained".to_string()));
        }
        if self.ids.is_empty() {
            return Ok(IndexSearchResult::new(vec![], vec![], 0.0));
        }

        let n_queries = vectors.len() / self.dim;
        if n_queries * self.dim != vectors.len() {
            return Err(IndexError::Unsupported("query dimension mismatch".to_string()));
        }
        if top_k == 0 {
            return Ok(IndexSearchResult::new(vec![], vec![], 0.0));
        }

        let nprobe = self.nprobe.min(self.nlist);
        let mut all_ids = vec![-1; n_queries * top_k];
        let mut all_dists = vec![f32::MAX; n_queries * top_k];

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &vectors[q_start..q_start + self.dim];
            let clusters = self.search_clusters(query_vec, nprobe);
            let mut q_residual_buf = vec![0.0f32; self.dim];
            let mut q_precomputed_buf = vec![0i16; self.dim];
            let mut merged = TopKAccumulator::new(top_k);

            for cluster_id in clusters {
                merged.merge(self.scan_cluster_with_buf(
                    cluster_id,
                    query_vec,
                    top_k,
                    &mut q_residual_buf,
                    &mut q_precomputed_buf,
                    Some(bitset),
                    self.inverted_list_rows.get(&cluster_id).map(|rows| rows.as_slice()),
                ));
            }

            let hits = merged.into_hits();
            let offset = q_idx * top_k;
            for (i, hit) in hits.into_iter().enumerate().take(top_k) {
                all_ids[offset + i] = hit.id;
                all_dists[offset + i] = hit.dist;
            }
        }

        Ok(IndexSearchResult::new(
            all_ids,
            all_dists,
            0.0,
        ))
    }

    fn range_search(
        &self,
        query: &Dataset,
        radius: f32,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let vectors = query.vectors();
        if !self.trained {
            return Err(IndexError::Unsupported("index not trained".to_string()));
        }
        if self.ids.is_empty() {
            return Ok(IndexSearchResult::new(vec![], vec![], 0.0));
        }

        let n_queries = vectors.len() / self.dim;
        if n_queries * self.dim != vectors.len() {
            return Err(IndexError::Unsupported("query dimension mismatch".to_string()));
        }

        let nprobe = self.nprobe.min(self.nlist);
        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &vectors[q_start..q_start + self.dim];
            let clusters = self.search_clusters(query_vec, nprobe);
            let mut q_residual_buf = vec![0.0f32; self.dim];
            let mut q_precomputed_buf = vec![0i16; self.dim];

            for cluster_id in clusters {
                let Some((cluster_ids, _)) = self.inverted_lists.get(&cluster_id) else {
                    continue;
                };
                if cluster_ids.is_empty() {
                    continue;
                }
                let cluster_all = cluster_ids.len();
                let hits = self
                    .scan_cluster_with_buf(
                        cluster_id,
                        query_vec,
                        cluster_all,
                        &mut q_residual_buf,
                        &mut q_precomputed_buf,
                        None,
                        None,
                    )
                    .into_hits();

                for hit in hits {
                    if hit.dist <= radius {
                        all_ids.push(hit.id);
                        all_dists.push(hit.dist);
                    }
                }
            }
        }

        Ok(IndexSearchResult::new(all_ids, all_dists, 0.0))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        let path = std::path::Path::new(path);
        self.save(path)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        let loaded = IvfSq8Index::load(std::path::Path::new(path), self.dim)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn has_raw_data(&self) -> bool {
        false
    }

    fn get_vector_by_ids(&self, _ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        Err(IndexError::Unsupported(
            "get_vector_by_ids not supported for IVF-SQ8 (lossy compression)".into(),
        ))
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        _bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        // IVF-SQ8 doesn't support native iterator, fallback to search
        // Use a large top_k to get all potential results
        let top_k = self.ids.len().max(1000);
        let vectors = query.vectors();

        let req = SearchRequest {
            top_k,
            nprobe: self.nprobe,
            filter: None,
            params: None,
            radius: None,
        };

        let api_result = self
            .search(vectors, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        // Create simple iterator from search results
        // Note: This is not a true streaming iterator, but provides compatibility
        let results: Vec<(i64, f32)> = api_result
            .ids
            .into_iter()
            .zip(api_result.distances)
            .collect();

        Ok(Box::new(IvfSq8AnnIterator::new(results)))
    }
}

/// Simple ANN iterator for IVF-SQ8 (fallback implementation)
pub struct IvfSq8AnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl IvfSq8AnnIterator {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl AnnIterator for IvfSq8AnnIterator {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexParams, IndexType, MetricType, SearchRequest};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_ivf_sq8_new() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::default(),
        };

        let index = IvfSq8Index::new(&config).unwrap();
        assert_eq!(index.ntotal(), 0);
        assert!(!index.trained);
    }

    #[test]
    fn test_ivf_sq8_save_load() {
        let dim = 32usize;
        let n = 500usize;
        let nlist = 16usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.r#gen::<f32>()).collect();

        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf(nlist, 16),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let path = std::path::Path::new("/tmp/ivf_sq8_test.bin");
        let _ = std::fs::remove_file(path);
        index.save(path).unwrap();

        let loaded = IvfSq8Index::load(path, dim).unwrap();

        let query = &vectors[0..dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 16,
            filter: None,
            params: None,
            radius: None,
        };
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

        let hits = gt_top10
            .iter()
            .filter(|id| result.ids.iter().take(10).any(|rid| rid == *id))
            .count();
        let recall = hits as f64 / 10.0;
        assert!(
            recall >= 0.8,
            "save/load ivf-sq8 recall@10 too low: {:.3}",
            recall
        );
    }

    #[test]
    fn test_ivf_sq8_serialize_deserialize_bytes_roundtrip() {
        let dim = 16usize;
        let n = 320usize;
        let nlist = 16usize;
        let mut rng = StdRng::seed_from_u64(42);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.r#gen::<f32>()).collect();

        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf(nlist, 8),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let bytes = index.serialize_to_bytes().unwrap();
        let restored = IvfSq8Index::deserialize_from_bytes(&bytes, dim).unwrap();

        let query = &vectors[..dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };

        let before = index.search(query, &req).unwrap();
        let after = restored.search(query, &req).unwrap();
        assert_eq!(before.ids, after.ids);
        assert_eq!(before.distances.len(), after.distances.len());
        for (lhs, rhs) in before.distances.iter().zip(after.distances.iter()) {
            assert!((lhs - rhs).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ivf_sq8_train_add_search() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf_sq8(4, 2),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();

        // Training data: 4 clusters
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 10.1,
        ];

        let trained_count = index.train(&vectors).unwrap();
        assert_eq!(trained_count, 4);

        // Add vectors
        let add_count = index.add(&vectors, None).unwrap();
        assert_eq!(add_count, 4);
        assert_eq!(index.ntotal(), 4);

        // Search
        let query = vec![0.05, 0.05, 0.05, 0.05];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 2);
        // Should find the closest vectors (ids 0 and 1)
        assert!(result.ids[0] == 0 || result.ids[0] == 1);
    }

    #[test]
    fn test_ivf_sq8_range_search_returns_hits_within_bounds() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf_sq8(4, 2),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, 0.0, //
            4.0, 0.0, 0.0, 0.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let hits = index.range_search(&query, 0.0, 1.5);

        assert!(!hits.is_empty());
        assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
        assert!(hits.iter().all(|(_, dist)| *dist >= 0.0 && *dist <= 1.5));
    }

    #[test]
    fn test_ivf_sq8_bitset_uses_internal_rows_not_external_ids() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf_sq8(4, 2),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, //
            0.1, 0.1, 0.1, 0.1, //
            5.0, 5.0, 5.0, 5.0, //
            6.0, 6.0, 6.0, 6.0,
        ];
        let ids = vec![100, 101, 102, 103];

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = Dataset::from_vectors(vec![0.01, 0.01, 0.01, 0.01], 4);
        let mut bitset = BitsetView::new(4);
        bitset.set(0, true);

        let result =
            crate::index::Index::search_with_bitset(&index, &query, 4, &bitset).unwrap();
        assert!(
            !result.ids.contains(&100),
            "bitset row 0 should filter the first internal row even when external id is 100"
        );
        assert!(
            result.ids.contains(&101),
            "bitset should only filter the targeted internal row, not nearby unfiltered rows"
        );
    }

    #[test]
    fn test_ivf_sq8_get_vector_by_ids_unsupported() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf_sq8(4, 2),
        };

        let mut index = IvfSq8Index::new(&config).unwrap();
        let vectors = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        ];
        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        assert!(!crate::index::Index::has_raw_data(&index));
        assert!(crate::index::Index::get_vector_by_ids(&index, &[0]).is_err());
    }

    #[test]
    fn test_ivf_sq8_index_type() {
        let config = IndexConfig {
            index_type: IndexType::IvfSq8,
            metric_type: MetricType::L2,
            dim: 8,
            data_type: crate::api::DataType::Float,
            params: IndexParams::ivf_sq8(10, 3),
        };

        assert_eq!(config.index_type, IndexType::IvfSq8);
        assert_eq!(config.params.nlist, Some(10));
        assert_eq!(config.params.nprobe, Some(3));
    }

    #[test]
    fn test_ivf_sq8_from_str() {
        assert_eq!("ivf_sq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("ivf-sq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("ivfsq8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
        assert_eq!("IVF_SQ8".parse::<IndexType>().ok(), Some(IndexType::IvfSq8));
    }

    #[test]
    fn test_import_faiss_ivfsq8_via_raw_bytes() {
        fn push_u8(buf: &mut Vec<u8>, v: u8) {
            buf.push(v);
        }
        fn push_i32(buf: &mut Vec<u8>, v: i32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fn push_u32(buf: &mut Vec<u8>, v: u32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fn push_i64(buf: &mut Vec<u8>, v: i64) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fn push_f32(buf: &mut Vec<u8>, v: f32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let dim = 4usize;
        let nlist = 2usize;
        let nprobe = 2usize;
        let ntotal = 4usize;
        let il_code_size = dim;

        let mut blob = Vec::new();

        // Top-level IwSq.
        blob.extend_from_slice(b"IwSq");
        push_i32(&mut blob, dim as i32);
        push_i64(&mut blob, ntotal as i64);
        push_i64(&mut blob, 0);
        push_i64(&mut blob, 0);
        push_u8(&mut blob, 1); // trained
        push_i32(&mut blob, 1); // METRIC_L2
        push_i64(&mut blob, nlist as i64);
        push_i64(&mut blob, nprobe as i64);

        // Coarse quantizer IxF2.
        blob.extend_from_slice(b"IxF2");
        push_i32(&mut blob, dim as i32);
        push_i64(&mut blob, nlist as i64);
        push_i64(&mut blob, 0);
        push_i64(&mut blob, 0);
        push_u8(&mut blob, 1);
        push_i32(&mut blob, 1);
        push_i64(&mut blob, (nlist * dim) as i64);
        for i in 0..(nlist * dim) {
            push_f32(&mut blob, i as f32 * 0.1);
        }

        // direct_map: NoMap.
        push_u8(&mut blob, 0);
        push_i64(&mut blob, 0);

        // ScalarQuantizer.
        push_i32(&mut blob, 0); // qtype QT_8bit
        push_i32(&mut blob, 0); // rangestat RS_minmax
        push_f32(&mut blob, 0.0); // rangestat_arg
        push_i32(&mut blob, dim as i32); // sq_d
        push_i64(&mut blob, dim as i64); // sq_code_size
        push_i64(&mut blob, (2 * dim) as i64); // trained_count
        for _ in 0..dim {
            push_f32(&mut blob, 0.0); // mins
        }
        for _ in 0..dim {
            push_f32(&mut blob, 1.0); // maxs
        }

        // IVFScalarQuantizer fields.
        push_i64(&mut blob, dim as i64); // ivsc_code_size
        push_u8(&mut blob, 1); // by_residual

        // Inverted lists.
        blob.extend_from_slice(b"ilar");
        push_i64(&mut blob, nlist as i64);
        push_i64(&mut blob, il_code_size as i64);
        blob.extend_from_slice(b"full");
        push_i64(&mut blob, nlist as i64); // sizes_count
        push_i64(&mut blob, 2);
        push_i64(&mut blob, 2);

        // list 0: 2 codes (2*dim bytes), 2 ids
        blob.extend_from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8]);
        push_i64(&mut blob, 10);
        push_i64(&mut blob, 11);
        // list 1
        blob.extend_from_slice(&[9u8, 10, 11, 12, 13, 14, 15, 16]);
        push_i64(&mut blob, 12);
        push_i64(&mut blob, 13);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &blob).unwrap();

        let imported =
            IvfSq8Index::import_from_faiss_file(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(imported.dim, 4);
        assert_eq!(imported.nlist, 2);
        assert_eq!(imported.nprobe, 2);
        assert!(imported.trained);
        let list0 = imported.inverted_lists.get(&0).unwrap();
        let list1 = imported.inverted_lists.get(&1).unwrap();
        assert_eq!(list0.0, vec![10, 11]);
        assert_eq!(list1.0, vec![12, 13]);
        assert_eq!(list0.1.len(), 8);
        assert_eq!(list1.1.len(), 8);
    }
}

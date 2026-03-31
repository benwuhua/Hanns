//! IVF-PQ Index Implementation
//!
//! Inverted File Index with Product Quantization.
//! Uses coarse quantizer (IVF) + fine quantizer (PQ) for compressed storage
//! and fast Asymmetric Distance Computation (ADC).
//!
//! OPT-003: 内存布局优化 - 使用 Vec 替代 HashMap
#![allow(dead_code)] // Some hot-path helpers are only reached from tests/feature-gated search lanes.

use crate::api::{
    DataType, IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult,
};
use crate::bitset::BitsetView;
use crate::quantization::kmeans::KMeansMetric;
use crate::quantization::opq::{OPQConfig, OptimizedProductQuantizer};
use crate::quantization::pq::{PQConfig, ProductQuantizer};
use crate::quantization::KMeans;
use crate::simd::{dot_product_f32, ip_batch_4, l2_batch_4_ptr, l2_distance_sq};
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone, Default, PartialEq)]
struct CompactInvertedLists {
    all_codes: Vec<u8>,
    all_ids: Vec<i64>,
    offsets: Vec<(usize, usize)>,
}

/// IVF-PQ Index
///
/// OPT-003: 内存布局优化 (HashMap → Vec)
/// - 使用 Vec<Vec<i64>> 替代 HashMap 存储 ID 列表
/// - 使用 Vec<Vec<u8>> 替代 HashMap 存储 PQ code 列表
/// - 优势：O(1) 索引，无哈希开销，缓存友好
#[allow(dead_code)]
pub struct IvfPqIndex {
    config: IndexConfig,
    dim: usize,
    nlist: usize,         // Number of clusters
    nprobe: usize,        // Number of clusters to search
    m: usize,             // Number of sub-quantizers
    nbits_per_idx: usize, // Bits per sub-vector

    /// Cluster centroids (coarse quantizer)
    centroids: Vec<f32>,
    /// Product quantizer (fine quantizer on residuals)
    pq: ProductQuantizer,
    /// Optional OPQ quantizer for rotated residual encoding
    opq: Option<OptimizedProductQuantizer>,
    /// Optional PQ centroids imported from FAISS binary files.
    /// When present, search uses this table directly instead of internal PQ training state.
    imported_pq_centroids: Option<Vec<f32>>,
    /// Enable OPQ path (default: true)
    use_opq: bool,
    /// Inverted lists (Vec 替代 HashMap - 性能优化)
    invlist_ids: Vec<Vec<i64>>, // [nlist] 个 ID 列表
    invlist_codes: Vec<Vec<u8>>, // [nlist] 个 PQ code 列表 (连续存储)
    /// Search-time compact storage for sequential scan.
    compact_invlists: CompactInvertedLists,
    /// All vectors (kept for save/load reconstruction)
    vectors: Vec<f32>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IvfPqHotPathAudit {
    pub dim: usize,
    pub coarse_centroid_count: usize,
    pub pq_subquantizers: usize,
    pub encoded_vector_count: usize,
    pub code_size_bytes: usize,
    pub total_code_bytes: usize,
    pub list_lengths: Vec<usize>,
    pub centroids: Vec<f32>,
}

impl IvfPqIndex {
    #[inline]
    fn compare_cluster_score(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
        a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
    }

    #[inline]
    fn compare_candidate_score(a: &(i64, f32), b: &(i64, f32)) -> std::cmp::Ordering {
        a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
    }

    #[inline]
    fn centroid_score(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => l2_distance_sq(a, b),
            MetricType::Ip | MetricType::Cosine => -dot_product_f32(a, b),
        }
    }

    fn score_all_centroids(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let mut cluster_dists = Vec::with_capacity(self.nlist);
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => unsafe {
                let query_ptr = query.as_ptr();
                let base_ptr = self.centroids.as_ptr();
                let mut c = 0usize;
                while c + 4 <= self.nlist {
                    let dists =
                        l2_batch_4_ptr(query_ptr, base_ptr.add(c * self.dim), self.dim, self.dim);
                    cluster_dists.push((c, dists[0]));
                    cluster_dists.push((c + 1, dists[1]));
                    cluster_dists.push((c + 2, dists[2]));
                    cluster_dists.push((c + 3, dists[3]));
                    c += 4;
                }
                while c < self.nlist {
                    let dist =
                        l2_distance_sq(query, &self.centroids[c * self.dim..(c + 1) * self.dim]);
                    cluster_dists.push((c, dist));
                    c += 1;
                }
            },
            MetricType::Ip | MetricType::Cosine => {
                let mut c = 0usize;
                while c + 4 <= self.nlist {
                    let dots = ip_batch_4(
                        query,
                        &self.centroids[c * self.dim..(c + 1) * self.dim],
                        &self.centroids[(c + 1) * self.dim..(c + 2) * self.dim],
                        &self.centroids[(c + 2) * self.dim..(c + 3) * self.dim],
                        &self.centroids[(c + 3) * self.dim..(c + 4) * self.dim],
                    );
                    cluster_dists.push((c, -dots[0]));
                    cluster_dists.push((c + 1, -dots[1]));
                    cluster_dists.push((c + 2, -dots[2]));
                    cluster_dists.push((c + 3, -dots[3]));
                    c += 4;
                }
                while c < self.nlist {
                    let dist =
                        -dot_product_f32(query, &self.centroids[c * self.dim..(c + 1) * self.dim]);
                    cluster_dists.push((c, dist));
                    c += 1;
                }
            }
        }
        cluster_dists
    }

    #[inline]
    fn dist_in_range(&self, dist: f32, radius: f32, range_filter: f32) -> bool {
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => radius <= dist && dist <= range_filter,
            MetricType::Ip | MetricType::Cosine => range_filter <= dist && dist <= radius,
        }
    }

    #[inline]
    fn subspace_score(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() == 2 && b.len() == 2 {
            return self.subspace_score_dim2(a, b);
        }
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| {
                    let d = x - y;
                    d * d
                })
                .sum(),
            MetricType::Ip | MetricType::Cosine => {
                -a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>()
            }
        }
    }

    #[inline]
    fn subspace_score_dim2(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => {
                let d0 = a[0] - b[0];
                let d1 = a[1] - b[1];
                d0 * d0 + d1 * d1
            }
            MetricType::Ip | MetricType::Cosine => -(a[0] * b[0] + a[1] * b[1]),
        }
    }

    #[inline]
    fn coarse_distance_bias(&self, query: &[f32], centroid: &[f32]) -> f32 {
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => 0.0,
            MetricType::Ip | MetricType::Cosine => -dot_product_f32(query, centroid),
        }
    }

    #[inline]
    fn metric_to_byte(metric: MetricType) -> u8 {
        match metric {
            MetricType::L2 => 0,
            MetricType::Ip => 1,
            MetricType::Cosine => 2,
            MetricType::Hamming => 3,
        }
    }

    #[inline]
    fn metric_from_byte(byte: u8) -> MetricType {
        match byte {
            1 => MetricType::Ip,
            2 => MetricType::Cosine,
            3 => MetricType::Hamming,
            _ => MetricType::L2,
        }
    }

    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let nlist = config.params.nlist.unwrap_or(100);
        let nprobe = config.params.nprobe.unwrap_or(8);
        let m = config.params.m.unwrap_or(8);
        let nbits = config.params.nbits_per_idx.unwrap_or(8);

        let pq_config = PQConfig::new(config.dim, m, nbits);
        let pq = ProductQuantizer::new(pq_config);
        let use_opq = false; // Disabled: power-iteration Procrustes diverges on SIFT-1M (plain PQ > OPQ)
        let opq = if use_opq {
            let mut opq_config = OPQConfig::new(config.dim, m, nbits);
            opq_config.niter = 1;
            opq_config.random_rotation = true;
            Some(OptimizedProductQuantizer::new(opq_config)?)
        } else {
            None
        };

        // 预分配 Vec 容量（性能优化）
        let invlist_ids = (0..nlist).map(|_| Vec::new()).collect();
        let invlist_codes = (0..nlist).map(|_| Vec::new()).collect();

        Ok(Self {
            config: config.clone(),
            dim: config.dim,
            nlist,
            nprobe,
            m,
            nbits_per_idx: nbits,
            centroids: Vec::new(),
            pq,
            opq,
            imported_pq_centroids: None,
            use_opq,
            invlist_ids,
            invlist_codes,
            compact_invlists: CompactInvertedLists {
                offsets: vec![(0, 0); nlist],
                ..Default::default()
            },
            vectors: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }

    /// Train: k-means clustering for coarse quantizer + PQ training on residuals
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        #[cfg(feature = "metrics")]
        let _timer = {
            let m = crate::metrics::init_metrics();
            crate::metrics::Timer::new(m.train_duration_seconds.clone())
        };

        let n = vectors.len() / self.dim;
        if n * self.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        // Step 1: Train coarse quantizer (parallel KMeans)
        self.train_ivf_centroids(vectors, n)?;

        // Clear inverted lists (already initialized in new())
        for i in 0..self.nlist {
            self.invlist_ids[i].clear();
            self.invlist_codes[i].clear();
        }

        // Step 2: Compute residuals for all training vectors
        let mut residuals = Vec::with_capacity(vectors.len());
        for i in 0..n {
            let start = i * self.dim;
            let vector = &vectors[start..start + self.dim];

            // Find nearest centroid
            let cluster = self.find_nearest_centroid(vector);

            // Compute residual: vector - centroid
            for (j, &value) in vector.iter().enumerate().take(self.dim) {
                residuals.push(value - self.centroids[cluster * self.dim + j]);
            }
        }

        // Step 3: Train fine quantizer on residuals
        self.train_fine_quantizer(n, &residuals)?;

        self.trained = true;
        tracing::info!(
            "Trained IVF-PQ with {} clusters, m={}, nbits={}",
            self.nlist,
            self.m,
            self.nbits_per_idx
        );
        Ok(())
    }

    #[inline]
    fn opq_enabled(&self) -> bool {
        self.use_opq
            && self
                .opq
                .as_ref()
                .map(|opq| opq.is_trained())
                .unwrap_or(false)
    }

    #[inline]
    fn active_code_size(&self) -> usize {
        if self.opq_enabled() {
            self.opq
                .as_ref()
                .map(|opq| opq.code_size())
                .unwrap_or(self.pq.code_size())
        } else {
            self.pq.code_size()
        }
    }

    fn rebuild_compact_invlists(&mut self) {
        let code_size = self.active_code_size();
        let total_codes = self.invlist_ids.iter().map(Vec::len).sum::<usize>();
        let total_code_bytes = self.invlist_codes.iter().map(Vec::len).sum::<usize>();

        let mut offsets = Vec::with_capacity(self.nlist);
        let mut all_ids = Vec::with_capacity(total_codes);
        let mut all_codes = Vec::with_capacity(total_code_bytes);

        let mut start = 0usize;
        for cluster in 0..self.nlist {
            let ids = &self.invlist_ids[cluster];
            let codes = &self.invlist_codes[cluster];
            debug_assert_eq!(codes.len(), ids.len() * code_size);
            offsets.push((start, ids.len()));
            all_ids.extend_from_slice(ids);
            all_codes.extend_from_slice(codes);
            start += ids.len();
        }

        self.compact_invlists = CompactInvertedLists {
            all_codes,
            all_ids,
            offsets,
        };
    }

    #[inline]
    fn cluster_scan_slice(&self, cluster: usize) -> (&[i64], &[u8]) {
        if let Some(&(offset, count)) = self.compact_invlists.offsets.get(cluster) {
            let code_size = self.active_code_size();
            let code_start = offset * code_size;
            let code_end = (offset + count) * code_size;
            if count == self.invlist_ids[cluster].len()
                && code_end.saturating_sub(code_start) == self.invlist_codes[cluster].len()
                && offset + count <= self.compact_invlists.all_ids.len()
                && code_end <= self.compact_invlists.all_codes.len()
            {
                let ids = &self.compact_invlists.all_ids[offset..offset + count];
                let codes = &self.compact_invlists.all_codes[code_start..code_end];
                return (ids, codes);
            }
        }

        (&self.invlist_ids[cluster], &self.invlist_codes[cluster])
    }

    fn train_fine_quantizer(&mut self, n: usize, residuals: &[f32]) -> Result<()> {
        if self.use_opq {
            if let Some(opq) = self.opq.as_mut() {
                return opq.train(n, residuals);
            }
        }
        self.pq.train(n, residuals)
    }

    fn encode_residual(&self, residual: &[f32]) -> Result<Vec<u8>> {
        if self.opq_enabled() {
            if let Some(opq) = self.opq.as_ref() {
                return opq.encode(residual);
            }
        }
        self.pq.encode(residual)
    }

    fn train_ivf_centroids(&mut self, vectors: &[f32], _n: usize) -> Result<()> {
        let mut km = KMeans::new(self.nlist, self.dim);
        if matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine) {
            km = km.with_metric(KMeansMetric::InnerProduct);
        }
        km.train(vectors);
        self.centroids = km.centroids().to_vec();
        Ok(())
    }

    /// Add vectors to index
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        #[cfg(feature = "metrics")]
        let _timer = {
            let m = crate::metrics::init_metrics();
            crate::metrics::Timer::new(m.add_duration_seconds.clone())
        };

        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        #[cfg(feature = "parallel")]
        {
            return self.add_parallel(vectors, ids, rayon::current_num_threads());
        }

        #[cfg(not(feature = "parallel"))]
        {
            let n = vectors.len() / self.dim;

            for i in 0..n {
                let start = i * self.dim;
                let end = start + self.dim;
                let vector = &vectors[start..end];

                // Find nearest centroid
                let cluster = self.find_nearest_centroid(vector);

                // Compute residual
                let mut residual = vec![0.0f32; self.dim];
                for j in 0..self.dim {
                    residual[j] = vector[j] - self.centroids[cluster * self.dim + j];
                }

                // PQ-encode the residual
                let code = self.encode_residual(&residual)?;

                let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
                self.next_id += 1;

                // Add to inverted lists (Vec optimization)
                self.invlist_ids[cluster].push(id);
                self.invlist_codes[cluster].extend_from_slice(&code);

                // Store original vector for save/load
                self.ids.push(id);
                self.vectors.extend_from_slice(vector);
            }

            self.rebuild_compact_invlists();
            tracing::debug!("Added {} vectors to IVF-PQ", n);
            return Ok(n);
        }
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
                "index must be trained first".to_string(),
            ));
        }

        let n = vectors.len() / self.dim;
        let dim = self.dim;
        let nlist = self.nlist;
        let metric_type = self.config.metric_type;
        let centroids = self.centroids.clone();

        // Parallel: assign vectors to clusters and PQ-encode residuals
        let assignments: Vec<(usize, i64, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let vector = &vectors[start..start + dim];

                // Find nearest centroid
                let mut min_dist = f32::MAX;
                let mut cluster = 0;
                for c in 0..nlist {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let dist = match metric_type {
                        MetricType::L2 | MetricType::Hamming => l2_distance_sq(vector, centroid),
                        MetricType::Ip | MetricType::Cosine => -dot_product_f32(vector, centroid),
                    };
                    if dist < min_dist {
                        min_dist = dist;
                        cluster = c;
                    }
                }

                // Compute residual
                let mut residual = vec![0.0f32; dim];
                for j in 0..dim {
                    residual[j] = vector[j] - centroids[cluster * dim + j];
                }

                // PQ-encode residual
                let code = self.encode_residual(&residual).unwrap();

                let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
                (cluster, id, code)
            })
            .collect();

        // Collect by cluster (using Vec for direct indexing)
        let mut cluster_ids: Vec<Vec<i64>> = (0..nlist).map(|_| Vec::new()).collect();
        let mut cluster_codes: Vec<Vec<u8>> = (0..nlist).map(|_| Vec::new()).collect();

        for (cluster, id, code) in assignments {
            cluster_ids[cluster].push(id);
            cluster_codes[cluster].extend_from_slice(&code);
        }

        // Merge into inverted lists
        for cluster in 0..nlist {
            self.invlist_ids[cluster].extend(&cluster_ids[cluster]);
            self.invlist_codes[cluster].extend(&cluster_codes[cluster]);
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

        self.rebuild_compact_invlists();
        tracing::debug!("Added {} vectors to IVF-PQ (parallel)", n);
        Ok(n)
    }

    pub fn hot_path_audit(&self) -> IvfPqHotPathAudit {
        IvfPqHotPathAudit {
            dim: self.dim,
            coarse_centroid_count: self.centroids.len() / self.dim,
            pq_subquantizers: self.pq.config().m,
            encoded_vector_count: self.ids.len(),
            code_size_bytes: self.pq.code_size(),
            total_code_bytes: self.invlist_codes.iter().map(Vec::len).sum(),
            list_lengths: self.invlist_ids.iter().map(Vec::len).collect(),
            centroids: self.centroids.clone(),
        }
    }

    pub fn coarse_probe_order(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        if query.len() != self.dim || self.centroids.is_empty() || nprobe == 0 {
            return Vec::new();
        }

        let cluster_dists = self.score_all_centroids(query);
        Self::select_top_clusters(cluster_dists, nprobe.min(self.nlist))
            .into_iter()
            .map(|(cluster, _)| cluster)
            .collect()
    }

    fn select_top_clusters(mut scored: Vec<(usize, f32)>, limit: usize) -> Vec<(usize, f32)> {
        if scored.is_empty() || limit == 0 {
            return Vec::new();
        }
        if limit >= scored.len() {
            scored.sort_by(Self::compare_cluster_score);
            return scored;
        }

        let nth = limit - 1;
        scored.select_nth_unstable_by(nth, Self::compare_cluster_score);
        scored.truncate(limit);
        scored.sort_by(Self::compare_cluster_score);
        scored
    }

    fn truncate_top_candidates(candidates: &mut Vec<(i64, f32)>, limit: usize) {
        if candidates.is_empty() || limit == 0 {
            candidates.clear();
            return;
        }
        if limit >= candidates.len() {
            candidates.sort_by(Self::compare_candidate_score);
            return;
        }

        let nth = limit - 1;
        candidates.select_nth_unstable_by(nth, Self::compare_candidate_score);
        candidates.truncate(limit);
        candidates.sort_by(Self::compare_candidate_score);
    }

    #[inline]
    fn adc_query_for_metric<'a>(&self, query: &'a [f32], query_residual: &'a [f32]) -> &'a [f32] {
        match self.config.metric_type {
            MetricType::Ip | MetricType::Cosine => query,
            MetricType::L2 | MetricType::Hamming => query_residual,
        }
    }

    #[inline]
    fn rotated_adc_query(
        &self,
        opq: &OptimizedProductQuantizer,
        query: &[f32],
        query_residual: &[f32],
    ) -> Vec<f32> {
        opq.apply_rotation_single(self.adc_query_for_metric(query, query_residual))
    }

    /// Precompute distance table for the vector used by ADC.
    /// Returns table[sub_q][centroid_idx] = subspace score between ADC query and PQ centroid.
    fn precompute_distance_table(&self, adc_query: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.dim / self.m;
        let ksub = 1usize << self.nbits_per_idx;
        let m = self.m;

        let mut table = Vec::with_capacity(m);
        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &adc_query[q_sub_offset..q_sub_offset + sub_dim];

            let centroids = self
                .pq_subquantizer_centroids(sub_q)
                .expect("pq centroids unavailable for subquantizer");
            let mut dists = Vec::with_capacity(ksub);

            for c in 0..ksub {
                let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                dists.push(self.subspace_score(query_sub, centroid));
            }
            table.push(dists);
        }
        table
    }

    fn precompute_distance_table_flat(&self, adc_query: &[f32]) -> Vec<f32> {
        let sub_dim = self.dim / self.m;
        let ksub = 1usize << self.nbits_per_idx;
        let m = self.m;

        let mut table = vec![0.0f32; m * ksub];
        if sub_dim == 2 {
            for sub_q in 0..m {
                let q_sub_offset = sub_q * 2;
                let query_sub = &adc_query[q_sub_offset..q_sub_offset + 2];

                let centroids = self
                    .pq_subquantizer_centroids(sub_q)
                    .expect("pq centroids unavailable for subquantizer");
                for c in 0..ksub {
                    let centroid = &centroids[c * 2..c * 2 + 2];
                    table[sub_q * ksub + c] = self.subspace_score_dim2(query_sub, centroid);
                }
            }
            return table;
        }

        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &adc_query[q_sub_offset..q_sub_offset + sub_dim];

            let centroids = self
                .pq_subquantizer_centroids(sub_q)
                .expect("pq centroids unavailable for subquantizer");
            for c in 0..ksub {
                let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                table[sub_q * ksub + c] = self.subspace_score(query_sub, centroid);
            }
        }
        table
    }

    #[inline]
    fn pq_subquantizer_centroids(&self, sub_q: usize) -> Option<&[f32]> {
        if sub_q >= self.m {
            return None;
        }
        if let Some(imported) = self.imported_pq_centroids.as_ref() {
            let ksub = 1usize << self.nbits_per_idx;
            let sub_dim = self.dim / self.m;
            let off = sub_q * ksub * sub_dim;
            return Some(&imported[off..off + ksub * sub_dim]);
        }
        self.pq.get_centroids(sub_q)
    }

    /// Precompute ADC distance table using OPQ centroids and rotated ADC query.
    fn precompute_distance_table_opq(
        &self,
        opq: &OptimizedProductQuantizer,
        rotated_adc_query: &[f32],
    ) -> Vec<Vec<f32>> {
        let m = opq.config().m;
        let ksub = opq.config().ksub();
        let sub_dim = opq.config().sub_dim();
        let centroids = opq.centroids();

        let mut table = Vec::with_capacity(m);
        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &rotated_adc_query[q_sub_offset..q_sub_offset + sub_dim];

            let mut dists = Vec::with_capacity(ksub);
            for c in 0..ksub {
                let centroid_offset = sub_q * ksub * sub_dim + c * sub_dim;
                let centroid = &centroids[centroid_offset..centroid_offset + sub_dim];
                dists.push(self.subspace_score(query_sub, centroid));
            }
            table.push(dists);
        }
        table
    }

    fn precompute_distance_table_opq_flat(
        &self,
        opq: &OptimizedProductQuantizer,
        rotated_adc_query: &[f32],
    ) -> Vec<f32> {
        let m = opq.config().m;
        let ksub = opq.config().ksub();
        let sub_dim = opq.config().sub_dim();
        let centroids = opq.centroids();

        let mut table = vec![0.0f32; m * ksub];
        if sub_dim == 2 {
            for sub_q in 0..m {
                let q_sub_offset = sub_q * 2;
                let query_sub = &rotated_adc_query[q_sub_offset..q_sub_offset + 2];
                for c in 0..ksub {
                    let centroid_offset = sub_q * ksub * 2 + c * 2;
                    let centroid = &centroids[centroid_offset..centroid_offset + 2];
                    table[sub_q * ksub + c] = self.subspace_score_dim2(query_sub, centroid);
                }
            }
            return table;
        }

        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &rotated_adc_query[q_sub_offset..q_sub_offset + sub_dim];
            for c in 0..ksub {
                let centroid_offset = sub_q * ksub * sub_dim + c * sub_dim;
                let centroid = &centroids[centroid_offset..centroid_offset + sub_dim];
                table[sub_q * ksub + c] = self.subspace_score(query_sub, centroid);
            }
        }
        table
    }

    /// Compute ADC distance using precomputed table
    fn adc_distance(&self, table: &[Vec<f32>], code: &[u8]) -> f32 {
        let m = self.pq.config().m;
        let nbits = self.pq.config().nbits;
        let mut total = 0.0f32;

        for sub_q in 0..m {
            let centroid_idx = if nbits == 8 {
                code[sub_q] as usize
            } else {
                // Generic bit extraction
                let byte_offset = sub_q * nbits / 8;
                let bit_offset = (sub_q * nbits) % 8;
                let mut idx = 0usize;
                for bit in 0..nbits {
                    let byte_idx = byte_offset + (bit_offset + bit) / 8;
                    let bit_idx = (bit_offset + bit) % 8;
                    if byte_idx < code.len() && (code[byte_idx] >> bit_idx) & 1 != 0 {
                        idx |= 1 << bit;
                    }
                }
                idx
            };
            total += table[sub_q][centroid_idx];
        }
        total
    }

    #[inline]
    fn adc_distance_flat(&self, table: &[f32], code: &[u8]) -> f32 {
        self.pq.compute_distance_with_table(table, code)
    }

    #[inline]
    fn fill_query_residual(&self, query: &[f32], centroid: &[f32], residual: &mut [f32]) {
        for ((dst, &q), &c) in residual.iter_mut().zip(query.iter()).zip(centroid.iter()) {
            *dst = q - c;
        }
    }

    /// Search using Asymmetric Distance Computation (ADC) - 优化版：并行 nprobe
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        #[cfg(feature = "metrics")]
        let _timer = {
            let m = crate::metrics::init_metrics();
            crate::metrics::Timer::new(m.search_duration_seconds.clone())
        };
        #[cfg(feature = "metrics")]
        crate::metrics::init_metrics().search_requests_total.inc();

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

        #[cfg(feature = "parallel")]
        if n_queries > 1 {
            return self.search_parallel(query, req, rayon::current_num_threads());
        }

        let nprobe = req.nprobe.max(1).min(self.nlist);
        let k = req.top_k;

        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();

        let code_size = self.active_code_size();
        let opq_enabled = self.opq_enabled();
        // For IP/Cosine, the ADC table depends only on the query (not the
        // centroid residual), so it can be computed once outside the probe loop.
        let is_ip = matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine);
        let avg_cluster_size = (self.ids.len() / self.nlist.max(1)).max(1);
        let mut query_residual = vec![0.0f32; self.dim];

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            let cluster_dists = self.score_all_centroids(query_vec);
            let cluster_dists = Self::select_top_clusters(cluster_dists, nprobe);

            let mut candidates: Vec<(i64, f32)> = Vec::with_capacity(avg_cluster_size * nprobe);

            let shared_table = if !opq_enabled && is_ip {
                Some(self.precompute_distance_table_flat(query_vec))
            } else {
                None
            };

            for (cluster, _) in cluster_dists.iter().copied() {
                let (ids, codes) = self.cluster_scan_slice(cluster);
                if ids.is_empty() {
                    continue;
                }
                let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
                let coarse_bias = self.coarse_distance_bias(query_vec, centroid);

                if opq_enabled {
                    self.fill_query_residual(query_vec, centroid, &mut query_residual);
                    let opq = self.opq.as_ref().unwrap();
                    let rotated_adc_query = self.rotated_adc_query(opq, query_vec, &query_residual);
                    let table = self.precompute_distance_table_opq_flat(opq, &rotated_adc_query);
                    for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                        let dist = self.adc_distance_flat(&table, code) + coarse_bias;
                        candidates.push((*id, dist));
                    }
                } else if let Some(ref table) = shared_table {
                    for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                        let dist = self.adc_distance_flat(table, code) + coarse_bias;
                        candidates.push((*id, dist));
                    }
                } else {
                    self.fill_query_residual(query_vec, centroid, &mut query_residual);
                    let table = self.precompute_distance_table_flat(&query_residual);
                    for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                        let dist = self.adc_distance_flat(&table, code) + coarse_bias;
                        candidates.push((*id, dist));
                    }
                }
            }

            Self::truncate_top_candidates(&mut candidates, k);
            for i in 0..k {
                if i < candidates.len() {
                    all_ids.push(candidates[i].0);
                    all_dists.push(candidates[i].1);
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    pub fn search_with_bitset(
        &self,
        query: &[f32],
        req: &SearchRequest,
        bitset: &BitsetView,
    ) -> Result<SearchResult> {
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

        let id_to_pos: std::collections::HashMap<i64, usize> = self
            .ids
            .iter()
            .copied()
            .enumerate()
            .map(|(pos, id)| (id, pos))
            .collect();

        let nprobe = req.nprobe.max(1).min(self.nlist);
        let k = req.top_k;
        let code_size = self.active_code_size();
        let opq_enabled = self.opq_enabled();

        let mut all_ids = Vec::with_capacity(n_queries * k);
        let mut all_dists = Vec::with_capacity(n_queries * k);

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            let cluster_dists = self.score_all_centroids(query_vec);
            let cluster_dists = Self::select_top_clusters(cluster_dists, nprobe);

            let mut candidates: Vec<(i64, f32)> = Vec::new();
            let mut query_residual = vec![0.0f32; self.dim];
            for (cluster, _) in cluster_dists.iter().copied() {
                let (ids, codes) = self.cluster_scan_slice(cluster);
                if ids.is_empty() {
                    continue;
                }
                let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
                let coarse_bias = self.coarse_distance_bias(query_vec, centroid);

                self.fill_query_residual(query_vec, centroid, &mut query_residual);

                let table = if opq_enabled {
                    let opq = self.opq.as_ref().expect("OPQ must be present when enabled");
                    let rotated_adc_query = self.rotated_adc_query(opq, query_vec, &query_residual);
                    Some(self.precompute_distance_table_opq_flat(opq, &rotated_adc_query))
                } else {
                    Some(self.precompute_distance_table_flat(
                        self.adc_query_for_metric(query_vec, &query_residual),
                    ))
                };

                for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                    let Some(&pos_in_index) = id_to_pos.get(id) else {
                        continue;
                    };
                    if bitset.test(pos_in_index) {
                        continue;
                    }

                    let dist = self.adc_distance_flat(table.as_ref().unwrap(), code) + coarse_bias;
                    candidates.push((*id, dist));
                }
            }

            Self::truncate_top_candidates(&mut candidates, k);
            for i in 0..k {
                if i < candidates.len() {
                    all_ids.push(candidates[i].0);
                    all_dists.push(candidates[i].1);
                } else {
                    all_ids.push(-1);
                    all_dists.push(f32::MAX);
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    pub fn range_search(&self, query: &[f32], radius: f32, range_filter: f32) -> Vec<(i64, f32)> {
        if self.ids.is_empty() || query.len() != self.dim {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let code_size = self.active_code_size();
        let opq_enabled = self.opq_enabled();
        let mut query_residual = vec![0.0f32; self.dim];

        for cluster in 0..self.nlist {
            let (ids, codes) = self.cluster_scan_slice(cluster);
            if ids.is_empty() {
                continue;
            }
            let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
            let coarse_bias = self.coarse_distance_bias(query, centroid);

            self.fill_query_residual(query, centroid, &mut query_residual);

            if opq_enabled {
                let opq = self.opq.as_ref().expect("OPQ must be present when enabled");
                let rotated_adc_query = self.rotated_adc_query(opq, query, &query_residual);
                let table = self.precompute_distance_table_opq_flat(opq, &rotated_adc_query);
                candidates.extend(
                    ids.iter()
                        .zip(codes.chunks(code_size))
                        .map(|(id, code)| (*id, self.adc_distance_flat(&table, code) + coarse_bias))
                        .filter(|(_, dist)| self.dist_in_range(*dist, radius, range_filter)),
                );
            } else {
                let adc_query = self.adc_query_for_metric(query, &query_residual);
                let table = self.precompute_distance_table_flat(adc_query);
                candidates.extend(
                    ids.iter()
                        .zip(codes.chunks(code_size))
                        .map(|(id, code)| (*id, self.adc_distance_flat(&table, code) + coarse_bias))
                        .filter(|(_, dist)| self.dist_in_range(*dist, radius, range_filter)),
                );
            }
        }

        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        candidates
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

        let nprobe = req.nprobe.max(1).min(self.nlist);
        let k = req.top_k;
        let dim = self.dim;
        let code_size = self.active_code_size();
        let opq_enabled = self.opq_enabled();
        let is_ip = matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine);
        let avg_cluster_size = (self.ids.len() / self.nlist.max(1)).max(1);

        let results: Vec<(Vec<i64>, Vec<f32>)> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * dim;
                let query_vec = &query[q_start..q_start + dim];

                let cluster_dists = self.score_all_centroids(query_vec);
                let cluster_dists = Self::select_top_clusters(cluster_dists, nprobe);

                let mut query_residual = vec![0.0f32; dim];
                let mut candidates: Vec<(i64, f32)> = Vec::with_capacity(avg_cluster_size * nprobe);

                let shared_table = if !opq_enabled && is_ip {
                    Some(self.precompute_distance_table_flat(query_vec))
                } else {
                    None
                };

                for (cluster, _) in cluster_dists.iter().copied() {
                    let (ids, codes) = self.cluster_scan_slice(cluster);
                    if ids.is_empty() {
                        continue;
                    }
                    let centroid = &self.centroids[cluster * dim..(cluster + 1) * dim];
                    let coarse_bias = self.coarse_distance_bias(query_vec, centroid);

                    if opq_enabled {
                        self.fill_query_residual(query_vec, centroid, &mut query_residual);
                        let opq = self.opq.as_ref().unwrap();
                        let rotated_adc_query =
                            self.rotated_adc_query(opq, query_vec, &query_residual);
                        let table =
                            self.precompute_distance_table_opq_flat(opq, &rotated_adc_query);
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance_flat(&table, code) + coarse_bias;
                            candidates.push((*id, dist));
                        }
                    } else if let Some(ref table) = shared_table {
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance_flat(table, code) + coarse_bias;
                            candidates.push((*id, dist));
                        }
                    } else {
                        self.fill_query_residual(query_vec, centroid, &mut query_residual);
                        let table = self.precompute_distance_table_flat(&query_residual);
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance_flat(&table, code) + coarse_bias;
                            candidates.push((*id, dist));
                        }
                    }
                }

                Self::truncate_top_candidates(&mut candidates, k);
                let mut ids = Vec::with_capacity(k);
                let mut dists = Vec::with_capacity(k);
                for i in 0..k {
                    if i < candidates.len() {
                        ids.push(candidates[i].0);
                        dists.push(candidates[i].1);
                    } else {
                        ids.push(-1);
                        dists.push(f32::MAX);
                    }
                }

                (ids, dists)
            })
            .collect();

        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();
        for (ids, dists) in results {
            all_ids.extend(ids);
            all_dists.extend(dists);
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        file.write_all(b"IVFPQ")?;
        file.write_all(&4u32.to_le_bytes())?; // version 4: persist PQ codebook + per-list ids/codes
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.nlist as u32).to_le_bytes())?;
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.nbits_per_idx as u32).to_le_bytes())?;
        file.write_all(&(self.nprobe as u32).to_le_bytes())?;
        file.write_all(&[Self::metric_to_byte(self.config.metric_type)])?;

        // Centroids
        let bytes: Vec<u8> = self
            .centroids
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        file.write_all(&bytes)?;

        // IDs
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        // Vectors (for reconstruction)
        let vec_bytes: Vec<u8> = self.vectors.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&vec_bytes)?;

        // Persist trained PQ codebook to avoid load-time retraining.
        let pq_centroids = self
            .imported_pq_centroids
            .as_deref()
            .unwrap_or(self.pq.centroids());
        file.write_all(&(pq_centroids.len() as u64).to_le_bytes())?;
        let pq_bytes: Vec<u8> = pq_centroids.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&pq_bytes)?;

        // Inverted lists with ids + PQ codes
        for i in 0..self.nlist {
            let count = self.invlist_ids[i].len() as u32;
            file.write_all(&count.to_le_bytes())?;
        }

        for i in 0..self.nlist {
            for &id in &self.invlist_ids[i] {
                file.write_all(&id.to_le_bytes())?;
            }
        }

        for i in 0..self.nlist {
            file.write_all(&self.invlist_codes[i])?;
        }

        Ok(())
    }

    /// Import an IVF-PQ index from native FAISS `write_index` binary file.
    ///
    /// Supported subset:
    /// - top-level fourcc: `IwPQ` / `IvPQ`
    /// - coarse quantizer fourcc: `IxF2` / `IxFI`
    /// - inverted lists fourcc: `ilar`
    pub fn import_from_faiss_file(path: &str) -> Result<Self> {
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
        fn read_f32_vec_with_count<R: Read>(r: &mut R) -> Result<Vec<f32>> {
            let n = read_i64(r)?;
            if n < 0 {
                return Err(crate::api::KnowhereError::Codec(
                    "negative vector length in faiss blob".to_string(),
                ));
            }
            let n = n as usize;
            let mut raw = vec![0u8; n * 4];
            r.read_exact(&mut raw)?;
            let mut out = Vec::with_capacity(n);
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        fn metric_from_faiss_i32(v: i32) -> Result<MetricType> {
            match v {
                0 => Ok(MetricType::L2),
                1 => Ok(MetricType::Ip),
                _ => Err(crate::api::KnowhereError::Codec(format!(
                    "unsupported faiss metric_type {} for ivfpq import",
                    v
                ))),
            }
        }

        let mut file = std::fs::File::open(path)?;

        // Top-level fourcc
        let mut fourcc = [0u8; 4];
        file.read_exact(&mut fourcc)?;
        if &fourcc != b"IwPQ" && &fourcc != b"IvPQ" {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported top-level fourcc {:?}, expected IwPQ/IvPQ",
                String::from_utf8_lossy(&fourcc)
            )));
        }
        let legacy = &fourcc == b"IvPQ";

        // IVF header
        let dim = read_u32(&mut file)? as usize;
        let _ntotal = read_i64(&mut file)?;
        let _dummy1 = read_i64(&mut file)?;
        let _dummy2 = read_i64(&mut file)?;
        let _is_trained = read_u8(&mut file)?;
        let metric_type = metric_from_faiss_i32(read_i32(&mut file)?)?;
        let nlist = read_i64(&mut file)? as usize;
        let nprobe = read_i64(&mut file)? as usize;

        // Nested coarse quantizer (IxF2 / IxFI only)
        let mut cq_fourcc = [0u8; 4];
        file.read_exact(&mut cq_fourcc)?;
        if &cq_fourcc != b"IxF2" && &cq_fourcc != b"IxFI" {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported coarse quantizer fourcc {:?}, expected IxF2/IxFI",
                String::from_utf8_lossy(&cq_fourcc)
            )));
        }

        let cq_dim = read_u32(&mut file)? as usize;
        let _cq_ntotal = read_i64(&mut file)?;
        let _cq_dummy1 = read_i64(&mut file)?;
        let _cq_dummy2 = read_i64(&mut file)?;
        let _cq_is_trained = read_u8(&mut file)?;
        let _cq_metric = read_i32(&mut file)?;
        let coarse_centroids = read_f32_vec_with_count(&mut file)?;
        if cq_dim != dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "coarse quantizer dim mismatch: cq_dim={} dim={}",
                cq_dim, dim
            )));
        }
        if coarse_centroids.len() != nlist * dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "coarse centroid size mismatch: got {}, expected {}",
                coarse_centroids.len(),
                nlist * dim
            )));
        }

        // direct_map: support both simplified i32 form and native char+vector form.
        let dm_pos = file.stream_position()?;
        let mut by_residual = None;
        let mut code_size = None;
        {
            let dm_type = read_u8(&mut file)?;
            let arr_len = read_i64(&mut file)?;
            let mut ok = arr_len >= 0;
            if ok {
                let skip = (arr_len as u64) * 8;
                file.seek(SeekFrom::Current(skip as i64))?;
                if dm_type == 2 {
                    let hash_len = read_i64(&mut file)?;
                    if hash_len < 0 {
                        ok = false;
                    } else {
                        let pair_bytes = (hash_len as u64).saturating_mul(16);
                        file.seek(SeekFrom::Current(pair_bytes as i64))?;
                    }
                }
                if ok {
                    by_residual = Some(read_u8(&mut file)?);
                    code_size = Some(read_i32(&mut file)? as usize);
                }
            }
        }
        if by_residual.is_none() || code_size.is_none() {
            file.seek(SeekFrom::Start(dm_pos))?;
            let _dm_type_i32 = read_i32(&mut file)?;
            by_residual = Some(read_u8(&mut file)?);
            code_size = Some(read_i32(&mut file)? as usize);
        }
        let by_residual = by_residual.unwrap_or(1);
        let code_size = code_size.unwrap_or(0);
        if by_residual == 0 {
            return Err(crate::api::KnowhereError::Codec(
                "import currently supports by_residual=1 only".to_string(),
            ));
        }

        // ProductQuantizer
        let pq_d = read_u32(&mut file)? as usize;
        let pq_m = read_u32(&mut file)? as usize;
        let pq_nbits = read_u32(&mut file)? as usize;
        if pq_d != dim {
            return Err(crate::api::KnowhereError::Codec(format!(
                "pq dim mismatch: pq_d={} dim={}",
                pq_d, dim
            )));
        }
        if pq_m == 0 || dim % pq_m != 0 {
            return Err(crate::api::KnowhereError::Codec(format!(
                "invalid pq_m={} for dim={}",
                pq_m, dim
            )));
        }
        let pq_centroids = read_f32_vec_with_count(&mut file)?;
        let expected_pq_centroids = pq_m * (1usize << pq_nbits) * (dim / pq_m);
        if pq_centroids.len() != expected_pq_centroids {
            return Err(crate::api::KnowhereError::Codec(format!(
                "pq centroids size mismatch: got {}, expected {}",
                pq_centroids.len(),
                expected_pq_centroids
            )));
        }
        let expected_code_size = (pq_m * pq_nbits) / 8;
        if code_size != expected_code_size {
            return Err(crate::api::KnowhereError::Codec(format!(
                "pq code_size mismatch: header={} expected={}",
                code_size, expected_code_size
            )));
        }

        // InvertedLists (ilar only)
        if !legacy {
            let mut il_fourcc = [0u8; 4];
            file.read_exact(&mut il_fourcc)?;
            if &il_fourcc != b"ilar" {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "unsupported invlists fourcc {:?}, expected ilar",
                    String::from_utf8_lossy(&il_fourcc)
                )));
            }
        }
        let il_nlist = read_i64(&mut file)? as usize;
        let il_code_size = read_i64(&mut file)? as usize;
        if il_nlist != nlist {
            return Err(crate::api::KnowhereError::Codec(format!(
                "invlists nlist mismatch: {} vs {}",
                il_nlist, nlist
            )));
        }
        if il_code_size != code_size {
            return Err(crate::api::KnowhereError::Codec(format!(
                "invlists code_size mismatch: {} vs {}",
                il_code_size, code_size
            )));
        }

        // list sizes: support both native ilar format (list_type + READVECTOR)
        // and simplified format (raw [i64; nlist]).
        let mut list_sizes = vec![0usize; nlist];
        let sizes_pos = file.stream_position()?;
        let mut list_type = [0u8; 4];
        file.read_exact(&mut list_type)?;
        if &list_type == b"full" || &list_type == b"sprs" {
            let sz_count = read_i64(&mut file)? as usize;
            if &list_type == b"full" {
                if sz_count != nlist {
                    return Err(crate::api::KnowhereError::Codec(format!(
                        "ilar full size count mismatch: {} vs {}",
                        sz_count, nlist
                    )));
                }
                for slot in &mut list_sizes {
                    *slot = read_i64(&mut file)? as usize;
                }
            } else {
                // sprs: pairs (idx, size)
                for _ in 0..(sz_count / 2) {
                    let idx = read_i64(&mut file)? as usize;
                    let sz = read_i64(&mut file)? as usize;
                    if idx < nlist {
                        list_sizes[idx] = sz;
                    }
                }
            }
        } else {
            // simplified format: rewind and read nlist raw sizes
            file.seek(SeekFrom::Start(sizes_pos))?;
            for slot in &mut list_sizes {
                *slot = read_i64(&mut file)? as usize;
            }
        }

        let mut invlist_ids = vec![Vec::<i64>::new(); nlist];
        let mut invlist_codes = vec![Vec::<u8>::new(); nlist];
        let mut all_ids = Vec::new();
        for i in 0..nlist {
            let sz = list_sizes[i];
            let mut codes = vec![0u8; sz * code_size];
            if !codes.is_empty() {
                file.read_exact(&mut codes)?;
            }
            let mut ids = vec![0i64; sz];
            for id in &mut ids {
                *id = read_i64(&mut file)?;
            }
            all_ids.extend_from_slice(&ids);
            invlist_ids[i] = ids;
            invlist_codes[i] = codes;
        }

        let cfg = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type,
            dim,
            data_type: DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(nlist),
                nprobe: Some(nprobe.max(1)),
                m: Some(pq_m),
                nbits_per_idx: Some(pq_nbits),
                ..Default::default()
            },
        };
        let mut index = Self::new(&cfg)?;
        index.centroids = coarse_centroids;
        index.invlist_ids = invlist_ids;
        index.invlist_codes = invlist_codes;
        index.ids = all_ids;
        index.vectors.clear();
        index.next_id = index.ids.iter().copied().max().map(|x| x + 1).unwrap_or(0);
        index.trained = true;
        index.imported_pq_centroids = Some(pq_centroids);
        index.use_opq = false;
        index.opq = None;
        index.rebuild_compact_invlists();
        Ok(index)
    }

    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let tmp = tempfile::NamedTempFile::new()?;
        self.save(tmp.path())?;
        let bytes = std::fs::read(tmp.path())?;
        Ok(bytes)
    }

    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        const HEADER_V2_LEN: usize = 5 + 4 + 4 + 4 + 4 + 4; // magic + version + dim/nlist/m/nbits
        if bytes.len() < HEADER_V2_LEN {
            return Err(crate::api::KnowhereError::Codec(
                "ivfpq bytes too short".to_string(),
            ));
        }
        if &bytes[..5] != b"IVFPQ" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid magic".to_string(),
            ));
        }

        let version =
            u32::from_le_bytes(bytes[5..9].try_into().map_err(|_| {
                crate::api::KnowhereError::Codec("invalid version bytes".to_string())
            })?);
        let dim = u32::from_le_bytes(
            bytes[9..13]
                .try_into()
                .map_err(|_| crate::api::KnowhereError::Codec("invalid dim bytes".to_string()))?,
        ) as usize;
        let nlist = u32::from_le_bytes(
            bytes[13..17]
                .try_into()
                .map_err(|_| crate::api::KnowhereError::Codec("invalid nlist bytes".to_string()))?,
        ) as usize;
        let m = u32::from_le_bytes(
            bytes[17..21]
                .try_into()
                .map_err(|_| crate::api::KnowhereError::Codec("invalid m bytes".to_string()))?,
        ) as usize;
        let nbits_per_idx = u32::from_le_bytes(
            bytes[21..25]
                .try_into()
                .map_err(|_| crate::api::KnowhereError::Codec("invalid nbits bytes".to_string()))?,
        ) as usize;
        let (nprobe, metric_type) = if version >= 3 {
            if bytes.len() < HEADER_V2_LEN + 4 + 1 {
                return Err(crate::api::KnowhereError::Codec(
                    "ivfpq bytes too short for v3 header".to_string(),
                ));
            }
            let nprobe = u32::from_le_bytes(bytes[25..29].try_into().map_err(|_| {
                crate::api::KnowhereError::Codec("invalid nprobe bytes".to_string())
            })?) as usize;
            let metric = Self::metric_from_byte(bytes[29]);
            (nprobe.max(1), metric)
        } else {
            (nlist.clamp(1, 8), MetricType::L2)
        };

        let cfg = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type,
            data_type: DataType::Float,
            dim,
            params: crate::api::IndexParams {
                nlist: Some(nlist),
                nprobe: Some(nprobe),
                m: Some(m),
                nbits_per_idx: Some(nbits_per_idx),
                ..Default::default()
            },
        };

        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), bytes)?;
        let mut index = Self::new(&cfg)?;
        index.load(tmp.path())?;
        Ok(index)
    }

    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 5];
        file.read_exact(&mut magic)?;
        if &magic != b"IVFPQ" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid magic".to_string(),
            ));
        }

        // Read version
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;
        let version = u32::from_le_bytes(ver);

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;

        let mut nlist_bytes = [0u8; 4];
        file.read_exact(&mut nlist_bytes)?;
        let nlist = u32::from_le_bytes(nlist_bytes) as usize;

        let mut m_bytes = [0u8; 4];
        file.read_exact(&mut m_bytes)?;
        let m = u32::from_le_bytes(m_bytes) as usize;

        let mut nbits_bytes = [0u8; 4];
        file.read_exact(&mut nbits_bytes)?;
        let nbits = u32::from_le_bytes(nbits_bytes) as usize;

        let mut nprobe = self.nprobe;
        let mut metric_type = self.config.metric_type;
        if version >= 3 {
            let mut nprobe_bytes = [0u8; 4];
            file.read_exact(&mut nprobe_bytes)?;
            nprobe = (u32::from_le_bytes(nprobe_bytes) as usize).max(1);
            let mut metric_byte = [0u8; 1];
            file.read_exact(&mut metric_byte)?;
            metric_type = Self::metric_from_byte(metric_byte[0]);
        }

        self.dim = dim;
        self.nlist = nlist;
        self.m = m;
        self.nbits_per_idx = nbits;
        self.nprobe = nprobe;
        self.config.dim = dim;
        self.config.metric_type = metric_type;
        self.config.params.nlist = Some(nlist);
        self.config.params.nprobe = Some(nprobe);
        self.config.params.m = Some(m);
        self.config.params.nbits_per_idx = Some(nbits);

        // Recreate quantizers with loaded parameters.
        self.pq = ProductQuantizer::new(PQConfig::new(self.dim, self.m, self.nbits_per_idx));
        self.opq = if self.use_opq {
            let mut opq_config = OPQConfig::new(self.dim, self.m, self.nbits_per_idx);
            opq_config.niter = 1;
            opq_config.random_rotation = true;
            Some(OptimizedProductQuantizer::new(opq_config)?)
        } else {
            None
        };
        self.imported_pq_centroids = None;
        self.invlist_ids = (0..self.nlist).map(|_| Vec::new()).collect();
        self.invlist_codes = (0..self.nlist).map(|_| Vec::new()).collect();
        self.compact_invlists = CompactInvertedLists {
            offsets: vec![(0, 0); self.nlist],
            ..Default::default()
        };

        // Load centroids
        let mut centroid_bytes = vec![0u8; nlist * dim * 4];
        file.read_exact(&mut centroid_bytes)?;
        self.centroids.clear();
        for chunk in centroid_bytes.chunks(4) {
            self.centroids
                .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        // Load IDs
        let mut id_count_bytes = [0u8; 8];
        file.read_exact(&mut id_count_bytes)?;
        let id_count = u64::from_le_bytes(id_count_bytes) as usize;

        self.ids.clear();
        for _ in 0..id_count {
            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            self.ids.push(i64::from_le_bytes(id_bytes));
        }

        // Load vectors
        let mut vec_bytes = vec![0u8; id_count * dim * 4];
        file.read_exact(&mut vec_bytes)?;
        self.vectors.clear();
        for chunk in vec_bytes.chunks(4) {
            self.vectors
                .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        if version >= 4 {
            let mut pq_count_bytes = [0u8; 8];
            file.read_exact(&mut pq_count_bytes)?;
            let pq_count = u64::from_le_bytes(pq_count_bytes) as usize;
            let mut pq_bytes = vec![0u8; pq_count * 4];
            file.read_exact(&mut pq_bytes)?;
            let mut pq_centroids = Vec::with_capacity(pq_count);
            for chunk in pq_bytes.chunks_exact(4) {
                pq_centroids.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            self.pq.set_centroids(pq_centroids)?;
            self.imported_pq_centroids = None;

            let mut list_counts = vec![0usize; self.nlist];
            for count in &mut list_counts {
                let mut count_bytes = [0u8; 4];
                file.read_exact(&mut count_bytes)?;
                *count = u32::from_le_bytes(count_bytes) as usize;
            }

            for (cluster, &count) in list_counts.iter().enumerate() {
                self.invlist_ids[cluster].clear();
                self.invlist_ids[cluster].reserve(count);
                for _ in 0..count {
                    let mut id_bytes = [0u8; 8];
                    file.read_exact(&mut id_bytes)?;
                    self.invlist_ids[cluster].push(i64::from_le_bytes(id_bytes));
                }
            }

            let code_size = self.active_code_size();
            for (cluster, &count) in list_counts.iter().enumerate() {
                let mut codes = vec![0u8; count * code_size];
                if !codes.is_empty() {
                    file.read_exact(&mut codes)?;
                }
                self.invlist_codes[cluster] = codes;
            }

            self.trained = true;
            self.next_id = self.ids.iter().copied().max().map(|id| id + 1).unwrap_or(0);
            self.rebuild_compact_invlists();
            return Ok(());
        }

        eprintln!(
            "warning: loading legacy IVFPQ v{} format; retraining fine quantizer as fallback",
            version
        );

        // Retrain PQ from loaded vectors and rebuild inverted lists
        if id_count > 0 {
            // Compute all residuals
            let mut residuals = Vec::with_capacity(id_count * dim);
            let mut assignments = Vec::with_capacity(id_count);

            for i in 0..id_count {
                let vector = &self.vectors[i * dim..(i + 1) * dim];
                let cluster = self.find_nearest_centroid(vector);
                assignments.push(cluster);
                for (j, &value) in vector.iter().enumerate().take(dim) {
                    residuals.push(value - self.centroids[cluster * dim + j]);
                }
            }

            // Train fine quantizer on residuals
            self.train_fine_quantizer(id_count, &residuals)?;

            // Rebuild inverted lists with PQ codes
            for i in 0..self.nlist {
                self.invlist_ids[i].clear();
                self.invlist_codes[i].clear();
            }

            for i in 0..id_count {
                let residual = &residuals[i * dim..(i + 1) * dim];
                let code = self.encode_residual(residual)?;
                let cluster = assignments[i];
                self.invlist_ids[cluster].push(self.ids[i]);
                self.invlist_codes[cluster].extend_from_slice(&code);
            }
        } else {
            for i in 0..nlist {
                self.invlist_ids[i].clear();
                self.invlist_codes[i].clear();
            }
        }

        self.trained = true;
        self.next_id = self.ids.iter().copied().max().map(|id| id + 1).unwrap_or(0);
        self.rebuild_compact_invlists();
        Ok(())
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = match self.config.metric_type {
                MetricType::L2 | MetricType::Hamming => l2_distance_sq(vector, centroid),
                MetricType::Ip | MetricType::Cosine => -dot_product_f32(vector, centroid),
            };
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
#[allow(deprecated)] // Test coverage still exercises the legacy IVF scaffold for comparison.
mod tests {
    use super::*;
    use crate::api::{IndexType, MetricType};
    use crate::faiss::ivf::IvfIndex;

    #[test]
    fn test_ivfpq_basic() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(4),
                m: Some(4), // 4 sub-quantizers, each handles 4 dims
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();

        // Generate training/index vectors
        let n = 200;
        let dim = 16;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 7 + j * 13) % 100) as f32 / 100.0);
            }
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        // Search for a vector that's in the index
        let query = vectors[0..dim].to_vec();
        let req = SearchRequest {
            top_k: 5,
            nprobe: 4,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
        // The closest result should be vector 0 itself
        assert_eq!(result.ids[0], 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_ivfpq_batch_search_matches_single_query_results() {
        let dim = 16;
        let n = 256;
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(8),
                nprobe: Some(4),
                m: Some(4),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 11 + j * 17) % 101) as f32 / 50.0);
            }
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let req = SearchRequest {
            top_k: 6,
            nprobe: 4,
            filter: None,
            params: None,
            radius: None,
        };
        let queries = &vectors[..4 * dim];

        let batch = index.search(queries, &req).unwrap();
        for q_idx in 0..4 {
            let single = index
                .search(&queries[q_idx * dim..(q_idx + 1) * dim], &req)
                .unwrap();
            assert_eq!(
                &batch.ids[q_idx * req.top_k..(q_idx + 1) * req.top_k],
                single.ids.as_slice()
            );
            assert_eq!(
                &batch.distances[q_idx * req.top_k..(q_idx + 1) * req.top_k],
                single.distances.as_slice()
            );
        }
    }

    #[test]
    fn test_ivfpq_recall() {
        // Test recall with clustered data
        let dim = 32;
        let n = 2000;
        let n_queries = 50;
        let k = 10;

        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(16),
                nprobe: Some(8),
                m: Some(8),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();

        // Generate vectors with some structure
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                let base = (i % 16) as f32 * 10.0;
                vectors.push(base + ((i * 7 + j * 13) % 100) as f32 / 10.0);
            }
        }

        index.train(&vectors).unwrap();

        let ids: Vec<i64> = (0..n as i64).collect();
        index.add(&vectors, Some(&ids)).unwrap();

        // Compute ground truth with brute force
        let queries = &vectors[0..n_queries * dim];
        let mut total_recall = 0.0;

        for q in 0..n_queries {
            let query = &queries[q * dim..(q + 1) * dim];

            // Brute force ground truth
            let mut gt: Vec<(i64, f32)> = (0..n)
                .map(|i| {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    let dist: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as i64, dist)
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_ids: std::collections::HashSet<i64> = gt.iter().take(k).map(|x| x.0).collect();

            // IVF-PQ search
            let req = SearchRequest {
                top_k: k,
                nprobe: 8,
                filter: None,
                params: None,
                radius: None,
            };
            let result = index.search(query, &req).unwrap();

            let found: std::collections::HashSet<i64> =
                result.ids.iter().filter(|&&id| id >= 0).copied().collect();

            let recall = gt_ids.intersection(&found).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / n_queries as f32;
        eprintln!("IVF-PQ R@{}: {:.1}%", k, avg_recall * 100.0);
        assert!(
            avg_recall > 0.80,
            "R@{} = {:.1}% (expected > 80%)",
            k,
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_ivfpq_ip_search_applies_coarse_bias_across_clusters() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::Ip,
            dim: 4,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(2),
                nprobe: Some(2),
                m: Some(2),
                nbits_per_idx: Some(1),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        index.centroids = vec![
            10.0, 0.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, 0.0,
        ];
        index.imported_pq_centroids = Some(vec![
            0.0, 0.0, 6.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ]);
        index.invlist_ids = vec![vec![0], vec![1]];
        index.invlist_codes = vec![vec![0b00], vec![0b01]];
        index.ids = vec![0, 1];
        index.trained = true;

        let req = SearchRequest {
            top_k: 1,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };
        let query = vec![10.0, 0.0, 0.0, 0.0];

        let result = index.search(&query, &req).unwrap();
        assert_eq!(
            result.ids[0], 0,
            "IP search must include the coarse centroid contribution when comparing candidates across clusters"
        );
    }

    /// Verify that IVF-PQ (dim=128) achieves reasonable recall with plain PQ (OPQ disabled).
    #[test]
    fn test_ivfpq_opq_recall_dim128() {
        let dim = 128;
        let n_clusters = 16;
        let n_per_cluster = 100;
        let n = n_clusters * n_per_cluster;
        let n_queries = 20;
        let k = 10;

        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(n_clusters),
                nprobe: Some(n_clusters), // full scan
                m: Some(8),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        // Well-separated clusters: cluster i is centered at (i*50, 0, ..., 0)
        let mut vectors = Vec::with_capacity(n * dim);
        for c in 0..n_clusters {
            for v in 0..n_per_cluster {
                for j in 0..dim {
                    let base = if j == 0 { (c as f32) * 50.0 } else { 0.0 };
                    let noise = ((c * 7919 + v * 6271 + j * 3137) % 100) as f32 / 100.0;
                    vectors.push(base + noise);
                }
            }
        }

        let mut index = IvfPqIndex::new(&config).unwrap();

        index.train(&vectors).unwrap();

        let ids: Vec<i64> = (0..n as i64).collect();
        index.add(&vectors, Some(&ids)).unwrap();

        let queries = &vectors[0..n_queries * dim];
        let mut total_recall = 0.0f32;

        for q in 0..n_queries {
            let query = &queries[q * dim..(q + 1) * dim];

            // Brute-force ground truth
            let mut gt: Vec<(i64, f32)> = (0..n)
                .map(|i| {
                    let v = &vectors[i * dim..(i + 1) * dim];
                    let dist: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as i64, dist)
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_ids: std::collections::HashSet<i64> = gt.iter().take(k).map(|x| x.0).collect();

            let req = SearchRequest {
                top_k: k,
                nprobe: n_clusters,
                filter: None,
                params: None,
                radius: None,
            };
            let result = index.search(query, &req).unwrap();

            // q==0: searching for exact self — must be rank 1
            if q == 0 {
                assert_eq!(
                    result.ids[0],
                    0,
                    "Exact-match query must return itself at rank 1. Got ids={:?}, dists={:?}",
                    &result.ids[..k.min(result.ids.len())],
                    &result.distances[..k.min(result.distances.len())]
                );
            }

            let found: std::collections::HashSet<i64> =
                result.ids.iter().filter(|&&id| id >= 0).copied().collect();
            let recall = gt_ids.intersection(&found).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / n_queries as f32;
        eprintln!("IVF-PQ (dim=128) R@{}: {:.1}%", k, avg_recall * 100.0);
        assert!(
            avg_recall > 0.50,
            "IVF-PQ R@{} = {:.1}% (expected > 50%)",
            k,
            avg_recall * 100.0
        );
    }

    #[test]
    fn ivfpq_hot_path_audit_exposes_residual_pq_state() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(2),
                m: Some(4),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut ivfpq = IvfPqIndex::new(&config).unwrap();
        let mut ivf = IvfIndex::new(16, 4);

        let n = 64;
        let dim = 16;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 11 + j * 17) % 97) as f32 / 97.0);
            }
        }

        ivfpq.train(&vectors).unwrap();
        ivfpq.add(&vectors, None).unwrap();

        ivf.train(&vectors);
        ivf.add(&vectors);

        let audit = ivfpq.hot_path_audit();

        assert_eq!(audit.coarse_centroid_count, 4);
        assert_eq!(audit.pq_subquantizers, 4);
        assert_eq!(audit.encoded_vector_count, n);
        assert!(
            audit.total_code_bytes >= n * audit.code_size_bytes,
            "real IVF-PQ hot path must store PQ code bytes for every inserted vector"
        );
        assert!(
            ivf.lists.iter().flatten().count() >= n,
            "placeholder IVF scaffold still stores raw vector positions in coarse lists"
        );
    }

    #[test]
    fn test_ivfpq_save_load_preserves_pq_state_and_lists() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(2),
                m: Some(4),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        let n = 128;
        let dim = 16;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 19 + j * 7) % 101) as f32 / 101.0);
            }
        }
        let ids: Vec<i64> = (0..n as i64).map(|x| x * 3 + 7).collect();

        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let expected_invlist_ids = index.invlist_ids.clone();
        let expected_invlist_codes = index.invlist_codes.clone();
        let expected_centroids = index.pq.centroids().to_vec();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        index.save(tmp.path()).unwrap();

        let mut loaded = IvfPqIndex::new(&config).unwrap();
        loaded.load(tmp.path()).unwrap();

        assert_eq!(loaded.invlist_ids, expected_invlist_ids);
        assert_eq!(loaded.invlist_codes, expected_invlist_codes);
        assert_eq!(loaded.pq.centroids(), expected_centroids.as_slice());
        assert!(loaded.imported_pq_centroids.is_none());
    }

    #[test]
    fn test_ivfpq_range_search_returns_hits_within_bounds() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(2),
                m: Some(4),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        let n = 64;
        let dim = 16;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 13 + j * 5) % 89) as f32 / 89.0);
            }
        }

        index.train(&vectors).unwrap();
        index.add(&vectors, None).unwrap();

        let query = vectors[0..dim].to_vec();
        let hits = index.range_search(&query, 0.0, 5.0);

        assert!(!hits.is_empty());
        assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
        assert!(hits.iter().all(|(_, dist)| *dist >= 0.0 && *dist <= 5.0));
    }

    #[test]
    fn test_ivfpq_search_with_bitset() {
        let dim = 16;
        let n = 200;
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim,
            data_type: crate::api::DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(4),
                m: Some(2),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 17 + j * 3) % 101) as f32 / 50.0);
            }
        }

        let ids: Vec<i64> = (0..n as i64).collect();
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        let query = vectors[0..dim].to_vec();
        let req = SearchRequest {
            top_k: 5,
            nprobe: 4,
            filter: None,
            params: None,
            radius: None,
        };

        let mut bitset = BitsetView::new(n);
        for i in 0..100 {
            bitset.set_bit(i);
        }

        let result = index.search_with_bitset(&query, &req, &bitset).unwrap();
        assert_eq!(result.ids.len(), 5);
        assert!(result.ids.iter().all(|&id| id == -1 || id >= 100));
    }

    #[test]
    fn test_ivfpq_precompute_distance_table_flat_matches_adc_lookup() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 4,
            data_type: DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(1),
                nprobe: Some(1),
                m: Some(2),
                nbits_per_idx: Some(1),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        index
            .pq
            .set_centroids(vec![
                0.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 1.0, 1.0,
            ])
            .unwrap();

        let adc_query = [0.25, 0.75, 0.1, 0.9];
        let table = index.precompute_distance_table_flat(&adc_query);

        assert_eq!(table.len(), index.m * (1usize << index.nbits_per_idx));

        let code = [1u8, 0u8];
        let expected = table[1] + table[1usize << index.nbits_per_idx];
        let actual = index.adc_distance_flat(&table, &code);
        assert!(
            (actual - expected).abs() < 1e-6,
            "flat ADC lookup must match direct table addressing: actual={actual}, expected={expected}"
        );
        assert!(
            (actual - index.pq.compute_distance_with_table(&table, &code)).abs() < 1e-6,
            "IVF-PQ search should be able to reuse ProductQuantizer::compute_distance_with_table()"
        );
    }

    #[test]
    fn test_ivfpq_compact_invlists_match_per_cluster_storage() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 16,
            data_type: DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(4),
                nprobe: Some(2),
                m: Some(4),
                nbits_per_idx: Some(8),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        let n = 128usize;
        let dim = 16usize;
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 23 + j * 9) % 113) as f32 / 31.0);
            }
        }

        let ids: Vec<i64> = (0..n as i64).map(|id| id * 7 + 3).collect();
        index.train(&vectors).unwrap();
        index.add(&vectors, Some(&ids)).unwrap();

        for cluster in 0..index.nlist {
            let (compact_ids, compact_codes) = index.cluster_scan_slice(cluster);
            assert_eq!(compact_ids, index.invlist_ids[cluster].as_slice());
            assert_eq!(compact_codes, index.invlist_codes[cluster].as_slice());
        }

        assert_eq!(
            index.compact_invlists.all_ids.len(),
            index.invlist_ids.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(
            index.compact_invlists.all_codes.len(),
            index.invlist_codes.iter().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn test_ivfpq_dim2_flat_distance_table_matches_scalar_path() {
        let config = IndexConfig {
            index_type: IndexType::IvfPq,
            metric_type: MetricType::L2,
            dim: 8,
            data_type: DataType::Float,
            params: crate::api::IndexParams {
                nlist: Some(2),
                nprobe: Some(1),
                m: Some(4),
                nbits_per_idx: Some(1),
                ..Default::default()
            },
        };

        let mut index = IvfPqIndex::new(&config).unwrap();
        index
            .pq
            .set_centroids(vec![
                0.0, 0.0, 1.0, 1.0, //
                0.0, 1.0, 1.0, 2.0, //
                1.0, 0.0, 2.0, 1.0, //
                1.0, 1.0, 2.0, 2.0,
            ])
            .unwrap();

        let adc_query = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6];
        let table = index.precompute_distance_table_flat(&adc_query);
        let sub_dim = index.dim / index.m;
        let ksub = 1usize << index.nbits_per_idx;
        let mut expected = vec![0.0f32; index.m * ksub];

        for sub_q in 0..index.m {
            let query_sub = &adc_query[sub_q * sub_dim..(sub_q + 1) * sub_dim];
            let centroids = index.pq_subquantizer_centroids(sub_q).unwrap();
            for c in 0..ksub {
                let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                expected[sub_q * ksub + c] = index.subspace_score(query_sub, centroid);
            }
        }

        assert_eq!(table.len(), expected.len());
        for (actual, expected) in table.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ivfpq_select_top_clusters_matches_full_sort_order() {
        let scored = vec![(0usize, 3.0f32), (1, 1.0), (2, 2.0), (3, 0.5), (4, 1.0)];

        let top3 = IvfPqIndex::select_top_clusters(scored.clone(), 3);
        assert_eq!(top3, vec![(3, 0.5), (1, 1.0), (4, 1.0)]);

        let all = IvfPqIndex::select_top_clusters(scored, 99);
        assert_eq!(all, vec![(3, 0.5), (1, 1.0), (4, 1.0), (2, 2.0), (0, 3.0)]);
    }

    #[test]
    fn test_import_faiss_ivfpq_roundtrip_via_raw_bytes() {
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
        let m = 2usize;
        let nbits = 8usize;
        let code_size = 2usize;

        let mut blob = Vec::new();

        // Top-level IwPQ
        blob.extend_from_slice(b"IwPQ");
        push_u32(&mut blob, dim as u32);
        push_i64(&mut blob, 3); // ntotal
        push_i64(&mut blob, 0); // dummy1
        push_i64(&mut blob, 0); // dummy2
        push_u8(&mut blob, 1); // trained
        push_i32(&mut blob, 0); // metric_type=L2
        push_i64(&mut blob, nlist as i64);
        push_i64(&mut blob, nprobe as i64);

        // Coarse quantizer: IxF2
        blob.extend_from_slice(b"IxF2");
        push_u32(&mut blob, dim as u32);
        push_i64(&mut blob, nlist as i64); // cq ntotal
        push_i64(&mut blob, 0);
        push_i64(&mut blob, 0);
        push_u8(&mut blob, 1);
        push_i32(&mut blob, 0); // L2
        push_i64(&mut blob, (nlist * dim) as i64); // READVECTOR count
        for i in 0..(nlist * dim) {
            push_f32(&mut blob, i as f32 * 0.1);
        }

        // NoMap direct_map: u8 type=0 + i64 arr_len=0
        push_u8(&mut blob, 0u8);
        push_i64(&mut blob, 0i64);

        // IVF-PQ payload
        push_u8(&mut blob, 1); // by_residual
        push_i32(&mut blob, code_size as i32);
        push_u32(&mut blob, dim as u32); // pq_d
        push_u32(&mut blob, m as u32); // pq_M
        push_u32(&mut blob, nbits as u32); // pq_nbits
        let pq_nf = m * (1usize << nbits) * (dim / m);
        push_i64(&mut blob, pq_nf as i64); // pq READVECTOR count
        for i in 0..pq_nf {
            push_f32(&mut blob, (i % 17) as f32 * 0.01);
        }

        // Inverted lists: ilar
        blob.extend_from_slice(b"ilar");
        push_i64(&mut blob, nlist as i64);
        push_i64(&mut blob, code_size as i64);
        // simplified per-list sizes (no full/sprs tag)
        push_i64(&mut blob, 2);
        push_i64(&mut blob, 1);
        // list0 codes + ids
        blob.extend_from_slice(&[1u8, 2u8, 3u8, 4u8]);
        push_i64(&mut blob, 11);
        push_i64(&mut blob, 12);
        // list1 codes + ids
        blob.extend_from_slice(&[5u8, 6u8]);
        push_i64(&mut blob, 13);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &blob).unwrap();

        let imported = IvfPqIndex::import_from_faiss_file(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(imported.dim, dim);
        assert_eq!(imported.nlist, nlist);
        assert_eq!(imported.m, m);
        assert_eq!(imported.nbits_per_idx, nbits);
        assert_eq!(imported.nprobe, nprobe);
        assert!(imported.trained);
        assert_eq!(imported.invlist_ids[0], vec![11, 12]);
        assert_eq!(imported.invlist_ids[1], vec![13]);
        assert_eq!(imported.invlist_codes[0], vec![1, 2, 3, 4]);
        assert_eq!(imported.invlist_codes[1], vec![5, 6]);
        assert_eq!(
            imported.imported_pq_centroids.as_ref().unwrap().len(),
            pq_nf
        );
    }
}

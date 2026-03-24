//! IVF-PQ Index Implementation
//!
//! Inverted File Index with Product Quantization.
//! Uses coarse quantizer (IVF) + fine quantizer (PQ) for compressed storage
//! and fast Asymmetric Distance Computation (ADC).
//!
//! OPT-003: 内存布局优化 - 使用 Vec 替代 HashMap

use crate::api::{DataType, IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};
use crate::quantization::KMeans;
use crate::quantization::opq::{OPQConfig, OptimizedProductQuantizer};
use crate::quantization::pq::{PQConfig, ProductQuantizer};
use crate::simd::{dot_product_f32, l2_distance_sq};
use std::io::{Read, Seek, SeekFrom};

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
    fn centroid_score(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric_type {
            MetricType::L2 | MetricType::Hamming => l2_distance_sq(a, b),
            MetricType::Ip | MetricType::Cosine => -dot_product_f32(a, b),
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
            self.opq.as_ref().map(|opq| opq.code_size()).unwrap_or(self.pq.code_size())
        } else {
            self.pq.code_size()
        }
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

        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| {
                let dist =
                    self.centroid_score(query, &self.centroids[c * self.dim..(c + 1) * self.dim]);
                (c, dist)
            })
            .collect();
        cluster_dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        cluster_dists
            .into_iter()
            .take(nprobe.min(self.nlist))
            .map(|(cluster, _)| cluster)
            .collect()
    }

    /// Precompute distance table for a query residual.
    /// Returns table[sub_q][centroid_idx] = L2 distance between query sub-vector and PQ centroid.
    fn precompute_distance_table(&self, query_residual: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.dim / self.m;
        let ksub = 1usize << self.nbits_per_idx;
        let m = self.m;

        let mut table = Vec::with_capacity(m);
        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &query_residual[q_sub_offset..q_sub_offset + sub_dim];

            let centroids = self
                .pq_subquantizer_centroids(sub_q)
                .expect("pq centroids unavailable for subquantizer");
            let mut dists = Vec::with_capacity(ksub);

            for c in 0..ksub {
                let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                let mut dist = 0.0f32;
                match self.config.metric_type {
                    MetricType::L2 | MetricType::Hamming => {
                        for d in 0..sub_dim {
                            let diff = query_sub[d] - centroid[d];
                            dist += diff * diff;
                        }
                    }
                    MetricType::Ip | MetricType::Cosine => {
                        for d in 0..sub_dim {
                            dist -= query_sub[d] * centroid[d];
                        }
                    }
                }
                dists.push(dist);
            }
            table.push(dists);
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

    /// Precompute ADC distance table using OPQ centroids and rotated query residual.
    fn precompute_distance_table_opq(
        &self,
        opq: &OptimizedProductQuantizer,
        rotated_residual: &[f32],
    ) -> Vec<Vec<f32>> {
        let m = opq.config().m;
        let ksub = opq.config().ksub();
        let sub_dim = opq.config().sub_dim();
        let centroids = opq.centroids();

        let mut table = Vec::with_capacity(m);
        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &rotated_residual[q_sub_offset..q_sub_offset + sub_dim];

            let mut dists = Vec::with_capacity(ksub);
            for c in 0..ksub {
                let centroid_offset = sub_q * ksub * sub_dim + c * sub_dim;
                let centroid = &centroids[centroid_offset..centroid_offset + sub_dim];
                let mut dist = 0.0f32;
                match self.config.metric_type {
                    MetricType::L2 | MetricType::Hamming => {
                        for i in 0..sub_dim {
                            let d = query_sub[i] - centroid[i];
                            dist += d * d;
                        }
                    }
                    MetricType::Ip | MetricType::Cosine => {
                        for i in 0..sub_dim {
                            dist -= query_sub[i] * centroid[i];
                        }
                    }
                }
                dists.push(dist);
            }
            table.push(dists);
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

        let nprobe = req.nprobe.max(1).min(self.nlist);
        let k = req.top_k;

        let mut all_ids = Vec::new();
        let mut all_dists = Vec::new();

        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            // Find nearest clusters (coarse quantizer)
            let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
                .map(|c| {
                    let dist = self.centroid_score(
                        query_vec,
                        &self.centroids[c * self.dim..(c + 1) * self.dim],
                    );
                    (c, dist)
                })
                .collect();
            cluster_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

            #[cfg(feature = "parallel")]
            {
                // 并行搜索 nprobe 个簇
                use rayon::prelude::*;

                // PQ code size per vector
                let code_size = self.active_code_size();
                let opq_enabled = self.opq_enabled();

                let candidates: Vec<(i64, f32)> = cluster_dists
                    .iter()
                    .take(nprobe)
                    .par_bridge()
                    .filter_map(|(cluster, _)| {
                        let ids = &self.invlist_ids[*cluster];
                        let codes = &self.invlist_codes[*cluster];
                        if ids.is_empty() {
                            return None;
                        }

                        // Compute query residual for this cluster
                        let mut query_residual = vec![0.0f32; self.dim];
                        for j in 0..self.dim {
                            query_residual[j] =
                                query_vec[j] - self.centroids[*cluster * self.dim + j];
                        }

                        if opq_enabled {
                            let opq = self.opq.as_ref().unwrap();
                            // Apply rotation once per cluster residual, then do ADC lookup.
                            let rotated_residual = opq.apply_rotation_single(&query_residual);
                            let table = self.precompute_distance_table_opq(opq, &rotated_residual);
                            Some(
                                ids.iter()
                                    .zip(codes.chunks(code_size))
                                    .map(|(id, code)| (*id, self.adc_distance(&table, code)))
                                    .collect::<Vec<_>>(),
                            )
                        } else {
                            // Precompute distance table
                            let table = self.precompute_distance_table(&query_residual);
                            // ADC search - iterate ids and codes together
                            Some(
                                ids.iter()
                                    .zip(codes.chunks(code_size))
                                    .map(|(id, code)| (*id, self.adc_distance(&table, code)))
                                    .collect::<Vec<_>>(),
                            )
                        }
                    })
                    .flatten()
                    .collect();

                // Sort and take top k
                let mut candidates = candidates;
                candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

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

            #[cfg(not(feature = "parallel"))]
            {
                // 串行搜索（fallback）
                let mut candidates: Vec<(i64, f32)> = Vec::new();

                // PQ code size per vector
                let code_size = self.active_code_size();
                let opq_enabled = self.opq_enabled();

                for &(cluster, _) in cluster_dists.iter().take(nprobe) {
                    let ids = &self.invlist_ids[cluster];
                    let codes = &self.invlist_codes[cluster];

                    if ids.is_empty() {
                        continue;
                    }

                    // Compute query residual for this cluster
                    let mut query_residual = vec![0.0f32; self.dim];
                    for j in 0..self.dim {
                        query_residual[j] = query_vec[j] - self.centroids[cluster * self.dim + j];
                    }

                    if opq_enabled {
                        let opq = self.opq.as_ref().unwrap();
                        let rotated_residual = opq.apply_rotation_single(&query_residual);
                        let table = self.precompute_distance_table_opq(opq, &rotated_residual);
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance(&table, code);
                            candidates.push((*id, dist));
                        }
                    } else {
                        // Precompute distance table for this cluster's residual
                        let table = self.precompute_distance_table(&query_residual);
                        // ADC: look up distances from table for each PQ code
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance(&table, code);
                            candidates.push((*id, dist));
                        }
                    }
                }

                // Sort and take top k
                candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

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
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
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
        let nlist = self.nlist;

        // Parallel search for each query
        let results: Vec<(Vec<i64>, Vec<f32>)> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * dim;
                let query_vec = &query[q_start..q_start + dim];

                // Find nearest clusters
                let mut cluster_dists: Vec<(usize, f32)> = (0..nlist)
                    .map(|c| {
                        let dist = self
                            .centroid_score(query_vec, &self.centroids[c * dim..(c + 1) * dim]);
                        (c, dist)
                    })
                    .collect();
                cluster_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

                // Search top nprobe clusters using ADC
                let mut candidates: Vec<(i64, f32)> = Vec::new();

                // PQ code size per vector
                let code_size = self.active_code_size();
                let opq_enabled = self.opq_enabled();

                for &(cluster, _) in cluster_dists.iter().take(nprobe) {
                    let ids = &self.invlist_ids[cluster];
                    let codes = &self.invlist_codes[cluster];

                    if ids.is_empty() {
                        continue;
                    }

                    // Compute query residual
                    let mut query_residual = vec![0.0f32; dim];
                    for j in 0..dim {
                        query_residual[j] = query_vec[j] - self.centroids[cluster * dim + j];
                    }

                    if opq_enabled {
                        let opq = self.opq.as_ref().unwrap();
                        let rotated_residual = opq.apply_rotation_single(&query_residual);
                        let table = self.precompute_distance_table_opq(opq, &rotated_residual);
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance(&table, code);
                            candidates.push((*id, dist));
                        }
                    } else {
                        // Precompute distance table
                        let table = self.precompute_distance_table(&query_residual);
                        // ADC lookup
                        for (id, code) in ids.iter().zip(codes.chunks(code_size)) {
                            let dist = self.adc_distance(&table, code);
                            candidates.push((*id, dist));
                        }
                    }
                }

                // Sort and take top k
                candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

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

        let version = u32::from_le_bytes(bytes[5..9].try_into().map_err(|_| {
            crate::api::KnowhereError::Codec("invalid version bytes".to_string())
        })?);
        let dim = u32::from_le_bytes(bytes[9..13].try_into().map_err(|_| {
            crate::api::KnowhereError::Codec("invalid dim bytes".to_string())
        })?) as usize;
        let nlist = u32::from_le_bytes(bytes[13..17].try_into().map_err(|_| {
            crate::api::KnowhereError::Codec("invalid nlist bytes".to_string())
        })?) as usize;
        let m = u32::from_le_bytes(bytes[17..21].try_into().map_err(|_| {
            crate::api::KnowhereError::Codec("invalid m bytes".to_string())
        })?) as usize;
        let nbits_per_idx = u32::from_le_bytes(bytes[21..25].try_into().map_err(|_| {
            crate::api::KnowhereError::Codec("invalid nbits bytes".to_string())
        })?) as usize;
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
        Ok(())
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = self.centroid_score(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
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
            let gt_ids: std::collections::HashSet<i64> =
                gt.iter().take(k).map(|x| x.0).collect();

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
                    result.ids[0], 0,
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
        assert_eq!(imported.imported_pq_centroids.as_ref().unwrap().len(), pq_nf);
    }
}

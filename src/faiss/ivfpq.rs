//! IVF-PQ Index Implementation
//!
//! Inverted File Index with Product Quantization.
//! Uses coarse quantizer (IVF) + fine quantizer (PQ) for compressed storage
//! and fast Asymmetric Distance Computation (ADC).
//!
//! OPT-003: 内存布局优化 - 使用 Vec 替代 HashMap

use crate::api::{IndexConfig, Result, SearchRequest, SearchResult};
use crate::executor::l2_distance;
use crate::quantization::KMeans;
use crate::quantization::opq::{OPQConfig, OptimizedProductQuantizer};
use crate::quantization::pq::{PQConfig, ProductQuantizer};
use crate::simd::l2_distance_sq;

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
        let use_opq = config.dim >= 64;
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
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index must be trained first".to_string(),
            ));
        }

        #[cfg(feature = "parallel")]
        {
            return self.add_parallel(vectors, ids, rayon::current_num_threads());
        }

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
        Ok(n)
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
                    let dist = l2_distance(vector, &centroids[c * dim..(c + 1) * dim]);
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
                let dist = l2_distance_sq(query, &self.centroids[c * self.dim..(c + 1) * self.dim]);
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
        let sub_dim = self.pq.config().sub_dim();
        let ksub = self.pq.config().ksub();
        let m = self.pq.config().m;

        let mut table = Vec::with_capacity(m);
        for sub_q in 0..m {
            let q_sub_offset = sub_q * sub_dim;
            let query_sub = &query_residual[q_sub_offset..q_sub_offset + sub_dim];

            let centroids = self.pq.get_centroids(sub_q).unwrap();
            let mut dists = Vec::with_capacity(ksub);

            for c in 0..ksub {
                let centroid = &centroids[c * sub_dim..(c + 1) * sub_dim];
                let mut dist = 0.0f32;
                for d in 0..sub_dim {
                    let diff = query_sub[d] - centroid[d];
                    dist += diff * diff;
                }
                dists.push(dist);
            }
            table.push(dists);
        }
        table
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
                for i in 0..sub_dim {
                    let d = query_sub[i] - centroid[i];
                    dist += d * d;
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
                    let dist = l2_distance_sq(
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
                        let dist =
                            l2_distance(query_vec, &self.centroids[c * dim..(c + 1) * dim]);
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
        file.write_all(&2u32.to_le_bytes())?; // version 2 for PQ format
        file.write_all(&(self.dim as u32).to_le_bytes())?;
        file.write_all(&(self.nlist as u32).to_le_bytes())?;
        file.write_all(&(self.m as u32).to_le_bytes())?;
        file.write_all(&(self.nbits_per_idx as u32).to_le_bytes())?;

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

        // Inverted lists with PQ codes
        for i in 0..self.nlist {
            let count = self.invlist_ids[i].len() as u32;
            file.write_all(&count.to_le_bytes())?;
        }

        // Write all PQ codes
        for i in 0..self.nlist {
            file.write_all(&self.invlist_codes[i])?;
        }

        Ok(())
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

        // Skip version
        let mut ver = [0u8; 4];
        file.read_exact(&mut ver)?;

        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;

        let mut nlist_bytes = [0u8; 4];
        file.read_exact(&mut nlist_bytes)?;
        let nlist = u32::from_le_bytes(nlist_bytes) as usize;

        let mut m_bytes = [0u8; 4];
        file.read_exact(&mut m_bytes)?;
        let _m = u32::from_le_bytes(m_bytes) as usize;

        let mut nbits_bytes = [0u8; 4];
        file.read_exact(&mut nbits_bytes)?;
        let _nbits = u32::from_le_bytes(nbits_bytes) as usize;

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
        self.next_id = self.ids.last().map(|&id| id + 1).unwrap_or(0);
        Ok(())
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;
        for (i, centroid) in self.centroids.chunks(self.dim).enumerate() {
            let dist = vector
                .iter()
                .zip(centroid)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
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
}

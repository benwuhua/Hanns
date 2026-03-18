//! IVF-SQ8 Index Implementation
//!
//! IVF (Inverted File) + SQ8 (Scalar Quantization 8-bit)
//! 内存优化索引，适合大规模数据

use std::collections::HashMap;

use crate::api::{IndexConfig, Result, SearchRequest, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::executor::l2_distance;
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::quantization::ScalarQuantizer;

type IvfSq8InvertedListSnapshot = Vec<(usize, Vec<(i64, Vec<u8>)>)>;

/// IVF-SQ8 Index
#[allow(dead_code)]
pub struct IvfSq8Index {
    config: IndexConfig,
    dim: usize,
    nlist: usize,  // Number of clusters
    nprobe: usize, // Number of clusters to search

    /// Cluster centroids
    centroids: Vec<f32>,
    /// Inverted lists: cluster_id -> list of (vector_id, quantized residual)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
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
            centroids: Vec::new(),
            inverted_lists: HashMap::new(),
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

            self.inverted_lists
                .entry(cluster)
                .or_default()
                .push((id, quantized));
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

        let mut all_ids = vec![-1; n_queries * k];
        let mut all_dists = vec![f32::MAX; n_queries * k];

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let q_indices: Vec<usize> = (0..n_queries).collect();

            let results: Vec<Vec<(i64, f32)>> = q_indices
                .par_iter()
                .map(|&q_idx| {
                    let q_start = q_idx * self.dim;
                    let query_vec = &query[q_start..q_start + self.dim];

                    let clusters = self.search_clusters(query_vec, nprobe);

                let mut candidates: Vec<(i64, f32)> = Vec::new();

                for &cluster_id in &clusters {
                    if let Some(list) = self.inverted_lists.get(&cluster_id) {
                        let centroid_vec =
                            &self.centroids[cluster_id * self.dim..(cluster_id + 1) * self.dim];
                        let q_residual: Vec<f32> = query_vec
                            .iter()
                            .zip(centroid_vec.iter())
                            .map(|(a, b)| a - b)
                            .collect();

                        for &(id, ref quantized) in list {
                            let dist = self.quantizer.sq_l2_asymmetric(&q_residual, quantized);
                            candidates.push((id, dist));
                        }
                    }
                }

                    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    candidates.truncate(k);
                    candidates
                })
                .collect();

            for (q_idx, res) in results.into_iter().enumerate() {
                let offset = q_idx * k;
                for (i, item) in res.into_iter().enumerate().take(k) {
                    all_ids[offset + i] = item.0;
                    all_dists[offset + i] = item.1;
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for q_idx in 0..n_queries {
                let q_start = q_idx * self.dim;
                let query_vec = &query[q_start..q_start + self.dim];

                let clusters = self.search_clusters(query_vec, nprobe);

                let mut candidates: Vec<(i64, f32)> = Vec::new();

                for &cluster_id in &clusters {
                    if let Some(list) = self.inverted_lists.get(&cluster_id) {
                        let centroid_vec =
                            &self.centroids[cluster_id * self.dim..(cluster_id + 1) * self.dim];
                        let q_residual: Vec<f32> = query_vec
                            .iter()
                            .zip(centroid_vec.iter())
                            .map(|(a, b)| a - b)
                            .collect();

                        for &(id, ref quantized) in list {
                            let dist = self.quantizer.sq_l2_asymmetric(&q_residual, quantized);
                            candidates.push((id, dist));
                        }
                    }
                }

                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                candidates.truncate(k);

                let offset = q_idx * k;
                for (i, item) in candidates.into_iter().enumerate().take(k) {
                    all_ids[offset + i] = item.0;
                    all_dists[offset + i] = item.1;
                }
            }
        }

        Ok(SearchResult::new(all_ids, all_dists, 0.0))
    }

    /// Find nearest centroid
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
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

    /// Search clusters
    fn search_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.nlist)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let dist = l2_distance(query, centroid);
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(nprobe).map(|(i, _)| i).collect()
    }

    /// Compute residual
    fn compute_residual(&self, vector: &[f32], cluster: usize) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        vector.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }

    /// Reconstruct vector from centroid + residual
    fn reconstruct(&self, _query: &[f32], cluster: usize, residual: &[f32]) -> Vec<f32> {
        let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
        centroid.iter().zip(residual).map(|(c, r)| c + r).collect()
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

        // Parallel: assign vectors to clusters
        let assignments: Vec<(usize, i64, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let vector = &vectors[start..start + dim];

                // Find nearest centroid
                let mut min_dist = f32::MAX;
                let mut cluster = 0;
                for c in 0..nlist {
                    let dist = l2_distance(vector, &centroids[c * dim..]);
                    if dist < min_dist {
                        min_dist = dist;
                        cluster = c;
                    }
                }

                // Compute residual
                let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
                let residual: Vec<f32> = vector.iter().zip(centroid).map(|(a, b)| a - b).collect();

                // Quantize residual
                let quantized = quantizer.encode(&residual);

                let id = ids.map(|ids| ids[i]).unwrap_or(i as i64);
                (cluster, id, quantized)
            })
            .collect();

        // Collect by cluster
        let mut cluster_data: HashMap<usize, Vec<(i64, Vec<u8>)>> = HashMap::new();
        for (cluster, id, quantized) in assignments {
            cluster_data
                .entry(cluster)
                .or_default()
                .push((id, quantized));
        }

        // Merge into inverted lists
        for (cluster, mut items) in cluster_data {
            let list = self.inverted_lists.entry(cluster).or_default();
            list.append(&mut items);
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
        let nlist = self.nlist;
        let centroids = self.centroids.clone();
        let inverted_lists: IvfSq8InvertedListSnapshot = self
            .inverted_lists
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        // Parallel search for each query
        let results: Vec<Vec<(i64, f32)>> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * dim;
                let query_vec = &query[q_start..q_start + dim];

                // Find nearest clusters
                let mut cluster_dists: Vec<(usize, f32)> = (0..nlist)
                    .map(|c| {
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        let dist = l2_distance(query_vec, centroid);
                        (c, dist)
                    })
                    .collect();
                cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let clusters: Vec<usize> = cluster_dists
                    .into_iter()
                    .take(nprobe)
                    .map(|(i, _)| i)
                    .collect();

                // Search clusters
                let mut candidates: Vec<(i64, f32)> = Vec::new();

                for &cluster_id in &clusters {
                    if let Some(list) = inverted_lists.iter().find(|(c, _)| *c == cluster_id) {
                        for &(id, ref quantized) in &list.1 {
                            let residual = self.quantizer.decode(quantized);
                            let reconstructed: Vec<f32> = centroids
                                [cluster_id * dim..(cluster_id + 1) * dim]
                                .iter()
                                .zip(residual.iter())
                                .map(|(c, r)| c + r)
                                .collect();
                            let dist = l2_distance(query_vec, &reconstructed);
                            candidates.push((id, dist));
                        }
                    }
                }

                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                candidates.truncate(k);
                candidates
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

    /// Get total number of vectors
    pub fn ntotal(&self) -> usize {
        self.ids.len()
    }

    /// Save index
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        // Magic
        file.write_all(b"IVFSQ8")?;

        // Dim
        file.write_all(&(self.dim as u32).to_le_bytes())?;

        // Nlist
        file.write_all(&(self.nlist as u32).to_le_bytes())?;

        // Centroids
        let centroid_bytes: Vec<u8> = self
            .centroids
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        file.write_all(&centroid_bytes)?;

        // Quantizer params
        file.write_all(&self.quantizer.min_val.to_le_bytes())?;
        file.write_all(&self.quantizer.max_val.to_le_bytes())?;
        file.write_all(&self.quantizer.scale.to_le_bytes())?;

        // IDs
        file.write_all(&(self.ids.len() as u64).to_le_bytes())?;
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }

        // Vectors
        let vec_bytes: Vec<u8> = self.vectors.iter().flat_map(|&f| f.to_le_bytes()).collect();
        file.write_all(&vec_bytes)?;

        Ok(())
    }

    /// Load index
    pub fn load(path: &std::path::Path, dim: usize) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;

        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"IVFSQ8" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid IVFSQ8 magic".to_string(),
            ));
        }

        let mut u32_buf = [0u8; 4];
        file.read_exact(&mut u32_buf)?;
        let stored_dim = u32::from_le_bytes(u32_buf) as usize;
        if stored_dim != dim {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "dim mismatch: load dim {} vs stored dim {}",
                dim, stored_dim
            )));
        }

        file.read_exact(&mut u32_buf)?;
        let nlist = u32::from_le_bytes(u32_buf) as usize;

        let mut centroids = vec![0.0f32; nlist * stored_dim];
        for value in &mut centroids {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *value = f32::from_le_bytes(buf);
        }

        let mut f32_buf = [0u8; 4];
        file.read_exact(&mut f32_buf)?;
        let min_val = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut f32_buf)?;
        let max_val = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut f32_buf)?;
        let scale = f32::from_le_bytes(f32_buf);

        let mut u64_buf = [0u8; 8];
        file.read_exact(&mut u64_buf)?;
        let n = u64::from_le_bytes(u64_buf) as usize;

        let mut ids = vec![0i64; n];
        for id in &mut ids {
            let mut ibuf = [0u8; 8];
            file.read_exact(&mut ibuf)?;
            *id = i64::from_le_bytes(ibuf);
        }

        let mut vectors = vec![0.0f32; n * stored_dim];
        for value in &mut vectors {
            let mut vbuf = [0u8; 4];
            file.read_exact(&mut vbuf)?;
            *value = f32::from_le_bytes(vbuf);
        }

        let mut quantizer = ScalarQuantizer::new(stored_dim, 8);
        quantizer.min_val = min_val;
        quantizer.max_val = max_val;
        quantizer.scale = scale;
        quantizer.offset = min_val;

        let mut index = Self {
            config: IndexConfig {
                index_type: crate::api::IndexType::IvfSq8,
                metric_type: crate::api::MetricType::L2,
                dim: stored_dim,
                data_type: crate::api::DataType::Float,
                params: crate::api::IndexParams::ivf(nlist, 8),
            },
            dim: stored_dim,
            nlist,
            nprobe: 8,
            centroids,
            inverted_lists: HashMap::new(),
            quantizer,
            vectors,
            ids,
            next_id: 0,
            trained: true,
        };

        index.next_id = index.ids.iter().copied().max().map_or(0, |m| m + 1);

        // Rebuild inverted lists from vectors + ids.
        let total = index.ids.len();
        for i in 0..total {
            let start = i * index.dim;
            let vector = &index.vectors[start..start + index.dim];
            let cluster = index.find_nearest_centroid(vector);
            let residual = index.compute_residual(vector, cluster);
            let quantized = index.quantizer.encode(&residual);
            let id = index.ids[i];
            index
                .inverted_lists
                .entry(cluster)
                .or_default()
                .push((id, quantized));
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
        // IVF-SQ8 doesn't have native bitset support, use default implementation
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

        // Filter out bitset-marked vectors
        let mut filtered_ids = Vec::new();
        let mut filtered_distances = Vec::new();

        for (id, dist) in api_result.ids.iter().zip(api_result.distances.iter()) {
            let idx = *id as usize;
            if idx >= bitset.len() || !bitset.get(idx) {
                filtered_ids.push(*id);
                filtered_distances.push(*dist);
            }
        }

        Ok(IndexSearchResult::new(
            filtered_ids,
            filtered_distances,
            api_result.elapsed_ms,
        ))
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
}

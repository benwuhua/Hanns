//! IVF-OPQ Index Implementation
//!
//! Inverted File Index with Optimized Product Quantization
//! Combines IVF coarse quantization with OPQ fine quantization for improved recall.

use std::collections::HashMap;

use crate::api::{IndexConfig, KnowhereError, Result, SearchRequest, SearchResult};
use crate::quantization::{
    OPQConfig, OptimizedProductQuantizer, ResidualPQConfig, ResidualProductQuantizer,
};

/// IVF-OPQ Index configuration
#[derive(Clone, Debug)]
pub struct IvfOpqConfig {
    /// Dimensionality of vectors
    pub dim: usize,
    /// Number of coarse clusters (nlist)
    pub nlist: usize,
    /// Number of probes during search
    pub nprobe: usize,
    /// OPQ: number of subquantizers
    pub m: usize,
    /// OPQ: bits per subvector
    pub nbits: usize,
    /// Use residual PQ instead of standard OPQ
    pub use_residual: bool,
    /// Number of OPQ training iterations
    pub opq_niter: usize,
}

impl Default for IvfOpqConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            nlist: 1024,
            nprobe: 32,
            m: 8,
            nbits: 8,
            use_residual: true,
            opq_niter: 20,
        }
    }
}

impl IvfOpqConfig {
    pub fn new(dim: usize, nlist: usize, m: usize, nbits: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: (nlist / 32).max(1),
            m,
            nbits,
            use_residual: true,
            opq_niter: 20,
        }
    }
}

/// IVF-OPQ Index
pub struct IvfOpqIndex {
    config: IvfOpqConfig,
    /// Coarse centroids (nlist x dim)
    coarse_centroids: Vec<f32>,
    /// OPQ quantizer for residuals
    opq: Option<OptimizedProductQuantizer>,
    /// Residual PQ quantizer
    residual_pq: Option<ResidualProductQuantizer>,
    /// Inverted lists: cluster_id -> list of (vector_id, residual_code)
    inverted_lists: HashMap<usize, Vec<(i64, Vec<u8>)>>,
    /// Total number of vectors
    ntotal: usize,
    /// Next ID to assign
    next_id: i64,
    /// Whether trained
    is_trained: bool,
}

impl IvfOpqIndex {
    /// Create a new IVF-OPQ index
    pub fn new(config: IvfOpqConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "Dimension must be > 0".to_string(),
            ));
        }

        if config.nlist == 0 {
            return Err(KnowhereError::InvalidArg("nlist must be > 0".to_string()));
        }

        if config.dim % config.m != 0 {
            return Err(KnowhereError::InvalidArg(format!(
                "Dimension {} must be divisible by m={}",
                config.dim, config.m
            )));
        }

        Ok(Self {
            config,
            coarse_centroids: Vec::new(),
            opq: None,
            residual_pq: None,
            inverted_lists: HashMap::new(),
            ntotal: 0,
            next_id: 0,
            is_trained: false,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &IvfOpqConfig {
        &self.config
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get total number of vectors
    pub fn ntotal(&self) -> usize {
        self.ntotal
    }

    /// Train the index
    pub fn train(&mut self, n: usize, x: &[f32]) -> Result<()> {
        if n == 0 || x.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "Cannot train with empty data".to_string(),
            ));
        }

        let expected_size = n * self.config.dim;
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats for {} vectors of dim {}, got {}",
                expected_size,
                n,
                self.config.dim,
                x.len()
            )));
        }

        // Step 1: Train coarse quantizer (k-means)
        self.train_coarse(n, x)?;

        // Step 2: Assign vectors to clusters and compute residuals
        let residuals_by_cluster = self.compute_residuals_by_cluster(n, x)?;

        // Step 3: Train OPQ/RPQ on residuals
        if self.config.use_residual {
            self.train_residual_pq(&residuals_by_cluster)?;
        } else {
            self.train_opq(&residuals_by_cluster)?;
        }

        // Step 4: Initialize inverted lists
        for i in 0..self.config.nlist {
            self.inverted_lists.insert(i, Vec::new());
        }

        self.is_trained = true;
        tracing::info!(
            "Trained IVF-OPQ with {} clusters, residual={}",
            self.config.nlist,
            self.config.use_residual
        );
        Ok(())
    }

    /// Train coarse quantizer using k-means
    fn train_coarse(&mut self, _n: usize, x: &[f32]) -> Result<()> {
        let dim = self.config.dim;
        let k = self.config.nlist;

        self.coarse_centroids = vec![0.0f32; k * dim];

        // Use k-means from quantization module
        let mut kmeans = crate::quantization::KMeans::new(k, dim);
        kmeans.train(x);

        self.coarse_centroids.copy_from_slice(kmeans.centroids());

        tracing::debug!("Trained coarse quantizer with {} centroids", k);
        Ok(())
    }

    /// Compute residuals for each cluster
    fn compute_residuals_by_cluster(
        &self,
        n: usize,
        x: &[f32],
    ) -> Result<HashMap<usize, Vec<f32>>> {
        let dim = self.config.dim;
        let mut residuals_by_cluster: HashMap<usize, Vec<f32>> = HashMap::new();

        // Initialize
        for i in 0..self.config.nlist {
            residuals_by_cluster.insert(i, Vec::new());
        }

        // Assign each vector to nearest centroid and compute residual
        for i in 0..n {
            let vec_offset = i * dim;
            let vector = &x[vec_offset..vec_offset + dim];

            // Find nearest coarse centroid
            let cluster = self.find_nearest_coarse_centroid(vector);

            // Compute residual
            let centroid_offset = cluster * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            let residuals = residuals_by_cluster.get_mut(&cluster).unwrap();
            for j in 0..dim {
                residuals.push(vector[j] - centroid[j]);
            }
        }

        Ok(residuals_by_cluster)
    }

    /// Find nearest coarse centroid
    fn find_nearest_coarse_centroid(&self, x: &[f32]) -> usize {
        let dim = self.config.dim;
        let k = self.config.nlist;

        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for c in 0..k {
            let centroid_offset = c * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            let dist = Self::l2_distance(x, centroid);

            if dist < best_dist {
                best_dist = dist;
                best_idx = c;
            }
        }

        best_idx
    }

    /// Train OPQ on residuals
    fn train_opq(&mut self, residuals_by_cluster: &HashMap<usize, Vec<f32>>) -> Result<()> {
        let dim = self.config.dim;
        let m = self.config.m;
        let nbits = self.config.nbits;
        let niter = self.config.opq_niter;

        // Collect all residuals
        let mut all_residuals = Vec::new();
        for residuals in residuals_by_cluster.values() {
            all_residuals.extend(residuals);
        }

        let n_residuals = all_residuals.len() / dim;

        if n_residuals == 0 {
            return Err(KnowhereError::InvalidArg(
                "No residuals to train OPQ".to_string(),
            ));
        }

        // Create and train OPQ
        let mut opq_config = OPQConfig::new(dim, m, nbits);
        opq_config.niter = niter;

        let mut opq = OptimizedProductQuantizer::new(opq_config)?;
        opq.train(n_residuals, &all_residuals)?;

        self.opq = Some(opq);

        tracing::debug!("Trained OPQ on {} residuals", n_residuals);
        Ok(())
    }

    /// Train Residual PQ on residuals
    fn train_residual_pq(&mut self, residuals_by_cluster: &HashMap<usize, Vec<f32>>) -> Result<()> {
        let dim = self.config.dim;
        let m = self.config.m;
        let nbits = self.config.nbits;

        // Collect all residuals
        let mut all_residuals = Vec::new();
        for residuals in residuals_by_cluster.values() {
            all_residuals.extend(residuals);
        }

        let n_residuals = all_residuals.len() / dim;

        if n_residuals == 0 {
            return Err(KnowhereError::InvalidArg(
                "No residuals to train Residual PQ".to_string(),
            ));
        }

        // Create and train Residual PQ
        // Use fewer coarse centroids for the residual quantizer since residuals are already centered
        let ncentroids = (self.config.nlist / 4).max(16);

        let rpq_config = ResidualPQConfig::new(dim, ncentroids, m, nbits);
        let mut residual_pq = ResidualProductQuantizer::new(rpq_config)?;
        residual_pq.train(n_residuals, &all_residuals)?;

        self.residual_pq = Some(residual_pq);

        tracing::debug!("Trained Residual PQ on {} residuals", n_residuals);
        Ok(())
    }

    /// Add vectors to the index
    pub fn add(&mut self, n: usize, x: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.is_trained {
            return Err(KnowhereError::InvalidArg(
                "Index must be trained first".to_string(),
            ));
        }

        let expected_size = n * self.config.dim;
        if x.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats for {} vectors, got {}",
                expected_size,
                n,
                x.len()
            )));
        }

        let dim = self.config.dim;

        for i in 0..n {
            let vec_offset = i * dim;
            let vector = &x[vec_offset..vec_offset + dim];

            // Find nearest coarse centroid
            let cluster = self.find_nearest_coarse_centroid(vector);

            // Compute residual
            let centroid_offset = cluster * dim;
            let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

            let mut residual = vec![0.0f32; dim];
            for j in 0..dim {
                residual[j] = vector[j] - centroid[j];
            }

            // Encode residual
            let code = if let Some(ref rpq) = self.residual_pq {
                rpq.encode(&residual)?
            } else if let Some(ref opq) = self.opq {
                opq.encode(&residual)?
            } else {
                return Err(KnowhereError::InternalError(
                    "No quantizer trained".to_string(),
                ));
            };

            // Get ID
            let id = ids.map(|ids| ids[i]).unwrap_or(self.next_id);
            self.next_id += 1;

            // Add to inverted list
            self.inverted_lists
                .get_mut(&cluster)
                .unwrap()
                .push((id, code));
        }

        self.ntotal += n;
        tracing::debug!("Added {} vectors to IVF-OPQ", n);
        Ok(n)
    }

    /// Search the index
    pub fn search(&self, nq: usize, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.is_trained {
            return Err(KnowhereError::InvalidArg(
                "Index must be trained first".to_string(),
            ));
        }

        if nq == 0 || query.is_empty() {
            return Err(KnowhereError::InvalidArg(
                "Cannot search with empty query".to_string(),
            ));
        }

        let expected_size = nq * self.config.dim;
        if query.len() != expected_size {
            return Err(KnowhereError::InvalidArg(format!(
                "Expected {} floats for {} queries of dim {}, got {}",
                expected_size,
                nq,
                self.config.dim,
                query.len()
            )));
        }

        let dim = self.config.dim;
        let k = req.top_k;
        let nprobe = req.nprobe.min(self.config.nlist).max(1);

        let mut all_ids = Vec::with_capacity(nq * k);
        let mut all_dists = Vec::with_capacity(nq * k);

        for q_idx in 0..nq {
            let q_offset = q_idx * dim;
            let q_vec = &query[q_offset..q_offset + dim];

            // Find nearest coarse clusters
            let mut cluster_dists: Vec<(usize, f32)> = (0..self.config.nlist)
                .map(|c| {
                    let centroid_offset = c * dim;
                    let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];
                    let dist = Self::l2_distance(q_vec, centroid);
                    (c, dist)
                })
                .collect();

            cluster_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Search top nprobe clusters
            let mut candidates: Vec<(i64, f32)> = Vec::new();

            for (cluster, _) in cluster_dists.iter().take(nprobe) {
                if let Some(list) = self.inverted_lists.get(cluster) {
                    for (id, code) in list {
                        // Compute distance
                        let dist = self.compute_distance_to_code(q_vec, code, *cluster);
                        candidates.push((*id, dist));
                    }
                }
            }

            // Sort and take top k
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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

    /// Compute distance from query to a code
    fn compute_distance_to_code(&self, query: &[f32], code: &[u8], cluster: usize) -> f32 {
        let dim = self.config.dim;
        let centroid_offset = cluster * dim;
        let centroid = &self.coarse_centroids[centroid_offset..centroid_offset + dim];

        // Compute query residual
        let mut query_residual = vec![0.0f32; dim];
        for j in 0..dim {
            query_residual[j] = query[j] - centroid[j];
        }

        // Compute distance using the appropriate quantizer
        if let Some(ref rpq) = self.residual_pq {
            rpq.compute_distance(&query_residual, code)
        } else if let Some(ref opq) = self.opq {
            opq.compute_distance(&query_residual, code)
        } else {
            f32::MAX
        }
    }

    /// Compute L2 distance
    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    }

    /// Reset the index
    pub fn reset(&mut self) {
        self.inverted_lists.clear();
        for i in 0..self.config.nlist {
            self.inverted_lists.insert(i, Vec::new());
        }
        self.ntotal = 0;
        self.next_id = 0;
    }

    /// Get code size
    pub fn code_size(&self) -> usize {
        if let Some(ref rpq) = self.residual_pq {
            rpq.code_size()
        } else if let Some(ref opq) = self.opq {
            opq.code_size()
        } else {
            self.config.m * self.config.nbits / 8
        }
    }
}

/// IVF-OPQ Index wrapper for API compatibility
pub struct IvfOpqIndexWrapper {
    index: IvfOpqIndex,
}

impl IvfOpqIndexWrapper {
    pub fn new(config: &IndexConfig) -> Result<Self> {
        let nlist = config.params.nlist.unwrap_or(1024);
        let m = config.params.m.unwrap_or(8);
        let nbits = config.params.nbits_per_idx.unwrap_or(8);

        let ivf_config = IvfOpqConfig::new(config.dim, nlist, m, nbits);
        let index = IvfOpqIndex::new(ivf_config)?;

        Ok(Self { index })
    }

    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.index.config().dim;
        self.index.train(n, vectors)
    }

    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        let n = vectors.len() / self.index.config().dim;
        self.index.add(n, vectors, ids)
    }

    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        let nq = query.len() / self.index.config().dim;
        self.index.search(nq, query, req)
    }

    pub fn ntotal(&self) -> usize {
        self.index.ntotal()
    }

    pub fn is_trained(&self) -> bool {
        self.index.is_trained()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<f32> {
        let mut vectors = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * dim + j) % 100) as f32 / 100.0);
            }
        }
        vectors
    }

    #[test]
    fn test_ivf_opq_config() {
        let config = IvfOpqConfig::new(128, 1024, 8, 8);
        assert_eq!(config.dim, 128);
        assert_eq!(config.nlist, 1024);
        assert_eq!(config.m, 8);
        assert!(config.use_residual);
    }

    #[test]
    fn test_ivf_opq_train_and_add() {
        let config = IvfOpqConfig::new(64, 64, 8, 8);
        let mut index = IvfOpqIndex::new(config).unwrap();

        // Train
        let train_data = create_test_vectors(1000, 64);
        index.train(1000, &train_data).unwrap();
        assert!(index.is_trained());

        // Add vectors
        let add_data = create_test_vectors(100, 64);
        let n_added = index.add(100, &add_data, None).unwrap();
        assert_eq!(n_added, 100);
        assert_eq!(index.ntotal(), 100);
    }

    #[test]
    fn test_ivf_opq_search() {
        let config = IvfOpqConfig::new(64, 64, 8, 8);
        let mut index = IvfOpqIndex::new(config).unwrap();

        // Train
        let train_data = create_test_vectors(1000, 64);
        index.train(1000, &train_data).unwrap();

        // Add vectors
        let add_data = create_test_vectors(100, 64);
        index.add(100, &add_data, None).unwrap();

        // Search
        let query = create_test_vectors(1, 64);
        let req = SearchRequest {
            top_k: 5,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };

        let result = index.search(1, &query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
        assert!(result.ids[0] != -1); // Should find at least one result
    }

    // Note: Standard OPQ mode (use_residual=false) requires more training data
    // and is tested separately in opq module tests
}

/// HNSW-PCA-USQ: PCA dimensionality reduction + flat brute-force + ExRaBitQ/USQ quantization
///
/// Phase 1 implementation: full-precision path
///   Build: raw [N×D] → PCA → projected [N×D'] → USQ encode → Vec<UsqEncoded>
///   Search: query [D] → PCA → [D'] → brute-force USQ score → top-k
///
/// Phase 2 (future): AVX512 fastscan beam search using UsqLayout/UsqFastScanState.
use crate::api::{KnowhereError, Result, SearchRequest, SearchResult as ApiSearchResult};
use crate::quantization::{
    usq::{UsqConfig, UsqEncoded, UsqQuantizer},
    PcaTransform,
};

pub struct HnswPcaUsqConfig {
    /// Original input dimensionality
    pub dim: usize,
    /// PCA target dimensionality (0 = no PCA)
    pub pca_dim: usize,
    /// USQ bits per dimension: 1, 4, or 8
    pub bits_per_dim: u8,
    /// Random rotation seed for ExRaBitQ
    pub rotation_seed: u64,
}

impl Default for HnswPcaUsqConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            pca_dim: 64,
            bits_per_dim: 4,
            rotation_seed: 42,
        }
    }
}

pub struct HnswPcaUsqIndex {
    orig_dim: usize,
    proj_dim: usize,
    pca: Option<PcaTransform>,
    quantizer: UsqQuantizer,
    encoded: Vec<UsqEncoded>,
    ids: Vec<i64>,
    next_id: i64,
    trained: bool,
}

impl HnswPcaUsqIndex {
    pub fn new(config: HnswPcaUsqConfig) -> Result<Self> {
        let proj_dim = if config.pca_dim > 0 && config.pca_dim < config.dim {
            config.pca_dim
        } else {
            config.dim
        };

        let usq_config = UsqConfig::new(proj_dim, config.bits_per_dim)
            .map_err(|e| KnowhereError::InvalidArg(format!("USQ config error: {e}")))?
            .with_seed(config.rotation_seed);

        Ok(Self {
            orig_dim: config.dim,
            proj_dim,
            pca: None,
            quantizer: UsqQuantizer::new(usq_config),
            encoded: Vec::new(),
            ids: Vec::new(),
            next_id: 0,
            trained: false,
        })
    }

    /// Train PCA + set USQ centroid from training data.
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.orig_dim;
        if n == 0 {
            return Err(KnowhereError::InvalidArg("empty training data".into()));
        }

        let projected = if self.proj_dim < self.orig_dim {
            let pca = PcaTransform::train(vectors, n, self.orig_dim, self.proj_dim);
            let proj = pca.apply(vectors, n);
            self.pca = Some(pca);
            proj
        } else {
            vectors.to_vec()
        };

        // Global centroid (mean) for USQ
        let mut centroid = vec![0.0f32; self.proj_dim];
        for row in projected.chunks_exact(self.proj_dim) {
            for (c, &v) in centroid.iter_mut().zip(row.iter()) {
                *c += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for c in &mut centroid {
            *c *= inv_n;
        }
        self.quantizer.set_centroid(&centroid);

        self.trained = true;
        Ok(())
    }

    /// Add vectors (call train first).
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("not trained".into()));
        }
        let n = vectors.len() / self.orig_dim;
        let projected = self.project_batch(vectors, n);

        for (i, row) in projected.chunks_exact(self.proj_dim).enumerate() {
            let enc = self.quantizer.encode(row);
            self.encoded.push(enc);
            let id = ids.map(|s| s[i]).unwrap_or(self.next_id);
            self.ids.push(id);
            self.next_id += 1;
        }
        Ok(n)
    }

    /// Search k-NN using brute-force USQ scoring.
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("not trained".into()));
        }
        let k = req.top_k.max(1);
        let q_proj = self.project_one(query);

        // Brute-force: compute USQ score for every stored vector
        let mut scored: Vec<(i64, f32)> = self
            .encoded
            .iter()
            .zip(self.ids.iter())
            .map(|(enc, &id)| {
                let score = self.quantizer.score(enc, &q_proj);
                (id, score)
            })
            .collect();

        // USQ score is an approximation of inner product; sort descending (larger = closer)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Pad to k
        let mut ids = Vec::with_capacity(k);
        let mut dists = Vec::with_capacity(k);
        for (id, score) in &scored {
            ids.push(*id);
            dists.push(*score);
        }
        while ids.len() < k {
            ids.push(-1);
            dists.push(f32::NEG_INFINITY);
        }

        Ok(ApiSearchResult::new(ids, dists, 0.0))
    }

    pub fn count(&self) -> usize {
        self.ids.len()
    }

    pub fn orig_dim(&self) -> usize {
        self.orig_dim
    }

    pub fn proj_dim(&self) -> usize {
        self.proj_dim
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn project_batch(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        if let Some(pca) = &self.pca {
            pca.apply(vectors, n)
        } else {
            vectors.to_vec()
        }
    }

    fn project_one(&self, x: &[f32]) -> Vec<f32> {
        if let Some(pca) = &self.pca {
            pca.apply_one(x)
        } else {
            x.to_vec()
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn make_data(n: usize, d: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        (0..n * d).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_hnsw_pca_usq_basic() {
        let dim = 64usize;
        let pca_dim = 32usize;
        let n = 200usize;
        let data = make_data(n, dim);

        let mut idx = HnswPcaUsqIndex::new(HnswPcaUsqConfig {
            dim,
            pca_dim,
            bits_per_dim: 4,
            rotation_seed: 42,
        })
        .unwrap();

        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();
        assert_eq!(idx.count(), n);

        let query = make_data(1, dim);
        let req = SearchRequest {
            top_k: 10,
            ..Default::default()
        };
        let result = idx.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 10);
        // All returned ids should be valid (>= 0)
        assert!(result.ids.iter().all(|&id| id >= 0));
    }

    #[test]
    fn test_hnsw_pca_usq_no_pca() {
        let dim = 32usize;
        let n = 100usize;
        let data = make_data(n, dim);

        let mut idx = HnswPcaUsqIndex::new(HnswPcaUsqConfig {
            dim,
            pca_dim: 0, // no PCA
            bits_per_dim: 8,
            rotation_seed: 0,
        })
        .unwrap();

        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();

        let query = make_data(1, dim);
        let req = SearchRequest {
            top_k: 5,
            ..Default::default()
        };
        let result = idx.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }

    #[test]
    fn test_hnsw_pca_usq_1bit() {
        let dim = 64usize;
        let n = 100usize;
        let data = make_data(n, dim);

        let mut idx = HnswPcaUsqIndex::new(HnswPcaUsqConfig {
            dim,
            pca_dim: 48,
            bits_per_dim: 1,
            rotation_seed: 7,
        })
        .unwrap();

        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();

        let query = make_data(1, dim);
        let req = SearchRequest {
            top_k: 5,
            ..Default::default()
        };
        let result = idx.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }
}

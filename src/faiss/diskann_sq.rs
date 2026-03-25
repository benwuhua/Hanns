//! DiskANN SQ Flash Index (scaffold)
//!
//! This index keeps a DiskANN graph on raw vectors while storing an additional
//! SQ8-compressed representation (optionally after PCA projection).

use crate::api::{IndexConfig, IndexParams, IndexType, MetricType, Result, SearchRequest};
use crate::faiss::diskann::{DiskAnnConfig, DiskAnnIndex};
use crate::quantization::{PcaTransform, Sq8Quantizer};

#[derive(Clone, Debug)]
pub struct DiskAnnSqConfig {
    pub base: DiskAnnConfig,
    pub use_pca: bool,
    pub pca_dim: usize,
    pub rerank_k: usize,
}

impl Default for DiskAnnSqConfig {
    fn default() -> Self {
        Self {
            base: DiskAnnConfig::default(),
            use_pca: false,
            pca_dim: 64,
            rerank_k: 64,
        }
    }
}

pub struct DiskAnnSqIndex {
    inner: DiskAnnIndex,
    sq: Sq8Quantizer,
    sq_codes: Vec<u8>,
    pca: Option<PcaTransform>,
    n: usize,
    d_in: usize,
    d_sq: usize,
    config: DiskAnnSqConfig,
}

impl DiskAnnSqIndex {
    pub fn new(dim: usize, metric_type: MetricType, config: DiskAnnSqConfig) -> Result<Self> {
        if dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }
        let mut inner_cfg = IndexConfig::new(IndexType::DiskAnn, metric_type, dim);
        inner_cfg.params = Self::to_index_params(&config.base);
        let inner = DiskAnnIndex::new(&inner_cfg)?;
        let sq = Sq8Quantizer::new(if config.use_pca { config.pca_dim } else { dim }, 8);

        Ok(Self {
            inner,
            sq,
            sq_codes: Vec::new(),
            pca: None,
            n: 0,
            d_in: dim,
            d_sq: if config.use_pca { config.pca_dim } else { dim },
            config,
        })
    }

    fn to_index_params(base: &DiskAnnConfig) -> IndexParams {
        IndexParams {
            max_degree: Some(base.max_degree),
            search_list_size: Some(base.search_list_size),
            construction_l: Some(base.construction_l),
            beamwidth: Some(base.beamwidth),
            disk_pq_dims: Some(base.disk_pq_dims),
            ..IndexParams::default()
        }
    }

    pub fn build(&mut self, data: &[f32], n: usize, d: usize) -> Result<()> {
        if d != self.d_in {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "dimension mismatch: expected {}, got {}",
                self.d_in, d
            )));
        }
        if n.checked_mul(d).unwrap_or(0) != data.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "input shape mismatch".to_string(),
            ));
        }

        let projected = if self.config.use_pca {
            let target_dim = self.config.pca_dim.clamp(1, d);
            let pca = PcaTransform::train(data, n, d, target_dim);
            let out = pca.apply(data, n);
            self.d_sq = target_dim;
            self.pca = Some(pca);
            out
        } else {
            self.d_sq = d;
            self.pca = None;
            data.to_vec()
        };

        self.sq = Sq8Quantizer::new(self.d_sq, 8);
        self.sq.train(&projected);
        self.sq_codes.clear();
        self.sq_codes.reserve(n * self.d_sq);
        for row in projected.chunks(self.d_sq) {
            let code = self.sq.encode(row);
            self.sq_codes.extend_from_slice(&code);
        }

        self.inner.train(data)?;
        self.n = n;
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(f32, u32)>> {
        let _candidate_k = (k.saturating_mul(2)).max(self.config.rerank_k).max(64);
        let _q_sq = if self.config.use_pca {
            self.pca
                .as_ref()
                .map(|p| p.apply_one(query))
                .unwrap_or_else(|| query.to_vec())
        } else {
            query.to_vec()
        };

        let req = SearchRequest {
            top_k: k,
            nprobe: self.config.base.search_list_size,
            ..SearchRequest::default()
        };
        let result = self.inner.search(query, &req)?;
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let dist = result.distances.get(i).copied().unwrap_or(f32::MAX);
            let id = result.ids.get(i).copied().unwrap_or(-1);
            let mapped = if id >= 0 { id as u32 } else { u32::MAX };
            out.push((dist, mapped));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_diskann_sq_build_and_search() {
        let n = 1000usize;
        let d = 32usize;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let config = DiskAnnSqConfig {
            use_pca: false,
            rerank_k: 20,
            ..DiskAnnSqConfig::default()
        };
        let mut index = DiskAnnSqIndex::new(d, MetricType::L2, config).unwrap();
        index.build(&data, n, d).unwrap();

        let query = &data[..d];
        let result = index.search(query, 10).unwrap();
        assert_eq!(result.len(), 10);
    }

    #[test]
    #[ignore = "DiskAnnSqIndex has no range_search API yet"]
    fn test_diskann_sq_range_search() {
        // Coverage placeholder for TEST-DISKANN-SQ-COVERAGE.
        // DiskAnnSqIndex currently exposes only top-k search.
    }

    #[test]
    #[ignore = "DiskAnnSqIndex has no save/load persistence API yet"]
    fn test_diskann_sq_save_load() {
        // Coverage placeholder for TEST-SAVELOAD-MISSING / TEST-DISKANN-SQ-COVERAGE.
        // DiskAnnSqIndex currently exposes no save/load methods to exercise.
    }
}

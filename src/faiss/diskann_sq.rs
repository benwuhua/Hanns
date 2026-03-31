//! DiskANN SQ Flash Index (scaffold)
//!
//! This index keeps a DiskANN graph on raw vectors while storing an additional
//! SQ8-compressed representation (optionally after PCA projection).
#![allow(deprecated)] // Legacy SQ wrapper intentionally composes deprecated DiskANN compatibility types.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::api::{IndexConfig, IndexParams, IndexType, MetricType, Result, SearchRequest};
use crate::faiss::diskann::{DiskAnnConfig, DiskAnnIndex};
use crate::quantization::{PcaTransform, Sq8Quantizer};
use crate::quantization::sq::QuantizerType;

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
    metric_type: MetricType,
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
            metric_type,
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

    pub fn range_search(&self, query: &[f32], radius: f32) -> Result<Vec<(f32, u32)>> {
        if radius.is_sign_negative() || self.n == 0 {
            return Ok(Vec::new());
        }

        let candidate_k = self.inner.ntotal().min(100);
        if candidate_k == 0 {
            return Ok(Vec::new());
        }

        let results = self.search(query, candidate_k)?;
        Ok(results
            .into_iter()
            .filter(|(dist, id)| *id != u32::MAX && *dist <= radius)
            .collect())
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        let graph_path = dir.join("diskann_graph.bin");
        let meta_path = dir.join("diskann_sq.meta");
        self.inner.save(&graph_path)?;

        let mut f = File::create(meta_path)?;
        f.write_all(b"DASQ")?;
        f.write_all(&1u32.to_le_bytes())?;

        write_u32(&mut f, metric_to_u32(self.metric_type))?;
        write_u64(&mut f, self.n as u64)?;
        write_u64(&mut f, self.d_in as u64)?;
        write_u64(&mut f, self.d_sq as u64)?;
        write_bool(&mut f, self.config.use_pca)?;
        write_u64(&mut f, self.config.pca_dim as u64)?;
        write_u64(&mut f, self.config.rerank_k as u64)?;

        write_diskann_config(&mut f, &self.config.base)?;
        write_sq8_quantizer(&mut f, &self.sq)?;
        write_vec_u8(&mut f, &self.sq_codes)?;
        write_pca_transform(&mut f, self.pca.as_ref())?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let graph_path = dir.join("diskann_graph.bin");
        let meta_path = dir.join("diskann_sq.meta");
        let mut f = File::open(meta_path)?;

        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"DASQ" {
            return Err(crate::api::KnowhereError::Codec(
                "invalid DiskAnnSqIndex magic".to_string(),
            ));
        }

        let version = read_u32(&mut f)?;
        if version != 1 {
            return Err(crate::api::KnowhereError::Codec(format!(
                "unsupported DiskAnnSqIndex version {version}"
            )));
        }

        let metric_type = metric_from_u32(read_u32(&mut f)?)?;
        let n = read_u64(&mut f)? as usize;
        let d_in = read_u64(&mut f)? as usize;
        let d_sq = read_u64(&mut f)? as usize;
        let use_pca = read_bool(&mut f)?;
        let pca_dim = read_u64(&mut f)? as usize;
        let rerank_k = read_u64(&mut f)? as usize;
        let base = read_diskann_config(&mut f)?;
        let sq = read_sq8_quantizer(&mut f)?;
        let sq_codes = read_vec_u8(&mut f)?;
        let pca = read_pca_transform(&mut f)?;

        let config = DiskAnnSqConfig {
            base,
            use_pca,
            pca_dim,
            rerank_k,
        };
        let mut index = Self::new(d_in, metric_type, config)?;
        index.inner.load(&graph_path)?;
        index.metric_type = metric_type;
        index.sq = sq;
        index.sq_codes = sq_codes;
        index.pca = pca;
        index.n = n;
        index.d_sq = d_sq;
        Ok(index)
    }
}

fn write_bool<W: Write>(w: &mut W, v: bool) -> Result<()> {
    w.write_all(&[u8::from(v)])?;
    Ok(())
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0] != 0)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_vec_f32<W: Write>(w: &mut W, values: &[f32]) -> Result<()> {
    write_u64(w, values.len() as u64)?;
    for &v in values {
        write_f32(w, v)?;
    }
    Ok(())
}

fn read_vec_f32<R: Read>(r: &mut R) -> Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_f32(r)?);
    }
    Ok(out)
}

fn write_vec_u8<W: Write>(w: &mut W, values: &[u8]) -> Result<()> {
    write_u64(w, values.len() as u64)?;
    w.write_all(values)?;
    Ok(())
}

fn read_vec_u8<R: Read>(r: &mut R) -> Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut out = vec![0u8; len];
    r.read_exact(&mut out)?;
    Ok(out)
}

fn metric_to_u32(metric: MetricType) -> u32 {
    match metric {
        MetricType::L2 => 0,
        MetricType::Ip => 1,
        MetricType::Cosine => 2,
        MetricType::Hamming => 3,
    }
}

fn metric_from_u32(metric: u32) -> Result<MetricType> {
    match metric {
        0 => Ok(MetricType::L2),
        1 => Ok(MetricType::Ip),
        2 => Ok(MetricType::Cosine),
        3 => Ok(MetricType::Hamming),
        other => Err(crate::api::KnowhereError::Codec(format!(
            "unknown metric type tag {other}"
        ))),
    }
}

fn quantizer_type_to_u32(q: QuantizerType) -> u32 {
    match q {
        QuantizerType::Uniform => 0,
        QuantizerType::Learned => 1,
        QuantizerType::Quant4 => 2,
    }
}

fn quantizer_type_from_u32(tag: u32) -> Result<QuantizerType> {
    match tag {
        0 => Ok(QuantizerType::Uniform),
        1 => Ok(QuantizerType::Learned),
        2 => Ok(QuantizerType::Quant4),
        other => Err(crate::api::KnowhereError::Codec(format!(
            "unknown quantizer type tag {other}"
        ))),
    }
}

fn write_diskann_config<W: Write>(w: &mut W, cfg: &DiskAnnConfig) -> Result<()> {
    write_u64(w, cfg.max_degree as u64)?;
    write_u64(w, cfg.search_list_size as u64)?;
    write_u64(w, cfg.construction_l as u64)?;
    write_f32(w, cfg.pq_code_budget_gb)?;
    write_f32(w, cfg.build_dram_budget_gb)?;
    write_u64(w, cfg.disk_pq_dims as u64)?;
    write_u64(w, cfg.pq_candidate_expand_pct as u64)?;
    write_u64(w, cfg.rerank_expand_pct as u64)?;
    write_bool(w, cfg.saturate_after_prune)?;
    write_u64(w, cfg.intra_batch_candidates as u64)?;
    write_u64(w, cfg.num_entry_points as u64)?;
    write_u64(w, cfg.build_degree_slack_pct as u64)?;
    write_u64(w, cfg.random_init_edges as u64)?;
    write_u64(w, cfg.build_parallel_batch_size as u64)?;
    write_u64(w, cfg.beamwidth as u64)?;
    write_f32(w, cfg.cache_dram_budget_gb)?;
    write_bool(w, cfg.enable_flash_layout)?;
    write_bool(w, cfg.flash_mmap_mode)?;
    write_bool(w, cfg.flash_ssd_mode)?;
    write_u64(w, cfg.flash_prefetch_batch as u64)?;
    write_bool(w, cfg.warm_up)?;
    write_f32(w, cfg.filter_threshold)?;
    write_bool(w, cfg.accelerate_build)?;
    write_u64(w, cfg.min_k as u64)?;
    write_u64(w, cfg.max_k as u64)?;
    write_bool(w, cfg.io_cutting_enabled)?;
    write_f32(w, cfg.io_cutting_threshold)?;
    Ok(())
}

fn read_diskann_config<R: Read>(r: &mut R) -> Result<DiskAnnConfig> {
    Ok(DiskAnnConfig {
        max_degree: read_u64(r)? as usize,
        search_list_size: read_u64(r)? as usize,
        construction_l: read_u64(r)? as usize,
        pq_code_budget_gb: read_f32(r)?,
        build_dram_budget_gb: read_f32(r)?,
        disk_pq_dims: read_u64(r)? as usize,
        pq_candidate_expand_pct: read_u64(r)? as usize,
        rerank_expand_pct: read_u64(r)? as usize,
        saturate_after_prune: read_bool(r)?,
        intra_batch_candidates: read_u64(r)? as usize,
        num_entry_points: read_u64(r)? as usize,
        build_degree_slack_pct: read_u64(r)? as usize,
        random_init_edges: read_u64(r)? as usize,
        build_parallel_batch_size: read_u64(r)? as usize,
        beamwidth: read_u64(r)? as usize,
        cache_dram_budget_gb: read_f32(r)?,
        enable_flash_layout: read_bool(r)?,
        flash_mmap_mode: read_bool(r)?,
        flash_ssd_mode: read_bool(r)?,
        flash_prefetch_batch: read_u64(r)? as usize,
        warm_up: read_bool(r)?,
        filter_threshold: read_f32(r)?,
        accelerate_build: read_bool(r)?,
        min_k: read_u64(r)? as usize,
        max_k: read_u64(r)? as usize,
        io_cutting_enabled: read_bool(r)?,
        io_cutting_threshold: read_f32(r)?,
    })
}

fn write_sq8_quantizer<W: Write>(w: &mut W, sq: &Sq8Quantizer) -> Result<()> {
    write_u64(w, sq.dim as u64)?;
    write_u64(w, sq.bit as u64)?;
    write_u32(w, quantizer_type_to_u32(sq.quantizer_type))?;
    write_f32(w, sq.min_val)?;
    write_f32(w, sq.max_val)?;
    write_f32(w, sq.scale)?;
    write_f32(w, sq.offset)?;
    Ok(())
}

fn read_sq8_quantizer<R: Read>(r: &mut R) -> Result<Sq8Quantizer> {
    Ok(Sq8Quantizer {
        dim: read_u64(r)? as usize,
        bit: read_u64(r)? as usize,
        quantizer_type: quantizer_type_from_u32(read_u32(r)?)?,
        min_val: read_f32(r)?,
        max_val: read_f32(r)?,
        scale: read_f32(r)?,
        offset: read_f32(r)?,
    })
}

fn write_pca_transform<W: Write>(w: &mut W, pca: Option<&PcaTransform>) -> Result<()> {
    write_bool(w, pca.is_some())?;
    if let Some(p) = pca {
        write_u64(w, p.d_in as u64)?;
        write_u64(w, p.d_out as u64)?;
        write_vec_f32(w, &p.mean)?;
        write_vec_f32(w, &p.components)?;
    }
    Ok(())
}

fn read_pca_transform<R: Read>(r: &mut R) -> Result<Option<PcaTransform>> {
    if !read_bool(r)? {
        return Ok(None);
    }

    Ok(Some(PcaTransform {
        d_in: read_u64(r)? as usize,
        d_out: read_u64(r)? as usize,
        mean: read_vec_f32(r)?,
        components: read_vec_f32(r)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::tempdir;

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
    fn test_diskann_sq_range_search() {
        let n = 100usize;
        let d = 16usize;
        let mut data = vec![0.0f32; n * d];
        for i in 1..n {
            data[i * d] = i as f32;
        }

        let mut index = DiskAnnSqIndex::new(d, MetricType::L2, DiskAnnSqConfig::default()).unwrap();
        index.build(&data, n, d).unwrap();

        let query = &data[..d];
        let radius = 0.05f32;
        let results = index.range_search(query, radius).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|(dist, _)| *dist <= radius));
        assert!(results.iter().any(|(dist, id)| *id == 0 && *dist <= radius));
    }

    #[test]
    fn test_diskann_sq_save_load() {
        let n = 256usize;
        let d = 16usize;
        let mut rng = StdRng::seed_from_u64(7);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();

        let config = DiskAnnSqConfig {
            use_pca: true,
            pca_dim: 8,
            rerank_k: 32,
            ..DiskAnnSqConfig::default()
        };
        let mut index = DiskAnnSqIndex::new(d, MetricType::L2, config).unwrap();
        index.build(&data, n, d).unwrap();

        let dir = tempdir().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = DiskAnnSqIndex::load(dir.path()).unwrap();
        let query = &data[..d];
        let result1 = index.search(query, 10).unwrap();
        let result2 = loaded.search(query, 10).unwrap();

        let ids1: Vec<u32> = result1.iter().map(|(_, id)| *id).collect();
        let ids2: Vec<u32> = result2.iter().map(|(_, id)| *id).collect();
        assert_eq!(ids1, ids2);
        assert!(loaded.pca.is_some());
        assert_eq!(loaded.sq_codes.len(), index.sq_codes.len());
    }
}

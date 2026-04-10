/// DiskANN-PCA-USQ index scaffold
///
/// Mirrors DiskAnnSqIndex but uses ExRaBitQ (USQ) for vector compression
/// instead of SQ8. The DiskANN Vamana graph drives search; USQ codes are
/// stored for future re-ranking (Phase 2).
///
/// Build path:
///   raw [N×D] → PQFlashIndex (DiskANN graph, original vectors)
///            ↘ PcaTransform → projected [N×D'] → UsqQuantizer::encode → codes
///
/// Search path (Phase 1):
///   query → PQFlashIndex::search → top-k
///
/// Phase 2 (future): re-rank PQFlashIndex candidates using USQ asymmetric distance.
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::api::{KnowhereError, MetricType, Result};
use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use crate::quantization::{
    usq::{UsqConfig, UsqEncoded, UsqQuantizer},
    PcaTransform,
};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DiskAnnPcaUsqConfig {
    pub base: AisaqConfig,
    /// Reduce to this many dimensions before USQ encoding (0 = no PCA)
    pub pca_dim: usize,
    /// ExRaBitQ bits per projected dimension: 1, 4, or 8
    pub bits_per_dim: u8,
    /// Random rotation seed
    pub rotation_seed: u64,
    /// Re-rank pool size (candidates from graph to score with USQ)
    pub rerank_k: usize,
}

impl Default for DiskAnnPcaUsqConfig {
    fn default() -> Self {
        Self {
            base: AisaqConfig::default(),
            pca_dim: 64,
            bits_per_dim: 4,
            rotation_seed: 42,
            rerank_k: 64,
        }
    }
}

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct DiskAnnPcaUsqIndex {
    inner: PQFlashIndex,
    metric_type: MetricType,
    quantizer: UsqQuantizer,
    codes: Vec<UsqEncoded>,
    pca: Option<PcaTransform>,
    n: usize,
    d_in: usize,
    d_proj: usize,
    config: DiskAnnPcaUsqConfig,
}

impl DiskAnnPcaUsqIndex {
    pub fn new(dim: usize, metric_type: MetricType, config: DiskAnnPcaUsqConfig) -> Result<Self> {
        if dim == 0 {
            return Err(KnowhereError::InvalidArg("dim must be > 0".into()));
        }
        let d_proj = if config.pca_dim > 0 && config.pca_dim < dim {
            config.pca_dim
        } else {
            dim
        };
        let usq_config = UsqConfig::new(d_proj, config.bits_per_dim)
            .map_err(|e| KnowhereError::InvalidArg(format!("USQ config error: {e}")))?
            .with_seed(config.rotation_seed);
        let inner = PQFlashIndex::new(config.base.clone(), metric_type, dim)?;

        Ok(Self {
            inner,
            metric_type,
            quantizer: UsqQuantizer::new(usq_config),
            codes: Vec::new(),
            pca: None,
            n: 0,
            d_in: dim,
            d_proj,
            config,
        })
    }

    /// Build the index from raw vectors.
    pub fn build(&mut self, data: &[f32], n: usize, d: usize) -> Result<()> {
        if d != self.d_in {
            return Err(KnowhereError::InvalidArg(format!(
                "dimension mismatch: expected {}, got {}",
                self.d_in, d
            )));
        }
        if n.checked_mul(d).unwrap_or(0) != data.len() {
            return Err(KnowhereError::InvalidArg("input shape mismatch".into()));
        }

        // PCA projection
        let projected = if self.d_proj < d {
            let pca = PcaTransform::train(data, n, d, self.d_proj);
            let proj = pca.apply(data, n);
            self.pca = Some(pca);
            proj
        } else {
            self.pca = None;
            data.to_vec()
        };

        // USQ centroid = global mean of projected data
        let mut centroid = vec![0.0f32; self.d_proj];
        for row in projected.chunks_exact(self.d_proj) {
            for (c, &v) in centroid.iter_mut().zip(row.iter()) {
                *c += v;
            }
        }
        let inv_n = 1.0 / n as f32;
        for c in &mut centroid {
            *c *= inv_n;
        }
        self.quantizer.set_centroid(&centroid);

        // Encode projected vectors
        self.codes.clear();
        self.codes.reserve(n);
        for row in projected.chunks_exact(self.d_proj) {
            self.codes.push(self.quantizer.encode(row));
        }

        // DiskANN graph on original vectors
        self.inner.add_with_ids(data, None)?;
        self.n = n;
        Ok(())
    }

    /// Search k-NN (Phase 1: graph search only; USQ re-ranking in Phase 2).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(f32, u32)>> {
        let result = self.inner.search(query, k)?;
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let dist = result.distances.get(i).copied().unwrap_or(f32::MAX);
            let id = result.ids.get(i).copied().unwrap_or(-1);
            let mapped = if id >= 0 { id as u32 } else { u32::MAX };
            out.push((dist, mapped));
        }
        Ok(out)
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn orig_dim(&self) -> usize {
        self.d_in
    }

    pub fn proj_dim(&self) -> usize {
        self.d_proj
    }

    // ── serialization ─────────────────────────────────────────────────────────

    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;

        // DiskANN graph
        let graph_path = dir.join("diskann_graph.bin");
        self.inner.save(&graph_path)?;

        // PCA-USQ metadata
        let meta_path = dir.join("diskann_pca_usq.meta");
        let mut f = File::create(meta_path)?;

        f.write_all(b"DAPUSQ")?; // magic
        f.write_all(&1u32.to_le_bytes())?; // version

        // scalar fields
        write_u64(&mut f, metric_to_u32(self.metric_type) as u64)?;
        write_u64(&mut f, self.n as u64)?;
        write_u64(&mut f, self.d_in as u64)?;
        write_u64(&mut f, self.d_proj as u64)?;
        write_u64(&mut f, self.config.pca_dim as u64)?;
        f.write_all(&[self.config.bits_per_dim])?;
        f.write_all(&self.config.rotation_seed.to_le_bytes())?;
        write_u64(&mut f, self.config.rerank_k as u64)?;

        // PCA transform
        write_pca_transform(&mut f, self.pca.as_ref())?;

        // USQ quantizer centroid
        {
            let centroid = self.quantizer.centroid();
            f.write_all(&1u8.to_le_bytes())
                .map_err(|e| KnowhereError::Codec(e.to_string()))?;
            write_u64(&mut f, centroid.len() as u64)?;
            for &v in centroid {
                f.write_all(&v.to_le_bytes())
                    .map_err(|e| KnowhereError::Codec(e.to_string()))?;
            }
        }

        // USQ encoded codes
        write_u64(&mut f, self.codes.len() as u64)?;
        for enc in &self.codes {
            write_u64(&mut f, enc.packed_bits.len() as u64)?;
            f.write_all(&enc.packed_bits)?;
            write_u64(&mut f, enc.sign_bits.len() as u64)?;
            f.write_all(&enc.sign_bits)?;
            f.write_all(&enc.norm.to_le_bytes())?;
            f.write_all(&enc.norm_sq.to_le_bytes())?;
            f.write_all(&enc.vmax.to_le_bytes())?;
            f.write_all(&enc.quant_quality.to_le_bytes())?;
        }

        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let graph_path = dir.join("diskann_graph.bin");
        let meta_path = dir.join("diskann_pca_usq.meta");
        let inner = PQFlashIndex::load(&graph_path)?;
        let mut f = File::open(meta_path)?;

        let mut magic = [0u8; 6];
        f.read_exact(&mut magic)?;
        if &magic != b"DAPUSQ" {
            return Err(KnowhereError::Codec(
                "invalid DiskAnnPcaUsqIndex magic".to_string(),
            ));
        }

        let version = read_u32(&mut f)?;
        if version != 1 {
            return Err(KnowhereError::Codec(format!(
                "unsupported DiskAnnPcaUsqIndex version {version}"
            )));
        }

        let metric_type = metric_from_u32(read_u64(&mut f)? as u32)?;
        let n = read_u64(&mut f)? as usize;
        let d_in = read_u64(&mut f)? as usize;
        let d_proj = read_u64(&mut f)? as usize;
        let pca_dim = read_u64(&mut f)? as usize;
        let bits_per_dim = read_u8(&mut f)?;
        let rotation_seed = read_u64(&mut f)?;
        let rerank_k = read_u64(&mut f)? as usize;
        let pca = read_pca_transform(&mut f)?;

        let mut quantizer = UsqQuantizer::new(
            UsqConfig::new(d_proj, bits_per_dim)
                .map_err(|e| KnowhereError::Codec(format!("USQ config error: {e}")))?
                .with_seed(rotation_seed),
        );

        if read_u8(&mut f)? != 0 {
            let centroid = read_vec_f32(&mut f)?;
            if centroid.len() != d_proj {
                return Err(KnowhereError::Codec(format!(
                    "invalid centroid length: expected {d_proj}, got {}",
                    centroid.len()
                )));
            }
            quantizer.set_centroid(&centroid);
        }

        let code_count = read_u64(&mut f)? as usize;
        let mut codes = Vec::with_capacity(code_count);
        for _ in 0..code_count {
            let packed_bits = read_vec_u8(&mut f)?;
            let sign_bits = read_vec_u8(&mut f)?;
            codes.push(UsqEncoded {
                packed_bits,
                sign_bits,
                norm: read_f32(&mut f)?,
                norm_sq: read_f32(&mut f)?,
                vmax: read_f32(&mut f)?,
                quant_quality: read_f32(&mut f)?,
            });
        }

        Ok(Self {
            inner,
            metric_type,
            quantizer,
            codes,
            pca,
            n,
            d_in,
            d_proj,
            config: DiskAnnPcaUsqConfig {
                base: AisaqConfig::default(),
                pca_dim,
                bits_per_dim,
                rotation_seed,
                rerank_k,
            },
        })
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn metric_to_u32(m: MetricType) -> u32 {
    match m {
        MetricType::L2 => 0,
        MetricType::Cosine => 1,
        _ => 99,
    }
}

fn metric_from_u32(metric: u32) -> Result<MetricType> {
    match metric {
        0 => Ok(MetricType::L2),
        1 => Ok(MetricType::Cosine),
        other => Err(KnowhereError::Codec(format!(
            "unknown DiskAnnPcaUsq metric type tag {other}"
        ))),
    }
}

fn read_u8(f: &mut impl Read) -> Result<u8> {
    let mut b = [0u8; 1];
    f.read_exact(&mut b)
        .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    Ok(b[0])
}

fn read_u32(f: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)
        .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    Ok(u32::from_le_bytes(b))
}

fn write_u64(f: &mut impl Write, v: u64) -> Result<()> {
    f.write_all(&v.to_le_bytes())
        .map_err(|e| KnowhereError::Codec(e.to_string()))
}

fn read_u64(f: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)
        .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32(f: &mut impl Read) -> Result<f32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)
        .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    Ok(f32::from_le_bytes(b))
}

fn read_vec_f32(f: &mut impl Read) -> Result<Vec<f32>> {
    let len = read_u64(f)? as usize;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_f32(f)?);
    }
    Ok(values)
}

fn read_vec_u8(f: &mut impl Read) -> Result<Vec<u8>> {
    let len = read_u64(f)? as usize;
    let mut values = vec![0u8; len];
    f.read_exact(&mut values)
        .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    Ok(values)
}

fn write_pca_transform(f: &mut impl Write, pca: Option<&PcaTransform>) -> Result<()> {
    if let Some(p) = pca {
        f.write_all(&1u8.to_le_bytes())
            .map_err(|e| KnowhereError::Codec(e.to_string()))?;
        write_u64(f, p.d_in as u64)?;
        write_u64(f, p.d_out as u64)?;
        write_u64(f, p.mean.len() as u64)?;
        for &v in &p.mean {
            f.write_all(&v.to_le_bytes())
                .map_err(|e| KnowhereError::Codec(e.to_string()))?;
        }
        write_u64(f, p.components.len() as u64)?;
        for &v in &p.components {
            f.write_all(&v.to_le_bytes())
                .map_err(|e| KnowhereError::Codec(e.to_string()))?;
        }
    } else {
        f.write_all(&0u8.to_le_bytes())
            .map_err(|e| KnowhereError::Codec(e.to_string()))?;
    }
    Ok(())
}

fn read_pca_transform(f: &mut impl Read) -> Result<Option<PcaTransform>> {
    if read_u8(f)? == 0 {
        return Ok(None);
    }

    Ok(Some(PcaTransform {
        d_in: read_u64(f)? as usize,
        d_out: read_u64(f)? as usize,
        mean: read_vec_f32(f)?,
        components: read_vec_f32(f)?,
    }))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::tempdir;

    fn make_data(n: usize, d: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(42);
        (0..n * d).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_diskann_pca_usq_build_search() {
        let dim = 16usize;
        let n = 64usize;
        let data = make_data(n, dim);

        let config = DiskAnnPcaUsqConfig {
            base: AisaqConfig {
                max_degree: 16,
                search_list_size: 32,
                ..Default::default()
            },
            pca_dim: 8,
            bits_per_dim: 4,
            rotation_seed: 42,
            rerank_k: 16,
        };

        let mut idx =
            DiskAnnPcaUsqIndex::new(dim, MetricType::L2, config).expect("new failed");
        idx.build(&data, n, dim).expect("build failed");
        assert_eq!(idx.count(), n);
        assert_eq!(idx.proj_dim(), 8);

        let query = make_data(1, dim);
        let results = idx.search(&query, 5).expect("search failed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_diskann_pca_usq_save_load_roundtrip() {
        let dim = 16usize;
        let n = 64usize;
        let data = make_data(n, dim);
        let dir = tempdir().expect("tempdir failed");

        let config = DiskAnnPcaUsqConfig {
            base: AisaqConfig {
                max_degree: 16,
                search_list_size: 32,
                ..Default::default()
            },
            pca_dim: 8,
            bits_per_dim: 4,
            rotation_seed: 7,
            rerank_k: 24,
        };

        let mut idx =
            DiskAnnPcaUsqIndex::new(dim, MetricType::L2, config.clone()).expect("new failed");
        idx.build(&data, n, dim).expect("build failed");
        idx.save(dir.path()).expect("save failed");

        let loaded = DiskAnnPcaUsqIndex::load(dir.path()).expect("load failed");
        assert_eq!(loaded.count(), idx.count());
        assert_eq!(loaded.orig_dim(), idx.orig_dim());
        assert_eq!(loaded.proj_dim(), idx.proj_dim());
        assert_eq!(loaded.config.pca_dim, config.pca_dim);
        assert_eq!(loaded.config.bits_per_dim, config.bits_per_dim);
        assert_eq!(loaded.config.rotation_seed, config.rotation_seed);
        assert_eq!(loaded.config.rerank_k, config.rerank_k);
        assert_eq!(loaded.quantizer.centroid(), idx.quantizer.centroid());
        assert_eq!(loaded.codes.len(), idx.codes.len());

        let query = &data[..dim];
        let results = loaded.search(query, 5).expect("search failed");
        assert_eq!(results.len(), 5);
    }
}

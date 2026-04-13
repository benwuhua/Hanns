/// HNSW-PCA-SQ: PCA dimensionality reduction + HNSW graph + SQ8 quantization
///
/// Build path:
///   raw [N×D] → PcaTransform::train → projected [N×D'] → HnswSqIndex(D')
///
/// Search path:
///   query [D] → pca.apply_one → [D'] → HnswSqIndex::search
use crate::api::{Result, SearchRequest, SearchResult as ApiSearchResult};
use crate::faiss::hnsw_quantized::HnswSqIndex;
use crate::quantization::PcaTransform;

pub struct HnswPcaSqConfig {
    /// Original input dimensionality
    pub dim: usize,
    /// PCA target dimensionality (0 = no PCA, use identity)
    pub pca_dim: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswPcaSqConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            pca_dim: 64,
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }
}

pub struct HnswPcaSqIndex {
    orig_dim: usize,
    proj_dim: usize,
    pca: Option<PcaTransform>,
    inner: HnswSqIndex,
}

impl HnswPcaSqIndex {
    pub fn new(config: HnswPcaSqConfig) -> Self {
        let proj_dim = if config.pca_dim > 0 && config.pca_dim < config.dim {
            config.pca_dim
        } else {
            config.dim
        };

        Self {
            orig_dim: config.dim,
            proj_dim,
            pca: None,
            inner: HnswSqIndex::new(proj_dim),
        }
    }

    /// Train PCA + SQ quantizer on raw vectors.
    pub fn train(&mut self, vectors: &[f32]) -> Result<usize> {
        let n = vectors.len() / self.orig_dim;
        if n == 0 {
            return Err(crate::api::KnowhereError::InvalidArg("empty data".into()));
        }

        let projected = if self.proj_dim < self.orig_dim {
            let pca = PcaTransform::train(vectors, n, self.orig_dim, self.proj_dim);
            let proj = pca.apply(vectors, n);
            self.pca = Some(pca);
            proj
        } else {
            vectors.to_vec()
        };

        self.inner.train(&projected)
    }

    /// Add vectors (must call train first).
    pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        let n = vectors.len() / self.orig_dim;
        let projected = self.project_batch(vectors, n);
        self.inner.add(&projected, ids)
    }

    /// Search k-NN.
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<ApiSearchResult> {
        let q_proj = self.project_one(query);
        self.inner.search(&q_proj, req)
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn orig_dim(&self) -> usize {
        self.orig_dim
    }

    pub fn proj_dim(&self) -> usize {
        self.proj_dim
    }

    // ── serialization ─────────────────────────────────────────────────────────

    pub fn save(
        &self,
        path: &std::path::Path,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let mut f = File::create(path)?;
        // magic + dimensions
        f.write_all(b"HNSWPCASQ")?;
        f.write_all(&(self.orig_dim as u32).to_le_bytes())?;
        f.write_all(&(self.proj_dim as u32).to_le_bytes())?;

        // PCA: present flag + components
        if let Some(pca) = &self.pca {
            f.write_all(&[1u8])?; // has_pca
                                  // mean
            f.write_all(&(pca.mean.len() as u32).to_le_bytes())?;
            for &v in &pca.mean {
                f.write_all(&v.to_le_bytes())?;
            }
            // components
            f.write_all(&(pca.components.len() as u32).to_le_bytes())?;
            for &v in &pca.components {
                f.write_all(&v.to_le_bytes())?;
            }
        } else {
            f.write_all(&[0u8])?; // no_pca
        }

        // inner HnswSqIndex — save to a temp path adjacent to main path
        let inner_path = path.with_extension("hnsw_sq_inner");
        self.inner.save(&inner_path)?;

        // embed inner file length + bytes into the outer file
        let inner_bytes = std::fs::read(&inner_path)?;
        std::fs::remove_file(&inner_path)?;
        f.write_all(&(inner_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&inner_bytes)?;

        Ok(())
    }

    pub fn load(path: &std::path::Path) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        use std::io::{Error, ErrorKind, Read};

        let bytes = std::fs::read(path)?;
        let mut cursor = std::io::Cursor::new(&bytes);

        let mut magic = [0u8; 9];
        cursor.read_exact(&mut magic)?;
        if &magic != b"HNSWPCASQ" {
            return Err(Box::new(Error::new(
                ErrorKind::InvalidData,
                "invalid HNSWPCASQ magic",
            )));
        }

        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let orig_dim = u32::from_le_bytes(u32_buf) as usize;
        cursor.read_exact(&mut u32_buf)?;
        let proj_dim = u32::from_le_bytes(u32_buf) as usize;

        let mut flag = [0u8; 1];
        cursor.read_exact(&mut flag)?;
        let pca = if flag[0] == 1 {
            cursor.read_exact(&mut u32_buf)?;
            let mean_len = u32::from_le_bytes(u32_buf) as usize;
            let mut mean = vec![0.0f32; mean_len];
            for v in &mut mean {
                let mut buf = [0u8; 4];
                cursor.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
            cursor.read_exact(&mut u32_buf)?;
            let comp_len = u32::from_le_bytes(u32_buf) as usize;
            let mut components = vec![0.0f32; comp_len];
            for v in &mut components {
                let mut buf = [0u8; 4];
                cursor.read_exact(&mut buf)?;
                *v = f32::from_le_bytes(buf);
            }
            Some(PcaTransform {
                mean,
                components,
                d_in: orig_dim,
                d_out: proj_dim,
            })
        } else {
            None
        };

        let mut u64_buf = [0u8; 8];
        cursor.read_exact(&mut u64_buf)?;
        let inner_len = u64::from_le_bytes(u64_buf) as usize;

        let pos = cursor.position() as usize;
        let inner_bytes = &bytes[pos..pos + inner_len];

        // write inner to temp path so HnswSqIndex::load can read it
        let tmp = path.with_extension("hnsw_sq_tmp");
        std::fs::write(&tmp, inner_bytes)?;
        let inner = HnswSqIndex::load(&tmp, proj_dim)?;
        std::fs::remove_file(&tmp)?;

        Ok(Self {
            orig_dim,
            proj_dim,
            pca,
            inner,
        })
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
    fn test_hnsw_pca_sq_basic() {
        let dim = 64usize;
        let pca_dim = 32usize;
        let n = 200usize;

        let data = make_data(n, dim);
        let mut idx = HnswPcaSqIndex::new(HnswPcaSqConfig {
            dim,
            pca_dim,
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        });

        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();
        assert_eq!(idx.count(), n);
        assert_eq!(idx.proj_dim(), pca_dim);

        let query = make_data(1, dim);
        let req = SearchRequest {
            top_k: 10,
            nprobe: 50,
            ..Default::default()
        };
        let result = idx.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 10);
    }

    #[test]
    fn test_hnsw_pca_sq_no_pca() {
        let dim = 32usize;
        let n = 100usize;
        let data = make_data(n, dim);

        let mut idx = HnswPcaSqIndex::new(HnswPcaSqConfig {
            dim,
            pca_dim: 0, // no PCA
            m: 8,
            ef_construction: 50,
            ef_search: 20,
        });

        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();
        assert_eq!(idx.proj_dim(), dim);

        let query = make_data(1, dim);
        let req = SearchRequest {
            top_k: 5,
            nprobe: 20,
            ..Default::default()
        };
        let result = idx.search(&query, &req).unwrap();
        assert_eq!(result.ids.len(), 5);
    }

    #[test]
    fn test_hnsw_pca_sq_save_load() {
        let dim = 48usize;
        let pca_dim = 24usize;
        let n = 150usize;
        let data = make_data(n, dim);

        let mut idx = HnswPcaSqIndex::new(HnswPcaSqConfig {
            dim,
            pca_dim,
            m: 8,
            ef_construction: 50,
            ef_search: 20,
        });
        idx.train(&data).unwrap();
        idx.add(&data, None).unwrap();

        let tmp = std::env::temp_dir().join("test_hnsw_pca_sq.bin");
        idx.save(&tmp).unwrap();

        let loaded = HnswPcaSqIndex::load(&tmp).unwrap();
        assert_eq!(loaded.orig_dim(), dim);
        assert_eq!(loaded.proj_dim(), pca_dim);
        assert_eq!(loaded.count(), n);

        std::fs::remove_file(&tmp).ok();
    }
}

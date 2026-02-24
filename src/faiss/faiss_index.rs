//! Faiss Index implementation using real Faiss library

use std::path::Path;

use crate::api::{IndexConfig, IndexType, MetricType, Result, SearchRequest, SearchResult};

/// Faiss index using real C++ library
pub struct FaissIndex {
    config: IndexConfig,
    /// Raw Faiss index pointer
    index: cxx::UniquePtr<ffi::FaissIndex>,
}

impl FaissIndex {
    /// Create a new Faiss index
    pub fn new(config: &IndexConfig) -> Result<Self> {
        if config.dim == 0 {
            return Err(crate::api::KnowhereError::InvalidArg(
                "dimension must be > 0".to_string(),
            ));
        }

        let metric = match config.metric_type {
            MetricType::L2 => 1,  // FAISS_METRIC_L2
            MetricType::Ip => 2,  // FAISS_METRIC_INNER_PRODUCT
            MetricType::Cosine => 2, // Use IP for cosine (normalize vectors)
        };

        let index = match config.index_type {
            IndexType::Flat => {
                ffi::faiss_index_new_flat(config.dim as i32, metric)
            }
            IndexType::IvfFlat => {
                // For IVF, create a flat quantizer first
                let quantizer = ffi::faiss_index_new_flat(config.dim as i32, metric);
                let nlist = config.params.nlist.unwrap_or(100) as i32;
                ffi::faiss_index_new_ivf_flat(&quantizer, config.dim as i32, nlist, metric)
            }
            _ => {
                // Fallback to flat for unsupported types
                tracing::warn!("Index type {:?} not supported, using Flat", config.index_type);
                ffi::faiss_index_new_flat(config.dim as i32, metric)
            }
        };

        if index.is_null() {
            return Err(crate::api::KnowhereError::Faiss(
                "Failed to create Faiss index".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
            index,
        })
    }

    /// Train the index
    pub fn train(&mut self, vectors: &[f32]) -> Result<()> {
        let n = vectors.len() / self.config.dim;
        if n * self.config.dim != vectors.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "vector dimension mismatch".to_string(),
            ));
        }

        let ret = unsafe {
            self.index.pin_mut().faiss_index_train(vectors.as_ptr(), n as i64)
        };

        if ret != 0 {
            return Err(crate::api::KnowhereError::Faiss(
                format!("Training failed with code {}", ret),
            ));
        }

        Ok(())
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: &[f32], _ids: Option<&[i64]>) -> Result<usize> {
        // Note: Faiss doesn't support custom IDs directly, we use internal IDs
        let n = vectors.len() / self.config.dim;
        
        unsafe {
            self.index.pin_mut().faiss_index_add(vectors.as_ptr(), n as i64);
        }

        Ok(n)
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        let n = query.len() / self.config.dim;
        if n * self.config.dim != query.len() {
            return Err(crate::api::KnowhereError::InvalidArg(
                "query dimension mismatch".to_string(),
            ));
        }

        let k = req.top_k as i32;
        
        // Allocate output buffers
        let mut distances = vec![0.0f32; n * k as usize];
        let mut labels = vec![-1i64; n * k as usize];

        unsafe {
            let ret = self.index.pin_mut().faiss_index_search(
                query.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );

            if ret != 0 {
                return Err(crate::api::KnowhereError::Faiss(
                    format!("Search failed with code {}", ret),
                ));
            }
        }

        // Convert to our result format
        let ids: Vec<i64> = labels;
        let dists: Vec<f32> = distances;

        Ok(SearchResult::new(ids, dists, 0.0))
    }

    /// Get number of vectors
    pub fn ntotal(&self) -> usize {
        self.index.faiss_index_ntotal() as usize
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.index.faiss_index_is_trained()
    }

    /// Save index to file
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::ffi::CString;
        
        let path_str = CString::new(path.to_string_lossy().as_bytes())?;
        
        let ret = unsafe {
            self.index.faiss_index_write(path_str.as_ptr())
        };

        if ret != 0 {
            return Err(crate::api::KnowhereError::Io(
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Save failed with code {}", ret),
                ),
            ));
        }

        Ok(())
    }

    /// Load index from file
    pub fn load(&mut self, path: &Path) -> Result<()> {
        use std::ffi::CString;
        
        let path_str = CString::new(path.to_string_lossy().as_bytes())?;
        
        let new_index = unsafe {
            ffi::faiss_index_read(path_str.as_ptr())
        };

        if new_index.is_null() {
            return Err(crate::api::KnowhereError::Io(
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Failed to load index",
                ),
            ));
        }

        self.index = new_index;
        Ok(())
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.config.dim
    }
}

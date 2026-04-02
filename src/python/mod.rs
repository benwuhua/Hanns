//! Python 绑定 - 供 Python 调用
//!
//! # Python 使用示例
//! ```python
//! import knowhere_rs
//!
//! # 创建索引
//! index = knowhere_rs.Index(
//!     index_type="hnsw",
//!     dimension=128,
//!     metric_type="l2"
//! )
//!
//! # 添加向量
//! import numpy as np
//! vectors = np.random.rand(1000, 128).astype(np.float32)
//! ids = np.arange(1000, dtype=np.int64)
//! index.add(vectors, ids)
//!
//! # 搜索
//! query = np.random.rand(1, 128).astype(np.float32)
//! result = index.search(query, k=10)
//! print(result.ids, result.distances)
//!
//! # 序列化
//! index.save("index.bin")
//! index2 = knowhere_rs.Index.load("index.bin")
//! ```
#![allow(clippy::useless_conversion)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

use crate::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest, SearchResult};
use crate::faiss::{HnswIndex, IvfUsqIndex, IvfFlatIndex, IvfPqIndex, MemIndex};

/// 内部索引枚举（避免 trait object 问题）
enum InnerIndex {
    Flat(Box<MemIndex>),
    Hnsw(Box<HnswIndex>),
    IvfFlat(Box<IvfFlatIndex>),
    IvfPq(Box<IvfPqIndex>),
    IvfUsq(Box<IvfUsqIndex>),
}

impl InnerIndex {
    fn train(&mut self, vectors: &[f32]) -> crate::api::Result<()> {
        match self {
            InnerIndex::Flat(idx) => idx.train(vectors),
            InnerIndex::Hnsw(idx) => idx.train(vectors),
            InnerIndex::IvfFlat(idx) => idx.train(vectors).map(|_| ()),
            InnerIndex::IvfPq(idx) => idx.train(vectors),
            InnerIndex::IvfUsq(idx) => idx.train(vectors),
        }
    }

    fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> crate::api::Result<usize> {
        match self {
            InnerIndex::Flat(idx) => idx.add(vectors, ids),
            InnerIndex::Hnsw(idx) => idx.add(vectors, ids),
            InnerIndex::IvfFlat(idx) => idx.add(vectors, ids),
            InnerIndex::IvfPq(idx) => idx.add(vectors, ids),
            InnerIndex::IvfUsq(idx) => idx.add(vectors, ids),
        }
    }

    fn search(&self, query: &[f32], req: &SearchRequest) -> crate::api::Result<SearchResult> {
        match self {
            InnerIndex::Flat(idx) => idx.search(query, req),
            InnerIndex::Hnsw(idx) => {
                let api_result = idx.search(query, req)?;
                // Convert ApiSearchResult to SearchResult
                Ok(SearchResult {
                    ids: api_result.ids,
                    distances: api_result.distances,
                    elapsed_ms: api_result.elapsed_ms,
                    num_visited: 0, // HNSW doesn't track this
                })
            }
            InnerIndex::IvfFlat(idx) => idx.search(query, req),
            InnerIndex::IvfPq(idx) => idx.search(query, req),
            InnerIndex::IvfUsq(idx) => idx.search(query, req),
        }
    }

    fn save(&self, path: &Path) -> crate::api::Result<()> {
        match self {
            InnerIndex::Flat(idx) => idx.save(path),
            InnerIndex::Hnsw(idx) => idx.save(path),
            InnerIndex::IvfFlat(_) => Err(crate::api::KnowhereError::InvalidArg(
                "IVF-Flat save is not implemented".to_string(),
            )),
            InnerIndex::IvfPq(idx) => idx.save(path),
            InnerIndex::IvfUsq(idx) => idx.save(path),
        }
    }

    fn load(&mut self, path: &Path) -> crate::api::Result<()> {
        match self {
            InnerIndex::Flat(idx) => idx.load(path),
            InnerIndex::Hnsw(idx) => idx.load(path),
            InnerIndex::IvfFlat(_) => Err(crate::api::KnowhereError::InvalidArg(
                "IVF-Flat load is not implemented".to_string(),
            )),
            InnerIndex::IvfPq(idx) => idx.load(path),
            InnerIndex::IvfUsq(idx) => {
                *idx = Box::new(IvfUsqIndex::load(path)?);
                Ok(())
            }
        }
    }

    fn ntotal(&self) -> usize {
        match self {
            InnerIndex::Flat(idx) => idx.ntotal(),
            InnerIndex::Hnsw(idx) => idx.ntotal(),
            InnerIndex::IvfFlat(idx) => idx.ntotal(),
            InnerIndex::IvfPq(idx) => idx.ntotal(),
            InnerIndex::IvfUsq(idx) => idx.ntotal(),
        }
    }

    fn index_type(&self) -> &str {
        match self {
            InnerIndex::Flat(_) => "flat",
            InnerIndex::Hnsw(_) => "hnsw",
            InnerIndex::IvfFlat(_) => "ivf_flat",
            InnerIndex::IvfPq(_) => "ivf_pq",
            InnerIndex::IvfUsq(_) => "ivf_usq",
        }
    }
}

/// Python 索引包装
#[pyclass(name = "Index")]
pub struct PyIndex {
    inner: Arc<RwLock<InnerIndex>>,
    config: IndexConfig,
}

#[allow(clippy::useless_conversion)]
#[pymethods]
impl PyIndex {
    /// 创建新索引
    #[new]
    #[pyo3(signature = (index_type, dimension, metric_type, ef_construction=None, ef_search=None, m=None, nlist=None, nprobe=None, nbits=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        index_type: &str,
        dimension: usize,
        metric_type: &str,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        m: Option<usize>,
        nlist: Option<usize>,
        nprobe: Option<usize>,
        nbits: Option<usize>,
    ) -> PyResult<Self> {
        let index_type_enum = match index_type.to_lowercase().as_str() {
            "flat" => IndexType::Flat,
            "hnsw" => IndexType::Hnsw,
            "ivf_flat" | "ivfflat" => IndexType::IvfFlat,
            "ivf_pq" | "ivfpq" => IndexType::IvfPq,
            "ivf_exrabitq" | "ivf-exrabitq" | "exrabitq" | "extended-rabitq"
            | "ivf_usq" | "ivf-usq" | "usq" => {
                IndexType::IvfUsq
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                "Unknown index type: {} (supported: flat, hnsw, ivf_flat, ivf_pq, ivf_exrabitq)",
                index_type
            )))
            }
        };

        let metric_type_enum = match metric_type.to_lowercase().as_str() {
            "l2" => MetricType::L2,
            "ip" | "inner_product" => MetricType::Ip,
            "cosine" => MetricType::Cosine,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric type: {}",
                    metric_type
                )))
            }
        };

        let config = IndexConfig {
            index_type: index_type_enum,
            dim: dimension,
            metric_type: metric_type_enum,
            data_type: crate::api::DataType::Float,
            params: IndexParams {
                ef_construction: Some(ef_construction.unwrap_or(400)),
                ef_search: Some(ef_search.unwrap_or(128)),
                m: Some(m.unwrap_or(8)), // IVF-PQ default: 8 sub-quantizers
                nlist: Some(nlist.unwrap_or(100)), // IVF default: 100 clusters
                nprobe: Some(nprobe.unwrap_or(8)), // IVF default: 8 probes
                nbits_per_idx: Some(nbits.unwrap_or(8)), // PQ default: 8 bits
                exrabitq_bits_per_dim: Some(nbits.unwrap_or(4)),
                ..Default::default()
            },
        };

        let index = match index_type_enum {
            IndexType::Flat => InnerIndex::Flat(Box::new(MemIndex::new(&config).map_err(|e| {
                PyValueError::new_err(format!("Failed to create Flat index: {:?}", e))
            })?)),
            IndexType::Hnsw => {
                InnerIndex::Hnsw(Box::new(HnswIndex::new(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create HNSW index: {:?}", e))
                })?))
            }
            IndexType::IvfFlat => {
                InnerIndex::IvfFlat(Box::new(IvfFlatIndex::new(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create IVF-Flat index: {:?}", e))
                })?))
            }
            IndexType::IvfPq => {
                InnerIndex::IvfPq(Box::new(IvfPqIndex::new(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create IVF-PQ index: {:?}", e))
                })?))
            }
            IndexType::IvfUsq => InnerIndex::IvfUsq(Box::new(
                IvfUsqIndex::from_index_config(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create IVF-ExRaBitQ index: {:?}", e))
                })?,
            )),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported index type: {:?}",
                    index_type_enum
                )))
            }
        };

        Ok(PyIndex {
            inner: Arc::new(RwLock::new(index)),
            config,
        })
    }

    /// 训练索引
    fn train(&mut self, py: Python, data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let data = data.as_array();
        let dim = data.ncols();

        if dim != self.config.dim {
            return Err(PyValueError::new_err(format!(
                "Data dimension {} doesn't match index dimension {}",
                dim, self.config.dim
            )));
        }

        let data_slice = data
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Data is not contiguous"))?;

        py.allow_threads(|| {
            self.inner
                .write()
                .train(data_slice)
                .map_err(|e| PyValueError::new_err(format!("Training failed: {:?}", e)))
        })
    }

    /// 添加向量
    fn add(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<usize> {
        let data = data.as_array();
        let ids = ids.as_array();

        let n = data.nrows();
        let dim = data.ncols();

        if dim != self.config.dim {
            return Err(PyValueError::new_err(format!(
                "Data dimension {} doesn't match index dimension {}",
                dim, self.config.dim
            )));
        }

        if ids.len() != n {
            return Err(PyValueError::new_err(format!(
                "Number of IDs {} doesn't match number of vectors {}",
                ids.len(),
                n
            )));
        }

        let data_slice = data
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Data is not contiguous"))?;

        let ids_slice: Vec<i64> = ids.to_vec();

        py.allow_threads(|| {
            self.inner
                .write()
                .add(data_slice, Some(&ids_slice))
                .map_err(|e| PyValueError::new_err(format!("Add failed: {:?}", e)))
        })
    }

    /// 搜索
    fn search(
        &self,
        py: Python,
        query: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<PySearchResult> {
        let query = query.as_array();
        let _n = query.nrows();
        let dim = query.ncols();

        if dim != self.config.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} doesn't match index dimension {}",
                dim, self.config.dim
            )));
        }

        let query_slice = query
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("Query is not contiguous"))?;

        let req = SearchRequest {
            top_k: k,
            nprobe: 10,
            filter: None,
            params: None,
            radius: None,
        };

        let result = py.allow_threads(|| {
            self.inner
                .read()
                .search(query_slice, &req)
                .map_err(|e| PyValueError::new_err(format!("Search failed: {:?}", e)))
        })?;

        Ok(PySearchResult { inner: result })
    }

    /// 序列化到文件
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .read()
            .save(Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("Save failed: {:?}", e)))
    }

    /// 从文件反序列化
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        // Read magic number to determine index type and dimension
        let mut file = File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        let (index_type_enum, config) = match &magic {
            b"KWFL" => {
                // Flat index: KWFL + version(4) + dim(4) + num(8)
                let mut version = [0u8; 4];
                file.read_exact(&mut version)?;

                let mut dim_bytes = [0u8; 4];
                file.read_exact(&mut dim_bytes)?;
                let dim = u32::from_le_bytes(dim_bytes) as usize;

                (
                    IndexType::Flat,
                    IndexConfig {
                        index_type: IndexType::Flat,
                        dim,
                        metric_type: MetricType::L2,
                        data_type: crate::api::DataType::Float,
                        params: IndexParams::default(),
                    },
                )
            }
            b"HNSW" => {
                // HNSW index: HNSW + version(4) + dim(4) + ...
                let mut version = [0u8; 4];
                file.read_exact(&mut version)?;

                let mut dim_bytes = [0u8; 4];
                file.read_exact(&mut dim_bytes)?;
                let dim = u32::from_le_bytes(dim_bytes) as usize;

                (
                    IndexType::Hnsw,
                    IndexConfig {
                        index_type: IndexType::Hnsw,
                        dim,
                        metric_type: MetricType::L2,
                        data_type: crate::api::DataType::Float,
                        params: IndexParams::default(),
                    },
                )
            }
            _ => {
                // Try IVFPQ (5 bytes)
                file.seek(SeekFrom::Start(0))?;
                let mut magic5 = [0u8; 5];
                file.read_exact(&mut magic5)?;

                if &magic5 == b"IVFPQ" {
                    // IVFPQ: IVFPQ(5) + version(4) + dim(4) + nlist(4) + m(4) + nbits(4)
                    let mut version = [0u8; 4];
                    file.read_exact(&mut version)?;

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

                    (
                        IndexType::IvfPq,
                        IndexConfig {
                            index_type: IndexType::IvfPq,
                            dim,
                            metric_type: MetricType::L2,
                            data_type: crate::api::DataType::Float,
                            params: IndexParams {
                                nlist: Some(nlist),
                                m: Some(m),
                                nbits_per_idx: Some(nbits),
                                ..Default::default()
                            },
                        },
                    )
                } else if &magic == b"IVFX" {
                    file.seek(SeekFrom::Start(0))?;
                    let mut magic8 = [0u8; 8];
                    file.read_exact(&mut magic8)?;
                    if &magic8 != b"IVFXRBTQ" {
                        return Err(PyValueError::new_err(format!(
                            "Invalid magic number: {:?}",
                            std::str::from_utf8(&magic)
                        )));
                    }

                    let mut version = [0u8; 4];
                    file.read_exact(&mut version)?;

                    let mut dim_bytes = [0u8; 4];
                    file.read_exact(&mut dim_bytes)?;
                    let dim = u32::from_le_bytes(dim_bytes) as usize;

                    (
                        IndexType::IvfUsq,
                        IndexConfig {
                            index_type: IndexType::IvfUsq,
                            dim,
                            metric_type: MetricType::L2,
                            data_type: crate::api::DataType::Float,
                            params: IndexParams::ivf_exrabitq(100, 8, 4),
                        },
                    )
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Invalid magic number: {:?}. Expected KWFL (Flat), HNSW (HNSW), IVFPQ (IVF-PQ), or IVFXRBTQ (IVF-ExRaBitQ)",
                        std::str::from_utf8(&magic)
                    )));
                }
            }
        };

        drop(file);

        let mut index = match index_type_enum {
            IndexType::Flat => InnerIndex::Flat(Box::new(MemIndex::new(&config).map_err(|e| {
                PyValueError::new_err(format!("Failed to create Flat index: {:?}", e))
            })?)),
            IndexType::Hnsw => {
                InnerIndex::Hnsw(Box::new(HnswIndex::new(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create HNSW index: {:?}", e))
                })?))
            }
            IndexType::IvfPq => {
                InnerIndex::IvfPq(Box::new(IvfPqIndex::new(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create IVF-PQ index: {:?}", e))
                })?))
            }
            IndexType::IvfUsq => InnerIndex::IvfUsq(Box::new(
                IvfUsqIndex::from_index_config(&config).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create IVF-ExRaBitQ index: {:?}", e))
                })?,
            )),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported index type: {:?}",
                    index_type_enum
                )))
            }
        };

        // Load from file
        index.load(Path::new(path)).map_err(|e| {
            PyValueError::new_err(format!("Failed to load index from {}: {:?}", path, e))
        })?;

        Ok(PyIndex {
            inner: Arc::new(RwLock::new(index)),
            config,
        })
    }

    /// 获取向量数量
    fn count(&self) -> usize {
        self.inner.read().ntotal()
    }

    /// 获取维度
    fn dimension(&self) -> usize {
        self.config.dim
    }

    /// 获取索引类型
    fn index_type(&self) -> String {
        self.inner.read().index_type().to_string()
    }
}

/// Python 搜索结果
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
    /// 获取结果 ID（返回 numpy 数组）
    #[getter]
    fn ids<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<i64>>> {
        Ok(self.inner.ids.clone().into_pyarray_bound(py))
    }

    /// 获取结果距离（返回 numpy 数组）
    #[getter]
    fn distances<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray1<f32>>> {
        Ok(self.inner.distances.clone().into_pyarray_bound(py))
    }

    /// 获取结果数量
    fn len(&self) -> usize {
        self.inner.ids.len()
    }
}

/// Python 模块 (PyO3 0.22 API)
#[pymodule]
#[pyo3(name = "knowhere_rs")]
fn knowhere_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PySearchResult>()?;

    // 添加版本信息
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_index() {
        // 测试创建 Flat 索引
        let index = PyIndex::new("flat", 128, "l2", None, None, None, None, None, None);
        assert!(index.is_ok());

        // 测试创建 HNSW 索引
        let index = PyIndex::new(
            "hnsw",
            128,
            "l2",
            Some(400),
            Some(128),
            Some(16),
            None,
            None,
            None,
        );
        assert!(index.is_ok());

        // 测试创建 IVF-ExRaBitQ 索引
        let index = PyIndex::new(
            "ivf_exrabitq",
            128,
            "l2",
            None,
            None,
            None,
            Some(32),
            Some(4),
            Some(4),
        )
        .expect("create ivf_exrabitq");
        assert_eq!(index.index_type(), "ivf_exrabitq");
    }

    #[test]
    fn test_invalid_index_type() {
        let result = PyIndex::new("invalid", 128, "l2", None, None, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_metric_type() {
        let result = PyIndex::new("flat", 128, "invalid", None, None, None, None, None, None);
        assert!(result.is_err());
    }
}

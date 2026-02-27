//! C API 绑定定义
//! 
//! 供 Milvus C++ 调用
//! 
//! # C API 使用示例
//! ```c
//! // 创建索引
//! CIndexConfig config = {
//!     .index_type = CIndexType_Flat,
//!     .dim = 128,
//!     .metric_type = 0,  // L2
//! };
//! CIndex* index = knowhere_create_index(config);
//! 
//! // 添加向量
//! float vectors[] = { ... };  // 1000 vectors * 128 dim
//! int64_t ids[] = { 0, 1, 2, ... };
//! knowhere_add_index(index, vectors, ids, 1000, 128);
//! 
//! // 搜索
//! float query[] = { ... };  // 1 * 128 dim
//! CSearchResult* result = knowhere_search(index, query, 1, 10, 128);
//! 
//! // 获取结果
//! for (size_t i = 0; i < result->num_results; i++) {
//!     int64_t id = result->ids[i];
//!     float dist = result->distances[i];
//!     printf("id=%ld, dist=%f\n", id, dist);
//! }
//! 
//! // 释放
//! knowhere_free_result(result);
//! knowhere_free_index(index);
//! ```

use std::path::Path;
use crate::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest, SearchResult as ApiSearchResult, Result as ApiResult};
use crate::faiss::{MemIndex, HnswIndex, ScaNNIndex, ScaNNConfig};

/// C API 错误码
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CError {
    Success = 0,
    NotFound = 1,
    InvalidArg = 2,
    Internal = 3,
    NotImplemented = 4,
    OutOfMemory = 5,
}

/// Index 类型枚举（C ABI 兼容）
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CIndexType {
    Flat = 0,
    Hnsw = 1,
    Scann = 2,
}

/// Metric 类型枚举
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CMetricType {
    L2 = 0,
    Ip = 1,
    Cosine = 2,
}

/// Index 配置（C ABI 兼容）
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CIndexConfig {
    pub index_type: CIndexType,
    pub metric_type: CMetricType,
    pub dim: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for CIndexConfig {
    fn default() -> Self {
        Self {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 0,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

/// C 风格的搜索结果
#[repr(C)]
#[derive(Debug)]
pub struct CSearchResult {
    pub ids: *mut i64,
    pub distances: *mut f32,
    pub num_results: usize,
    pub elapsed_ms: f32,
}

/// 包装索引对象 - 只支持 Flat 和 HNSW
struct IndexWrapper {
    flat: Option<MemIndex>,
    hnsw: Option<HnswIndex>,
    dim: usize,
}

impl IndexWrapper {
    fn new(config: CIndexConfig) -> Option<Self> {
        let dim = config.dim;
        if dim == 0 {
            return None;
        }
        
        let metric: MetricType = match config.metric_type {
            CMetricType::L2 => MetricType::L2,
            CMetricType::Ip => MetricType::Ip,
            CMetricType::Cosine => MetricType::Cosine,
        };
        
        match config.index_type {
            CIndexType::Flat => {
                let index_config = IndexConfig {
                    index_type: IndexType::Flat,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                let flat = MemIndex::new(&index_config).ok()?;
                Some(Self { flat: Some(flat), hnsw: None, dim })
            }
            CIndexType::Hnsw => {
                let mut index_config = IndexConfig {
                    index_type: IndexType::Hnsw,
                    metric_type: metric,
                    dim,
                    params: IndexParams::default(),
                };
                if config.ef_construction > 0 {
                    index_config.params.ef_construction = Some(config.ef_construction);
                }
                if config.ef_search > 0 {
                    index_config.params.ef_search = Some(config.ef_search);
                }
                let hnsw = HnswIndex::new(&index_config).ok()?;
                Some(Self { flat: None, hnsw: Some(hnsw), dim })
            }
            CIndexType::Scann => {
                // SCANN not yet supported in FFI
                None
            }
        }
    }
    
    fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize, CError> {
        if let Some(ref mut idx) = self.flat {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw {
            idx.add(vectors, ids).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn train(&mut self, vectors: &[f32]) -> Result<(), CError> {
        if let Some(ref mut idx) = self.flat {
            idx.train(vectors).map_err(|_| CError::Internal)
        } else if let Some(ref mut idx) = self.hnsw {
            idx.train(vectors).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn search(&self, query: &[f32], top_k: usize) -> Result<ApiSearchResult, CError> {
        let req = SearchRequest {
            top_k,
            nprobe: 8,
            filter: None,
            params: None,
            radius: None,
        };
        
        if let Some(ref idx) = self.flat {
            idx.search(query, &req).map_err(|_| CError::Internal)
        } else if let Some(ref idx) = self.hnsw {
            idx.search(query, &req).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
    }
    
    fn count(&self) -> usize {
        if let Some(ref idx) = self.flat {
            idx.ntotal()
        } else if let Some(ref idx) = self.hnsw {
            idx.ntotal()
        } else {
            0
        }
    }
    
    fn dim(&self) -> usize {
        self.dim
    }
}

/// 创建索引
#[no_mangle]
pub extern "C" fn knowhere_create_index(config: CIndexConfig) -> *mut std::ffi::c_void {
    match IndexWrapper::new(config) {
        Some(wrapper) => {
            let boxed = Box::new(wrapper);
            Box::into_raw(boxed) as *mut std::ffi::c_void
        }
        None => std::ptr::null_mut(),
    }
}

/// 释放索引
#[no_mangle]
pub extern "C" fn knowhere_free_index(index: *mut std::ffi::c_void) {
    if !index.is_null() {
        unsafe {
            Box::from_raw(index as *mut IndexWrapper);
        }
    }
}

/// 添加向量到索引
#[no_mangle]
pub extern "C" fn knowhere_add_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    ids: *const i64,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }
    
    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        
        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        let ids_slice = if !ids.is_null() {
            Some(std::slice::from_raw_parts(ids, count))
        } else {
            None
        };
        
        match index.add(vectors_slice, ids_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 训练索引
#[no_mangle]
pub extern "C" fn knowhere_train_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }
    
    unsafe {
        let index = &mut *(index as *mut IndexWrapper);
        
        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        
        match index.train(vectors_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}

/// 搜索
#[no_mangle]
pub extern "C" fn knowhere_search(
    index: *const std::ffi::c_void,
    query: *const f32,
    count: usize,
    top_k: usize,
    dim: usize,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 {
        return std::ptr::null_mut();
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        
        let query_slice = std::slice::from_raw_parts(query, count * dim);
        
        match index.search(query_slice, top_k) {
            Ok(result) => {
                let mut ids = result.ids;
                let mut distances = result.distances;
                
                let num_results = ids.len();
                let ids_ptr = ids.as_mut_ptr();
                let distances_ptr = distances.as_mut_ptr();
                
                // 防止析构函数释放内存
                std::mem::forget(ids);
                std::mem::forget(distances);
                
                let csr = CSearchResult {
                    ids: ids_ptr,
                    distances: distances_ptr,
                    num_results,
                    elapsed_ms: result.elapsed_ms as f32,
                };
                
                Box::into_raw(Box::new(csr))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
}

/// 获取索引中的向量数
#[no_mangle]
pub extern "C" fn knowhere_get_index_count(index: *const std::ffi::c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        index.count()
    }
}

/// 获取索引维度
#[no_mangle]
pub extern "C" fn knowhere_get_index_dim(index: *const std::ffi::c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let index = &*(index as *const IndexWrapper);
        index.dim()
    }
}

/// 释放搜索结果
#[no_mangle]
pub extern "C" fn knowhere_free_result(result: *mut CSearchResult) {
    if !result.is_null() {
        unsafe {
            let r = &mut *result;
            if !r.ids.is_null() {
                Vec::from_raw_parts(r.ids, r.num_results, r.num_results);
            }
            if !r.distances.is_null() {
                Vec::from_raw_parts(r.distances, r.num_results, r.num_results);
            }
            Box::from_raw(result);
        }
    }
}

// ========== BitsetView C 包装 ==========

use crate::bitset::BitsetView;

/// BitsetView C 包装
#[repr(C)]
pub struct CBitset {
    data: *mut u64,
    len: usize,
}

impl From<&BitsetView> for CBitset {
    fn from(bitset: &BitsetView) -> Self {
        let slice = bitset.as_slice();
        let mut vec = slice.to_vec();
        vec.shrink_to_fit();
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        
        Self {
            data: ptr,
            len: bitset.len(),
        }
    }
}

impl Drop for CBitset {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                Vec::from_raw_parts(self.data, 0, self.len / 64 + 1);
            }
        }
    }
}

/// 创建 Bitset
#[no_mangle]
pub extern "C" fn knowhere_bitset_create(len: usize) -> *mut CBitset {
    let bitset = BitsetView::new(len);
    let cb = CBitset::from(&bitset);
    Box::into_raw(Box::new(cb))
}

/// 释放 Bitset
#[no_mangle]
pub extern "C" fn knowhere_bitset_free(bitset: *mut CBitset) {
    if !bitset.is_null() {
        unsafe { Box::from_raw(bitset); }
    }
}

/// 设置位
#[no_mangle]
pub extern "C" fn knowhere_bitset_set(_bitset: &mut CBitset, _index: usize, _value: bool) {
    // TODO: 需要修改 BitsetView 支持可变引用
}

/// 获取位
#[no_mangle]
pub extern "C" fn knowhere_bitset_get(_bitset: &CBitset, _index: usize) -> bool {
    // TODO: 需要实现
    false
}

/// 统计
#[no_mangle]
pub extern "C" fn knowhere_bitset_count(_bitset: &CBitset) -> usize {
    // TODO: 需要实现
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitset_create() {
        let ptr = knowhere_bitset_create(100);
        assert!(!ptr.is_null());
        knowhere_bitset_free(ptr);
    }
    
    #[test]
    fn test_create_flat_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Flat,
            metric_type: CMetricType::L2,
            dim: 128,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        let count = knowhere_get_index_count(index);
        assert_eq!(count, 0);
        
        let dim = knowhere_get_index_dim(index);
        assert_eq!(dim, 128);
        
        knowhere_free_index(index);
    }
    
    #[test]
    fn test_create_hnsw_index() {
        let config = CIndexConfig {
            index_type: CIndexType::Hnsw,
            metric_type: CMetricType::L2,
            dim: 128,
            ef_construction: 200,
            ef_search: 64,
            ..Default::default()
        };
        
        let index = knowhere_create_index(config);
        assert!(!index.is_null());
        
        let dim = knowhere_get_index_dim(index);
        assert_eq!(dim, 128);
        
        knowhere_free_index(index);
    }
}

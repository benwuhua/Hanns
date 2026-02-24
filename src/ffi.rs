//! C API 绑定定义
//! 
//! 供 Milvus C++ 调用

use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{Index, IndexError, SearchResult as IndexSearchResult};

/// C API 错误码
#[repr(i32)]
pub enum CError {
    Success = 0,
    NotFound = 1,
    InvalidArg = 2,
    Internal = 3,
    NotImplemented = 4,
}

impl From<IndexError> for CError {
    fn from(e: IndexError) -> Self {
        match e {
            IndexError::Empty => CError::InvalidArg,
            _ => CError::Internal,
        }
    }
}

impl From<crate::api::KnowhereError> for CError {
    fn from(e: crate::api::KnowhereError) -> Self {
        CError::Internal
    }
}

/// C 风格的搜索结果
#[repr(C)]
pub struct CSearchResult {
    pub ids: *mut i64,
    pub distances: *mut f32,
    pub num_results: usize,
    pub elapsed_ms: f32,
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
pub extern "C" fn knowhere_bitset_set(bitset: &mut CBitset, index: usize, value: bool) {
    // 注意：需要修改 BitsetView 支持可变引用
}

/// 获取位
#[no_mangle]
pub extern "C" fn knowhere_bitset_get(bitset: &CBitset, index: usize) -> bool {
    // 需要实现
    false
}

/// 统计
#[no_mangle]
pub extern "C" fn knowhere_bitset_count(bitset: &CBitset) -> usize {
    0
}

/// Index 类型枚举
#[repr(i32)]
pub enum CIndexType {
    Hnsw = 0,
    IvfFlat = 1,
    IvfPq = 2,
    DiskAnn = 3,
}

/// Index 配置
#[repr(C)]
pub struct CIndexConfig {
    pub index_type: CIndexType,
    pub dim: usize,
    pub metric_type: i32, // 0=L2, 1=IP
    pub ef_construction: usize,
    pub ef_search: usize,
    pub nlist: usize,
    pub nprobe: usize,
}

/// 动态 Index 指针
#[repr(C)]
pub struct CIndex {
    _priv: [u8; 0],
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
}

//! IVF-ExRaBitQ C API 绑定
//!
//! 提供 C 语言接口用于构建、搜索、保存/加载 IVF-ExRaBitQ 索引

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

use crate::api::SearchRequest;

use super::ivf_usq::{IvfUsqConfig, IvfUsqIndex};

/// IVF-USQ index handle (backward-compatible with IVF-ExRaBitQ)
pub struct IvfUsqIndexHandle {
    index: IvfUsqIndex,
}

/// 构建 IVF-ExRaBitQ 索引
///
/// # Safety
/// `data` must point to at least `n * dim` valid `f32` values. When `ids` is non-null, it must
/// point to at least `n` valid `i64` values.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_build(
    dim: u32,
    nlist: u32,
    bits_per_dim: u32,
    data: *const f32,
    n: u64,
    ids: *const i64,
) -> *mut IvfUsqIndexHandle {
    if data.is_null() || dim == 0 || nlist == 0 || bits_per_dim == 0 || n == 0 {
        return std::ptr::null_mut();
    }

    let data_slice = std::slice::from_raw_parts(data, (n * dim as u64) as usize);
    let ids_slice = if ids.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(ids, n as usize))
    };

    let config = IvfUsqConfig::new(dim as usize, nlist as usize, bits_per_dim as usize);
    let mut index = IvfUsqIndex::new(config);

    if index.train(data_slice).is_err() {
        return std::ptr::null_mut();
    }

    if index.add(data_slice, ids_slice).is_err() {
        return std::ptr::null_mut();
    }

    Box::into_raw(Box::new(IvfUsqIndexHandle { index }))
}

/// 从文件加载 IVF-ExRaBitQ 索引
///
/// # Safety
/// `path` must be a valid, NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_load(
    path: *const c_char,
) -> *mut IvfUsqIndexHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match IvfUsqIndex::load(Path::new(path_str)) {
        Ok(index) => Box::into_raw(Box::new(IvfUsqIndexHandle { index })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// 保存 IVF-ExRaBitQ 索引到文件
///
/// # Safety
/// `index` must be a valid handle returned by this module and `path` must be a valid,
/// NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_save(
    index: *mut IvfUsqIndexHandle,
    path: *const c_char,
) -> i32 {
    if index.is_null() || path.is_null() {
        return -1;
    }

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match (&mut *index).index.save(Path::new(path_str)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// 搜索 IVF-ExRaBitQ 索引
///
/// # Safety
/// `index`, `query`, `distances`, and `labels` must be valid for reads/writes according to the
/// sizes implied by `k` and the index dimension.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_search(
    index: *const IvfUsqIndexHandle,
    query: *const f32,
    k: u32,
    nprobe: u32,
    distances: *mut f32,
    labels: *mut i64,
) -> i32 {
    if index.is_null() || query.is_null() || distances.is_null() || labels.is_null() || k == 0 {
        return -1;
    }

    let handle = &*index;
    let query_slice = std::slice::from_raw_parts(query, handle.index.config().dim);
    let distances_slice = std::slice::from_raw_parts_mut(distances, k as usize);
    let labels_slice = std::slice::from_raw_parts_mut(labels, k as usize);
    let req = SearchRequest {
        top_k: k as usize,
        nprobe: nprobe as usize,
        filter: None,
        params: None,
        radius: None,
    };

    match handle.index.search(query_slice, &req) {
        Ok(result) => {
            let len = result.ids.len().min(k as usize);
            labels_slice[..len].copy_from_slice(&result.ids[..len]);
            distances_slice[..len].copy_from_slice(&result.distances[..len]);
            for idx in len..k as usize {
                labels_slice[idx] = -1;
                distances_slice[idx] = f32::MAX;
            }
            0
        }
        Err(_) => -1,
    }
}

/// 批量搜索 IVF-ExRaBitQ 索引
///
/// # Safety
/// `index`, `queries`, `distances`, and `labels` must be valid for reads/writes according to the
/// sizes implied by `nq`, `k`, and the index dimension.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_batch_search(
    index: *const IvfUsqIndexHandle,
    queries: *const f32,
    nq: u64,
    k: u32,
    nprobe: u32,
    distances: *mut f32,
    labels: *mut i64,
) -> i32 {
    if index.is_null()
        || queries.is_null()
        || distances.is_null()
        || labels.is_null()
        || nq == 0
        || k == 0
    {
        return -1;
    }

    let handle = &*index;
    let dim = handle.index.config().dim;
    let queries_slice = std::slice::from_raw_parts(queries, (nq * dim as u64) as usize);
    let distances_slice = std::slice::from_raw_parts_mut(distances, (nq * k as u64) as usize);
    let labels_slice = std::slice::from_raw_parts_mut(labels, (nq * k as u64) as usize);

    for query_idx in 0..nq as usize {
        let query = &queries_slice[query_idx * dim..(query_idx + 1) * dim];
        let req = SearchRequest {
            top_k: k as usize,
            nprobe: nprobe as usize,
            filter: None,
            params: None,
            radius: None,
        };
        let offset = query_idx * k as usize;

        match handle.index.search(query, &req) {
            Ok(result) => {
                let len = result.ids.len().min(k as usize);
                labels_slice[offset..offset + len].copy_from_slice(&result.ids[..len]);
                distances_slice[offset..offset + len].copy_from_slice(&result.distances[..len]);
                for idx in len..k as usize {
                    labels_slice[offset + idx] = -1;
                    distances_slice[offset + idx] = f32::MAX;
                }
            }
            Err(_) => {
                for idx in 0..k as usize {
                    labels_slice[offset + idx] = -1;
                    distances_slice[offset + idx] = f32::MAX;
                }
            }
        }
    }

    0
}

/// 检查索引是否有原始数据
///
/// # Safety
/// `index` must be a valid handle returned by this module or null.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_has_raw_data(
    index: *const IvfUsqIndexHandle,
) -> i32 {
    if index.is_null() {
        return 0;
    }

    if (&*index).index.has_raw_data() {
        1
    } else {
        0
    }
}

/// 获取索引中的向量数量
///
/// # Safety
/// `index` must be a valid handle returned by this module or null.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_count(index: *const IvfUsqIndexHandle) -> u64 {
    if index.is_null() {
        return 0;
    }

    (&*index).index.count() as u64
}

/// 获取索引大小（字节）
///
/// # Safety
/// `index` must be a valid handle returned by this module or null.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_size(index: *const IvfUsqIndexHandle) -> u64 {
    if index.is_null() {
        return 0;
    }

    (&*index).index.size() as u64
}

/// 释放索引资源
///
/// # Safety
/// `index` must be a valid handle returned by this module or null. It must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_free(index: *mut IvfUsqIndexHandle) {
    if !index.is_null() {
        let _ = Box::from_raw(index);
    }
}

/// 设置搜索时的 nprobe 参数
///
/// # Safety
/// `index` must be a valid handle returned by this module.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_set_nprobe(
    index: *mut IvfUsqIndexHandle,
    nprobe: u32,
) -> i32 {
    if index.is_null() {
        return -1;
    }

    (&mut *index).index.set_nprobe(nprobe as usize);
    0
}

/// 获取索引维度
///
/// # Safety
/// `index` must be a valid handle returned by this module or null.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_dim(index: *const IvfUsqIndexHandle) -> u32 {
    if index.is_null() {
        return 0;
    }

    (&*index).index.config().dim as u32
}

/// 获取索引的 nlist 参数
///
/// # Safety
/// `index` must be a valid handle returned by this module or null.
#[no_mangle]
pub unsafe extern "C" fn knowhere_ivf_exrabitq_nlist(index: *const IvfUsqIndexHandle) -> u32 {
    if index.is_null() {
        return 0;
    }

    (&*index).index.config().nlist as u32
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_ffi_build_search_exrabitq() {
        unsafe {
            let dim = 16usize;
            let n = 100usize;
            let mut data = vec![0.0f32; n * dim];
            for i in 0..n {
                for j in 0..dim {
                    data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
                }
            }

            let index_ptr = knowhere_ivf_exrabitq_build(
                dim as u32,
                4,
                4,
                data.as_ptr(),
                n as u64,
                std::ptr::null(),
            );
            assert!(!index_ptr.is_null());

            let query = vec![0.5f32; dim];
            let mut distances = vec![0.0f32; 10];
            let mut labels = vec![0i64; 10];

            let ret = knowhere_ivf_exrabitq_search(
                index_ptr,
                query.as_ptr(),
                10,
                2,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            knowhere_ivf_exrabitq_free(index_ptr);
        }
    }

    #[test]
    fn test_ffi_save_load_exrabitq() {
        unsafe {
            let dim = 16usize;
            let n = 100usize;
            let mut data = vec![0.0f32; n * dim];
            for i in 0..n {
                for j in 0..dim {
                    data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
                }
            }

            let index_ptr = knowhere_ivf_exrabitq_build(
                dim as u32,
                4,
                4,
                data.as_ptr(),
                n as u64,
                std::ptr::null(),
            );
            assert!(!index_ptr.is_null());

            let dir = tempdir().unwrap();
            let path = dir.path().join("test_ivf_exrabitq.bin");
            let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

            assert_eq!(knowhere_ivf_exrabitq_save(index_ptr, c_path.as_ptr()), 0);
            knowhere_ivf_exrabitq_free(index_ptr);

            let loaded_ptr = knowhere_ivf_exrabitq_load(c_path.as_ptr());
            assert!(!loaded_ptr.is_null());
            assert_eq!(knowhere_ivf_exrabitq_count(loaded_ptr), n as u64);
            assert_eq!(knowhere_ivf_exrabitq_dim(loaded_ptr), dim as u32);
            assert_eq!(knowhere_ivf_exrabitq_nlist(loaded_ptr), 4);

            knowhere_ivf_exrabitq_free(loaded_ptr);
        }
    }
}

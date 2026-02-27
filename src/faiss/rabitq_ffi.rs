//! IVF-RaBitQ C API 绑定
//! 
//! 提供 C 语言接口用于构建、搜索、保存/加载 IVF-RaBitQ 索引

use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::path::Path;

use super::ivf_rabitq::{IvfRaBitqIndex, IvfRaBitqConfig};

/// IVF-RaBitQ 索引句柄
pub struct IvfRaBitqIndexHandle {
    index: IvfRaBitqIndex,
}

/// 构建 IVF-RaBitQ 索引
/// 
/// # Arguments
/// * `dim` - 向量维度
/// * `nlist` - 聚类数量
/// * `data` - 训练数据指针 [n * dim]
/// * `n` - 训练数据数量
/// * `ids` - 可选的向量 IDs
/// 
/// # Returns
/// 返回索引句柄，失败返回 NULL
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_build(
    dim: u32,
    nlist: u32,
    data: *const f32,
    n: u64,
    ids: *const i64,
) -> *mut IvfRaBitqIndexHandle {
    if data.is_null() || dim == 0 || nlist == 0 || n == 0 {
        return std::ptr::null_mut();
    }
    
    let data_slice = unsafe {
        std::slice::from_raw_parts(data, (n * dim as u64) as usize)
    };
    
    let ids_slice = if !ids.is_null() {
        Some(unsafe { std::slice::from_raw_parts(ids, n as usize) })
    } else {
        None
    };
    
    let config = IvfRaBitqConfig::new(dim as usize, nlist as usize);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 训练
    if let Err(_) = index.train(data_slice) {
        return std::ptr::null_mut();
    }
    
    // 添加向量
    if let Err(_) = index.add(data_slice, ids_slice) {
        return std::ptr::null_mut();
    }
    
    let handle = Box::new(IvfRaBitqIndexHandle { index });
    Box::into_raw(handle)
}

/// 从文件加载 IVF-RaBitQ 索引
/// 
/// # Arguments
/// * `path` - 文件路径
/// 
/// # Returns
/// 返回索引句柄，失败返回 NULL
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_load(
    path: *const c_char,
) -> *mut IvfRaBitqIndexHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };
    
    match IvfRaBitqIndex::load(Path::new(path_str)) {
        Ok(index) => {
            let handle = Box::new(IvfRaBitqIndexHandle { index });
            Box::into_raw(handle)
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// 保存 IVF-RaBitQ 索引到文件
/// 
/// # Arguments
/// * `index` - 索引句柄
/// * `path` - 文件路径
/// 
/// # Returns
/// 0 表示成功，非 0 表示失败
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_save(
    index: *mut IvfRaBitqIndexHandle,
    path: *const c_char,
) -> i32 {
    if index.is_null() || path.is_null() {
        return -1;
    }
    
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    let handle = unsafe { &mut *index };
    
    match handle.index.save(Path::new(path_str)) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// 搜索 IVF-RaBitQ 索引
/// 
/// # Arguments
/// * `index` - 索引句柄
/// * `query` - 查询向量 [dim]
/// * `k` - 返回的最近邻数量
/// * `nprobe` - 搜索的聚类数量
/// * `distances` - 输出距离数组 [k]
/// * `labels` - 输出 IDs 数组 [k]
/// 
/// # Returns
/// 0 表示成功，非 0 表示失败
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_search(
    index: *const IvfRaBitqIndexHandle,
    query: *const f32,
    k: u32,
    nprobe: u32,
    distances: *mut f32,
    labels: *mut i64,
) -> i32 {
    if index.is_null() || query.is_null() || distances.is_null() || labels.is_null() || k == 0 {
        return -1;
    }
    
    let handle = unsafe { &*index };
    let query_slice = unsafe { std::slice::from_raw_parts(query, handle.index.config().dim) };
    let distances_slice = unsafe { std::slice::from_raw_parts_mut(distances, k as usize) };
    let labels_slice = unsafe { std::slice::from_raw_parts_mut(labels, k as usize) };
    
    // 创建搜索请求
    use crate::api::SearchRequest;
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
            for i in 0..len {
                labels_slice[i] = result.ids[i];
                distances_slice[i] = result.distances[i];
            }
            0
        }
        Err(_) => -1,
    }
}

/// 批量搜索 IVF-RaBitQ 索引
/// 
/// # Arguments
/// * `index` - 索引句柄
/// * `queries` - 查询向量数组 [nq * dim]
/// * `nq` - 查询数量
/// * `k` - 每个查询返回的最近邻数量
/// * `nprobe` - 搜索的聚类数量
/// * `distances` - 输出距离数组 [nq * k]
/// * `labels` - 输出 IDs 数组 [nq * k]
/// 
/// # Returns
/// 0 表示成功，非 0 表示失败
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_batch_search(
    index: *const IvfRaBitqIndexHandle,
    queries: *const f32,
    nq: u64,
    k: u32,
    nprobe: u32,
    distances: *mut f32,
    labels: *mut i64,
) -> i32 {
    if index.is_null() || queries.is_null() || distances.is_null() || labels.is_null() || k == 0 || nq == 0 {
        return -1;
    }
    
    let handle = unsafe { &*index };
    let dim = handle.index.config().dim;
    
    let queries_slice = unsafe { 
        std::slice::from_raw_parts(queries, (nq * dim as u64) as usize) 
    };
    let distances_slice = unsafe { 
        std::slice::from_raw_parts_mut(distances, (nq * k as u64) as usize) 
    };
    let labels_slice = unsafe { 
        std::slice::from_raw_parts_mut(labels, (nq * k as u64) as usize) 
    };
    
    use crate::api::SearchRequest;
    
    let mut offset = 0;
    for i in 0..nq as usize {
        let query = &queries_slice[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: k as usize,
            nprobe: nprobe as usize,
            filter: None,
            params: None,
            radius: None,
        };
        
        match handle.index.search(query, &req) {
            Ok(result) => {
                let len = result.ids.len().min(k as usize);
                for j in 0..len {
                    labels_slice[offset + j] = result.ids[j];
                    distances_slice[offset + j] = result.distances[j];
                }
                // 填充剩余部分
                for j in len..k as usize {
                    labels_slice[offset + j] = -1;
                    distances_slice[offset + j] = f32::MAX;
                }
            }
            Err(_) => {
                for j in 0..k as usize {
                    labels_slice[offset + j] = -1;
                    distances_slice[offset + j] = f32::MAX;
                }
            }
        }
        
        offset += k as usize;
    }
    
    0
}

/// 检查索引是否有原始数据
/// 
/// # Returns
/// 1 表示有原始数据，0 表示没有
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_has_raw_data(
    index: *const IvfRaBitqIndexHandle,
) -> i32 {
    if index.is_null() {
        return 0;
    }
    
    let handle = unsafe { &*index };
    if handle.index.has_raw_data() { 1 } else { 0 }
}

/// 获取索引中的向量数量
/// 
/// # Returns
/// 返回向量数量，失败返回 0
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_count(
    index: *const IvfRaBitqIndexHandle,
) -> u64 {
    if index.is_null() {
        return 0;
    }
    
    let handle = unsafe { &*index };
    handle.index.count() as u64
}

/// 获取索引大小（字节）
/// 
/// # Returns
/// 返回索引大小，失败返回 0
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_size(
    index: *const IvfRaBitqIndexHandle,
) -> u64 {
    if index.is_null() {
        return 0;
    }
    
    let handle = unsafe { &*index };
    handle.index.size() as u64
}

/// 释放索引资源
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_free(
    index: *mut IvfRaBitqIndexHandle,
) {
    if !index.is_null() {
        unsafe {
            let _ = Box::from_raw(index);
        }
    }
}

/// 设置搜索时的 nprobe 参数
/// 
/// # Returns
/// 0 表示成功，非 0 表示失败
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_set_nprobe(
    index: *mut IvfRaBitqIndexHandle,
    nprobe: u32,
) -> i32 {
    if index.is_null() {
        return -1;
    }
    
    let handle = unsafe { &mut *index };
    handle.index.set_nprobe(nprobe as usize);
    0
}

/// 获取索引维度
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_dim(
    index: *const IvfRaBitqIndexHandle,
) -> u32 {
    if index.is_null() {
        return 0;
    }
    
    let handle = unsafe { &*index };
    handle.index.config().dim as u32
}

/// 获取索引的 nlist 参数
#[no_mangle]
pub extern "C" fn knowhere_ivf_rabitq_nlist(
    index: *const IvfRaBitqIndexHandle,
) -> u32 {
    if index.is_null() {
        return 0;
    }
    
    let handle = unsafe { &*index };
    handle.index.config().nlist as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_ffi_build_search() {
        // 生成测试数据
        let dim = 16;
        let n = 100;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        // 构建索引
        let index_ptr = knowhere_ivf_rabitq_build(
            dim,
            4,
            data.as_ptr(),
            n as u64,
            std::ptr::null(),
        );
        
        assert!(!index_ptr.is_null());
        
        // 搜索
        let query = vec![0.5f32; dim];
        let mut distances = vec![0.0f32; 10];
        let mut labels = vec![0i64; 10];
        
        let ret = knowhere_ivf_rabitq_search(
            index_ptr,
            query.as_ptr(),
            10,
            2,
            distances.as_mut_ptr(),
            labels.as_mut_ptr(),
        );
        
        assert_eq!(ret, 0);
        
        // 清理
        knowhere_ivf_rabitq_free(index_ptr);
    }
    
    #[test]
    fn test_ffi_save_load() {
        // 生成测试数据
        let dim = 16;
        let n = 100;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        // 构建索引
        let index_ptr = knowhere_ivf_rabitq_build(
            dim,
            4,
            data.as_ptr(),
            n as u64,
            std::ptr::null(),
        );
        
        assert!(!index_ptr.is_null());
        
        // 保存
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_ivf_rabitq.bin");
        let path_str = path.to_str().unwrap();
        let c_path = std::ffi::CString::new(path_str).unwrap();
        
        let ret = knowhere_ivf_rabitq_save(index_ptr, c_path.as_ptr());
        assert_eq!(ret, 0);
        
        // 释放原索引
        knowhere_ivf_rabitq_free(index_ptr);
        
        // 加载
        let loaded_ptr = knowhere_ivf_rabitq_load(c_path.as_ptr());
        assert!(!loaded_ptr.is_null());
        
        // 验证
        assert_eq!(knowhere_ivf_rabitq_count(loaded_ptr), n as u64);
        assert_eq!(knowhere_ivf_rabitq_dim(loaded_ptr), dim);
        
        // 清理
        knowhere_ivf_rabitq_free(loaded_ptr);
    }
    
    #[test]
    fn test_ffi_batch_search() {
        // 生成测试数据
        let dim = 16;
        let n = 100;
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }
        
        // 构建索引
        let index_ptr = knowhere_ivf_rabitq_build(
            dim,
            4,
            data.as_ptr(),
            n as u64,
            std::ptr::null(),
        );
        
        assert!(!index_ptr.is_null());
        
        // 批量搜索
        let nq = 5;
        let k = 10;
        let mut queries = vec![0.5f32; nq * dim];
        let mut distances = vec![0.0f32; nq * k];
        let mut labels = vec![0i64; nq * k];
        
        let ret = knowhere_ivf_rabitq_batch_search(
            index_ptr,
            queries.as_ptr(),
            nq as u64,
            k,
            2,
            distances.as_mut_ptr(),
            labels.as_mut_ptr(),
        );
        
        assert_eq!(ret, 0);
        
        // 清理
        knowhere_ivf_rabitq_free(index_ptr);
    }
}

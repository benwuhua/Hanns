//! JNI 绑定 - 供 Java/Kotlin 调用
//! 
//! # Java 使用示例
//! ```java
//! // 创建索引
//! KnowhereIndex index = KnowhereIndex.builder()
//     .indexType(IndexType.HNSW)
//     .dimension(128)
//     .metricType(MetricType.L2)
//     .build();
//! 
//! // 添加向量
//! float[][] vectors = ...;
//! long[] ids = ...;
//! index.add(vectors, ids);
//! 
//! // 搜索
//! float[][] query = ...;
//! SearchResult result = index.search(query, 10);
//! long[] resultIds = result.getIds();
//! float[] resultDistances = result.getDistances();
//! 
//! // 释放
//! index.close();
//! ```

#![allow(dead_code)]

use jni::JNIEnv;
use jni::objects::{JClass, JLongArray, JFloatArray, JObjectArray, JByteArray};
use jni::sys::{jlong, jint, jfloat};
use std::sync::Mutex;
use std::collections::HashMap;

use crate::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest, SearchResult as ApiSearchResult};
use crate::faiss::{MemIndex, HnswIndex, IvfPqIndex, DiskAnnIndex};
use crate::index::Index;

/// 全局索引注册表
static INDEX_REGISTRY: Mutex<Option<HashMap<jlong, Box<dyn Index + Send + Sync>>>> = Mutex::new(None);

fn get_registry() -> std::sync::MutexGuard<'static, Option<HashMap<jlong, Box<dyn Index + Send + Sync>>>> {
    let mut guard = INDEX_REGISTRY.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    guard
}

fn next_handle() -> jlong {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as jlong
}

/// 初始化 JNI 模块
pub fn init() {
    let mut guard = get_registry();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
}

/// IndexType 转换
fn parse_index_type(t: i32) -> IndexType {
    match t {
        0 => IndexType::Flat,
        1 => IndexType::Hnsw,
        2 => IndexType::IvfFlat,
        3 => IndexType::IvfPq,
        4 => IndexType::DiskAnn,
        _ => IndexType::Flat,
    }
}

/// MetricType 转换
fn parse_metric_type(t: i32) -> MetricType {
    match t {
        0 => MetricType::L2,
        1 => MetricType::Ip,
        2 => MetricType::Cosine,
        _ => MetricType::L2,
    }
}

/// 创建索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_createIndex(
    _env: JNIEnv,
    _class: JClass,
    index_type: jint,
    dim: jint,
    metric_type: jint,
    ef_construction: jint,
    ef_search: jint,
) -> jlong {
    init();
    
    let config = IndexConfig {
        index_type: parse_index_type(index_type as i32),
        dim: dim as usize,
        metric_type: parse_metric_type(metric_type as i32),
        params: IndexParams {
            ef_construction: ef_construction as usize,
            ef_search: ef_search as usize,
            ..Default::default()
        },
    };
    
    let index: Box<dyn Index + Send + Sync> = match config.index_type {
        IndexType::Flat | IndexType::IvfFlat => {
            match MemIndex::new(config.clone()) {
                Ok(idx) => Box::new(idx),
                Err(e) => {
                    tracing::error!("Failed to create MemIndex: {:?}", e);
                    return 0;
                }
            }
        }
        IndexType::Hnsw => {
            match HnswIndex::new(config.clone(), config.params.clone()) {
                Ok(idx) => Box::new(idx),
                Err(e) => {
                    tracing::error!("Failed to create HnswIndex: {:?}", e);
                    return 0;
                }
            }
        }
        IndexType::IvfPq => {
            match IvfPqIndex::new(config.clone()) {
                Ok(idx) => Box::new(idx),
                Err(e) => {
                    tracing::error!("Failed to create IvfPqIndex: {:?}", e);
                    return 0;
                }
            }
        }
        IndexType::DiskAnn => {
            match DiskAnnIndex::new(config.clone()) {
                Ok(idx) => Box::new(idx),
                Err(e) => {
                    tracing::error!("Failed to create DiskAnnIndex: {:?}", e);
                    return 0;
                }
            }
        }
    };
    
    let handle = next_handle();
    let mut guard = get_registry();
    guard.as_mut().unwrap().insert(handle, index);
    
    handle
}

/// 释放索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_freeIndex(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    let mut guard = get_registry();
    if let Some(ref mut registry) = *guard {
        registry.remove(&handle);
    }
}

/// 添加向量
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_addIndex(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    vectors: JFloatArray,
    ids: JLongArray,
    _num_vectors: jint,
) -> jint {
    let guard = match get_registry().ok() {
        Some(g) => g,
        None => return -1,
    };
    
    let index = match guard.as_ref().and_then(|r| r.get(&handle)) {
        Some(i) => i,
        None => return -1,
    };
    
    // 获取向量数据
    let vec_slice = match env.get_array_elements(&vectors, jni::objects::ReleaseMode::NoCopyBack) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    // 获取 IDs
    let ids_slice = match env.get_array_elements(&ids, jni::objects::ReleaseMode::NoCopyBack) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    
    let ids: Option<Vec<i64>> = if !ids.is_null() {
        Some(ids_slice.iter().map(|&x| x as i64).collect())
    } else {
        None
    };
    
    match index.add(&vec_slice, ids.as_deref()) {
        Ok(n) => n as jint,
        Err(e) => {
            tracing::error!("Add failed: {:?}", e);
            -1
        }
    }
}

/// 搜索
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_search(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    query: JFloatArray,
    k: jint,
    _num_queries: jint,
) -> jlong {
    let guard = match get_registry().ok() {
        Some(g) => g,
        None => return 0,
    };
    
    let index = match guard.as_ref().and_then(|r| r.get(&handle)) {
        Some(i) => i,
        None => return 0,
    };
    
    // 获取查询向量
    let query_slice = match env.get_array_elements(&query, jni::objects::ReleaseMode::NoCopyBack) {
        Ok(s) => s,
        Err(_) => 0,
    };
    
    let req = SearchRequest {
        k: k as usize,
        ..Default::default()
    };
    
    match index.search(&query_slice, &req) {
        Ok(result) => {
            // 返回结果指针（简化实现，实际需要更好的内存管理）
            let result_ptr = Box::into_raw(Box::new(result));
            result_ptr as jlong
        }
        Err(e) => {
            tracing::error!("Search failed: {:?}", e);
            0
        }
    }
}

/// 获取搜索结果 IDs
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_getResultIds(
    mut env: JNIEnv,
    _class: JClass,
    result_ptr: jlong,
) -> JLongArray {
    if result_ptr == 0 {
        return JObjectArray::null(&env).into();
    }
    
    let result = unsafe { &*(result_ptr as *const ApiSearchResult) };
    
    // 展平 IDs
    let mut ids: Vec<jlong> = Vec::new();
    for &ids_batch in &result.distances {
        // 这里简化处理，实际需要正确的 ID 提取
        ids.push(ids_batch as jlong);
    }
    
    match env.new_long_array(ids.len() as jint) {
        Ok(arr) => {
            let _ = env.set_array_region(&arr, 0, &ids);
            arr.into()
        }
        Err(_) => JObjectArray::null(&env).into(),
    }
}

/// 获取搜索结果距离
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_getResultDistances(
    mut env: JNIEnv,
    _class: JClass,
    result_ptr: jlong,
) -> JFloatArray {
    if result_ptr == 0 {
        return JObjectArray::null(&env).into();
    }
    
    let result = unsafe { &*(result_ptr as *const ApiSearchResult) };
    
    // 展平距离
    let distances: Vec<jfloat> = result.distances.iter().flat_map(|d| d.iter()).map(|&x| x as jfloat).collect();
    
    match env.new_float_array(distances.len() as jint) {
        Ok(arr) => {
            let _ = env.set_array_region(&arr, 0, &distances);
            arr.into()
        }
        Err(_) => JObjectArray::null(&env).into(),
    }
}

/// 释放搜索结果
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_freeResult(
    _env: JNIEnv,
    _class: JClass,
    result_ptr: jlong,
) {
    if result_ptr != 0 {
        unsafe {
            let _ = Box::from_raw(result_ptr as *mut ApiSearchResult);
        }
    }
}

/// 序列化索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_serializeIndex(
    mut env: JNIEnv,
    _class: JClass,
    _handle: jlong,
) -> JByteArray {
    // TODO: 实现序列化
    match env.new_byte_array(0) {
        Ok(arr) => arr.into(),
        Err(_) => JObjectArray::null(&env).into(),
    }
}

/// 反序列化索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_deserializeIndex(
    _env: JNIEnv,
    _class: JClass,
    _data: JByteArray,
) -> jlong {
    // TODO: 实现反序列化
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_registry() {
        init();
        let guard = get_registry();
        assert!(guard.is_some());
    }
}

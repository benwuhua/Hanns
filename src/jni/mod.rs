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

use jni::objects::{JByteArray, JClass, JFloatArray, JLongArray};
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;
use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;
use tempfile::NamedTempFile;

use crate::api::{
    DataType, IndexConfig, IndexParams, IndexType, KnowhereError, MetricType, SearchRequest,
    SearchResult as ApiSearchResult,
};
use crate::faiss::{
    DiskAnnIndex, HnswIndex, IvfExRaBitqIndex, IvfFlatIndex, IvfPqIndex, IvfSq8Index, MemIndex,
};

enum RegisteredIndex {
    Mem(Box<MemIndex>),
    Hnsw(Box<HnswIndex>),
    IvfFlat(Box<IvfFlatIndex>),
    IvfPq(Box<IvfPqIndex>),
    IvfExRaBitq(Box<IvfExRaBitqIndex>),
    IvfSq8(Box<IvfSq8Index>),
    DiskAnn(Box<DiskAnnIndex>),
}

impl RegisteredIndex {
    fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> crate::api::Result<usize> {
        match self {
            RegisteredIndex::Mem(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::Hnsw(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::IvfFlat(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::IvfPq(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::IvfExRaBitq(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::IvfSq8(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
            RegisteredIndex::DiskAnn(idx) => match idx.add(vectors, ids) {
                Ok(count) => Ok(count),
                Err(err) if needs_training(&err) => {
                    idx.train(vectors)?;
                    idx.add(vectors, ids)
                }
                Err(err) => Err(err),
            },
        }
    }

    fn search(&self, query: &[f32], req: &SearchRequest) -> crate::api::Result<ApiSearchResult> {
        match self {
            RegisteredIndex::Mem(idx) => idx.search(query, req),
            RegisteredIndex::Hnsw(idx) => idx.search(query, req),
            RegisteredIndex::IvfFlat(idx) => idx.search(query, req),
            RegisteredIndex::IvfPq(idx) => idx.search(query, req),
            RegisteredIndex::IvfExRaBitq(idx) => idx.search(query, req),
            RegisteredIndex::IvfSq8(idx) => idx.search(query, req),
            RegisteredIndex::DiskAnn(idx) => idx.search(query, req),
        }
    }

    fn serialize_to_bytes(&self) -> crate::api::Result<Vec<u8>> {
        match self {
            RegisteredIndex::Mem(idx) => idx.serialize_to_memory(),
            RegisteredIndex::Hnsw(idx) => save_to_temp_bytes(|path| idx.save(path)),
            RegisteredIndex::IvfFlat(idx) => save_to_temp_bytes(|path| idx.save(path)),
            RegisteredIndex::IvfPq(idx) => save_to_temp_bytes(|path| idx.save(path)),
            RegisteredIndex::IvfExRaBitq(idx) => save_to_temp_bytes(|path| idx.save(path)),
            RegisteredIndex::IvfSq8(idx) => idx.serialize_to_bytes(),
            RegisteredIndex::DiskAnn(idx) => save_to_temp_bytes(|path| idx.save(path)),
        }
    }
}

/// 全局索引注册表
static INDEX_REGISTRY: Mutex<Option<HashMap<jlong, RegisteredIndex>>> = Mutex::new(None);
static RESULT_REGISTRY: Mutex<Option<HashMap<jlong, ApiSearchResult>>> = Mutex::new(None);

fn get_registry() -> std::sync::MutexGuard<'static, Option<HashMap<jlong, RegisteredIndex>>> {
    let mut guard = INDEX_REGISTRY.lock().unwrap();
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    guard
}

fn get_result_registry() -> std::sync::MutexGuard<'static, Option<HashMap<jlong, ApiSearchResult>>>
{
    let mut guard = RESULT_REGISTRY.lock().unwrap();
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
    drop(guard);

    let mut result_guard = get_result_registry();
    if result_guard.is_none() {
        *result_guard = Some(HashMap::new());
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
        5 => IndexType::IvfSq8,
        6 => IndexType::IvfExRaBitq,
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

fn build_config(
    index_type: IndexType,
    dim: usize,
    metric_type: MetricType,
    params: IndexParams,
) -> IndexConfig {
    IndexConfig {
        index_type,
        metric_type,
        dim,
        data_type: DataType::Float,
        params,
    }
}

fn build_registered_index(config: &IndexConfig) -> crate::api::Result<RegisteredIndex> {
    match config.index_type {
        IndexType::Flat => Ok(RegisteredIndex::Mem(Box::new(MemIndex::new(config)?))),
        IndexType::Hnsw => Ok(RegisteredIndex::Hnsw(Box::new(HnswIndex::new(config)?))),
        IndexType::IvfFlat => Ok(RegisteredIndex::IvfFlat(Box::new(IvfFlatIndex::new(
            config,
        )?))),
        IndexType::IvfPq => Ok(RegisteredIndex::IvfPq(Box::new(IvfPqIndex::new(config)?))),
        IndexType::IvfExRaBitq => Ok(RegisteredIndex::IvfExRaBitq(Box::new(
            IvfExRaBitqIndex::from_index_config(config)?,
        ))),
        IndexType::IvfSq8 => Ok(RegisteredIndex::IvfSq8(Box::new(IvfSq8Index::new(config)?))),
        IndexType::DiskAnn => Ok(RegisteredIndex::DiskAnn(Box::new(DiskAnnIndex::new(
            config,
        )?))),
        _ => Err(KnowhereError::InvalidArg(format!(
            "unsupported JNI index type: {:?}",
            config.index_type
        ))),
    }
}

fn read_u32_at(bytes: &[u8], offset: usize) -> crate::api::Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| KnowhereError::Codec("data too short".to_string()))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn save_to_temp_bytes<F>(save_fn: F) -> crate::api::Result<Vec<u8>>
where
    F: FnOnce(&std::path::Path) -> crate::api::Result<()>,
{
    let file = NamedTempFile::new()?;
    save_fn(file.path())?;
    Ok(fs::read(file.path())?)
}

fn load_from_temp_bytes<F>(bytes: &[u8], load_fn: F) -> crate::api::Result<()>
where
    F: FnOnce(&std::path::Path) -> crate::api::Result<()>,
{
    let file = NamedTempFile::new()?;
    fs::write(file.path(), bytes)?;
    load_fn(file.path())
}

fn needs_training(err: &KnowhereError) -> bool {
    matches!(err, KnowhereError::IndexNotTrained(_))
        || matches!(err, KnowhereError::InvalidArg(message) if message.contains("trained first"))
}

fn read_float_array(env: &mut JNIEnv, array: &JFloatArray) -> Option<Vec<f32>> {
    let len = env.get_array_length(array).ok()? as usize;
    let mut values = vec![0.0f32; len];
    env.get_float_array_region(array, 0, &mut values).ok()?;
    Some(values)
}

fn read_long_array(env: &mut JNIEnv, array: &JLongArray) -> Option<Vec<i64>> {
    if array.is_null() {
        return None;
    }
    let len = env.get_array_length(array).ok()? as usize;
    let mut values = vec![0i64; len];
    env.get_long_array_region(array, 0, &mut values).ok()?;
    Some(values)
}

fn build_result_array<'a>(env: &mut JNIEnv<'a>, values: &[jlong]) -> JLongArray<'a> {
    match env.new_long_array(values.len() as jint) {
        Ok(array) => {
            let _ = env.set_long_array_region(&array, 0, values);
            array
        }
        Err(_) => JLongArray::default(),
    }
}

fn build_distance_array<'a>(env: &mut JNIEnv<'a>, values: &[jfloat]) -> JFloatArray<'a> {
    match env.new_float_array(values.len() as jint) {
        Ok(array) => {
            let _ = env.set_float_array_region(&array, 0, values);
            array
        }
        Err(_) => JFloatArray::default(),
    }
}

fn build_byte_array<'a>(env: &mut JNIEnv<'a>, values: &[u8]) -> JByteArray<'a> {
    match env.new_byte_array(values.len() as jint) {
        Ok(array) => {
            let payload: Vec<i8> = values.iter().map(|&value| value as i8).collect();
            let _ = env.set_byte_array_region(&array, 0, &payload);
            array
        }
        Err(_) => JByteArray::default(),
    }
}

fn deserialize_registered_index(bytes: &[u8]) -> crate::api::Result<RegisteredIndex> {
    if bytes.starts_with(b"KWIX") {
        let dim = read_u32_at(bytes, 8)? as usize;
        let config = build_config(IndexType::Flat, dim, MetricType::L2, IndexParams::default());
        let mut index = MemIndex::new(&config)?;
        index.deserialize_from_memory(bytes)?;
        return Ok(RegisteredIndex::Mem(Box::new(index)));
    }

    if bytes.starts_with(b"HNSW") {
        let dim = read_u32_at(bytes, 8)? as usize;
        let config = build_config(IndexType::Hnsw, dim, MetricType::L2, IndexParams::default());
        let mut index = HnswIndex::new(&config)?;
        load_from_temp_bytes(bytes, |path| index.load(path))?;
        return Ok(RegisteredIndex::Hnsw(Box::new(index)));
    }

    if bytes.starts_with(b"DANN") {
        let dim = read_u32_at(bytes, 8)? as usize;
        let config = build_config(
            IndexType::DiskAnn,
            dim,
            MetricType::L2,
            IndexParams::default(),
        );
        let mut index = DiskAnnIndex::new(&config)?;
        load_from_temp_bytes(bytes, |path| index.load(path))?;
        return Ok(RegisteredIndex::DiskAnn(Box::new(index)));
    }

    if bytes.starts_with(b"IVFPQ") {
        let dim = read_u32_at(bytes, 9)? as usize;
        let params = IndexParams {
            nlist: Some(read_u32_at(bytes, 13)? as usize),
            m: Some(read_u32_at(bytes, 17)? as usize),
            nbits_per_idx: Some(read_u32_at(bytes, 21)? as usize),
            ..Default::default()
        };
        let config = build_config(IndexType::IvfPq, dim, MetricType::L2, params);
        let mut index = IvfPqIndex::new(&config)?;
        load_from_temp_bytes(bytes, |path| index.load(path))?;
        return Ok(RegisteredIndex::IvfPq(Box::new(index)));
    }

    if bytes.starts_with(b"IVFXRBTQ") {
        let file = NamedTempFile::new()?;
        fs::write(file.path(), bytes)?;
        let index = IvfExRaBitqIndex::load(file.path())?;
        return Ok(RegisteredIndex::IvfExRaBitq(Box::new(index)));
    }

    if bytes.starts_with(b"IVFFLAT") {
        let dim = read_u32_at(bytes, 7)? as usize;
        let file = NamedTempFile::new()?;
        fs::write(file.path(), bytes)?;
        let index = IvfFlatIndex::load(file.path(), dim)?;
        return Ok(RegisteredIndex::IvfFlat(Box::new(index)));
    }

    if bytes.starts_with(b"IVFSQ8") {
        let dim = read_u32_at(bytes, 6)? as usize;
        let index = IvfSq8Index::deserialize_from_bytes(bytes, dim)?;
        return Ok(RegisteredIndex::IvfSq8(Box::new(index)));
    }

    Err(KnowhereError::Codec(
        "unsupported JNI serialization format".to_string(),
    ))
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

    let config = build_config(
        parse_index_type(index_type),
        dim as usize,
        parse_metric_type(metric_type),
        IndexParams {
            ef_construction: Some(ef_construction as usize),
            ef_search: Some(ef_search as usize),
            ..Default::default()
        },
    );

    let index = match build_registered_index(&config) {
        Ok(index) => index,
        Err(e) => {
            tracing::error!("Failed to create JNI index: {:?}", e);
            return 0;
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
    num_vectors: jint,
) -> jint {
    let vectors = match read_float_array(&mut env, &vectors) {
        Some(values) => values,
        None => return -1,
    };

    let ids = read_long_array(&mut env, &ids);

    let mut guard = get_registry();
    let registry = match guard.as_mut() {
        Some(registry) => registry,
        None => return -1,
    };
    let index = match registry.get_mut(&handle) {
        Some(index) => index,
        None => return -1,
    };

    if let Some(ref row_ids) = ids {
        if num_vectors < 0 || row_ids.len() != num_vectors as usize {
            return -1;
        }
    }

    match index.add(&vectors, ids.as_deref()) {
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
    let query = match read_float_array(&mut env, &query) {
        Some(values) => values,
        None => return 0,
    };

    let req = SearchRequest {
        top_k: k as usize,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };

    let guard = get_registry();
    let registry = match guard.as_ref() {
        Some(registry) => registry,
        None => return 0,
    };
    let index = match registry.get(&handle) {
        Some(index) => index,
        None => return 0,
    };

    match index.search(&query, &req) {
        Ok(result) => {
            let result_handle = next_handle();
            let mut result_guard = get_result_registry();
            result_guard.as_mut().unwrap().insert(result_handle, result);
            result_handle
        }
        Err(e) => {
            tracing::error!("Search failed: {:?}", e);
            0
        }
    }
}

/// 获取搜索结果 IDs
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_getResultIds<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
    result_ptr: jlong,
) -> JLongArray<'a> {
    if result_ptr == 0 {
        return JLongArray::default();
    }

    let guard = get_result_registry();
    let result = match guard
        .as_ref()
        .and_then(|registry| registry.get(&result_ptr))
    {
        Some(result) => result,
        None => return JLongArray::default(),
    };

    let ids: Vec<jlong> = result.ids.iter().copied().map(|id| id as jlong).collect();
    build_result_array(&mut env, &ids)
}

/// 获取搜索结果距离
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_getResultDistances<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
    result_ptr: jlong,
) -> JFloatArray<'a> {
    if result_ptr == 0 {
        return JFloatArray::default();
    }

    let guard = get_result_registry();
    let result = match guard
        .as_ref()
        .and_then(|registry| registry.get(&result_ptr))
    {
        Some(result) => result,
        None => return JFloatArray::default(),
    };

    let distances: Vec<jfloat> = result
        .distances
        .iter()
        .copied()
        .map(|distance| distance as jfloat)
        .collect();
    build_distance_array(&mut env, &distances)
}

/// 释放搜索结果
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_freeResult(
    _env: JNIEnv,
    _class: JClass,
    result_ptr: jlong,
) {
    if result_ptr != 0 {
        let mut guard = get_result_registry();
        if let Some(registry) = guard.as_mut() {
            registry.remove(&result_ptr);
        }
    }
}

/// 序列化索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_serializeIndex<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
    handle: jlong,
) -> JByteArray<'a> {
    if handle == 0 {
        return JByteArray::default();
    }

    let guard = get_registry();
    let registry = match guard.as_ref() {
        Some(registry) => registry,
        None => return JByteArray::default(),
    };
    let index = match registry.get(&handle) {
        Some(index) => index,
        None => return JByteArray::default(),
    };

    match index.serialize_to_bytes() {
        Ok(bytes) => build_byte_array(&mut env, &bytes),
        Err(e) => {
            tracing::error!("Serialize failed: {:?}", e);
            JByteArray::default()
        }
    }
}

/// 反序列化索引
#[no_mangle]
pub extern "system" fn Java_io_milvus_knowhere_KnowhereNative_deserializeIndex(
    env: JNIEnv,
    _class: JClass,
    data: JByteArray,
) -> jlong {
    if data.is_null() {
        return 0;
    }

    let bytes = match env.convert_byte_array(&data) {
        Ok(bytes) => bytes,
        Err(_) => return 0,
    };
    if bytes.is_empty() {
        return 0;
    }

    match deserialize_registered_index(&bytes) {
        Ok(index) => {
            let handle = next_handle();
            let mut guard = get_registry();
            guard.as_mut().unwrap().insert(handle, index);
            handle
        }
        Err(e) => {
            tracing::error!("Deserialize failed: {:?}", e);
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{IndexParams, IndexType, MetricType, SearchRequest};

    #[test]
    fn test_registry() {
        init();
        let guard = get_registry();
        assert!(guard.is_some());
    }

    fn gen_vectors(n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|i| ((i % 97) as f32) / 97.0).collect()
    }

    fn train_registered_index(
        index: &mut RegisteredIndex,
        vectors: &[f32],
    ) -> crate::api::Result<()> {
        match index {
            RegisteredIndex::Mem(_) => Ok(()),
            RegisteredIndex::Hnsw(idx) => idx.train(vectors).map(|_| ()),
            RegisteredIndex::IvfFlat(idx) => idx.train(vectors).map(|_| ()),
            RegisteredIndex::IvfPq(idx) => idx.train(vectors).map(|_| ()),
            RegisteredIndex::IvfExRaBitq(idx) => idx.train(vectors).map(|_| ()),
            RegisteredIndex::IvfSq8(idx) => idx.train(vectors).map(|_| ()),
            RegisteredIndex::DiskAnn(idx) => idx.train(vectors).map(|_| ()),
        }
    }

    fn run_round_trip(index_type: IndexType) {
        let dim = 16usize;
        let n = 100usize;
        let qn = 5usize;
        let vectors = gen_vectors(n, dim);
        let queries = vectors[..qn * dim].to_vec();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        let params = IndexParams {
            nlist: Some(16),
            nprobe: Some(8),
            ..Default::default()
        };
        let config = build_config(index_type, dim, MetricType::L2, params);
        let mut index = build_registered_index(&config).expect("create index");
        train_registered_index(&mut index, &vectors).expect("train index");
        index
            .add(&vectors, Some(&ids))
            .expect("add vectors with ids");

        let req = SearchRequest {
            top_k: 5,
            nprobe: 8,
            ..Default::default()
        };

        let before: Vec<ApiSearchResult> = queries
            .chunks(dim)
            .map(|q| index.search(q, &req).expect("search before serialize"))
            .collect();

        for result in &before {
            assert!(!result.ids.is_empty());
            assert!(result.ids[0] >= 0);
            assert!(ids.contains(&result.ids[0]));
        }

        let bytes = index.serialize_to_bytes().expect("serialize");
        let restored = deserialize_registered_index(&bytes).expect("deserialize");

        let after: Vec<ApiSearchResult> = queries
            .chunks(dim)
            .map(|q| restored.search(q, &req).expect("search after deserialize"))
            .collect();

        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(after.iter()) {
            assert_eq!(b.ids, a.ids);
            assert_eq!(b.distances.len(), a.distances.len());
            for (bd, ad) in b.distances.iter().zip(a.distances.iter()) {
                assert_eq!(bd.to_bits(), ad.to_bits());
            }
        }
    }

    #[test]
    fn test_jni_ivf_flat_round_trip() {
        run_round_trip(IndexType::IvfFlat);
    }

    #[test]
    fn test_jni_ivf_sq8_round_trip() {
        run_round_trip(IndexType::IvfSq8);
    }

    #[test]
    fn test_jni_ivf_exrabitq_round_trip() {
        run_round_trip(IndexType::IvfExRaBitq);
    }
}

# IVF-FLAT-CC Implementation Summary

## Overview
实现了 IVF-FLAT-CC (并发版本) 索引，基于 C++ knowhere 的 `faiss::IndexIVFFlatCC` 设计。

## Files Changed

### New Files
- `src/faiss/ivf_flat_cc.rs` - IVF-FLAT-CC 并发版本实现

### Modified Files
- `src/faiss/mod.rs` - 导出 `IvfFlatCcIndex`
- `src/api/index.rs` - 添加 `IndexType::IvfFlatCc` 枚举值和 `ssize` 参数
- `src/api/mod.rs` - 添加 `KnowhereError::InternalError` 变体
- `src/codec/index.rs` - 添加 IvfFlatCc 的序列化支持 (type ID: 9)

## Key Features

### 并发支持
- 使用 `Arc<RwLock<>>` 保护所有共享状态
- 线程安全的 `train()`, `add()`, `search()` 操作
- 支持并发插入和搜索

### 参数
- `nlist`: IVF 聚类数量
- `nprobe`: 搜索时探查的聚类数量
- `ssize`: Segment size，用于并发操作控制

### API 使用示例

```rust
use knowhere_rs::api::{IndexConfig, IndexType, MetricType, IndexParams, SearchRequest};
use knowhere_rs::faiss::IvfFlatCcIndex;

// 创建配置
let config = IndexConfig {
    index_type: IndexType::IvfFlatCc,
    metric_type: MetricType::L2,
    dim: 128,
    params: IndexParams::ivf_cc(100, 10, 1024), // nlist=100, nprobe=10, ssize=1024
};

// 创建索引
let index = IvfFlatCcIndex::new(&config).unwrap();

// 训练
let train_data = vec![/* ... */];
index.train(&train_data).unwrap();

// 添加向量
let vectors = vec![/* ... */];
index.add(&vectors, None).unwrap();

// 搜索
let query = vec![/* ... */];
let req = SearchRequest {
    top_k: 10,
    nprobe: 10,
    filter: None,
    params: None,
    radius: None,
};
let result = index.search(&query, &req).unwrap();
```

### 并发使用示例

```rust
use std::sync::Arc;
use std::thread;

let index = Arc::new(IvfFlatCcIndex::new(&config).unwrap());

// 并发添加
let mut handles = vec![];
for i in 0..4 {
    let index_clone = Arc::clone(&index);
    let handle = thread::spawn(move || {
        let vectors = vec![/* ... */];
        index_clone.add(&vectors, None).unwrap()
    });
    handles.push(handle);
}

// 并发搜索
for i in 0..4 {
    let index_clone = Arc::clone(&index);
    let handle = thread::spawn(move || {
        let query = vec![/* ... */];
        index_clone.search(&query, &req).unwrap()
    });
    handles.push(handle);
}
```

## Tests

所有 5 个单元测试通过：
- `test_ivf_flat_cc_new` - 创建索引
- `test_ivf_flat_cc_train_add_search` - 基本功能测试
- `test_ivf_flat_cc_get_vectors` - 按 ID 获取向量
- `test_ivf_flat_cc_concurrent_add` - 并发插入测试
- `test_ivf_flat_cc_concurrent_search` - 并发搜索测试

```
running 5 tests
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 287 filtered out
```

## Implementation Notes

1. **线程安全**: 所有共享数据使用 `Arc<RwLock<>>` 保护
2. **读锁优化**: `search()` 只使用读锁，支持高并发查询
3. **写锁保护**: `train()` 和 `add()` 使用写锁保证数据一致性
4. **错误处理**: 锁获取失败时返回 `KnowhereError::InternalError`
5. **兼容性**: 与现有 `IvfFlatIndex` API 保持一致

## Reference

C++ knowhere 实现参考：
- `faiss::IndexIVFFlatCC` in `/Users/ryan/Code/vectorDB/knowhere/thirdparty/faiss/faiss/IndexIVFFlat.h`
- knowhere wrapper in `/Users/ryan/Code/vectorDB/knowhere/src/index/ivf/ivf.cc`

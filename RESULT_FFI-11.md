# FFI-11: HasRawData C API 实现结果

## 任务概述
添加 `HasRawData` C API 以检查索引是否包含原始向量数据。

## 实现内容

### 1. Index Trait 扩展 (src/index.rs)
添加了 `has_raw_data()` 方法到 Index trait：

```rust
/// 检查索引是否包含原始数据 (HasRawData)
/// 
/// 用于判断索引是否存储了原始向量数据，以便支持 GetVectorByIds 等操作。
/// 
/// # Returns
/// true 如果索引包含原始数据，false 否则
fn has_raw_data(&self) -> bool {
    // 默认实现：返回 false
    false
}
```

### 2. MemIndex 实现 (src/faiss/mem_index.rs)
```rust
/// Check if this index contains raw data
/// 
/// MemIndex (Flat index) always stores raw vectors
pub fn has_raw_data(&self) -> bool {
    true
}
```

### 3. HnswIndex 实现 (src/faiss/hnsw.rs)
```rust
/// Check if this index contains raw data
/// 
/// HNSW index stores raw vectors in the graph nodes
pub fn has_raw_data(&self) -> bool {
    true
}
```

### 4. ScaNNIndex 实现 (src/faiss/scann.rs)
```rust
/// Check if this index contains raw data
/// 
/// ScaNN stores raw vectors for re-ranking when reorder_k > 0
pub fn has_raw_data(&self) -> bool {
    self.config.reorder_k > 0
}
```

### 5. C API 函数 (src/ffi.rs)
```rust
/// 检查索引是否包含原始数据 (HasRawData)
/// 
/// 用于判断索引是否存储了原始向量数据，以便支持 GetVectorByIds 等操作。
/// 
/// # Arguments
/// * `index` - 索引指针 (由 knowhere_create_index 创建)
/// 
/// # Returns
/// 1 如果索引包含原始数据，0 否则
#[no_mangle]
pub extern "C" fn knowhere_has_raw_data(index: *const std::ffi::c_void) -> i32 {
    if index.is_null() {
        return 0;
    }
    
    unsafe {
        let wrapper = &*(index as *const IndexWrapper);
        // Check which index type is active and call has_raw_data
        if let Some(ref flat) = wrapper.flat {
            if flat.has_raw_data() {
                return 1;
            }
        }
        if let Some(ref hnsw) = wrapper.hnsw {
            if hnsw.has_raw_data() {
                return 1;
            }
        }
        if let Some(ref scann) = wrapper.scann {
            if scann.has_raw_data() {
                return 1;
            }
        }
        0
    }
}
```

## 文件改动

| 文件 | 改动 |
|------|------|
| src/index.rs | +10 行 (trait 方法) |
| src/faiss/mem_index.rs | +7 行 |
| src/faiss/hnsw.rs | +7 行 |
| src/faiss/scann.rs | +7 行 |
| src/ffi.rs | +38 行 |

## 编译验证
```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo check
```

结果：✅ 编译通过（仅有预存在的警告）

## C API 使用示例

```c
#include <knowhere_rs.h>

// 创建索引
CIndexConfig config = {
    .index_type = CIndexType_Flat,
    .dim = 128,
    .metric_type = CMetricType_L2,
};
CIndex* index = knowhere_create_index(config);

// 检查是否有原始数据
int has_raw = knowhere_has_raw_data(index);
if (has_raw) {
    // 可以使用 GetVectorByIds
    printf("Index has raw data, GetVectorByIds is supported\n");
} else {
    printf("Index does not have raw data\n");
}

// 清理
knowhere_free_index(index);
```

## 与 C++ knowhere 的对应关系

C++ knowhere 中的实现：
- `include/knowhere/index/index_node.h`: `HasRawData(const std::string& metric_type) const`
- `src/index/flat/flat.cc`: Flat 实现
- `src/index/hnsw/hnsw.cc`: HNSW 实现

Rust 实现简化了接口，不需要 metric_type 参数，因为：
1. Rust 实现中，has_raw_data 的决策不依赖于 metric type
2. 保持了 C API 的简洁性

## 后续工作
- 可能需要添加单元测试验证 has_raw_data 的行为
- 考虑是否需要在 C++ 层也添加对应的 C API 包装

## 时间戳
完成时间：2026-02-27 16:XX (Asia/Shanghai)

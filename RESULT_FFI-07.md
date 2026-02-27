# FFI-07: GetVectorByIds C API 实现结果

## 任务
为 knowhere-rs 添加 GetVectorByIds 功能的 C API 绑定

## 实现内容

### 1. 修改的文件
- `src/ffi.rs` - 主要修改文件

### 2. 新增的 C API

#### 2.1 CGetVectorResult 结构体
```c
typedef struct {
    const float* vectors;    // 向量数据 (num_ids * dim)
    size_t num_ids;          // 成功获取的向量数量
    size_t dim;              // 向量维度
    int64_t* ids;            // 对应的 ID 数组
} CGetVectorResult;
```

#### 2.2 knowhere_get_vector_by_ids 函数
```c
CGetVectorResult* knowhere_get_vector_by_ids(
    const void* index,      // 索引指针
    const int64_t* ids,     // ID 数组
    size_t num_ids,         // ID 数量
    size_t dim              // 向量维度
);
```

#### 2.3 knowhere_free_get_vector_result 函数
```c
void knowhere_free_get_vector_result(CGetVectorResult* result);
```

### 3. 内部实现

#### IndexWrapper::get_vectors 方法
- 已存在于代码中，调用 `idx.get_vector_by_ids(ids)`
- **Flat (MemIndex)**: ✅ 支持，直接通过 ID 访问向量
- **HNSW**: ❌ NotImplemented (未存储原始向量)
- **ScaNN**: ❌ NotImplemented (未存储原始向量)

#### GetVectorByIds 行为
- 通过 ID 数组获取对应的原始向量数据
- 支持批量获取多个向量
- 如果 ID 不存在，返回的 num_ids 会少于输入数量
- 维度不匹配时返回 NULL

### 4. C API 使用示例

```c
// 创建 Flat 索引
CIndexConfig config = {
    .index_type = CIndexType_Flat,
    .metric_type = CMetricType_L2,
    .dim = 128,
};
CIndex* index = knowhere_create_index(config);

// 添加向量
float vectors[] = { ... };  // N * 128
int64_t ids[] = { 0, 1, 2, ... };
knowhere_add_index(index, vectors, ids, N, 128);

// 通过 ID 获取向量
int64_t query_ids[] = { 0, 5, 9 };
CGetVectorResult* result = knowhere_get_vector_by_ids(
    index, query_ids, 3, 128
);

if (result != NULL) {
    // 访问向量数据
    for (size_t i = 0; i < result->num_ids; i++) {
        const float* vec = &result->vectors[i * result->dim];
        printf("ID %ld: [%f, %f, ...]\n", result->ids[i], vec[0], vec[1]);
    }
    
    knowhere_free_get_vector_result(result);
}

knowhere_free_index(index);
```

### 5. 测试覆盖

已添加以下测试用例：
- `test_get_vector_by_ids_flat` - Flat 索引单 ID 测试
- `test_get_vector_by_ids_multiple` - 多个 ID 测试
- `test_get_vector_by_ids_nonexistent` - 不存在的 ID 测试
- `test_get_vector_by_ids_null_index` - 空索引指针测试
- `test_get_vector_by_ids_dimension_mismatch` - 维度不匹配测试
- `test_get_vector_by_ids_hnsw_not_implemented` - HNSW NotImplemented 测试
- `test_free_get_vector_result_null` - 释放空结果测试

所有测试均通过。

### 6. 编译验证

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo check
cargo test --lib ffi
```

结果：
- ✅ 编译成功（97 个警告，无错误）
- ✅ 28 个 FFI 测试全部通过
- ✅ 新增 7 个 GetVectorByIds 测试全部通过

### 7. 与 C++ 参考对齐

参考 C++ knowhere 的 `VectorMemIndex::GetVector()` 方法：
- ✅ 通过 ID 获取向量数据
- ✅ 支持批量获取
- ✅ 返回原始向量数据（std::vector<uint8_t> / float*）
- ⏳ 稀疏向量支持（待后续添加）

### 8. 后续工作

1. 为 HNSW 索引添加 GetVectorByIds 支持（需要存储原始向量）
2. 为 ScaNN 索引添加 GetVectorByIds 支持
3. 添加稀疏向量 GetVectorByIds 支持
4. 优化内存布局（当前使用连续 float 数组）

## 总结

成功实现了 GetVectorByIds 的 C API 绑定，包括：
- CGetVectorResult 结果结构体
- knowhere_get_vector_by_ids 搜索函数
- knowhere_free_get_vector_result 释放函数
- 完整的测试覆盖（7 个测试用例）

当前支持 Flat (MemIndex) 索引的 GetVectorByIds，HNSW 和 ScaNN 的 GetVectorByIds 将在后续实现。

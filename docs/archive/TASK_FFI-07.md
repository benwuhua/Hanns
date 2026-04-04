# FFI-07: GetVectorByIds C API 实现

## 任务描述
为 knowhere-rs 添加 GetVectorByIds 功能的 C API 绑定

## C++ 参考实现

参考 C++ knowhere 的 `VectorMemIndex::GetVector()` 方法：
- 调用 `index_.GetVectorByIds(dataset)` 
- 输入：包含 ID 的 Dataset
- 输出：`std::vector<uint8_t>` 原始向量数据

## 实现要求

### 1. C API 设计

```c
// 获取向量数据（按 IDs）
typedef struct {
    const float* vectors;    // 向量数据 (num_ids * dim)
    size_t num_ids;          // 成功获取的向量数量
    size_t dim;              // 向量维度
    int64_t* ids;            // 对应的 ID 数组（可能少于输入，如果某些 ID 不存在）
} CGetVectorResult;

CGetVectorResult* knowhere_get_vector_by_ids(
    const void* index,           // 索引指针
    const int64_t* ids,          // 要获取的 ID 数组
    size_t num_ids,              // ID 数量
    size_t dim                   // 向量维度
);

void knowhere_free_vector_result(CGetVectorResult* result);
```

### 2. 内部实现

在 `IndexWrapper` 中添加：
```rust
impl IndexWrapper {
    pub fn get_vector_by_ids(&self, ids: &[i64], dim: usize) -> Result<Vec<f32>>;
}
```

### 3. 支持的索引类型

- ✅ Flat (MemIndex) - 直接通过 ID 访问向量
- ⏳ HNSW - 需要存储原始向量
- ⏳ ScaNN - 可能需要存储原始向量

### 4. 测试用例

- `test_get_vector_by_ids_flat` - Flat 索引基本测试
- `test_get_vector_by_ids_multiple` - 多个 ID
- `test_get_vector_by_ids_nonexistent` - 不存在的 ID
- `test_get_vector_by_ids_null_index` - 空索引指针
- `test_free_vector_result_null` - 释放空结果

## 文件修改

- `src/ffi.rs` - 添加 C API 绑定和测试

## 验收标准

- [ ] `cargo check` 编译通过
- [ ] `cargo test --lib ffi` 所有测试通过
- [ ] 新增至少 5 个测试用例
- [ ] 代码风格与现有 FFI 代码一致

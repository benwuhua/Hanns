## 开发与审查报告 [2026-02-27 14:40]

### 完成任务
- **任务名:** FFI-08: 添加索引序列化 C API (Save/Load/Deserialize)
- **改动:**
  - `src/ffi.rs` +380/-0 行 (新增序列化/反序列化 C API 和测试)

### 具体改动

#### 1. src/ffi.rs - C API 层

**新增函数:**
- `knowhere_deserialize_index()` - 从 CBinarySet 反序列化到索引
  - 输入：索引指针 + CBinarySet 指针
  - 输出：CError 错误码
  - 支持 Flat 索引的内存反序列化

**新增结构体:**
- `CBinary` - 二进制数据块 (已有)
- `CBinarySet` - 二进制数据集合 (已有)

**新增测试:**
- `test_serialize_flat_index()` - 测试 Flat 索引序列化
- `test_save_load_flat_index()` - 测试 Flat 索引文件保存/加载
- `test_save_load_hnsw_index()` - 测试 HNSW 索引文件保存/加载
- `test_deserialize_index()` - 测试反序列化功能
- `test_deserialize_null_index()` - 测试空指针处理
- `test_deserialize_empty_binset()` - 测试空 BinarySet 处理
- `test_serialize_null_index()` - 测试空索引序列化
- `test_save_null_index()` - 测试空索引保存
- `test_load_null_index()` - 测试空索引加载

**IndexWrapper 方法:**
- `serialize()` - 序列化索引到内存 (Vec<u8>)
- `deserialize()` - 从内存反序列化索引
- `save()` - 保存索引到文件
- `load()` - 从文件加载索引

### 审查结果
✅ 通过

**检查项:**
- [x] `cargo check` 编译通过（96 warnings，无 errors）
- [x] `cargo test ffi::tests` - 15 个测试全部通过
- [x] `cargo test test_deserialize` - 3 个反序列化测试通过
- [x] 内存管理正确（使用 Box::into_raw / Box::from_raw 模式）
- [x] C API 风格与现有代码一致
- [x] 错误处理完善（null 指针检查，空数据检查）

### C API 使用示例

```c
// 序列化索引到内存
CBinarySet* binset = knowhere_serialize_index(source_index);
if (binset != NULL) {
    // 使用 binset->keys[i], binset->values[i].data/size 访问数据
    knowhere_free_binary_set(binset);
}

// 反序列化到目标索引
int result = knowhere_deserialize_index(target_index, binset);
if (result == 0) {
    // 反序列化成功
}

// 保存索引到文件
int save_result = knowhere_save_index(index, "/path/to/index.bin");

// 从文件加载索引
int load_result = knowhere_load_index(index, "/path/to/index.bin");
```

### 支持的索引类型

| 索引类型 | 内存序列化 | 文件序列化 |
|---------|----------|----------|
| Flat/MemIndex | ✅ | ✅ |
| HNSW | ❌ (NotImplemented) | ✅ |
| ScaNN | ❌ (NotImplemented) | ❌ (NotImplemented) |

### 新增任务
无 (序列化 API 已完整实现)

### 待办
- [ ] FFI-09: 添加 RangeSearch C API 支持
- [ ] FFI-10: 添加 Bitset 过滤搜索支持
- [ ] FFI-11: 添加 HasRawData C API 支持
- [ ] FFI-12: 添加索引统计信息 C API (Dim, Size, Count)
- [ ] IDX-08: 实现 HNSW-PRQ 索引 (Progressive Residual Quantization)
- [ ] IDX-09: 实现 IVF-RABITQ 索引

### 技术说明

**序列化格式:**
- Flat 索引：使用 `serialize_to_memory()` / `deserialize_from_memory()`
- HNSW 索引：仅支持文件序列化 (`save()` / `load()`)
- ScaNN 索引：暂不支持序列化

**内存管理:**
- CBinarySet 由 Rust 分配，调用者需使用 `knowhere_free_binary_set()` 释放
- CBinary 数据由 Rust 分配，包含在 CBinarySet 释放中
- Key 字符串使用 CString::into_raw()，在释放时转换回 CString 自动释放

**错误处理:**
- 空指针返回 `CError::InvalidArg`
- 空 BinarySet (count=0) 返回 `CError::InvalidArg`
- 不支持的索引类型返回 `CError::NotImplemented`
- 内部错误返回 `CError::Internal`

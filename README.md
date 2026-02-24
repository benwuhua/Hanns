# KnowHere-Rust

Rust 实现的向量搜索引擎，对齐 Milvus KnowHere。

## 特性

- **BitsetView** - 软删除支持
- **Dataset** - 数据集抽象
- **Metrics** - 距离度量 (L2/IP/Cosine/Hamming)
- **SIMD** - ARM NEON / x86 SSE/AVX 支持
- **Index Trait** - 统一索引接口
- **FFI** - C API 暴露

## 构建

```bash
cargo build --release
```

或使用脚本：
```bash
./build.sh release
```

## 测试

```bash
cargo test
```

## 项目结构

```
src/
├── bitset.rs    # 软删除
├── dataset.rs   # 数据集
├── metrics.rs   # 距离度量
├── simd.rs     # SIMD 计算
├── index.rs    # Index Trait
├── ffi.rs       # C API
└── faiss/      # 索引实现
    ├── mem_index.rs   # Flat
    ├── hnsw.rs      # HNSW
    ├── ivfpq.rs     # IVF-PQ
    └── diskann.rs   # DiskANN
```

## 测试结果

- 总计: **34 tests passed**
- 编译: ✅
- 构建: ✅

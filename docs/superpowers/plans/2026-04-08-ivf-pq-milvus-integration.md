# Plan: IVF-PQ Milvus 集成

**日期**: 2026-04-08
**目标**: 将 Milvus 中 `INDEX_FAISS_IVFPQ` 接入 knowhere-rs，验证真实数据集 recall 是否达标（合成随机数据上限低是已知的，Cohere/SIFT 上应更高）

---

## 背景

### 当前状态
- IVF-PQ 在合成 random 数据上 recall 低（m=32: 0.720），但这是数据分布问题，非实现 bug
- `CIndexConfig` 缺少 `pq_m` 字段 → 需要添加（Rust + C++ 双侧）
- `IvfPqIndex::new()` 默认 m=8（`config.params.m.unwrap_or(8)`）
- Milvus IVF-PQ 参数：`nlist`、`m`、`nbits`（默认8）

### 关键文件
- Rust FFI: `/data/work/milvus-rs-integ/knowhere-rs/src/ffi.rs`（本地 + rsync）
- Shim: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/`

---

## Tasks

### Task 1: 在 CIndexConfig 中添加 pq_m 字段

**文件**: `src/ffi.rs`，`pub struct CIndexConfig`（line 112 附近）

添加字段：
```rust
    pub pq_m: usize,   // IVF-PQ: number of sub-quantizers
    pub pq_nbits: usize, // IVF-PQ: bits per sub-quantizer (default 8)
```

在 `impl Default for CIndexConfig` 中添加：
```rust
    pq_m: 32,      // 合理默认：dim/pq_m = 768/32 = 24 维/sub-quantizer
    pq_nbits: 8,
```

在 `CIndexType::IvfPq => {` 分支（line 846 附近）修改 `IndexParams` 传入：
```rust
    params: IndexParams {
        nlist: Some(config.num_clusters.max(1)),
        nprobe: Some(config.nprobe.max(1)),
        m: Some(config.pq_m.max(1)),  // 添加这行
        nbits: Some(if config.pq_nbits > 0 { config.pq_nbits } else { 8 }), // 添加这行
        ..Default::default()
    },
```

**注意**: 添加字段后需要更新所有 `CIndexConfig { ... }` 的 struct literal 地方，或者确保它们都用 `..Default::default()`。检查 ffi.rs 末尾的测试里有没有直接 literal 构造，加 `pq_m: 32, pq_nbits: 8`。

### Task 2: rsync RS 到 hannsdb-x86 + 编译

```bash
rsync -av /path/to/knowhere-rs/src/ffi.rs hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/src/ffi.rs
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ && CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target ~/.cargo/bin/cargo build --release --manifest-path knowhere-rs/Cargo.toml 2>&1 | tail -5'
```

### Task 3: 创建 ivf_pq_rust_node.cpp

路径: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/ivf_pq_rust_node.cpp`

基于 `ivf_flat_rust_node.cpp` 修改：
- 类名: `IvfPqRustNode`
- `ffi_config.index_type = CIndexType::IvfPq;`（值 = 17）
- `Type()` 返回 `IndexEnum::INDEX_FAISS_IVFPQ`
- `HasRawData()` 返回 `false`（PQ 是有损压缩）
- `EnsureIndex()` 中读取 `m` 参数：

```cpp
// 读取 m（Milvus 用 "m" key，默认 dim/8 四舍五入到整数）
const auto pq_m = IvfPqGetSizeT(config, "m").value_or(
    static_cast<size_t>((configured_dim + 7) / 8));
const auto nbits = IvfPqGetSizeT(config, "nbits").value_or(8);

ffi_config.pq_m = pq_m;
ffi_config.pq_nbits = nbits;
```

**注意**: 需要同步修改 `cabi_bridge.hpp` 中的 `CIndexConfig` C 结构体，加入 `size_t pq_m; size_t pq_nbits;`

### Task 4: 修改 cabi_bridge.hpp

路径: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/cabi_bridge.hpp`

找到 `CIndexConfig` C struct，在末尾添加：
```cpp
    size_t pq_m;
    size_t pq_nbits;
```

（需要与 Rust 端 `CIndexConfig` 字段顺序完全一致！用 `#[repr(C)]` 对应）

### Task 5: 更新 index_factory.h + CMakeLists.txt

在 `index_factory.h` 的 IVF-Flat 路由之后插入：
```cpp
        if (name == IndexEnum::INDEX_FAISS_IVFPQ) {
            if constexpr (std::is_same_v<T, float>) {
                if (!std::getenv("KNOWHERE_RS_IVFPQ_BYPASS")) {
                    return Index<IndexNode>(MakeIvfPqRustNode());
                }
            }
        }
```
并添加 `MakeIvfPqRustNode()` 声明。

CMakeLists.txt 中加入 `ivf_pq_rust_node.cpp`。

### Task 6: 重编 Milvus + 重启 + Smoke test

同 IVF-Flat 集成步骤。

Smoke test 用 dim=128, nlist=64, m=8, nprobe=8：

```python
col.create_index("vec", {"index_type": "IVF_PQ", "metric_type": "L2",
    "params": {"nlist": 64, "m": 8, "nbits": 8}})
results = col.search(..., search_params={"metric_type": "L2", "params": {"nprobe": 8}}, limit=10)
```

检查 top1 是自身（recall = 1.0 for self-query）。

### Task 7: Recall + QPS benchmark

**数据集**: 与 IVF-Flat 相同（100K×768D, IP, Cohere shape）

**配置**: nlist=1024, m=32（768/32=24 dims/sub-quantizer，合理），nprobe=64

期望 recall @ 真实数据 ≥ 0.85（若 <0.70 说明实现有问题）

---

## 关键约束

`CIndexConfig` 是 `#[repr(C)]` struct，字段顺序必须与 C++ 侧 `CIndexConfig` 完全一致。
新增字段 `pq_m`/`pq_nbits` 必须同步更新两侧，且顺序相同。

---

## 完成标准

- Milvus IVF-PQ smoke test 通过（self-query recall=1.0）
- 100K×768D recall@10 ≥ 0.85（m=32, nprobe=64）
- RS vs native QPS 对比完成

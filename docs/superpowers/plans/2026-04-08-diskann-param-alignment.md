# DiskANN Parameter Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 DiskANN 的 `pq_code_budget_gb`、`build_dram_budget_gb`、`disk_pq_dims`、`beamwidth` 四个参数从 Milvus 侧打通到 RS `AisaqConfig`，并修复 shim 中 `max_degree` 和 `search_list_size` 的错误 fallback 值。

**Architecture:** 参数通道是 `Milvus config JSON → diskann_rust_node.cpp (shim) → CIndexConfig (C ABI struct) → ffi.rs → AisaqConfig`。需要同步修改三个文件：① C++ shim (ABI struct + 提取逻辑)，② Rust `CIndexConfig` struct (镜像 ABI)，③ Rust `ffi.rs` DiskAnn 分支（使用新字段）。两端 struct 必须字段顺序一致（`#[repr(C)]`）。

**Tech Stack:** Rust (ffi.rs, diskann_aisaq.rs), C++ (cabi_bridge.hpp, diskann_rust_node.cpp on hannsdb-x86)

**Key files:**
- Local: `src/ffi.rs` — `CIndexConfig` struct (line 112) + `CIndexType::DiskAnn` arm (line 1092)
- Remote x86: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/cabi_bridge.hpp` — C struct definition
- Remote x86: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/diskann_rust_node.cpp` — shim EnsureIndex() (line ~315)

---

### Task 1: 扩展 `CIndexConfig` — Rust 侧

**Files:**
- Modify: `src/ffi.rs:112-150`

**Step 1: 在 CIndexConfig struct 末尾追加 DiskANN 专用字段**

在 `data_type: i32` 之后、`}` 之前添加：

```rust
    // DiskANN-specific parameters
    pub pq_code_budget_gb: f32,
    pub build_dram_budget_gb: f32,
    pub disk_pq_dims: usize,
    pub beamwidth: usize,
```

结果（line 112-131）：
```rust
#[repr(C)]
pub struct CIndexConfig {
    pub index_type: CIndexType,
    pub metric_type: CMetricType,
    pub dim: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub num_partitions: usize,
    pub num_centroids: usize,
    pub reorder_k: usize,
    // PRQ parameters for HNSW-PRQ
    pub prq_nsplits: usize,
    pub prq_msub: usize,
    pub prq_nbits: usize,
    // IVF-RaBitQ parameters
    pub num_clusters: usize,
    pub nprobe: usize,
    /// Data type (0 = Float, 100 = Binary, etc.) - matches Milvus VecType enum
    pub data_type: i32,
    // DiskANN-specific parameters
    pub pq_code_budget_gb: f32,
    pub build_dram_budget_gb: f32,
    pub disk_pq_dims: usize,
    pub beamwidth: usize,
}
```

**Step 2: 更新 Default impl，添加新字段默认值**

在 `Default for CIndexConfig` 的 `data_type: 101` 之后添加：

```rust
            pq_code_budget_gb: 0.0,
            build_dram_budget_gb: 0.0,
            disk_pq_dims: 0,
            beamwidth: 8,
```

**Step 3: 验证编译**

```bash
cargo check 2>&1 | grep "^error" | head -10
```

Expected: 0 errors（只有 C++ struct 还没同步，但 Rust 侧应干净）

**Step 4: 更新 `CIndexType::DiskAnn` 分支使用新字段**

在 `src/ffi.rs` 的 `DiskAnn` 分支 (line 1092~1110)，将 `AisaqConfig` 构建替换为：

```rust
CIndexType::DiskAnn => {
    use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
    let max_degree = if config.ef_construction > 0 {
        config.ef_construction
    } else {
        48
    };
    let search_list_size = if config.ef_search > 0 {
        config.ef_search
    } else {
        128
    };
    let beamwidth = if config.beamwidth > 0 { config.beamwidth } else { 8 };
    let aisaq_config = AisaqConfig {
        max_degree,
        search_list_size,
        disk_pq_dims: config.disk_pq_dims,
        pq_code_budget_gb: config.pq_code_budget_gb,
        build_dram_budget_gb: config.build_dram_budget_gb,
        beamwidth,
        ..AisaqConfig::default()
    };
    let diskann = PQFlashIndex::new(aisaq_config, metric, dim).ok()?;
    // ... rest unchanged
```

**Step 5: 验证编译**

```bash
cargo check 2>&1 | grep "^error" | head -10
```

Expected: 0 errors

**Step 6: Commit**

```bash
git add src/ffi.rs
git commit -m "feat(ffi): extend CIndexConfig with DiskANN params (pq_code_budget_gb, build_dram_budget_gb, disk_pq_dims, beamwidth)"
```

---

### Task 2: 同步 C++ 侧 — cabi_bridge.hpp + diskann_rust_node.cpp

**Files:**
- Remote x86: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/cabi_bridge.hpp`
- Remote x86: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/diskann_rust_node.cpp`

**Step 1: 更新 cabi_bridge.hpp — CIndexConfig struct**

在 `int32_t data_type;` 之后添加（与 Rust 侧顺序完全一致）：

```cpp
    // DiskANN-specific parameters
    float pq_code_budget_gb;
    float build_dram_budget_gb;
    size_t disk_pq_dims;
    size_t beamwidth;
```

**Note on C++ field initialization:** C++ 的 `CIndexConfig ffi_config{}` 会零初始化所有字段，新字段自动为 0/0.0，安全。

**Step 2: 更新 diskann_rust_node.cpp EnsureIndex()**

修改 `EnsureIndex()` 中的参数提取部分（当前 line ~315-328）：

```cpp
CIndexConfig ffi_config{};
ffi_config.index_type = CIndexType::DiskAnn;
ffi_config.metric_type = ToCMetric(metric_type_);
ffi_config.dim = configured_dim;
// max_degree maps to ef_construction in CIndexConfig
// Native knowhere default is 48; use 48 as fallback (not 56)
ffi_config.ef_construction =
    GetSizeT(config, "max_degree").value_or(48);
// search_list_size maps to ef_search in CIndexConfig
// Native knowhere default at search time is max(k,1); use 128 for build
ffi_config.ef_search =
    GetSizeT(config, "search_list_size").value_or(128);
ffi_config.data_type = 101;

// DiskANN-specific parameters
if (const auto v = GetFloat(config, "pq_code_budget_gb"); v.has_value())
    ffi_config.pq_code_budget_gb = v.value();
if (const auto v = GetFloat(config, "build_dram_budget_gb"); v.has_value())
    ffi_config.build_dram_budget_gb = v.value();
if (const auto v = GetSizeT(config, "disk_pq_dims"); v.has_value())
    ffi_config.disk_pq_dims = v.value();
if (const auto v = GetSizeT(config, "beamwidth"); v.has_value())
    ffi_config.beamwidth = v.value();
```

**Note:** 需要确认 `GetFloat` helper 是否已存在于 diskann_rust_node.cpp；若无则添加：
```cpp
static std::optional<float>
GetFloat(const Config& config, const char* key) {
    const auto it = config.find(key);
    if (it == config.end()) return std::nullopt;
    if (it->is_number_float()) return it->get<float>();
    if (it->is_number_integer()) return static_cast<float>(it->get<int64_t>());
    return std::nullopt;
}
```

**Step 3: 重建 Milvus shim**

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/build && make -j8 2>&1 | tail -20'
```

Expected: build succeeds without error

**Step 4: 重启 Milvus 并验证 healthz**

```bash
ssh hannsdb-x86 'pkill -f "milvus run" || true; sleep 3'
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ && nohup bash scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_restart.log 2>&1 &'
sleep 20 && ssh hannsdb-x86 'curl -s http://127.0.0.1:9091/healthz'
```

Expected: `{"status":"healthy"}`

**Step 5: Smoke test — 建一个小 DiskANN collection，确认 pq_code_budget_gb=0.0 行为不变**

在 x86 运行快速 smoke test：
```python
# /tmp/smoke_diskann.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np, time
connections.connect(host="127.0.0.1", port="19530")
col_name = "diskann_smoke"
if utility.has_collection(col_name): utility.drop_collection(col_name)
schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True), FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)])
col = Collection(col_name, schema)
vecs = np.random.randn(10000, 128).astype(np.float32); vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
col.insert({"id": list(range(10000)), "vector": vecs.tolist()}); col.flush()
col.create_index("vector", {"index_type": "DISKANN", "metric_type": "IP", "params": {"max_degree": 56, "search_list_size": 100, "pq_code_budget_gb": 0.0, "build_dram_budget_gb": 32.0, "num_threads": 16}})
utility.wait_for_index_building_complete(col_name)
col.load(); time.sleep(5)
q = np.random.randn(1, 128).astype(np.float32); q /= np.linalg.norm(q)
res = col.search(q.tolist(), "vector", {"metric_type": "IP", "params": {"search_list": 100}}, limit=10)
print(f"SMOKE OK: got {len(res[0])} results, top_dist={res[0][0].distance:.4f}")
```

```bash
scp /tmp/smoke_diskann.py hannsdb-x86:/tmp/ && ssh hannsdb-x86 'python3 /tmp/smoke_diskann.py'
```

Expected: `SMOKE OK: got 10 results, top_dist=...`

**Step 6: Commit（本地 Rust 更改已在 Task 1 提交；此 step 记录 x86 shim 变更）**

在 x86 shim repo 提交（如果有独立 git）：
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim && git add src/cabi_bridge.hpp src/diskann_rust_node.cpp && git commit -m "feat(diskann): wire pq_code_budget_gb/build_dram_budget_gb/disk_pq_dims/beamwidth; fix max_degree/search_list_size fallbacks"'
```

---

### Task 3: 验证参数传递正确性

**Step 1: 验证 beamwidth 参数可从 Milvus 侧控制**

写一个小测试：对同一 collection 分别用 `beamwidth=1` 和 `beamwidth=8` 搜索，QPS 应不同（beamwidth=1 更慢，精度影响随 disk 模式更明显；NoPQ in-memory 模式下影响有限但应可见）。

这是可选验证步骤，用于确认通道打通。

**Step 2: 更新 benchmark result 记录**

在 `benchmark_results/diskann_milvus_rs_2026-04-08.md` 末尾追加一节：

```markdown
## R2 后续：参数对齐 (2026-04-08)

- `pq_code_budget_gb` / `build_dram_budget_gb` / `disk_pq_dims` / `beamwidth` 已打通
- Shim fallback 修复：`max_degree` 56→48，`search_list_size` 100→128
- NoPQ in-memory 行为不变（pq_code_budget_gb=0.0 默认）
```

**Step 3: Commit**

```bash
git add benchmark_results/diskann_milvus_rs_2026-04-08.md
git commit -m "docs(diskann): note param alignment fix in R2 benchmark record"
```

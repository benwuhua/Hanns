# IVF-SQ8 Milvus Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route Milvus `INDEX_FAISS_IVFSQ8` to the Rust IVF-SQ8 implementation, replacing the native C++ path.

**Architecture:** Three-layer change — (1) Rust FFI: add `knowhere_set_nprobe` + fix hardcoded nprobe=8 in search path; (2) C++ shim: `cabi_bridge.hpp` enum + new `ivf_sq8_rust_node.cpp` + `index_factory.h` intercept; (3) Build Milvus + benchmark.

**Tech Stack:** Rust (`src/ffi.rs`), C++ (hannsdb-x86 shim), pymilvus, VectorDBBench

---

## Context

### Current State

**Rust side (`src/ffi.rs`)**: `CIndexType::IvfSq8 = 9` exists, full IvfSq8 create/train/add/search/save/load implemented. `CIndexConfig` already has `num_centroids` and `nprobe` fields. **Problem**: IVF-SQ8 search at line ~1382 hardcodes `nprobe: 8` — needs dynamic nprobe support via `knowhere_set_nprobe`.

**C++ shim** (`/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/`):
- `cabi_bridge.hpp`: `CIndexType` enum only has `Flat=0, Hnsw=1, SparseInverted=12, SparseWand=13, DiskAnn=19` — missing `IvfSq8=9`
- `index_factory.h`: IVF-SQ8 currently falls through to `MakeRawDataIndexNode` (native C++) — need to intercept before that
- Missing: `ivf_sq8_rust_node.cpp` (analogous to `hnsw_rust_node.cpp`)

**Binary set key**: All RS indexes use `"index_data"` key (confirmed in ffi.rs line 3705).

**Type() string**: `IndexEnum::INDEX_FAISS_IVFSQ8` = `"IVF_SQ8"` (from `include/knowhere/comp/index_param.h`).

### Reference Files

- Template shim: `hnsw_rust_node.cpp` (797 lines) at shim `src/`
- IVF-SQ8 Rust impl: `src/faiss/ivf_sq8.rs` (searchable via `SearchRequest { nprobe, top_k, ... }`)
- `IndexWrapper::search()` in `src/ffi.rs` at line ~1381: hardcoded `nprobe: 8`
- `set_ef_search` pattern (line 1447–1457): model for `set_nprobe`
- `knowhere_set_ef_search` export (line 2538–2552): model for `knowhere_set_nprobe`

### Key Paths

```
Local repo:    /Users/ryan/.openclaw/workspace-builder/knowhere-rs/
Remote source: hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
Remote build:  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs-target/
Shim src:      hannsdb-x86:/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/
Shim header:   hannsdb-x86:/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h
Build:         ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/build && make -j8 2>&1 | tail -20'
Rsync:         rsync -az --exclude=target --exclude='data/' /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
```

---

## Task 1: Rust FFI — Add `knowhere_set_nprobe` + Fix Hardcoded nprobe

**Files:**
- Modify: `src/ffi.rs`

**Scope**: This task has 3 sub-changes, all in `src/ffi.rs`.

### Step 1: Add `nprobe` field to `IndexWrapper`

In `IndexWrapper` struct (around line 305, where other fields like `dim` are declared):
```rust
nprobe: usize,  // IVF search-time nprobe (default 8, overridable via knowhere_set_nprobe)
```
In all `IndexWrapper` constructors (anywhere `dim,` is set as the last field), add `nprobe: 8,`.

### Step 2: Fix hardcoded `nprobe: 8` in search

In `IndexWrapper::search()`, find lines ~1382–1411 (the IvfSq8 and IvfFlat arms):
```rust
// BEFORE (line ~1384):
nprobe: 8,
// AFTER:
nprobe: self.nprobe,
```
Do this for both the `ivf_sq8` arm and the `ivf_flat` arm.

### Step 3: Add `set_nprobe` method + `knowhere_set_nprobe` export

After `fn set_ef_search` (line ~1447):
```rust
fn set_nprobe(&mut self, nprobe: usize) -> Result<(), CError> {
    if self.ivf_sq8.is_some() || self.ivf_flat.is_some() {
        self.nprobe = nprobe;
        Ok(())
    } else {
        Err(CError::InvalidArg)
    }
}
```

After `knowhere_set_ef_search` export (line ~2552):
```rust
/// Override IVF search-time nprobe on an existing index handle.
#[no_mangle]
pub extern "C" fn knowhere_set_nprobe(index: *mut std::ffi::c_void, nprobe: usize) -> i32 {
    if index.is_null() || nprobe == 0 {
        return CError::InvalidArg as i32;
    }
    unsafe {
        let wrapper = &mut *(index as *mut IndexWrapper);
        match wrapper.set_nprobe(nprobe) {
            Ok(()) => CError::Success as i32,
            Err(err) => err as i32,
        }
    }
}
```

### Step 4: Build locally

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build 2>&1 | grep "^error"
```
Expected: no errors.

### Step 5: Rsync to hannsdb-x86 + build Rust there

```bash
rsync -az --exclude=target --exclude='data/' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/

ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/knowhere-rs && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  ~/.cargo/bin/cargo build --release 2>&1 | tail -5'
```
Expected: `Compiling knowhere-rs...` then `Finished`.

### Step 6: Commit

```bash
git add src/ffi.rs
git commit -m "feat(ffi): add knowhere_set_nprobe + fix hardcoded nprobe=8 in IVF search path"
```

---

## Task 2: C++ Shim — cabi_bridge.hpp + index_factory.h + ivf_sq8_rust_node.cpp

**Files (all on hannsdb-x86):**
- Modify: `knowhere-rs-shim/src/cabi_bridge.hpp`
- Modify: `knowhere-rs-shim/index_factory.h`
- Create: `knowhere-rs-shim/src/ivf_sq8_rust_node.cpp`

### Step 1: Patch `cabi_bridge.hpp` — add IvfSq8 to enum

Find the `CIndexType` enum:
```cpp
enum class CIndexType : int32_t {
    Flat = 0,
    Hnsw = 1,
    SparseInverted = 12,
    SparseWand = 13,
    DiskAnn = 19,
};
```
Add `IvfSq8 = 9,` after `Hnsw = 1,`:
```cpp
enum class CIndexType : int32_t {
    Flat = 0,
    Hnsw = 1,
    IvfSq8 = 9,
    SparseInverted = 12,
    SparseWand = 13,
    DiskAnn = 19,
};
```

Also add the `knowhere_set_nprobe` declaration to `cabi_bridge.hpp` (next to `knowhere_set_ef_search`):
```cpp
extern "C" int32_t knowhere_set_nprobe(void* index, size_t nprobe);
```

### Step 2: Patch `index_factory.h` — intercept IVF-SQ8 before RawDataIndexNode

Find the declaration block near lines 319-325 where other `Make...RustNode` functions are declared. Add:
```cpp
std::shared_ptr<IndexNode> MakeIvfSq8RustNode();
```

Find the dispatch block (around line 367) where `INDEX_FAISS_IVFSQ8` currently falls into the `MakeRawDataIndexNode` block. **Before** that block, insert:
```cpp
if (name == IndexEnum::INDEX_FAISS_IVFSQ8 ||
    name == IndexEnum::INDEX_FAISS_IVFSQ) {
    if constexpr (std::is_same_v<T, float>) {
        return Index<IndexNode>(MakeIvfSq8RustNode());
    }
}
```

### Step 3: Create `ivf_sq8_rust_node.cpp`

Create `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/ivf_sq8_rust_node.cpp` with the following content.

Key design differences vs `hnsw_rust_node.cpp`:
- No raw dataset storage (IVF-SQ8 has sufficient recall, no brute-force fallback)
- `EnsureIndex`: use `CIndexType::IvfSq8`, read `nlist` from index config, read `nprobe` from config (default 32)
- `Search`: call `knowhere_set_nprobe(handle_, nprobe)` before `knowhere_search`
- `Type()`: returns `IndexEnum::INDEX_FAISS_IVFSQ8`
- No `AnnIterator`, no `GetVectorByIds` (return `not_implemented`)
- No `MaybeNormalizeVectors` (IVF-SQ8 handles metric natively)
- `Deserialize`: use `"index_data"` key (confirmed)

```cpp
#include "knowhere/index/index_factory.h"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cabi_bridge.hpp"
#include "knowhere/config.h"
#include "status.hpp"

namespace knowhere {
namespace {

std::optional<size_t>
GetSizeT(const Config& config, const char* key) {
    if (!config.contains(key)) return std::nullopt;
    const auto& v = config.at(key);
    if (v.is_number_unsigned()) return v.get<size_t>();
    if (v.is_number_integer()) {
        auto i = v.get<int64_t>();
        if (i >= 0) return static_cast<size_t>(i);
        return std::nullopt;
    }
    if (v.is_string()) {
        const auto& s = v.get_ref<const std::string&>();
        if (!s.empty()) return static_cast<size_t>(std::stoull(s));
    }
    return std::nullopt;
}

std::optional<size_t>
GetSearchSizeT(const Config& config, const char* key) {
    const auto* vp = FindSearchParamValue(config, key);
    if (!vp) return std::nullopt;
    const auto& v = *vp;
    if (v.is_number_unsigned()) return v.get<size_t>();
    if (v.is_number_integer()) {
        auto i = v.get<int64_t>();
        if (i >= 0) return static_cast<size_t>(i);
        return std::nullopt;
    }
    if (v.is_string()) {
        const auto& s = v.get_ref<const std::string&>();
        if (!s.empty()) return static_cast<size_t>(std::stoull(s));
    }
    return std::nullopt;
}

std::string
GetStr(const Config& config, const char* key, std::string fallback) {
    if (!config.contains(key)) return fallback;
    const auto& v = config.at(key);
    if (v.is_string()) return v.get<std::string>();
    return fallback;
}

CMetricType
MetricToCMetric(const std::string& m) {
    if (m == metric::IP) return CMetricType::Ip;
    if (m == metric::COSINE) return CMetricType::Cosine;
    return CMetricType::L2;
}

class IvfSq8RustNode : public IndexNode {
 public:
    IvfSq8RustNode() = default;
    ~IvfSq8RustNode() override { Reset(); }

    Status
    Train(const DataSet& dataset, const Config& config) override {
        RETURN_IF_ERROR(EnsureIndex(dataset.GetDim(), config));
        const auto* vecs = static_cast<const float*>(dataset.GetTensor());
        if (!vecs || dataset.GetRows() <= 0 || dataset.GetDim() <= 0)
            return Status::invalid_args;
        return ToStatus(knowhere_train_index(handle_,
                                             vecs,
                                             static_cast<size_t>(dataset.GetRows()),
                                             static_cast<size_t>(dataset.GetDim())));
    }

    Status
    Add(const DataSet& dataset, const Config& config) override {
        RETURN_IF_ERROR(EnsureIndex(dataset.GetDim(), config));
        const auto* vecs = static_cast<const float*>(dataset.GetTensor());
        if (!vecs || dataset.GetRows() <= 0 || dataset.GetDim() <= 0)
            return Status::invalid_args;
        const auto code = knowhere_add_index(handle_,
                                              vecs,
                                              dataset.GetIds(),
                                              static_cast<size_t>(dataset.GetRows()),
                                              static_cast<size_t>(dataset.GetDim()));
        if (const auto s = ToStatus(code); s != Status::success) return s;
        RefreshStats();
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset,
           const Config& config,
           const BitsetView& bitset) const override {
        if (!handle_)
            return ErrorExpected<DataSetPtr>(Status::empty_index, "ivf_sq8 not initialized");

        const auto* query = static_cast<const float*>(dataset.GetTensor());
        const auto topk = GetSizeT(config, meta::TOPK).value_or(10);
        if (!query || dataset.GetRows() <= 0 || dataset.GetDim() <= 0 || topk == 0)
            return ErrorExpected<DataSetPtr>(Status::invalid_args, "invalid search dataset");

        // Honor search-time nprobe
        if (const auto nprobe = GetSearchSizeT(config, indexparam::NPROBE); nprobe.has_value()) {
            knowhere_set_nprobe(handle_, nprobe.value());
        }

        std::vector<uint64_t> bitset_words;
        CBitset cbitset{};
        CSearchResult* raw = nullptr;

        if (bitset.empty()) {
            raw = knowhere_search(handle_,
                                   query,
                                   static_cast<size_t>(dataset.GetRows()),
                                   topk,
                                   static_cast<size_t>(dataset.GetDim()));
        } else {
            bitset_words.resize((bitset.size() + 63) / 64);
            std::memcpy(bitset_words.data(), bitset.data(), bitset.byte_size());
            cbitset.data = bitset_words.data();
            cbitset.len = bitset.size();
            cbitset.cap_words = bitset_words.size();
            raw = knowhere_search_with_bitset(handle_,
                                               query,
                                               static_cast<size_t>(dataset.GetRows()),
                                               topk,
                                               static_cast<size_t>(dataset.GetDim()),
                                               &cbitset);
        }

        if (!raw)
            return ErrorExpected<DataSetPtr>(Status::invalid_index_error, "rust ffi returned null");

        std::unique_ptr<CSearchResult, void(*)(CSearchResult*)> result(raw, knowhere_free_result);
        return BuildSearchDataSet(dataset.GetRows(), topk, *result);
    }

    expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSet&, const Config&, const BitsetView&) const override {
        return ErrorExpected<std::vector<IteratorPtr>>(Status::not_implemented,
                                                        "iterator not supported for ivf_sq8");
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet&, const Config&, const BitsetView&) const override {
        return ErrorExpected<DataSetPtr>(Status::not_implemented, "range search not supported");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet&) const override {
        return ErrorExpected<DataSetPtr>(Status::not_implemented, "GetVectorByIds not supported");
    }

    bool HasRawData(const std::string&) const override { return false; }

    expected<DataSetPtr>
    GetIndexMeta(const Config&) const override {
        return ErrorExpected<DataSetPtr>(Status::not_implemented, "GetIndexMeta not supported");
    }

    Status
    Serialize(BinarySet& binary_set) const override {
        if (!handle_) return Status::empty_index;
        std::unique_ptr<CBinarySet, void(*)(CBinarySet*)> ffi_set(
            knowhere_serialize_index(handle_), knowhere_free_binary_set);
        if (!ffi_set || ffi_set->count == 0 || !ffi_set->values)
            return Status::invalid_binary_set;

        for (size_t i = 0; i < ffi_set->count; ++i) {
            const auto& b = ffi_set->values[i];
            const auto* key = (ffi_set->keys && ffi_set->keys[i]) ? ffi_set->keys[i] : "index_data";
            auto owned = std::shared_ptr<uint8_t[]>(new uint8_t[b.size]);
            std::memcpy(owned.get(), b.data, b.size);
            binary_set.Append(key, owned, b.size);
        }
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binary_set, const Config& config) override {
        RETURN_IF_ERROR(EnsureIndex(dim_ > 0 ? dim_ : 0, config));
        auto binary = binary_set.GetByNames({"index_data"});
        if (!binary || !binary->data || binary->size <= 0)
            return Status::invalid_binary_set;

        CBinary ffi_binary{ .data = binary->data.get(), .size = binary->size };
        const char* key = "index_data";
        char* ffi_key = const_cast<char*>(key);
        CBinarySet ffi_set{ .keys = &ffi_key, .values = &ffi_binary, .count = 1 };
        const auto s = ToStatus(knowhere_deserialize_index(handle_, &ffi_set));
        if (s == Status::success) RefreshStats();
        return s;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        RETURN_IF_ERROR(EnsureIndex(dim_ > 0 ? dim_ : 0, config));
        const auto s = ToStatus(knowhere_load_index(handle_, filename.c_str()));
        if (s == Status::success) RefreshStats();
        return s;
    }

    std::unique_ptr<BaseConfig> CreateConfig() const override {
        return std::make_unique<BaseConfig>();
    }

    int64_t Dim() const override { return dim_; }
    int64_t Size() const override { return size_; }
    int64_t Count() const override { return count_; }
    std::string Type() const override { return IndexEnum::INDEX_FAISS_IVFSQ8; }

 private:
    Status
    EnsureIndex(int64_t dataset_dim, const Config& config) {
        if (handle_) return Status::success;

        const auto configured_dim = GetSizeT(config, meta::DIM).value_or(
            static_cast<size_t>(dataset_dim));
        if (configured_dim == 0) return Status::invalid_args;

        metric_type_ = GetStr(config, meta::METRIC_TYPE, metric::L2);
        dim_ = static_cast<int64_t>(configured_dim);

        const auto nlist = GetSizeT(config, indexparam::NLIST).value_or(1024);
        const auto nprobe = GetSizeT(config, indexparam::NPROBE).value_or(32);

        CIndexConfig ffi_config{};
        ffi_config.index_type    = CIndexType::IvfSq8;
        ffi_config.metric_type   = MetricToCMetric(metric_type_);
        ffi_config.dim           = configured_dim;
        ffi_config.num_centroids = nlist;
        ffi_config.nprobe        = nprobe;
        ffi_config.data_type     = 101;

        handle_ = knowhere_create_index(ffi_config);
        return handle_ ? Status::success : Status::invalid_args;
    }

    expected<DataSetPtr>
    BuildSearchDataSet(int64_t nq, size_t topk, const CSearchResult& result) const {
        const auto total = static_cast<size_t>(nq) * topk;
        if (!result.ids || !result.distances || result.num_results < total)
            return ErrorExpected<DataSetPtr>(Status::invalid_index_error, "short result buffers");

        auto* ids = new int64_t[total];
        auto* dists = new float[total];
        std::copy(result.ids, result.ids + total, ids);
        std::copy(result.distances, result.distances + total, dists);

        auto ds = std::make_shared<DataSet>();
        ds->SetRows(nq);
        ds->SetDim(static_cast<int64_t>(topk));
        ds->SetIds(ids);
        ds->SetDistance(dists);
        ds->SetIsOwner(true);
        return ds;
    }

    void RefreshStats() {
        count_ = static_cast<int64_t>(knowhere_get_index_count(handle_));
        dim_   = static_cast<int64_t>(knowhere_get_index_dim(handle_));
        size_  = static_cast<int64_t>(knowhere_get_index_size(handle_));
    }

    void Reset() {
        if (handle_) {
            knowhere_free_index(handle_);
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
    int64_t dim_ = 0, size_ = 0, count_ = 0;
    std::string metric_type_ = metric::L2;
};

}  // namespace

std::shared_ptr<IndexNode>
MakeIvfSq8RustNode() {
    return std::make_shared<IvfSq8RustNode>();
}

}  // namespace knowhere
```

### Step 4: Commit on hannsdb-x86

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim && git add src/cabi_bridge.hpp src/ivf_sq8_rust_node.cpp index_factory.h && git commit -m "feat: add IVF-SQ8 Rust shim node"'
```

---

## Task 3: Build Milvus + Smoke Test

### Step 1: Rebuild Milvus shim

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/build && make -j8 2>&1 | tail -30'
```
Expected: `[100%] Built target milvus` or similar success.

### Step 2: Restart Milvus

```bash
ssh hannsdb-x86 'pkill -f "milvus run" || true'
sleep 5
ssh hannsdb-x86 'nohup bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_restart.log 2>&1 &'
sleep 30
ssh hannsdb-x86 'curl -s http://127.0.0.1:9091/healthz'
```
Expected: `OK`

### Step 3: Smoke test via pymilvus

Run this smoke test inline via SSH:

```python
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

connections.connect(host="127.0.0.1", port="19530")
N, DIM = 5000, 128
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("v", DataType.FLOAT_VECTOR, dim=DIM),
])
np.random.seed(1)
vecs = np.random.randn(N, DIM).astype(np.float32)

if utility.has_collection("ivfsq8_smoke"): utility.drop_collection("ivfsq8_smoke")
col = Collection("ivfsq8_smoke", schema)
col.insert([list(range(N)), vecs.tolist()])
col.flush()
col.create_index("v", {"index_type": "IVF_SQ8", "metric_type": "L2",
                        "params": {"nlist": 128}})
utility.wait_for_index_building_complete("ivfsq8_smoke")
col.load()

q = np.random.randn(1, DIM).astype(np.float32)
res = col.search(q.tolist(), "v", {"metric_type": "L2", "params": {"nprobe": 16}}, limit=5)
print(f"SMOKE OK: {len(res[0])} results, top dist={res[0][0].distance:.4f}")
```

Expected: `SMOKE OK: 5 results, top dist=...`

### Step 4: (If build fails) Debug approach

- Check compile errors: `grep -i "error\|undefined" /tmp/build.log`
- Common issue: missing `indexparam::NLIST` or `indexparam::NPROBE` — check include headers in shim
- Common issue: `knowhere_set_nprobe` not found — confirm rsync+cargo build completed in Task 1

---

## Task 4: Authority Benchmark — RS vs Native IVF-SQ8 on Cohere-1M

### Step 1: Check if native Milvus binary exists

```bash
ssh hannsdb-x86 'ls -la /data/work/milvus-rs-integ/milvus-native-bin/ 2>/dev/null || echo "no native binary"'
```

### Step 2: Run RS IVF-SQ8 VectorDBBench

Check what case IDs correspond to IVF-SQ8 in VectorDBBench:
```bash
ssh hannsdb-x86 '/data/work/VectorDBBench/.venv/bin/python3 -m vectordb_bench --help 2>&1 | head -20'
```

Run RS IVF-SQ8 benchmark with Cohere-1M:
```bash
ssh hannsdb-x86 'cd /data/work/VectorDBBench && \
  /data/work/VectorDBBench/.venv/bin/python3 run_benchmark.py \
  --db Milvus --case-type Performance1536D1M \
  --index IVF_SQ8 --nlist 1024 --nprobe 32 \
  --num-concurrency 1 5 10 20 30 40 60 80 \
  2>&1 | tee /tmp/ivfsq8_rs_bench.log'
```

**Note**: If the above command format doesn't work, check `benchmark_results/` for how previous HNSW benchmarks were run and adapt accordingly. The VectorDBBench command varies by version.

### Step 3: Record results

Save results to `benchmark_results/ivf_sq8_milvus_rs_2026-04-08.md` with format:
```
# IVF-SQ8 Milvus RS Benchmark — 2026-04-08

Dataset: Cohere-1M (768D, IP metric)
Machine: hannsdb-x86
Index: IVF_SQ8, nlist=1024

| Concurrency | QPS | Recall |
|------------|-----|--------|
| serial | ... | ... |
| c=20 | ... | ... |
| c=80 | ... | ... |
```

### Step 4: Ingest into wiki

Update `wiki/benchmarks/authority-numbers.md` with IVF-SQ8 Milvus results.
Update `wiki/log.md` (append at top).

---

## Success Criteria

1. `cargo build` passes locally with `knowhere_set_nprobe` exported
2. Milvus builds on hannsdb-x86 without errors
3. Smoke test returns results for `IVF_SQ8` index type
4. Authority benchmark QPS recorded and wiki updated

## Anti-patterns to Avoid

- Do NOT change the `CIndexConfig` struct field order (ABI breakage)
- Do NOT skip rsync before Milvus build (shim will call old Rust .so)
- Do NOT forget `indexparam::NLIST` include — check `hnsw_rust_node.cpp` includes for guidance
- Do NOT try to implement `AnnIterator` — return `not_implemented` (out of scope)

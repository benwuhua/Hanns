# Plan: IVF-Flat Milvus 集成

**日期**: 2026-04-08
**目标**: 将 Milvus 中 `INDEX_FAISS_IVFFLAT` 从 native C++ 切换到 knowhere-rs 实现，测量 RS vs native QPS 对比
**预期收益**: standalone 已验证 4.76× faster than native 8T（SIFT-1M, nprobe=32）

---

## 背景

### 当前 shim 路由（index_factory.h）
```
INDEX_HNSW → MakeHnswRustNode()          ✅
INDEX_DISKANN → MakeDiskAnnRustNode()     ✅
INDEX_FAISS_IVFSQ8 → MakeIvfSq8RustNode() ✅
INDEX_FAISS_IVFFLAT → MakeRawDataIndexNode() ❌ (native fallthrough)
```

### 关键路径
- Shim src: `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/`
- 新建文件: `src/ivf_flat_rust_node.cpp`（镜像 `ivf_sq8_rust_node.cpp`）
- 修改: `index_factory.h`（两处，shim 和 include 目录下各一个）
- RS repo: `/data/work/milvus-rs-integ/knowhere-rs/`
- Build target: `/data/work/milvus-rs-integ/knowhere-rs-target/`
- Cargo: `~/.cargo/bin/cargo`

### IVF-Flat vs IVF-SQ8 差异
- `CIndexType::IvfFlat = 18`（vs IvfSq8 = 9）
- `HasRawData()` 应返回 `true`（IVF-Flat 存储原始向量）
- `Type()` 返回 `IndexEnum::INDEX_FAISS_IVFFLAT`
- `GetVectorByIds()` 可以支持（IVF-Flat 有原始向量）
- 其他逻辑（Train/Add/Search/Serialize/Deserialize）与 IVF-SQ8 完全一致

### FFI 接口确认
`CIndexType::IvfFlat = 18` 在 `src/ffi.rs` line 95 已有。
IVF-Flat 走通用 `knowhere_create_index(ffi_config)` 路径，与 IVF-SQ8 相同。

---

## Tasks

### Task 1: 创建 ivf_flat_rust_node.cpp

在 hannsdb-x86 上创建文件:
`/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/ivf_flat_rust_node.cpp`

基于 `ivf_sq8_rust_node.cpp` 修改：
- 类名: `IvfFlatRustNode`
- `ffi_config.index_type = CIndexType::IvfFlat;`（值为 18）
- `Type()` 返回 `IndexEnum::INDEX_FAISS_IVFFLAT`
- `HasRawData()` 返回 `true`
- `GetVectorByIds()` 暂时保留 not_implemented（避免 FFI 复杂性）

**文件内容**（完整 cpp）：
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

// ---- 复用 ivf_sq8 的辅助函数（GetSizeT / GetSearchSizeT / GetStr / MetricToCMetric）----
// （在同一 translation unit，不重复定义；改用 static 避免 ODR 冲突）

static std::optional<size_t>
IvfFlatGetSizeT(const Config& config, const char* key) {
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

static std::optional<size_t>
IvfFlatGetSearchSizeT(const Config& config, const char* key) {
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

static std::string
IvfFlatGetStr(const Config& config, const char* key, std::string fallback) {
    if (!config.contains(key)) return fallback;
    const auto& v = config.at(key);
    if (v.is_string()) return v.get<std::string>();
    return fallback;
}

static CMetricType
IvfFlatMetricToCMetric(const std::string& m) {
    if (m == metric::IP) return CMetricType::Ip;
    if (m == metric::COSINE) return CMetricType::Cosine;
    return CMetricType::L2;
}

class IvfFlatRustNode : public IndexNode {
 public:
    IvfFlatRustNode() = default;
    ~IvfFlatRustNode() override { Reset(); }

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
            return ErrorExpected<DataSetPtr>(Status::empty_index, "ivf_flat not initialized");

        const auto* query = static_cast<const float*>(dataset.GetTensor());
        const auto topk = IvfFlatGetSizeT(config, meta::TOPK).value_or(10);
        if (!query || dataset.GetRows() <= 0 || dataset.GetDim() <= 0 || topk == 0)
            return ErrorExpected<DataSetPtr>(Status::invalid_args, "invalid search dataset");

        if (const auto nprobe = IvfFlatGetSearchSizeT(config, indexparam::NPROBE); nprobe.has_value()) {
            const auto update_status = ToStatus(knowhere_set_nprobe(handle_, nprobe.value()));
            if (update_status != Status::success) {
                return ErrorExpected<DataSetPtr>(update_status, "failed to set ivf_flat nprobe");
            }
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
                                                        "iterator not supported for ivf_flat");
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet&, const Config&, const BitsetView&) const override {
        return ErrorExpected<DataSetPtr>(Status::not_implemented, "range search not supported");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet&) const override {
        return ErrorExpected<DataSetPtr>(Status::not_implemented, "GetVectorByIds not supported for ivf_flat shim");
    }

    bool HasRawData(const std::string&) const override { return true; }

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
    std::string Type() const override { return IndexEnum::INDEX_FAISS_IVFFLAT; }

 private:
    Status
    EnsureIndex(int64_t dataset_dim, const Config& config) {
        if (handle_) return Status::success;

        const auto configured_dim = IvfFlatGetSizeT(config, meta::DIM).value_or(
            static_cast<size_t>(dataset_dim));
        if (configured_dim == 0) return Status::invalid_args;

        metric_type_ = IvfFlatGetStr(config, meta::METRIC_TYPE, metric::L2);
        dim_ = static_cast<int64_t>(configured_dim);

        const auto nlist = IvfFlatGetSizeT(config, indexparam::NLIST).value_or(1024);
        const auto nprobe = IvfFlatGetSizeT(config, indexparam::NPROBE).value_or(32);

        CIndexConfig ffi_config{};
        ffi_config.index_type    = CIndexType::IvfFlat;
        ffi_config.metric_type   = IvfFlatMetricToCMetric(metric_type_);
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
MakeIvfFlatRustNode() {
    return std::make_shared<IvfFlatRustNode>();
}

}  // namespace knowhere
```

### Task 2: 修改 index_factory.h（两处）

在 `INDEX_FAISS_IVFSQ8` 路由块之后、`INDEX_FAISS_IDMAP` fallthrough 之前，插入：
```cpp
if (name == IndexEnum::INDEX_FAISS_IVFFLAT ||
    name == IndexEnum::INDEX_FAISS_IVFFLAT_CC) {
    if constexpr (std::is_same_v<T, float>) {
        if (!std::getenv("KNOWHERE_RS_IVFFLAT_BYPASS")) {
            return Index<IndexNode>(MakeIvfFlatRustNode());
        }
    }
}
```
同时在文件顶部添加声明：`std::shared_ptr<IndexNode> MakeIvfFlatRustNode();`

修改两处：
1. `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h`
2. `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/index/index_factory.h`

### Task 3: 更新 CMakeLists.txt

确认新的 cpp 文件被编译。检查：
```bash
grep -n "ivf_sq8_rust_node\|SOURCES\|add_library" /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/CMakeLists.txt
```
在同一位置添加 `ivf_flat_rust_node.cpp`。

### Task 4: 重新编译 Milvus shim + 重启

```bash
cd /data/work/milvus-rs-integ/milvus-src
# 重编 shim（只重编修改部分）
cmake --build build --target knowhere -- -j$(nproc) 2>&1 | tail -20

# 或直接重启 Milvus（如果 shim 是动态库，hot-reload）
pkill -9 -f "milvus run standalone"; sleep 3
RESET_RUNTIME_STATE=false nohup bash scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_ivfflat.log 2>&1 &
sleep 20
```

### Task 5: 验证 + 写 benchmark 脚本

**验证脚本** `/tmp/ivfflat_smoke.py`：
- 创建 collection，IVF_FLAT index，nlist=1024
- insert 10K vectors，build index，搜索 k=10
- 检查返回结果非空，recall > 0.9

**诊断 benchmark** `/tmp/ivfflat_diag.py`：
- 与 ivfsq8_diag.py 结构相同
- c=1, c=20, c=80，每轮 8s
- nprobe=32（recall 目标 ≥0.95）
- RS vs native（KNOWHERE_RS_IVFFLAT_BYPASS=1 env var）

### Task 6: wiki 更新 + commit

更新 `wiki/benchmarks/authority-numbers.md` 添加 IVF-Flat Milvus 并发 QPS 表。
更新 `wiki/log.md` 添加集成记录。

---

## 预期结果

IVF-Flat standalone 4.76× faster → Milvus 环境下预计也有明显优势（IVF-Flat 是 brute-force 计算密集，RS SIMD 优化应显现）。

预计 c=80 QPS >> 140（IVF-Flat 无 SQ8 的量化延迟，纯 SIMD L2/IP）。

---

## 反模式

- 若 Milvus build 报找不到 `MakeIvfFlatRustNode`：检查 CMakeLists.txt 是否包含了 cpp 文件
- 若 knowhere_set_nprobe 不支持 IVF-Flat：检查 ffi.rs 里 IvfFlat 分支（line 806）
- 若 recall 异常低（<0.5）：检查 metric_type 是否正确传入（IP vs L2）

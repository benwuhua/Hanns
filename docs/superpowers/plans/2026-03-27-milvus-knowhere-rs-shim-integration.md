# Milvus knowhere-rs Shim Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-stage Milvus standalone smoke path on the remote x86 VM where Milvus can create, load, and search a float `HNSW` index through a new C++ `knowhere-rs-shim`, without changing `knowhere-rs` source code.

**Architecture:** Keep `knowhere-rs` unchanged and add a thin C++ compatibility layer inside the Milvus thirdparty/build boundary. The shim should expose only the minimum Knowhere-shaped surface Milvus needs for `HNSW` memory-index flow, while internally calling the existing `knowhere-rs` C ABI. All capability gaps discovered during the slice must be documented as issues before adding code outside the current stage-1 scope.

**Tech Stack:** Milvus source tree under `/Users/ryan/code/milvus`, `knowhere-rs` under `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`, CMake via `scripts/core_build.sh`, C++17, Rust `cdylib`/`staticlib`, remote x86 Linux under `/data/work`, Python smoke via `pymilvus`

---

## File Map

- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0001-stage1-scope.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0002-abi-gap.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0003-unsupported-surface.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0004-binaryset-contract.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0005-resource-mmap-stubs.md`
- Modify: `/Users/ryan/code/milvus/internal/core/thirdparty/CMakeLists.txt`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/CMakeLists.txt`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/version.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/object.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/expected.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/config.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/binaryset.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/bitsetview.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/dataset.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/index_node.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/comp/index_param.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/index/index_factory.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/cabi_bridge.hpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/status.hpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/index_factory.cpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/hnsw_rust_node.cpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/tests/compile_surface_smoke.cpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/tests/hnsw_roundtrip_smoke.cpp`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_env.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_sync.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/build_remote.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/start_standalone_remote.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/smoke_hnsw.py`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/run_remote_smoke.sh`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/remote-runbook.md`

## Chunk 1: Issue-First Guardrails

### Task 1: Record the hard scope and known gaps before writing adapter code

**Files:**
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0001-stage1-scope.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0002-abi-gap.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0003-unsupported-surface.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0004-binaryset-contract.md`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/issues/0005-resource-mmap-stubs.md`

- [ ] **Step 1: Create the issue directory**

Run:

```bash
mkdir -p /Users/ryan/code/milvus/docs/knowhere-rs-shim/issues
```

Expected: directory exists and is empty.

- [ ] **Step 2: Write the scope issue**

Put these non-negotiables into `0001-stage1-scope.md`:

```md
# 0001 Stage 1 Scope

- Remote x86 only
- Milvus standalone only
- float vector only
- HNSW only
- create_index / load / search only
- do not modify knowhere-rs source code
- do not modify Milvus query/segcore/indexbuilder core behavior
```

- [ ] **Step 3: Write the ABI-gap issue**

List the exact known surface mismatch:

```md
- Milvus expects knowhere C++ headers and object model
- knowhere-rs currently exposes Rust-owned C ABI
- shim must cover Version/DataSet/IndexNode/IndexFactory/BinarySet/config helpers
```

- [ ] **Step 4: Write the unsupported-surface issue**

Mark these as out of stage 1:

```md
- RangeSearch
- GetVectorByIds
- iterators
- IVF family
- DiskANN
- sparse
- GPU
- mmap / disk-load semantics
```

- [ ] **Step 5: Write BinarySet and resource stub issues**

Document:

```md
- BinarySet may start as a single-blob contract for HNSW only
- resource estimation can be conservative
- mmap/UseDiskLoad should default to unsupported in stage 1
```

- [ ] **Step 6: Verify the issue set exists**

Run:

```bash
find /Users/ryan/code/milvus/docs/knowhere-rs-shim/issues -maxdepth 1 -type f | sort
```

Expected: the five markdown files appear.

- [ ] **Step 7: Commit the issue inventory**

```bash
cd /Users/ryan/code/milvus
git add docs/knowhere-rs-shim/issues
git commit -m "docs(shim): add knowhere-rs integration issue inventory"
```

## Chunk 2: Build Switch and Compile Surface

### Task 2: Add a Milvus build-layer switch that can replace official Knowhere with the shim

**Files:**
- Modify: `/Users/ryan/code/milvus/internal/core/thirdparty/CMakeLists.txt`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/CMakeLists.txt`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/tests/compile_surface_smoke.cpp`

- [ ] **Step 1: Write a failing compile-surface smoke**

Create `compile_surface_smoke.cpp`:

```cpp
#include "knowhere/version.h"
#include "knowhere/dataset.h"
#include "knowhere/index_node.h"
#include "knowhere/index/index_factory.h"

int main() {
    auto ds = knowhere::GenDataSet(1, 4, nullptr);
    (void)ds;
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    (void)version;
    return 0;
}
```

- [ ] **Step 2: Wire a failing CMake option before the shim exists**

Add `MILVUS_USE_KNOWHERE_RS_SHIM` to `internal/core/thirdparty/CMakeLists.txt` and, when `ON`, point build flow to `knowhere-rs-shim` instead of `knowhere`.

- [ ] **Step 3: Run configure/build to verify red**

Run:

```bash
cd /Users/ryan/code/milvus
CMAKE_EXTRA_ARGS="-DMILVUS_USE_KNOWHERE_RS_SHIM=ON" make build-cpp
```

Expected: FAIL because the shim target and headers do not exist yet.

- [ ] **Step 4: Add the shim CMake scaffold**

Create `knowhere-rs-shim/CMakeLists.txt` with:

```cmake
add_library(knowhere SHARED)
target_include_directories(knowhere PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KNOWHERE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "Path to knowhere include directory")

add_executable(knowhere_rs_compile_surface_smoke tests/compile_surface_smoke.cpp)
target_link_libraries(knowhere_rs_compile_surface_smoke PRIVATE knowhere)
```

- [ ] **Step 5: Re-run configure/build to verify the next failure is now missing symbols or headers, not missing switch plumbing**

Run:

```bash
cd /Users/ryan/code/milvus
CMAKE_EXTRA_ARGS="-DMILVUS_USE_KNOWHERE_RS_SHIM=ON" make build-cpp
```

Expected: FAIL later in compilation, proving the build switch is active.

- [ ] **Step 6: Commit the build switch scaffold**

```bash
cd /Users/ryan/code/milvus
git add internal/core/thirdparty/CMakeLists.txt internal/core/thirdparty/knowhere-rs-shim
git commit -m "feat(shim): add Milvus build switch for knowhere-rs shim"
```

### Task 3: Implement the minimum public header surface so Milvus can compile against the shim

**Files:**
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/version.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/object.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/expected.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/config.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/binaryset.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/bitsetview.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/dataset.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/index_node.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/comp/index_param.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/include/knowhere/index/index_factory.h`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/status.hpp`

- [ ] **Step 1: Write the failing compile test target explicitly**

Run:

```bash
cd /Users/ryan/code/milvus/cmake_build
cmake --build . --target knowhere_rs_compile_surface_smoke -j8
```

Expected: FAIL on missing Knowhere headers or declarations.

- [ ] **Step 2: Add the immutable constants and config aliases first**

Put into `comp/index_param.h`:

```cpp
namespace knowhere {
using IndexType = std::string;
using MetricType = std::string;
namespace IndexEnum { inline constexpr const char* INDEX_HNSW = "HNSW"; }
namespace metric {
inline constexpr const char* L2 = "L2";
inline constexpr const char* IP = "IP";
inline constexpr const char* COSINE = "COSINE";
}
namespace meta {
inline constexpr const char* DIM = "dim";
inline constexpr const char* ROWS = "rows";
inline constexpr const char* TENSOR = "tensor";
inline constexpr const char* IDS = "ids";
inline constexpr const char* DISTANCE = "distance";
inline constexpr const char* TOPK = "k";
inline constexpr const char* METRIC_TYPE = "metric_type";
}
namespace indexparam {
inline constexpr const char* HNSW_M = "M";
inline constexpr const char* EFCONSTRUCTION = "efConstruction";
inline constexpr const char* EF = "ef";
}
}
```

- [ ] **Step 3: Add `Version`, `Object`, `expected`, `Status`, `BitsetView`, and `DataSet`**

Keep them deliberately small and header-only where practical.

- [ ] **Step 4: Add a stub `IndexNode` and `IndexFactory` declaration surface**

Use the same method names Milvus consumes, but leave behavior stubbed until the direct HNSW roundtrip test drives the implementation.

- [ ] **Step 5: Rebuild the compile surface target**

Run:

```bash
cd /Users/ryan/code/milvus/cmake_build
cmake --build . --target knowhere_rs_compile_surface_smoke -j8
```

Expected: PASS.

- [ ] **Step 6: Commit the public surface**

```bash
cd /Users/ryan/code/milvus
git add internal/core/thirdparty/knowhere-rs-shim/include internal/core/thirdparty/knowhere-rs-shim/src/status.hpp
git commit -m "feat(shim): add minimal knowhere public header surface"
```

## Chunk 3: HNSW Shim Node

### Task 4: Add a direct HNSW roundtrip smoke that fails before the Rust bridge exists

**Files:**
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/tests/hnsw_roundtrip_smoke.cpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/cabi_bridge.hpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/index_factory.cpp`
- Create: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/src/hnsw_rust_node.cpp`
- Modify: `/Users/ryan/code/milvus/internal/core/thirdparty/knowhere-rs-shim/CMakeLists.txt`

- [ ] **Step 1: Write the failing HNSW roundtrip smoke**

Create `hnsw_roundtrip_smoke.cpp`:

```cpp
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/comp/index_param.h"

int main() {
    float base[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float query[4] = {1,0,0,0};

    auto build_ds = knowhere::GenDataSet(4, 4, base);
    auto query_ds = knowhere::GenDataSet(1, 4, query);

    auto idx = knowhere::IndexFactory::Instance().Create<float>(knowhere::IndexEnum::INDEX_HNSW, 0).value();
    auto cfg = knowhere::Json{
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::indexparam::HNSW_M, 8},
        {knowhere::indexparam::EFCONSTRUCTION, 32},
        {knowhere::indexparam::EF, 16},
        {knowhere::meta::TOPK, 2},
    };

    auto st = idx.Build(*build_ds, cfg, true);
    if (st != knowhere::Status::success) return 1;
    auto res = idx.Search(query_ds, cfg, knowhere::BitsetView{});
    return res.has_value() ? 0 : 2;
}
```

- [ ] **Step 2: Build and run the test to verify red**

Run:

```bash
cd /Users/ryan/code/milvus/cmake_build
cmake --build . --target knowhere_rs_hnsw_roundtrip_smoke -j8
ctest -R knowhere_rs_hnsw_roundtrip_smoke --output-on-failure
```

Expected: FAIL because `IndexFactory::Create`, `Build`, or `Search` are not implemented.

- [ ] **Step 3: Add the C ABI bridge header**

Declare only the `knowhere-rs` symbols actually used in stage 1:

```cpp
extern "C" {
void* knowhere_create_index(CIndexConfig config);
int knowhere_add_index(void* index, const float* vectors, const int64_t* ids, size_t rows, size_t dim);
CSearchResult* knowhere_search(void* index, const float* query, size_t nq, size_t topk, size_t dim);
CBinarySet* knowhere_serialize_index(const void* index);
int knowhere_deserialize_index(void* index, const CBinarySet* binset);
int knowhere_save_index(const void* index, const char* path);
int knowhere_load_index(void* index, const char* path);
}
```

- [ ] **Step 4: Implement `HnswRustNode` with only stage-1 methods**

Required behavior:

```cpp
- create and own a `knowhere-rs` index handle
- translate HNSW config keys into `CIndexConfig`
- `Build` -> create + add/build path
- `Search` -> return ids/distances in a Knowhere-style `DataSet`
- `Serialize`/`Deserialize` -> bridge through existing memory ABI
- `DeserializeFromFile` -> bridge through existing file ABI
- unsupported methods return `Status::not_implemented`
```

- [ ] **Step 5: Implement the minimal `IndexFactory` wrapper**

Only support:

```cpp
Create<float>("HNSW", ...)
FeatureCheck(...) -> false except minimal `MMAP=false`
UseDiskLoad(...) -> false
IndexStaticFaced<float>::HasRawData(...) -> true or documented conservative value
IndexStaticFaced<float>::EstimateLoadResource(...) -> conservative memory estimate
```

- [ ] **Step 6: Link the shim to `libknowhere_rs`**

Update shim `CMakeLists.txt` so the `knowhere` target links against the Rust library path supplied by a cache variable such as `KNOWHERE_RS_LIB_PATH`.

- [ ] **Step 7: Re-run the HNSW roundtrip smoke**

Run:

```bash
cd /Users/ryan/code/milvus
CMAKE_EXTRA_ARGS="-DMILVUS_USE_KNOWHERE_RS_SHIM=ON -DKNOWHERE_RS_LIB_PATH=/data/work/milvus-rs-integ/knowhere-rs/target/release/libknowhere_rs.so" make build-cpp
cd /Users/ryan/code/milvus/cmake_build
ctest -R knowhere_rs_hnsw_roundtrip_smoke --output-on-failure
```

Expected: PASS.

- [ ] **Step 8: Commit the shim node**

```bash
cd /Users/ryan/code/milvus
git add internal/core/thirdparty/knowhere-rs-shim
git commit -m "feat(shim): add HNSW knowhere-rs bridge node"
```

## Chunk 4: Remote Standalone Smoke

### Task 5: Add remote scripts that build the Rust library, build Milvus with shim mode, and start standalone

**Files:**
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_env.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_sync.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/build_remote.sh`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/start_standalone_remote.sh`
- Create: `/Users/ryan/code/milvus/docs/knowhere-rs-shim/remote-runbook.md`

- [ ] **Step 1: Write the environment file**

Put these canonical roots into `remote_env.sh`:

```bash
export INTEG_ROOT=/data/work/milvus-rs-integ
export MILVUS_ROOT=$INTEG_ROOT/milvus
export KNOWHERE_RS_ROOT=$INTEG_ROOT/knowhere-rs
export SHIM_LOG_DIR=$INTEG_ROOT/artifacts
export LD_LIBRARY_PATH=$MILVUS_ROOT/internal/core/output/lib:$KNOWHERE_RS_ROOT/target/release:${LD_LIBRARY_PATH}
```

- [ ] **Step 2: Write the sync script**

It should:

```bash
- create `/data/work/milvus-rs-integ`
- sync `/Users/ryan/code/milvus` -> `$MILVUS_ROOT`
- sync `/Users/ryan/.openclaw/workspace-builder/knowhere-rs` -> `$KNOWHERE_RS_ROOT`
```

- [ ] **Step 3: Write the remote build script**

It should run:

```bash
cd "$KNOWHERE_RS_ROOT"
cargo build --release --features ffi

cd "$MILVUS_ROOT"
CMAKE_EXTRA_ARGS="-DMILVUS_USE_KNOWHERE_RS_SHIM=ON -DKNOWHERE_RS_LIB_PATH=$KNOWHERE_RS_ROOT/target/release/libknowhere_rs.so" \
make build-cpp
make milvus
```

- [ ] **Step 4: Write the standalone start script**

It should:

```bash
- source `remote_env.sh`
- ensure `SHIM_LOG_DIR` exists
- stop any stale standalone process
- start `bin/milvus run standalone --run-with-subprocess`
- wait for port 19530 readiness
```

- [ ] **Step 5: Verify the scripts are syntactically valid**

Run:

```bash
bash -n /Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_env.sh
bash -n /Users/ryan/code/milvus/scripts/knowhere-rs-shim/remote_sync.sh
bash -n /Users/ryan/code/milvus/scripts/knowhere-rs-shim/build_remote.sh
bash -n /Users/ryan/code/milvus/scripts/knowhere-rs-shim/start_standalone_remote.sh
```

Expected: no output, exit code 0.

- [ ] **Step 6: Commit the remote scripts**

```bash
cd /Users/ryan/code/milvus
git add scripts/knowhere-rs-shim docs/knowhere-rs-shim/remote-runbook.md
git commit -m "feat(shim): add remote build and standalone run scripts"
```

### Task 6: Add the end-to-end Milvus standalone smoke script and run it on the remote x86 host

**Files:**
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/smoke_hnsw.py`
- Create: `/Users/ryan/code/milvus/scripts/knowhere-rs-shim/run_remote_smoke.sh`

- [ ] **Step 1: Write the failing smoke script**

Create `smoke_hnsw.py`:

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

connections.connect(alias="default", host="127.0.0.1", port="19530")

name = "kh_rs_hnsw_smoke"
if utility.has_collection(name):
    utility.drop_collection(name)

schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=4),
])

col = Collection(name=name, schema=schema)
col.insert([
    [0, 1, 2, 3],
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
])
col.flush()
col.create_index("vec", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 32}})
col.load()
res = col.search([[1,0,0,0]], "vec", {"metric_type": "L2", "params": {"ef": 16}}, limit=2)
assert len(res) == 1
assert len(res[0]) == 2
print("SMOKE_OK")
```

- [ ] **Step 2: Write the remote runner**

`run_remote_smoke.sh` should:

```bash
- source `remote_env.sh`
- run `python3 -m pip install pymilvus`
- execute `smoke_hnsw.py`
- tee output to `$SHIM_LOG_DIR/smoke_hnsw.log`
```

- [ ] **Step 3: Run the smoke script before starting standalone to verify red**

Run on remote:

```bash
bash /data/work/milvus-rs-integ/milvus/scripts/knowhere-rs-shim/run_remote_smoke.sh
```

Expected: FAIL with connection error or index creation failure before the full remote lane is up.

- [ ] **Step 4: Run the full remote lane**

Run on remote:

```bash
bash /data/work/milvus-rs-integ/milvus/scripts/knowhere-rs-shim/remote_sync.sh
bash /data/work/milvus-rs-integ/milvus/scripts/knowhere-rs-shim/build_remote.sh
bash /data/work/milvus-rs-integ/milvus/scripts/knowhere-rs-shim/start_standalone_remote.sh
bash /data/work/milvus-rs-integ/milvus/scripts/knowhere-rs-shim/run_remote_smoke.sh
```

Expected:

```text
SMOKE_OK
```

- [ ] **Step 5: Archive the authoritative evidence**

Collect:

```bash
ls -la /data/work/milvus-rs-integ/artifacts
tail -n 100 /data/work/milvus-rs-integ/artifacts/smoke_hnsw.log
tail -n 100 /tmp/standalone.log
```

Expected: logs exist and show successful `create_index`, `load`, and `search`.

- [ ] **Step 6: Commit the smoke lane**

```bash
cd /Users/ryan/code/milvus
git add scripts/knowhere-rs-shim/smoke_hnsw.py scripts/knowhere-rs-shim/run_remote_smoke.sh
git commit -m "test(shim): add standalone HNSW smoke verification"
```

## Completion Checklist

- [ ] issue inventory exists before adapter expansion
- [ ] Milvus can build with `MILVUS_USE_KNOWHERE_RS_SHIM=ON`
- [ ] compile-surface smoke passes
- [ ] direct HNSW shim roundtrip smoke passes
- [ ] remote `knowhere-rs` release build succeeds without source changes
- [ ] Milvus standalone starts on remote x86 with shim libraries on `LD_LIBRARY_PATH`
- [ ] `smoke_hnsw.py` prints `SMOKE_OK`
- [ ] unsupported features remain documented as issues instead of being silently widened into scope

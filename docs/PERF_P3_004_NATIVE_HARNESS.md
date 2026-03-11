# PERF-P3-004 Native Benchmark Harness Enablement

Last updated: 2026-03-09 13:12 Asia/Shanghai

## 2026-03-11 update

The "harness enabled" conclusion below is stale for the freshly rebuilt official upstream binary.

- On the active official native workspace `/data/work/knowhere-native-src` at commit `bc613be25bee42c7dfdb9d62501db9bdbabcfda7`, both `benchmark_float_qps --gtest_list_tests` and `Benchmark_float_qps.TEST_HNSW` now abort before entering the benchmark body with `Metric name grpc.xds_client.resource_updates_valid has already been registered.`
- The earlier successful HNSW capture came from a pre-rebuild binary and must not be treated as evidence that the current official harness is still available.
- A follow-up isolated `WITH_LIGHT` probe narrowed a possible workaround path, but it did not unblock the harness:
  - `src/index/sparse/sparse_inverted_index.h` still contains `KNOHWERE_WITH_LIGHT` typo guards
  - `src/index/sparse/sparse_index_node.cc` misses `#include "knowhere/comp/task.h"` for `WaitAllSuccess`
  - even with `conan install -o with_light=True`, fetched `milvus-common` still fails configure because it unconditionally requires `opentelemetry-cpp`
- Current truth: PERF-P3-004 should be treated as regressed/blocked on current official upstream until either a minimal patch set is carried explicitly in the remote harness or upstream fixes the native dependency graph.

See also `docs/PERF_P3_005_NATIVE_VS_RS.md` for the up-to-date blocker chain.

## Goal

把 `clustered_l2 + HNSW` 的 native knowhere 侧 benchmark 入口收敛成一个**可重复执行、可映射到 knowhere-rs schema** 的最小 runbook；如果本轮仍不能直接跑通，则必须把 blocker 缩成具体依赖/补丁目标，而不是停留在“native harness unavailable”的宽泛表述。

## Current conclusion

本轮已把 PERF-P3-004 从“定位 blocker”推进到**实际拿到 native benchmark executable surface**：

- 远端源码现在应固定在**隔离 native workspace**：`/data/work/knowhere-native-src/benchmark/hdf5/benchmark_float_qps.cpp`
- native baseline 源默认使用 **Zilliz 官方 knowhere**：`https://github.com/zilliztech/knowhere.git`
- 统一探针 `scripts/remote/native_benchmark_probe.sh` 现已具备两段 bootstrap：
  - 若远端存在 `/usr/src/googletest`，直接本地构建/安装临时 GTest
  - 若不存在，则自动 shallow clone `googletest v1.14.0` 到 `/tmp/googletest-src` 后构建/安装到 `/tmp/gtest-install`
- probe 现使用实际安装产物路径：
  - `gtest_config=/tmp/gtest-install/lib/cmake/GTest/GTestConfig.cmake`
  - `GTEST_LIBRARY=/tmp/gtest-install/lib/libgtest.a`
  - `GTEST_MAIN_LIBRARY=/tmp/gtest-install/lib/libgtest_main.a`
- `WITH_BENCHMARK=ON` 的远端配置与 `benchmark_float_qps` 构建现已成功
- 运行 `benchmark_float_qps --gtest_list_tests` 时，还需补齐 Conan 产物的运行时库搜索路径；probe 已自动从 `/root/.conan/data/{folly,gflags}` 解析对应 package lib 目录并注入 `LD_LIBRARY_PATH`
- 当前 probe 已成功产出 `gtest_list.log`，证明 binary surface 真正可执行，而不是只停留在 target 声明或 link 阶段

因此本任务的 blocker 已从“GTest 发现失败”推进为**native benchmark 入口已打通，可进入数据/fixture 与基线生成阶段**。

## Remote native command path

### 1. Configure a benchmark-capable native build dir

```bash
ssh root@knowhere-x86-hk-proxy '
set -euo pipefail
src=/data/work/knowhere-native-src
base=/data/work/knowhere-native-src/build/Release
build=/data/work/knowhere-native-build-benchmark
rm -rf "$build"
mkdir -p "$build"
cmake -S "$src" -B "$build" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$base/generators/conan_toolchain.cmake" \
  -Dfolly_DIR="$base/generators" \
  -Dfmt_DIR="$base/generators" \
  -Dglog_DIR="$base/generators" \
  -Dprometheus-cpp_DIR="$base/generators" \
  -DxxHash_DIR="$base/generators" \
  -Dsimde_DIR="$base/generators" \
  -Dopentelemetry-cpp_DIR="$base/generators" \
  -Dopentelemetry-proto_DIR="$base/generators" \
  -DWITH_BENCHMARK=ON -DWITH_UT=OFF -DWITH_DISKANN=OFF \
  > "$build/configure.log" 2>&1
'
```

### 2. Build the minimal native entrypoint

```bash
ssh root@knowhere-x86-hk-proxy '
cmake --build /data/work/knowhere-native-build-benchmark --target benchmark_float_qps -j4 \
  > /data/work/knowhere-native-build-benchmark/build.log 2>&1
'
```

### 3. Probe the executable surface before running a full dataset

```bash
ssh root@knowhere-x86-hk-proxy '
/data/work/knowhere-native-build-benchmark/benchmark/benchmark_float_qps --gtest_list_tests \
  > /data/work/knowhere-native-build-benchmark/gtest_list.log 2>&1
'
```

## Logs / artifacts

- Remote probe runner: `scripts/remote/native_benchmark_probe.sh`
- Native configure log: `/data/work/knowhere-native-logs/configure.log`
- Native build log: `/data/work/knowhere-native-logs/build.log`
- Native gtest listing log: `/data/work/knowhere-native-logs/gtest_list.log`（仅当 build 成功时生成）
- knowhere-rs candidate-path artifact: `benchmark_results/cross_dataset_sampling.json`
- Field-mapping helper: `src/bin/native_benchmark_qps_parser.rs`

## Verified result from this round

Probe 已验证以下事实：

1. 远端初始失败根因不是 benchmark target 缺失，而是 **`/usr/src/googletest` 在当前 x86 环境中不存在**，导致临时 GTest bootstrap 直接失败。
2. 在缺少系统源码目录时，改为 clone `googletest v1.14.0` 并安装到 `/tmp/gtest-install` 后，`WITH_BENCHMARK=ON` 的配置与 `benchmark_float_qps` 构建均可成功完成。
3. 构建成功后，`benchmark_float_qps` 的第一层运行时失败来自 Conan 依赖的动态库搜索路径（例如 `libfolly_exception_tracer_base.so.0.58.0-dev`、`libgflags_nothreads.so.2.2`）；补齐 `LD_LIBRARY_PATH` 后，`--gtest_list_tests` 可以正常输出。
4. 因此 native benchmark 入口已经打通，下一步不再是修 benchmark/CMake 的发现链路，而是进入数据/fixture 与 baseline 生成。

## Output schema mapping

`benchmark_float_qps` 的 HNSW 路径会先打印 header 行，再打印每个线程的耗时/QPS（源码位于 `benchmark/hdf5/benchmark_float_qps.cpp`）：

```text
[0.245 s] clustered_l2_4k | HNSW(fp32) | M=16 | efConstruction=200, ef=128, k=10, R@=0.9300
  thread_num =  8, elapse =  0.050s, VPS = 2000.000
```

映射到 knowhere-rs 同方法学的字段约定如下：

- `R@=` → `recall_at_10`
- `VPS` → `qps`
- `elapse` → `runtime_seconds`
- `dataset/index/params/thread_num` 作为 context metadata 保留

本 repo 已提供 parser：

```bash
cargo run --bin native_benchmark_qps_parser -- \
  --input /path/to/native_qps.log --thread-num 8
```

输出示例：

```json
{
  "dataset": "clustered_l2_4k",
  "index": "HNSW(fp32)",
  "params": "M=16 | efConstruction=200, ef=128, k=10",
  "thread_num": 8,
  "recall_at_10": 0.93,
  "qps": 2000.0,
  "runtime_seconds": 0.05,
  "source": "knowhere_cpp_benchmark_float_qps"
}
```

## Methodology note

当前 native knowhere 自带 benchmark 入口偏向 HDF5/ANN benchmark 套件，不直接包含 `clustered_l2` synthetic generator；native 基线现在必须运行在**隔离目录**（`/data/work/knowhere-native-*`），避免与 `pipnn` 的远端工作树互相污染。因此本轮先完成两件事：

1. 固化 **native HNSW benchmark entrypoint + log schema**
2. 把 blocker 缩到 **GTest dependency** 这一层

下一轮若要真正对齐 `clustered_l2 + HNSW`，有两个最小延伸方向：

- **Preferred:** 在 native side 增加一个最小 synthetic dataset adapter / fixture，复用 `benchmark_float_qps` 的输出格式
- **Fallback:** 用现有 HDF5 benchmark 入口先证明 command path 与 field mapping 可执行，再补 synthetic dataset support

## Recommended next exec/plan target

1. 优先修 `benchmark/CMakeLists.txt` / 顶层 benchmark 配置，使其能消费 probe 已准备好的 `/tmp/gtest-install`（或等价的系统/Conan GTest）
2. 拿到 `benchmark_float_qps --gtest_list_tests` 成功日志，确认 binary surface 已真正可执行
3. 若随后仍卡在 link/build，再收敛 BLAS/OpenBLAS 链接路径，而不是重新回到“入口不可用”的宽泛描述
4. 只有在以上两步完成后，再决定是补 `clustered_l2` fixture，还是先用原生 HDF5 harness 打首个 native-vs-rs schema-aligned baseline

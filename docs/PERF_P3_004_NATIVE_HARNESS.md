# PERF-P3-004 Native Benchmark Harness Enablement

Last updated: 2026-03-09 13:12 Asia/Shanghai

## Goal

把 `clustered_l2 + HNSW` 的 native knowhere 侧 benchmark 入口收敛成一个**可重复执行、可映射到 knowhere-rs schema** 的最小 runbook；如果本轮仍不能直接跑通，则必须把 blocker 缩成具体依赖/补丁目标，而不是停留在“native harness unavailable”的宽泛表述。

## Current conclusion

本轮把 blocker 继续压缩成**可复现的远端 benchmark probe**，并确认 native harness 已不是“找不到入口”，而是**稳定卡在 benchmark 依赖发现链路**：

- 远端源码存在 benchmark 源：`/data/work/knowhere-src/benchmark/hdf5/benchmark_float_qps.cpp`
- 仓内新增统一探针：`scripts/remote/native_benchmark_probe.sh`
- 该探针会在远端先从 `/usr/src/googletest` 构建临时 GTest 安装，再用现成 Conan toolchain 重配 `WITH_BENCHMARK=ON`
- 当前 probe 输出已固定为：
  - `build_dir=/data/work/knowhere-build-benchmark`
  - `configure_log=/data/work/knowhere-build-benchmark/configure.log`
  - `build_log=/data/work/knowhere-build-benchmark/build.log`
  - `gtest_config=/tmp/gtest-install/lib/x86_64-linux-gnu/cmake/GTest/GTestConfig.cmake`
  - `build_exit_code=2`
- `configure.log` 现稳定复现 `benchmark/CMakeLists.txt` 里的 `find_package(GTest REQUIRED)` 失败；即使系统已装 `libgtest-dev/googletest`，远端 knowhere benchmark 这一条配置路径仍拿不到 `GTEST_LIBRARY/GTEST_INCLUDE_DIR/GTEST_MAIN_LIBRARY`

因此当前最小 concrete patch target 已进一步明确为：**修 benchmark 配置链路，让远端 knowhere benchmark 消费临时/系统 GTest，而不是继续停在模块发现失败**。

## Remote native command path

### 1. Configure a benchmark-capable native build dir

```bash
ssh root@knowhere-x86-hk-proxy '
set -euo pipefail
src=/data/work/knowhere-src
base=/data/work/knowhere-src/build/Release
build=/data/work/knowhere-build-benchmark
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
cmake --build /data/work/knowhere-build-benchmark --target benchmark_float_qps -j4 \
  > /data/work/knowhere-build-benchmark/build.log 2>&1
'
```

### 3. Probe the executable surface before running a full dataset

```bash
ssh root@knowhere-x86-hk-proxy '
/data/work/knowhere-build-benchmark/benchmark/benchmark_float_qps --gtest_list_tests \
  > /data/work/knowhere-build-benchmark/gtest_list.log 2>&1
'
```

## Logs / artifacts

- Remote probe runner: `scripts/remote/native_benchmark_probe.sh`
- Native configure log: `/data/work/knowhere-build-benchmark/configure.log`
- Native build log: `/data/work/knowhere-build-benchmark/build.log`
- Native gtest listing log: `/data/work/knowhere-build-benchmark/gtest_list.log`（仅当 build 成功时生成）
- knowhere-rs candidate-path artifact: `benchmark_results/cross_dataset_sampling.json`
- Field-mapping helper: `src/bin/native_benchmark_qps_parser.rs`

## Verified blocker from this round

Observed tail from `/data/work/knowhere-build-benchmark/configure.log`:

```text
-- Found HDF5: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so ...
CMake Error at /usr/share/cmake-3.28/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
  Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR GTEST_MAIN_LIBRARY)
```

Meaning:

1. `benchmark_float_qps` is a **real native target**, not a missing-source problem.
2. Reusing `build/Release/generators` already fixes the earlier `folly` discovery failure.
3. The next minimal step is **GTest availability**, not broad remote cleanup.

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

当前 native knowhere 自带 benchmark 入口偏向 HDF5/ANN benchmark 套件，不直接包含 `clustered_l2` synthetic generator；因此本轮先完成两件事：

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

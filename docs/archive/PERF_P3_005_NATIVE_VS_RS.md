# PERF-P3-005 Native-vs-RS 首条对照现状

Last updated: 2026-03-11 09:35 Asia/Shanghai

## 2026-03-11 update

After re-running the official native bootstrap/probe flow on remote x86, the blocker is now tighter and more honest than the 2026-03-10 note:

1. The active official workspace is `/data/work/knowhere-native-src` at commit `bc613be25bee42c7dfdb9d62501db9bdbabcfda7`.
2. The freshly rebuilt official `benchmark_float_qps` binary now aborts in both `--gtest_list_tests` and `Benchmark_float_qps.TEST_HNSW` with:

   ```text
   Metric name grpc.xds_client.resource_updates_valid has already been registered.
   ```

3. That invalidates any previous claim that the official native benchmark harness is currently "available"; the older successful HNSW run came from a pre-rebuild binary, not the current official rebuild.
4. A non-destructive `WITH_LIGHT` workaround probe did not unblock the current upstream tree:
   - `src/index/sparse/sparse_inverted_index.h` still contains `KNOHWERE_WITH_LIGHT` typo guards
   - `src/index/sparse/sparse_index_node.cc` fails under light mode because `WaitAllSuccess` is not declared until `knowhere/comp/task.h` is included
   - after fixing those two source-level defects in an isolated worktree and re-running `conan install -o with_light=True`, configure still fails because fetched `milvus-common` unconditionally executes `find_package(opentelemetry-cpp)`
5. Therefore `baseline-native-benchmark-smoke` remains failing on current official upstream. The next honest options are:
   - carry a minimal, auditable temporary patch set inside the remote probe flow
   - or explicitly classify official native smoke as blocked at this upstream commit and stop pretending that the benchmark harness is ready for methodology comparison

Relevant remote artifacts from this round:

- `/data/work/knowhere-native-logs/gtest_list.log`
- `/data/work/knowhere-native-logs/native_hnsw_qps_20260311T003259Z.log`
- `/data/work/knowhere-native-logs/light2-build.log`
- `/data/work/knowhere-native-logs/light3-conan-install.log`
- `/data/work/knowhere-native-logs/light3-configure.log`

## 结论

本轮**仍未完成 `clustered_l2 + HNSW` 的 native-vs-rs recall-gated 对照基线**，但 blocker 已从“运行时重复注册”继续压实到**native CMake 链接可见性设计问题**：`knowhere` 作为 shared lib 使用 `PUBLIC` 方式外泄 `milvus-common` 与一整串 otel/grpc 静态依赖，导致 `benchmark_float_qps` 一边载入 `libknowhere.so -> libmilvus-common.so`，一边又把同一批 grpc/otel runtime 直接静态链接进自身，最终在 `--gtest_list_tests` 前触发重复 metrics 注册 abort。

而且 blocker 已比上一轮更具体：

1. 隔离 native workspace 的源码目录已存在（`/data/work/knowhere-native-src`），但当前并没有可直接执行的 `benchmark_float_qps` binary。
2. 新证据表明，当前最前置的失败点不是 dataset 本身，而是 **native benchmark build prerequisite 缺失**：远端 PATH 上没有 `conan`，且 `/data/work/knowhere-native-src/build/Release/generators/conan_toolchain.cmake` 不存在，导致 `scripts/remote/native_benchmark_probe.sh` 在 CMake configure 前就无法满足 toolchain 前置条件。
3. 本轮已把这一步从“口头 blocker”落成可复用脚本 `scripts/remote/native_bootstrap.sh`：它会在远端创建独立 `conan` venv（固定 `conan<2`，避免 upstream `conanfile.py` 与 Conan 2 不兼容）、执行 profile detect，并在 `/data/work/knowhere-native-src/build/Release` 触发 `conan install` 生成 toolchain。
4. 截至本轮结束，bootstrap 已真实启动并进入依赖编译阶段（日志显示正在构建 `boost/1.83.0` / 配置 `hdf5/1.14.5`），但 `conan_toolchain.cmake` 尚未落地，因此 `benchmark_float_qps` 还不能恢复构建。
5. 只有在补齐 Conan/bootstrap、重新构建出 `benchmark_float_qps` 之后，才有资格继续验证上一轮已经收敛出的 dataset blocker：upstream benchmark 仍硬编码绑定 `sift-128-euclidean` HDF5 fixture，而不是 `clustered_l2` synthetic fixture。
6. 因而当前 blocker 应拆成两个串行前置，而不是混写成一个宽泛 blocker：
   - **Stage A（当前活跃）**：补齐 isolated native benchmark 的 Conan/toolchain/bootstrap，使 `benchmark_float_qps --gtest_list_tests` / `TEST_HNSW` 至少可执行。
   - **Stage B（后续）**：在 binary 真正可执行后，再解决 `clustered_l2` fixture / dataset adapter 缺口，形成同分布 native-vs-rs 对照。

## 本轮新增工具

新增脚本：

- `scripts/remote/native_bootstrap.sh`
- `scripts/remote/native_hnsw_qps_capture.sh`

用途：
- `native_bootstrap.sh`：在远端创建独立 Conan 1.x venv、执行 profile detect，并生成 isolated native benchmark 所需的 Conan toolchain / generators
- `native_hnsw_qps_capture.sh`：复用已打通的远端 `benchmark_float_qps` binary surface，统一补齐 Conan runtime `LD_LIBRARY_PATH`，执行 `Benchmark_float_qps.TEST_HNSW`，固化日志路径供后续 parser / 比对脚本直接消费

示例：

```bash
bash scripts/remote/native_bootstrap.sh
bash scripts/remote/native_hnsw_qps_capture.sh
```

若 native case 失败，脚本会返回非零退出码并打印远端日志路径。

## 已验证证据

### 1. native binary surface 可执行

远端 `gtest_list` 已确认：

```text
Benchmark_float_qps.
  TEST_IDMAP
  TEST_IVF_FLAT
  TEST_IVF_SQ8
  TEST_IVF_PQ
  TEST_HNSW
  TEST_SCANN
```

这说明 `benchmark_float_qps` 不只是能 build/link，**真实运行面已存在**。

### 2. HNSW case 当前仍绑 HDF5 fixture

远端源码检索显示：

- `benchmark/hdf5/benchmark_float_qps.cpp` 中 `set_ann_test_name("sift-128-euclidean")`
- `benchmark/prepare.sh` 也默认准备 `sift-128-euclidean.hdf5`

因此当前 native benchmark methodology 仍是 HDF5/ANN-benchmark 风格，不是 knowhere-rs 侧 Phase-5 计划里的 `clustered_l2` synthetic sampling。

### 3. 真实执行失败已缩成具体 blocker

`Benchmark_float_qps.TEST_HNSW` 远端真实运行结果表明：

- 缺少 `sift-128-euclidean.hdf5`
- HDF5 open 失败后进入非法 dataset class
- 后续 build path 误读大数量，最终抛出 `std::bad_alloc`

这意味着当前不是“benchmark harness unavailable”，而是：

- harness 已可运行
- **但缺少可用于本任务口径的 fixture**

## 与 knowhere-rs 基线的当前关系

knowhere-rs 侧已有 `clustered_l2 / HNSW` 可用 artifact：

- `benchmark_results/cross_dataset_sampling.json`
- 行：`dataset=clustered_l2`、`index=HNSW`
- 当前结果：
  - `recall_at_10 = 0.9520`
  - `qps = 1780.8553`
  - `runtime_seconds = 0.05615`
  - `confidence = trusted`

但由于 native side 还没有**同分布、同参数、同 recall gate** 的可解析 row，本轮还不能给出“领先 / parity / 落后”的有效结论。

## 本轮新增根因证据

### 4. duplicate grpc metrics 的更细根因：共享库 `PUBLIC` 依赖外泄

远端 `benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt` 已进一步说明问题不只是“两个产物都含同名字符串”，而是**CMake 链接面本身把重复 runtime 装配进了可执行程序**：

- `benchmark_float_qps` 显式链接 `../libknowhere.so`
- 同一条 link line 又显式链接 `../milvus-common-build/libmilvus-common.so`
- 同一条 link line 还显式拼入 `libopentelemetry_exporter_otlp_grpc.a`、`libgrpc++.a`、`libgrpc.a` 等静态库

而这些依赖本应主要作为 `libknowhere.so` / `libmilvus-common.so` 的内部实现依赖存在，不应该再由 benchmark 可执行重复持有一份 runtime registry。

进一步在隔离 native repo 上做了两轮最小验证：

### 4.1 粗暴降级为全 PRIVATE 会立即暴露 public header 泄漏

将 upstream `CMakeLists.txt` 里的

```cmake
target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})
```

临时改成

```cmake
target_link_libraries(knowhere PRIVATE ${KNOWHERE_LINKER_LIBS})
```

之后，`benchmark_float_qps` 的 link line 确实明显收缩，不再直接把 `milvus-common.so` 与 otel/grpc 大串依赖拖进来；但 benchmark 编译阶段随即因 `nlohmann/json.hpp` 缺失失败，说明 upstream 当前把“头文件可见性”和“实现链接依赖”混在了一起。

### 4.2 更细粒度 split 仍证明 public header 依赖已经越过 knowhere 自身边界

本轮又补了一次更细粒度 probe：

- 保留 `Boost::boost` / `nlohmann_json::nlohmann_json` / `glog::glog` / `simde::simde` 为 `PUBLIC`
- 将 `faiss` / `prometheus-cpp` / `fmt` / `Folly::folly` / `milvus-common` 下沉为 `PRIVATE`
- 通过脚本 `scripts/remote/native_usage_requirements_probe.sh` 在 isolated native repo 上自动改写、重配、重编译并恢复现场

结果依然没有进入运行时：`benchmark_float_qps.cpp` 在编译阶段就因为 `milvus-common-src/include/common/OpContext.h` 间接暴露 `folly/CancellationToken.h` 而失败。这说明问题不只是 `nlohmann_json` 这种轻量 header-only 依赖，而是 **knowhere 的 public header 面已经透传了 milvus-common / folly 级别的上游类型**。换句话说，真正需要拆分的不只是 `target_link_libraries(PUBLIC/PRIVATE)`，还包括：

- public header 是否必须直接包含 `milvus-common` 的类型定义
- 是否需要把 `OpContext` 之类依赖改成前向声明 / pimpl / 本地抽象
- benchmark 是否应该只消费更窄的稳定 public facade，而不是穿透到当前这层 header graph

这让下一步修复方向变得更明确：

- **不是**继续盲查 runtime abort
- **而是**在 native repo 里拆分 `knowhere` 的 public usage requirements（headers/include dirs）与 private runtime deps（milvus-common / otel / grpc）

## 本轮新增证据（2026-03-10 09:24 Asia/Shanghai）

### 4.3 `OpContext` 公开 include 只是第一层，去掉后 blocker 立即前移到 header 内联线程池语义

本轮新增了两个可复用 probe：

- `scripts/remote/native_usage_requirements_probe.sh`
- `scripts/remote/native_public_header_probe.sh`

第二个 probe 在 isolated native repo 上做了两步**可逆**实验：

1. 继续维持 `knowhere` 的 `PUBLIC/PRIVATE` link split（仅保留 `Boost/nlohmann_json/glog/simde` 为 PUBLIC）。
2. 将 `include/knowhere/index/index_node.h` 中 `common/OpContext.h` 改为纯前向声明，再重编 `benchmark_float_qps`。

结果表明：

- 原先的 `milvus-common -> folly/CancellationToken.h` compile blocker 确实被越过了；
- 但 benchmark consumer 随即卡在 **同一个 public header 的更深层 inline 实现**：`index_node.h` 内联代码直接依赖 `folly::Future`、`ThreadPool`、`WaitAllSuccess`，这些符号来自 `knowhere/comp/task.h` / Folly 线程池栈；
- 如果进一步粗暴去掉这组 heavy includes，失败会从 benchmark compile 前移到 `libknowhere.so` 自身编译阶段，因为 `index_node.h` 的 696/801/832 等位置直接在头文件里使用这些类型，而不是只在 `.cc` 中使用。

本轮（2026-03-10 10:06 Asia/Shanghai）再次复跑 `scripts/remote/native_public_header_probe.sh`，结果稳定复现：首个 compile error 仍是 `index_node.h:696/801/832` 上的 `folly::Future` / `ThreadPool` / `WaitAllSuccess`，说明这不是一次性远端环境抖动，而是可重复的 public-header 设计问题。

又在本轮（2026-03-10 10:20 Asia/Shanghai）按相同入口追加复跑并远端 grep `public_header_probe.build.log`，首个错误序列仍未变化：`folly not declared`、`ThreadPool has not been declared`、`WaitAllSuccess was not declared in this scope`，位置依旧锁定在 `index_node.h:696/801/832`。这进一步确认 blocker 已稳定到 header-level inline runtime 依赖，而不是 configure/toolchain 偶发噪音。

本轮（2026-03-10 10:40 Asia/Shanghai）再次完整执行 `bash -n scripts/remote/native_public_header_probe.sh && bash scripts/remote/native_public_header_probe.sh` 后，`public_header_probe.build.log` 的 first-error 仍然完全一致；build 尾部虽然还会出现 `src/common/comp/brute_force.cc` 的编译失败，但那属于前序 header 失配触发的大面积级联报错，不改变首个稳定 blocker 的定性。

这把 blocker 再次收紧成一个更明确的工程结论：

> 当前 native 问题不是“补一个缺失 include dir”就能解决，而是 `include/knowhere/index/index_node.h` 同时承载了对外 ABI/接口定义和内部并行执行实现，导致 benchmark 这类 consumer 被迫继承 `milvus-common + folly + ThreadPool` 整条依赖链。

换句话说，若要让 `benchmark_float_qps` 在不重复静态装配整串 runtime 依赖的前提下成立，native 侧需要做的不是继续 tweak link line，而是**拆 public facade**：

- 将 `OpContext` 保持为前向声明或更窄抽象；
- 将 `folly::Future` / `ThreadPool` / `WaitAllSuccess` 等并行细节从 `index_node.h` 对外头文件中移走，改为 `.cc` / pimpl / internal facade；
- 让 benchmark 只消费稳定 API，而不是吃下当前整张 implementation header graph。

## 下一步最小闭环建议

优先级最高的不是继续修 benchmark harness，而是先修正 native 链接/usage-requirement，再谈 fixture：

1. **Preferred：** 在 native side 为 `benchmark_float_qps` 增加最小 `clustered_l2` synthetic dataset adapter，直接复用现有 header/thread log schema。
2. **更前置的工程前提：** 先把 `index_node.h` 的 public facade 与内部并行/runtime 依赖拆开；否则 `benchmark_float_qps` 这类 consumer 仍会被迫继承 `milvus-common/folly` 依赖链，PUBLIC/PRIVATE split 只会不断暴露下一个 leak 点。
3. **Fallback：** 若短期只能走 HDF5，则至少先补齐 native side 的可下载 fixture 与参数固定口径，然后把 knowhere-rs 也切到同一数据集/metric/profile 做首条对照；但这会偏离当前 Phase-5 已选中的 `clustered_l2` 路线。

在完成上述任一项之前，本任务应保持 `exec_blocked`，而不是虚报已形成 leadership verdict。

## 本轮新增证据（2026-03-10 11:03 Asia/Shanghai）

### 4.4 header 内联 runtime 不是终点，最小同步化 probe 后 first-error 已前移到 private `.cc`

本轮直接增强了 `scripts/remote/native_public_header_probe.sh`：

- 保持此前的 `PUBLIC/PRIVATE` usage-requirement split；
- 继续把 `index_node.h` 中 `OpContext` 改为前向声明；
- 不再只删除 `knowhere/comp/task.h` 等 heavy include，而是**把 `index_node.h` 里 3 处 `folly::Future` / `ThreadPool` / `WaitAllSuccess` 的内联调度逻辑临时降级为同步直跑**，用来验证 benchmark-facing public header 的最小可编译面。

复跑 `bash -n scripts/remote/native_public_header_probe.sh && bash scripts/remote/native_public_header_probe.sh` 后，远端 `public_header_probe.build.log` 的 first-error 已发生实质迁移：

- **不再**首先报 `include/knowhere/index/index_node.h:696/801/832` 上的 `folly::Future` / `ThreadPool` / `WaitAllSuccess` 未声明；
- **新的首个稳定错误**变成 `src/index/sparse/sparse_index_node.cc:142`：`WaitAllSuccess` 未声明。

这说明两件事：

1. `index_node.h` 的 public-header inline runtime 依赖，确实是 benchmark consumer 最前置的 compile blocker；
2. 一旦把这些 inline runtime 逻辑从 public header 面拿掉，编译就能继续深入到 `knowhere` 私有实现文件，随后暴露出新的、更健康的依赖缺口：**私有 `.cc` 文件此前在“偷吃” `index_node.h -> knowhere/comp/task.h` 的传递性 include**。

结合本地源码可见：

- `src/index/sparse/sparse_index_node.cc` 自己显式包含了 `knowhere/thread_pool.h`，但**没有**包含 `knowhere/comp/task.h`；
- 该文件内部却直接使用了 `std::vector<folly::Future<folly::Unit>>` 与 `WaitAllSuccess(futs)`；
- `WaitAllSuccess` 的真实声明位于 `include/knowhere/comp/task.h`。

因此本轮把修复方向进一步收紧为一个明确的最小工程动作：

> native 侧要想真正把 `index_node.h` 缩成 benchmark-facing public facade，不能只改 header；还需要把各个私有实现文件对 `task.h` / Folly runtime 的依赖改成**显式私有 include**，而不是继续依赖 public header 的传递性暴露。

这比上一轮的结论更具体：

- 上一轮只能证明“public header 混入了 runtime 语义”；
- 本轮已经证明“把这层 runtime 语义挪走后，编译面能前进，且首个补丁点落在 private `sparse_index_node.cc` 的 include hygiene 上”。

因此下一轮最小可执行动作应改为：

1. 继续沿用当前同步化 public-header probe；
2. 在 isolated native repo 中为 `src/index/sparse/sparse_index_node.cc` 等真实使用 `WaitAllSuccess`/`folly::Future` 的私有实现补显式 `#include "knowhere/comp/task.h"`；
3. 观察 build first-error 是否继续前移，直到 `benchmark_float_qps` 恢复可编译/可执行面，随后再回到 dataset adapter / native-vs-rs artifact 阶段。

## 本轮新增证据（2026-03-10 11:20 Asia/Shanghai）

### 4.5 compile/link blocker 已越过，`benchmark_float_qps --gtest_list_tests` 恢复可执行

本轮继续增强 `scripts/remote/native_public_header_probe.sh`，在同一可逆 probe 内完成两类修复：

1. 为 private 实现 `src/index/sparse/sparse_index_node.cc` 补显式 `#include "knowhere/comp/task.h"`，不再依赖 `index_node.h` 的传递性 include；
2. 运行时不再把整个 `/root/.conan/data` 注入 `LD_LIBRARY_PATH`，而是从 `benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt` 提取当前 build 实际使用的 `-Wl,-rpath`，只保留本次构建命中的 `glog 0.7.1` / `gflags 2.2.2` 等精确 runtime 路径。

结果链路如下：

- 第一次复跑后，`build_exit_code=0`，说明 public-header + private include hygiene 已足够让 `benchmark_float_qps` 完整编译链接；
- 随后 `gtest_exit_code` 先从 `127`（缺 `libgflags_nothreads.so`）推进到 `1`（错误地把 Conan cache 全量塞入 `LD_LIBRARY_PATH`，触发 `glog 0.6.0`/`0.7.1` 重复 flag 注册）；
- 将 runtime 路径收窄到 link line 实际声明的 rpath 后，`gtest_exit_code=0`，`public_header_probe.gtest_list.log` 成功列出：`TEST_IDMAP` / `TEST_IVF_FLAT` / `TEST_IVF_SQ8` / `TEST_IVF_PQ` / `TEST_HNSW` / `TEST_SCANN`。

这意味着当前主阻塞已不再是 native benchmark harness 的 compile/link/runtime availability，而是**方法学对齐**：

- native side 仍是 `benchmark/hdf5/benchmark_float_qps.cpp` 的 `sift-128-euclidean` HDF5 fixture 路线；
- Rust side 当前可信 artifact 来自 `clustered_l2`；
- 两边尚未形成同分布、同参数、同 recall gate 的可比 row。

因此当前大任务仍保持 `BASELINE-P3-001`，但子阶段应从 `native_public_header_minimal_fix` 前移到 **`native_fixture_alignment`**：下一步不该再花轮次修 public header，而应直接处理 native fixture / dataset adapter 对齐，或明确留下 no-go 证据。

## 本轮新增证据（2026-03-10 11:43 Asia/Shanghai）

### 4.6 `TEST_HNSW` 已可直接执行，但当前失败已稳定收敛到 HDF5 fixture / methodology 缺口

本轮做了两件最小闭环动作：

1. 将 `scripts/remote/native_hnsw_qps_capture.sh` 从“扫描整个 Conan cache 塞进 `LD_LIBRARY_PATH`”改为**从 `benchmark_float_qps.dir/link.txt` 提取精确 rpath**，避免再次触发 `glog/gflags` 多版本重复注册。
2. 直接远端执行：

```bash
bash scripts/remote/native_hnsw_qps_capture.sh
```

结果：

- binary surface 真实可用，脚本成功启动 `Benchmark_float_qps.TEST_HNSW`
- 远端日志：`/data/work/knowhere-native-logs/native_hnsw_qps_20260310T034248Z.log`
- 退出码：`1`
- 首组稳定错误不再是 compile/link/runtime registry，而是：
  - `Loading HDF5 file: sift-128-euclidean.hdf5`
  - `Illegal dataset class type`
  - 随后在 index load/build 路径抛出 `std::bad_array_new_length`

同轮用更直接的 probe 也拿到了同一结论：

- `benchmark/hdf5/benchmark_float_qps.cpp:366` 仍固定 `set_ann_test_name("sift-128-euclidean")`
- `benchmark/prepare.sh` 仍固定准备 `sift-128-euclidean.hdf5`，且脚本本身还存在落盘问题：它使用 `wget -P $SIFT_FILE ...`，其中 `$SIFT_FILE` 被设置成完整文件路径而不是目录；这既不会把 fixture 放到 benchmark 当前工作目录，也容易把准备动作变成“创建同名目录前缀”而非“写出同名 HDF5 文件”
- 为避免继续把问题混成“单纯缺数据集”，本轮补了 `scripts/remote/native_fetch_hdf5_fixture.sh`，改为直接把 fixture 下载到远端 native repo 根目录下的 `${fixture_name}.hdf5`，并验证 HDF5 magic；probe 显示 URL 可达、下载可推进，说明当前 blocker 已从“网络/链接不可达”进一步收敛为“upstream benchmark 仍硬编码 HDF5 fixture，且默认 prepare 脚本不可靠”
- 若 fixture 缺失或内容/路径不合法，native HNSW case 会在 HDF5 load 之后进入错误 dataset class，并把后续 build/load 逻辑带偏到异常大 row count，最终抛出 `std::bad_alloc` / `std::bad_array_new_length`

这把当前 blocker 再收紧了一层：

> 现在已经不是“native benchmark 还跑不起来”，而是“native benchmark 已经能跑到 case body，但仍被 upstream `sift-128-euclidean` HDF5 methodology 绑定，且在缺失合法 fixture 时直接失败”。

因此下一轮最小动作应是二选一：

1. **Adapter 路线：** 为 native side 增加 `clustered_l2` synthetic adapter，直接形成与 Rust baseline 同分布的 row；
2. **No-go 路线：** 固化当前 `sift-128-euclidean` 依赖为 methodology blocker，并明确记录：若不接受 HDF5 fixture/download 前置，本阶段无法给出 clustered-l2 native-vs-rs leadership verdict。
